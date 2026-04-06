#include "ch13_qk_fused_path.h"

#include "ch13_gguf_runtime.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

#ifndef CH13_QK_FUSED_SPV_PATH
#define CH13_QK_FUSED_SPV_PATH ""
#endif

#ifndef CH13_MLP_FUSED_SPV_PATH
#define CH13_MLP_FUSED_SPV_PATH ""
#endif

[[noreturn]] void fail(const std::string & message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

void check_vk(VkResult result, const char * what) {
    if (result != VK_SUCCESS) {
        fail(std::string(what) + " failed with VkResult=" + std::to_string(static_cast<int>(result)));
    }
}

struct LocalCtx {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ~LocalCtx() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx) ggml_free(ctx);
    }
};

std::string layer_key(int il, const std::string & suffix) {
    return "runtime.layers." + std::string(il < 10 ? "0" : "") + std::to_string(il) + "." + suffix;
}

std::vector<float> tensor_to_f32(ggml_tensor * tensor) {
    std::vector<float> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float> tensor_to_f32_expanded(ggml_tensor * tensor) {
    if (tensor->type == GGML_TYPE_F32) {
        return tensor_to_f32(tensor);
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(ggml_fp16_t));
        std::vector<float> out(raw.size());
        ggml_fp16_to_fp32_row(raw.data(), out.data(), static_cast<int64_t>(raw.size()));
        return out;
    }
    fail(std::string("unsupported tensor type for f32 expansion: ") + ggml_type_name(tensor->type));
}

Ch13QKFusedResult run_qk_graph_path(
    Ch13RuntimeGGUF & runtime,
    const Ch13QKFusedConfig & cfg,
    const std::vector<float> & attn_input,
    const std::string & backend_name) {
    if ((int)attn_input.size() != cfg.hidden * cfg.seq_len) {
        fail("run_qk_graph_path: attn_input shape mismatch");
    }

    LocalCtx local;
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    local.ctx = ggml_init(params);
    if (!local.ctx) fail("failed to init qk graph ctx");

    ggml_cgraph * gf = ggml_new_graph_custom(local.ctx, 128, false);
    if (!gf) fail("failed to init qk graph");

    ggml_tensor * inp = ggml_new_tensor_2d(local.ctx, GGML_TYPE_F32, cfg.hidden, cfg.seq_len);
    ggml_set_name(inp, "qk_graph.input");
    ggml_set_input(inp);

    ggml_tensor * pos = ggml_new_tensor_1d(local.ctx, GGML_TYPE_I32, cfg.seq_len);
    ggml_set_name(pos, "qk_graph.pos");
    ggml_set_input(pos);

    ggml_tensor * q_proj = ggml_mul_mat(local.ctx, runtime.get_tensor(layer_key(cfg.layer, "q_proj.weight").c_str()), inp);
    ggml_tensor * k_proj = ggml_mul_mat(local.ctx, runtime.get_tensor(layer_key(cfg.layer, "k_proj.weight").c_str()), inp);
    ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);

    ggml_tensor * q_3d = ggml_reshape_3d(local.ctx, q_proj, cfg.head_dim, cfg.n_head, cfg.seq_len);
    ggml_tensor * k_3d = ggml_reshape_3d(local.ctx, k_proj, cfg.head_dim, cfg.n_kv_head, cfg.seq_len);

    ggml_tensor * q_norm = ggml_mul(local.ctx, ggml_rms_norm(local.ctx, q_3d, cfg.rms_norm_eps), ggml_repeat(local.ctx, runtime.get_tensor(layer_key(cfg.layer, "q_norm.weight").c_str()), q_3d));
    ggml_tensor * k_norm = ggml_mul(local.ctx, ggml_rms_norm(local.ctx, k_3d, cfg.rms_norm_eps), ggml_repeat(local.ctx, runtime.get_tensor(layer_key(cfg.layer, "k_norm.weight").c_str()), k_3d));

    ggml_tensor * q_rope = ggml_rope_ext(local.ctx, q_norm, pos, nullptr, cfg.head_dim, GGML_ROPE_TYPE_NEOX, cfg.rope_n_ctx_orig, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope = ggml_rope_ext(local.ctx, k_norm, pos, nullptr, cfg.head_dim, GGML_ROPE_TYPE_NEOX, cfg.rope_n_ctx_orig, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_set_name(q_rope, "qk_graph.q_rope");
    ggml_set_name(k_rope, "qk_graph.k_rope");
    ggml_set_output(q_rope);
    ggml_set_output(k_rope);
    ggml_build_forward_expand(gf, q_rope);
    ggml_build_forward_expand(gf, k_rope);

    const bool use_cpu = backend_name == "cpu";
    if (use_cpu) {
        local.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime.backend));
        if (!local.allocr) fail("failed to init qk graph gallocr");
        if (!ggml_gallocr_alloc_graph(local.allocr, gf)) fail("failed to alloc qk graph");
    } else {
        ggml_backend_sched_reset(runtime.sched);
        if (!ggml_backend_sched_alloc_graph(runtime.sched, gf)) fail("failed to alloc qk graph");
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qk_graph.input"), attn_input.data(), 0, attn_input.size() * sizeof(float));
    std::vector<int32_t> pos_i(cfg.seq_len);
    for (int i = 0; i < cfg.seq_len; ++i) pos_i[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qk_graph.pos"), pos_i.data(), 0, pos_i.size() * sizeof(int32_t));

    if (use_cpu) {
        if (ggml_backend_graph_compute(runtime.backend, gf) != GGML_STATUS_SUCCESS) fail("failed qk graph compute");
    } else {
        if (ggml_backend_sched_graph_compute(runtime.sched, gf) != GGML_STATUS_SUCCESS) fail("failed qk graph compute");
        ggml_backend_sched_reset(runtime.sched);
    }

    return {
        tensor_to_f32(ggml_graph_get_tensor(gf, "qk_graph.q_rope")),
        tensor_to_f32(ggml_graph_get_tensor(gf, "qk_graph.k_rope")),
    };
}

std::vector<uint32_t> read_spirv(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        fail("missing CH13_QK_FUSED_SPV_PATH compile definition");
    }
    FILE * file = std::fopen(path, "rb");
    if (!file) {
        fail(std::string("failed to open SPIR-V file: ") + path);
    }
    std::fseek(file, 0, SEEK_END);
    long size = std::ftell(file);
    std::rewind(file);
    if (size <= 0 || (size % 4) != 0) {
        std::fclose(file);
        fail(std::string("invalid SPIR-V file size: ") + path);
    }
    std::vector<uint32_t> data(static_cast<size_t>(size) / 4);
    const size_t read_count = std::fread(data.data(), 1, static_cast<size_t>(size), file);
    std::fclose(file);
    if (read_count != static_cast<size_t>(size)) {
        fail(std::string("failed to read SPIR-V file: ") + path);
    }
    return data;
}

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) {
            continue;
        }
        if ((props.memoryTypes[i].propertyFlags & required_flags) == required_flags) {
            return i;
        }
    }
    fail("failed to find suitable Vulkan memory type");
}

struct VkBufferHost {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void * mapped = nullptr;
    VkDeviceSize size = 0;
};

struct VkQKFusedPush {
    uint32_t hidden;
    uint32_t seq_len;
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_kv_head;
    float eps;
    float theta_scale;
};

struct LocalVulkanQK {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family_index = UINT32_MAX;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<VkBufferHost> buffers;

    void clear_buffers() {
        for (auto & buf : buffers) {
            if (buf.mapped) {
                vkUnmapMemory(device, buf.memory);
                buf.mapped = nullptr;
            }
            if (buf.buffer) {
                vkDestroyBuffer(device, buf.buffer, nullptr);
                buf.buffer = VK_NULL_HANDLE;
            }
            if (buf.memory) {
                vkFreeMemory(device, buf.memory, nullptr);
                buf.memory = VK_NULL_HANDLE;
            }
        }
        buffers.clear();
    }

    ~LocalVulkanQK() {
        clear_buffers();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader_module) vkDestroyShaderModule(device, shader_module, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    VkBufferHost & create_host_buffer(VkDeviceSize size) {
        buffers.push_back({});
        auto & out = buffers.back();
        out.size = size;

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check_vk(vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer), "vkCreateBuffer");

        VkMemoryRequirements mem_req{};
        vkGetBufferMemoryRequirements(device, out.buffer, &mem_req);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_req.size;
        alloc_info.memoryTypeIndex = find_memory_type(
            physical_device,
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        check_vk(vkAllocateMemory(device, &alloc_info, nullptr, &out.memory), "vkAllocateMemory");
        check_vk(vkBindBufferMemory(device, out.buffer, out.memory, 0), "vkBindBufferMemory");
        check_vk(vkMapMemory(device, out.memory, 0, size, 0, &out.mapped), "vkMapMemory");
        return out;
    }

    void init() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "ch13-qk-fused";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "none";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;
        check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

        uint32_t physical_device_count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (physical_device_count == 0) {
            fail("no Vulkan physical devices found for ch13 fused qk path");
        }
        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        check_vk(vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data()), "vkEnumeratePhysicalDevices(list)");

        for (VkPhysicalDevice candidate : physical_devices) {
            uint32_t queue_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
            std::vector<VkQueueFamilyProperties> queues(queue_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
            for (uint32_t i = 0; i < queue_count; ++i) {
                if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physical_device = candidate;
                    queue_family_index = i;
                    break;
                }
            }
            if (physical_device != VK_NULL_HANDLE) break;
        }
        if (physical_device == VK_NULL_HANDLE) {
            fail("failed to find a Vulkan physical device with a compute queue");
        }

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        check_vk(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queue_family_index, 0, &queue);

        VkDescriptorSetLayoutBinding bindings[7]{};
        for (uint32_t i = 0; i < 7; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo set_layout_info{};
        set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_layout_info.bindingCount = 7;
        set_layout_info.pBindings = bindings;
        check_vk(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(VkQKFusedPush);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

        const auto spirv = read_spirv(CH13_QK_FUSED_SPV_PATH);
        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = spirv.size() * sizeof(uint32_t);
        shader_info.pCode = spirv.data();
        check_vk(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo shader_stage{};
        shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage.module = shader_module;
        shader_stage.pName = "main";

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage;
        pipeline_info.layout = pipeline_layout;
        check_vk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines");

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 7;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_vk(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &set_layout;
        check_vk(vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set), "vkAllocateDescriptorSets");

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.queueFamilyIndex = queue_family_index;
        check_vk(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo command_buffer_info{};
        command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_info.commandPool = command_pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;
        check_vk(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check_vk(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");
    }

    Ch13QKFusedResult run(const Ch13QKFusedConfig & cfg,
                          const std::vector<float> & attn_input,
                          const std::vector<float> & q_weight,
                          const std::vector<float> & k_weight,
                          const std::vector<float> & q_norm_weight,
                          const std::vector<float> & k_norm_weight) {
        clear_buffers();
        buffers.reserve(7);

        const size_t input_bytes = attn_input.size() * sizeof(float);
        const size_t q_weight_bytes = q_weight.size() * sizeof(float);
        const size_t k_weight_bytes = k_weight.size() * sizeof(float);
        const size_t q_norm_bytes = q_norm_weight.size() * sizeof(float);
        const size_t k_norm_bytes = k_norm_weight.size() * sizeof(float);
        const size_t q_out_elems = static_cast<size_t>(cfg.head_dim) * static_cast<size_t>(cfg.n_head) * static_cast<size_t>(cfg.seq_len);
        const size_t k_out_elems = static_cast<size_t>(cfg.head_dim) * static_cast<size_t>(cfg.n_kv_head) * static_cast<size_t>(cfg.seq_len);
        const size_t q_out_bytes = q_out_elems * sizeof(float);
        const size_t k_out_bytes = k_out_elems * sizeof(float);

        auto & input_buf = create_host_buffer(input_bytes);
        auto & q_weight_buf = create_host_buffer(q_weight_bytes);
        auto & k_weight_buf = create_host_buffer(k_weight_bytes);
        auto & q_norm_buf = create_host_buffer(q_norm_bytes);
        auto & k_norm_buf = create_host_buffer(k_norm_bytes);
        auto & q_out_buf = create_host_buffer(q_out_bytes);
        auto & k_out_buf = create_host_buffer(k_out_bytes);

        std::memcpy(input_buf.mapped, attn_input.data(), input_bytes);
        std::memcpy(q_weight_buf.mapped, q_weight.data(), q_weight_bytes);
        std::memcpy(k_weight_buf.mapped, k_weight.data(), k_weight_bytes);
        std::memcpy(q_norm_buf.mapped, q_norm_weight.data(), q_norm_bytes);
        std::memcpy(k_norm_buf.mapped, k_norm_weight.data(), k_norm_bytes);
        std::memset(q_out_buf.mapped, 0, q_out_bytes);
        std::memset(k_out_buf.mapped, 0, k_out_bytes);

        VkDescriptorBufferInfo infos[7]{};
        const VkBufferHost * host_bufs[7] = { &input_buf, &q_weight_buf, &k_weight_buf, &q_norm_buf, &k_norm_buf, &q_out_buf, &k_out_buf };
        VkWriteDescriptorSet writes[7]{};
        for (uint32_t i = 0; i < 7; ++i) {
            infos[i].buffer = host_bufs[i]->buffer;
            infos[i].offset = 0;
            infos[i].range = host_bufs[i]->size;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

        const float theta_scale = std::pow(cfg.rope_freq_base, -2.0f / static_cast<float>(cfg.head_dim));
        const VkQKFusedPush push {
            static_cast<uint32_t>(cfg.hidden),
            static_cast<uint32_t>(cfg.seq_len),
            static_cast<uint32_t>(cfg.head_dim),
            static_cast<uint32_t>(cfg.n_head),
            static_cast<uint32_t>(cfg.n_kv_head),
            cfg.rms_norm_eps,
            theta_scale,
        };
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(command_buffer, 1, static_cast<uint32_t>(cfg.n_head + cfg.n_kv_head), static_cast<uint32_t>(cfg.seq_len));
        check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        check_vk(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
        check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        check_vk(vkResetFences(device, 1, &fence), "vkResetFences");
        check_vk(vkResetCommandPool(device, command_pool, 0), "vkResetCommandPool");

        Ch13QKFusedResult out;
        out.q_rope.resize(q_out_elems);
        out.k_rope.resize(k_out_elems);
        std::memcpy(out.q_rope.data(), q_out_buf.mapped, q_out_bytes);
        std::memcpy(out.k_rope.data(), k_out_buf.mapped, k_out_bytes);
        return out;
    }
};

struct VkMLPFusedPush {
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t seq_len;
    float eps;
};

struct LocalVulkanMLP {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family_index = UINT32_MAX;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<VkBufferHost> buffers;

    void clear_buffers() {
        for (auto & buf : buffers) {
            if (buf.mapped) {
                vkUnmapMemory(device, buf.memory);
                buf.mapped = nullptr;
            }
            if (buf.buffer) {
                vkDestroyBuffer(device, buf.buffer, nullptr);
                buf.buffer = VK_NULL_HANDLE;
            }
            if (buf.memory) {
                vkFreeMemory(device, buf.memory, nullptr);
                buf.memory = VK_NULL_HANDLE;
            }
        }
        buffers.clear();
    }

    ~LocalVulkanMLP() {
        clear_buffers();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader_module) vkDestroyShaderModule(device, shader_module, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    VkBufferHost & create_host_buffer(VkDeviceSize size) {
        buffers.push_back({});
        auto & out = buffers.back();
        out.size = size;

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check_vk(vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer), "vkCreateBuffer");

        VkMemoryRequirements mem_req{};
        vkGetBufferMemoryRequirements(device, out.buffer, &mem_req);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_req.size;
        alloc_info.memoryTypeIndex = find_memory_type(
            physical_device,
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        check_vk(vkAllocateMemory(device, &alloc_info, nullptr, &out.memory), "vkAllocateMemory");
        check_vk(vkBindBufferMemory(device, out.buffer, out.memory, 0), "vkBindBufferMemory");
        check_vk(vkMapMemory(device, out.memory, 0, size, 0, &out.mapped), "vkMapMemory");
        return out;
    }

    void init() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "ch13-mlp-fused";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "none";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;
        check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

        uint32_t physical_device_count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (physical_device_count == 0) fail("no Vulkan physical devices found for ch13 fused mlp path");
        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        check_vk(vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data()), "vkEnumeratePhysicalDevices(list)");
        for (VkPhysicalDevice candidate : physical_devices) {
            uint32_t queue_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
            std::vector<VkQueueFamilyProperties> queues(queue_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
            for (uint32_t i = 0; i < queue_count; ++i) {
                if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physical_device = candidate;
                    queue_family_index = i;
                    break;
                }
            }
            if (physical_device != VK_NULL_HANDLE) break;
        }
        if (physical_device == VK_NULL_HANDLE) fail("failed to find Vulkan physical device with compute queue");

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        check_vk(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queue_family_index, 0, &queue);

        VkDescriptorSetLayoutBinding bindings[7]{};
        for (uint32_t i = 0; i < 7; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo set_layout_info{};
        set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_layout_info.bindingCount = 7;
        set_layout_info.pBindings = bindings;
        check_vk(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(VkMLPFusedPush);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

        const auto spirv = read_spirv(CH13_MLP_FUSED_SPV_PATH);
        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = spirv.size() * sizeof(uint32_t);
        shader_info.pCode = spirv.data();
        check_vk(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo shader_stage{};
        shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage.module = shader_module;
        shader_stage.pName = "main";

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage;
        pipeline_info.layout = pipeline_layout;
        check_vk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines");

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 7;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_vk(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &set_layout;
        check_vk(vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set), "vkAllocateDescriptorSets");

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.queueFamilyIndex = queue_family_index;
        check_vk(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo command_buffer_info{};
        command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_info.commandPool = command_pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;
        check_vk(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check_vk(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");
    }

    std::pair<std::vector<float>, std::vector<float>> run(
        int hidden,
        int intermediate,
        int seq_len,
        float eps,
        const std::vector<float> & attn_residual,
        const std::vector<float> & norm_weight,
        const std::vector<float> & gate_weight,
        const std::vector<float> & up_weight,
        const std::vector<float> & down_weight) {
        clear_buffers();
        buffers.reserve(7);

        const size_t input_bytes = attn_residual.size() * sizeof(float);
        const size_t norm_bytes = norm_weight.size() * sizeof(float);
        const size_t gate_bytes = gate_weight.size() * sizeof(float);
        const size_t up_bytes = up_weight.size() * sizeof(float);
        const size_t down_bytes = down_weight.size() * sizeof(float);
        const size_t out_bytes = size_t(hidden) * size_t(seq_len) * sizeof(float);

        auto & input_buf = create_host_buffer(input_bytes);
        auto & norm_buf = create_host_buffer(norm_bytes);
        auto & gate_buf = create_host_buffer(gate_bytes);
        auto & up_buf = create_host_buffer(up_bytes);
        auto & down_buf = create_host_buffer(down_bytes);
        auto & mlp_down_buf = create_host_buffer(out_bytes);
        auto & layer_out_buf = create_host_buffer(out_bytes);

        std::memcpy(input_buf.mapped, attn_residual.data(), input_bytes);
        std::memcpy(norm_buf.mapped, norm_weight.data(), norm_bytes);
        std::memcpy(gate_buf.mapped, gate_weight.data(), gate_bytes);
        std::memcpy(up_buf.mapped, up_weight.data(), up_bytes);
        std::memcpy(down_buf.mapped, down_weight.data(), down_bytes);
        std::memset(mlp_down_buf.mapped, 0, out_bytes);
        std::memset(layer_out_buf.mapped, 0, out_bytes);

        VkDescriptorBufferInfo infos[7]{};
        const VkBufferHost * host_bufs[7] = { &input_buf, &norm_buf, &gate_buf, &up_buf, &down_buf, &mlp_down_buf, &layer_out_buf };
        VkWriteDescriptorSet writes[7]{};
        for (uint32_t i = 0; i < 7; ++i) {
            infos[i].buffer = host_bufs[i]->buffer;
            infos[i].offset = 0;
            infos[i].range = host_bufs[i]->size;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

        const VkMLPFusedPush push {
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(intermediate),
            static_cast<uint32_t>(seq_len),
            eps,
        };
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(command_buffer, 1, static_cast<uint32_t>(seq_len), 1);
        check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        check_vk(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
        check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        check_vk(vkResetFences(device, 1, &fence), "vkResetFences");
        check_vk(vkResetCommandPool(device, command_pool, 0), "vkResetCommandPool");

        std::vector<float> mlp_down(size_t(hidden) * size_t(seq_len));
        std::vector<float> layer_out(size_t(hidden) * size_t(seq_len));
        std::memcpy(mlp_down.data(), mlp_down_buf.mapped, out_bytes);
        std::memcpy(layer_out.data(), layer_out_buf.mapped, out_bytes);
        return {std::move(mlp_down), std::move(layer_out)};
    }
};

Ch13QKFusedResult run_qk_custom_vulkan_path(
    Ch13RuntimeGGUF & runtime,
    const Ch13QKFusedConfig & cfg,
    const std::vector<float> & attn_input) {
    if (cfg.head_dim > 128) {
        fail("custom qk fused shader currently supports head_dim <= 128");
    }
    if ((int)attn_input.size() != cfg.hidden * cfg.seq_len) {
        fail("run_qk_custom_vulkan_path: attn_input shape mismatch");
    }

    ggml_tensor * q_weight_t = runtime.get_tensor(layer_key(cfg.layer, "q_proj.weight").c_str());
    ggml_tensor * k_weight_t = runtime.get_tensor(layer_key(cfg.layer, "k_proj.weight").c_str());
    ggml_tensor * q_norm_t = runtime.get_tensor(layer_key(cfg.layer, "q_norm.weight").c_str());
    ggml_tensor * k_norm_t = runtime.get_tensor(layer_key(cfg.layer, "k_norm.weight").c_str());

    if ((int)q_weight_t->ne[0] != cfg.hidden || (int)q_weight_t->ne[1] != cfg.head_dim * cfg.n_head) {
        fail("unexpected q_proj weight shape for custom qk fused path");
    }
    if ((int)k_weight_t->ne[0] != cfg.hidden || (int)k_weight_t->ne[1] != cfg.head_dim * cfg.n_kv_head) {
        fail("unexpected k_proj weight shape for custom qk fused path");
    }
    if ((int)q_norm_t->ne[0] != cfg.head_dim || q_norm_t->ne[1] != 1) {
        fail("unexpected q_norm weight shape for custom qk fused path");
    }
    if ((int)k_norm_t->ne[0] != cfg.head_dim || k_norm_t->ne[1] != 1) {
        fail("unexpected k_norm weight shape for custom qk fused path");
    }

    static LocalVulkanQK vkqk;
    static bool vkqk_ready = false;
    if (!vkqk_ready) {
        vkqk.init();
        vkqk_ready = true;
    }
    return vkqk.run(
        cfg,
        attn_input,
        tensor_to_f32_expanded(q_weight_t),
        tensor_to_f32_expanded(k_weight_t),
        tensor_to_f32(q_norm_t),
        tensor_to_f32(k_norm_t));
}

std::pair<std::vector<float>, std::vector<float>> run_mlp_custom_vulkan_path(
    Ch13RuntimeGGUF & runtime,
    int layer,
    const std::vector<float> & attn_residual) {
    const int hidden = (int) runtime.get_u32("runtime.hidden_size");
    const int intermediate = (int) runtime.get_u32("runtime.intermediate_size");
    if ((int) attn_residual.size() % hidden != 0) {
        fail("run_mlp_custom_vulkan_path: attn_residual shape mismatch");
    }
    const int seq_len = (int) attn_residual.size() / hidden;

    ggml_tensor * norm_t = runtime.get_tensor(layer_key(layer, "post_attention_norm.weight").c_str());
    ggml_tensor * gate_t = runtime.get_tensor(layer_key(layer, "gate_proj.weight").c_str());
    ggml_tensor * up_t = runtime.get_tensor(layer_key(layer, "up_proj.weight").c_str());
    ggml_tensor * down_t = runtime.get_tensor(layer_key(layer, "down_proj.weight").c_str());

    static LocalVulkanMLP vkmlp;
    static bool vkmlp_ready = false;
    if (!vkmlp_ready) {
        vkmlp.init();
        vkmlp_ready = true;
    }

    return vkmlp.run(
        hidden,
        intermediate,
        seq_len,
        runtime.get_f32("runtime.rms_norm_eps"),
        attn_residual,
        tensor_to_f32(norm_t),
        tensor_to_f32_expanded(gate_t),
        tensor_to_f32_expanded(up_t),
        tensor_to_f32_expanded(down_t));
}

} // namespace

Ch13QKFusedResult ch13_run_qk_fused_path(
    Ch13RuntimeGGUF & runtime,
    const Ch13QKFusedConfig & cfg,
    const std::vector<float> & attn_input,
    const std::string & backend_name) {
    if (backend_name == "cpu" || backend_name == "gpu-ggml") {
        return run_qk_graph_path(runtime, cfg, attn_input, backend_name == "cpu" ? "cpu" : "gpu");
    }

    return run_qk_custom_vulkan_path(runtime, cfg, attn_input);
}

Ch13LayerFusedQKResult ch13_run_layer_with_fused_qk(
    Ch13RuntimeGGUF & rt,
    int il,
    const std::vector<float> & layer_input) {
    const uint32_t hidden = rt.get_u32("runtime.hidden_size");
    const uint32_t head_dim = rt.get_u32("runtime.head_dim");
    const uint32_t n_head = rt.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt.get_u32("runtime.n_kv_head");
    if (layer_input.size() % hidden != 0) {
        fail("ch13_run_layer_with_fused_qk: layer_input size is not divisible by hidden");
    }
    const uint32_t seq_len = static_cast<uint32_t>(layer_input.size() / hidden);
    const float eps = rt.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    if (layer_input.size() != static_cast<size_t>(hidden) * static_cast<size_t>(seq_len)) {
        fail("ch13_run_layer_with_fused_qk: layer_input shape mismatch");
    }

    // Stage A: attn_input + v_proj
    LocalCtx stage_a;
    stage_a.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!stage_a.ctx) fail("failed to init fused stage_a ctx");
    ggml_cgraph * gf_a = ggml_new_graph_custom(stage_a.ctx, 128, false);
    if (!gf_a) fail("failed to init fused stage_a graph");

    ggml_tensor * inp_a = ggml_new_tensor_2d(stage_a.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp_a, "fused_layer.input");
    ggml_set_input(inp_a);

    ggml_tensor * attn_input = ggml_mul(stage_a.ctx,
        ggml_rms_norm(stage_a.ctx, inp_a, eps),
        ggml_repeat(stage_a.ctx, rt.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp_a));
    ggml_tensor * v_proj = ggml_mul_mat(stage_a.ctx, rt.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input);
    ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);
    ggml_set_name(attn_input, "fused_layer.attn_input");
    ggml_set_name(v_proj, "fused_layer.v_proj");
    ggml_set_output(attn_input);
    ggml_set_output(v_proj);
    ggml_build_forward_expand(gf_a, attn_input);
    ggml_build_forward_expand(gf_a, v_proj);

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf_a)) fail("failed to alloc fused stage_a graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_a, "fused_layer.input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(rt.sched, gf_a) != GGML_STATUS_SUCCESS) fail("failed to compute fused stage_a graph");
    const auto attn_input_vec = tensor_to_f32(ggml_graph_get_tensor(gf_a, "fused_layer.attn_input"));
    const auto v_proj_vec = tensor_to_f32(ggml_graph_get_tensor(gf_a, "fused_layer.v_proj"));
    ggml_backend_sched_reset(rt.sched);

    Ch13QKFusedConfig cfg;
    cfg.layer = il;
    cfg.hidden = (int) hidden;
    cfg.head_dim = (int) head_dim;
    cfg.n_head = (int) n_head;
    cfg.n_kv_head = (int) n_kv_head;
    cfg.seq_len = (int) seq_len;
    cfg.rms_norm_eps = eps;
    cfg.rope_freq_base = rope_base;
    cfg.rope_n_ctx_orig = rope_ctx;
    const auto qk = ch13_run_qk_fused_path(rt, cfg, attn_input_vec, "gpu");

    // Stage B: attention + MLP
    LocalCtx stage_b;
    stage_b.ctx = ggml_init({48 * 1024 * 1024, nullptr, true});
    if (!stage_b.ctx) fail("failed to init fused stage_b ctx");
    ggml_cgraph * gf_b = ggml_new_graph_custom(stage_b.ctx, 256, false);
    if (!gf_b) fail("failed to init fused stage_b graph");

    ggml_tensor * inp_b = ggml_new_tensor_2d(stage_b.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp_b, "fused_layer_b.input");
    ggml_set_input(inp_b);
    ggml_tensor * q_in = ggml_new_tensor_3d(stage_b.ctx, GGML_TYPE_F32, head_dim, n_head, seq_len);
    ggml_set_name(q_in, "fused_layer_b.q_in");
    ggml_set_input(q_in);
    ggml_tensor * k_in = ggml_new_tensor_3d(stage_b.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(k_in, "fused_layer_b.k_in");
    ggml_set_input(k_in);
    ggml_tensor * v_in = ggml_new_tensor_3d(stage_b.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(v_in, "fused_layer_b.v_in");
    ggml_set_input(v_in);

    ggml_tensor * q_mat = ggml_dup(stage_b.ctx, q_in);
    ggml_tensor * k_mat = ggml_dup(stage_b.ctx, k_in);
    ggml_tensor * q_attn = ggml_permute(stage_b.ctx, q_mat, 0, 2, 1, 3);
    ggml_tensor * k_attn = ggml_permute(stage_b.ctx, k_mat, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(stage_b.ctx, v_in, 0, 2, 1, 3);

    ggml_tensor * attn = ggml_flash_attn_ext(stage_b.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor * attn_out = ggml_cont(stage_b.ctx, ggml_reshape_2d(stage_b.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
    ggml_tensor * o_proj = ggml_mul_mat(stage_b.ctx, rt.get_tensor(layer_key(il, "o_proj.weight").c_str()), attn_out);
    ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
    ggml_tensor * attn_residual = ggml_add(stage_b.ctx, inp_b, o_proj);
    ggml_tensor * mlp_in = ggml_mul(stage_b.ctx,
        ggml_rms_norm(stage_b.ctx, attn_residual, eps),
        ggml_repeat(stage_b.ctx, rt.get_tensor(layer_key(il, "post_attention_norm.weight").c_str()), attn_residual));
    ggml_tensor * gate = ggml_mul_mat(stage_b.ctx, rt.get_tensor(layer_key(il, "gate_proj.weight").c_str()), mlp_in);
    ggml_tensor * up = ggml_mul_mat(stage_b.ctx, rt.get_tensor(layer_key(il, "up_proj.weight").c_str()), mlp_in);
    ggml_mul_mat_set_prec(gate, GGML_PREC_F32);
    ggml_mul_mat_set_prec(up, GGML_PREC_F32);
    ggml_tensor * mlp_act = ggml_swiglu_split(stage_b.ctx, gate, up);
    ggml_tensor * mlp_down = ggml_mul_mat(stage_b.ctx, rt.get_tensor(layer_key(il, "down_proj.weight").c_str()), mlp_act);
    ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);
    ggml_tensor * layer_out = ggml_add(stage_b.ctx, attn_residual, mlp_down);

    ggml_set_name(attn_out, "fused_layer_b.attn_out");
    ggml_set_name(o_proj, "fused_layer_b.o_proj");
    ggml_set_name(attn_residual, "fused_layer_b.attn_residual");
    ggml_set_name(mlp_down, "fused_layer_b.mlp_down");
    ggml_set_name(layer_out, "fused_layer_b.layer_out");
    for (ggml_tensor * t : {attn_out, o_proj, attn_residual, mlp_down, layer_out}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf_b, t);
    }

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf_b)) fail("failed to alloc fused stage_b graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "fused_layer_b.input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "fused_layer_b.q_in"), qk.q_rope.data(), 0, qk.q_rope.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "fused_layer_b.k_in"), qk.k_rope.data(), 0, qk.k_rope.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "fused_layer_b.v_in"), v_proj_vec.data(), 0, v_proj_vec.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(rt.sched, gf_b) != GGML_STATUS_SUCCESS) fail("failed to compute fused stage_b graph");

    Ch13LayerFusedQKResult out {
        qk.q_rope,
        qk.k_rope,
        tensor_to_f32(ggml_graph_get_tensor(gf_b, "fused_layer_b.attn_out")),
        tensor_to_f32(ggml_graph_get_tensor(gf_b, "fused_layer_b.o_proj")),
        tensor_to_f32(ggml_graph_get_tensor(gf_b, "fused_layer_b.attn_residual")),
        tensor_to_f32(ggml_graph_get_tensor(gf_b, "fused_layer_b.mlp_down")),
        tensor_to_f32(ggml_graph_get_tensor(gf_b, "fused_layer_b.layer_out")),
    };
    ggml_backend_sched_reset(rt.sched);
    return out;
}

Ch13LayerFusedQKResult ch13_run_layer_plain_ggml(
    Ch13RuntimeGGUF & rt,
    int il,
    const std::vector<float> & layer_input) {
    const uint32_t hidden = rt.get_u32("runtime.hidden_size");
    const uint32_t head_dim = rt.get_u32("runtime.head_dim");
    const uint32_t n_head = rt.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt.get_u32("runtime.n_kv_head");
    if (layer_input.size() % hidden != 0) {
        fail("ch13_run_layer_plain_ggml: layer_input size is not divisible by hidden");
    }
    const uint32_t seq_len = static_cast<uint32_t>(layer_input.size() / hidden);
    const float eps = rt.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    LocalCtx ctx;
    ctx.ctx = ggml_init({64 * 1024 * 1024, nullptr, true});
    if (!ctx.ctx) fail("failed to init plain layer ctx");
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.ctx, 256, false);
    if (!gf) fail("failed to init plain layer graph");

    ggml_tensor * inp = ggml_new_tensor_2d(ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp, "plain_layer.input");
    ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos, "plain_layer.pos");
    ggml_set_input(pos);

    ggml_tensor * attn_input = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, inp, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp));
    ggml_tensor * q_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "q_proj.weight").c_str()), attn_input);
    ggml_tensor * k_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "k_proj.weight").c_str()), attn_input);
    ggml_tensor * v_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input);
    ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);
    ggml_tensor * q_3d = ggml_reshape_3d(ctx.ctx, q_proj, head_dim, n_head, seq_len);
    ggml_tensor * k_3d = ggml_reshape_3d(ctx.ctx, k_proj, head_dim, n_kv_head, seq_len);
    ggml_tensor * v_3d = ggml_reshape_3d(ctx.ctx, v_proj, head_dim, n_kv_head, seq_len);
    ggml_tensor * q_norm = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, q_3d, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "q_norm.weight").c_str()), q_3d));
    ggml_tensor * k_norm = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, k_3d, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "k_norm.weight").c_str()), k_3d));
    ggml_tensor * q_rope = ggml_rope_ext(ctx.ctx, q_norm, pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope = ggml_rope_ext(ctx.ctx, k_norm, pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q_rope = ggml_cont(ctx.ctx, q_rope);
    k_rope = ggml_cont(ctx.ctx, k_rope);
    ggml_tensor * q_attn = ggml_permute(ctx.ctx, q_rope, 0, 2, 1, 3);
    ggml_tensor * k_attn = ggml_permute(ctx.ctx, k_rope, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(ctx.ctx, v_3d, 0, 2, 1, 3);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor * attn_out = ggml_cont(ctx.ctx, ggml_reshape_2d(ctx.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
    ggml_tensor * o_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "o_proj.weight").c_str()), attn_out);
    ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
    ggml_tensor * attn_residual = ggml_add(ctx.ctx, inp, o_proj);
    ggml_tensor * mlp_in = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, attn_residual, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "post_attention_norm.weight").c_str()), attn_residual));
    ggml_tensor * gate = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "gate_proj.weight").c_str()), mlp_in);
    ggml_tensor * up = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "up_proj.weight").c_str()), mlp_in);
    ggml_mul_mat_set_prec(gate, GGML_PREC_F32);
    ggml_mul_mat_set_prec(up, GGML_PREC_F32);
    ggml_tensor * mlp_act = ggml_swiglu_split(ctx.ctx, gate, up);
    ggml_tensor * mlp_down = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "down_proj.weight").c_str()), mlp_act);
    ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);
    ggml_tensor * layer_out = ggml_add(ctx.ctx, attn_residual, mlp_down);

    for (ggml_tensor * t : {q_rope, k_rope, attn_out, o_proj, attn_residual, mlp_down, layer_out}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to alloc plain layer graph");
    ggml_backend_tensor_set(inp, layer_input.data(), 0, layer_input.size() * sizeof(float));
    std::vector<int32_t> pos_i(seq_len);
    for (uint32_t i = 0; i < seq_len; ++i) pos_i[i] = (int32_t) i;
    ggml_backend_tensor_set(pos, pos_i.data(), 0, pos_i.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute plain layer graph");

    Ch13LayerFusedQKResult out {
        tensor_to_f32(q_rope),
        tensor_to_f32(k_rope),
        tensor_to_f32(attn_out),
        tensor_to_f32(o_proj),
        tensor_to_f32(attn_residual),
        tensor_to_f32(mlp_down),
        tensor_to_f32(layer_out),
    };
    ggml_backend_sched_reset(rt.sched);
    return out;
}

static std::vector<float> ch13_run_backbone_prefix_plain(
    Ch13RuntimeGGUF & rt,
    const std::vector<float> & x_input,
    int end_layer_exclusive) {
    const uint32_t hidden = rt.get_u32("runtime.hidden_size");
    const uint32_t head_dim = rt.get_u32("runtime.head_dim");
    const uint32_t n_head = rt.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt.get_u32("runtime.n_kv_head");
    if (x_input.size() % hidden != 0) {
        fail("ch13_run_backbone_prefix_plain: x_input size is not divisible by hidden");
    }
    const uint32_t seq_len = static_cast<uint32_t>(x_input.size() / hidden);
    const float eps = rt.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    LocalCtx ctx;
    ctx.ctx = ggml_init({256 * 1024 * 1024, nullptr, true});
    if (!ctx.ctx) fail("failed to init prefix plain ctx");
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.ctx, 4096, false);
    if (!gf) fail("failed to init prefix plain graph");

    ggml_tensor * pos = ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos, "prefix_plain.pos");
    ggml_set_input(pos);

    ggml_tensor * cur = ggml_new_tensor_2d(ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(cur, "prefix_plain.input");
    ggml_set_input(cur);

    for (int il = 0; il < end_layer_exclusive; ++il) {
        ggml_tensor * attn_input = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, cur, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "attn_norm.weight").c_str()), cur));
        ggml_tensor * q_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "q_proj.weight").c_str()), attn_input);
        ggml_tensor * k_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "k_proj.weight").c_str()), attn_input);
        ggml_tensor * v_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input);
        ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
        ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);
        ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);
        ggml_tensor * q_3d = ggml_reshape_3d(ctx.ctx, q_proj, head_dim, n_head, seq_len);
        ggml_tensor * k_3d = ggml_reshape_3d(ctx.ctx, k_proj, head_dim, n_kv_head, seq_len);
        ggml_tensor * v_3d = ggml_reshape_3d(ctx.ctx, v_proj, head_dim, n_kv_head, seq_len);
        ggml_tensor * q_norm = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, q_3d, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "q_norm.weight").c_str()), q_3d));
        ggml_tensor * k_norm = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, k_3d, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "k_norm.weight").c_str()), k_3d));
        ggml_tensor * q_rope = ggml_rope_ext(ctx.ctx, q_norm, pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_rope = ggml_rope_ext(ctx.ctx, k_norm, pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q_rope = ggml_cont(ctx.ctx, q_rope);
        k_rope = ggml_cont(ctx.ctx, k_rope);
        ggml_tensor * q_attn = ggml_permute(ctx.ctx, q_rope, 0, 2, 1, 3);
        ggml_tensor * k_attn = ggml_permute(ctx.ctx, k_rope, 0, 2, 1, 3);
        ggml_tensor * v_attn = ggml_permute(ctx.ctx, v_3d, 0, 2, 1, 3);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        ggml_tensor * attn_out = ggml_cont(ctx.ctx, ggml_reshape_2d(ctx.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
        ggml_tensor * o_proj = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "o_proj.weight").c_str()), attn_out);
        ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
        ggml_tensor * attn_residual = ggml_add(ctx.ctx, cur, o_proj);
        ggml_tensor * mlp_in = ggml_mul(ctx.ctx, ggml_rms_norm(ctx.ctx, attn_residual, eps), ggml_repeat(ctx.ctx, rt.get_tensor(layer_key(il, "post_attention_norm.weight").c_str()), attn_residual));
        ggml_tensor * gate = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "gate_proj.weight").c_str()), mlp_in);
        ggml_tensor * up = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "up_proj.weight").c_str()), mlp_in);
        ggml_mul_mat_set_prec(gate, GGML_PREC_F32);
        ggml_mul_mat_set_prec(up, GGML_PREC_F32);
        ggml_tensor * mlp_act = ggml_swiglu_split(ctx.ctx, gate, up);
        ggml_tensor * mlp_down = ggml_mul_mat(ctx.ctx, rt.get_tensor(layer_key(il, "down_proj.weight").c_str()), mlp_act);
        ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);
        cur = ggml_add(ctx.ctx, attn_residual, mlp_down);
    }

    ggml_set_name(cur, "prefix_plain.out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to alloc prefix plain graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "prefix_plain.input"), x_input.data(), 0, x_input.size() * sizeof(float));
    std::vector<int32_t> pos_i(seq_len);
    for (uint32_t i = 0; i < seq_len; ++i) pos_i[i] = (int32_t) i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "prefix_plain.pos"), pos_i.data(), 0, pos_i.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute prefix plain graph");

    auto out = tensor_to_f32(ggml_graph_get_tensor(gf, "prefix_plain.out"));
    ggml_backend_sched_reset(rt.sched);
    return out;
}

Ch13BackboneFusedQKResult ch13_run_backbone_with_fused_qk(
    Ch13RuntimeGGUF & rt,
    const std::vector<float> & x_input,
    int fused_from_layer) {
    const uint32_t hidden = rt.get_u32("runtime.hidden_size");
    if (x_input.size() % hidden != 0) {
        fail("ch13_run_backbone_with_fused_qk: x_input size is not divisible by hidden");
    }
    const uint32_t seq_len = static_cast<uint32_t>(x_input.size() / hidden);
    const uint32_t n_layer = rt.get_u32("runtime.n_layer");
    const float eps = rt.get_f32("runtime.rms_norm_eps");

    if (x_input.size() != static_cast<size_t>(hidden) * static_cast<size_t>(seq_len)) {
        fail("ch13_run_backbone_with_fused_qk: x_input shape mismatch");
    }

    std::vector<float> cur = x_input;
    if (fused_from_layer > 0) {
        cur = ch13_run_backbone_prefix_plain(rt, x_input, fused_from_layer);
        for (uint32_t il = fused_from_layer; il < n_layer; ++il) {
            cur = ch13_run_layer_with_fused_qk(rt, static_cast<int>(il), cur).layer_out;
        }
    } else {
        for (uint32_t il = 0; il < n_layer; ++il) {
            cur = ch13_run_layer_with_fused_qk(rt, static_cast<int>(il), cur).layer_out;
        }
    }

    LocalCtx final_ctx;
    final_ctx.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!final_ctx.ctx) fail("failed to init fused backbone final ctx");
    ggml_cgraph * gf_final = ggml_new_graph_custom(final_ctx.ctx, 64, false);
    if (!gf_final) fail("failed to init fused backbone final graph");

    ggml_tensor * final_in = ggml_new_tensor_2d(final_ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(final_in, "runtime.backbone_fused.final_in");
    ggml_set_input(final_in);
    ggml_tensor * final_hidden = ggml_mul(final_ctx.ctx, ggml_rms_norm(final_ctx.ctx, final_in, eps), ggml_repeat(final_ctx.ctx, rt.get_tensor("runtime.output_norm.weight"), final_in));
    ggml_set_name(final_hidden, "runtime.backbone_fused.final_hidden");
    ggml_tensor * logits = ggml_mul_mat(final_ctx.ctx, rt.get_tensor("runtime.audio_heads.weight"), final_hidden);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_set_name(logits, "runtime.backbone_fused.logits");
    ggml_set_output(final_hidden);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf_final, final_hidden);
    ggml_build_forward_expand(gf_final, logits);

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf_final)) fail("failed to alloc fused backbone final graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_final, "runtime.backbone_fused.final_in"), cur.data(), 0, cur.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(rt.sched, gf_final) != GGML_STATUS_SUCCESS) fail("failed to compute fused backbone final graph");

    Ch13BackboneFusedQKResult out {
        tensor_to_f32(ggml_graph_get_tensor(gf_final, "runtime.backbone_fused.final_hidden")),
        tensor_to_f32(ggml_graph_get_tensor(gf_final, "runtime.backbone_fused.logits")),
    };
    ggml_backend_sched_reset(rt.sched);
    return out;
}
