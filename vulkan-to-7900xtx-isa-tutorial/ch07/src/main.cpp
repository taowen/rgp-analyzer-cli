#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string shader_path;
    uint32_t seq_len = 64;
    uint32_t head_dim = 64;
    uint32_t dispatches = 64;
    uint32_t warmup = 2;
    uint32_t repeats = 8;
};

struct PushConstants {
    uint32_t seq_len;
    uint32_t head_dim;
    uint32_t repeats;
    float scale;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void check_vk(VkResult result, const char *what) {
    if (result != VK_SUCCESS) {
        fail(std::string(what) + " failed with VkResult=" + std::to_string(result));
    }
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("failed to open file: " + path);
    }

    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<size_t>(size));
    if (!input.read(data.data(), size)) {
        fail("failed to read file: " + path);
    }
    return data;
}

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    fail("failed to find suitable memory type");
}

Options parse_args(int argc, char **argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--shader") {
            options.shader_path = require_value("--shader");
        } else if (arg == "--seq-len") {
            options.seq_len = static_cast<uint32_t>(std::stoul(require_value("--seq-len")));
        } else if (arg == "--head-dim") {
            options.head_dim = static_cast<uint32_t>(std::stoul(require_value("--head-dim")));
        } else if (arg == "--dispatches") {
            options.dispatches = static_cast<uint32_t>(std::stoul(require_value("--dispatches")));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<uint32_t>(std::stoul(require_value("--warmup")));
        } else if (arg == "--repeats") {
            options.repeats = static_cast<uint32_t>(std::stoul(require_value("--repeats")));
        } else {
            fail("unknown argument: " + arg);
        }
    }

    if (options.shader_path.empty()) {
        fail("usage: ch07_attention --shader <file.spv> [--seq-len N] [--head-dim N] [--dispatches N] "
             "[--warmup N] [--repeats N]");
    }

    if (options.seq_len == 0 || options.seq_len > 64) {
        fail("seq_len must be in [1, 64] for this chapter");
    }
    if (options.head_dim == 0 || options.head_dim > 64) {
        fail("head_dim must be in [1, 64] for this chapter");
    }

    return options;
}

float input_value(uint32_t row, uint32_t col, float scale) {
    const float a = static_cast<float>((row * 13u + col * 17u) % 37u);
    const float b = static_cast<float>((row * 5u + col * 7u + 3u) % 29u);
    return scale * (0.15f * a - 0.11f * b);
}

void fill_attention_inputs(uint32_t seq_len, uint32_t head_dim, std::vector<float> &q, std::vector<float> &k,
                           std::vector<float> &v) {
    q.resize(static_cast<size_t>(seq_len) * head_dim);
    k.resize(static_cast<size_t>(seq_len) * head_dim);
    v.resize(static_cast<size_t>(seq_len) * head_dim);

    for (uint32_t row = 0; row < seq_len; ++row) {
        for (uint32_t col = 0; col < head_dim; ++col) {
            const size_t idx = static_cast<size_t>(row) * head_dim + col;
            q[idx] = input_value(row, col, 0.07f);
            k[idx] = input_value(col, row + 1u, 0.05f);
            v[idx] = input_value(row + col, col + 3u, 0.09f);
        }
    }
}

std::vector<float> cpu_attention_reference(const std::vector<float> &q, const std::vector<float> &k,
                                           const std::vector<float> &v, uint32_t seq_len, uint32_t head_dim) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> out(static_cast<size_t>(seq_len) * head_dim, 0.0f);
    std::vector<float> scores(seq_len, 0.0f);

    for (uint32_t row = 0; row < seq_len; ++row) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (uint32_t key = 0; key < seq_len; ++key) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += q[static_cast<size_t>(row) * head_dim + d] * k[static_cast<size_t>(key) * head_dim + d];
            }
            scores[key] = dot * scale;
            max_score = std::max(max_score, scores[key]);
        }

        float denom = 0.0f;
        for (uint32_t key = 0; key < seq_len; ++key) {
            scores[key] = std::exp(scores[key] - max_score);
            denom += scores[key];
        }

        for (uint32_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (uint32_t key = 0; key < seq_len; ++key) {
                const float weight = scores[key] / denom;
                acc += weight * v[static_cast<size_t>(key) * head_dim + d];
            }
            out[static_cast<size_t>(row) * head_dim + d] = acc;
        }
    }

    return out;
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

Buffer create_buffer(VkPhysicalDevice physical_device, VkDevice device, VkDeviceSize size) {
    Buffer result{};
    result.size = size;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check_vk(vkCreateBuffer(device, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        physical_device, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check_vk(vkAllocateMemory(device, &alloc_info, nullptr, &result.memory), "vkAllocateMemory");
    check_vk(vkBindBufferMemory(device, result.buffer, result.memory, 0), "vkBindBufferMemory");

    return result;
}

void upload_floats(VkDevice device, const Buffer &buffer, const std::vector<float> &values) {
    void *mapped = nullptr;
    check_vk(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped), "vkMapMemory(upload)");
    std::memcpy(mapped, values.data(), values.size() * sizeof(float));
    vkUnmapMemory(device, buffer.memory);
}

std::vector<float> download_floats(VkDevice device, const Buffer &buffer, size_t count) {
    void *mapped = nullptr;
    check_vk(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped), "vkMapMemory(download)");
    std::vector<float> out(count);
    std::memcpy(out.data(), mapped, out.size() * sizeof(float));
    vkUnmapMemory(device, buffer.memory);
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<char> shader_code = read_file(options.shader_path);

        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        fill_attention_inputs(options.seq_len, options.head_dim, q, k, v);
        const std::vector<float> expected = cpu_attention_reference(q, k, v, options.seq_len, options.head_dim);

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "ch07-attention";
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        VkInstance instance = VK_NULL_HANDLE;
        check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

        uint32_t device_count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (device_count == 0) {
            fail("no Vulkan physical devices found");
        }

        std::vector<VkPhysicalDevice> physical_devices(device_count);
        check_vk(vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data()),
                 "vkEnumeratePhysicalDevices(list)");

        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        uint32_t queue_family_index = UINT32_MAX;
        VkPhysicalDeviceProperties physical_props{};

        for (VkPhysicalDevice candidate : physical_devices) {
            vkGetPhysicalDeviceProperties(candidate, &physical_props);
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
            for (uint32_t i = 0; i < family_count; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physical_device = candidate;
                    queue_family_index = i;
                    break;
                }
            }
            if (physical_device != VK_NULL_HANDLE && physical_props.vendorID == 0x1002) {
                break;
            }
        }

        if (physical_device == VK_NULL_HANDLE || queue_family_index == UINT32_MAX) {
            fail("failed to find compute-capable Vulkan device");
        }

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.pEnabledFeatures = &features;

        VkDevice device = VK_NULL_HANDLE;
        check_vk(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");

        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, queue_family_index, 0, &queue);

        const size_t matrix_elements = static_cast<size_t>(options.seq_len) * options.head_dim;
        const VkDeviceSize matrix_size = static_cast<VkDeviceSize>(matrix_elements * sizeof(float));

        Buffer q_buffer = create_buffer(physical_device, device, matrix_size);
        Buffer k_buffer = create_buffer(physical_device, device, matrix_size);
        Buffer v_buffer = create_buffer(physical_device, device, matrix_size);
        Buffer out_buffer = create_buffer(physical_device, device, matrix_size);

        upload_floats(device, q_buffer, q);
        upload_floats(device, k_buffer, k);
        upload_floats(device, v_buffer, v);
        upload_floats(device, out_buffer, std::vector<float>(matrix_elements, 0.0f));

        std::vector<VkDescriptorSetLayoutBinding> bindings(4);
        for (uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo set_layout_info{};
        set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        set_layout_info.pBindings = bindings.data();

        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout),
                 "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout),
                 "vkCreatePipelineLayout");

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = shader_code.size();
        shader_info.pCode = reinterpret_cast<const uint32_t *>(shader_code.data());

        VkShaderModule shader_module = VK_NULL_HANDLE;
        check_vk(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stage_info{};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = shader_module;
        stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        check_vk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
                 "vkCreateComputePipelines");

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = static_cast<uint32_t>(bindings.size());

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo alloc_set_info{};
        alloc_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_set_info.descriptorPool = descriptor_pool;
        alloc_set_info.descriptorSetCount = 1;
        alloc_set_info.pSetLayouts = &set_layout;

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        check_vk(vkAllocateDescriptorSets(device, &alloc_set_info, &descriptor_set), "vkAllocateDescriptorSets");

        std::vector<VkDescriptorBufferInfo> buffer_infos(4);
        buffer_infos[0] = {q_buffer.buffer, 0, matrix_size};
        buffer_infos[1] = {k_buffer.buffer, 0, matrix_size};
        buffer_infos[2] = {v_buffer.buffer, 0, matrix_size};
        buffer_infos[3] = {out_buffer.buffer, 0, matrix_size};

        std::vector<VkWriteDescriptorSet> writes(4);
        for (uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.queueFamilyIndex = queue_family_index;

        VkCommandPool command_pool = VK_NULL_HANDLE;
        check_vk(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo command_buffer_info{};
        command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_info.commandPool = command_pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        check_vk(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");

        VkQueryPoolCreateInfo query_pool_info{};
        query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_pool_info.queryCount = 2;

        VkQueryPool query_pool = VK_NULL_HANDLE;
        check_vk(vkCreateQueryPool(device, &query_pool_info, nullptr, &query_pool), "vkCreateQueryPool");

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

        vkCmdResetQueryPool(command_buffer, query_pool, 0, 2);
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set,
                                0, nullptr);

        PushConstants push_constants{};
        push_constants.seq_len = options.seq_len;
        push_constants.head_dim = options.head_dim;
        push_constants.repeats = options.repeats;
        push_constants.scale = 1.0f / std::sqrt(static_cast<float>(options.head_dim));

        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                           &push_constants);

        for (uint32_t i = 0; i < options.dispatches; ++i) {
            vkCmdDispatch(command_buffer, options.seq_len, 1, 1);
        }

        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
        check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        auto run_submit = [&]() {
            check_vk(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
            check_vk(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        };

        for (uint32_t i = 0; i < options.warmup; ++i) {
            run_submit();
        }
        run_submit();

        uint64_t timestamps[2] = {};
        check_vk(vkGetQueryPoolResults(device, query_pool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                 "vkGetQueryPoolResults");

        const std::vector<float> output = download_floats(device, out_buffer, matrix_elements);

        float max_abs_error = 0.0f;
        double checksum = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            max_abs_error = std::max(max_abs_error, std::fabs(output[i] - expected[i]));
            checksum += output[i];
        }

        const double gpu_total_ns =
            static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(physical_props.limits.timestampPeriod);
        const double gpu_total_ms = gpu_total_ns * 1e-6;
        const double gpu_avg_dispatch_us = gpu_total_ns * 1e-3 / static_cast<double>(options.dispatches);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "device: " << physical_props.deviceName << "\n";
        std::cout << "shader: " << options.shader_path << "\n";
        std::cout << "seq_len: " << options.seq_len << "\n";
        std::cout << "head_dim: " << options.head_dim << "\n";
        std::cout << "dispatches: " << options.dispatches << "\n";
        std::cout << "warmup: " << options.warmup << "\n";
        std::cout << "repeats: " << options.repeats << "\n";
        std::cout << "timestamp_period_ns: " << physical_props.limits.timestampPeriod << "\n";
        std::cout << "gpu_total_ms: " << gpu_total_ms << "\n";
        std::cout << "gpu_avg_dispatch_us: " << gpu_avg_dispatch_us << "\n";
        std::cout << "max_abs_error: " << max_abs_error << "\n";
        std::cout << "checksum: " << checksum << "\n";
        if (max_abs_error > 1e-3f) {
            fail("attention output verification failed");
        }
        std::cout << "dispatch_ok shader=" << options.shader_path << " seq_len=" << options.seq_len
                  << " head_dim=" << options.head_dim << " dispatches=" << options.dispatches
                  << " repeats=" << options.repeats << " checksum=" << checksum << "\n";

        vkDestroyQueryPool(device, query_pool, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyBuffer(device, out_buffer.buffer, nullptr);
        vkFreeMemory(device, out_buffer.memory, nullptr);
        vkDestroyBuffer(device, v_buffer.buffer, nullptr);
        vkFreeMemory(device, v_buffer.memory, nullptr);
        vkDestroyBuffer(device, k_buffer.buffer, nullptr);
        vkFreeMemory(device, k_buffer.memory, nullptr);
        vkDestroyBuffer(device, q_buffer.buffer, nullptr);
        vkFreeMemory(device, q_buffer.memory, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
