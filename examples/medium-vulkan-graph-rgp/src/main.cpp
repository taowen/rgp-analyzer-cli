#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class SubmitMode {
    Batched,
    Phase,
    MultiCommandBuffer,
};

struct Options {
    std::string preprocess_shader;
    std::string mix_shader;
    std::string reduce_shader;
    uint32_t element_count = 1u << 20;
    uint32_t graph_iterations = 32;
    uint32_t warmup = 2;
    uint32_t scale = 16;
    uint32_t dispatches_per_phase = 6;
    uint32_t barrier_every = 2;
    bool use_secondary_command_buffers = false;
    bool enable_labels = true;
    SubmitMode submit_mode = SubmitMode::Batched;
};

struct PushConstants {
    uint32_t element_count;
    uint32_t iteration;
    uint32_t phase;
    uint32_t scale;
};

struct PipelineBundle {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const char* label = "";
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check_vk(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        fail(std::string(what) + " failed with VkResult=" + std::to_string(result));
    }
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("failed to open file: " + path);
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!input.read(data.data(), size)) fail("failed to read file: " + path);
    return data;
}

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    fail("failed to find suitable memory type");
}

SubmitMode parse_submit_mode(const std::string& value) {
    if (value == "batched") return SubmitMode::Batched;
    if (value == "phase") return SubmitMode::Phase;
    if (value == "multi-cmdbuf") return SubmitMode::MultiCommandBuffer;
    fail("unknown submit mode: " + value);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) fail(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--preprocess") {
            options.preprocess_shader = require_value("--preprocess");
        } else if (arg == "--mix") {
            options.mix_shader = require_value("--mix");
        } else if (arg == "--reduce") {
            options.reduce_shader = require_value("--reduce");
        } else if (arg == "--elements") {
            options.element_count = static_cast<uint32_t>(std::stoul(require_value("--elements")));
        } else if (arg == "--graph-iterations") {
            options.graph_iterations = static_cast<uint32_t>(std::stoul(require_value("--graph-iterations")));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<uint32_t>(std::stoul(require_value("--warmup")));
        } else if (arg == "--scale") {
            options.scale = static_cast<uint32_t>(std::stoul(require_value("--scale")));
        } else if (arg == "--dispatches-per-phase") {
            options.dispatches_per_phase = std::max(1u, static_cast<uint32_t>(std::stoul(require_value("--dispatches-per-phase"))));
        } else if (arg == "--barrier-every") {
            options.barrier_every = std::max(1u, static_cast<uint32_t>(std::stoul(require_value("--barrier-every"))));
        } else if (arg == "--submit-mode") {
            options.submit_mode = parse_submit_mode(require_value("--submit-mode"));
        } else if (arg == "--secondary-cmdbuf") {
            options.use_secondary_command_buffers = true;
        } else if (arg == "--no-labels") {
            options.enable_labels = false;
        } else {
            fail("unknown argument: " + arg);
        }
    }
    if (options.preprocess_shader.empty() || options.mix_shader.empty() || options.reduce_shader.empty()) {
        fail("usage: medium_graph_compute --preprocess <spv> --mix <spv> --reduce <spv> [--submit-mode batched|phase|multi-cmdbuf] [--elements N] [--graph-iterations N] [--warmup N] [--scale N] [--dispatches-per-phase N] [--barrier-every N] [--secondary-cmdbuf] [--no-labels]");
    }
    return options;
}

VkShaderModule create_shader_module(VkDevice device, const std::vector<char>& shader_code) {
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = shader_code.size();
    shader_info.pCode = reinterpret_cast<const uint32_t*>(shader_code.data());
    VkShaderModule shader_module = VK_NULL_HANDLE;
    check_vk(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");
    return shader_module;
}

VkPipeline create_compute_pipeline(VkDevice device, VkPipelineLayout pipeline_layout, VkShaderModule shader_module) {
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
    check_vk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines");
    return pipeline;
}

void add_memory_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr);
}

void dispatch_phase(
    VkCommandBuffer cmd,
    VkPipelineLayout pipeline_layout,
    const PipelineBundle& bundle,
    uint32_t group_count,
    const PushConstants& pc) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bundle.pipeline);
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmd, group_count, 1, 1);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "medium-vulkan-graph-rgp";
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        VkInstance instance = VK_NULL_HANDLE;
        check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

        uint32_t device_count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (device_count == 0) fail("no Vulkan physical devices found");
        std::vector<VkPhysicalDevice> physical_devices(device_count);
        check_vk(vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data()), "vkEnumeratePhysicalDevices(list)");

        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        uint32_t queue_family_index = std::numeric_limits<uint32_t>::max();
        for (VkPhysicalDevice candidate : physical_devices) {
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
            if (physical_device != VK_NULL_HANDLE) break;
        }
        if (physical_device == VK_NULL_HANDLE) fail("failed to find compute-capable Vulkan device");

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

        VkDevice device = VK_NULL_HANDLE;
        check_vk(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");

        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, queue_family_index, 0, &queue);

        const VkDeviceSize buffer_size = static_cast<VkDeviceSize>(options.element_count) * sizeof(uint32_t);

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = buffer_size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        std::array<VkBuffer, 2> buffers{VK_NULL_HANDLE, VK_NULL_HANDLE};
        std::array<VkDeviceMemory, 2> memories{VK_NULL_HANDLE, VK_NULL_HANDLE};
        for (size_t i = 0; i < buffers.size(); ++i) {
            check_vk(vkCreateBuffer(device, &buffer_info, nullptr, &buffers[i]), "vkCreateBuffer");
            VkMemoryRequirements memory_requirements{};
            vkGetBufferMemoryRequirements(device, buffers[i], &memory_requirements);
            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = memory_requirements.size;
            alloc_info.memoryTypeIndex = find_memory_type(
                physical_device,
                memory_requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            check_vk(vkAllocateMemory(device, &alloc_info, nullptr, &memories[i]), "vkAllocateMemory");
            check_vk(vkBindBufferMemory(device, buffers[i], memories[i], 0), "vkBindBufferMemory");
            void* mapped = nullptr;
            check_vk(vkMapMemory(device, memories[i], 0, buffer_size, 0, &mapped), "vkMapMemory");
            auto* values = static_cast<uint32_t*>(mapped);
            for (uint32_t j = 0; j < options.element_count; ++j) {
                values[j] = static_cast<uint32_t>((i + 1) * 1315423911u ^ j);
            }
            vkUnmapMemory(device, memories[i]);
        }

        VkDescriptorSetLayoutBinding bindings[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 2;
        layout_info.pBindings = bindings;

        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 2;

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
        alloc_set_info.pSetLayouts = &descriptor_set_layout;

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        check_vk(vkAllocateDescriptorSets(device, &alloc_set_info, &descriptor_set), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo buffer_infos[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            buffer_infos[i].buffer = buffers[i];
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = buffer_size;
        }

        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        std::array<PipelineBundle, 3> pipelines{};
        const std::array<std::pair<std::string, const char*>, 3> shaders{{
            {options.preprocess_shader, "phase_preprocess"},
            {options.mix_shader, "phase_mix"},
            {options.reduce_shader, "phase_reduce"},
        }};
        for (size_t i = 0; i < pipelines.size(); ++i) {
            const auto shader_code = read_file(shaders[i].first);
            pipelines[i].shader_module = create_shader_module(device, shader_code);
            pipelines[i].pipeline = create_compute_pipeline(device, pipeline_layout, pipelines[i].shader_module);
            pipelines[i].label = shaders[i].second;
        }

        VkCommandPoolCreateInfo cmd_pool_info{};
        cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmd_pool_info.queueFamilyIndex = queue_family_index;
        cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkCommandPool command_pool = VK_NULL_HANDLE;
        check_vk(vkCreateCommandPool(device, &cmd_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

        const uint32_t command_buffer_count = options.submit_mode == SubmitMode::MultiCommandBuffer ? 3u : 1u;
        VkCommandBufferAllocateInfo command_alloc_info{};
        command_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_alloc_info.commandPool = command_pool;
        command_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_alloc_info.commandBufferCount = command_buffer_count;

        std::vector<VkCommandBuffer> command_buffers(command_buffer_count);
        check_vk(vkAllocateCommandBuffers(device, &command_alloc_info, command_buffers.data()), "vkAllocateCommandBuffers");

        std::vector<VkCommandBuffer> secondary_command_buffers;
        if (options.use_secondary_command_buffers) {
            VkCommandBufferAllocateInfo secondary_alloc_info{};
            secondary_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            secondary_alloc_info.commandPool = command_pool;
            secondary_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            secondary_alloc_info.commandBufferCount = 3;
            secondary_command_buffers.resize(3);
            check_vk(vkAllocateCommandBuffers(device, &secondary_alloc_info, secondary_command_buffers.data()), "vkAllocateCommandBuffers(secondary)");
        }

        VkQueryPoolCreateInfo query_info{};
        query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = (options.graph_iterations + options.warmup + 1) * 8u;

        VkQueryPool query_pool = VK_NULL_HANDLE;
        check_vk(vkCreateQueryPool(device, &query_info, nullptr, &query_pool), "vkCreateQueryPool");

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        check_vk(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

        const uint32_t group_count = (options.element_count + 63u) / 64u;
        const uint32_t reduce_group_count = group_count;

        auto begin_cmd = [&](VkCommandBuffer cmd) {
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check_vk(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        };

        auto begin_secondary_cmd = [&](VkCommandBuffer cmd) {
            VkCommandBufferInheritanceInfo inheritance{};
            inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            begin_info.pInheritanceInfo = &inheritance;
            check_vk(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer(secondary)");
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        };

        auto end_and_submit = [&](VkCommandBuffer cmd) {
            check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
            check_vk(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
            check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            check_vk(vkResetFences(device, 1, &fence), "vkResetFences");
        };

        uint32_t query_index = 0;
        for (uint32_t iteration = 0; iteration < options.warmup + options.graph_iterations; ++iteration) {
            const bool measure = iteration >= options.warmup;
            const uint32_t logical_iteration = iteration - options.warmup;

            if (options.submit_mode == SubmitMode::Batched) {
                VkCommandBuffer cmd = command_buffers[0];
                check_vk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
                begin_cmd(cmd);
                if (measure) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_index++);
                for (uint32_t phase = 0; phase < pipelines.size(); ++phase) {
                    for (uint32_t dispatch_index = 0; dispatch_index < options.dispatches_per_phase; ++dispatch_index) {
                        PushConstants pc{options.element_count, logical_iteration, phase * 1024u + dispatch_index, options.scale + dispatch_index};
                        dispatch_phase(cmd, pipeline_layout, pipelines[phase], phase == 2 ? reduce_group_count : group_count, pc);
                        const bool phase_end = dispatch_index + 1 == options.dispatches_per_phase;
                        if (!phase_end && ((dispatch_index + 1) % options.barrier_every) == 0) add_memory_barrier(cmd);
                    }
                    if (phase + 1 < pipelines.size()) add_memory_barrier(cmd);
                }
                if (measure) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_index++);
                end_and_submit(cmd);
            } else if (options.submit_mode == SubmitMode::Phase) {
                for (uint32_t phase = 0; phase < pipelines.size(); ++phase) {
                    VkCommandBuffer cmd = command_buffers[0];
                    check_vk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
                    begin_cmd(cmd);
                    if (measure) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_index++);
                    for (uint32_t dispatch_index = 0; dispatch_index < options.dispatches_per_phase; ++dispatch_index) {
                        PushConstants pc{options.element_count, logical_iteration, phase * 1024u + dispatch_index, options.scale + dispatch_index};
                        dispatch_phase(cmd, pipeline_layout, pipelines[phase], phase == 2 ? reduce_group_count : group_count, pc);
                        if (dispatch_index + 1 < options.dispatches_per_phase && ((dispatch_index + 1) % options.barrier_every) == 0) {
                            add_memory_barrier(cmd);
                        }
                    }
                    if (measure) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_index++);
                    end_and_submit(cmd);
                }
            } else {
                for (uint32_t phase = 0; phase < pipelines.size(); ++phase) {
                    VkCommandBuffer cmd = command_buffers[phase];
                    check_vk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
                    begin_cmd(cmd);
                    if (measure) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_index++);
                    if (options.use_secondary_command_buffers) {
                        VkCommandBuffer secondary = secondary_command_buffers[phase];
                        check_vk(vkResetCommandBuffer(secondary, 0), "vkResetCommandBuffer(secondary)");
                        begin_secondary_cmd(secondary);
                        for (uint32_t dispatch_index = 0; dispatch_index < options.dispatches_per_phase; ++dispatch_index) {
                            PushConstants pc{options.element_count, logical_iteration, phase * 1024u + dispatch_index, options.scale + dispatch_index};
                            dispatch_phase(secondary, pipeline_layout, pipelines[phase], phase == 2 ? reduce_group_count : group_count, pc);
                            if (dispatch_index + 1 < options.dispatches_per_phase && ((dispatch_index + 1) % options.barrier_every) == 0) {
                                add_memory_barrier(secondary);
                            }
                        }
                        check_vk(vkEndCommandBuffer(secondary), "vkEndCommandBuffer(secondary)");
                        vkCmdExecuteCommands(cmd, 1, &secondary);
                    } else {
                        for (uint32_t dispatch_index = 0; dispatch_index < options.dispatches_per_phase; ++dispatch_index) {
                            PushConstants pc{options.element_count, logical_iteration, phase * 1024u + dispatch_index, options.scale + dispatch_index};
                            dispatch_phase(cmd, pipeline_layout, pipelines[phase], phase == 2 ? reduce_group_count : group_count, pc);
                            if (dispatch_index + 1 < options.dispatches_per_phase && ((dispatch_index + 1) % options.barrier_every) == 0) {
                                add_memory_barrier(cmd);
                            }
                        }
                    }
                    if (measure) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_index++);
                    check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
                }

                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());
                submit_info.pCommandBuffers = command_buffers.data();
                check_vk(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
                check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
                check_vk(vkResetFences(device, 1, &fence), "vkResetFences");
            }
        }

        std::vector<uint64_t> timestamps(query_index);
        if (!timestamps.empty()) {
            check_vk(vkGetQueryPoolResults(
                device,
                query_pool,
                0,
                query_index,
                timestamps.size() * sizeof(uint64_t),
                timestamps.data(),
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device, &properties);
        const double timestamp_period_ns = properties.limits.timestampPeriod;
        std::cout << "medium_graph_compute submit_mode="
                  << (options.submit_mode == SubmitMode::Batched ? "batched" :
                      options.submit_mode == SubmitMode::Phase ? "phase" : "multi-cmdbuf")
                  << " graph_iterations=" << options.graph_iterations
                  << " warmup=" << options.warmup
                  << " element_count=" << options.element_count
                  << " scale=" << options.scale
                  << " dispatches_per_phase=" << options.dispatches_per_phase
                  << " barrier_every=" << options.barrier_every
                  << " secondary_cmbuf=" << (options.use_secondary_command_buffers ? 1 : 0)
                  << '\n';
        if (query_index >= 2) {
            const double total_ns = static_cast<double>(timestamps.back() - timestamps.front()) * timestamp_period_ns;
            std::cout << "measured_gpu_ms=" << (total_ns / 1.0e6) << '\n';
        }

        vkDestroyFence(device, fence, nullptr);
        vkDestroyQueryPool(device, query_pool, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        for (const auto& pipeline : pipelines) {
            vkDestroyPipeline(device, pipeline.pipeline, nullptr);
            vkDestroyShaderModule(device, pipeline.shader_module, nullptr);
        }
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        for (size_t i = 0; i < buffers.size(); ++i) {
            vkDestroyBuffer(device, buffers[i], nullptr);
            vkFreeMemory(device, memories[i], nullptr);
        }
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
