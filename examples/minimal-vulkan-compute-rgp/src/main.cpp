#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class WorkloadMode {
    Single,
    MultiPipeline,
    MultiCommandBuffer,
    BarrierMix,
};

struct Options {
    std::string shader_path;
    std::string secondary_shader_path;
    uint32_t element_count = 4u * 1024u * 1024u;
    uint32_t dispatches = 80;
    uint32_t warmup = 5;
    uint32_t iterations = 64;
    uint32_t command_buffers = 1;
    uint32_t barrier_every = 2;
    WorkloadMode mode = WorkloadMode::Single;
};

struct PushConstants {
    uint32_t element_count;
    uint32_t seed;
    uint32_t iterations;
};

struct PipelineBundle {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::string label;
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
        if ((type_bits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    fail("failed to find suitable memory type");
}

WorkloadMode parse_mode(const std::string& value) {
    if (value == "single") return WorkloadMode::Single;
    if (value == "multi-pipeline") return WorkloadMode::MultiPipeline;
    if (value == "multi-cmdbuf") return WorkloadMode::MultiCommandBuffer;
    if (value == "barrier-mix") return WorkloadMode::BarrierMix;
    fail("unknown mode: " + value);
}

const char* mode_name(WorkloadMode mode) {
    switch (mode) {
        case WorkloadMode::Single: return "single";
        case WorkloadMode::MultiPipeline: return "multi-pipeline";
        case WorkloadMode::MultiCommandBuffer: return "multi-cmdbuf";
        case WorkloadMode::BarrierMix: return "barrier-mix";
    }
    return "unknown";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--shader") {
            options.shader_path = require_value("--shader");
        } else if (arg == "--shader-secondary") {
            options.secondary_shader_path = require_value("--shader-secondary");
        } else if (arg == "--elements") {
            options.element_count = static_cast<uint32_t>(std::stoul(require_value("--elements")));
        } else if (arg == "--dispatches") {
            options.dispatches = static_cast<uint32_t>(std::stoul(require_value("--dispatches")));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<uint32_t>(std::stoul(require_value("--warmup")));
        } else if (arg == "--iterations") {
            options.iterations = static_cast<uint32_t>(std::stoul(require_value("--iterations")));
        } else if (arg == "--mode") {
            options.mode = parse_mode(require_value("--mode"));
        } else if (arg == "--command-buffers") {
            options.command_buffers = std::max(1u, static_cast<uint32_t>(std::stoul(require_value("--command-buffers"))));
        } else if (arg == "--barrier-every") {
            options.barrier_every = std::max(1u, static_cast<uint32_t>(std::stoul(require_value("--barrier-every"))));
        } else {
            fail("unknown argument: " + arg);
        }
    }

    if (options.shader_path.empty()) {
        fail("usage: minimal_compute --shader <file.spv> [--shader-secondary <file.spv>] [--mode single|multi-pipeline|multi-cmdbuf|barrier-mix] [--command-buffers N] [--barrier-every N] [--elements N] [--dispatches N] [--warmup N] [--iterations N]");
    }

    if (options.secondary_shader_path.empty() && options.mode == WorkloadMode::MultiPipeline) {
        fail("--mode multi-pipeline requires --shader-secondary");
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<char> shader_code = read_file(options.shader_path);
        const std::vector<char> secondary_shader_code =
            options.secondary_shader_path.empty() ? shader_code : read_file(options.secondary_shader_path);

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "minimal-compute-rgp";
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
        check_vk(vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data()), "vkEnumeratePhysicalDevices(list)");

        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        uint32_t queue_family_index = std::numeric_limits<uint32_t>::max();
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
        if (physical_device == VK_NULL_HANDLE || queue_family_index == std::numeric_limits<uint32_t>::max()) {
            fail("failed to find compute-capable Vulkan device");
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

        VkBuffer buffer = VK_NULL_HANDLE;
        check_vk(vkCreateBuffer(device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");

        VkMemoryRequirements memory_requirements{};
        vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = memory_requirements.size;
        alloc_info.memoryTypeIndex = find_memory_type(
            physical_device,
            memory_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
        check_vk(vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory), "vkAllocateMemory");
        check_vk(vkBindBufferMemory(device, buffer, buffer_memory, 0), "vkBindBufferMemory");

        void* mapped = nullptr;
        check_vk(vkMapMemory(device, buffer_memory, 0, buffer_size, 0, &mapped), "vkMapMemory");
        auto* data = static_cast<uint32_t*>(mapped);
        for (uint32_t i = 0; i < options.element_count; ++i) {
            data[i] = i * 2654435761u;
        }
        vkUnmapMemory(device, buffer_memory);

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo set_layout_info{};
        set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_layout_info.bindingCount = 1;
        set_layout_info.pBindings = &binding;

        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

        std::vector<PipelineBundle> pipelines;
        pipelines.push_back({create_shader_module(device, shader_code), VK_NULL_HANDLE, "primary"});
        pipelines.back().pipeline = create_compute_pipeline(device, pipeline_layout, pipelines.back().shader_module);
        if (options.mode == WorkloadMode::MultiPipeline) {
            pipelines.push_back({create_shader_module(device, secondary_shader_code), VK_NULL_HANDLE, "secondary"});
            pipelines.back().pipeline = create_compute_pipeline(device, pipeline_layout, pipelines.back().shader_module);
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 1;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo descriptor_alloc_info{};
        descriptor_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_alloc_info.descriptorPool = descriptor_pool;
        descriptor_alloc_info.descriptorSetCount = 1;
        descriptor_alloc_info.pSetLayouts = &set_layout;

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        check_vk(vkAllocateDescriptorSets(device, &descriptor_alloc_info, &descriptor_set), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo descriptor_buffer_info{};
        descriptor_buffer_info.buffer = buffer;
        descriptor_buffer_info.offset = 0;
        descriptor_buffer_info.range = buffer_size;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &descriptor_buffer_info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.queueFamilyIndex = queue_family_index;

        VkCommandPool command_pool = VK_NULL_HANDLE;
        check_vk(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

        const uint32_t command_buffer_count =
            options.mode == WorkloadMode::MultiCommandBuffer ? std::max(2u, options.command_buffers) : std::max(1u, options.command_buffers);

        VkCommandBufferAllocateInfo cmd_alloc_info{};
        cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc_info.commandPool = command_pool;
        cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc_info.commandBufferCount = command_buffer_count;

        std::vector<VkCommandBuffer> command_buffers(command_buffer_count, VK_NULL_HANDLE);
        check_vk(vkAllocateCommandBuffers(device, &cmd_alloc_info, command_buffers.data()), "vkAllocateCommandBuffers");

        VkQueryPoolCreateInfo query_info{};
        query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = command_buffer_count * 2;

        VkQueryPool query_pool = VK_NULL_HANDLE;
        check_vk(vkCreateQueryPool(device, &query_info, nullptr, &query_pool), "vkCreateQueryPool");

        const uint32_t workgroup_size = 64;
        const uint32_t group_count = (options.element_count + workgroup_size - 1) / workgroup_size;

        auto pipeline_for_dispatch = [&](uint32_t global_dispatch_index) -> const PipelineBundle& {
            if (options.mode == WorkloadMode::MultiPipeline && pipelines.size() > 1) {
                return pipelines[global_dispatch_index % pipelines.size()];
            }
            return pipelines[0];
        };

        auto record_command_buffer = [&](VkCommandBuffer cmd, uint32_t command_buffer_index, uint32_t dispatch_base, uint32_t dispatch_count, uint32_t seed_base) {
            check_vk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check_vk(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");

            const uint32_t query_base = command_buffer_index * 2;
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_base + 0);

            for (uint32_t local_dispatch_index = 0; local_dispatch_index < dispatch_count; ++local_dispatch_index) {
                const uint32_t global_dispatch_index = dispatch_base + local_dispatch_index;
                const PipelineBundle& bundle = pipeline_for_dispatch(global_dispatch_index);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bundle.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

                PushConstants pc{options.element_count, seed_base + global_dispatch_index, options.iterations};
                vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, group_count, 1, 1);

                if (options.mode == WorkloadMode::BarrierMix && ((local_dispatch_index + 1) % options.barrier_every) == 0) {
                    VkBufferMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.buffer = buffer;
                    barrier.offset = 0;
                    barrier.size = buffer_size;
                    vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        0,
                        nullptr,
                        1,
                        &barrier,
                        0,
                        nullptr);
                }
            }

            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_base + 1);
            check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        };

        auto record_and_submit = [&](uint32_t dispatches, uint32_t seed_base) -> double {
            vkResetQueryPool(device, query_pool, 0, command_buffer_count * 2);

            const uint32_t dispatches_per_buffer = std::max(1u, (dispatches + command_buffer_count - 1) / command_buffer_count);
            uint32_t dispatched = 0;
            uint32_t active_command_buffers = 0;
            for (uint32_t i = 0; i < command_buffer_count && dispatched < dispatches; ++i) {
                const uint32_t remaining = dispatches - dispatched;
                const uint32_t current_dispatches = std::min(dispatches_per_buffer, remaining);
                record_command_buffer(command_buffers[i], i, dispatched, current_dispatches, seed_base);
                dispatched += current_dispatches;
                active_command_buffers = i + 1;
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            check_vk(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = active_command_buffers;
            submit_info.pCommandBuffers = command_buffers.data();
            check_vk(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
            check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            vkDestroyFence(device, fence, nullptr);

            std::vector<uint64_t> timestamps(static_cast<size_t>(active_command_buffers) * 2u);
            check_vk(
                vkGetQueryPoolResults(
                    device,
                    query_pool,
                    0,
                    active_command_buffers * 2,
                    timestamps.size() * sizeof(uint64_t),
                    timestamps.data(),
                    sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                "vkGetQueryPoolResults");

            double total_ms = 0.0;
            for (uint32_t i = 0; i < active_command_buffers; ++i) {
                const uint64_t begin = timestamps[static_cast<size_t>(i) * 2u + 0u];
                const uint64_t end = timestamps[static_cast<size_t>(i) * 2u + 1u];
                total_ms += static_cast<double>(end - begin) * static_cast<double>(physical_props.limits.timestampPeriod) / 1'000'000.0;
            }
            return total_ms;
        };

        if (options.warmup > 0) {
            record_and_submit(options.warmup, 1);
        }

        const double total_ms = record_and_submit(options.dispatches, 1000);

        std::cout << "device: " << physical_props.deviceName << "\n";
        std::cout << "mode: " << mode_name(options.mode) << "\n";
        std::cout << "shader_primary: " << options.shader_path << "\n";
        if (!options.secondary_shader_path.empty()) {
            std::cout << "shader_secondary: " << options.secondary_shader_path << "\n";
        }
        std::cout << "command_buffers: " << command_buffer_count << "\n";
        std::cout << "elements: " << options.element_count << "\n";
        std::cout << "dispatches: " << options.dispatches << "\n";
        std::cout << "iterations: " << options.iterations << "\n";
        std::cout << "timestamp_period_ns: " << physical_props.limits.timestampPeriod << "\n";
        std::cout << "gpu_total_ms: " << total_ms << "\n";
        std::cout << "gpu_avg_dispatch_us: " << (total_ms * 1000.0 / std::max(1u, options.dispatches)) << "\n";

        vkDestroyQueryPool(device, query_pool, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        for (const PipelineBundle& bundle : pipelines) {
            vkDestroyPipeline(device, bundle.pipeline, nullptr);
            vkDestroyShaderModule(device, bundle.shader_module, nullptr);
        }
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, buffer_memory, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
