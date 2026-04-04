#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct PushConstants {
    uint32_t element_count;
    uint32_t multiplier;
    uint32_t bias;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

[[noreturn]] void fail(const std::string &message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

void check_vk(VkResult result, const char *what) {
    if (result != VK_SUCCESS) {
        fail(std::string(what) + " failed with VkResult=" + std::to_string(static_cast<int>(result)));
    }
}

std::vector<uint32_t> read_binary_file(const char *path) {
    FILE *file = std::fopen(path, "rb");
    if (file == nullptr) {
        fail(std::string("failed to open file: ") + path);
    }
    std::fseek(file, 0, SEEK_END);
    long size = std::ftell(file);
    std::rewind(file);
    if (size <= 0 || (size % 4) != 0) {
        std::fclose(file);
        fail(std::string("invalid SPIR-V file size: ") + path);
    }
    std::vector<uint32_t> data(static_cast<size_t>(size) / 4);
    size_t read_count = std::fread(data.data(), 1, static_cast<size_t>(size), file);
    std::fclose(file);
    if (read_count != static_cast<size_t>(size)) {
        fail(std::string("failed to read file: ") + path);
    }
    return data;
}

uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t type_bits,
    VkMemoryPropertyFlags required_flags
) {
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

Buffer create_host_visible_storage_buffer(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size
) {
    Buffer out{};

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
    return out;
}

uint32_t baseline_value(const std::vector<uint32_t>& input, uint32_t idx, uint32_t multiplier, uint32_t bias) {
    return input[idx] * multiplier + bias;
}

uint32_t memory_heavy_value(const std::vector<uint32_t>& input, uint32_t idx, uint32_t multiplier, uint32_t bias) {
    const uint32_t count = static_cast<uint32_t>(input.size());
    const uint32_t a = input[idx];
    const uint32_t b = input[(idx * 13u + 7u) % count];
    const uint32_t c = input[(idx + 17u) % count];
    const uint32_t d = input[(idx ^ 31u) % count];
    const uint32_t mixed = ((a + b) ^ (c * 3u + d * 5u));
    return mixed * multiplier + bias;
}

uint32_t sync_heavy_value(const std::vector<uint32_t>& input, uint32_t idx, uint32_t multiplier, uint32_t bias) {
    constexpr uint32_t workgroup_size = 64;
    const uint32_t group_base = (idx / workgroup_size) * workgroup_size;
    const uint32_t local_id = idx % workgroup_size;
    const uint32_t neighbor_id = group_base + ((local_id + 1u) % workgroup_size);
    const uint32_t base = input[idx] * multiplier + bias;
    const uint32_t neighbor = input[neighbor_id] * multiplier + bias;
    return base + (neighbor & 1u);
}

}  // namespace

int main(int argc, char **argv) {
    const char *shader_path = (argc >= 2) ? argv[1] : "./build/shaders/baseline.comp.spv";
    const char *variant = (argc >= 3) ? argv[2] : "baseline";
    const uint32_t dispatch_repeats = (argc >= 4) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 128u;

    constexpr uint32_t kElementCount = 256;
    constexpr uint32_t kMultiplier = 3;
    constexpr uint32_t kBias = 7;
    const VkDeviceSize buffer_size = sizeof(uint32_t) * kElementCount;
    std::vector<uint32_t> input_values(kElementCount);
    for (uint32_t i = 0; i < kElementCount; ++i) {
        input_values[i] = i * 17u + 3u;
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "vulkan-to-7900xtx-isa-ch03";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "none";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    uint32_t physical_device_count = 0;
    check_vk(vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr), "vkEnumeratePhysicalDevices(count)");
    if (physical_device_count == 0) fail("no Vulkan physical devices found");

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    check_vk(vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data()), "vkEnumeratePhysicalDevices(list)");

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t queue_family_index = UINT32_MAX;
    VkPhysicalDeviceProperties picked_props{};
    for (VkPhysicalDevice candidate : physical_devices) {
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
        for (uint32_t i = 0; i < queue_count; ++i) {
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                physical_device = candidate;
                queue_family_index = i;
                vkGetPhysicalDeviceProperties(candidate, &picked_props);
                break;
            }
        }
        if (physical_device != VK_NULL_HANDLE) break;
    }
    if (physical_device == VK_NULL_HANDLE) fail("failed to find a Vulkan physical device with a compute queue");

    std::cout << "variant=" << variant
              << " device=" << picked_props.deviceName
              << " vendor=0x" << std::hex << picked_props.vendorID
              << " device=0x" << picked_props.deviceID
              << std::dec << " queue_family=" << queue_family_index << std::endl;

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

    Buffer input_buffer = create_host_visible_storage_buffer(physical_device, device, buffer_size);
    Buffer output_buffer = create_host_visible_storage_buffer(physical_device, device, buffer_size);
    {
        void *mapped = nullptr;
        check_vk(vkMapMemory(device, input_buffer.memory, 0, buffer_size, 0, &mapped), "vkMapMemory(input)");
        std::memcpy(mapped, input_values.data(), static_cast<size_t>(buffer_size));
        vkUnmapMemory(device, input_buffer.memory);
        check_vk(vkMapMemory(device, output_buffer.memory, 0, buffer_size, 0, &mapped), "vkMapMemory(output)");
        std::memset(mapped, 0, static_cast<size_t>(buffer_size));
        vkUnmapMemory(device, output_buffer.memory);
    }

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 2;
    set_layout_info.pBindings = bindings;

    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    check_vk(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

    std::vector<uint32_t> spirv = read_binary_file(shader_path);
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv.size() * sizeof(uint32_t);
    shader_info.pCode = spirv.data();

    VkShaderModule shader_module = VK_NULL_HANDLE;
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

    VkPipeline pipeline = VK_NULL_HANDLE;
    check_vk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines");

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

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check_vk(vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set), "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo input_info{};
    input_info.buffer = input_buffer.buffer;
    input_info.offset = 0;
    input_info.range = buffer_size;
    VkDescriptorBufferInfo output_info{};
    output_info.buffer = output_buffer.buffer;
    output_info.offset = 0;
    output_info.range = buffer_size;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &input_info;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &output_info;
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

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

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    check_vk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

    PushConstants push_constants{kElementCount, kMultiplier, kBias};
    constexpr uint32_t workgroup_size = 64;
    const uint32_t workgroup_count = (kElementCount + workgroup_size - 1) / workgroup_size;
    for (uint32_t i = 0; i < std::max(1u, dispatch_repeats); ++i) {
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(command_buffer, workgroup_count, 1, 1);
    }

    check_vk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check_vk(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    check_vk(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
    check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    uint32_t checksum = 0;
    {
        void *mapped = nullptr;
        check_vk(vkMapMemory(device, output_buffer.memory, 0, buffer_size, 0, &mapped), "vkMapMemory(readback)");
        auto* values = static_cast<uint32_t*>(mapped);
        for (uint32_t idx = 0; idx < kElementCount; ++idx) {
            uint32_t expected = 0;
            std::string variant_name = variant;
            if (variant_name == "baseline") {
                expected = baseline_value(input_values, idx, kMultiplier, kBias);
            } else if (variant_name == "memory_heavy") {
                expected = memory_heavy_value(input_values, idx, kMultiplier, kBias);
            } else if (variant_name == "sync_heavy") {
                expected = sync_heavy_value(input_values, idx, kMultiplier, kBias);
            } else {
                vkUnmapMemory(device, output_buffer.memory);
                fail(std::string("unknown variant: ") + variant_name);
            }
            if (values[idx] != expected) {
                vkUnmapMemory(device, output_buffer.memory);
                fail(
                    "buffer validation failed at index=" + std::to_string(idx) +
                    " expected=" + std::to_string(expected) +
                    " actual=" + std::to_string(values[idx]));
            }
            checksum += values[idx];
        }
        vkUnmapMemory(device, output_buffer.memory);
    }

    std::cout << "dispatch_ok variant=" << variant
              << " element_count=" << kElementCount
              << " workgroups=" << workgroup_count
              << " repeats=" << dispatch_repeats
              << " checksum=" << checksum << std::endl;

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    vkDestroyBuffer(device, input_buffer.buffer, nullptr);
    vkFreeMemory(device, input_buffer.memory, nullptr);
    vkDestroyBuffer(device, output_buffer.buffer, nullptr);
    vkFreeMemory(device, output_buffer.memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
