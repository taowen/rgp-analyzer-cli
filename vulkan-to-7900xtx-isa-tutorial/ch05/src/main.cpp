#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct PushConstants {
    uint32_t matrix_size;
};

struct Half {
    uint16_t bits = 0;
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
    if (file == nullptr) fail(std::string("failed to open file: ") + path);
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
    if (read_count != static_cast<size_t>(size)) fail(std::string("failed to read file: ") + path);
    return data;
}

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) continue;
        if ((props.memoryTypes[i].propertyFlags & required_flags) == required_flags) return i;
    }
    fail("failed to find suitable Vulkan memory type");
}

Buffer create_host_visible_storage_buffer(VkPhysicalDevice physical_device, VkDevice device, VkDeviceSize size) {
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

uint16_t float_to_half_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x00001000u) >> 13));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x00001000u) >> 13));
}

void cpu_matmul(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c, uint32_t n) {
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < n; ++k) acc += a[row * n + k] * b[k * n + col];
            c[row * n + col] = acc;
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    const char *shader_path = (argc >= 2) ? argv[1] : "./build/shaders/wmma_tile16.comp.spv";
    const char *variant = (argc >= 3) ? argv[2] : "wmma_tile16";
    const uint32_t dispatch_repeats = (argc >= 4) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 32u;

    const bool use_row2 = std::strcmp(variant, "wmma_row2") == 0;
    const bool use_k2 = std::strcmp(variant, "wmma_k2") == 0;
    const bool use_tile16 = std::strcmp(variant, "wmma_tile16") == 0;
    if (!use_row2 && !use_k2 && !use_tile16) fail(std::string("unknown variant: ") + variant);

    constexpr uint32_t kMatrixSize = 32;
    constexpr uint32_t kElementCount = kMatrixSize * kMatrixSize;
    constexpr VkDeviceSize kInputBufferSize = sizeof(uint16_t) * kElementCount;
    constexpr VkDeviceSize kOutputBufferSize = sizeof(float) * kElementCount;

    std::vector<float> host_a_f32(kElementCount);
    std::vector<float> host_b_f32(kElementCount);
    std::vector<float> host_b_transposed(kElementCount);
    std::vector<Half> host_a(kElementCount);
    std::vector<Half> host_b(kElementCount);
    std::vector<float> host_c_expected(kElementCount, 0.0f);

    for (uint32_t i = 0; i < kElementCount; ++i) {
        host_a_f32[i] = static_cast<float>((i % 13u) + 1u);
        host_b_f32[i] = static_cast<float>(((i * 5u) % 17u) + 1u);
    }
    for (uint32_t row = 0; row < kMatrixSize; ++row) {
        for (uint32_t col = 0; col < kMatrixSize; ++col) {
            host_b_transposed[col * kMatrixSize + row] = host_b_f32[row * kMatrixSize + col];
        }
    }
    for (uint32_t i = 0; i < kElementCount; ++i) {
        host_a[i].bits = float_to_half_bits(host_a_f32[i]);
        host_b[i].bits = float_to_half_bits(host_b_transposed[i]);
    }
    cpu_matmul(host_a_f32, host_b_f32, host_c_expected, kMatrixSize);

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "vulkan-to-7900xtx-isa-ch05";
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

    uint32_t extension_count = 0;
    check_vk(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr), "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> extensions(extension_count);
    check_vk(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data()), "vkEnumerateDeviceExtensionProperties(list)");

    bool has_cooperative_matrix_extension = false;
    for (const auto &ext : extensions) {
        if (std::strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            has_cooperative_matrix_extension = true;
            break;
        }
    }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features{};
    cooperative_matrix_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &cooperative_matrix_features;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_cooperative_matrix_properties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

    std::vector<VkCooperativeMatrixPropertiesKHR> cooperative_matrix_props;
    if (has_cooperative_matrix_extension && get_cooperative_matrix_properties != nullptr) {
        uint32_t property_count = 0;
        check_vk(get_cooperative_matrix_properties(physical_device, &property_count, nullptr), "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(count)");
        cooperative_matrix_props.resize(property_count);
        for (auto &prop : cooperative_matrix_props) {
            prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            prop.pNext = nullptr;
        }
        check_vk(get_cooperative_matrix_properties(physical_device, &property_count, cooperative_matrix_props.data()), "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(list)");
    }

    std::cout << "cooperative_matrix_extension=" << (has_cooperative_matrix_extension ? 1 : 0)
              << " cooperative_matrix_feature=" << (cooperative_matrix_features.cooperativeMatrix ? 1 : 0)
              << " cooperative_matrix_proc=" << (get_cooperative_matrix_properties != nullptr ? 1 : 0)
              << " cooperative_matrix_property_count=" << cooperative_matrix_props.size()
              << std::endl;

    bool has_target_shape = false;
    for (size_t i = 0; i < cooperative_matrix_props.size(); ++i) {
        const auto &prop = cooperative_matrix_props[i];
        std::cout << "coopmat_property[" << i << "]"
                  << " M=" << prop.MSize
                  << " N=" << prop.NSize
                  << " K=" << prop.KSize
                  << " AType=" << prop.AType
                  << " BType=" << prop.BType
                  << " CType=" << prop.CType
                  << " ResultType=" << prop.ResultType
                  << " scope=" << prop.scope
                  << std::endl;
        if (prop.MSize == 16 &&
            prop.NSize == 16 &&
            prop.KSize == 16 &&
            prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            prop.scope == VK_SCOPE_SUBGROUP_KHR) {
            has_target_shape = true;
        }
    }
    if (!has_cooperative_matrix_extension || !cooperative_matrix_features.cooperativeMatrix || !has_target_shape) {
        fail("this chapter requires 16x16x16 f16->f32 subgroup cooperative matrix support");
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    const char *device_extensions[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = device_extensions;
    cooperative_matrix_features.pNext = nullptr;
    device_info.pNext = &cooperative_matrix_features;

    VkDevice device = VK_NULL_HANDLE;
    check_vk(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family_index, 0, &queue);

    Buffer buffer_a = create_host_visible_storage_buffer(physical_device, device, kInputBufferSize);
    Buffer buffer_b = create_host_visible_storage_buffer(physical_device, device, kInputBufferSize);
    Buffer buffer_c = create_host_visible_storage_buffer(physical_device, device, kOutputBufferSize);

    {
        void *mapped = nullptr;
        check_vk(vkMapMemory(device, buffer_a.memory, 0, kInputBufferSize, 0, &mapped), "vkMapMemory(A)");
        std::memcpy(mapped, host_a.data(), static_cast<size_t>(kInputBufferSize));
        vkUnmapMemory(device, buffer_a.memory);

        check_vk(vkMapMemory(device, buffer_b.memory, 0, kInputBufferSize, 0, &mapped), "vkMapMemory(B)");
        std::memcpy(mapped, host_b.data(), static_cast<size_t>(kInputBufferSize));
        vkUnmapMemory(device, buffer_b.memory);

        check_vk(vkMapMemory(device, buffer_c.memory, 0, kOutputBufferSize, 0, &mapped), "vkMapMemory(C)");
        std::memset(mapped, 0, static_cast<size_t>(kOutputBufferSize));
        vkUnmapMemory(device, buffer_c.memory);
    }

    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 3;
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
    pool_size.descriptorCount = 3;

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

    VkDescriptorBufferInfo info_a{buffer_a.buffer, 0, kInputBufferSize};
    VkDescriptorBufferInfo info_b{buffer_b.buffer, 0, kInputBufferSize};
    VkDescriptorBufferInfo info_c{buffer_c.buffer, 0, kOutputBufferSize};

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info_a;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &info_b;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &info_c;
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

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

    PushConstants push_constants{kMatrixSize};
    const uint32_t dispatch_x = use_row2 ? 1u : 2u;
    const uint32_t dispatch_y = 2u;

    for (uint32_t i = 0; i < std::max(1u, dispatch_repeats); ++i) {
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(command_buffer, dispatch_x, dispatch_y, 1);
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

    double checksum = 0.0;
    {
        void *mapped = nullptr;
        check_vk(vkMapMemory(device, buffer_c.memory, 0, kOutputBufferSize, 0, &mapped), "vkMapMemory(readback)");
        auto *values = static_cast<float *>(mapped);
        for (uint32_t i = 0; i < kElementCount; ++i) {
            float expected = host_c_expected[i];
            float actual = values[i];
            if (std::fabs(actual - expected) > 0.05f) {
                vkUnmapMemory(device, buffer_c.memory);
                fail("matrix validation failed at index=" + std::to_string(i) +
                     " expected=" + std::to_string(expected) +
                     " actual=" + std::to_string(actual));
            }
            checksum += actual;
        }
        vkUnmapMemory(device, buffer_c.memory);
    }

    std::cout << "dispatch_ok variant=" << variant
              << " matrix_size=" << kMatrixSize
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
    vkDestroyBuffer(device, buffer_a.buffer, nullptr);
    vkFreeMemory(device, buffer_a.memory, nullptr);
    vkDestroyBuffer(device, buffer_b.buffer, nullptr);
    vkFreeMemory(device, buffer_b.memory, nullptr);
    vkDestroyBuffer(device, buffer_c.buffer, nullptr);
    vkFreeMemory(device, buffer_c.memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
