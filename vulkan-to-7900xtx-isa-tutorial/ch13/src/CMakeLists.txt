cmake_minimum_required(VERSION 3.21)
project(ch13_runtime_gguf LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED COMPONENTS glslc)

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml-build EXCLUDE_FROM_ALL)

set(CH13_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${CH13_SHADER_DIR})

set(CH13_QK_FUSED_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/qk_fused.comp)
set(CH13_QK_FUSED_SHADER_SPV ${CH13_SHADER_DIR}/qk_fused.comp.spv)
set(CH13_MLP_FUSED_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mlp_fused.comp)
set(CH13_MLP_FUSED_SHADER_SPV ${CH13_SHADER_DIR}/mlp_fused.comp.spv)
add_custom_command(
    OUTPUT ${CH13_QK_FUSED_SHADER_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=compute --target-env=vulkan1.1 --target-spv=spv1.3 ${CH13_QK_FUSED_SHADER_SRC} -O -o ${CH13_QK_FUSED_SHADER_SPV}
    DEPENDS ${CH13_QK_FUSED_SHADER_SRC}
    COMMENT "Compiling ch13 fused qk Vulkan shader"
    VERBATIM)
add_custom_command(
    OUTPUT ${CH13_MLP_FUSED_SHADER_SPV}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=compute --target-env=vulkan1.1 --target-spv=spv1.3 ${CH13_MLP_FUSED_SHADER_SRC} -O -o ${CH13_MLP_FUSED_SHADER_SPV}
    DEPENDS ${CH13_MLP_FUSED_SHADER_SRC}
    COMMENT "Compiling ch13 fused mlp Vulkan shader"
    VERBATIM)
add_custom_target(ch13_shaders DEPENDS ${CH13_QK_FUSED_SHADER_SPV} ${CH13_MLP_FUSED_SHADER_SPV})

add_executable(ch13_pack_runtime_gguf pack_runtime_gguf.cpp)
target_include_directories(ch13_pack_runtime_gguf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_pack_runtime_gguf PRIVATE ggml ggml-base)

add_executable(ch13_inspect_runtime_gguf inspect_runtime_gguf.cpp)
target_include_directories(ch13_inspect_runtime_gguf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_inspect_runtime_gguf PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_smoke runtime_gguf_smoke.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_smoke PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_first_graph runtime_gguf_first_graph.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_first_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_first_graph PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_layer0 runtime_gguf_layer0.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_layer0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_layer0 PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_backbone runtime_gguf_backbone.cpp ch13_qk_fused_path.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_backbone PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_backbone PRIVATE ggml ggml-base Vulkan::Vulkan)
target_compile_definitions(ch13_runtime_gguf_backbone PRIVATE CH13_QK_FUSED_SPV_PATH=\"${CH13_QK_FUSED_SHADER_SPV}\" CH13_MLP_FUSED_SPV_PATH=\"${CH13_MLP_FUSED_SHADER_SPV}\")
add_dependencies(ch13_runtime_gguf_backbone ch13_shaders)

add_executable(ch13_runtime_gguf_layer_probe runtime_gguf_layer_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_layer_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_layer_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_mlp_probe runtime_gguf_mlp_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_mlp_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_mlp_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_attn_probe runtime_gguf_attn_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_attn_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_attn_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_linear_probe runtime_gguf_linear_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_linear_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_linear_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_attention_core_probe runtime_gguf_attention_core_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_attention_core_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_attention_core_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_rope_probe runtime_gguf_rope_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_rope_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_rope_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_attention_mix_probe runtime_gguf_attention_mix_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_attention_mix_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_attention_mix_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_attention_hybrid_probe runtime_gguf_attention_hybrid_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_attention_hybrid_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_attention_hybrid_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_attn_qk_f32w_probe runtime_gguf_attn_qk_f32w_probe.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_attn_qk_f32w_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_attn_qk_f32w_probe PRIVATE ggml ggml-base)

add_executable(ch13_runtime_gguf_backbone_tail_hybrid runtime_gguf_backbone_tail_hybrid.cpp ch13_qk_fused_path.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_backbone_tail_hybrid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_backbone_tail_hybrid PRIVATE ggml ggml-base Vulkan::Vulkan)
target_compile_definitions(ch13_runtime_gguf_backbone_tail_hybrid PRIVATE CH13_QK_FUSED_SPV_PATH=\"${CH13_QK_FUSED_SHADER_SPV}\" CH13_MLP_FUSED_SPV_PATH=\"${CH13_MLP_FUSED_SHADER_SPV}\")
add_dependencies(ch13_runtime_gguf_backbone_tail_hybrid ch13_shaders)

add_executable(ch13_runtime_qk_fused_probe runtime_qk_fused_probe.cpp ch13_qk_fused_path.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_qk_fused_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_qk_fused_probe PRIVATE ggml ggml-base Vulkan::Vulkan)
target_compile_definitions(ch13_runtime_qk_fused_probe PRIVATE CH13_QK_FUSED_SPV_PATH=\"${CH13_QK_FUSED_SHADER_SPV}\" CH13_MLP_FUSED_SPV_PATH=\"${CH13_MLP_FUSED_SHADER_SPV}\")
add_dependencies(ch13_runtime_qk_fused_probe ch13_shaders)

add_executable(ch13_runtime_gguf_attn_fused_qk_probe runtime_gguf_attn_fused_qk_probe.cpp ch13_qk_fused_path.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_attn_fused_qk_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_attn_fused_qk_probe PRIVATE ggml ggml-base Vulkan::Vulkan)
target_compile_definitions(ch13_runtime_gguf_attn_fused_qk_probe PRIVATE CH13_QK_FUSED_SPV_PATH=\"${CH13_QK_FUSED_SHADER_SPV}\" CH13_MLP_FUSED_SPV_PATH=\"${CH13_MLP_FUSED_SHADER_SPV}\")
add_dependencies(ch13_runtime_gguf_attn_fused_qk_probe ch13_shaders)

add_executable(ch13_runtime_gguf_layer_fused_qk_probe runtime_gguf_layer_fused_qk_probe.cpp ch13_qk_fused_path.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_layer_fused_qk_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_layer_fused_qk_probe PRIVATE ggml ggml-base Vulkan::Vulkan)
target_compile_definitions(ch13_runtime_gguf_layer_fused_qk_probe PRIVATE CH13_QK_FUSED_SPV_PATH=\"${CH13_QK_FUSED_SHADER_SPV}\" CH13_MLP_FUSED_SPV_PATH=\"${CH13_MLP_FUSED_SHADER_SPV}\")
add_dependencies(ch13_runtime_gguf_layer_fused_qk_probe ch13_shaders)

add_executable(ch13_runtime_gguf_iterative_decode runtime_gguf_iterative_decode.cpp ch13_qk_fused_path.cpp ch13_gguf_runtime.cpp)
target_include_directories(ch13_runtime_gguf_iterative_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch13_runtime_gguf_iterative_decode PRIVATE ggml ggml-base Vulkan::Vulkan)
target_compile_definitions(ch13_runtime_gguf_iterative_decode PRIVATE CH13_QK_FUSED_SPV_PATH=\"${CH13_QK_FUSED_SHADER_SPV}\" CH13_MLP_FUSED_SPV_PATH=\"${CH13_MLP_FUSED_SHADER_SPV}\")
add_dependencies(ch13_runtime_gguf_iterative_decode ch13_shaders)
