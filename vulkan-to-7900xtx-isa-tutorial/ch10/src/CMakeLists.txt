cmake_minimum_required(VERSION 3.21)
project(ch10_ggml_isa LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml-build EXCLUDE_FROM_ALL)

add_executable(ch10_ggml_isa main.cpp)
target_include_directories(ch10_ggml_isa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/ggml/include)
target_link_libraries(ch10_ggml_isa PRIVATE ggml ggml-vulkan)
