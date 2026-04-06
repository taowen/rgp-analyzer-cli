#pragma once

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>

struct Ch13RuntimeGGUF {
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx = nullptr;
    ggml_context * tmp_ctx = nullptr;
    gguf_context * gguf_ctx = nullptr;

    ~Ch13RuntimeGGUF();

    void load(const char * path, enum ggml_backend_dev_type backend_type = GGML_BACKEND_DEVICE_TYPE_GPU);

    uint32_t get_u32(const char * key) const;
    float get_f32(const char * key) const;
    std::string get_str(const char * key) const;
    ggml_tensor * get_tensor(const char * name) const;
    int64_t tensor_count() const;
};

bool ch13_load_from_gguf_to_backend(const char * fname, ggml_context * dst_ctx, gguf_context * gguf_ctx);
float ch13_checksum_tensor_f32(ggml_tensor * tensor);
int ch13_checksum_tensor_i32(ggml_tensor * tensor);
int ch13_checksum_tensor_i8(ggml_tensor * tensor);
