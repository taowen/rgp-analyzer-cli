#include "ch13_gguf_runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

}

Ch13RuntimeGGUF::~Ch13RuntimeGGUF() {
    if (gguf_ctx) gguf_free(gguf_ctx);
    if (ctx) ggml_free(ctx);
    if (tmp_ctx) ggml_free(tmp_ctx);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (sched) ggml_backend_sched_free(sched);
    if (cpu_backend) ggml_backend_free(cpu_backend);
    if (backend) ggml_backend_free(backend);
}

void Ch13RuntimeGGUF::load(const char * path, enum ggml_backend_dev_type backend_type) {
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx =*/ &tmp_ctx,
    };
    gguf_ctx = gguf_init_from_file(path, params);
    if (!gguf_ctx || !tmp_ctx) {
        fail("failed to load gguf");
    }

    ggml_backend_load_all();

    backend = ggml_backend_init_by_type(backend_type, nullptr);
    if (!backend) {
        fail("failed to init backend");
    }

    ggml_backend_t backends[2] = { backend, nullptr };
    int n_backends = 1;

    if (backend_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!cpu_backend) {
            fail("failed to init cpu backend");
        }
        backends[1] = cpu_backend;
        n_backends = 2;
    }

    sched = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) {
        fail("failed to init backend scheduler");
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    ggml_init_params model_params = {
        /*.mem_size =*/ ggml_tensor_overhead() * (n_tensors + 16),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ctx = ggml_init(model_params);
    if (!ctx) {
        fail("failed to init model context");
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        if (!src) {
            fail(std::string("missing source tensor in tmp ctx: ") + name);
        }
        ggml_tensor * dst = ggml_dup_tensor(ctx, src);
        ggml_set_name(dst, name);
    }

    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fail("failed to allocate backend tensors");
    }
    if (!ch13_load_from_gguf_to_backend(path, ctx, gguf_ctx)) {
        fail("failed to copy gguf tensors into backend");
    }
}

uint32_t Ch13RuntimeGGUF::get_u32(const char * key) const {
    const auto id = gguf_find_key(gguf_ctx, key);
    if (id < 0) fail(std::string("missing key: ") + key);
    return gguf_get_val_u32(gguf_ctx, id);
}

float Ch13RuntimeGGUF::get_f32(const char * key) const {
    const auto id = gguf_find_key(gguf_ctx, key);
    if (id < 0) fail(std::string("missing key: ") + key);
    return gguf_get_val_f32(gguf_ctx, id);
}

std::string Ch13RuntimeGGUF::get_str(const char * key) const {
    const auto id = gguf_find_key(gguf_ctx, key);
    if (id < 0) fail(std::string("missing key: ") + key);
    return gguf_get_val_str(gguf_ctx, id);
}

ggml_tensor * Ch13RuntimeGGUF::get_tensor(const char * name) const {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fail(std::string("missing tensor: ") + name);
    }
    return t;
}

int64_t Ch13RuntimeGGUF::tensor_count() const {
    return gguf_get_n_tensors(gguf_ctx);
}

bool ch13_load_from_gguf_to_backend(const char * fname, ggml_context * dst_ctx, gguf_context * gguf_ctx) {
    FILE * f = fopen(fname, "rb");
    if (!f) return false;

    const size_t buf_size = 4 * 1024 * 1024;
    std::vector<char> buf(buf_size);

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * tensor = ggml_get_tensor(dst_ctx, name);
        if (!tensor) continue;

        const size_t offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
        if (fseek(f, static_cast<long>(offs), SEEK_SET) != 0) {
            fclose(f);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t nbytes_cpy = std::min(buf_size, nbytes - pos);
            if (fread(buf.data(), 1, nbytes_cpy, f) != nbytes_cpy) {
                fclose(f);
                return false;
            }
            ggml_backend_tensor_set(tensor, buf.data(), pos, nbytes_cpy);
        }
    }

    fclose(f);
    return true;
}

float ch13_checksum_tensor_f32(ggml_tensor * tensor) {
    const size_t n = ggml_nelements(tensor);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(tensor, buf.data(), 0, n * sizeof(float));
    double s = 0.0;
    for (float v : buf) s += v;
    return static_cast<float>(s);
}

int ch13_checksum_tensor_i32(ggml_tensor * tensor) {
    const size_t n = ggml_nelements(tensor);
    std::vector<int32_t> buf(n);
    ggml_backend_tensor_get(tensor, buf.data(), 0, n * sizeof(int32_t));
    long long s = 0;
    for (int32_t v : buf) s += v;
    return static_cast<int>(s);
}

int ch13_checksum_tensor_i8(ggml_tensor * tensor) {
    const size_t n = ggml_nelements(tensor);
    std::vector<int8_t> buf(n);
    ggml_backend_tensor_get(tensor, buf.data(), 0, n * sizeof(int8_t));
    long long s = 0;
    for (int8_t v : buf) s += v;
    return static_cast<int>(s);
}
