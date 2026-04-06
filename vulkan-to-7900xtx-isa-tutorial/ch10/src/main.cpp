#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kRowsA = 64;
constexpr int kColsA = 64;
constexpr int kRowsB = 16;
constexpr int kColsB = 64;

static void log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    std::fputs(text, stderr);
}

[[noreturn]] void fail(const std::string & message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

struct Model {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> graph_buffer;
};

std::vector<float> make_matrix_a() {
    std::vector<float> data(kRowsA * kColsA);
    for (int row = 0; row < kRowsA; ++row) {
        for (int col = 0; col < kColsA; ++col) {
            data[row * kColsA + col] = static_cast<float>(((row * 17 + col * 13) % 31) - 15) / 8.0f;
        }
    }
    return data;
}

std::vector<float> make_matrix_b_transposed() {
    std::vector<float> data(kRowsB * kColsB);
    for (int row = 0; row < kRowsB; ++row) {
        for (int col = 0; col < kColsB; ++col) {
            data[row * kColsB + col] = static_cast<float>(((row * 19 + col * 7) % 29) - 14) / 7.0f;
        }
    }
    return data;
}

double checksum_f64(const std::vector<float> & values) {
    double sum = 0.0;
    for (float v : values) {
        sum += static_cast<double>(v);
    }
    return sum;
}

void init_model(Model & model) {
    ggml_time_init();
    ggml_log_set(log_callback, nullptr);

    const int vk_device_count = ggml_backend_vk_get_device_count();
    if (vk_device_count <= 0) {
        fail("ggml Vulkan backend found no Vulkan devices");
    }

    char desc[256] = {};
    ggml_backend_vk_get_device_description(0, desc, sizeof(desc));

    model.backend = ggml_backend_vk_init(0);
    if (!model.backend) {
        fail("ggml_backend_vk_init(0) failed");
    }

    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!model.cpu_backend) {
        fail("ggml_backend_init_by_type(CPU) failed");
    }

    ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
    model.sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!model.sched) {
        fail("ggml_backend_sched_new failed");
    }

    std::cout << "ggml_backend=" << ggml_backend_name(model.backend) << std::endl;
    std::cout << "ggml_device_desc[0]=" << desc << std::endl;
}

ggml_cgraph * build_graph(Model & model) {
    const size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.graph_buffer.resize(buf_size);

    ggml_init_params params = {
        /* .mem_size   = */ buf_size,
        /* .mem_buffer = */ model.graph_buffer.data(),
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("ggml_init failed");
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kColsA, kRowsA);
    model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kColsB, kRowsB);

    ggml_set_name(model.a, "A");
    ggml_set_name(model.b, "B_transposed");

    ggml_tensor * result = ggml_mul_mat(ctx, model.a, model.b);
    ggml_set_name(result, "mul_mat_result");
    ggml_build_forward_expand(gf, result);

    const int n_nodes = ggml_graph_n_nodes(gf);
    std::cout << "graph_nodes=" << n_nodes << std::endl;
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        std::cout
            << "node" << i
            << "=" << ggml_op_name(node->op)
            << " name=" << ggml_get_name(node)
            << " result_ne=[" << node->ne[0] << "," << node->ne[1] << "," << node->ne[2] << "," << node->ne[3] << "]"
            << std::endl;
    }

    ggml_free(ctx);
    return gf;
}

std::vector<float> compute(Model & model, ggml_cgraph * gf, int repeats) {
    const std::vector<float> host_a = make_matrix_a();
    const std::vector<float> host_b = make_matrix_b_transposed();

    ggml_backend_sched_reset(model.sched);
    if (!ggml_backend_sched_alloc_graph(model.sched, gf)) {
        fail("ggml_backend_sched_alloc_graph failed");
    }

    ggml_backend_tensor_set(model.a, host_a.data(), 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, host_b.data(), 0, ggml_nbytes(model.b));

    for (int i = 0; i < repeats; ++i) {
        const enum ggml_status status = ggml_backend_sched_graph_compute(model.sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fail("ggml_backend_sched_graph_compute failed on repeat " + std::to_string(i));
        }
    }

    ggml_tensor * result = ggml_graph_node(gf, -1);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(result)));
    ggml_backend_tensor_get(result, out.data(), 0, ggml_nbytes(result));
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    const int repeats = (argc >= 2) ? std::max(1, std::atoi(argv[1])) : 16;

    Model model;
    init_model(model);
    ggml_cgraph * graph = build_graph(model);
    const std::vector<float> out = compute(model, graph, repeats);

    std::cout << "repeat_count=" << repeats << std::endl;
    std::cout << "checksum=" << checksum_f64(out) << std::endl;
    std::cout << "sample_out="
              << out[0] << ","
              << out[1] << ","
              << out[2] << ","
              << out[3] << std::endl;

    ggml_backend_sched_free(model.sched);
    ggml_backend_free(model.backend);
    ggml_backend_free(model.cpu_backend);
    return 0;
}
