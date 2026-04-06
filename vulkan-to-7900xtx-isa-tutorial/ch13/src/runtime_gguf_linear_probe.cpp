#include "ch13_gguf_runtime.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

struct ComputeCtx {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ~ComputeCtx() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx) ggml_free(ctx);
    }
};

std::vector<float> read_f32(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("failed to open file: " + path);
    in.seekg(0, std::ios::end);
    const size_t nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (nbytes % sizeof(float) != 0) fail("invalid f32 file: " + path);
    std::vector<float> out(nbytes / sizeof(float));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(nbytes));
    if (!in) fail("failed reading file: " + path);
    return out;
}

std::vector<float> tensor_to_f32(ggml_tensor * tensor) {
    std::vector<float> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

struct DiffStats {
    float max_abs_diff = 0.0f;
    double mean_abs_diff = 0.0;
};

DiffStats diff_stats(const std::vector<float> & got, const std::vector<float> & ref, const std::string & name) {
    if (got.size() != ref.size()) fail("size mismatch for " + name);
    DiffStats stats;
    double sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        stats.max_abs_diff = std::max(stats.max_abs_diff, d);
        sum += d;
    }
    stats.mean_abs_diff = got.empty() ? 0.0 : (sum / static_cast<double>(got.size()));
    return stats;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 6 && argc != 7) {
        fail("usage: ch13_runtime_gguf_linear_probe <runtime.gguf> <weight_tensor_name> <input_ref_f32.bin> <output_ref_f32.bin> <label> [gpu|cpu]");
    }

    const std::string gguf_path = argv[1];
    const std::string weight_name = argv[2];
    const std::string input_path = argv[3];
    const std::string output_path = argv[4];
    const std::string label = argv[5];
    const std::string backend_name = argc >= 7 ? argv[6] : "gpu";
    const bool use_cpu = backend_name == "cpu";

    Ch13RuntimeGGUF rt;
    rt.load(gguf_path.c_str(), use_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);

    ggml_tensor * weight = rt.get_tensor(weight_name.c_str());
    const int64_t in_dim = weight->ne[0];
    const int64_t out_dim = weight->ne[1];

    const auto input_ref = read_f32(input_path);
    if (input_ref.size() % static_cast<size_t>(in_dim) != 0) {
        fail("input does not match weight input dim");
    }
    const int64_t seq = static_cast<int64_t>(input_ref.size() / static_cast<size_t>(in_dim));

    ComputeCtx compute;
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    compute.ctx = ggml_init(params);
    if (!compute.ctx) fail("failed to init compute ctx");

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 64, false);
    if (!gf) fail("failed to init graph");

    ggml_tensor * inp = ggml_new_tensor_2d(compute.ctx, GGML_TYPE_F32, in_dim, seq);
    ggml_set_name(inp, "linear_probe.input");
    ggml_set_input(inp);

    ggml_tensor * out = ggml_mul_mat(compute.ctx, weight, inp);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "linear_probe.output");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    if (use_cpu) {
        compute.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        if (!compute.allocr) fail("failed to init CPU gallocr");
        if (!ggml_gallocr_alloc_graph(compute.allocr, gf)) fail("failed to alloc graph");
    } else {
        ggml_backend_sched_reset(rt.sched);
        if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to alloc graph");
    }

    ggml_tensor * inp_alloc = ggml_graph_get_tensor(gf, "linear_probe.input");
    if (!inp_alloc) fail("failed to find linear_probe.input");
    ggml_backend_tensor_set(inp_alloc, input_ref.data(), 0, input_ref.size() * sizeof(float));

    if (use_cpu) {
        if (ggml_backend_graph_compute(rt.backend, gf) != GGML_STATUS_SUCCESS) fail("failed to compute graph");
    } else {
        if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute graph");
    }

    const auto output_ref = read_f32(output_path);
    const auto diff = diff_stats(tensor_to_f32(out), output_ref, label);

    std::cout << "probe.backend=" << backend_name << "\n";
    std::cout << "probe.label=" << label << "\n";
    std::cout << "probe.out_dim=" << out_dim << "\n";
    std::cout << "probe.seq=" << seq << "\n";
    std::cout << "probe.max_abs_diff=" << diff.max_abs_diff << "\n";
    std::cout << "probe.mean_abs_diff=" << diff.mean_abs_diff << "\n";
    std::cout << "linear_probe_ok=true\n";

    if (!use_cpu) {
        ggml_backend_sched_reset(rt.sched);
    }
    return 0;
}
