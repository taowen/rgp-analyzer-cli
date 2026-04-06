#include "ch13_gguf_runtime.h"

#include "ggml-alloc.h"

#include <algorithm>
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
    if (!in) fail("failed to open reference file: " + path);
    in.seekg(0, std::ios::end);
    const size_t nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (nbytes % sizeof(float) != 0) fail("invalid f32 ref size: " + path);
    std::vector<float> out(nbytes / sizeof(float));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(nbytes));
    if (!in) fail("failed reading reference file: " + path);
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

std::string layer_key(int il, const std::string & suffix) {
    return "runtime.layers." + std::string(il < 10 ? "0" : "") + std::to_string(il) + "." + suffix;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        fail("usage: ch13_runtime_gguf_mlp_probe <runtime.gguf> <layer_probe_dir> <layer_idx> [gpu|cpu]");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];
    const std::string backend_name = argc >= 5 ? argv[4] : "gpu";
    const bool use_cpu = backend_name == "cpu";

    Ch13RuntimeGGUF rt;
    rt.load(argv[1], use_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);

    ComputeCtx compute;
    ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    compute.ctx = ggml_init(params);
    if (!compute.ctx) fail("failed to init compute ctx");

    const uint32_t hidden = rt.get_u32("runtime.hidden_size");
    const uint32_t seq_len = rt.get_u32("runtime.cond_seq_len");
    const float eps = rt.get_f32("runtime.rms_norm_eps");

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 256, false);
    if (!gf) fail("failed to init compute graph");

    ggml_tensor * inp = ggml_new_tensor_2d(compute.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp, "runtime.mlp_probe.input");
    ggml_set_input(inp);

    ggml_tensor * post_attn_norm_weight = rt.get_tensor(layer_key(il, "post_attention_norm.weight").c_str());
    ggml_tensor * gate_proj_weight = rt.get_tensor(layer_key(il, "gate_proj.weight").c_str());
    ggml_tensor * up_proj_weight = rt.get_tensor(layer_key(il, "up_proj.weight").c_str());
    ggml_tensor * down_proj_weight = rt.get_tensor(layer_key(il, "down_proj.weight").c_str());

    ggml_tensor * post_attn_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, inp, eps), ggml_repeat(compute.ctx, post_attn_norm_weight, inp));
    ggml_tensor * gate_proj = ggml_mul_mat(compute.ctx, gate_proj_weight, post_attn_norm);
    ggml_tensor * up_proj = ggml_mul_mat(compute.ctx, up_proj_weight, post_attn_norm);
    ggml_mul_mat_set_prec(gate_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(up_proj, GGML_PREC_F32);
    ggml_tensor * gate_silu = ggml_silu(compute.ctx, gate_proj);
    ggml_tensor * mlp_act = ggml_mul(compute.ctx, gate_silu, up_proj);
    ggml_tensor * mlp_down = ggml_mul_mat(compute.ctx, down_proj_weight, mlp_act);
    ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);

    ggml_set_name(post_attn_norm, "mlp_probe.post_attn_norm");
    ggml_set_name(gate_proj, "mlp_probe.gate_proj");
    ggml_set_name(up_proj, "mlp_probe.up_proj");
    ggml_set_name(gate_silu, "mlp_probe.gate_silu");
    ggml_set_name(mlp_act, "mlp_probe.mlp_act");
    ggml_set_name(mlp_down, "mlp_probe.mlp_down");

    for (ggml_tensor * t : {post_attn_norm, gate_proj, up_proj, gate_silu, mlp_act, mlp_down}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    if (use_cpu) {
        compute.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        if (!compute.allocr) fail("failed to init CPU gallocr");
        if (!ggml_gallocr_alloc_graph(compute.allocr, gf)) fail("failed to allocate MLP probe graph (cpu)");
    } else {
        ggml_backend_sched_reset(rt.sched);
        if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to allocate MLP probe graph");
    }

    auto input_data = read_f32(probe_dir + "/attn_residual_ref_f32.bin");
    ggml_tensor * inp_alloc = ggml_graph_get_tensor(gf, "runtime.mlp_probe.input");
    if (!inp_alloc) fail("failed to locate runtime.mlp_probe.input");
    ggml_backend_tensor_set(inp_alloc, input_data.data(), 0, input_data.size() * sizeof(float));

    if (use_cpu) {
        if (ggml_backend_graph_compute(rt.backend, gf) != GGML_STATUS_SUCCESS) fail("failed to compute MLP probe (cpu)");
    } else {
        if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute MLP probe (gpu)");
    }

    const std::pair<const char *, const char *> refs[] = {
        {"mlp_probe.post_attn_norm", "post_attn_norm_ref_f32.bin"},
        {"mlp_probe.gate_proj", "gate_proj_ref_f32.bin"},
        {"mlp_probe.up_proj", "up_proj_ref_f32.bin"},
        {"mlp_probe.gate_silu", "gate_silu_ref_f32.bin"},
        {"mlp_probe.mlp_act", "mlp_act_ref_f32.bin"},
        {"mlp_probe.mlp_down", "mlp_down_ref_f32.bin"},
    };

    std::cout << "probe.layer=" << il << "\n";
    std::cout << "probe.backend=" << backend_name << "\n";
    for (const auto & item : refs) {
        auto * t = ggml_graph_get_tensor(gf, item.first);
        if (!t) fail(std::string("missing graph tensor: ") + item.first);
        const auto diff = diff_stats(tensor_to_f32(t), read_f32(probe_dir + "/" + item.second), item.first);
        std::cout << item.first << ".max_abs_diff=" << diff.max_abs_diff << "\n";
        std::cout << item.first << ".mean_abs_diff=" << diff.mean_abs_diff << "\n";
    }

    std::cout << "mlp_probe_ok=true\n";
    if (!use_cpu) {
        ggml_backend_sched_reset(rt.sched);
    }
    return 0;
}
