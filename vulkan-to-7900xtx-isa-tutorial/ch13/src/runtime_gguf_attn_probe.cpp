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
        fail("usage: ch13_runtime_gguf_attn_probe <runtime.gguf> <layer_probe_dir> <layer_idx> [gpu|cpu]");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];
    const std::string backend_name = argc >= 5 ? argv[4] : "gpu";
    const bool use_cpu = backend_name == "cpu";

    Ch13RuntimeGGUF rt;
    rt.load(argv[1], use_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);

    ComputeCtx compute;
    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    compute.ctx = ggml_init(params);
    if (!compute.ctx) fail("failed to init compute ctx");

    const uint32_t hidden = rt.get_u32("runtime.hidden_size");
    const uint32_t head_dim = rt.get_u32("runtime.head_dim");
    const uint32_t n_head = rt.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt.get_u32("runtime.n_kv_head");
    const uint32_t seq_len = rt.get_u32("runtime.cond_seq_len");
    const float eps = rt.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 256, false);
    if (!gf) fail("failed to init compute graph");

    ggml_tensor * inp = ggml_new_tensor_2d(compute.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp, "runtime.attn_probe.input");
    ggml_set_input(inp);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(compute.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "runtime.attn_probe.inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * attn_norm_weight = rt.get_tensor(layer_key(il, "attn_norm.weight").c_str());
    ggml_tensor * q_proj_weight = rt.get_tensor(layer_key(il, "q_proj.weight").c_str());
    ggml_tensor * k_proj_weight = rt.get_tensor(layer_key(il, "k_proj.weight").c_str());
    ggml_tensor * v_proj_weight = rt.get_tensor(layer_key(il, "v_proj.weight").c_str());
    ggml_tensor * q_norm_weight = rt.get_tensor(layer_key(il, "q_norm.weight").c_str());
    ggml_tensor * k_norm_weight = rt.get_tensor(layer_key(il, "k_norm.weight").c_str());
    ggml_tensor * o_proj_weight = rt.get_tensor(layer_key(il, "o_proj.weight").c_str());

    ggml_tensor * attn_input = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, inp, eps), ggml_repeat(compute.ctx, attn_norm_weight, inp));
    ggml_tensor * q_proj = ggml_mul_mat(compute.ctx, q_proj_weight, attn_input);
    ggml_tensor * k_proj = ggml_mul_mat(compute.ctx, k_proj_weight, attn_input);
    ggml_tensor * v_proj = ggml_mul_mat(compute.ctx, v_proj_weight, attn_input);
    ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);

    ggml_tensor * q_3d = ggml_reshape_3d(compute.ctx, q_proj, head_dim, n_head, seq_len);
    ggml_tensor * k_3d = ggml_reshape_3d(compute.ctx, k_proj, head_dim, n_kv_head, seq_len);
    ggml_tensor * v_3d = ggml_reshape_3d(compute.ctx, v_proj, head_dim, n_kv_head, seq_len);
    ggml_tensor * q_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, q_3d, eps), ggml_repeat(compute.ctx, q_norm_weight, q_3d));
    ggml_tensor * k_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, k_3d, eps), ggml_repeat(compute.ctx, k_norm_weight, k_3d));
    ggml_tensor * q_rope = ggml_rope_ext(compute.ctx, q_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope = ggml_rope_ext(compute.ctx, k_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * q_rope_mat = ggml_dup(compute.ctx, q_rope);
    ggml_tensor * k_rope_mat = ggml_dup(compute.ctx, k_rope);
    ggml_tensor * q_attn = ggml_permute(compute.ctx, q_rope_mat, 0, 2, 1, 3);
    ggml_tensor * k_attn = ggml_permute(compute.ctx, k_rope_mat, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(compute.ctx, v_3d, 0, 2, 1, 3);

    ggml_tensor * attn = ggml_flash_attn_ext(compute.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor * attn_out = ggml_cont(compute.ctx, ggml_reshape_2d(compute.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
    ggml_tensor * o_proj = ggml_mul_mat(compute.ctx, o_proj_weight, attn_out);
    ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
    ggml_tensor * attn_residual = ggml_add(compute.ctx, inp, o_proj);

    ggml_set_name(attn_input, "attn_probe.attn_input");
    ggml_set_name(q_proj, "attn_probe.q_proj");
    ggml_set_name(k_proj, "attn_probe.k_proj");
    ggml_set_name(v_proj, "attn_probe.v_proj");
    ggml_set_name(q_norm, "attn_probe.q_norm");
    ggml_set_name(k_norm, "attn_probe.k_norm");
    ggml_set_name(q_rope, "attn_probe.q_rope");
    ggml_set_name(k_rope, "attn_probe.k_rope");
    ggml_set_name(attn_out, "attn_probe.attn_out");
    ggml_set_name(o_proj, "attn_probe.o_proj");
    ggml_set_name(attn_residual, "attn_probe.attn_residual");

    for (ggml_tensor * t : {attn_input, q_proj, k_proj, v_proj, q_norm, k_norm, q_rope, k_rope, attn_out, o_proj, attn_residual}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    if (use_cpu) {
        compute.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        if (!compute.allocr) fail("failed to init CPU gallocr");
        if (!ggml_gallocr_alloc_graph(compute.allocr, gf)) fail("failed to allocate attention probe graph (cpu)");
    } else {
        ggml_backend_sched_reset(rt.sched);
        if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to allocate attention probe graph");
    }

    auto input_data = read_f32(probe_dir + "/layer_input_ref_f32.bin");
    ggml_tensor * inp_alloc = ggml_graph_get_tensor(gf, "runtime.attn_probe.input");
    if (!inp_alloc) fail("failed to locate runtime.attn_probe.input");
    ggml_backend_tensor_set(inp_alloc, input_data.data(), 0, input_data.size() * sizeof(float));

    std::vector<int32_t> pos(seq_len);
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t) i;
    ggml_tensor * pos_alloc = ggml_graph_get_tensor(gf, "runtime.attn_probe.inp_pos");
    if (!pos_alloc) fail("failed to locate runtime.attn_probe.inp_pos");
    ggml_backend_tensor_set(pos_alloc, pos.data(), 0, pos.size() * sizeof(int32_t));

    if (use_cpu) {
        if (ggml_backend_graph_compute(rt.backend, gf) != GGML_STATUS_SUCCESS) fail("failed to compute attention probe (cpu)");
    } else {
        if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute attention probe (gpu)");
    }

    const std::pair<const char *, const char *> refs[] = {
        {"attn_probe.attn_input", "attn_input_ref_f32.bin"},
        {"attn_probe.q_proj", "q_proj_ref_f32.bin"},
        {"attn_probe.k_proj", "k_proj_ref_f32.bin"},
        {"attn_probe.v_proj", "v_proj_ref_f32.bin"},
        {"attn_probe.q_norm", "q_norm_ref_f32.bin"},
        {"attn_probe.k_norm", "k_norm_ref_f32.bin"},
        {"attn_probe.q_rope", "q_rope_ref_f32.bin"},
        {"attn_probe.k_rope", "k_rope_ref_f32.bin"},
        {"attn_probe.attn_out", "attn_out_ref_f32.bin"},
        {"attn_probe.o_proj", "o_proj_ref_f32.bin"},
        {"attn_probe.attn_residual", "attn_residual_ref_f32.bin"},
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

    std::cout << "attn_probe_ok=true\n";
    if (!use_cpu) {
        ggml_backend_sched_reset(rt.sched);
    }
    return 0;
}
