#include "ch13_gguf_runtime.h"
#include "ch13_qk_fused_path.h"

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
    if ((nbytes % sizeof(float)) != 0) fail("invalid f32 ref size: " + path);
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
    if (argc != 4) {
        fail("usage: ch13_runtime_gguf_attn_fused_qk_probe <runtime.gguf> <layer_probe_dir> <layer_idx>");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];

    Ch13RuntimeGGUF rt;
    rt.load(argv[1], GGML_BACKEND_DEVICE_TYPE_GPU);

    const uint32_t hidden = rt.get_u32("runtime.hidden_size");
    const uint32_t head_dim = rt.get_u32("runtime.head_dim");
    const uint32_t n_head = rt.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt.get_u32("runtime.n_kv_head");
    const uint32_t seq_len = rt.get_u32("runtime.cond_seq_len");
    const float eps = rt.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    const auto layer_input = read_f32(probe_dir + "/layer_input_ref_f32.bin");

    // Stage A: compute attn_input + v_proj with ggml/Vulkan.
    ComputeCtx stage_a;
    ggml_init_params params_a = { 16 * 1024 * 1024, nullptr, true };
    stage_a.ctx = ggml_init(params_a);
    if (!stage_a.ctx) fail("failed to init stage_a ctx");
    ggml_cgraph * gf_a = ggml_new_graph_custom(stage_a.ctx, 128, false);
    if (!gf_a) fail("failed to init stage_a graph");

    ggml_tensor * inp_a = ggml_new_tensor_2d(stage_a.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp_a, "attn_fused.input");
    ggml_set_input(inp_a);

    ggml_tensor * attn_input = ggml_mul(stage_a.ctx,
        ggml_rms_norm(stage_a.ctx, inp_a, eps),
        ggml_repeat(stage_a.ctx, rt.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp_a));
    ggml_tensor * v_proj = ggml_mul_mat(stage_a.ctx, rt.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input);
    ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);
    ggml_set_name(attn_input, "attn_fused.attn_input");
    ggml_set_name(v_proj, "attn_fused.v_proj");
    ggml_set_output(attn_input);
    ggml_set_output(v_proj);
    ggml_build_forward_expand(gf_a, attn_input);
    ggml_build_forward_expand(gf_a, v_proj);

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf_a)) fail("failed to alloc stage_a graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_a, "attn_fused.input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(rt.sched, gf_a) != GGML_STATUS_SUCCESS) fail("failed to compute stage_a graph");
    const auto attn_input_vec = tensor_to_f32(ggml_graph_get_tensor(gf_a, "attn_fused.attn_input"));
    const auto v_proj_vec = tensor_to_f32(ggml_graph_get_tensor(gf_a, "attn_fused.v_proj"));
    ggml_backend_sched_reset(rt.sched);

    Ch13QKFusedConfig cfg;
    cfg.layer = il;
    cfg.hidden = (int) hidden;
    cfg.head_dim = (int) head_dim;
    cfg.n_head = (int) n_head;
    cfg.n_kv_head = (int) n_kv_head;
    cfg.seq_len = (int) seq_len;
    cfg.rms_norm_eps = eps;
    cfg.rope_freq_base = rope_base;
    cfg.rope_n_ctx_orig = rope_ctx;

    const auto qk = ch13_run_qk_fused_path(rt, cfg, attn_input_vec, "gpu");

    // Stage B: attention + o_proj + residual using fused q/k.
    ComputeCtx stage_b;
    ggml_init_params params_b = { 32 * 1024 * 1024, nullptr, true };
    stage_b.ctx = ggml_init(params_b);
    if (!stage_b.ctx) fail("failed to init stage_b ctx");
    ggml_cgraph * gf_b = ggml_new_graph_custom(stage_b.ctx, 128, false);
    if (!gf_b) fail("failed to init stage_b graph");

    ggml_tensor * inp_b = ggml_new_tensor_2d(stage_b.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp_b, "attn_fused_b.input");
    ggml_set_input(inp_b);
    ggml_tensor * q_in = ggml_new_tensor_3d(stage_b.ctx, GGML_TYPE_F32, head_dim, n_head, seq_len);
    ggml_set_name(q_in, "attn_fused_b.q_in");
    ggml_set_input(q_in);
    ggml_tensor * k_in = ggml_new_tensor_3d(stage_b.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(k_in, "attn_fused_b.k_in");
    ggml_set_input(k_in);
    ggml_tensor * v_in = ggml_new_tensor_3d(stage_b.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(v_in, "attn_fused_b.v_in");
    ggml_set_input(v_in);

    ggml_tensor * q_mat = ggml_dup(stage_b.ctx, q_in);
    ggml_tensor * k_mat = ggml_dup(stage_b.ctx, k_in);
    ggml_tensor * q_attn = ggml_permute(stage_b.ctx, q_mat, 0, 2, 1, 3);
    ggml_tensor * k_attn = ggml_permute(stage_b.ctx, k_mat, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(stage_b.ctx, v_in, 0, 2, 1, 3);

    ggml_tensor * attn = ggml_flash_attn_ext(stage_b.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor * attn_out = ggml_cont(stage_b.ctx, ggml_reshape_2d(stage_b.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
    ggml_tensor * o_proj = ggml_mul_mat(stage_b.ctx, rt.get_tensor(layer_key(il, "o_proj.weight").c_str()), attn_out);
    ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
    ggml_tensor * attn_residual = ggml_add(stage_b.ctx, inp_b, o_proj);

    ggml_set_name(attn_out, "attn_fused_b.attn_out");
    ggml_set_name(o_proj, "attn_fused_b.o_proj");
    ggml_set_name(attn_residual, "attn_fused_b.attn_residual");
    ggml_set_output(attn_out);
    ggml_set_output(o_proj);
    ggml_set_output(attn_residual);
    ggml_build_forward_expand(gf_b, attn_out);
    ggml_build_forward_expand(gf_b, o_proj);
    ggml_build_forward_expand(gf_b, attn_residual);

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf_b)) fail("failed to alloc stage_b graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "attn_fused_b.input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "attn_fused_b.q_in"), qk.q_rope.data(), 0, qk.q_rope.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "attn_fused_b.k_in"), qk.k_rope.data(), 0, qk.k_rope.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "attn_fused_b.v_in"), v_proj_vec.data(), 0, v_proj_vec.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(rt.sched, gf_b) != GGML_STATUS_SUCCESS) fail("failed to compute stage_b graph");

    const auto q_diff = diff_stats(qk.q_rope, read_f32(probe_dir + "/q_rope_ref_f32.bin"), "q_rope");
    const auto k_diff = diff_stats(qk.k_rope, read_f32(probe_dir + "/k_rope_ref_f32.bin"), "k_rope");
    const auto attn_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf_b, "attn_fused_b.attn_out")), read_f32(probe_dir + "/attn_out_ref_f32.bin"), "attn_out");
    const auto o_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf_b, "attn_fused_b.o_proj")), read_f32(probe_dir + "/o_proj_ref_f32.bin"), "o_proj");
    const auto residual_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf_b, "attn_fused_b.attn_residual")), read_f32(probe_dir + "/attn_residual_ref_f32.bin"), "attn_residual");

    std::cout << "probe.layer=" << il << "\n";
    std::cout << "probe.backend=gpu-fused-qk\n";
    std::cout << "attn_fused.q_rope.max_abs_diff=" << q_diff.max_abs_diff << "\n";
    std::cout << "attn_fused.q_rope.mean_abs_diff=" << q_diff.mean_abs_diff << "\n";
    std::cout << "attn_fused.k_rope.max_abs_diff=" << k_diff.max_abs_diff << "\n";
    std::cout << "attn_fused.k_rope.mean_abs_diff=" << k_diff.mean_abs_diff << "\n";
    std::cout << "attn_fused.attn_out.max_abs_diff=" << attn_diff.max_abs_diff << "\n";
    std::cout << "attn_fused.attn_out.mean_abs_diff=" << attn_diff.mean_abs_diff << "\n";
    std::cout << "attn_fused.o_proj.max_abs_diff=" << o_diff.max_abs_diff << "\n";
    std::cout << "attn_fused.o_proj.mean_abs_diff=" << o_diff.mean_abs_diff << "\n";
    std::cout << "attn_fused.attn_residual.max_abs_diff=" << residual_diff.max_abs_diff << "\n";
    std::cout << "attn_fused.attn_residual.mean_abs_diff=" << residual_diff.mean_abs_diff << "\n";
    std::cout << "attn_fused_probe_ok=true\n";

    ggml_backend_sched_reset(rt.sched);
    return 0;
}
