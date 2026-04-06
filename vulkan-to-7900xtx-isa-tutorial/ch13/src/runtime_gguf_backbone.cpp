#include "ch13_gguf_runtime.h"
#include "ch13_qk_fused_path.h"

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
    ~ComputeCtx() {
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

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4 && argc != 5) {
        fail("usage: ch13_runtime_gguf_backbone <runtime.gguf> <backbone_refs_dir> [gpu|cpu] [cpu_from_layer]");
    }

    const std::string backend_name = argc >= 4 ? argv[3] : "gpu";
    const bool use_cpu = backend_name == "cpu";
    const int cpu_from_layer = argc >= 5 ? std::atoi(argv[4]) : -1;

    Ch13RuntimeGGUF rt;
    rt.load(argv[1], use_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);

    ComputeCtx compute;
    ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
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
    const uint32_t n_layer = rt.get_u32("runtime.n_layer");
    const float eps = rt.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    const std::string ref_dir = argv[2];

    if (backend_name == "gpu_fused_qk") {
        const auto x_input = tensor_to_f32(rt.get_tensor("runtime.iterative.cond.x_input_ref"));
        const int fused_from_layer = 0;
        const auto result = ch13_run_backbone_with_fused_qk(rt, x_input, fused_from_layer);
        const auto final_diff = diff_stats(result.final_hidden, read_f32(ref_dir + "/final_hidden_ref_f32.bin"), "final_hidden");
        const auto logits_diff = diff_stats(result.logits, read_f32(ref_dir + "/logits_ref_f32.bin"), "logits");

        std::cout << "general.architecture=" << rt.get_str("general.architecture") << "\n";
        std::cout << "runtime.backend=" << backend_name << "\n";
        std::cout << "runtime.fused_from_layer=" << fused_from_layer << "\n";
        std::cout << "runtime.cpu_from_layer=" << cpu_from_layer << "\n";
        std::cout << "runtime.n_layer=" << n_layer << "\n";
        std::cout << "backbone.final_hidden.max_abs_diff=" << final_diff.max_abs_diff << "\n";
        std::cout << "backbone.final_hidden.mean_abs_diff=" << final_diff.mean_abs_diff << "\n";
        std::cout << "backbone.logits.max_abs_diff=" << logits_diff.max_abs_diff << "\n";
        std::cout << "backbone.logits.mean_abs_diff=" << logits_diff.mean_abs_diff << "\n";
        std::cout << "backbone_graph_ok=true\n";
        return 0;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 4096, false);
    if (!gf) fail("failed to init compute graph");

    ggml_tensor * inp_pos = ggml_new_tensor_1d(compute.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "runtime.backbone.inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * cur = rt.get_tensor("runtime.iterative.cond.x_input_ref");
    std::vector<ggml_tensor *> layer_outs;
    layer_outs.reserve(n_layer);

    for (uint32_t il = 0; il < n_layer; ++il) {
        char name[128];

        ggml_tensor * attn_norm_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".attn_norm.weight").c_str());
        ggml_tensor * q_proj_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".q_proj.weight").c_str());
        ggml_tensor * k_proj_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".k_proj.weight").c_str());
        ggml_tensor * v_proj_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".v_proj.weight").c_str());
        ggml_tensor * q_norm_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".q_norm.weight").c_str());
        ggml_tensor * k_norm_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".k_norm.weight").c_str());
        ggml_tensor * o_proj_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".o_proj.weight").c_str());
        ggml_tensor * post_attn_norm_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".post_attention_norm.weight").c_str());
        ggml_tensor * gate_proj_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".gate_proj.weight").c_str());
        ggml_tensor * up_proj_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".up_proj.weight").c_str());
        ggml_tensor * down_proj_weight = rt.get_tensor((std::string("runtime.layers.") + (il < 10 ? "0" : "") + std::to_string(il) + ".down_proj.weight").c_str());

        std::vector<ggml_tensor *> layer_nodes;

        ggml_tensor * residual = cur;
        ggml_tensor * attn_in = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, cur, eps), ggml_repeat(compute.ctx, attn_norm_weight, cur));
        ggml_tensor * q_proj = ggml_mul_mat(compute.ctx, q_proj_weight, attn_in);
        ggml_tensor * k_proj = ggml_mul_mat(compute.ctx, k_proj_weight, attn_in);
        ggml_tensor * v_proj = ggml_mul_mat(compute.ctx, v_proj_weight, attn_in);
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
        q_rope = ggml_cont(compute.ctx, q_rope);
        k_rope = ggml_cont(compute.ctx, k_rope);
        ggml_tensor * q_attn = ggml_permute(compute.ctx, q_rope, 0, 2, 1, 3);
        ggml_tensor * k_attn = ggml_permute(compute.ctx, k_rope, 0, 2, 1, 3);
        ggml_tensor * v_attn = ggml_permute(compute.ctx, v_3d, 0, 2, 1, 3);
        ggml_tensor * attn = ggml_flash_attn_ext(compute.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        ggml_tensor * attn_2d = ggml_cont(compute.ctx, ggml_reshape_2d(compute.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
        ggml_tensor * o_proj = ggml_mul_mat(compute.ctx, o_proj_weight, attn_2d);
        ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
        ggml_tensor * attn_residual = ggml_add(compute.ctx, residual, o_proj);
        ggml_tensor * mlp_in = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, attn_residual, eps), ggml_repeat(compute.ctx, post_attn_norm_weight, attn_residual));
        ggml_tensor * gate = ggml_mul_mat(compute.ctx, gate_proj_weight, mlp_in);
        ggml_tensor * up = ggml_mul_mat(compute.ctx, up_proj_weight, mlp_in);
        ggml_mul_mat_set_prec(gate, GGML_PREC_F32);
        ggml_mul_mat_set_prec(up, GGML_PREC_F32);
        ggml_tensor * mlp_act = ggml_mul(compute.ctx, ggml_silu(compute.ctx, gate), up);
        ggml_tensor * mlp_down = ggml_mul_mat(compute.ctx, down_proj_weight, mlp_act);
        ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);
        cur = ggml_add(compute.ctx, attn_residual, mlp_down);

        layer_nodes = {
            attn_in, q_proj, k_proj, v_proj, q_3d, k_3d, v_3d, q_norm, k_norm, q_rope, k_rope,
            q_attn, k_attn, v_attn, attn, attn_2d, o_proj, attn_residual, mlp_in, gate, up, mlp_act, mlp_down, cur
        };
        if (!use_cpu && cpu_from_layer >= 0 && static_cast<int>(il) >= cpu_from_layer && rt.cpu_backend) {
            for (ggml_tensor * t : layer_nodes) {
                ggml_backend_sched_set_tensor_backend(rt.sched, t, rt.cpu_backend);
            }
        }

        snprintf(name, sizeof(name), "runtime.backbone.layer_%02u_out", il);
        ggml_set_name(cur, name);
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        layer_outs.push_back(cur);
    }

    ggml_tensor * final_hidden = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, cur, eps), ggml_repeat(compute.ctx, rt.get_tensor("runtime.output_norm.weight"), cur));
    ggml_set_name(final_hidden, "runtime.backbone.final_hidden");
    ggml_set_output(final_hidden);
    ggml_build_forward_expand(gf, final_hidden);

    ggml_tensor * logits = ggml_mul_mat(compute.ctx, rt.get_tensor("runtime.audio_heads.weight"), final_hidden);
    ggml_set_name(logits, "runtime.backbone.logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    if (!use_cpu && cpu_from_layer >= 0 && rt.cpu_backend) {
        ggml_backend_sched_set_tensor_backend(rt.sched, final_hidden, rt.cpu_backend);
        ggml_backend_sched_set_tensor_backend(rt.sched, logits, rt.cpu_backend);
    }

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to allocate backbone graph");
    std::vector<int32_t> pos(seq_len);
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t) i;
    ggml_tensor * inp_pos_alloc = ggml_graph_get_tensor(gf, "runtime.backbone.inp_pos");
    if (!inp_pos_alloc) fail("failed to locate runtime.backbone.inp_pos");
    ggml_backend_tensor_set(inp_pos_alloc, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute backbone graph");

    float backbone_max = 0.0f;
    double backbone_mean = 0.0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const auto diff = diff_stats(tensor_to_f32(layer_outs[il]), read_f32(ref_dir + "/layer_" + (il < 10 ? "0" : "") + std::to_string(il) + "_out_ref_f32.bin"), "layer");
        backbone_max = std::max(backbone_max, diff.max_abs_diff);
        backbone_mean += diff.mean_abs_diff;
        std::cout << "backbone.layer_" << (il < 10 ? "0" : "") << il << ".max_abs_diff=" << diff.max_abs_diff << "\n";
        std::cout << "backbone.layer_" << (il < 10 ? "0" : "") << il << ".mean_abs_diff=" << diff.mean_abs_diff << "\n";
    }
    backbone_mean /= n_layer;
    const auto final_diff = diff_stats(tensor_to_f32(final_hidden), read_f32(ref_dir + "/final_hidden_ref_f32.bin"), "final_hidden");
    const auto logits_diff = diff_stats(tensor_to_f32(logits), read_f32(ref_dir + "/logits_ref_f32.bin"), "logits");

    std::cout << "general.architecture=" << rt.get_str("general.architecture") << "\n";
    std::cout << "runtime.backend=" << backend_name << "\n";
    std::cout << "runtime.cpu_from_layer=" << cpu_from_layer << "\n";
    std::cout << "runtime.n_layer=" << n_layer << "\n";
    std::cout << "backbone.max_layer_abs_diff=" << backbone_max << "\n";
    std::cout << "backbone.mean_layer_mean_abs_diff=" << backbone_mean << "\n";
    std::cout << "backbone.final_hidden.max_abs_diff=" << final_diff.max_abs_diff << "\n";
    std::cout << "backbone.final_hidden.mean_abs_diff=" << final_diff.mean_abs_diff << "\n";
    std::cout << "backbone.logits.max_abs_diff=" << logits_diff.max_abs_diff << "\n";
    std::cout << "backbone.logits.mean_abs_diff=" << logits_diff.mean_abs_diff << "\n";
    std::cout << "backbone_graph_ok=true\n";

    ggml_backend_sched_reset(rt.sched);
    return 0;
}
