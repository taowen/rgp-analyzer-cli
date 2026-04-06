#include "ch13_gguf_runtime.h"

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

struct ComputeCtx { ggml_context * ctx = nullptr; ~ComputeCtx() { if (ctx) ggml_free(ctx); } };

std::vector<float> read_f32(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail(std::string("failed to open reference file: ") + path);
    }
    in.seekg(0, std::ios::end);
    const size_t nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (nbytes % sizeof(float) != 0) {
        fail(std::string("invalid f32 ref size: ") + path);
    }
    std::vector<float> out(nbytes / sizeof(float));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(nbytes));
    if (!in) {
        fail(std::string("failed reading reference file: ") + path);
    }
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

DiffStats diff_stats(const std::vector<float> & got, const std::vector<float> & ref, const char * name) {
    if (got.size() != ref.size()) {
        fail(std::string("size mismatch for ") + name);
    }
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
    if (argc != 3) {
        fail("usage: ch13_runtime_gguf_layer0 <runtime.gguf> <layer0_refs_dir>");
    }

    Ch13RuntimeGGUF rt;
    rt.load(argv[1]);

    ggml_tensor * x_input_ref = rt.get_tensor("runtime.iterative.cond.x_input_ref");
    ggml_tensor * attn_norm_weight = rt.get_tensor("runtime.layers.00.attn_norm.weight");
    ggml_tensor * q_proj_weight = rt.get_tensor("runtime.layers.00.q_proj.weight");
    ggml_tensor * k_proj_weight = rt.get_tensor("runtime.layers.00.k_proj.weight");
    ggml_tensor * v_proj_weight = rt.get_tensor("runtime.layers.00.v_proj.weight");
    ggml_tensor * q_norm_weight = rt.get_tensor("runtime.layers.00.q_norm.weight");
    ggml_tensor * k_norm_weight = rt.get_tensor("runtime.layers.00.k_norm.weight");
    ggml_tensor * o_proj_weight = rt.get_tensor("runtime.layers.00.o_proj.weight");
    ggml_tensor * post_attention_norm_weight = rt.get_tensor("runtime.layers.00.post_attention_norm.weight");
    ggml_tensor * gate_proj_weight = rt.get_tensor("runtime.layers.00.gate_proj.weight");
    ggml_tensor * up_proj_weight = rt.get_tensor("runtime.layers.00.up_proj.weight");
    ggml_tensor * down_proj_weight = rt.get_tensor("runtime.layers.00.down_proj.weight");

    ComputeCtx compute;
    ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    compute.ctx = ggml_init(params);
    if (!compute.ctx) {
        fail("failed to init compute ctx");
    }

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 128, false);
    if (!gf) {
        fail("failed to init compute graph");
    }

    ggml_tensor * rms = ggml_rms_norm(compute.ctx, x_input_ref, rt.get_f32("runtime.rms_norm_eps"));
    ggml_tensor * attn_scale = ggml_repeat(compute.ctx, attn_norm_weight, rms);
    ggml_tensor * attn_input = ggml_mul(compute.ctx, rms, attn_scale);
    ggml_tensor * q_proj = ggml_mul_mat(compute.ctx, q_proj_weight, attn_input);
    ggml_tensor * k_proj = ggml_mul_mat(compute.ctx, k_proj_weight, attn_input);
    ggml_tensor * v_proj = ggml_mul_mat(compute.ctx, v_proj_weight, attn_input);
    ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);
    ggml_tensor * inp_pos = ggml_new_tensor_1d(compute.ctx, GGML_TYPE_I32, rt.get_u32("runtime.cond_seq_len"));
    ggml_set_name(inp_pos, "runtime.layer0.inp_pos");
    ggml_set_input(inp_pos);
    ggml_tensor * q_proj_3d = ggml_reshape_3d(compute.ctx, q_proj, rt.get_u32("runtime.head_dim"), rt.get_u32("runtime.n_head"), rt.get_u32("runtime.cond_seq_len"));
    ggml_tensor * k_proj_3d = ggml_reshape_3d(compute.ctx, k_proj, rt.get_u32("runtime.head_dim"), rt.get_u32("runtime.n_kv_head"), rt.get_u32("runtime.cond_seq_len"));
    ggml_tensor * q_rms = ggml_rms_norm(compute.ctx, q_proj_3d, rt.get_f32("runtime.rms_norm_eps"));
    ggml_tensor * k_rms = ggml_rms_norm(compute.ctx, k_proj_3d, rt.get_f32("runtime.rms_norm_eps"));
    ggml_tensor * q_norm = ggml_mul(compute.ctx, q_rms, ggml_repeat(compute.ctx, q_norm_weight, q_rms));
    ggml_tensor * k_norm = ggml_mul(compute.ctx, k_rms, ggml_repeat(compute.ctx, k_norm_weight, k_rms));
    ggml_tensor * q_rope = ggml_rope_ext(
        compute.ctx, q_norm, inp_pos, nullptr,
        (int) rt.get_u32("runtime.head_dim"), GGML_ROPE_TYPE_NEOX, (int) rt.get_u32("runtime.rope_n_ctx_orig"),
        rt.get_f32("runtime.rope_freq_base"), 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope = ggml_rope_ext(
        compute.ctx, k_norm, inp_pos, nullptr,
        (int) rt.get_u32("runtime.head_dim"), GGML_ROPE_TYPE_NEOX, (int) rt.get_u32("runtime.rope_n_ctx_orig"),
        rt.get_f32("runtime.rope_freq_base"), 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * v_proj_3d = ggml_reshape_3d(compute.ctx, v_proj, rt.get_u32("runtime.head_dim"), rt.get_u32("runtime.n_kv_head"), rt.get_u32("runtime.cond_seq_len"));
    ggml_tensor * q_attn = ggml_permute(compute.ctx, q_rope, 0, 2, 1, 3);
    ggml_tensor * k_attn = ggml_permute(compute.ctx, k_rope, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(compute.ctx, v_proj_3d, 0, 2, 1, 3);
    ggml_tensor * attn_out = ggml_flash_attn_ext(compute.ctx, q_attn, k_attn, v_attn, nullptr, 1.0f / std::sqrt((float) rt.get_u32("runtime.head_dim")), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
    ggml_tensor * attn_out_2d = ggml_cont(compute.ctx, ggml_reshape_2d(compute.ctx, attn_out, attn_out->ne[0] * attn_out->ne[1], attn_out->ne[2] * attn_out->ne[3]));
    ggml_tensor * o_proj = ggml_mul_mat(compute.ctx, o_proj_weight, attn_out_2d);
    ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
    ggml_tensor * attn_residual = ggml_add(compute.ctx, x_input_ref, o_proj);
    ggml_tensor * post_attn_norm = ggml_mul(
        compute.ctx,
        ggml_rms_norm(compute.ctx, attn_residual, rt.get_f32("runtime.rms_norm_eps")),
        ggml_repeat(compute.ctx, post_attention_norm_weight, attn_residual));
    ggml_tensor * gate_proj = ggml_mul_mat(compute.ctx, gate_proj_weight, post_attn_norm);
    ggml_tensor * up_proj = ggml_mul_mat(compute.ctx, up_proj_weight, post_attn_norm);
    ggml_mul_mat_set_prec(gate_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(up_proj, GGML_PREC_F32);
    ggml_tensor * mlp_down = ggml_mul_mat(compute.ctx, down_proj_weight, ggml_mul(compute.ctx, ggml_silu(compute.ctx, gate_proj), up_proj));
    ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);
    ggml_tensor * layer0_out = ggml_add(compute.ctx, attn_residual, mlp_down);
    ggml_set_name(attn_input, "runtime.layer0.attn_input");
    ggml_set_name(q_proj, "runtime.layer0.q_proj");
    ggml_set_name(k_proj, "runtime.layer0.k_proj");
    ggml_set_name(v_proj, "runtime.layer0.v_proj");
    ggml_set_name(q_norm, "runtime.layer0.q_norm");
    ggml_set_name(k_norm, "runtime.layer0.k_norm");
    ggml_set_name(q_rope, "runtime.layer0.q_rope");
    ggml_set_name(k_rope, "runtime.layer0.k_rope");
    ggml_set_name(attn_out_2d, "runtime.layer0.attn_out");
    ggml_set_name(o_proj, "runtime.layer0.o_proj");
    ggml_set_name(attn_residual, "runtime.layer0.attn_residual");
    ggml_set_name(post_attn_norm, "runtime.layer0.post_attn_norm");
    ggml_set_name(gate_proj, "runtime.layer0.gate_proj");
    ggml_set_name(up_proj, "runtime.layer0.up_proj");
    ggml_set_name(mlp_down, "runtime.layer0.mlp_down");
    ggml_set_name(layer0_out, "runtime.layer0.out");
    ggml_set_output(attn_input);
    ggml_set_output(q_proj);
    ggml_set_output(k_proj);
    ggml_set_output(v_proj);
    ggml_set_output(q_norm);
    ggml_set_output(k_norm);
    ggml_set_output(q_rope);
    ggml_set_output(k_rope);
    ggml_set_output(attn_out_2d);
    ggml_set_output(o_proj);
    ggml_set_output(attn_residual);
    ggml_set_output(post_attn_norm);
    ggml_set_output(gate_proj);
    ggml_set_output(up_proj);
    ggml_set_output(mlp_down);
    ggml_set_output(layer0_out);
    ggml_build_forward_expand(gf, attn_input);
    ggml_build_forward_expand(gf, q_proj);
    ggml_build_forward_expand(gf, k_proj);
    ggml_build_forward_expand(gf, v_proj);
    ggml_build_forward_expand(gf, q_norm);
    ggml_build_forward_expand(gf, k_norm);
    ggml_build_forward_expand(gf, q_rope);
    ggml_build_forward_expand(gf, k_rope);
    ggml_build_forward_expand(gf, attn_out_2d);
    ggml_build_forward_expand(gf, o_proj);
    ggml_build_forward_expand(gf, attn_residual);
    ggml_build_forward_expand(gf, post_attn_norm);
    ggml_build_forward_expand(gf, gate_proj);
    ggml_build_forward_expand(gf, up_proj);
    ggml_build_forward_expand(gf, mlp_down);
    ggml_build_forward_expand(gf, layer0_out);

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) {
        fail("failed to allocate layer0 graph");
    }
    std::vector<int32_t> pos((size_t) rt.get_u32("runtime.cond_seq_len"));
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t) i;
    ggml_tensor * inp_pos_alloc = ggml_graph_get_tensor(gf, "runtime.layer0.inp_pos");
    if (!inp_pos_alloc) {
        fail("failed to locate runtime.layer0.inp_pos in graph");
    }
    ggml_backend_tensor_set(inp_pos_alloc, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) {
        fail("failed to compute layer0 graph on backend");
    }

    const std::string ref_dir = argv[2];
    const DiffStats attn_diff = diff_stats(tensor_to_f32(attn_input), read_f32((ref_dir + "/attn_input_ref_f32.bin").c_str()), "attn_input");
    const DiffStats q_diff = diff_stats(tensor_to_f32(q_proj), read_f32((ref_dir + "/q_proj_ref_f32.bin").c_str()), "q_proj");
    const DiffStats k_diff = diff_stats(tensor_to_f32(k_proj), read_f32((ref_dir + "/k_proj_ref_f32.bin").c_str()), "k_proj");
    const DiffStats v_diff = diff_stats(tensor_to_f32(v_proj), read_f32((ref_dir + "/v_proj_ref_f32.bin").c_str()), "v_proj");
    const DiffStats qn_diff = diff_stats(tensor_to_f32(q_norm), read_f32((ref_dir + "/q_norm_ref_f32.bin").c_str()), "q_norm");
    const DiffStats kn_diff = diff_stats(tensor_to_f32(k_norm), read_f32((ref_dir + "/k_norm_ref_f32.bin").c_str()), "k_norm");
    const DiffStats qr_diff = diff_stats(tensor_to_f32(q_rope), read_f32((ref_dir + "/q_rope_ref_f32.bin").c_str()), "q_rope");
    const DiffStats kr_diff = diff_stats(tensor_to_f32(k_rope), read_f32((ref_dir + "/k_rope_ref_f32.bin").c_str()), "k_rope");
    const DiffStats attn_out_diff = diff_stats(tensor_to_f32(attn_out_2d), read_f32((ref_dir + "/attn_out_ref_f32.bin").c_str()), "attn_out");
    const DiffStats oproj_diff = diff_stats(tensor_to_f32(o_proj), read_f32((ref_dir + "/o_proj_ref_f32.bin").c_str()), "o_proj");
    const DiffStats attn_residual_diff = diff_stats(tensor_to_f32(attn_residual), read_f32((ref_dir + "/attn_residual_ref_f32.bin").c_str()), "attn_residual");
    const DiffStats post_attn_norm_diff = diff_stats(tensor_to_f32(post_attn_norm), read_f32((ref_dir + "/post_attn_norm_ref_f32.bin").c_str()), "post_attn_norm");
    const DiffStats gate_proj_diff = diff_stats(tensor_to_f32(gate_proj), read_f32((ref_dir + "/gate_proj_ref_f32.bin").c_str()), "gate_proj");
    const DiffStats up_proj_diff = diff_stats(tensor_to_f32(up_proj), read_f32((ref_dir + "/up_proj_ref_f32.bin").c_str()), "up_proj");
    const DiffStats mlp_down_diff = diff_stats(tensor_to_f32(mlp_down), read_f32((ref_dir + "/mlp_down_ref_f32.bin").c_str()), "mlp_down");
    const DiffStats layer0_out_diff = diff_stats(tensor_to_f32(layer0_out), read_f32((ref_dir + "/layer0_out_ref_f32.bin").c_str()), "layer0_out");

    std::cout << "general.architecture=" << rt.get_str("general.architecture") << "\n";
    std::cout << "runtime.cond_seq_len=" << rt.get_u32("runtime.cond_seq_len") << "\n";
    std::cout << "layer0.attn_input.max_abs_diff=" << attn_diff.max_abs_diff << "\n";
    std::cout << "layer0.attn_input.mean_abs_diff=" << attn_diff.mean_abs_diff << "\n";
    std::cout << "layer0.q_proj.max_abs_diff=" << q_diff.max_abs_diff << "\n";
    std::cout << "layer0.q_proj.mean_abs_diff=" << q_diff.mean_abs_diff << "\n";
    std::cout << "layer0.k_proj.max_abs_diff=" << k_diff.max_abs_diff << "\n";
    std::cout << "layer0.k_proj.mean_abs_diff=" << k_diff.mean_abs_diff << "\n";
    std::cout << "layer0.v_proj.max_abs_diff=" << v_diff.max_abs_diff << "\n";
    std::cout << "layer0.v_proj.mean_abs_diff=" << v_diff.mean_abs_diff << "\n";
    std::cout << "layer0.q_norm.max_abs_diff=" << qn_diff.max_abs_diff << "\n";
    std::cout << "layer0.q_norm.mean_abs_diff=" << qn_diff.mean_abs_diff << "\n";
    std::cout << "layer0.k_norm.max_abs_diff=" << kn_diff.max_abs_diff << "\n";
    std::cout << "layer0.k_norm.mean_abs_diff=" << kn_diff.mean_abs_diff << "\n";
    std::cout << "layer0.q_rope.max_abs_diff=" << qr_diff.max_abs_diff << "\n";
    std::cout << "layer0.q_rope.mean_abs_diff=" << qr_diff.mean_abs_diff << "\n";
    std::cout << "layer0.k_rope.max_abs_diff=" << kr_diff.max_abs_diff << "\n";
    std::cout << "layer0.k_rope.mean_abs_diff=" << kr_diff.mean_abs_diff << "\n";
    std::cout << "layer0.attn_out.max_abs_diff=" << attn_out_diff.max_abs_diff << "\n";
    std::cout << "layer0.attn_out.mean_abs_diff=" << attn_out_diff.mean_abs_diff << "\n";
    std::cout << "layer0.o_proj.max_abs_diff=" << oproj_diff.max_abs_diff << "\n";
    std::cout << "layer0.o_proj.mean_abs_diff=" << oproj_diff.mean_abs_diff << "\n";
    std::cout << "layer0.attn_residual.max_abs_diff=" << attn_residual_diff.max_abs_diff << "\n";
    std::cout << "layer0.attn_residual.mean_abs_diff=" << attn_residual_diff.mean_abs_diff << "\n";
    std::cout << "layer0.post_attn_norm.max_abs_diff=" << post_attn_norm_diff.max_abs_diff << "\n";
    std::cout << "layer0.post_attn_norm.mean_abs_diff=" << post_attn_norm_diff.mean_abs_diff << "\n";
    std::cout << "layer0.gate_proj.max_abs_diff=" << gate_proj_diff.max_abs_diff << "\n";
    std::cout << "layer0.gate_proj.mean_abs_diff=" << gate_proj_diff.mean_abs_diff << "\n";
    std::cout << "layer0.up_proj.max_abs_diff=" << up_proj_diff.max_abs_diff << "\n";
    std::cout << "layer0.up_proj.mean_abs_diff=" << up_proj_diff.mean_abs_diff << "\n";
    std::cout << "layer0.mlp_down.max_abs_diff=" << mlp_down_diff.max_abs_diff << "\n";
    std::cout << "layer0.mlp_down.mean_abs_diff=" << mlp_down_diff.mean_abs_diff << "\n";
    std::cout << "layer0.out.max_abs_diff=" << layer0_out_diff.max_abs_diff << "\n";
    std::cout << "layer0.out.mean_abs_diff=" << layer0_out_diff.mean_abs_diff << "\n";
    std::cout << "layer0_graph_ok=true\n";
    ggml_backend_sched_reset(rt.sched);
    return 0;
}
