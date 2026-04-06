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

std::string layer_key(int il, const std::string & suffix) {
    return "runtime.layers." + std::string(il < 10 ? "0" : "") + std::to_string(il) + "." + suffix;
}

std::string layer_ref_path(const std::string & base, int il) {
    return base + "/layer_" + (il < 10 ? "0" : "") + std::to_string(il) + "_out_ref_f32.bin";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        fail("usage: ch13_runtime_gguf_backbone_tail_hybrid <runtime.gguf> <backbone_refs_dir> <start_layer> [gpu|cpu_qk|gpu_fused_qk]");
    }

    const std::string refs_dir = argv[2];
    const int start_layer = std::atoi(argv[3]);
    const std::string mode = argc >= 5 ? argv[4] : "gpu";
    const bool cpu_qk = mode == "cpu_qk";
    const bool gpu_fused_qk = mode == "gpu_fused_qk";

    Ch13RuntimeGGUF rt_gpu;
    rt_gpu.load(argv[1], GGML_BACKEND_DEVICE_TYPE_GPU);
    Ch13RuntimeGGUF rt_cpu;
    if (cpu_qk) {
        rt_cpu.load(argv[1], GGML_BACKEND_DEVICE_TYPE_CPU);
    }

    const uint32_t hidden = rt_gpu.get_u32("runtime.hidden_size");
    const uint32_t head_dim = rt_gpu.get_u32("runtime.head_dim");
    const uint32_t n_head = rt_gpu.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt_gpu.get_u32("runtime.n_kv_head");
    const uint32_t seq_len = rt_gpu.get_u32("runtime.cond_seq_len");
    const uint32_t n_layer = rt_gpu.get_u32("runtime.n_layer");
    const float eps = rt_gpu.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt_gpu.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt_gpu.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    std::vector<float> cur;
    if (start_layer == 0) {
        cur = tensor_to_f32(rt_gpu.get_tensor("runtime.iterative.cond.x_input_ref"));
    } else {
        cur = read_f32(layer_ref_path(refs_dir, start_layer - 1));
    }

    std::vector<float> pos_f;
    std::vector<int32_t> pos_i(seq_len);
    for (size_t i = 0; i < pos_i.size(); ++i) pos_i[i] = (int32_t) i;

    float tail_max = 0.0f;
    double tail_mean = 0.0;
    int tail_layers = 0;

    for (int il = start_layer; il < (int)n_layer; ++il) {
        std::vector<float> qk_cpu_q;
        std::vector<float> qk_cpu_k;
        std::vector<float> qk_fused_q;
        std::vector<float> qk_fused_k;
        std::vector<float> fused_attn_input;
        std::vector<float> fused_v_proj;

        if (cpu_qk) {
            ComputeCtx cpu_ctx;
            cpu_ctx.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
            if (!cpu_ctx.ctx) fail("failed to init cpu ctx");
            ggml_cgraph * gf = ggml_new_graph_custom(cpu_ctx.ctx, 128, false);
            if (!gf) fail("failed to init cpu graph");

            ggml_tensor * inp = ggml_new_tensor_2d(cpu_ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
            ggml_set_name(inp, "tail_cpu.input");
            ggml_set_input(inp);
            ggml_tensor * inp_pos = ggml_new_tensor_1d(cpu_ctx.ctx, GGML_TYPE_I32, seq_len);
            ggml_set_name(inp_pos, "tail_cpu.inp_pos");
            ggml_set_input(inp_pos);

            ggml_tensor * attn_input = ggml_mul(cpu_ctx.ctx, ggml_rms_norm(cpu_ctx.ctx, inp, eps), ggml_repeat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp));
            ggml_tensor * q_proj = ggml_mul_mat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "q_proj.weight").c_str()), attn_input);
            ggml_tensor * k_proj = ggml_mul_mat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "k_proj.weight").c_str()), attn_input);
            ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
            ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);
            ggml_tensor * q_3d = ggml_reshape_3d(cpu_ctx.ctx, q_proj, head_dim, n_head, seq_len);
            ggml_tensor * k_3d = ggml_reshape_3d(cpu_ctx.ctx, k_proj, head_dim, n_kv_head, seq_len);
            ggml_tensor * q_norm = ggml_mul(cpu_ctx.ctx, ggml_rms_norm(cpu_ctx.ctx, q_3d, eps), ggml_repeat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "q_norm.weight").c_str()), q_3d));
            ggml_tensor * k_norm = ggml_mul(cpu_ctx.ctx, ggml_rms_norm(cpu_ctx.ctx, k_3d, eps), ggml_repeat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "k_norm.weight").c_str()), k_3d));
            ggml_tensor * q_rope = ggml_rope_ext(cpu_ctx.ctx, q_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * k_rope = ggml_rope_ext(cpu_ctx.ctx, k_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_set_name(q_rope, "tail_cpu.q_rope");
            ggml_set_name(k_rope, "tail_cpu.k_rope");
            for (ggml_tensor * t : {q_rope, k_rope}) {
                ggml_set_output(t);
                ggml_build_forward_expand(gf, t);
            }
            cpu_ctx.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt_cpu.backend));
            if (!cpu_ctx.allocr) fail("failed to init cpu gallocr");
            if (!ggml_gallocr_alloc_graph(cpu_ctx.allocr, gf)) fail("failed to alloc cpu graph");
            ggml_tensor * cpu_in = ggml_graph_get_tensor(gf, "tail_cpu.input");
            if (!cpu_in) fail("missing graph tensor: tail_cpu.input");
            ggml_backend_tensor_set(cpu_in, cur.data(), 0, cur.size() * sizeof(float));
            ggml_tensor * cpu_pos = ggml_graph_get_tensor(gf, "tail_cpu.inp_pos");
            if (!cpu_pos) fail("missing graph tensor: tail_cpu.inp_pos");
            ggml_backend_tensor_set(cpu_pos, pos_i.data(), 0, pos_i.size() * sizeof(int32_t));
            if (ggml_backend_graph_compute(rt_cpu.backend, gf) != GGML_STATUS_SUCCESS) fail("failed cpu qk graph");
            qk_cpu_q = tensor_to_f32(ggml_graph_get_tensor(gf, "tail_cpu.q_rope"));
            qk_cpu_k = tensor_to_f32(ggml_graph_get_tensor(gf, "tail_cpu.k_rope"));
        }

        if (gpu_fused_qk) {
            ComputeCtx stage_a;
            stage_a.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
            if (!stage_a.ctx) fail("failed to init fused stage_a ctx");
            ggml_cgraph * gf_a = ggml_new_graph_custom(stage_a.ctx, 128, false);
            if (!gf_a) fail("failed to init fused stage_a graph");

            ggml_tensor * inp_a = ggml_new_tensor_2d(stage_a.ctx, GGML_TYPE_F32, hidden, seq_len);
            ggml_set_name(inp_a, "tail_fused.input");
            ggml_set_input(inp_a);

            ggml_tensor * attn_input_a = ggml_mul(stage_a.ctx, ggml_rms_norm(stage_a.ctx, inp_a, eps), ggml_repeat(stage_a.ctx, rt_gpu.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp_a));
            ggml_tensor * v_proj_a = ggml_mul_mat(stage_a.ctx, rt_gpu.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input_a);
            ggml_mul_mat_set_prec(v_proj_a, GGML_PREC_F32);
            ggml_set_name(attn_input_a, "tail_fused.attn_input");
            ggml_set_name(v_proj_a, "tail_fused.v_proj");
            ggml_set_output(attn_input_a);
            ggml_set_output(v_proj_a);
            ggml_build_forward_expand(gf_a, attn_input_a);
            ggml_build_forward_expand(gf_a, v_proj_a);

            ggml_backend_sched_reset(rt_gpu.sched);
            if (!ggml_backend_sched_alloc_graph(rt_gpu.sched, gf_a)) fail("failed to alloc fused stage_a graph");
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf_a, "tail_fused.input"), cur.data(), 0, cur.size() * sizeof(float));
            if (ggml_backend_sched_graph_compute(rt_gpu.sched, gf_a) != GGML_STATUS_SUCCESS) fail("failed fused stage_a graph");
            fused_attn_input = tensor_to_f32(ggml_graph_get_tensor(gf_a, "tail_fused.attn_input"));
            fused_v_proj = tensor_to_f32(ggml_graph_get_tensor(gf_a, "tail_fused.v_proj"));
            ggml_backend_sched_reset(rt_gpu.sched);

            Ch13QKFusedConfig cfg;
            cfg.layer = il;
            cfg.hidden = static_cast<int>(hidden);
            cfg.head_dim = static_cast<int>(head_dim);
            cfg.n_head = static_cast<int>(n_head);
            cfg.n_kv_head = static_cast<int>(n_kv_head);
            cfg.seq_len = static_cast<int>(seq_len);
            cfg.rms_norm_eps = eps;
            cfg.rope_freq_base = rope_base;
            cfg.rope_n_ctx_orig = rope_ctx;
            const auto qk = ch13_run_qk_fused_path(rt_gpu, cfg, fused_attn_input, "gpu");
            qk_fused_q = qk.q_rope;
            qk_fused_k = qk.k_rope;
        }

        ComputeCtx gpu_ctx;
        gpu_ctx.ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
        if (!gpu_ctx.ctx) fail("failed to init gpu ctx");
        ggml_cgraph * gf = ggml_new_graph_custom(gpu_ctx.ctx, 256, false);
        if (!gf) fail("failed to init gpu graph");

        ggml_tensor * inp = ggml_new_tensor_2d(gpu_ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
        ggml_set_name(inp, "tail_gpu.input");
        ggml_set_input(inp);
        ggml_tensor * inp_pos = ggml_new_tensor_1d(gpu_ctx.ctx, GGML_TYPE_I32, seq_len);
        ggml_set_name(inp_pos, "tail_gpu.inp_pos");
        ggml_set_input(inp_pos);
        ggml_tensor * q_in = nullptr;
        ggml_tensor * k_in = nullptr;
        ggml_tensor * v_in = nullptr;
        const bool external_qk = cpu_qk || gpu_fused_qk;
        const bool external_v = gpu_fused_qk;
        if (external_qk) {
            q_in = ggml_new_tensor_3d(gpu_ctx.ctx, GGML_TYPE_F32, head_dim, n_head, seq_len);
            k_in = ggml_new_tensor_3d(gpu_ctx.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
            ggml_set_name(q_in, "tail_gpu.q_in");
            ggml_set_name(k_in, "tail_gpu.k_in");
            ggml_set_input(q_in);
            ggml_set_input(k_in);
        }
        if (external_v) {
            v_in = ggml_new_tensor_3d(gpu_ctx.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
            ggml_set_name(v_in, "tail_gpu.v_in");
            ggml_set_input(v_in);
        }

        ggml_tensor * attn_input = ggml_mul(gpu_ctx.ctx, ggml_rms_norm(gpu_ctx.ctx, inp, eps), ggml_repeat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp));
        ggml_tensor * q_proj = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "q_proj.weight").c_str()), attn_input);
        ggml_tensor * k_proj = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "k_proj.weight").c_str()), attn_input);
        ggml_tensor * v_proj = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input);
        ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
        ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);
        ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);
        ggml_tensor * q_3d = ggml_reshape_3d(gpu_ctx.ctx, q_proj, head_dim, n_head, seq_len);
        ggml_tensor * k_3d = ggml_reshape_3d(gpu_ctx.ctx, k_proj, head_dim, n_kv_head, seq_len);
        ggml_tensor * v_3d = ggml_reshape_3d(gpu_ctx.ctx, v_proj, head_dim, n_kv_head, seq_len);
        ggml_tensor * q_norm = ggml_mul(gpu_ctx.ctx, ggml_rms_norm(gpu_ctx.ctx, q_3d, eps), ggml_repeat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "q_norm.weight").c_str()), q_3d));
        ggml_tensor * k_norm = ggml_mul(gpu_ctx.ctx, ggml_rms_norm(gpu_ctx.ctx, k_3d, eps), ggml_repeat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "k_norm.weight").c_str()), k_3d));
        ggml_tensor * q_rope = ggml_rope_ext(gpu_ctx.ctx, q_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_rope = ggml_rope_ext(gpu_ctx.ctx, k_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * q_use = external_qk ? q_in : q_rope;
        ggml_tensor * k_use = external_qk ? k_in : k_rope;
        ggml_tensor * q_attn = ggml_permute(gpu_ctx.ctx, q_use, 0, 2, 1, 3);
        ggml_tensor * k_attn = ggml_permute(gpu_ctx.ctx, k_use, 0, 2, 1, 3);
        ggml_tensor * v_use = external_v ? v_in : v_3d;
        ggml_tensor * v_attn = ggml_permute(gpu_ctx.ctx, v_use, 0, 2, 1, 3);
        ggml_tensor * attn = ggml_flash_attn_ext(gpu_ctx.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        ggml_tensor * attn_2d = ggml_cont(gpu_ctx.ctx, ggml_reshape_2d(gpu_ctx.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
        ggml_tensor * o_proj = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "o_proj.weight").c_str()), attn_2d);
        ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
        ggml_tensor * attn_residual = ggml_add(gpu_ctx.ctx, inp, o_proj);
        ggml_tensor * mlp_in = ggml_mul(gpu_ctx.ctx, ggml_rms_norm(gpu_ctx.ctx, attn_residual, eps), ggml_repeat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "post_attention_norm.weight").c_str()), attn_residual));
        ggml_tensor * gate = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "gate_proj.weight").c_str()), mlp_in);
        ggml_tensor * up = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "up_proj.weight").c_str()), mlp_in);
        ggml_mul_mat_set_prec(gate, GGML_PREC_F32);
        ggml_mul_mat_set_prec(up, GGML_PREC_F32);
        ggml_tensor * mlp_act = ggml_mul(gpu_ctx.ctx, ggml_silu(gpu_ctx.ctx, gate), up);
        ggml_tensor * mlp_down = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "down_proj.weight").c_str()), mlp_act);
        ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);
        ggml_tensor * out = ggml_add(gpu_ctx.ctx, attn_residual, mlp_down);
        ggml_set_name(out, "tail_gpu.out");
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        ggml_backend_sched_reset(rt_gpu.sched);
        if (!ggml_backend_sched_alloc_graph(rt_gpu.sched, gf)) fail("failed to alloc gpu graph");
        auto set_if_present = [&](const char * name, const void * data, size_t nbytes) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (t) {
                ggml_backend_tensor_set(t, data, 0, nbytes);
            }
        };
        set_if_present("tail_gpu.input", cur.data(), cur.size() * sizeof(float));
        set_if_present("tail_gpu.inp_pos", pos_i.data(), pos_i.size() * sizeof(int32_t));
        if (cpu_qk) {
            set_if_present("tail_gpu.q_in", qk_cpu_q.data(), qk_cpu_q.size() * sizeof(float));
            set_if_present("tail_gpu.k_in", qk_cpu_k.data(), qk_cpu_k.size() * sizeof(float));
        } else if (gpu_fused_qk) {
            set_if_present("tail_gpu.q_in", qk_fused_q.data(), qk_fused_q.size() * sizeof(float));
            set_if_present("tail_gpu.k_in", qk_fused_k.data(), qk_fused_k.size() * sizeof(float));
            set_if_present("tail_gpu.v_in", fused_v_proj.data(), fused_v_proj.size() * sizeof(float));
        }
        if (ggml_backend_sched_graph_compute(rt_gpu.sched, gf) != GGML_STATUS_SUCCESS) fail("failed gpu layer graph");
        cur = tensor_to_f32(ggml_graph_get_tensor(gf, "tail_gpu.out"));
        ggml_backend_sched_reset(rt_gpu.sched);

        const auto diff = diff_stats(cur, read_f32(layer_ref_path(refs_dir, il)), "tail layer");
        tail_max = std::max(tail_max, diff.max_abs_diff);
        tail_mean += diff.mean_abs_diff;
        tail_layers++;
        std::cout << "tail.layer_" << il << ".max_abs_diff=" << diff.max_abs_diff << "\n";
        std::cout << "tail.layer_" << il << ".mean_abs_diff=" << diff.mean_abs_diff << "\n";
    }

    tail_mean /= std::max(1, tail_layers);
    const auto final_diff = diff_stats(cur, read_f32(refs_dir + "/layer_27_out_ref_f32.bin"), "tail final");

    ComputeCtx final_ctx;
    final_ctx.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!final_ctx.ctx) fail("failed to init final ctx");
    ggml_cgraph * gf_final = ggml_new_graph_custom(final_ctx.ctx, 64, false);
    if (!gf_final) fail("failed to init final graph");

    ggml_tensor * final_in = ggml_new_tensor_2d(final_ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(final_in, "tail_final.input");
    ggml_set_input(final_in);
    ggml_tensor * final_hidden = ggml_mul(final_ctx.ctx,
        ggml_rms_norm(final_ctx.ctx, final_in, eps),
        ggml_repeat(final_ctx.ctx, rt_gpu.get_tensor("runtime.output_norm.weight"), final_in));
    ggml_tensor * logits = ggml_mul_mat(final_ctx.ctx, rt_gpu.get_tensor("runtime.audio_heads.weight"), final_hidden);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_set_name(final_hidden, "tail_final.hidden");
    ggml_set_name(logits, "tail_final.logits");
    ggml_set_output(final_hidden);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf_final, final_hidden);
    ggml_build_forward_expand(gf_final, logits);

    ggml_backend_sched_reset(rt_gpu.sched);
    if (!ggml_backend_sched_alloc_graph(rt_gpu.sched, gf_final)) fail("failed to alloc final graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_final, "tail_final.input"), cur.data(), 0, cur.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(rt_gpu.sched, gf_final) != GGML_STATUS_SUCCESS) fail("failed final graph");

    const auto final_hidden_diff = diff_stats(
        tensor_to_f32(ggml_graph_get_tensor(gf_final, "tail_final.hidden")),
        read_f32(refs_dir + "/final_hidden_ref_f32.bin"),
        "tail final hidden");
    const auto logits_diff = diff_stats(
        tensor_to_f32(ggml_graph_get_tensor(gf_final, "tail_final.logits")),
        read_f32(refs_dir + "/logits_ref_f32.bin"),
        "tail logits");
    ggml_backend_sched_reset(rt_gpu.sched);

    std::cout << "tail.mode=" << mode << "\n";
    std::cout << "tail.start_layer=" << start_layer << "\n";
    std::cout << "tail.max_layer_abs_diff=" << tail_max << "\n";
    std::cout << "tail.mean_layer_mean_abs_diff=" << tail_mean << "\n";
    std::cout << "tail.final.max_abs_diff=" << final_diff.max_abs_diff << "\n";
    std::cout << "tail.final.mean_abs_diff=" << final_diff.mean_abs_diff << "\n";
    std::cout << "tail.final_hidden.max_abs_diff=" << final_hidden_diff.max_abs_diff << "\n";
    std::cout << "tail.final_hidden.mean_abs_diff=" << final_hidden_diff.mean_abs_diff << "\n";
    std::cout << "tail.logits.max_abs_diff=" << logits_diff.max_abs_diff << "\n";
    std::cout << "tail.logits.mean_abs_diff=" << logits_diff.mean_abs_diff << "\n";
    std::cout << "tail_hybrid_ok=true\n";
    return 0;
}
