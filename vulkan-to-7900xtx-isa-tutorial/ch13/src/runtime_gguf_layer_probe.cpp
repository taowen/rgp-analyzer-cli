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

struct PeakStats {
    size_t index = 0;
    float got = 0.0f;
    float ref = 0.0f;
    float abs_diff = 0.0f;
    float got_abs_max = 0.0f;
    float ref_abs_max = 0.0f;
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

PeakStats peak_stats(const std::vector<float> & got, const std::vector<float> & ref, const std::string & name) {
    if (got.size() != ref.size()) fail("size mismatch for " + name);
    PeakStats stats;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        if (diff > stats.abs_diff) {
            stats.index = i;
            stats.got = got[i];
            stats.ref = ref[i];
            stats.abs_diff = diff;
        }
        stats.got_abs_max = std::max(stats.got_abs_max, std::fabs(got[i]));
        stats.ref_abs_max = std::max(stats.ref_abs_max, std::fabs(ref[i]));
    }
    return stats;
}

std::string layer_key(int il, const std::string & suffix) {
    return "runtime.layers." + std::string(il < 10 ? "0" : "") + std::to_string(il) + "." + suffix;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5 && argc != 6) {
        fail("usage: ch13_runtime_gguf_layer_probe <runtime.gguf> <layer_probe_dir> <layer_idx> [gpu|cpu] [cpu_stage]");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];
    const std::string backend_name = argc >= 5 ? argv[4] : "gpu";
    const std::string cpu_stage = argc >= 6 ? argv[5] : "";
    const bool do_f16_branches = backend_name != "cpu";

    Ch13RuntimeGGUF rt;
    rt.load(argv[1], backend_name == "cpu" ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);

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

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 512, false);
    if (!gf) fail("failed to init compute graph");

    ggml_tensor * inp = ggml_new_tensor_2d(compute.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp, "runtime.probe.input");
    ggml_set_input(inp);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(compute.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "runtime.probe.inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * attn_norm_weight = rt.get_tensor(layer_key(il, "attn_norm.weight").c_str());
    ggml_tensor * q_proj_weight = rt.get_tensor(layer_key(il, "q_proj.weight").c_str());
    ggml_tensor * k_proj_weight = rt.get_tensor(layer_key(il, "k_proj.weight").c_str());
    ggml_tensor * v_proj_weight = rt.get_tensor(layer_key(il, "v_proj.weight").c_str());
    ggml_tensor * q_norm_weight = rt.get_tensor(layer_key(il, "q_norm.weight").c_str());
    ggml_tensor * k_norm_weight = rt.get_tensor(layer_key(il, "k_norm.weight").c_str());
    ggml_tensor * o_proj_weight = rt.get_tensor(layer_key(il, "o_proj.weight").c_str());
    ggml_tensor * post_attn_norm_weight = rt.get_tensor(layer_key(il, "post_attention_norm.weight").c_str());
    ggml_tensor * gate_proj_weight = rt.get_tensor(layer_key(il, "gate_proj.weight").c_str());
    ggml_tensor * up_proj_weight = rt.get_tensor(layer_key(il, "up_proj.weight").c_str());
    ggml_tensor * down_proj_weight = rt.get_tensor(layer_key(il, "down_proj.weight").c_str());

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
    q_rope = ggml_cont(compute.ctx, q_rope);
    k_rope = ggml_cont(compute.ctx, k_rope);

    ggml_tensor * q_attn = ggml_permute(compute.ctx, q_rope, 0, 2, 1, 3);
    ggml_tensor * k_attn = ggml_permute(compute.ctx, k_rope, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(compute.ctx, v_3d, 0, 2, 1, 3);
    ggml_tensor * attn = ggml_flash_attn_ext(compute.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor * attn_out = ggml_cont(compute.ctx, ggml_reshape_2d(compute.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
    ggml_tensor * o_proj = ggml_mul_mat(compute.ctx, o_proj_weight, attn_out);
    ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);

    ggml_tensor * attn_residual = ggml_add(compute.ctx, inp, o_proj);
    ggml_tensor * post_attn_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, attn_residual, eps), ggml_repeat(compute.ctx, post_attn_norm_weight, attn_residual));
    ggml_tensor * gate_proj = ggml_mul_mat(compute.ctx, gate_proj_weight, post_attn_norm);
    ggml_tensor * up_proj = ggml_mul_mat(compute.ctx, up_proj_weight, post_attn_norm);
    ggml_mul_mat_set_prec(gate_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(up_proj, GGML_PREC_F32);
    ggml_tensor * gate_silu = ggml_silu(compute.ctx, gate_proj);
    ggml_tensor * mlp_act = ggml_mul(compute.ctx, gate_silu, up_proj);

    ggml_tensor * post_attn_norm_f16w = do_f16_branches ? ggml_cast(compute.ctx, post_attn_norm, GGML_TYPE_F16) : nullptr;
    ggml_tensor * gate_w_f16 = do_f16_branches ? ggml_cast(compute.ctx, gate_proj_weight, GGML_TYPE_F16) : nullptr;
    ggml_tensor * up_w_f16 = do_f16_branches ? ggml_cast(compute.ctx, up_proj_weight, GGML_TYPE_F16) : nullptr;
    ggml_tensor * down_w_f16 = do_f16_branches ? ggml_cast(compute.ctx, down_proj_weight, GGML_TYPE_F16) : nullptr;
    ggml_tensor * gate_proj_f16w = do_f16_branches ? ggml_mul_mat(compute.ctx, gate_w_f16, post_attn_norm_f16w) : nullptr;
    ggml_tensor * up_proj_f16w = do_f16_branches ? ggml_mul_mat(compute.ctx, up_w_f16, post_attn_norm_f16w) : nullptr;
    if (gate_proj_f16w) ggml_mul_mat_set_prec(gate_proj_f16w, GGML_PREC_F32);
    if (up_proj_f16w) ggml_mul_mat_set_prec(up_proj_f16w, GGML_PREC_F32);
    ggml_tensor * gate_proj_f16w_out = do_f16_branches ? ggml_cast(compute.ctx, gate_proj_f16w, GGML_TYPE_F32) : nullptr;
    ggml_tensor * up_proj_f16w_out = do_f16_branches ? ggml_cast(compute.ctx, up_proj_f16w, GGML_TYPE_F32) : nullptr;
    ggml_tensor * gate_silu_f16w = do_f16_branches ? ggml_silu(compute.ctx, ggml_cast(compute.ctx, gate_proj_f16w, GGML_TYPE_F16)) : nullptr;
    ggml_tensor * mlp_act_f16w = do_f16_branches ? ggml_mul(compute.ctx, gate_silu_f16w, ggml_cast(compute.ctx, up_proj_f16w, GGML_TYPE_F16)) : nullptr;
    ggml_tensor * mlp_act_f16w_out = do_f16_branches ? ggml_cast(compute.ctx, mlp_act_f16w, GGML_TYPE_F32) : nullptr;
    ggml_tensor * mlp_down_f16w = do_f16_branches ? ggml_mul_mat(compute.ctx, down_w_f16, ggml_cast(compute.ctx, mlp_act_f16w, GGML_TYPE_F16)) : nullptr;
    if (mlp_down_f16w) ggml_mul_mat_set_prec(mlp_down_f16w, GGML_PREC_F32);
    ggml_tensor * mlp_down_f16w_out = do_f16_branches ? ggml_cast(compute.ctx, mlp_down_f16w, GGML_TYPE_F32) : nullptr;

    ggml_tensor * post_attn_norm_f16 = do_f16_branches ? ggml_cast(compute.ctx, post_attn_norm, GGML_TYPE_F16) : nullptr;
    ggml_tensor * gate_proj_f16in = do_f16_branches ? ggml_mul_mat(compute.ctx, gate_proj_weight, post_attn_norm_f16) : nullptr;
    ggml_tensor * up_proj_f16in = do_f16_branches ? ggml_mul_mat(compute.ctx, up_proj_weight, post_attn_norm_f16) : nullptr;
    if (gate_proj_f16in) ggml_mul_mat_set_prec(gate_proj_f16in, GGML_PREC_F32);
    if (up_proj_f16in) ggml_mul_mat_set_prec(up_proj_f16in, GGML_PREC_F32);
    ggml_tensor * mlp_act_f16path = do_f16_branches ? ggml_mul(
        compute.ctx,
        ggml_silu(compute.ctx, ggml_cast(compute.ctx, gate_proj_f16in, GGML_TYPE_F16)),
        ggml_cast(compute.ctx, up_proj_f16in, GGML_TYPE_F16)) : nullptr;
    ggml_tensor * mlp_act_f16 = do_f16_branches ? ggml_cast(compute.ctx, mlp_act, GGML_TYPE_F16) : nullptr;
    ggml_tensor * mlp_down = ggml_mul_mat(compute.ctx, down_proj_weight, mlp_act);
    ggml_tensor * mlp_down_f16 = do_f16_branches ? ggml_mul_mat(compute.ctx, down_proj_weight, mlp_act_f16) : nullptr;
    ggml_mul_mat_set_prec(mlp_down, GGML_PREC_F32);
    if (mlp_down_f16) ggml_mul_mat_set_prec(mlp_down_f16, GGML_PREC_F32);
    ggml_tensor * mlp_down_f16_out = do_f16_branches ? ggml_cast(compute.ctx, mlp_down_f16, GGML_TYPE_F32) : nullptr;
    ggml_tensor * mlp_act_f16path_out = do_f16_branches ? ggml_cast(compute.ctx, mlp_act_f16path, GGML_TYPE_F32) : nullptr;
    ggml_tensor * layer_out = ggml_add(compute.ctx, attn_residual, mlp_down);

    if (!cpu_stage.empty()) {
        const auto cpu_backend = rt.cpu_backend ? rt.cpu_backend : rt.backend;
        const auto pin_cpu = [&](ggml_tensor * t) {
            ggml_backend_sched_set_tensor_backend(rt.sched, t, cpu_backend);
        };
        if (cpu_stage == "attn") {
            pin_cpu(attn_out);
        } else if (cpu_stage == "o_proj") {
            pin_cpu(o_proj);
        } else if (cpu_stage == "gate") {
            pin_cpu(gate_proj);
        } else if (cpu_stage == "up") {
            pin_cpu(up_proj);
        } else if (cpu_stage == "silu") {
            pin_cpu(gate_silu);
        } else if (cpu_stage == "mlp_act") {
            pin_cpu(mlp_act);
        } else if (cpu_stage == "mlp_down") {
            pin_cpu(mlp_down);
        } else if (cpu_stage == "mlp_chain") {
            for (ggml_tensor * t : {gate_proj, up_proj, gate_silu, mlp_act, mlp_down, layer_out}) {
                pin_cpu(t);
            }
        } else if (cpu_stage == "attn_chain") {
            for (ggml_tensor * t : {attn_out, o_proj, attn_residual}) {
                pin_cpu(t);
            }
        }
    }

    ggml_set_name(attn_input, "probe.attn_input");
    ggml_set_name(q_proj, "probe.q_proj");
    ggml_set_name(k_proj, "probe.k_proj");
    ggml_set_name(v_proj, "probe.v_proj");
    ggml_set_name(q_norm, "probe.q_norm");
    ggml_set_name(k_norm, "probe.k_norm");
    ggml_set_name(q_rope, "probe.q_rope");
    ggml_set_name(k_rope, "probe.k_rope");
    ggml_set_name(attn_out, "probe.attn_out");
    ggml_set_name(o_proj, "probe.o_proj");
    ggml_set_name(attn_residual, "probe.attn_residual");
    ggml_set_name(post_attn_norm, "probe.post_attn_norm");
    ggml_set_name(gate_proj, "probe.gate_proj");
    ggml_set_name(up_proj, "probe.up_proj");
    ggml_set_name(gate_silu, "probe.gate_silu");
    ggml_set_name(mlp_act, "probe.mlp_act");
    if (gate_proj_f16w_out) ggml_set_name(gate_proj_f16w_out, "probe.gate_proj_f16w");
    if (up_proj_f16w_out) ggml_set_name(up_proj_f16w_out, "probe.up_proj_f16w");
    if (mlp_act_f16w_out) ggml_set_name(mlp_act_f16w_out, "probe.mlp_act_f16w");
    if (mlp_down_f16w_out) ggml_set_name(mlp_down_f16w_out, "probe.mlp_down_f16w");
    if (gate_proj_f16in) ggml_set_name(gate_proj_f16in, "probe.gate_proj_f16in");
    if (up_proj_f16in) ggml_set_name(up_proj_f16in, "probe.up_proj_f16in");
    if (mlp_act_f16path_out) ggml_set_name(mlp_act_f16path_out, "probe.mlp_act_f16path");
    ggml_set_name(mlp_down, "probe.mlp_down");
    if (mlp_down_f16_out) ggml_set_name(mlp_down_f16_out, "probe.mlp_down_f16");
    ggml_set_name(layer_out, "probe.layer_out");

    for (ggml_tensor * t : {attn_input, q_proj, k_proj, v_proj, q_norm, k_norm, q_rope, k_rope, attn_out, o_proj, attn_residual, post_attn_norm, gate_proj, up_proj, gate_silu, mlp_act, mlp_down, layer_out}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }
    for (ggml_tensor * t : {gate_proj_f16w_out, up_proj_f16w_out, mlp_act_f16w_out, mlp_down_f16w_out, gate_proj_f16in, up_proj_f16in, mlp_act_f16path_out, mlp_down_f16_out}) {
        if (!t) continue;
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    const bool use_cpu_direct = backend_name == "cpu";
    if (use_cpu_direct) {
        compute.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        if (!compute.allocr) fail("failed to init CPU gallocr");
        if (!ggml_gallocr_alloc_graph(compute.allocr, gf)) fail("failed to allocate probe graph (cpu direct)");
    } else {
        ggml_backend_sched_reset(rt.sched);
        if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to allocate probe graph");
    }

    auto input_data = read_f32(probe_dir + "/layer_input_ref_f32.bin");
    ggml_tensor * inp_alloc = ggml_graph_get_tensor(gf, "runtime.probe.input");
    if (!inp_alloc) fail("failed to locate runtime.probe.input");
    ggml_backend_tensor_set(inp_alloc, input_data.data(), 0, input_data.size() * sizeof(float));

    std::vector<int32_t> pos(seq_len);
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t) i;
    ggml_tensor * pos_alloc = ggml_graph_get_tensor(gf, "runtime.probe.inp_pos");
    if (!pos_alloc) fail("failed to locate runtime.probe.inp_pos");
    ggml_backend_tensor_set(pos_alloc, pos.data(), 0, pos.size() * sizeof(int32_t));

    if (use_cpu_direct) {
        if (ggml_backend_graph_compute(rt.backend, gf) != GGML_STATUS_SUCCESS) fail("failed to compute probe graph (cpu direct)");
    } else {
        if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute probe graph");
    }

    const std::pair<const char *, const char *> refs[] = {
        {"probe.attn_input", "attn_input_ref_f32.bin"},
        {"probe.q_proj", "q_proj_ref_f32.bin"},
        {"probe.k_proj", "k_proj_ref_f32.bin"},
        {"probe.v_proj", "v_proj_ref_f32.bin"},
        {"probe.q_norm", "q_norm_ref_f32.bin"},
        {"probe.k_norm", "k_norm_ref_f32.bin"},
        {"probe.q_rope", "q_rope_ref_f32.bin"},
        {"probe.k_rope", "k_rope_ref_f32.bin"},
        {"probe.attn_out", "attn_out_ref_f32.bin"},
        {"probe.o_proj", "o_proj_ref_f32.bin"},
        {"probe.attn_residual", "attn_residual_ref_f32.bin"},
        {"probe.post_attn_norm", "post_attn_norm_ref_f32.bin"},
        {"probe.gate_proj", "gate_proj_ref_f32.bin"},
        {"probe.up_proj", "up_proj_ref_f32.bin"},
        {"probe.gate_silu", "gate_silu_ref_f32.bin"},
        {"probe.mlp_act", "mlp_act_ref_f32.bin"},
        {"probe.mlp_down", "mlp_down_ref_f32.bin"},
        {"probe.layer_out", "layer_out_ref_f32.bin"},
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
    if (do_f16_branches) {
        {
            auto * t = ggml_graph_get_tensor(gf, "probe.mlp_down_f16");
            if (!t) fail("missing graph tensor: probe.mlp_down_f16");
            const auto diff = diff_stats(tensor_to_f32(t), read_f32(probe_dir + "/mlp_down_ref_f32.bin"), "probe.mlp_down_f16");
            std::cout << "probe.mlp_down_f16.max_abs_diff=" << diff.max_abs_diff << "\n";
            std::cout << "probe.mlp_down_f16.mean_abs_diff=" << diff.mean_abs_diff << "\n";
        }
        {
            auto * t = ggml_graph_get_tensor(gf, "probe.gate_proj_f16in");
            if (!t) fail("missing graph tensor: probe.gate_proj_f16in");
            const auto diff = diff_stats(tensor_to_f32(t), read_f32(probe_dir + "/gate_proj_ref_f32.bin"), "probe.gate_proj_f16in");
            std::cout << "probe.gate_proj_f16in.max_abs_diff=" << diff.max_abs_diff << "\n";
            std::cout << "probe.gate_proj_f16in.mean_abs_diff=" << diff.mean_abs_diff << "\n";
        }
        {
            auto * t = ggml_graph_get_tensor(gf, "probe.up_proj_f16in");
            if (!t) fail("missing graph tensor: probe.up_proj_f16in");
            const auto diff = diff_stats(tensor_to_f32(t), read_f32(probe_dir + "/up_proj_ref_f32.bin"), "probe.up_proj_f16in");
            std::cout << "probe.up_proj_f16in.max_abs_diff=" << diff.max_abs_diff << "\n";
            std::cout << "probe.up_proj_f16in.mean_abs_diff=" << diff.mean_abs_diff << "\n";
        }
        {
            auto * t = ggml_graph_get_tensor(gf, "probe.mlp_act_f16path");
            if (!t) fail("missing graph tensor: probe.mlp_act_f16path");
            const auto diff = diff_stats(tensor_to_f32(t), read_f32(probe_dir + "/mlp_act_ref_f32.bin"), "probe.mlp_act_f16path");
            std::cout << "probe.mlp_act_f16path.max_abs_diff=" << diff.max_abs_diff << "\n";
            std::cout << "probe.mlp_act_f16path.mean_abs_diff=" << diff.mean_abs_diff << "\n";
        }
        for (const auto & item : std::vector<std::pair<std::string, std::string>>{
                 {"probe.gate_proj_f16w", "gate_proj_ref_f32.bin"},
                 {"probe.up_proj_f16w", "up_proj_ref_f32.bin"},
                 {"probe.mlp_act_f16w", "mlp_act_ref_f32.bin"},
                 {"probe.mlp_down_f16w", "mlp_down_ref_f32.bin"},
             }) {
            auto * t = ggml_graph_get_tensor(gf, item.first.c_str());
            if (!t) fail("missing graph tensor: " + item.first);
            const auto diff = diff_stats(tensor_to_f32(t), read_f32(probe_dir + "/" + item.second), item.first);
            std::cout << item.first << ".max_abs_diff=" << diff.max_abs_diff << "\n";
            std::cout << item.first << ".mean_abs_diff=" << diff.mean_abs_diff << "\n";
        }
    }
    for (const auto & item : std::vector<std::pair<std::string, std::string>>{
             {"probe.gate_proj", "gate_proj_ref_f32.bin"},
             {"probe.up_proj", "up_proj_ref_f32.bin"},
             {"probe.gate_silu", "gate_silu_ref_f32.bin"},
             {"probe.mlp_act", "mlp_act_ref_f32.bin"},
             {"probe.mlp_down", "mlp_down_ref_f32.bin"},
         }) {
        auto * t = ggml_graph_get_tensor(gf, item.first.c_str());
        if (!t) fail("missing graph tensor: " + item.first);
        const auto peak = peak_stats(tensor_to_f32(t), read_f32(probe_dir + "/" + item.second), item.first);
        std::cout << item.first << ".peak_index=" << peak.index << "\n";
        std::cout << item.first << ".peak_got=" << peak.got << "\n";
        std::cout << item.first << ".peak_ref=" << peak.ref << "\n";
        std::cout << item.first << ".peak_abs_diff=" << peak.abs_diff << "\n";
        std::cout << item.first << ".got_abs_max=" << peak.got_abs_max << "\n";
        std::cout << item.first << ".ref_abs_max=" << peak.ref_abs_max << "\n";
    }
    std::cout << "layer_probe_ok=true\n";
    if (!use_cpu_direct) {
        ggml_backend_sched_reset(rt.sched);
    }
    return 0;
}
