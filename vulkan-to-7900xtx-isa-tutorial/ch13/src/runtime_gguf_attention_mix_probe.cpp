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

std::string layer_key(int il, const std::string & suffix) {
    return "runtime.layers." + std::string(il < 10 ? "0" : "") + std::to_string(il) + "." + suffix;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        fail("usage: ch13_runtime_gguf_attention_mix_probe <runtime.gguf> <layer_probe_dir> <layer_idx> <mode> [gpu|cpu]");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];
    const std::string mode = argv[4]; // none | ref_q | ref_k | ref_qk | ref_v | ref_qkv
    const std::string backend_name = argc >= 6 ? argv[5] : "gpu";
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
    ggml_set_name(inp, "attention_mix.input");
    ggml_set_input(inp);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(compute.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "attention_mix.inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * q_ref = ggml_new_tensor_3d(compute.ctx, GGML_TYPE_F32, head_dim, n_head, seq_len);
    ggml_set_name(q_ref, "attention_mix.q_ref");
    ggml_set_input(q_ref);
    ggml_tensor * k_ref = ggml_new_tensor_3d(compute.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(k_ref, "attention_mix.k_ref");
    ggml_set_input(k_ref);
    ggml_tensor * v_ref = ggml_new_tensor_3d(compute.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(v_ref, "attention_mix.v_ref");
    ggml_set_input(v_ref);

    ggml_tensor * attn_input = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, inp, eps), ggml_repeat(compute.ctx, rt.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp));
    ggml_tensor * q_proj = ggml_mul_mat(compute.ctx, rt.get_tensor(layer_key(il, "q_proj.weight").c_str()), attn_input);
    ggml_tensor * k_proj = ggml_mul_mat(compute.ctx, rt.get_tensor(layer_key(il, "k_proj.weight").c_str()), attn_input);
    ggml_tensor * v_proj = ggml_mul_mat(compute.ctx, rt.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input);
    ggml_mul_mat_set_prec(q_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(k_proj, GGML_PREC_F32);
    ggml_mul_mat_set_prec(v_proj, GGML_PREC_F32);

    ggml_tensor * q_3d = ggml_reshape_3d(compute.ctx, q_proj, head_dim, n_head, seq_len);
    ggml_tensor * k_3d = ggml_reshape_3d(compute.ctx, k_proj, head_dim, n_kv_head, seq_len);
    ggml_tensor * v_3d = ggml_reshape_3d(compute.ctx, v_proj, head_dim, n_kv_head, seq_len);
    ggml_tensor * q_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, q_3d, eps), ggml_repeat(compute.ctx, rt.get_tensor(layer_key(il, "q_norm.weight").c_str()), q_3d));
    ggml_tensor * k_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, k_3d, eps), ggml_repeat(compute.ctx, rt.get_tensor(layer_key(il, "k_norm.weight").c_str()), k_3d));
    ggml_tensor * q_rope = ggml_rope_ext(compute.ctx, q_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope = ggml_rope_ext(compute.ctx, k_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * q_use = q_rope;
    ggml_tensor * k_use = k_rope;
    ggml_tensor * v_use = v_3d;
    if (mode == "ref_q" || mode == "ref_qk" || mode == "ref_qkv") q_use = q_ref;
    if (mode == "ref_k" || mode == "ref_qk" || mode == "ref_qkv") k_use = k_ref;
    if (mode == "ref_v" || mode == "ref_qkv") {
        v_use = v_ref;
    }

    ggml_tensor * q_attn = ggml_permute(compute.ctx, q_use, 0, 2, 1, 3);
    ggml_tensor * k_attn = ggml_permute(compute.ctx, k_use, 0, 2, 1, 3);
    ggml_tensor * v_attn = ggml_permute(compute.ctx, v_use, 0, 2, 1, 3);
    ggml_tensor * attn = ggml_flash_attn_ext(compute.ctx, q_attn, k_attn, v_attn, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_tensor * attn_out = ggml_cont(compute.ctx, ggml_reshape_2d(compute.ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]));
    ggml_tensor * o_proj = ggml_mul_mat(compute.ctx, rt.get_tensor(layer_key(il, "o_proj.weight").c_str()), attn_out);
    ggml_mul_mat_set_prec(o_proj, GGML_PREC_F32);
    ggml_tensor * attn_residual = ggml_add(compute.ctx, inp, o_proj);

    ggml_set_name(attn_out, "attention_mix.attn_out");
    ggml_set_name(o_proj, "attention_mix.o_proj");
    ggml_set_name(attn_residual, "attention_mix.attn_residual");
    for (ggml_tensor * t : {attn_out, o_proj, attn_residual}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    if (use_cpu) {
        compute.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        if (!compute.allocr) fail("failed to init CPU gallocr");
        if (!ggml_gallocr_alloc_graph(compute.allocr, gf)) fail("failed to alloc graph");
    } else {
        ggml_backend_sched_reset(rt.sched);
        if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) fail("failed to alloc graph");
    }

    auto layer_input = read_f32(probe_dir + "/layer_input_ref_f32.bin");
    auto q_ref_data = read_f32(probe_dir + "/q_rope_ref_f32.bin");
    auto k_ref_data = read_f32(probe_dir + "/k_rope_ref_f32.bin");
    auto v_ref_data = read_f32(probe_dir + "/v_proj_ref_f32.bin");
    auto set_if_present = [&](const char * name, const std::vector<float> & data) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (t) {
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        }
    };
    set_if_present("attention_mix.input", layer_input);
    set_if_present("attention_mix.q_ref", q_ref_data);
    set_if_present("attention_mix.k_ref", k_ref_data);
    set_if_present("attention_mix.v_ref", v_ref_data);
    std::vector<int32_t> pos(seq_len);
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t) i;
    ggml_tensor * pos_tensor = ggml_graph_get_tensor(gf, "attention_mix.inp_pos");
    if (pos_tensor) {
        ggml_backend_tensor_set(pos_tensor, pos.data(), 0, pos.size() * sizeof(int32_t));
    }

    if (use_cpu) {
        if (ggml_backend_graph_compute(rt.backend, gf) != GGML_STATUS_SUCCESS) fail("failed to compute graph");
    } else {
        if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute graph");
    }

    const auto attn_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf, "attention_mix.attn_out")), read_f32(probe_dir + "/attn_out_ref_f32.bin"), "attn_out");
    const auto oproj_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf, "attention_mix.o_proj")), read_f32(probe_dir + "/o_proj_ref_f32.bin"), "o_proj");
    const auto resid_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf, "attention_mix.attn_residual")), read_f32(probe_dir + "/attn_residual_ref_f32.bin"), "attn_residual");

    std::cout << "probe.layer=" << il << "\n";
    std::cout << "probe.backend=" << backend_name << "\n";
    std::cout << "probe.mode=" << mode << "\n";
    std::cout << "attention_mix.attn_out.max_abs_diff=" << attn_diff.max_abs_diff << "\n";
    std::cout << "attention_mix.attn_out.mean_abs_diff=" << attn_diff.mean_abs_diff << "\n";
    std::cout << "attention_mix.o_proj.max_abs_diff=" << oproj_diff.max_abs_diff << "\n";
    std::cout << "attention_mix.o_proj.mean_abs_diff=" << oproj_diff.mean_abs_diff << "\n";
    std::cout << "attention_mix.attn_residual.max_abs_diff=" << resid_diff.max_abs_diff << "\n";
    std::cout << "attention_mix.attn_residual.mean_abs_diff=" << resid_diff.mean_abs_diff << "\n";
    std::cout << "attention_mix_probe_ok=true\n";

    if (!use_cpu) {
        ggml_backend_sched_reset(rt.sched);
    }
    return 0;
}
