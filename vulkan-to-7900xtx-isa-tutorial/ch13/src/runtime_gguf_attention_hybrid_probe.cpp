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
    if (argc != 4) {
        fail("usage: ch13_runtime_gguf_attention_hybrid_probe <runtime.gguf> <layer_probe_dir> <layer_idx>");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];

    Ch13RuntimeGGUF rt_cpu;
    rt_cpu.load(argv[1], GGML_BACKEND_DEVICE_TYPE_CPU);
    Ch13RuntimeGGUF rt_gpu;
    rt_gpu.load(argv[1], GGML_BACKEND_DEVICE_TYPE_GPU);

    const uint32_t hidden = rt_cpu.get_u32("runtime.hidden_size");
    const uint32_t head_dim = rt_cpu.get_u32("runtime.head_dim");
    const uint32_t n_head = rt_cpu.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt_cpu.get_u32("runtime.n_kv_head");
    const uint32_t seq_len = rt_cpu.get_u32("runtime.cond_seq_len");
    const float eps = rt_cpu.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt_cpu.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt_cpu.get_u32("runtime.rope_n_ctx_orig");
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);

    const auto layer_input = read_f32(probe_dir + "/layer_input_ref_f32.bin");

    // CPU pass: compute q_rope / k_rope
    ComputeCtx cpu_ctx;
    cpu_ctx.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!cpu_ctx.ctx) fail("failed to init cpu ctx");
    ggml_cgraph * gf_cpu = ggml_new_graph_custom(cpu_ctx.ctx, 128, false);
    if (!gf_cpu) fail("failed to init cpu graph");

    ggml_tensor * inp_cpu = ggml_new_tensor_2d(cpu_ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp_cpu, "hybrid_cpu.input");
    ggml_set_input(inp_cpu);
    ggml_tensor * inp_pos_cpu = ggml_new_tensor_1d(cpu_ctx.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos_cpu, "hybrid_cpu.inp_pos");
    ggml_set_input(inp_pos_cpu);

    ggml_tensor * attn_input_cpu = ggml_mul(cpu_ctx.ctx, ggml_rms_norm(cpu_ctx.ctx, inp_cpu, eps), ggml_repeat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp_cpu));
    ggml_tensor * q_proj_cpu = ggml_mul_mat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "q_proj.weight").c_str()), attn_input_cpu);
    ggml_tensor * k_proj_cpu = ggml_mul_mat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "k_proj.weight").c_str()), attn_input_cpu);
    ggml_mul_mat_set_prec(q_proj_cpu, GGML_PREC_F32);
    ggml_mul_mat_set_prec(k_proj_cpu, GGML_PREC_F32);
    ggml_tensor * q_3d_cpu = ggml_reshape_3d(cpu_ctx.ctx, q_proj_cpu, head_dim, n_head, seq_len);
    ggml_tensor * k_3d_cpu = ggml_reshape_3d(cpu_ctx.ctx, k_proj_cpu, head_dim, n_kv_head, seq_len);
    ggml_tensor * q_norm_cpu = ggml_mul(cpu_ctx.ctx, ggml_rms_norm(cpu_ctx.ctx, q_3d_cpu, eps), ggml_repeat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "q_norm.weight").c_str()), q_3d_cpu));
    ggml_tensor * k_norm_cpu = ggml_mul(cpu_ctx.ctx, ggml_rms_norm(cpu_ctx.ctx, k_3d_cpu, eps), ggml_repeat(cpu_ctx.ctx, rt_cpu.get_tensor(layer_key(il, "k_norm.weight").c_str()), k_3d_cpu));
    ggml_tensor * q_rope_cpu = ggml_rope_ext(cpu_ctx.ctx, q_norm_cpu, inp_pos_cpu, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope_cpu = ggml_rope_ext(cpu_ctx.ctx, k_norm_cpu, inp_pos_cpu, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_set_name(q_rope_cpu, "hybrid_cpu.q_rope");
    ggml_set_name(k_rope_cpu, "hybrid_cpu.k_rope");
    for (ggml_tensor * t : {q_rope_cpu, k_rope_cpu}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf_cpu, t);
    }
    cpu_ctx.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt_cpu.backend));
    if (!cpu_ctx.allocr) fail("failed to init cpu gallocr");
    if (!ggml_gallocr_alloc_graph(cpu_ctx.allocr, gf_cpu)) fail("failed to alloc cpu graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_cpu, "hybrid_cpu.input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    std::vector<int32_t> pos(seq_len);
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t) i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_cpu, "hybrid_cpu.inp_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(rt_cpu.backend, gf_cpu) != GGML_STATUS_SUCCESS) fail("failed cpu qk pass");
    auto q_cpu = tensor_to_f32(ggml_graph_get_tensor(gf_cpu, "hybrid_cpu.q_rope"));
    auto k_cpu = tensor_to_f32(ggml_graph_get_tensor(gf_cpu, "hybrid_cpu.k_rope"));

    // GPU pass: compute v -> attention -> o_proj with CPU q/k
    ComputeCtx gpu_ctx;
    gpu_ctx.ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    if (!gpu_ctx.ctx) fail("failed to init gpu ctx");
    ggml_cgraph * gf_gpu = ggml_new_graph_custom(gpu_ctx.ctx, 160, false);
    if (!gf_gpu) fail("failed to init gpu graph");

    ggml_tensor * inp_gpu = ggml_new_tensor_2d(gpu_ctx.ctx, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(inp_gpu, "hybrid_gpu.input");
    ggml_set_input(inp_gpu);
    ggml_tensor * q_in = ggml_new_tensor_3d(gpu_ctx.ctx, GGML_TYPE_F32, head_dim, n_head, seq_len);
    ggml_set_name(q_in, "hybrid_gpu.q_in");
    ggml_set_input(q_in);
    ggml_tensor * k_in = ggml_new_tensor_3d(gpu_ctx.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(k_in, "hybrid_gpu.k_in");
    ggml_set_input(k_in);

    ggml_tensor * attn_input_gpu = ggml_mul(gpu_ctx.ctx, ggml_rms_norm(gpu_ctx.ctx, inp_gpu, eps), ggml_repeat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "attn_norm.weight").c_str()), inp_gpu));
    ggml_tensor * v_proj_gpu = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "v_proj.weight").c_str()), attn_input_gpu);
    ggml_mul_mat_set_prec(v_proj_gpu, GGML_PREC_F32);
    ggml_tensor * v_3d_gpu = ggml_reshape_3d(gpu_ctx.ctx, v_proj_gpu, head_dim, n_kv_head, seq_len);
    ggml_tensor * q_attn_gpu = ggml_permute(gpu_ctx.ctx, q_in, 0, 2, 1, 3);
    ggml_tensor * k_attn_gpu = ggml_permute(gpu_ctx.ctx, k_in, 0, 2, 1, 3);
    ggml_tensor * v_attn_gpu = ggml_permute(gpu_ctx.ctx, v_3d_gpu, 0, 2, 1, 3);
    ggml_tensor * attn_gpu = ggml_flash_attn_ext(gpu_ctx.ctx, q_attn_gpu, k_attn_gpu, v_attn_gpu, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn_gpu, GGML_PREC_F32);
    ggml_tensor * attn_out_gpu = ggml_cont(gpu_ctx.ctx, ggml_reshape_2d(gpu_ctx.ctx, attn_gpu, attn_gpu->ne[0] * attn_gpu->ne[1], attn_gpu->ne[2] * attn_gpu->ne[3]));
    ggml_tensor * o_proj_gpu = ggml_mul_mat(gpu_ctx.ctx, rt_gpu.get_tensor(layer_key(il, "o_proj.weight").c_str()), attn_out_gpu);
    ggml_mul_mat_set_prec(o_proj_gpu, GGML_PREC_F32);
    ggml_set_name(attn_out_gpu, "hybrid_gpu.attn_out");
    ggml_set_name(o_proj_gpu, "hybrid_gpu.o_proj");
    for (ggml_tensor * t : {attn_out_gpu, o_proj_gpu}) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf_gpu, t);
    }
    ggml_backend_sched_reset(rt_gpu.sched);
    if (!ggml_backend_sched_alloc_graph(rt_gpu.sched, gf_gpu)) fail("failed to alloc gpu graph");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_gpu, "hybrid_gpu.input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_gpu, "hybrid_gpu.q_in"), q_cpu.data(), 0, q_cpu.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_gpu, "hybrid_gpu.k_in"), k_cpu.data(), 0, k_cpu.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(rt_gpu.sched, gf_gpu) != GGML_STATUS_SUCCESS) fail("failed gpu hybrid pass");

    const auto attn_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf_gpu, "hybrid_gpu.attn_out")), read_f32(probe_dir + "/attn_out_ref_f32.bin"), "attn_out");
    const auto oproj_diff = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf_gpu, "hybrid_gpu.o_proj")), read_f32(probe_dir + "/o_proj_ref_f32.bin"), "o_proj");
    std::cout << "probe.layer=" << il << "\n";
    std::cout << "probe.mode=cpu_qk_gpu_rest\n";
    std::cout << "attention_hybrid.attn_out.max_abs_diff=" << attn_diff.max_abs_diff << "\n";
    std::cout << "attention_hybrid.attn_out.mean_abs_diff=" << attn_diff.mean_abs_diff << "\n";
    std::cout << "attention_hybrid.o_proj.max_abs_diff=" << oproj_diff.max_abs_diff << "\n";
    std::cout << "attention_hybrid.o_proj.mean_abs_diff=" << oproj_diff.mean_abs_diff << "\n";
    std::cout << "attention_hybrid_probe_ok=true\n";
    ggml_backend_sched_reset(rt_gpu.sched);
    return 0;
}
