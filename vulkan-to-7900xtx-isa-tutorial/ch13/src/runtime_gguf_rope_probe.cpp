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
    if (argc != 4 && argc != 5) {
        fail("usage: ch13_runtime_gguf_rope_probe <runtime.gguf> <layer_probe_dir> <layer_idx> [gpu|cpu]");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];
    const std::string backend_name = argc >= 5 ? argv[4] : "gpu";
    const bool use_cpu = backend_name == "cpu";

    Ch13RuntimeGGUF rt;
    rt.load(argv[1], use_cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);

    ComputeCtx compute;
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    compute.ctx = ggml_init(params);
    if (!compute.ctx) fail("failed to init compute ctx");

    const uint32_t head_dim = rt.get_u32("runtime.head_dim");
    const uint32_t n_head = rt.get_u32("runtime.n_head");
    const uint32_t n_kv_head = rt.get_u32("runtime.n_kv_head");
    const uint32_t seq_len = rt.get_u32("runtime.cond_seq_len");
    const float eps = rt.get_f32("runtime.rms_norm_eps");
    const float rope_base = rt.get_f32("runtime.rope_freq_base");
    const int rope_ctx = (int) rt.get_u32("runtime.rope_n_ctx_orig");

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 96, false);
    if (!gf) fail("failed to init graph");

    ggml_tensor * q_in = ggml_new_tensor_3d(compute.ctx, GGML_TYPE_F32, head_dim, n_head, seq_len);
    ggml_set_name(q_in, "rope_probe.q_in");
    ggml_set_input(q_in);

    ggml_tensor * k_in = ggml_new_tensor_3d(compute.ctx, GGML_TYPE_F32, head_dim, n_kv_head, seq_len);
    ggml_set_name(k_in, "rope_probe.k_in");
    ggml_set_input(k_in);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(compute.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "rope_probe.inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * q_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, q_in, eps), ggml_repeat(compute.ctx, rt.get_tensor(layer_key(il, "q_norm.weight").c_str()), q_in));
    ggml_tensor * k_norm = ggml_mul(compute.ctx, ggml_rms_norm(compute.ctx, k_in, eps), ggml_repeat(compute.ctx, rt.get_tensor(layer_key(il, "k_norm.weight").c_str()), k_in));
    ggml_tensor * q_rope = ggml_rope_ext(compute.ctx, q_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope = ggml_rope_ext(compute.ctx, k_norm, inp_pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, rope_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_set_name(q_norm, "rope_probe.q_norm");
    ggml_set_name(k_norm, "rope_probe.k_norm");
    ggml_set_name(q_rope, "rope_probe.q_rope");
    ggml_set_name(k_rope, "rope_probe.k_rope");
    for (ggml_tensor * t : {q_norm, k_norm, q_rope, k_rope}) {
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

    auto q_in_ref = read_f32(probe_dir + "/q_proj_ref_f32.bin");
    auto k_in_ref = read_f32(probe_dir + "/k_proj_ref_f32.bin");
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rope_probe.q_in"), q_in_ref.data(), 0, q_in_ref.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rope_probe.k_in"), k_in_ref.data(), 0, k_in_ref.size() * sizeof(float));
    std::vector<int32_t> pos(seq_len);
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t) i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rope_probe.inp_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));

    if (use_cpu) {
        if (ggml_backend_graph_compute(rt.backend, gf) != GGML_STATUS_SUCCESS) fail("failed to compute graph");
    } else {
        if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) fail("failed to compute graph");
    }

    const auto qn = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf, "rope_probe.q_norm")), read_f32(probe_dir + "/q_norm_ref_f32.bin"), "q_norm");
    const auto kn = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf, "rope_probe.k_norm")), read_f32(probe_dir + "/k_norm_ref_f32.bin"), "k_norm");
    const auto qr = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf, "rope_probe.q_rope")), read_f32(probe_dir + "/q_rope_ref_f32.bin"), "q_rope");
    const auto kr = diff_stats(tensor_to_f32(ggml_graph_get_tensor(gf, "rope_probe.k_rope")), read_f32(probe_dir + "/k_rope_ref_f32.bin"), "k_rope");

    std::cout << "probe.layer=" << il << "\n";
    std::cout << "probe.backend=" << backend_name << "\n";
    std::cout << "rope_probe.q_norm.max_abs_diff=" << qn.max_abs_diff << "\n";
    std::cout << "rope_probe.q_norm.mean_abs_diff=" << qn.mean_abs_diff << "\n";
    std::cout << "rope_probe.k_norm.max_abs_diff=" << kn.max_abs_diff << "\n";
    std::cout << "rope_probe.k_norm.mean_abs_diff=" << kn.mean_abs_diff << "\n";
    std::cout << "rope_probe.q_rope.max_abs_diff=" << qr.max_abs_diff << "\n";
    std::cout << "rope_probe.q_rope.mean_abs_diff=" << qr.mean_abs_diff << "\n";
    std::cout << "rope_probe.k_rope.max_abs_diff=" << kr.max_abs_diff << "\n";
    std::cout << "rope_probe.k_rope.mean_abs_diff=" << kr.mean_abs_diff << "\n";
    std::cout << "rope_probe_ok=true\n";

    if (!use_cpu) {
        ggml_backend_sched_reset(rt.sched);
    }
    return 0;
}
