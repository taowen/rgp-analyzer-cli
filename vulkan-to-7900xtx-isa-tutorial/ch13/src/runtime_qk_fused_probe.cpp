#include "ch13_qk_fused_path.h"

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
    if (argc != 4 && argc != 5) {
        fail("usage: runtime_qk_fused_probe <runtime.gguf> <layer_probe_dir> <layer_idx> [gpu|cpu]");
    }

    const int layer = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];
    const std::string backend_name = argc >= 5 ? argv[4] : "gpu";

    Ch13RuntimeGGUF runtime;
    runtime.load(argv[1], backend_name == "cpu" ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);

    Ch13QKFusedConfig cfg;
    cfg.layer = layer;
    cfg.hidden = (int) runtime.get_u32("runtime.hidden_size");
    cfg.head_dim = (int) runtime.get_u32("runtime.head_dim");
    cfg.n_head = (int) runtime.get_u32("runtime.n_head");
    cfg.n_kv_head = (int) runtime.get_u32("runtime.n_kv_head");
    cfg.seq_len = (int) runtime.get_u32("runtime.cond_seq_len");
    cfg.rms_norm_eps = runtime.get_f32("runtime.rms_norm_eps");
    cfg.rope_freq_base = runtime.get_f32("runtime.rope_freq_base");
    cfg.rope_n_ctx_orig = (int) runtime.get_u32("runtime.rope_n_ctx_orig");

    auto attn_input = read_f32(probe_dir + "/attn_input_ref_f32.bin");
    auto q_ref = read_f32(probe_dir + "/q_rope_ref_f32.bin");
    auto k_ref = read_f32(probe_dir + "/k_rope_ref_f32.bin");

    auto out = ch13_run_qk_fused_path(runtime, cfg, attn_input, backend_name);
    auto q_diff = diff_stats(out.q_rope, q_ref, "q_rope");
    auto k_diff = diff_stats(out.k_rope, k_ref, "k_rope");

    std::cout << "probe.layer=" << layer << "\n";
    std::cout << "probe.backend=" << backend_name << "\n";
    std::cout << "qk_fused.q_rope.max_abs_diff=" << q_diff.max_abs_diff << "\n";
    std::cout << "qk_fused.q_rope.mean_abs_diff=" << q_diff.mean_abs_diff << "\n";
    std::cout << "qk_fused.k_rope.max_abs_diff=" << k_diff.max_abs_diff << "\n";
    std::cout << "qk_fused.k_rope.mean_abs_diff=" << k_diff.mean_abs_diff << "\n";
    std::cout << "qk_fused_probe_ok=true\n";
    return 0;
}
