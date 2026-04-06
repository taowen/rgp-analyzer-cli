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
    if (argc != 4) {
        fail("usage: ch13_runtime_gguf_layer_fused_qk_probe <runtime.gguf> <layer_probe_dir> <layer_idx>");
    }

    const int il = std::atoi(argv[3]);
    const std::string probe_dir = argv[2];
    const auto layer_input = read_f32(probe_dir + "/layer_input_ref_f32.bin");

    Ch13RuntimeGGUF rt;
    rt.load(argv[1], GGML_BACKEND_DEVICE_TYPE_GPU);
    const auto out = ch13_run_layer_with_fused_qk(rt, il, layer_input);

    const auto attn_diff = diff_stats(out.attn_out, read_f32(probe_dir + "/attn_out_ref_f32.bin"), "attn_out");
    const auto o_diff = diff_stats(out.o_proj, read_f32(probe_dir + "/o_proj_ref_f32.bin"), "o_proj");
    const auto residual_diff = diff_stats(out.attn_residual, read_f32(probe_dir + "/attn_residual_ref_f32.bin"), "attn_residual");
    const auto mlp_down_diff = diff_stats(out.mlp_down, read_f32(probe_dir + "/mlp_down_ref_f32.bin"), "mlp_down");
    const auto layer_out_diff = diff_stats(out.layer_out, read_f32(probe_dir + "/layer_out_ref_f32.bin"), "layer_out");

    std::cout << "probe.layer=" << il << "\n";
    std::cout << "probe.backend=gpu-fused-qk\n";
    std::cout << "layer_fused.attn_out.max_abs_diff=" << attn_diff.max_abs_diff << "\n";
    std::cout << "layer_fused.attn_out.mean_abs_diff=" << attn_diff.mean_abs_diff << "\n";
    std::cout << "layer_fused.o_proj.max_abs_diff=" << o_diff.max_abs_diff << "\n";
    std::cout << "layer_fused.o_proj.mean_abs_diff=" << o_diff.mean_abs_diff << "\n";
    std::cout << "layer_fused.attn_residual.max_abs_diff=" << residual_diff.max_abs_diff << "\n";
    std::cout << "layer_fused.attn_residual.mean_abs_diff=" << residual_diff.mean_abs_diff << "\n";
    std::cout << "layer_fused.mlp_down.max_abs_diff=" << mlp_down_diff.max_abs_diff << "\n";
    std::cout << "layer_fused.mlp_down.mean_abs_diff=" << mlp_down_diff.mean_abs_diff << "\n";
    std::cout << "layer_fused.layer_out.max_abs_diff=" << layer_out_diff.max_abs_diff << "\n";
    std::cout << "layer_fused.layer_out.mean_abs_diff=" << layer_out_diff.mean_abs_diff << "\n";
    std::cout << "layer_fused_probe_ok=true\n";
    return 0;
}
