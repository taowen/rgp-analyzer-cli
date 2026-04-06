#include "ch13_gguf_runtime.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

struct ComputeCtx { ggml_context * ctx = nullptr; ~ComputeCtx() { if (ctx) ggml_free(ctx); } };

float checksum_output_f32(ggml_tensor * tensor) {
    const size_t n = ggml_nelements(tensor);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(tensor, buf.data(), 0, n * sizeof(float));
    double sum = 0.0;
    for (float v : buf) {
        sum += v;
    }
    return static_cast<float>(sum);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        fail("usage: ch13_runtime_gguf_first_graph <runtime.gguf>");
    }

    Ch13RuntimeGGUF rt;
    rt.load(argv[1]);

    ggml_tensor * x_input_ref = rt.get_tensor("runtime.iterative.cond.x_input_ref");
    ggml_tensor * output_norm = rt.get_tensor("runtime.output_norm.weight");
    ggml_tensor * audio_heads = rt.get_tensor("runtime.audio_heads.weight");

    ComputeCtx compute;
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    compute.ctx = ggml_init(params);
    if (!compute.ctx) {
        fail("failed to init compute ctx");
    }

    ggml_cgraph * gf = ggml_new_graph_custom(compute.ctx, 64, false);
    if (!gf) {
        fail("failed to init compute graph");
    }

    ggml_tensor * rms = ggml_rms_norm(compute.ctx, x_input_ref, rt.get_f32("runtime.rms_norm_eps"));
    ggml_tensor * scale = ggml_repeat(compute.ctx, output_norm, rms);
    ggml_tensor * normed = ggml_mul(compute.ctx, rms, scale);
    ggml_tensor * logits = ggml_mul_mat(compute.ctx, audio_heads, normed);
    ggml_set_name(logits, "runtime.first_graph.logits");
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(rt.sched);
    if (!ggml_backend_sched_alloc_graph(rt.sched, gf)) {
        fail("failed to allocate compute graph");
    }
    if (ggml_backend_sched_graph_compute(rt.sched, gf) != GGML_STATUS_SUCCESS) {
        fail("failed to compute first graph on backend");
    }

    std::cout << "general.architecture=" << rt.get_str("general.architecture") << "\n";
    std::cout << "runtime.hidden_size=" << rt.get_u32("runtime.hidden_size") << "\n";
    std::cout << "runtime.cond_seq_len=" << rt.get_u32("runtime.cond_seq_len") << "\n";
    std::cout << "runtime.audio_vocab_size=" << rt.get_u32("runtime.audio_vocab_size") << "\n";
    std::cout << "runtime.num_audio_codebook=" << rt.get_u32("runtime.num_audio_codebook") << "\n";
    std::cout << "logits_ne0=" << logits->ne[0] << "\n";
    std::cout << "logits_ne1=" << logits->ne[1] << "\n";
    std::cout << "x_input_ref_checksum=" << ch13_checksum_tensor_f32(x_input_ref) << "\n";
    std::cout << "first_graph_logits_checksum=" << checksum_output_f32(logits) << "\n";
    std::cout << "first_graph_ok=true\n";
    ggml_backend_sched_reset(rt.sched);
    return 0;
}
