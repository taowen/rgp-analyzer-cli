#include "ggml.h"
#include "gguf.h"
#include <iostream>
#include <string>
#include <vector>

namespace {
[[noreturn]] void fail(const std::string & m) { std::cerr << m << std::endl; std::exit(1); }
void require_tensor(ggml_context * ctx, const char * name) {
    if (!ggml_get_tensor(ctx, name)) fail(std::string("missing tensor: ") + name);
}
}

int main(int argc, char ** argv) {
    if (argc != 2) fail("usage: ch13_inspect_runtime_gguf <runtime.gguf>");
    ggml_context * tctx = nullptr;
    gguf_init_params params{ true, &tctx };
    gguf_context * gctx = gguf_init_from_file(argv[1], params);
    if (!gctx || !tctx) fail("failed to load gguf");
    auto get_i32 = [&](const char * key) { auto id = gguf_find_key(gctx, key); if (id < 0) fail(std::string("missing key: ")+key); return gguf_get_val_i32(gctx, id); };
    auto get_u32 = [&](const char * key) { auto id = gguf_find_key(gctx, key); if (id < 0) fail(std::string("missing key: ")+key); return gguf_get_val_u32(gctx, id); };
    auto get_str = [&](const char * key) { auto id = gguf_find_key(gctx, key); if (id < 0) fail(std::string("missing key: ")+key); return std::string(gguf_get_val_str(gctx, id)); };

    std::cout << "general.architecture=" << get_str("general.architecture") << "\n";
    std::cout << "general.name=" << get_str("general.name") << "\n";
    std::cout << "runtime.n_layer=" << get_u32("runtime.n_layer") << "\n";
    std::cout << "runtime.hidden_size=" << get_u32("runtime.hidden_size") << "\n";
    std::cout << "runtime.num_step=" << get_u32("runtime.num_step") << "\n";
    std::cout << "runtime.cond_seq_len=" << get_u32("runtime.cond_seq_len") << "\n";
    std::cout << "runtime.target_len=" << get_u32("runtime.target_len") << "\n";
    std::cout << "runtime.quantizer_count=" << get_u32("runtime.quantizer_count") << "\n";
    require_tensor(tctx, "runtime.output_norm.weight");
    require_tensor(tctx, "runtime.audio_heads.weight");
    require_tensor(tctx, "runtime.iterative.cond.input_ids");
    require_tensor(tctx, "runtime.iterative.cond.audio_mask");
    require_tensor(tctx, "runtime.audio.weights.fc2__weight.f32.bin");
    require_tensor(tctx, "runtime.audio.weights.acoustic_decoder__conv1__weight.f32.bin");
    require_tensor(tctx, "runtime.audio.acoustic_decoder.block.0.conv_t1.weight");
    std::cout << "tensor_count=" << gguf_get_n_tensors(gctx) << "\n";
    std::cout << "kv_count=" << gguf_get_n_kv(gctx) << "\n";
    gguf_free(gctx); ggml_free(tctx); return 0;
}
