#include "ch13_gguf_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        fail("usage: ch13_runtime_gguf_smoke <runtime.gguf>");
    }

    Ch13RuntimeGGUF rt;
    rt.load(argv[1]);

    ggml_tensor * output_norm = rt.get_tensor("runtime.output_norm.weight");
    ggml_tensor * audio_heads = rt.get_tensor("runtime.audio_heads.weight");
    ggml_tensor * cond_input_ids = rt.get_tensor("runtime.iterative.cond.input_ids");
    ggml_tensor * cond_audio_mask = rt.get_tensor("runtime.iterative.cond.audio_mask");

    std::cout << "general.architecture=" << rt.get_str("general.architecture") << "\n";
    std::cout << "runtime.n_layer=" << rt.get_u32("runtime.n_layer") << "\n";
    std::cout << "runtime.hidden_size=" << rt.get_u32("runtime.hidden_size") << "\n";
    std::cout << "runtime.target_len=" << rt.get_u32("runtime.target_len") << "\n";
    std::cout << "runtime.num_step=" << rt.get_u32("runtime.num_step") << "\n";
    std::cout << "tensor_count=" << rt.tensor_count() << "\n";
    std::cout << "output_norm_checksum=" << ch13_checksum_tensor_f32(output_norm) << "\n";
    std::cout << "audio_heads_checksum=" << ch13_checksum_tensor_f32(audio_heads) << "\n";
    std::cout << "cond_input_ids_checksum=" << ch13_checksum_tensor_i32(cond_input_ids) << "\n";
    std::cout << "cond_audio_mask_checksum=" << ch13_checksum_tensor_i8(cond_audio_mask) << "\n";
    std::cout << "backend_load_ok=true\n";
    return 0;
}
