#include "ch13_gguf_runtime.h"
#include "ch13_qk_fused_path.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

std::vector<float> tensor_to_f32(ggml_tensor * tensor) {
    std::vector<float> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<int32_t> tensor_to_i32(ggml_tensor * tensor) {
    std::vector<int32_t> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(int32_t));
    return out;
}

std::vector<int8_t> tensor_to_i8(ggml_tensor * tensor) {
    std::vector<int8_t> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(int8_t));
    return out;
}

std::vector<float> read_f32_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("failed to open file: " + path);
    in.seekg(0, std::ios::end);
    const size_t nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if ((nbytes % sizeof(float)) != 0) fail("invalid f32 file size: " + path);
    std::vector<float> out(nbytes / sizeof(float));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(nbytes));
    if (!in) fail("failed reading file: " + path);
    return out;
}

std::vector<int32_t> read_i32_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("failed to open file: " + path);
    in.seekg(0, std::ios::end);
    const size_t nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if ((nbytes % sizeof(int32_t)) != 0) fail("invalid i32 file size: " + path);
    std::vector<int32_t> out(nbytes / sizeof(int32_t));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(nbytes));
    if (!in) fail("failed reading file: " + path);
    return out;
}

void write_i32_file(const std::string & path, const std::vector<int32_t> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) fail("failed to open output file: " + path);
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(int32_t)));
    if (!out) fail("failed writing output file: " + path);
}

std::vector<uint32_t> get_u32_array(gguf_context * gguf_ctx, const char * key) {
    const auto id = gguf_find_key(gguf_ctx, key);
    if (id < 0) fail(std::string("missing array key: ") + key);
    const auto n = gguf_get_arr_n(gguf_ctx, id);
    const auto ty = gguf_get_arr_type(gguf_ctx, id);
    if (ty == GGUF_TYPE_UINT32) {
        const auto * data = static_cast<const uint32_t *>(gguf_get_arr_data(gguf_ctx, id));
        return std::vector<uint32_t>(data, data + n);
    }
    if (ty == GGUF_TYPE_INT32) {
        const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(gguf_ctx, id));
        std::vector<uint32_t> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint32_t>(data[i]);
        return out;
    }
    fail(std::string("unexpected array type for key: ") + key);
}

float logsumexp(const float * x, int n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) mx = std::max(mx, x[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp(double(x[i] - mx));
    return mx + std::log(float(sum));
}

// logits layout from ggml mul_mat output flattened as [seq][c*vocab]
inline float logits_at(const std::vector<float> & logits, int seq, int seq_len, int codebook, int vocab, int vocab_size, int num_codebook) {
    GGML_UNUSED(seq_len);
    const int feature = codebook * vocab_size + vocab;
    return logits[seq * (num_codebook * vocab_size) + feature];
}

std::vector<float> prepare_embed_inputs(
    const std::vector<int32_t> & input_ids, // [C,S] flattened c-major with seq contiguous
    const std::vector<int8_t> & audio_mask, // [S]
    int codebook_count,
    int seq_len,
    int hidden,
    const std::vector<int32_t> & codebook_offsets,
    const std::unordered_map<int32_t, int> & text_id_to_row,
    const std::vector<float> & text_embed_table,
    const std::vector<float> & audio_embed_table) {

    std::vector<float> out(size_t(seq_len) * hidden, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        float * dst = out.data() + size_t(t) * hidden;
        if (audio_mask[t]) {
            for (int c = 0; c < codebook_count; ++c) {
                const int32_t token = input_ids[c * seq_len + t];
                const int32_t shifted = token + codebook_offsets[c];
                const float * src = audio_embed_table.data() + size_t(shifted) * hidden;
                for (int h = 0; h < hidden; ++h) dst[h] += src[h];
            }
        } else {
            const int32_t token = input_ids[t];
            auto it = text_id_to_row.find(token);
            if (it == text_id_to_row.end()) fail("missing text embedding row for token id");
            const float * src = text_embed_table.data() + size_t(it->second) * hidden;
            std::copy(src, src + hidden, dst);
        }
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        fail("usage: ch13_runtime_gguf_iterative_decode <runtime.gguf> <output_dir> <reference_tokens_i32.bin>");
    }

    const std::string gguf_path = argv[1];
    const std::string out_dir = argv[2];
    const std::string ref_tokens_path = argv[3];

    Ch13RuntimeGGUF rt;
    rt.load(gguf_path.c_str(), GGML_BACKEND_DEVICE_TYPE_GPU);

    const int hidden = (int) rt.get_u32("runtime.hidden_size");
    const int codebook_count = (int) rt.get_u32("runtime.num_audio_codebook");
    const int target_len = (int) rt.get_u32("runtime.target_len");
    const int cond_seq_len = (int) rt.get_u32("runtime.cond_seq_len");
    const int vocab_size = (int) rt.get_u32("runtime.audio_vocab_size");
    const int audio_mask_id = vocab_size - 1;
    const float guidance_scale = 2.0f;
    const float layer_penalty_factor = 5.0f;

    const auto schedule_u32 = get_u32_array(rt.gguf_ctx, "runtime.schedule");
    std::vector<int> schedule(schedule_u32.begin(), schedule_u32.end());

    const auto cond_input_ids = tensor_to_i32(rt.get_tensor("runtime.iterative.cond.input_ids"));
    const auto cond_audio_mask_i8 = tensor_to_i8(rt.get_tensor("runtime.iterative.cond.audio_mask"));
    std::vector<int8_t> cond_audio_mask(cond_audio_mask_i8.begin(), cond_audio_mask_i8.end());
    const auto codebook_offsets = tensor_to_i32(rt.get_tensor("runtime.iterative.cond.codebook_offsets"));
    const auto text_unique_ids = tensor_to_i32(rt.get_tensor("runtime.iterative.cond.text_unique_ids"));
    const auto text_embed_table = tensor_to_f32(rt.get_tensor("runtime.iterative.cond.text_embed_table"));
    const auto audio_embed_table = tensor_to_f32(rt.get_tensor("runtime.iterative.cond.audio_embed_table"));

    std::unordered_map<int32_t, int> text_id_to_row;
    for (int i = 0; i < (int) text_unique_ids.size(); ++i) text_id_to_row[text_unique_ids[i]] = i;

    const int cond_target_start = cond_seq_len - target_len;
    std::vector<int32_t> sample_tokens(codebook_count * target_len, audio_mask_id);
    std::vector<int32_t> uncond_input_ids(codebook_count * target_len, audio_mask_id);
    std::vector<int8_t> uncond_audio_mask(target_len, 1);

    std::vector<int32_t> cond_work = cond_input_ids;

    for (int step = 0; step < (int) schedule.size(); ++step) {
        // update cond and uncond tensors from current sample_tokens
        for (int c = 0; c < codebook_count; ++c) {
            for (int t = 0; t < target_len; ++t) {
                const int idx = c * target_len + t;
                cond_work[c * cond_seq_len + (cond_target_start + t)] = sample_tokens[idx];
                uncond_input_ids[idx] = sample_tokens[idx];
            }
        }

        const auto cond_x = prepare_embed_inputs(cond_work, cond_audio_mask, codebook_count, cond_seq_len, hidden, codebook_offsets, text_id_to_row, text_embed_table, audio_embed_table);
        const auto uncond_x = prepare_embed_inputs(uncond_input_ids, uncond_audio_mask, codebook_count, target_len, hidden, codebook_offsets, text_id_to_row, text_embed_table, audio_embed_table);

        const auto cond_out = ch13_run_backbone_with_fused_qk(rt, cond_x, 0);
        const auto uncond_out = ch13_run_backbone_with_fused_qk(rt, uncond_x, 0);

        const int k = schedule[step];
        std::vector<int32_t> pred_tokens(codebook_count * target_len, 0);
        std::vector<float> scores(codebook_count * target_len, -std::numeric_limits<float>::infinity());
        std::vector<float> buf_c(vocab_size), buf_u(vocab_size), buf_mix(vocab_size);

        for (int c = 0; c < codebook_count; ++c) {
            for (int t = 0; t < target_len; ++t) {
                const int cond_pos = cond_target_start + t;
                const int uncond_pos = t;
                for (int v = 0; v < vocab_size; ++v) {
                    buf_c[v] = logits_at(cond_out.logits, cond_pos, cond_seq_len, c, v, vocab_size, codebook_count);
                    buf_u[v] = logits_at(uncond_out.logits, uncond_pos, target_len, c, v, vocab_size, codebook_count);
                }
                const float lse_c = logsumexp(buf_c.data(), vocab_size);
                const float lse_u = logsumexp(buf_u.data(), vocab_size);
                for (int v = 0; v < vocab_size; ++v) {
                    const float c_logp = buf_c[v] - lse_c;
                    const float u_logp = buf_u[v] - lse_u;
                    buf_mix[v] = c_logp + guidance_scale * (c_logp - u_logp);
                }
                const float lse_mix = logsumexp(buf_mix.data(), vocab_size);
                int best_v = 0;
                float best_logp = -std::numeric_limits<float>::infinity();
                for (int v = 0; v < vocab_size; ++v) {
                    float lp = buf_mix[v] - lse_mix;
                    if (v == audio_mask_id) lp = -std::numeric_limits<float>::infinity();
                    if (lp > best_logp) {
                        best_logp = lp;
                        best_v = v;
                    }
                }
                const int idx = c * target_len + t;
                pred_tokens[idx] = best_v;
                scores[idx] = best_logp - float(c) * layer_penalty_factor;
                if (sample_tokens[idx] != audio_mask_id) {
                    scores[idx] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        std::vector<int> order(scores.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + std::min<int>(k, (int)order.size()), order.end(),
            [&](int a, int b) { return scores[a] > scores[b]; });
        for (int i = 0; i < k && i < (int) order.size(); ++i) {
            const int idx = order[i];
            if (!std::isfinite(scores[idx])) continue;
            sample_tokens[idx] = pred_tokens[idx];
        }
    }

    const auto ref_tokens = read_i32_file(ref_tokens_path);
    int mismatch = 0;
    for (size_t i = 0; i < std::min(sample_tokens.size(), ref_tokens.size()); ++i) {
        if (sample_tokens[i] != ref_tokens[i]) mismatch++;
    }

    const std::string out_tokens = out_dir + "/generated_tokens_fused_i32.bin";
    write_i32_file(out_tokens, sample_tokens);

    std::cout << "iterative_decode.backend=gpu_fused_qk\n";
    std::cout << "iterative_decode.target_len=" << target_len << "\n";
    std::cout << "iterative_decode.codebook_count=" << codebook_count << "\n";
    std::cout << "iterative_decode.schedule_steps=" << schedule.size() << "\n";
    std::cout << "iterative_decode.token_mismatch_count=" << mismatch << "\n";
    std::cout << "iterative_decode.output_tokens=" << out_tokens << "\n";
    std::cout << "iterative_decode_ok=true\n";
    return 0;
}
