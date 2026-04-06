#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace {
[[noreturn]] void fail(const std::string & m) { std::cerr << m << std::endl; std::exit(1); }
std::unordered_map<std::string, std::string> read_kv(const fs::path & p) {
    std::ifstream in(p); if (!in) fail("failed to open " + p.string());
    std::unordered_map<std::string,std::string> out; std::string line;
    while (std::getline(in,line)) { auto pos=line.find('='); if (pos!=std::string::npos) out[line.substr(0,pos)] = line.substr(pos+1); }
    return out;
}
std::string read_text(const fs::path & p) { std::ifstream in(p); if (!in) fail("failed to open " + p.string()); std::stringstream ss; ss<<in.rdbuf(); return ss.str(); }
std::vector<char> read_bin(const fs::path & p) { std::ifstream in(p, std::ios::binary); if (!in) fail("failed to open " + p.string()); in.seekg(0,std::ios::end); size_t n=(size_t)in.tellg(); in.seekg(0,std::ios::beg); std::vector<char> d(n); in.read(d.data(), (std::streamsize)n); if(!in) fail("failed reading "+p.string()); return d; }
std::vector<int64_t> parse_shape(const std::string & manifest, const std::string & needle) {
    auto key = std::string("\"name\": \"") + needle + "\"";
    auto pos = manifest.find(key); if (pos == std::string::npos) fail("shape entry not found for " + needle);
    auto sp = manifest.find("\"shape\": [", pos); if (sp == std::string::npos) fail("shape not found for " + needle);
    auto l = manifest.find('[', sp); auto r = manifest.find(']', l); std::string inner = manifest.substr(l+1, r-l-1);
    std::vector<int64_t> dims; std::stringstream ss(inner); std::string item; while(std::getline(ss,item,',')) { std::stringstream ws(item); int64_t v=0; ws>>v; dims.push_back(v);} return dims;
}
std::vector<int64_t> rev(const std::vector<int64_t> & d) { return std::vector<int64_t>(d.rbegin(), d.rend()); }
void add_f32(gguf_context *g, ggml_context *tctx, std::vector<std::vector<char>> & storage, const fs::path & p, const std::string & name, const std::vector<int64_t> & dims) {
    auto rd = rev(dims); ggml_tensor * t = ggml_new_tensor(tctx, GGML_TYPE_F32, (int)rd.size(), rd.data()); ggml_set_name(t, name.c_str()); gguf_add_tensor(g,t); storage.push_back(read_bin(p)); gguf_set_tensor_data(g, name.c_str(), storage.back().data());
}
void add_i32(gguf_context *g, ggml_context *tctx, std::vector<std::vector<char>> & storage, const fs::path & p, const std::string & name, const std::vector<int64_t> & dims) {
    auto rd = rev(dims); ggml_tensor * t = ggml_new_tensor(tctx, GGML_TYPE_I32, (int)rd.size(), rd.data()); ggml_set_name(t, name.c_str()); gguf_add_tensor(g,t); storage.push_back(read_bin(p)); gguf_set_tensor_data(g, name.c_str(), storage.back().data());
}
void add_i8(gguf_context *g, ggml_context *tctx, std::vector<std::vector<char>> & storage, const fs::path & p, const std::string & name, const std::vector<int64_t> & dims) {
    auto rd = rev(dims); ggml_tensor * t = ggml_new_tensor(tctx, GGML_TYPE_I8, (int)rd.size(), rd.data()); ggml_set_name(t, name.c_str()); gguf_add_tensor(g,t); storage.push_back(read_bin(p)); gguf_set_tensor_data(g, name.c_str(), storage.back().data());
}
}
int main(int argc, char ** argv) {
    if (argc != 3) fail("usage: ch13_pack_runtime_gguf <chapter-output-dir> <out.gguf>");
    fs::path out_dir = argv[1]; fs::path export_dir = out_dir / "export"; fs::path iter_dir = out_dir / "iterative_decode"; fs::path audio_dir = out_dir / "audio_tokenizer_decode";
    auto meta = read_kv(export_dir / "meta.txt"); auto bmeta = read_kv(export_dir / "backbone_meta.txt"); auto imeta = read_kv(iter_dir / "meta.txt"); auto ameta = read_kv(audio_dir / "meta.txt"); auto amanifest = read_text(audio_dir / "manifest.json");
    gguf_context * g = gguf_init_empty(); ggml_init_params p{ ggml_tensor_overhead()*2048, nullptr, true }; ggml_context * tctx = ggml_init(p); if (!g || !tctx) fail("failed to init gguf/ggml contexts");
    std::vector<std::vector<char>> storage; storage.reserve(1024);
    gguf_set_val_str(g, "general.architecture", "ch13-runtime");
    gguf_set_val_str(g, "general.name", "ch13-vulkan-omnivoice-runtime");
    gguf_set_val_u32(g, "runtime.hidden_size", (uint32_t)std::stoi(meta.at("hidden_size")));
    gguf_set_val_u32(g, "runtime.n_head", (uint32_t)std::stoi(meta.at("n_head")));
    gguf_set_val_u32(g, "runtime.n_kv_head", (uint32_t)std::stoi(meta.at("n_kv_head")));
    gguf_set_val_u32(g, "runtime.head_dim", (uint32_t)std::stoi(meta.at("head_dim")));
    gguf_set_val_u32(g, "runtime.q_out", (uint32_t)std::stoi(meta.at("q_out")));
    gguf_set_val_u32(g, "runtime.k_out", (uint32_t)std::stoi(meta.at("k_out")));
    gguf_set_val_u32(g, "runtime.v_out", (uint32_t)std::stoi(meta.at("v_out")));
    gguf_set_val_u32(g, "runtime.n_layer", (uint32_t)std::stoi(bmeta.at("n_layer")));
    gguf_set_val_u32(g, "runtime.intermediate_size", (uint32_t)std::stoi(bmeta.at("intermediate_size")));
    gguf_set_val_u32(g, "runtime.audio_vocab_size", (uint32_t)std::stoi(bmeta.at("audio_vocab_size")));
    gguf_set_val_u32(g, "runtime.num_audio_codebook", (uint32_t)std::stoi(bmeta.at("num_audio_codebook")));
    gguf_set_val_u32(g, "runtime.rope_n_ctx_orig", (uint32_t)std::stoi(bmeta.at("max_position_embeddings")));
    gguf_set_val_f32(g, "runtime.rope_freq_base", std::stof(bmeta.at("rope_freq_base")));
    gguf_set_val_f32(g, "runtime.rms_norm_eps", std::stof(meta.at("eps")));
    gguf_set_val_u32(g, "runtime.cond_seq_len", (uint32_t)std::stoi(imeta.at("cond_seq_len")));
    gguf_set_val_u32(g, "runtime.target_len", (uint32_t)std::stoi(imeta.at("target_len")));
    gguf_set_val_u32(g, "runtime.num_step", (uint32_t)std::stoi(imeta.at("num_step")));
    gguf_set_val_u32(g, "runtime.tokenizer_sample_rate", (uint32_t)std::stoi(ameta.at("tokenizer_sample_rate")));
    gguf_set_val_u32(g, "runtime.quantizer_count", (uint32_t)std::stoi(ameta.at("quantizer_count")));
    gguf_set_val_u32(g, "runtime.codebook_size", (uint32_t)std::stoi(ameta.at("codebook_size")));
    { std::stringstream ss(imeta.at("schedule")); std::string item; std::vector<int32_t> sched; while(std::getline(ss,item,',')) if(!item.empty()) sched.push_back(std::stoi(item)); gguf_set_arr_data(g, "runtime.schedule", GGUF_TYPE_INT32, sched.data(), sched.size()); }
    add_f32(g,tctx,storage, export_dir/"output_norm_weight_f32.bin", "runtime.output_norm.weight", {std::stoll(meta.at("hidden_size"))});
    add_f32(g,tctx,storage, export_dir/"audio_heads_weight_f32.bin", "runtime.audio_heads.weight", {std::stoll(bmeta.at("audio_vocab_size"))*std::stoll(bmeta.at("num_audio_codebook")), std::stoll(meta.at("hidden_size"))});
    auto cond_meta = read_kv(iter_dir/"cond/meta.txt");
    add_i32(g,tctx,storage, iter_dir/"cond/input_ids_i32.bin", "runtime.iterative.cond.input_ids", {std::stoll(imeta.at("num_codebook")), std::stoll(imeta.at("cond_seq_len"))});
    add_i8(g,tctx,storage, iter_dir/"cond/audio_mask_u8.bin", "runtime.iterative.cond.audio_mask", {std::stoll(imeta.at("cond_seq_len"))});
    add_i32(g,tctx,storage, iter_dir/"cond/codebook_offsets_i32.bin", "runtime.iterative.cond.codebook_offsets", {std::stoll(imeta.at("num_codebook"))});
    add_i32(g,tctx,storage, iter_dir/"cond/text_unique_ids_i32.bin", "runtime.iterative.cond.text_unique_ids", {std::stoll(cond_meta.at("text_unique_count"))});
    add_f32(g,tctx,storage, iter_dir/"cond/text_embed_table_f32.bin", "runtime.iterative.cond.text_embed_table", {std::stoll(cond_meta.at("text_unique_count")), std::stoll(meta.at("hidden_size"))});
    add_i32(g,tctx,storage, iter_dir/"cond/audio_unique_shifted_ids_i32.bin", "runtime.iterative.cond.audio_unique_ids", {std::stoll(cond_meta.at("audio_unique_count"))});
    add_f32(g,tctx,storage, iter_dir/"cond/audio_embed_table_f32.bin", "runtime.iterative.cond.audio_embed_table", {std::stoll(cond_meta.at("audio_unique_count")), std::stoll(meta.at("hidden_size"))});
    add_f32(g,tctx,storage, iter_dir/"cond/x_input_ref_f32.bin", "runtime.iterative.cond.x_input_ref", {std::stoll(imeta.at("cond_seq_len")), std::stoll(meta.at("hidden_size"))});
    add_f32(g,tctx,storage, iter_dir/"cond/rope_cos_f32.bin", "runtime.iterative.cond.rope_cos", {std::stoll(imeta.at("cond_seq_len")), std::stoll(meta.at("head_dim"))});
    add_f32(g,tctx,storage, iter_dir/"cond/rope_sin_f32.bin", "runtime.iterative.cond.rope_sin", {std::stoll(imeta.at("cond_seq_len")), std::stoll(meta.at("head_dim"))});
    for (int il=0; il<std::stoi(bmeta.at("n_layer")); ++il) {
        char key[96]; fs::path layer_dir = export_dir/"layers"/(std::string("layer_") + (il<10?"0":"") + std::to_string(il));
        snprintf(key,sizeof(key),"runtime.layers.%02d.attn_norm.weight",il); add_f32(g,tctx,storage, layer_dir/"attn_norm_weight_f32.bin", key, {std::stoll(meta.at("hidden_size"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.q_proj.weight",il); add_f32(g,tctx,storage, layer_dir/"q_proj_weight_f32.bin", key, {std::stoll(meta.at("q_out")), std::stoll(meta.at("hidden_size"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.k_proj.weight",il); add_f32(g,tctx,storage, layer_dir/"k_proj_weight_f32.bin", key, {std::stoll(meta.at("k_out")), std::stoll(meta.at("hidden_size"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.v_proj.weight",il); add_f32(g,tctx,storage, layer_dir/"v_proj_weight_f32.bin", key, {std::stoll(meta.at("v_out")), std::stoll(meta.at("hidden_size"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.q_norm.weight",il); add_f32(g,tctx,storage, layer_dir/"q_norm_weight_f32.bin", key, {std::stoll(meta.at("head_dim"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.k_norm.weight",il); add_f32(g,tctx,storage, layer_dir/"k_norm_weight_f32.bin", key, {std::stoll(meta.at("head_dim"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.o_proj.weight",il); add_f32(g,tctx,storage, layer_dir/"o_proj_weight_f32.bin", key, {std::stoll(meta.at("hidden_size")), std::stoll(meta.at("q_out"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.post_attention_norm.weight",il); add_f32(g,tctx,storage, layer_dir/"post_attention_norm_weight_f32.bin", key, {std::stoll(meta.at("hidden_size"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.gate_proj.weight",il); add_f32(g,tctx,storage, layer_dir/"gate_proj_weight_f32.bin", key, {std::stoll(bmeta.at("intermediate_size")), std::stoll(meta.at("hidden_size"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.up_proj.weight",il); add_f32(g,tctx,storage, layer_dir/"up_proj_weight_f32.bin", key, {std::stoll(bmeta.at("intermediate_size")), std::stoll(meta.at("hidden_size"))});
        snprintf(key,sizeof(key),"runtime.layers.%02d.down_proj.weight",il); add_f32(g,tctx,storage, layer_dir/"down_proj_weight_f32.bin", key, {std::stoll(meta.at("hidden_size")), std::stoll(bmeta.at("intermediate_size"))});
    }
    add_i32(g,tctx,storage, audio_dir/"tokens_i32.bin", "runtime.audio.tokens", {std::stoll(ameta.at("num_codebooks")), std::stoll(ameta.at("seq_len"))});
    const std::vector<std::pair<std::string,std::vector<int64_t>>> fixed = {
        {"weights/fc2__weight.f32.bin", {256,1024}}, {"weights/fc2__bias.f32.bin", {256}},
        {"weights/acoustic_decoder__conv1__weight.f32.bin", {1024,256,7}}, {"weights/acoustic_decoder__conv1__bias.f32.bin", {1024}},
        {"weights/acoustic_decoder__snake1__alpha.f32.bin", {1,32,1}}, {"weights/acoustic_decoder__conv2__weight.f32.bin", {1,32,7}}, {"weights/acoustic_decoder__conv2__bias.f32.bin", {1}},
    };
    for (auto & kv : fixed) { std::string key = "runtime.audio." + kv.first; for (char & c : key) if (c=='/'||c=='-') c='.'; add_f32(g,tctx,storage, audio_dir/kv.first, key, kv.second); }
    const std::vector<std::pair<std::string,std::string>> entries = {
        {"snake1.alpha","snake1__alpha"}, {"conv_t1.weight","conv_t1__weight"}, {"conv_t1.bias","conv_t1__bias"},
        {"res_unit1.snake1.alpha","res_unit1__snake1__alpha"}, {"res_unit1.conv1.weight","res_unit1__conv1__weight"}, {"res_unit1.conv1.bias","res_unit1__conv1__bias"}, {"res_unit1.snake2.alpha","res_unit1__snake2__alpha"}, {"res_unit1.conv2.weight","res_unit1__conv2__weight"}, {"res_unit1.conv2.bias","res_unit1__conv2__bias"},
        {"res_unit2.snake1.alpha","res_unit2__snake1__alpha"}, {"res_unit2.conv1.weight","res_unit2__conv1__weight"}, {"res_unit2.conv1.bias","res_unit2__conv1__bias"}, {"res_unit2.snake2.alpha","res_unit2__snake2__alpha"}, {"res_unit2.conv2.weight","res_unit2__conv2__weight"}, {"res_unit2.conv2.bias","res_unit2__conv2__bias"},
        {"res_unit3.snake1.alpha","res_unit3__snake1__alpha"}, {"res_unit3.conv1.weight","res_unit3__conv1__weight"}, {"res_unit3.conv1.bias","res_unit3__conv1__bias"}, {"res_unit3.snake2.alpha","res_unit3__snake2__alpha"}, {"res_unit3.conv2.weight","res_unit3__conv2__weight"}, {"res_unit3.conv2.bias","res_unit3__conv2__bias"},
    };
    for (int bi=0; bi<5; ++bi) for (const auto & e : entries) { std::string mname = "acoustic_decoder.block." + std::to_string(bi) + "." + e.first; auto dims = parse_shape(amanifest,mname); fs::path file = audio_dir/"weights"/("acoustic_decoder__block__" + std::to_string(bi) + "__" + e.second + ".f32.bin"); std::string key = "runtime.audio." + mname; add_f32(g,tctx,storage,file,key,dims); }
    if (!gguf_write_to_file(g, argv[2], false)) fail("failed to write gguf");
    gguf_free(g); ggml_free(tctx); std::cout << "packed " << argv[2] << std::endl; return 0;
}
