#pragma once

#include <string>
#include <vector>

struct Ch13QKFusedConfig {
    int layer = 0;
    int hidden = 0;
    int head_dim = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int seq_len = 0;
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    int rope_n_ctx_orig = 0;
};

struct Ch13QKFusedResult {
    std::vector<float> q_rope;
    std::vector<float> k_rope;
};

struct Ch13LayerFusedQKResult {
    std::vector<float> q_rope;
    std::vector<float> k_rope;
    std::vector<float> attn_out;
    std::vector<float> o_proj;
    std::vector<float> attn_residual;
    std::vector<float> mlp_down;
    std::vector<float> layer_out;
};

struct Ch13BackboneFusedQKResult {
    std::vector<float> final_hidden;
    std::vector<float> logits; // [seq, C*V] flattened row-major as ggml tensor copy layout
};

class Ch13RuntimeGGUF;

Ch13QKFusedResult ch13_run_qk_fused_path(
    Ch13RuntimeGGUF & runtime,
    const Ch13QKFusedConfig & cfg,
    const std::vector<float> & attn_input,
    const std::string & backend_name);

Ch13LayerFusedQKResult ch13_run_layer_with_fused_qk(
    Ch13RuntimeGGUF & runtime,
    int layer,
    const std::vector<float> & layer_input);

Ch13LayerFusedQKResult ch13_run_layer_plain_ggml(
    Ch13RuntimeGGUF & runtime,
    int layer,
    const std::vector<float> & layer_input);

Ch13BackboneFusedQKResult ch13_run_backbone_with_fused_qk(
    Ch13RuntimeGGUF & runtime,
    const std::vector<float> & x_input,
    int fused_from_layer = 0);
