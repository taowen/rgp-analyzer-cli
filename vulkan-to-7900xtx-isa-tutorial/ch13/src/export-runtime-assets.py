#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

LLAMA_CPP_GGUF_PY = Path.home() / "projects" / "llama.cpp" / "gguf-py"
if LLAMA_CPP_GGUF_PY.is_dir():
    sys.path.insert(0, str(LLAMA_CPP_GGUF_PY))
import gguf  # type: ignore[import-not-found]


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.astype(np.float32).tofile(path)


def write_i32(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().to(torch.int32).contiguous().cpu().numpy()
    arr.astype(np.int32).tofile(path)


def write_u8(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().to(torch.uint8).contiguous().cpu().numpy()
    arr.astype(np.uint8).tofile(path)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_f32_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).contiguous().cpu().numpy().astype(np.float32, copy=False)


def to_f16_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float16).contiguous().cpu().numpy().astype(np.float16, copy=False)


def to_i32_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.int32).contiguous().cpu().numpy().astype(np.int32, copy=False)


def to_i8_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.int8).contiguous().cpu().numpy().astype(np.int8, copy=False)


def force_eager_attention(model: OmniVoice) -> None:
    llm = model.llm
    llm.config._attn_implementation = "eager"
    if hasattr(model.config, "llm_config"):
        model.config.llm_config._attn_implementation = "eager"


def export_embed_inputs(model: OmniVoice, input_ids: torch.Tensor, audio_mask: torch.Tensor, out_dir: Path) -> dict:
    seq_len = int(input_ids.shape[-1])
    hidden = int(model.config.llm_config.hidden_size)
    input_ids_slice = input_ids[0].contiguous()
    audio_mask_slice = audio_mask[0].contiguous()

    text_ids = input_ids_slice[0]
    text_unique_ids, _ = torch.unique(text_ids, sorted=True, return_inverse=True)
    text_embed_table = model.get_input_embeddings().weight.detach()[text_unique_ids].to(torch.float32)

    audio_unique_ids = torch.arange(
        model.audio_embeddings.weight.shape[0], dtype=torch.int64, device=input_ids_slice.device
    )
    audio_embed_table = model.audio_embeddings.weight.detach().to(torch.float32)

    hs = model._prepare_embed_inputs(input_ids, audio_mask)[0].to(torch.float32)
    hs_b = hs.unsqueeze(0).to(model.llm.layers[0].self_attn.q_proj.weight.dtype)
    position_ids = torch.arange(seq_len, device=hs_b.device).unsqueeze(0)
    cos, sin = model.llm.rotary_emb(hs_b, position_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_i32(out_dir / "input_ids_i32.bin", input_ids_slice.to(torch.int32))
    write_u8(out_dir / "audio_mask_u8.bin", audio_mask_slice.to(torch.uint8))
    write_i32(out_dir / "codebook_offsets_i32.bin", model.codebook_layer_offsets.to(torch.int32))
    write_i32(out_dir / "text_unique_ids_i32.bin", text_unique_ids.to(torch.int32))
    write_f32(out_dir / "text_embed_table_f32.bin", text_embed_table)
    write_i32(out_dir / "audio_unique_shifted_ids_i32.bin", audio_unique_ids.to(torch.int32))
    write_f32(out_dir / "audio_embed_table_f32.bin", audio_embed_table)
    write_f32(out_dir / "x_input_ref_f32.bin", hs)
    write_f32(out_dir / "rope_cos_f32.bin", cos.squeeze(0))
    write_f32(out_dir / "rope_sin_f32.bin", sin.squeeze(0))

    meta = {
        "seq_len": seq_len,
        "hidden_size": hidden,
        "text_unique_count": int(text_unique_ids.numel()),
        "audio_unique_count": int(audio_unique_ids.numel()),
        "audio_table_mode": "full",
    }
    (out_dir / "meta.txt").write_text("\n".join(f"{k}={v}" for k, v in meta.items()) + "\n", encoding="utf-8")
    return meta


def export_full_backbone_weights(model: OmniVoice, export_dir: Path) -> dict:
    llm = model.llm
    layers_dir = export_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    for il, layer in enumerate(llm.layers):
        layer_dir = layers_dir / f"layer_{il:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        write_f32(layer_dir / "attn_norm_weight_f32.bin", layer.input_layernorm.weight)
        write_f32(layer_dir / "q_proj_weight_f32.bin", layer.self_attn.q_proj.weight)
        write_f32(layer_dir / "k_proj_weight_f32.bin", layer.self_attn.k_proj.weight)
        write_f32(layer_dir / "v_proj_weight_f32.bin", layer.self_attn.v_proj.weight)
        write_f32(layer_dir / "q_norm_weight_f32.bin", layer.self_attn.q_norm.weight)
        write_f32(layer_dir / "k_norm_weight_f32.bin", layer.self_attn.k_norm.weight)
        write_f32(layer_dir / "o_proj_weight_f32.bin", layer.self_attn.o_proj.weight)
        write_f32(layer_dir / "post_attention_norm_weight_f32.bin", layer.post_attention_layernorm.weight)
        write_f32(layer_dir / "gate_proj_weight_f32.bin", layer.mlp.gate_proj.weight)
        write_f32(layer_dir / "up_proj_weight_f32.bin", layer.mlp.up_proj.weight)
        write_f32(layer_dir / "down_proj_weight_f32.bin", layer.mlp.down_proj.weight)

    write_f32(export_dir / "output_norm_weight_f32.bin", llm.norm.weight)
    write_f32(export_dir / "audio_heads_weight_f32.bin", model.audio_heads.weight)
    payload = {
        "n_layer": int(llm.config.num_hidden_layers),
        "hidden_size": int(llm.config.hidden_size),
        "head_dim": int(llm.config.head_dim),
        "n_head": int(llm.config.num_attention_heads),
        "n_kv_head": int(llm.config.num_key_value_heads),
        "max_position_embeddings": int(llm.config.max_position_embeddings),
        "rope_freq_base": float((llm.config.rope_scaling or {}).get("rope_theta", 10000.0)),
        "intermediate_size": int(llm.layers[0].mlp.gate_proj.weight.shape[0]),
        "audio_vocab_size": int(model.config.audio_vocab_size),
        "num_audio_codebook": int(model.config.num_audio_codebook),
    }
    save_json(export_dir / "backbone_manifest.json", payload)
    (export_dir / "backbone_meta.txt").write_text("\n".join(f"{k}={v}" for k, v in payload.items()) + "\n", encoding="utf-8")
    return payload


def export_iterative_setup(model: OmniVoice, text: str, output_dir: Path, gen_config: OmniVoiceGenerationConfig) -> dict:
    full_task = model._preprocess_all(text=text, language=None, ref_text=None, ref_audio=None, voice_clone_prompt=None, instruct=None, preprocess_prompt=gen_config.preprocess_prompt, speed=None, duration=None)
    inputs = model._prepare_inference_inputs(full_task.texts[0], full_task.target_lens[0], full_task.ref_texts[0], full_task.ref_audio_tokens[0], full_task.langs[0], full_task.instructs[0], gen_config.denoise)
    target_len = int(full_task.target_lens[0])
    total_mask = target_len * int(model.config.num_audio_codebook)
    timesteps = model.__class__.__dict__["_generate_iterative"].__globals__["_get_time_steps"](t_start=0.0, t_end=1.0, num_step=gen_config.num_step + 1, t_shift=gen_config.t_shift).tolist()
    rem = total_mask
    schedule = []
    for step in range(gen_config.num_step):
        num = rem if step == gen_config.num_step - 1 else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)

    out_dir = output_dir / "iterative_decode"
    out_dir.mkdir(parents=True, exist_ok=True)
    embed_meta = export_embed_inputs(model, inputs["input_ids"], inputs["audio_mask"], out_dir / "cond")

    x_embed = model._prepare_embed_inputs(inputs["input_ids"], inputs["audio_mask"])
    layer0 = model.llm.layers[0]
    layer0_in = x_embed.to(layer0.self_attn.q_proj.weight.dtype)
    attn_input_t = layer0.input_layernorm(layer0_in)
    attn_input = attn_input_t.detach().to(torch.float32)[0]
    q_proj = layer0.self_attn.q_proj(attn_input_t).detach().to(torch.float32)[0]
    k_proj = layer0.self_attn.k_proj(attn_input_t).detach().to(torch.float32)[0]
    v_proj = layer0.self_attn.v_proj(attn_input_t).detach().to(torch.float32)[0]
    seq_len = attn_input.shape[0]
    head_dim = int(model.llm.config.head_dim)
    n_head = int(model.llm.config.num_attention_heads)
    n_kv_head = int(model.llm.config.num_key_value_heads)
    q_norm = layer0.self_attn.q_norm(q_proj.view(seq_len, n_head, head_dim)).detach().to(torch.float32)
    k_norm = layer0.self_attn.k_norm(k_proj.view(seq_len, n_kv_head, head_dim)).detach().to(torch.float32)
    v_heads = v_proj.view(seq_len, n_kv_head, head_dim).detach().to(torch.float32)
    position_ids = torch.arange(seq_len, device=attn_input_t.device).unsqueeze(0)
    rope_dummy = torch.empty((1, seq_len, model.llm.config.hidden_size), dtype=attn_input_t.dtype, device=attn_input_t.device)
    cos, sin = model.llm.rotary_emb(rope_dummy, position_ids)
    q_rope, k_rope = apply_rotary_pos_emb(
        q_norm.unsqueeze(0).transpose(1, 2),
        k_norm.unsqueeze(0).transpose(1, 2),
        cos,
        sin,
    )
    k_rep = k_rope.repeat_interleave(layer0.self_attn.num_key_value_groups, dim=1)
    v_rep = v_heads.unsqueeze(0).transpose(1, 2).repeat_interleave(layer0.self_attn.num_key_value_groups, dim=1)
    attn_scores = torch.matmul(q_rope, k_rep.transpose(-2, -1)) * layer0.self_attn.scaling
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_probs, v_rep)
    attn_out_merge = attn_out.transpose(1, 2).contiguous().view(seq_len, n_head * head_dim)
    o_proj_ref = layer0.self_attn.o_proj(attn_out_merge.to(layer0.self_attn.o_proj.weight.dtype)).detach().to(torch.float32)
    attn_residual = (x_embed.detach().to(torch.float32)[0] + o_proj_ref).contiguous()
    post_attn_input_t = attn_residual.unsqueeze(0).to(layer0.mlp.gate_proj.weight.dtype)
    post_attn_norm = layer0.post_attention_layernorm(post_attn_input_t).detach().to(torch.float32)[0]
    post_attn_norm_t = post_attn_norm.unsqueeze(0).to(layer0.mlp.gate_proj.weight.dtype)
    gate_proj_ref = layer0.mlp.gate_proj(post_attn_norm_t).detach().to(torch.float32)[0]
    up_proj_ref = layer0.mlp.up_proj(post_attn_norm_t).detach().to(torch.float32)[0]
    gate_silu_ref = torch.nn.functional.silu(gate_proj_ref).contiguous()
    mlp_act_ref = (gate_silu_ref * up_proj_ref).contiguous()
    mlp_down_ref = layer0.mlp.down_proj(mlp_act_ref.to(layer0.mlp.down_proj.weight.dtype)).detach().to(torch.float32)
    layer0_out_ref = (attn_residual + mlp_down_ref).contiguous()
    layer0_ref_dir = out_dir / "cond" / "layer0_refs"
    layer0_ref_dir.mkdir(parents=True, exist_ok=True)
    write_f32(layer0_ref_dir / "attn_input_ref_f32.bin", attn_input)
    write_f32(layer0_ref_dir / "q_proj_ref_f32.bin", q_proj)
    write_f32(layer0_ref_dir / "k_proj_ref_f32.bin", k_proj)
    write_f32(layer0_ref_dir / "v_proj_ref_f32.bin", v_proj)
    write_f32(layer0_ref_dir / "q_norm_ref_f32.bin", q_norm.contiguous())
    write_f32(layer0_ref_dir / "k_norm_ref_f32.bin", k_norm.contiguous())
    write_f32(layer0_ref_dir / "q_rope_ref_f32.bin", q_rope.squeeze(0).permute(1, 0, 2).contiguous())
    write_f32(layer0_ref_dir / "k_rope_ref_f32.bin", k_rope.squeeze(0).permute(1, 0, 2).contiguous())
    write_f32(layer0_ref_dir / "attn_out_ref_f32.bin", attn_out_merge.contiguous())
    write_f32(layer0_ref_dir / "o_proj_ref_f32.bin", o_proj_ref.contiguous())
    write_f32(layer0_ref_dir / "attn_residual_ref_f32.bin", attn_residual)
    write_f32(layer0_ref_dir / "post_attn_norm_ref_f32.bin", post_attn_norm)
    write_f32(layer0_ref_dir / "gate_proj_ref_f32.bin", gate_proj_ref)
    write_f32(layer0_ref_dir / "up_proj_ref_f32.bin", up_proj_ref)
    write_f32(layer0_ref_dir / "gate_silu_ref_f32.bin", gate_silu_ref)
    write_f32(layer0_ref_dir / "mlp_act_ref_f32.bin", mlp_act_ref)
    write_f32(layer0_ref_dir / "mlp_down_ref_f32.bin", mlp_down_ref)
    write_f32(layer0_ref_dir / "layer0_out_ref_f32.bin", layer0_out_ref)
    (layer0_ref_dir / "meta.txt").write_text(
        "\n".join([
            f"seq_len={attn_input.shape[0]}",
            f"hidden_size={attn_input.shape[1]}",
            f"q_out={q_proj.shape[1]}",
            f"k_out={k_proj.shape[1]}",
            f"v_out={v_proj.shape[1]}",
            f"head_dim={head_dim}",
            f"n_head={n_head}",
            f"n_kv_head={n_kv_head}",
        ]) + "\n",
        encoding="utf-8",
    )

    backbone_ref_dir = out_dir / "cond" / "backbone_refs"
    backbone_ref_dir.mkdir(parents=True, exist_ok=True)
    attn_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=attn_input_t.device)
    x_backbone = x_embed.to(layer0.self_attn.q_proj.weight.dtype)
    probe_layers = {19, 20, 21, 22, 23, 24, 25, 26, 27}
    for il, layer in enumerate(model.llm.layers):
        if il in probe_layers:
            probe_dir = backbone_ref_dir / f"layer_{il:02d}_probe"
            probe_dir.mkdir(parents=True, exist_ok=True)
            layer_input = x_backbone.detach().to(torch.float32)[0]
            layer_norm_in = layer.input_layernorm(x_backbone).detach().to(torch.float32)[0]
            q_proj = layer.self_attn.q_proj(layer.input_layernorm(x_backbone)).detach().to(torch.float32)[0]
            k_proj = layer.self_attn.k_proj(layer.input_layernorm(x_backbone)).detach().to(torch.float32)[0]
            v_proj = layer.self_attn.v_proj(layer.input_layernorm(x_backbone)).detach().to(torch.float32)[0]
            q_norm = layer.self_attn.q_norm(q_proj.view(seq_len, n_head, head_dim)).detach().to(torch.float32)
            k_norm = layer.self_attn.k_norm(k_proj.view(seq_len, n_kv_head, head_dim)).detach().to(torch.float32)
            v_heads = v_proj.view(seq_len, n_kv_head, head_dim).detach().to(torch.float32)
            q_rope, k_rope = apply_rotary_pos_emb(
                q_norm.unsqueeze(0).transpose(1, 2),
                k_norm.unsqueeze(0).transpose(1, 2),
                cos,
                sin,
            )
            k_rep = k_rope.repeat_interleave(layer.self_attn.num_key_value_groups, dim=1)
            v_rep = v_heads.unsqueeze(0).transpose(1, 2).repeat_interleave(layer.self_attn.num_key_value_groups, dim=1)
            attn_scores = torch.matmul(q_rope, k_rep.transpose(-2, -1)) * layer.self_attn.scaling
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_probs, v_rep)
            attn_out_merge = attn_out.transpose(1, 2).contiguous().view(seq_len, n_head * head_dim)
            o_proj_ref = layer.self_attn.o_proj(attn_out_merge.to(layer.self_attn.o_proj.weight.dtype)).detach().to(torch.float32)
            attn_residual = (layer_input + o_proj_ref).contiguous()
            post_attn_norm = layer.post_attention_layernorm(attn_residual.unsqueeze(0).to(layer.mlp.gate_proj.weight.dtype)).detach().to(torch.float32)[0]
            post_attn_norm_t = post_attn_norm.unsqueeze(0).to(layer.mlp.gate_proj.weight.dtype)
            gate_proj_ref = layer.mlp.gate_proj(post_attn_norm_t).detach().to(torch.float32)[0]
            up_proj_ref = layer.mlp.up_proj(post_attn_norm_t).detach().to(torch.float32)[0]
            gate_silu_ref = torch.nn.functional.silu(gate_proj_ref).contiguous()
            mlp_act_ref = (gate_silu_ref * up_proj_ref).contiguous()
            mlp_down_ref = layer.mlp.down_proj(mlp_act_ref.to(layer.mlp.down_proj.weight.dtype)).detach().to(torch.float32)
            layer_out_ref = (attn_residual + mlp_down_ref).contiguous()

            write_f32(probe_dir / "layer_input_ref_f32.bin", layer_input)
            write_f32(probe_dir / "attn_input_ref_f32.bin", layer_norm_in)
            write_f32(probe_dir / "q_proj_ref_f32.bin", q_proj)
            write_f32(probe_dir / "k_proj_ref_f32.bin", k_proj)
            write_f32(probe_dir / "v_proj_ref_f32.bin", v_proj)
            write_f32(probe_dir / "q_norm_ref_f32.bin", q_norm.contiguous())
            write_f32(probe_dir / "k_norm_ref_f32.bin", k_norm.contiguous())
            write_f32(probe_dir / "q_rope_ref_f32.bin", q_rope.squeeze(0).permute(1, 0, 2).contiguous())
            write_f32(probe_dir / "k_rope_ref_f32.bin", k_rope.squeeze(0).permute(1, 0, 2).contiguous())
            write_f32(probe_dir / "attn_out_ref_f32.bin", attn_out_merge.contiguous())
            write_f32(probe_dir / "o_proj_ref_f32.bin", o_proj_ref.contiguous())
            write_f32(probe_dir / "attn_residual_ref_f32.bin", attn_residual)
            write_f32(probe_dir / "post_attn_norm_ref_f32.bin", post_attn_norm)
            write_f32(probe_dir / "gate_proj_ref_f32.bin", gate_proj_ref)
            write_f32(probe_dir / "up_proj_ref_f32.bin", up_proj_ref)
            write_f32(probe_dir / "gate_silu_ref_f32.bin", gate_silu_ref)
            write_f32(probe_dir / "mlp_act_ref_f32.bin", mlp_act_ref)
            write_f32(probe_dir / "mlp_down_ref_f32.bin", mlp_down_ref)
            write_f32(probe_dir / "layer_out_ref_f32.bin", layer_out_ref)

        x_backbone = layer(
            x_backbone,
            attention_mask=attn_mask,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
        write_f32(backbone_ref_dir / f"layer_{il:02d}_out_ref_f32.bin", x_backbone.detach().to(torch.float32)[0])
    final_hidden = model.llm.norm(x_backbone).detach().to(torch.float32)[0]
    logits_ref = model.audio_heads(final_hidden.to(model.audio_heads.weight.dtype)).detach().to(torch.float32)
    write_f32(backbone_ref_dir / "final_hidden_ref_f32.bin", final_hidden)
    write_f32(backbone_ref_dir / "logits_ref_f32.bin", logits_ref)
    (backbone_ref_dir / "meta.txt").write_text(
        "\n".join([
            f"seq_len={seq_len}",
            f"hidden_size={model.llm.config.hidden_size}",
            f"n_layer={model.llm.config.num_hidden_layers}",
            f"audio_vocab_total={model.config.audio_vocab_size * model.config.num_audio_codebook}",
        ]) + "\n",
        encoding="utf-8",
    )

    payload = {
        "target_len": target_len,
        "cond_seq_len": int(inputs["input_ids"].shape[-1]),
        "num_codebook": int(model.config.num_audio_codebook),
        "audio_vocab_size": int(model.config.audio_vocab_size),
        "schedule": schedule,
        "num_step": int(gen_config.num_step),
        "guidance_scale": float(gen_config.guidance_scale),
        "layer_penalty_factor": float(gen_config.layer_penalty_factor),
        "position_temperature": float(gen_config.position_temperature),
        "class_temperature": float(gen_config.class_temperature),
        "ref_rms": float(full_task.ref_rms[0]) if full_task.ref_rms[0] is not None else 0.1,
        "sampling_rate": int(model.sampling_rate),
        "cond": embed_meta,
    }
    save_json(out_dir / "manifest.json", payload)
    (out_dir / "meta.txt").write_text(
        "\n".join([
            f"target_len={payload['target_len']}",
            f"cond_seq_len={payload['cond_seq_len']}",
            f"num_codebook={payload['num_codebook']}",
            f"audio_vocab_size={payload['audio_vocab_size']}",
            f"num_step={payload['num_step']}",
            f"guidance_scale={payload['guidance_scale']}",
            f"layer_penalty_factor={payload['layer_penalty_factor']}",
            f"position_temperature={payload['position_temperature']}",
            f"class_temperature={payload['class_temperature']}",
            f"ref_rms={payload['ref_rms']}",
            f"sampling_rate={payload['sampling_rate']}",
            f"schedule={','.join(str(v) for v in schedule)}",
        ]) + "\n",
        encoding="utf-8",
    )
    return payload


def export_audio_tokenizer_weights(model: OmniVoice, raw_tokens: torch.Tensor, output_dir: Path) -> dict:
    tokenizer = model.audio_tokenizer
    out_dir = output_dir / "audio_tokenizer_decode"
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    write_i32(out_dir / "tokens_i32.bin", raw_tokens.to(torch.int32).cpu())
    manifest = []
    def export_tensor(name: str, tensor: torch.Tensor):
        safe = name.replace('.', '__')
        path = weights_dir / f"{safe}.f32.bin"
        write_f32(path, tensor)
        manifest.append({"name": name, "file": path.name, "shape": [int(x) for x in tensor.shape], "dtype": "float32"})
    for i, q in enumerate(tokenizer.quantizer.quantizers):
        export_tensor(f"quantizer.quantizers.{i}.codebook.embed", q.codebook.embed)
        export_tensor(f"quantizer.quantizers.{i}.project_out.weight", q.project_out.weight)
        export_tensor(f"quantizer.quantizers.{i}.project_out.bias", q.project_out.bias)
    for name, tensor in tokenizer.fc2.state_dict().items():
        export_tensor(f"fc2.{name}", tensor)
    for name, tensor in tokenizer.acoustic_decoder.state_dict().items():
        export_tensor(f"acoustic_decoder.{name}", tensor)
    payload = {
        "num_codebooks": int(raw_tokens.shape[0]),
        "seq_len": int(raw_tokens.shape[1]),
        "tokenizer_sample_rate": int(tokenizer.config.sample_rate),
        "hop_length": int(tokenizer.config.acoustic_model_config.hop_length),
        "quantizer_count": int(tokenizer.quantizer.num_quantizers),
        "codebook_size": int(tokenizer.quantizer.codebook_size),
        "weights": manifest,
    }
    save_json(out_dir / "manifest.json", payload)
    (out_dir / "meta.txt").write_text("\n".join([
        f"num_codebooks={payload['num_codebooks']}",
        f"seq_len={payload['seq_len']}",
        f"tokenizer_sample_rate={payload['tokenizer_sample_rate']}",
        f"hop_length={payload['hop_length']}",
        f"quantizer_count={payload['quantizer_count']}",
        f"codebook_size={payload['codebook_size']}",
    ]) + "\n", encoding="utf-8")
    return payload


def write_runtime_gguf_direct(
    model: OmniVoice,
    text: str,
    out_dir: Path,
    raw_tokens: torch.Tensor,
    iterative_manifest: dict,
    tokenizer_manifest: dict,
) -> Path:
    writer = gguf.GGUFWriter(path=None, arch="ch13-runtime")
    writer.add_name("ch13-vulkan-omnivoice-runtime")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    llm = model.llm
    cfg = llm.config
    rope_freq_base = float((cfg.rope_scaling or {}).get("rope_theta", 10000.0))
    writer.add_uint32("runtime.hidden_size", int(cfg.hidden_size))
    writer.add_uint32("runtime.n_head", int(cfg.num_attention_heads))
    writer.add_uint32("runtime.n_kv_head", int(cfg.num_key_value_heads))
    writer.add_uint32("runtime.head_dim", int(cfg.head_dim))
    writer.add_uint32("runtime.q_out", int(llm.layers[0].self_attn.q_proj.weight.shape[0]))
    writer.add_uint32("runtime.k_out", int(llm.layers[0].self_attn.k_proj.weight.shape[0]))
    writer.add_uint32("runtime.v_out", int(llm.layers[0].self_attn.v_proj.weight.shape[0]))
    writer.add_uint32("runtime.n_layer", int(cfg.num_hidden_layers))
    writer.add_uint32("runtime.intermediate_size", int(cfg.intermediate_size))
    writer.add_uint32("runtime.audio_vocab_size", int(model.config.audio_vocab_size))
    writer.add_uint32("runtime.num_audio_codebook", int(model.config.num_audio_codebook))
    writer.add_uint32("runtime.cond_seq_len", int(iterative_manifest["cond_seq_len"]))
    writer.add_uint32("runtime.target_len", int(iterative_manifest["target_len"]))
    writer.add_uint32("runtime.num_step", int(iterative_manifest["num_step"]))
    writer.add_uint32("runtime.tokenizer_sample_rate", int(tokenizer_manifest["tokenizer_sample_rate"]))
    writer.add_uint32("runtime.quantizer_count", int(tokenizer_manifest["quantizer_count"]))
    writer.add_uint32("runtime.codebook_size", int(tokenizer_manifest["codebook_size"]))
    writer.add_uint32("runtime.rope_n_ctx_orig", int(cfg.max_position_embeddings))
    writer.add_float32("runtime.rope_freq_base", rope_freq_base)
    writer.add_float32("runtime.rms_norm_eps", float(cfg.rms_norm_eps))
    writer.add_array("runtime.schedule", [int(v) for v in iterative_manifest["schedule"]])

    inputs = model._prepare_inference_inputs(
        iterative_manifest.get("text", text) if "text" in iterative_manifest else text,
        iterative_manifest["target_len"],
        None, None, None, None,
        denoise=True,
    )
    cond_input_ids = inputs["input_ids"]
    cond_audio_mask = inputs["audio_mask"]
    cond_seq_len = int(cond_input_ids.shape[-1])
    hidden = int(cfg.hidden_size)
    x_input_ref = model._prepare_embed_inputs(cond_input_ids, cond_audio_mask)[0].to(torch.float32)

    input_ids_slice = cond_input_ids[0].contiguous()
    audio_mask_slice = cond_audio_mask[0].contiguous()
    text_ids = input_ids_slice[0]
    text_unique_ids, _ = torch.unique(text_ids, sorted=True, return_inverse=True)
    text_embed_table = model.get_input_embeddings().weight.detach()[text_unique_ids].to(torch.float32)
    audio_unique_ids = torch.arange(model.audio_embeddings.weight.shape[0], dtype=torch.int64, device=input_ids_slice.device)
    audio_embed_table = model.audio_embeddings.weight.detach().to(torch.float32)

    writer.add_tensor("runtime.output_norm.weight", to_f32_np(llm.norm.weight))
    writer.add_tensor("runtime.audio_heads.weight", to_f32_np(model.audio_heads.weight))
    writer.add_tensor("runtime.iterative.cond.input_ids", to_i32_np(input_ids_slice))
    writer.add_tensor("runtime.iterative.cond.audio_mask", to_i8_np(audio_mask_slice.to(torch.int8)))
    writer.add_tensor("runtime.iterative.cond.codebook_offsets", to_i32_np(model.codebook_layer_offsets))
    writer.add_tensor("runtime.iterative.cond.text_unique_ids", to_i32_np(text_unique_ids))
    writer.add_tensor("runtime.iterative.cond.text_embed_table", to_f32_np(text_embed_table))
    writer.add_tensor("runtime.iterative.cond.audio_unique_ids", to_i32_np(audio_unique_ids))
    writer.add_tensor("runtime.iterative.cond.audio_embed_table", to_f32_np(audio_embed_table))
    writer.add_tensor("runtime.iterative.cond.x_input_ref", to_f32_np(x_input_ref))

    hs_b = x_input_ref.unsqueeze(0).to(llm.layers[0].self_attn.q_proj.weight.dtype)
    position_ids = torch.arange(cond_seq_len, device=hs_b.device).unsqueeze(0)
    cos, sin = llm.rotary_emb(hs_b, position_ids)
    writer.add_tensor("runtime.iterative.cond.rope_cos", to_f32_np(cos.squeeze(0)))
    writer.add_tensor("runtime.iterative.cond.rope_sin", to_f32_np(sin.squeeze(0)))

    for il, layer in enumerate(llm.layers):
        prefix = f"runtime.layers.{il:02d}."
        writer.add_tensor(prefix + "attn_norm.weight", to_f32_np(layer.input_layernorm.weight))
        writer.add_tensor(prefix + "q_proj.weight", to_f16_np(layer.self_attn.q_proj.weight))
        writer.add_tensor(prefix + "k_proj.weight", to_f16_np(layer.self_attn.k_proj.weight))
        writer.add_tensor(prefix + "v_proj.weight", to_f16_np(layer.self_attn.v_proj.weight))
        writer.add_tensor(prefix + "q_norm.weight", to_f32_np(layer.self_attn.q_norm.weight))
        writer.add_tensor(prefix + "k_norm.weight", to_f32_np(layer.self_attn.k_norm.weight))
        writer.add_tensor(prefix + "o_proj.weight", to_f16_np(layer.self_attn.o_proj.weight))
        writer.add_tensor(prefix + "post_attention_norm.weight", to_f32_np(layer.post_attention_layernorm.weight))
        writer.add_tensor(prefix + "gate_proj.weight", to_f16_np(layer.mlp.gate_proj.weight))
        writer.add_tensor(prefix + "up_proj.weight", to_f16_np(layer.mlp.up_proj.weight))
        writer.add_tensor(prefix + "down_proj.weight", to_f16_np(layer.mlp.down_proj.weight))

    writer.add_tensor("runtime.audio.tokens", to_i32_np(raw_tokens))
    tokenizer = model.audio_tokenizer
    for i, q in enumerate(tokenizer.quantizer.quantizers):
        writer.add_tensor(f"runtime.audio.quantizer.quantizers.{i}.codebook.embed", to_f32_np(q.codebook.embed))
        writer.add_tensor(f"runtime.audio.quantizer.quantizers.{i}.project_out.weight", to_f32_np(q.project_out.weight))
        writer.add_tensor(f"runtime.audio.quantizer.quantizers.{i}.project_out.bias", to_f32_np(q.project_out.bias))
    writer.add_tensor("runtime.audio.weights.fc2__weight.f32.bin", to_f32_np(tokenizer.fc2.weight))
    writer.add_tensor("runtime.audio.weights.fc2__bias.f32.bin", to_f32_np(tokenizer.fc2.bias))
    writer.add_tensor("runtime.audio.weights.acoustic_decoder__conv1__weight.f32.bin", to_f32_np(tokenizer.acoustic_decoder.conv1.weight))
    writer.add_tensor("runtime.audio.weights.acoustic_decoder__conv1__bias.f32.bin", to_f32_np(tokenizer.acoustic_decoder.conv1.bias))
    writer.add_tensor("runtime.audio.weights.acoustic_decoder__snake1__alpha.f32.bin", to_f32_np(tokenizer.acoustic_decoder.snake1.alpha))
    writer.add_tensor("runtime.audio.weights.acoustic_decoder__conv2__weight.f32.bin", to_f32_np(tokenizer.acoustic_decoder.conv2.weight))
    writer.add_tensor("runtime.audio.weights.acoustic_decoder__conv2__bias.f32.bin", to_f32_np(tokenizer.acoustic_decoder.conv2.bias))

    block_entries = [
        ("snake1.alpha", "snake1__alpha"),
        ("conv_t1.weight", "conv_t1__weight"),
        ("conv_t1.bias", "conv_t1__bias"),
        ("res_unit1.snake1.alpha", "res_unit1__snake1__alpha"),
        ("res_unit1.conv1.weight", "res_unit1__conv1__weight"),
        ("res_unit1.conv1.bias", "res_unit1__conv1__bias"),
        ("res_unit1.snake2.alpha", "res_unit1__snake2__alpha"),
        ("res_unit1.conv2.weight", "res_unit1__conv2__weight"),
        ("res_unit1.conv2.bias", "res_unit1__conv2__bias"),
        ("res_unit2.snake1.alpha", "res_unit2__snake1__alpha"),
        ("res_unit2.conv1.weight", "res_unit2__conv1__weight"),
        ("res_unit2.conv1.bias", "res_unit2__conv1__bias"),
        ("res_unit2.snake2.alpha", "res_unit2__snake2__alpha"),
        ("res_unit2.conv2.weight", "res_unit2__conv2__weight"),
        ("res_unit2.conv2.bias", "res_unit2__conv2__bias"),
        ("res_unit3.snake1.alpha", "res_unit3__snake1__alpha"),
        ("res_unit3.conv1.weight", "res_unit3__conv1__weight"),
        ("res_unit3.conv1.bias", "res_unit3__conv1__bias"),
        ("res_unit3.snake2.alpha", "res_unit3__snake2__alpha"),
        ("res_unit3.conv2.weight", "res_unit3__conv2__weight"),
        ("res_unit3.conv2.bias", "res_unit3__conv2__bias"),
    ]
    for bi, block in enumerate(tokenizer.acoustic_decoder.block):
        for path_name, file_name in block_entries:
            cur = block
            for part in path_name.split("."):
                cur = getattr(cur, part)
            writer.add_tensor(
                f"runtime.audio.acoustic_decoder.block.{bi}.{path_name}",
                to_f32_np(cur),
            )

    gguf_path = out_dir / "ch13-runtime.gguf"
    writer.write_header_to_file(path=gguf_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return gguf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--model', default='k2-fsa/OmniVoice')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seq-len', type=int, default=16)
    ap.add_argument('--write-gguf-direct', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    export_dir = out_dir / 'export'
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model
    snap = Path.home() / '.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/d39ac7fc8434dd452494b5061090af007d2a3ec0'
    if model_id == 'k2-fsa/OmniVoice' and snap.exists():
        model_id = str(snap)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    model = OmniVoice.from_pretrained(model_id, device_map=args.device, dtype=torch.float16, local_files_only=True)
    force_eager_attention(model)
    gen_config = OmniVoiceGenerationConfig(position_temperature=0.0, class_temperature=0.0)

    audios = model.generate(text=args.text, generation_config=gen_config)
    waveform = audios[0].detach().cpu().squeeze(0).numpy().astype(np.float32)
    sf.write(out_dir / 'generated.wav', waveform, model.sampling_rate)

    full_task = model._preprocess_all(text=args.text, language=None, ref_text=None, ref_audio=None, voice_clone_prompt=None, instruct=None, preprocess_prompt=gen_config.preprocess_prompt, speed=None, duration=None)
    raw_tokens = model._generate_iterative(full_task, gen_config)[0].detach().to(torch.int32).cpu()
    write_i32(out_dir / 'generated_tokens_i32.bin', raw_tokens)
    ref_rms = float(full_task.ref_rms[0]) if full_task.ref_rms[0] is not None else 0.1
    (out_dir / 'generated_tokens_meta.txt').write_text('\n'.join([
        f'num_codebook={int(raw_tokens.shape[0])}',
        f'seq_len={int(raw_tokens.shape[1])}',
        f'sampling_rate={int(model.sampling_rate)}',
        f'ref_rms={ref_rms}',
    ]) + '\n', encoding='utf-8')

    embed_inputs = model._prepare_inference_inputs(full_task.texts[0], min(full_task.target_lens[0], 32), full_task.ref_texts[0], full_task.ref_audio_tokens[0], full_task.langs[0], full_task.instructs[0], denoise=True)
    x = model._prepare_embed_inputs(embed_inputs['input_ids'], embed_inputs['audio_mask'])[0].to(torch.float32)
    actual_seq_len = min(args.seq_len, int(x.shape[0]))
    slice_meta = export_embed_inputs(model, embed_inputs['input_ids'][:, :, :actual_seq_len], embed_inputs['audio_mask'][:, :actual_seq_len], export_dir / 'slice_input_min')
    backbone_manifest = export_full_backbone_weights(model, export_dir)
    manifest = {
        'text': args.text,
        'slice': 'runtime_minimal',
        'seq_len': actual_seq_len,
        'input_total_len': int(x.shape[0]),
        'hidden_size': int(x.shape[1]),
        'head_dim': int(model.llm.config.head_dim),
        'n_head': int(model.llm.config.num_attention_heads),
        'n_kv_head': int(model.llm.config.num_key_value_heads),
        'q_out': int(model.llm.layers[0].self_attn.q_proj.weight.shape[0]),
        'k_out': int(model.llm.layers[0].self_attn.k_proj.weight.shape[0]),
        'v_out': int(model.llm.layers[0].self_attn.v_proj.weight.shape[0]),
        'eps': float(model.llm.config.rms_norm_eps),
        'text_unique_count': int(slice_meta['text_unique_count']),
        'audio_unique_count': int(slice_meta['audio_unique_count']),
        'backbone': backbone_manifest,
    }
    save_json(export_dir / 'manifest.json', manifest)
    (export_dir / 'meta.txt').write_text('\n'.join([
        f"seq_len={manifest['seq_len']}",
        f"input_total_len={manifest['input_total_len']}",
        f"hidden_size={manifest['hidden_size']}",
        f"head_dim={manifest['head_dim']}",
        f"n_head={manifest['n_head']}",
        f"n_kv_head={manifest['n_kv_head']}",
        f"q_out={manifest['q_out']}",
        f"k_out={manifest['k_out']}",
        f"v_out={manifest['v_out']}",
        f"text_unique_count={manifest['text_unique_count']}",
        f"audio_unique_count={manifest['audio_unique_count']}",
        f"eps={manifest['eps']:.9f}",
    ]) + '\n', encoding='utf-8')

    iterative_manifest = export_iterative_setup(model, args.text, out_dir, gen_config)
    tokenizer_manifest = export_audio_tokenizer_weights(model, raw_tokens, out_dir)
    gguf_path = None
    if args.write_gguf_direct:
        gguf_path = write_runtime_gguf_direct(model, args.text, out_dir, raw_tokens, iterative_manifest, tokenizer_manifest)
    save_json(out_dir / 'inference.json', {
        'text': args.text,
        'model': args.model,
        'device': args.device,
        'sampling_rate': int(model.sampling_rate),
        'generated_wav': str(out_dir / 'generated.wav'),
        'generated_tokens': str(out_dir / 'generated_tokens_i32.bin'),
        'export_dir': str(export_dir),
        'export_manifest': manifest,
        'iterative_decode_manifest': iterative_manifest,
        'audio_tokenizer_decode_manifest': tokenizer_manifest,
        'direct_runtime_gguf': str(gguf_path) if gguf_path else None,
    })
    print(json.dumps({'ok': True, 'output_dir': str(out_dir), 'runtime_export': str(export_dir)}, indent=2))


if __name__ == '__main__':
    main()
