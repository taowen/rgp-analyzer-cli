#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LLAMA_CPP_GGUF_PY = Path.home() / "projects" / "llama.cpp" / "gguf-py"
if LLAMA_CPP_GGUF_PY.is_dir():
    sys.path.insert(0, str(LLAMA_CPP_GGUF_PY))
import gguf  # type: ignore[import-not-found]


def stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    af = a.astype(np.float32, copy=False)
    bf = b.astype(np.float32, copy=False)
    d = np.abs(af - bf)
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "ref_abs_max": float(np.abs(bf).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-gguf", required=True)
    ap.add_argument("--official-gguf", required=True)
    args = ap.parse_args()

    rt = gguf.GGUFReader(args.runtime_gguf)
    off = gguf.GGUFReader(args.official_gguf)

    rt_map = {t.name: t for t in rt.tensors}
    off_map = {t.name: t for t in off.tensors}

    pairs = [
        ("runtime.output_norm.weight", "output_norm.weight"),
        ("runtime.layers.00.attn_norm.weight", "blk.0.attn_norm.weight"),
        ("runtime.layers.00.q_proj.weight", "blk.0.attn_q.weight"),
        ("runtime.layers.00.k_proj.weight", "blk.0.attn_k.weight"),
        ("runtime.layers.00.v_proj.weight", "blk.0.attn_v.weight"),
        ("runtime.layers.00.o_proj.weight", "blk.0.attn_output.weight"),
        ("runtime.layers.00.post_attention_norm.weight", "blk.0.ffn_norm.weight"),
        ("runtime.layers.00.gate_proj.weight", "blk.0.ffn_gate.weight"),
        ("runtime.layers.00.up_proj.weight", "blk.0.ffn_up.weight"),
        ("runtime.layers.00.down_proj.weight", "blk.0.ffn_down.weight"),
        ("runtime.layers.23.q_proj.weight", "blk.23.attn_q.weight"),
        ("runtime.layers.23.o_proj.weight", "blk.23.attn_output.weight"),
        ("runtime.layers.23.gate_proj.weight", "blk.23.ffn_gate.weight"),
        ("runtime.layers.23.up_proj.weight", "blk.23.ffn_up.weight"),
        ("runtime.layers.23.down_proj.weight", "blk.23.ffn_down.weight"),
    ]

    report: dict[str, object] = {
        "runtime_tensor_count": len(rt.tensors),
        "official_tensor_count": len(off.tensors),
        "pairs": {},
    }

    for rt_name, off_name in pairs:
        a = rt_map[rt_name].data
        b = off_map[off_name].data
        report["pairs"][f"{rt_name} :: {off_name}"] = {
            "runtime_shape": list(a.shape),
            "official_shape": list(b.shape),
            "runtime_dtype": str(a.dtype),
            "official_dtype": str(b.dtype),
            **stats(a, b),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
