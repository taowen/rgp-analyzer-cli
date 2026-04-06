#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--omnivoice-snapshot",
        default=str(
            Path.home()
            / ".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/d39ac7fc8434dd452494b5061090af007d2a3ec0"
        ),
    )
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    snap = Path(args.omnivoice_snapshot)
    out = Path(args.output_dir)
    if not snap.is_dir():
        raise SystemExit(f"snapshot not found: {snap}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))["llm_config"]
    cfg["_name_or_path"] = str(out)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md"):
        src = snap / name
        if src.exists():
            shutil.copy2(src, out / name)

    tensors: dict[str, object] = {}
    with safe_open(snap / "model.safetensors", framework="pt", device="cpu") as f:
        llm_keys = [k for k in f.keys() if k.startswith("llm.")]
        for name in llm_keys:
            tensors[name[len("llm.") :]] = f.get_tensor(name).contiguous()

    save_file(tensors, str(out / "model.safetensors"))

    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(out),
                "tensor_count": len(tensors),
                "sample_tensors": list(tensors.keys())[:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
