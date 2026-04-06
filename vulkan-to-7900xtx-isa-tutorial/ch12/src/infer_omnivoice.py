#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="en")
    parser.add_argument("--instruct", default="female, american accent")
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float16,
    )
    t1 = time.perf_counter()

    audios = model.generate(
        text=args.text,
        language=args.language,
        instruct=args.instruct,
        num_step=args.num_step,
        duration=args.duration,
        guidance_scale=args.guidance_scale,
    )
    t2 = time.perf_counter()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_np = audios[0].detach().cpu().squeeze(0).to(torch.float32).numpy()
    sf.write(str(output), np.asarray(audio_np), model.sampling_rate)

    payload = {
        "text": args.text,
        "model": args.model,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sampling_rate": model.sampling_rate,
        "load_ms": round((t1 - t0) * 1000.0, 2),
        "generate_ms": round((t2 - t1) * 1000.0, 2),
        "total_ms": round((t2 - t0) * 1000.0, 2),
        "output_wav": str(output),
    }
    Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
