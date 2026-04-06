#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--asr-file", required=True)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    asr_text = Path(args.asr_file).read_text(encoding="utf-8").strip()
    target_norm = normalize(args.target)
    asr_norm = normalize(asr_text)

    target_tokens = target_norm.split()
    asr_tokens = asr_norm.split()
    matched_prefix = 0
    for t, a in zip(target_tokens, asr_tokens):
        if t != a:
            break
        matched_prefix += 1

    payload = {
        "target_text": args.target,
        "asr_text": asr_text,
        "target_normalized": target_norm,
        "asr_normalized": asr_norm,
        "target_token_count": len(target_tokens),
        "asr_token_count": len(asr_tokens),
        "matched_prefix_tokens": matched_prefix,
        "exact_match": target_norm == asr_norm,
    }
    Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
