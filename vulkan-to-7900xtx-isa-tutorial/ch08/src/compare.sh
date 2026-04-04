#!/usr/bin/env bash
set -euo pipefail
cli_root="/home/taowen/projects/rgp-analyzer-cli"
chapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="${cli_root}/src" python3 -m rgp_analyzer_cli compare-shader-focus \
    "${chapter_dir}/captures/attention_naive.rgp" \
    "${chapter_dir}/captures/flash_attention.rgp" \
    --source-file "${chapter_dir}/src/shaders/flash_attention.comp"
