#!/usr/bin/env bash
set -euo pipefail
cli_root="/home/taowen/projects/rgp-analyzer-cli"
chapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
capture="${chapter_dir}/captures/flash_attention.rgp"
source_file="${chapter_dir}/src/shaders/flash_attention.comp"
PYTHONPATH="${cli_root}/src" python3 -m rgp_analyzer_cli shader-focus "${capture}" --source-file "${source_file}"
echo
PYTHONPATH="${cli_root}/src" python3 -m rgp_analyzer_cli code-object-isa "${capture}" --source-file "${source_file}"
