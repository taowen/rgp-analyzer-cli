#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${chapter_dir}/.." && pwd)"
cli_root="/home/taowen/projects/rgp-analyzer-cli"
capture="${chapter_dir}/captures/attention_naive.rgp"
source_file="${chapter_dir}/src/shaders/attention_naive.comp"

if [[ ! -f "${capture}" ]]; then
    echo "missing capture: ${capture}" >&2
    exit 1
fi

PYTHONPATH="${cli_root}/src" python3 -m rgp_analyzer_cli shader-focus \
    "${capture}" \
    --source-file "${source_file}"

echo
PYTHONPATH="${cli_root}/src" python3 -m rgp_analyzer_cli code-object-isa \
    "${capture}" \
    --source-file "${source_file}"
