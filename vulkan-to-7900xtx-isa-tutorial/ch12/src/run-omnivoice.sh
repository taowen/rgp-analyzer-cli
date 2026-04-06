#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
venv_dir="${chapter_dir}/.venv"
output_dir="${chapter_dir}/output"

text="${1:-This is a short speech synthesis test running on an AMD Radeon graphics card.}"

mkdir -p "${output_dir}"
source "${venv_dir}/bin/activate"

python "${script_dir}/infer_omnivoice.py" \
  --text "${text}" \
  --output "${output_dir}/generated.wav" \
  --json-output "${output_dir}/inference.json"
