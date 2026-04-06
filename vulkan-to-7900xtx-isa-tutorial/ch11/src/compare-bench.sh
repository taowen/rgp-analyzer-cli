#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
baseline="${chapter_dir}/captures/bench-w0.rgp"
candidate="${chapter_dir}/captures/bench-w2.rgp"

if [[ ! -f "${baseline}" || ! -f "${candidate}" ]]; then
  echo "missing captures: ${baseline} or ${candidate}" >&2
  exit 1
fi

cd "${repo_root}"
PYTHONPATH=src python3 -m rgp_analyzer_cli compare-shader-focus "${baseline}" "${candidate}"
