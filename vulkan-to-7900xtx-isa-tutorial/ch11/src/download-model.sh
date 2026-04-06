#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
whisper_dir="${repo_root}/third_party/whisper.cpp"
model_name="${1:-tiny.en}"

cd "${whisper_dir}"
./models/download-ggml-model.sh "${model_name}" ./models
