#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
text="${1:-This is a short speech synthesis test running on an AMD Radeon graphics card.}"

bash "${script_dir}/run-omnivoice.sh" "${text}"
bash "${script_dir}/verify-asr.sh" "${text}"
