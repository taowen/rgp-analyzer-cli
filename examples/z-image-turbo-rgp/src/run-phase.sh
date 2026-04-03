#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

phase="${1:-}"
if [[ -z "${phase}" ]]; then
    echo "usage: $0 <full|condition|diffusion> [prompt...]" >&2
    exit 1
fi
shift

normalized_phase="$(normalize_phase_name "${phase}")"
prompt="${*:-a red robot reading a book in a library}"

if [[ -n "${normalized_phase}" ]]; then
    env ZIMAGE_STOP_AFTER_PHASE="${normalized_phase}" bash "${script_dir}/run.sh" "${prompt}"
else
    bash "${script_dir}/run.sh" "${prompt}"
fi
