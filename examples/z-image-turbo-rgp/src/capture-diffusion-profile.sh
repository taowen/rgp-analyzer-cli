#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prompt="${*:-a red robot reading a book in a library}"

env \
    ZIMAGE_PROFILE_MODE=rgp_profile \
    ZIMAGE_STOP_AFTER_PHASE=diffusion \
    ZIMAGE_STEPS="${ZIMAGE_STEPS:-2}" \
    ZIMAGE_WIDTH="${ZIMAGE_WIDTH:-256}" \
    ZIMAGE_HEIGHT="${ZIMAGE_HEIGHT:-256}" \
    ZIMAGE_REPEAT=1 \
    bash "${script_dir}/capture-rgp.sh" "${prompt}"
