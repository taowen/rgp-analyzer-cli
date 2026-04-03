#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${ZIMAGE_STEPS:=4}"
: "${ZIMAGE_WIDTH:=512}"
: "${ZIMAGE_HEIGHT:=512}"
: "${ZIMAGE_REPEAT:=2}"

export ZIMAGE_STEPS
export ZIMAGE_WIDTH
export ZIMAGE_HEIGHT
export ZIMAGE_REPEAT

echo "focused capture config: steps=${ZIMAGE_STEPS} width=${ZIMAGE_WIDTH} height=${ZIMAGE_HEIGHT} repeat=${ZIMAGE_REPEAT}"
exec bash "${script_dir}/capture-rgp.sh" "$@"
