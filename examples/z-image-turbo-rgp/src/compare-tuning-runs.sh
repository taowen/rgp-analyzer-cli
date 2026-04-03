#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 4 ]]; then
  echo "usage: $(basename "$0") <baseline-perf.json> <baseline.rgp> <candidate-perf.json> <candidate.rgp>" >&2
  exit 2
fi

python3 "${script_dir}/compare-tuning-runs.py" "$@"
