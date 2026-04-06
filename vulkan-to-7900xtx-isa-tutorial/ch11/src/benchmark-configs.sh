#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
whisper_dir="${repo_root}/third_party/whisper.cpp"
runs="${1:-5}"

bash "${script_dir}/build.sh" >/dev/null
bash "${script_dir}/download-model.sh" tiny.en >/dev/null

run_case() {
  local label="$1"
  shift
  local totals_file
  totals_file="$(mktemp)"
  trap 'rm -f "${totals_file}"' RETURN

  echo "=== ${label} ==="
  for i in $(seq 1 "${runs}"); do
    local total
    total="$(
      ./build-vk/bin/whisper-cli \
        -m ./models/ggml-tiny.en.bin \
        -f ./samples/jfk.wav \
        "$@" 2>&1 \
        | sed -n 's/.*total time = *\([0-9.]*\) ms.*/\1/p' \
        | tail -n 1
    )"
    printf 'run[%d]=%s ms\n' "${i}" "${total}"
    printf '%s\n' "${total}" >> "${totals_file}"
  done

  python3 - "${totals_file}" <<'PY'
import statistics
import sys
path = sys.argv[1]
vals = [float(x.strip()) for x in open(path) if x.strip()]
vals_sorted = sorted(vals)
print(f"median={statistics.median(vals):.2f} ms min={min(vals):.2f} ms max={max(vals):.2f} ms")
print(f"sorted={','.join(f'{v:.2f}' for v in vals_sorted)}")
PY
  echo
}

cd "${whisper_dir}"
run_case "baseline -t 4" -t 4
run_case "optimized -t 12 -bo 1 -bs 1 -nf -mc 16" -t 12 -bo 1 -bs 1 -nf -mc 16
