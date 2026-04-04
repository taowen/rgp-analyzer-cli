# Z-Image-Turbo RGP Example

This example drives `rgp-analyzer-cli` with a real Vulkan text-to-image workload instead of a microbenchmark.

It uses:

- a vendored copy of `amd-vulkan-tutorial/ch02/src`
- local model weights under `examples/z-image-turbo-rgp/models/zimage`
- optional weight sync from a mounted Windows `C:\\games`
- Linux `.rgp` capture with `RADV`
- `rgp-analyzer-cli` for stitch, decode, dispatch ISA, and runtime summaries

Current role in the repo:

- real-workload profiling target
- proof that `rgp-analyzer-cli` can guide tuning on a non-toy Vulkan compute stack
- home of the `ggml perf -> diffusion-only rgp_profile -> compare` workflow

Current release scope for this example:

- real diffusion-phase capture on Linux + RADV
- focused shader compare against checked-in `.rgp` baselines
- report-oriented metric output via `metrics-doc`

## Files

```text
examples/z-image-turbo-rgp/
  README.md
  app/
    ... vendored direct-txt2img source tree ...
  models/
    zimage/
  src/
    common.sh
    mount-windows-games.sh
    sync-weights.sh
    build.sh
    run.sh
    run-perf.sh
    run-phase-perf.sh
    probe-runtime-knobs.sh
    perf-then-capture.sh
    capture-rgp.sh
    analyze-latest.sh
  captures/
```

## Requirements

- Linux
- AMD GPU with Vulkan
- `RADV` for `MESA_VK_TRACE=rgp`
- model weights already present either:
  - in `examples/z-image-turbo-rgp/models/zimage`
  - or on a mounted Windows drive under `C:\\games`

## Sync weights

```bash
cd ~/projects/rgp-analyzer-cli/examples/z-image-turbo-rgp/src
bash ./sync-weights.sh
```

This script:

- reuses example-local weights if already present
- otherwise tries `/mnt/win-c/games/amd-vulkan-tutorial/shared/models/zimage`
- otherwise tries `/mnt/win-c/games/koboldcpp/models/zimage`
- and will attempt to mount the largest NTFS partition to `/mnt/win-c` if needed

## Build

```bash
bash ./build.sh
```

## Run real inference

```bash
bash ./run.sh "a red robot reading a book in a library"
```

To stop after a specific phase instead of producing a final image:

```bash
bash ./src/run-phase.sh condition "a red robot reading a book in a library"
bash ./src/run-phase.sh diffusion "a red robot reading a book in a library"
```

To probe whether an individual phase is producing usable GPU capture:

```bash
bash ./src/probe-phases.sh "a red robot reading a book in a library"
```

To measure Vulkan op timings for a single phase instead of the full generate path:

```bash
bash ./src/run-phase-perf.sh diffusion "a red robot reading a book in a library"
```

To reproduce the default sparse-trace behavior and compare it against the fixed profiling path:

```bash
bash ./src/reproduce-profile-fix.sh "a red robot reading a book in a library"
```

To probe which profiling knobs are actually required for dense `.rgp` output:

```bash
bash ./src/probe-profile-knobs.sh "a red robot reading a book in a library"
```

To probe which real runtime knobs actually change diffusion performance:

```bash
bash ./src/probe-runtime-knobs.sh "a red robot reading a book in a library"
```

To compare a real perf A/B against a profiling-mode `.rgp` A/B in one shot:

```bash
bash ./src/compare-tuning-runs.sh \
  ./perf-logs/runtime-knob-probes/baseline.json \
  ./captures/baseline-current.rgp \
  ./perf-logs/runtime-knob-probes/force_k_quant_medium.json \
  ./captures/force-k-quant-medium.rgp
```

To compare one focused shader before and after a backend or shader change:

```bash
PYTHONPATH=../../src python3 -m rgp_analyzer_cli shader-focus \
  ./captures/baseline-current.rgp \
  --code-object-index 1 \
  --source-file ./app/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp

PYTHONPATH=../../src python3 -m rgp_analyzer_cli compare-shader-focus \
  ./captures/baseline-current.rgp \
  ./captures/baseline-splitk-heuristic.rgp \
  --code-object-index 1 \
  --source-file ./app/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp

PYTHONPATH=../../src python3 -m rgp_analyzer_cli compare-shader-focus \
  ./captures/baseline-current.rgp \
  ./captures/baseline-splitk-heuristic.rgp \
  --code-object-index 1 \
  --source-file ./app/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp \
  --source-excerpt

PYTHONPATH=../../src python3 -m rgp_analyzer_cli metrics-doc --format report
```

The last command emits a report-ready metrics chapter. The checked-in reference lives at
[docs/metrics.md](/home/taowen/projects/rgp-analyzer-cli/docs/metrics.md).

The current workload uses the vendored `direct-txt2img` source tree at 1024x1024 with 8 steps.
Generation size can be adjusted with:

```bash
ZIMAGE_STEPS=4 ZIMAGE_WIDTH=512 ZIMAGE_HEIGHT=512 ZIMAGE_REPEAT=2 bash ./run.sh "a red robot reading a book in a library"
```

## Capture `.rgp`

```bash
bash ./capture-rgp.sh "a red robot reading a book in a library"
```

To capture only the conditioning or latent-generation portion:

```bash
bash ./src/capture-phase-rgp.sh condition "a red robot reading a book in a library"
bash ./src/capture-phase-rgp.sh diffusion "a red robot reading a book in a library"
```

To capture the proven profiling-friendly real diffusion path, use:

```bash
bash ./src/capture-diffusion-profile.sh "a red robot reading a book in a library"
```

That enables:

- `GGML_VK_DISABLE_FUSION=1`
- `GGML_VK_PROFILE_NODES_PER_SUBMIT=1`
- `--stop-after-phase diffusion`

This is the first path in this example that consistently upgrades the real workload from `trace_quality=resource_only` to `trace_quality=dispatch_isa`. Current knob-isolation experiments show:

- `disable_fusion + nodes_per_submit=1` is sufficient
- `disable_graph_optimize` is not required
- `matmul_bytes_per_submit=0` is not required
- `nodes_per_submit=2` already falls back to `resource_only`

When `shader-triage` reports:

- `trace_quality: level=resource_only`
- `profiling_constraints: submit_dilution_suspected=True`

the current `ggml-vulkan` execution organization is still too coarse for `.rgp` to preserve dispatch/instruction-level runtime evidence, even though SQTT payload is present.

For a shorter profiling window on the same real model, use:

```bash
bash ./capture-focused-rgp.sh "a red robot reading a book in a library"
```

That keeps the workload real but reduces the generation window to a smaller image and fewer steps, which is often better for short `.rgp` capture experiments.

Recommended workflow for real tuning:

```bash
bash ./perf-then-capture.sh "a red robot reading a book in a library"
```

This now runs the proven diffusion-only profile path for capture, not the older full-run capture path.

If you want the full three-step workflow in one command:

```bash
bash ./src/profile-and-analyze.sh "a red robot reading a book in a library"
```

The latest capture is copied to:

```text
examples/z-image-turbo-rgp/captures/latest.rgp
```

This compute-only workload should use `TRACE_MODE=per_submit`. Mesa's trigger path is tied to frame/present handling, so `TRACE_MODE=trigger` is not the reliable path here.

When `TRACE_MODE=per_submit` generates multiple `.rgp` files, the script snapshots `direct-txt2img_*.rgp` while the app is running, copies the resulting captures into `captures/`, writes `captures/last-capture-manifest.txt`, and promotes the largest new capture to `latest.rgp`.
`analyze-latest.sh` also ranks the captures in that manifest by `trace_quality`, decoded instructions, and mapped dispatch count, so the largest file is no longer the only signal.

## Analyze the latest capture

```bash
bash ./analyze-latest.sh
```

That runs:

- workload evidence summary: latest `ggml` perf + `trace_quality`
- latest `ggml` Vulkan perf summary, if available
- `inspect`
- `resource-summary`
- `stitch-report`
- `decode-sqtt`
- `shader-triage`

This example is intended to expose what still breaks or gets fuzzy once the profiler sees a real model inference workload.

Current best-practice workflow inside this example is:

1. `run-phase-perf.sh diffusion`
2. identify the hottest Vulkan ops from the `ggml` perf log
3. `capture-diffusion-profile.sh`
4. `analyze-latest.sh`

Current runtime-knob probe results on the bundled diffusion-only workload are:

- `flash_attn` is slower than baseline
- `diffusion_conv_direct` is also slower than baseline
- `clip_cpu=0` is much slower than baseline
- `GGML_VK_DISABLE_F16=1` is dramatically slower than baseline
- increasing `split_k` for the hot q4_K/q5_K diffusion matmuls is a real win on this RADV NAVI31 setup
- the bundled ggml-vulkan copy now carries a conservative AMD k-quant split-k heuristic derived from that result

So the current tuning target remains the quantized `MUL_MAT q4_K/q5_K` path rather than higher-level runtime switches.

If the `.rgp` side reports `sparse_runtime_trace=True`, treat the `.rgp` as resource/stitch evidence only and keep the perf log as the primary runtime source for this workload.

The compact `workload_evidence` block is the quickest way to see whether the current `.rgp` is usable for runtime hotspot work or only for resource/stitch inspection.
It now also shows the capture profile mode, stop phase, app exit status, and submit policy used for the current trace.
It also shows `profiling_constraints`, which makes the current execution-organization diagnosis explicit:

- `submit_dilution_suspected=True` means SQTT payload exists, but dispatch/instruction-level runtime evidence was diluted away by the current `ggml-vulkan` execution shape.
- `submit_dilution_suspected=False` means the current capture retained dispatch-level runtime evidence.

To compare the real-model trace against the dense bundled microbenchmark trace:

```bash
python3 ./src/compare-trace-quality.py \
  ../minimal-vulkan-compute-rgp/captures/latest.rgp \
  ./captures/latest.rgp
```
