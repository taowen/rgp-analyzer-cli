# rgp-analyzer-cli

CLI for Linux/RADV `.rgp` captures, aimed at Vulkan compute shader tuning without opening RGP GUI.
It reports capture structure, runtime signals, stitch confidence, and shader-resource facts. It does not try to prescribe optimizations.
The old v1 code is archived under [`bak/rgp_analyzer_cli_v1`](/home/taowen/projects/rgp-analyzer-cli/bak/rgp_analyzer_cli_v1). The canonical package is `rgp_analyzer_cli`.

## What I need from this tool

When iterating on a Vulkan compute shader, I want to answer these questions quickly:

- did the capture succeed?
- which code objects and pipelines are in the trace?
- can `.rgp` correlate `Code Object -> PSO -> Loader Event` cleanly?
- what does runtime SQTT say about instruction mix?
- what do the embedded code objects say about `VGPR / SGPR / LDS / scratch`?
- can the top hotspot be mapped back to a shader symbol or at least to likely code objects?

This repo is built around that loop.

## Fast path

Install:

```bash
cd ~/projects/rgp-analyzer-cli
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Smoke check:

```bash
bash ./scripts/release-check.sh
```

Extended scenario check:

```bash
bash ./examples/minimal-vulkan-compute-rgp/src/validate-scenarios.sh --analyze-only --skip-decode
```

Typical capture on Linux + RADV:

```bash
MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 ./your_app
capture="$(ls -1t /tmp/*.rgp | head -n 1)"
```

Minimal analysis loop:

```bash
rgp-analyzer stitch-report "$capture"
rgp-analyzer decode-sqtt "$capture" --build-helper
rgp-analyzer resource-summary "$capture"
rgp-analyzer dispatch-isa-map "$capture"
rgp-analyzer shader-triage "$capture" --build-helper
```

Heavy commands (`decode-sqtt`, `dispatch-isa-map`, `shader-triage`) now cache JSON results under `.cache/rgp-analyzer-cli`. Use `--no-cache` when you explicitly want a cold rerun.

## Commands That Matter

- `stitch-report`
  Shows the authoritative `.rgp` stitch model:
  `Code Object -> internal_pipeline_hash -> PSO -> Loader Event`.
  Use this before trusting hotspot mapping.

- `decode-sqtt`
  Runs the ROCm helper on SQTT streams and emits runtime instruction-category summaries plus stitched hotspot hints.

- `resource-summary`
  Extracts AMDGPU metadata note values from embedded code objects:
  `vgpr_count`, `sgpr_count`, `lds_size`, `scratch_memory_size`, `wavefront_size`.

- `shader-triage`
  Highest-level signal summary. Combines runtime SQTT, ISA summary, resource metadata, and stitch facts into a compact report.
  It now also reports `trace_quality.runtime_evidence_level`, so sparse real-model captures can be identified quickly.

- `shader-focus`
  Shader-tuning view for the hottest code object in a capture. It compresses the current capture to:
  `VGPR / SGPR / LDS / scratch`, runtime stall and occupancy, top hotspot cost, top dispatch-ISA PCs, and runtime proxies such as
  `global_mem`, `lds`, `sync_wait`, and `IMMED`-heavy behavior.
  The runtime summary is still capture-global; the most shader-specific evidence is the focused hotspot bucket plus `top_pcs`.

- `compare-shader-focus`
  A/B compare for shader work. It keeps the compare centered on the focused code object and highlights whether the candidate:
  changes occupancy, sync-wait pressure, LDS-heavy behavior, average stall, or hotspot cost per hit.

- `dispatch-isa-map`
  Uses the `.rgp` stitch model plus vendored raw SQTT decode to recover dispatch-segment ISA evidence without relying on an external `tinygrad` checkout.

## Current stitch model

Primary path:

1. read `internal_pipeline_hash` from embedded AMDGPU metadata
2. correlate to `PSO_CORRELATION`
3. correlate to `CODE_OBJECT_LOADER_EVENTS`
4. use recovered `BIND_PIPELINE` and API lifecycle markers as runtime context

Important constraint:

- ROCm's native decoder still rejects Mesa's relocatable `AMDGPU-PAL` ELF as a runtime-loadable code object.
- Because of that, hotspot stitching still relies on `.rgp` metadata and fallback logic on top of the ROCm summary.
- The raw SQTT packet decoder is vendored into this repo; there is no runtime dependency on an external `tinygrad` checkout anymore.

Current behavior:

- single-kernel captures can map the top hotspot back to `_amdgpu_cs_main+0x0`
- multi-pipeline captures fall back to dispatch-span-weighted candidate ranking per stream

## What this tool is good at

- validating `.rgp` structure
- extracting code objects, PSOs, loader events, and SQTT streams
- disassembling embedded code objects
- recovering `VGPR / SGPR / LDS / scratch`
- summarizing runtime instruction mix
- reconstructing enough runtime context to say which code objects are plausible hotspot owners

## What it still does not do

- exact RGP-GUI-equivalent runtime-PC stitching for Mesa captures
- full barrier / occupancy / stall-reason interpretation
- full SQTT packet semantics
- replacing RGA for deep static resource analysis

## Example

There is a bundled end-to-end example:

```bash
cd ~/projects/rgp-analyzer-cli/examples/minimal-vulkan-compute-rgp/src
bash ./compile-shaders.sh
bash ./build.sh
bash ./capture-rgp.sh reg_pressure
bash ./capture-scenarios.sh
bash ./analyze-latest.sh
```

That example now includes:

- single-pipeline capture
- multi-pipeline capture
- multi-command-buffer capture
- barrier-heavy capture

See [examples/minimal-vulkan-compute-rgp/README.md](/home/taowen/projects/rgp-analyzer-cli/examples/minimal-vulkan-compute-rgp/README.md).

## Practical workflow

If I am tuning a compute shader, the shortest useful loop is:

1. capture `.rgp`
2. run `stitch-report`
3. run `decode-sqtt`
4. run `resource-summary`
5. run `shader-triage`
6. use `shader-focus` or `compare-shader-focus` when changing a shader or backend heuristic
7. if still ambiguous, inspect dispatch-level ISA evidence with `dispatch-isa-map`
8. if that is still not enough, open RGP or hand the ELF to RGA
