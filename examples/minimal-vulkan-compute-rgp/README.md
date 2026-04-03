# Minimal Vulkan Compute RGP Example

This example shows the full loop inside this repository:

1. build a Vulkan compute microbenchmark
2. run it normally
3. capture an `.rgp` trace on Linux with `RADV`
4. analyze the trace with `rgp-analyzer`
5. disassemble embedded code object ELF to AMDGPU ISA
6. summarize ISA mix for agent-friendly consumption
7. preview SQTT raw streams in structured form
8. decode SQTT streams through ROCm thread-trace decoder
9. stitch decoded hotspots back to the embedded code object and symbol metadata

## Files

```text
examples/minimal-vulkan-compute-rgp/
  README.md
  src/
    main.cpp
    build.sh
    compile-shaders.sh
    run.sh
    capture-rgp.sh
    capture-scenarios.sh
    validate-scenarios.sh
    analyze-latest.sh
    shaders/
      baseline.comp
      reg_pressure.comp
      lds_mix.comp
```

## Requirements

- Linux
- Vulkan runtime
- `glslc`
- C++ compiler
- AMD GPU with Vulkan
- `RADV` if you want CLI `.rgp` capture with `MESA_VK_TRACE=rgp`

## Build

```bash
cd ~/projects/rgp-analyzer-cli/examples/minimal-vulkan-compute-rgp/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run.sh baseline
bash ./run.sh reg_pressure
bash ./run.sh baseline --mode multi-pipeline --shader-secondary ./build/shaders/lds_mix.spv
```

The program prints timestamp-query-based GPU timing for repeated compute dispatches.

## Capture `.rgp`

```bash
bash ./capture-rgp.sh reg_pressure
bash ./capture-rgp.sh baseline --mode multi-pipeline --shader-secondary ./build/shaders/lds_mix.spv
bash ./capture-scenarios.sh
bash ./validate-scenarios.sh
bash ./validate-scenarios.sh --analyze-only
```

This uses:

```bash
MESA_VK_TRACE=rgp
MESA_VK_TRACE_PER_SUBMIT=1
RADV_THREAD_TRACE_BUFFER_SIZE=65536
```

The example script uses `67108864` by default, which is `64 MiB` in bytes.

The script copies the latest generated `.rgp` into:

```text
examples/minimal-vulkan-compute-rgp/captures/
```

## Parse the latest capture

If you installed the tool:

```bash
rgp-analyzer inspect ../captures/latest.rgp
rgp-analyzer events ../captures/latest.rgp
rgp-analyzer code-objects ../captures/latest.rgp --show-strings
rgp-analyzer disassemble-code-objects ../captures/latest.rgp --limit 1 --symbol _amdgpu_cs_main
rgp-analyzer isa-summary ../captures/latest.rgp --limit 1 --symbol _amdgpu_cs_main
rgp-analyzer sqtt ../captures/latest.rgp
rgp-analyzer scan-sqtt ../captures/latest.rgp --stream-limit 2 --dword-limit 16
rgp-analyzer general-api-markers ../captures/latest.rgp --stream-limit 1
rgp-analyzer decoder-bridge ../captures/latest.rgp
rgp-analyzer decode-sqtt ../captures/latest.rgp --build-helper
rgp-analyzer decode-sqtt ../captures/latest.rgp --build-helper --strict --hotspot-limit 16
rgp-analyzer resource-summary ../captures/latest.rgp
rgp-analyzer shader-triage ../captures/latest.rgp --build-helper
```

Without installation:

```bash
cd ~/projects/rgp-analyzer-cli
PYTHONPATH=src python3 -m rgp_analyzer_cli inspect examples/minimal-vulkan-compute-rgp/captures/latest.rgp
```

Or use the bundled helper:

```bash
cd ~/projects/rgp-analyzer-cli/examples/minimal-vulkan-compute-rgp/src
bash ./analyze-latest.sh
```

## End-to-end demo

```bash
cd ~/projects/rgp-analyzer-cli/examples/minimal-vulkan-compute-rgp/src
bash ./capture-rgp.sh reg_pressure
bash ./analyze-latest.sh
```

The current example verifies:

- `.rgp` generation on Linux with `RADV`
- queue event parsing
- code object extraction
- AMDGPU ISA disassembly via `llvm-objdump`
- ISA mix summary suitable for automated heuristics
- SQTT descriptor parsing
- SQTT raw stream preview
- raw `General API` marker scanning for command-context hints
- high-confidence command-context hints for dispatch / push-constant / timestamp markers
- SQTT decode to wave / instruction-category summaries when `rocprof-trace-decoder` is available
- resource metadata extraction from the embedded AMDGPU note
- high-level shader triage findings for tuning loops
- fallback hotspot stitching back to `_amdgpu_cs_main` for the single-kernel example capture
- dispatch-span-weighted hotspot candidate ranking for multi-pipeline captures instead of a false single-code-object stitch
- scenario-matrix regression across `single-baseline`, `multi-pipeline`, `multi-cmdbuf`, and `barrier-mix`
