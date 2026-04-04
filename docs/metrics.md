# RGP Analyzer Metrics Reference

This document is the report-friendly reference for the metrics emitted by `shader-focus`, `compare-shader-focus`, and related reports.

The same content can be emitted from the CLI:

```bash
cd ~/projects/rgp-analyzer-cli
PYTHONPATH=src python3 -m rgp_analyzer_cli metrics-doc --format markdown
PYTHONPATH=src python3 -m rgp_analyzer_cli metrics-doc --format report
```

## Capture Scope

### `trace_quality.level`

- `kind`: enum
- `source`: RGP container + thread-trace decode + dispatch/ISA mapping
- `units`: level
- `values`: resource_only, dispatch_isa
- `meaning`: Overall evidence depth available in the capture.

### `profiling_constraints.submit_dilution_suspected`

- `kind`: bool
- `source`: thread-trace diagnostics
- `units`: boolean
- `meaning`: Whether SQTT exists but dispatch/instruction evidence is too diluted to support shader tuning.

### `capture_capabilities.*`

- `kind`: bool-set
- `source`: tool capability model
- `units`: boolean
- `meaning`: Which metric families are actually available on the current capture path.

## Static Resource Metrics

### `vgpr_count`

- `kind`: static
- `source`: embedded code object metadata
- `units`: registers per wave
- `meaning`: Vector general-purpose registers declared by the focused shader.

### `sgpr_count`

- `kind`: static
- `source`: embedded code object metadata
- `units`: registers per wave
- `meaning`: Scalar general-purpose registers declared by the focused shader.

### `lds_size`

- `kind`: static
- `source`: embedded code object metadata
- `units`: bytes
- `meaning`: Static LDS allocation for the focused shader.

### `scratch_memory_size`

- `kind`: static
- `source`: embedded code object metadata
- `units`: bytes
- `meaning`: Static scratch allocation for the focused shader.

### `wavefront_size`

- `kind`: static
- `source`: embedded code object metadata
- `units`: lanes
- `meaning`: Wavefront width declared by the focused shader.

## Runtime Totals

### `instructions`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: decoded instruction samples
- `meaning`: Total decoded dynamic instruction count in the capture-global runtime profile.

### `avg_stall_per_inst`

- `kind`: derived
- `source`: ROCm thread-trace decoder
- `units`: stall cycles per decoded instruction
- `meaning`: Average accumulated stall cycles per decoded instruction sample.

### `stall_share_of_duration`

- `kind`: derived
- `source`: ROCm thread-trace decoder
- `units`: ratio
- `meaning`: Fraction of total runtime duration accumulated as stall.

### `stalled_instruction_share`

- `kind`: derived
- `source`: ROCm thread-trace decoder
- `units`: ratio
- `meaning`: Fraction of decoded instruction samples marked stalled.

### `avg_wave_lifetime`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: cycles
- `meaning`: Average accumulated lifetime per traced wave in the current runtime window.

### `max_wave_lifetime`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: cycles
- `meaning`: Maximum accumulated lifetime among traced waves.

## Occupancy Metrics

### `occupancy_average_active / runtime_average_active`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: active-wave proxy
- `meaning`: Capture-window average active wave proxy. Use for A/B comparison, not as literal waves-per-CU.

### `occupancy_max_active / runtime_max_active`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: active-wave proxy
- `meaning`: Maximum active wave proxy observed in the traced window.

## Instruction Category Metrics

### `VALU / SALU / VMEM / SMEM / LDS / IMMED counts`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: decoded instruction samples
- `meaning`: Dynamic instruction counts by category.

### `<category>.duration_total`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: accumulated cycles
- `meaning`: Accumulated runtime duration attributed to this category across traced waves.

### `<category>.stall_total`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: accumulated stall cycles
- `meaning`: Accumulated stall cycles attributed to this category across traced waves.

### `stall_over_duration`

- `kind`: derived
- `source`: ROCm thread-trace decoder
- `units`: ratio
- `meaning`: Category stall_total divided by duration_total.

## Wave-State Metrics

### `EXEC share`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: ratio
- `meaning`: Fraction of wave-state duration spent actively executing instructions.

### `WAIT share`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: ratio
- `meaning`: Fraction of wave-state duration spent waiting on sync-style instructions such as waitcnt/barrier.

### `IDLE share`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: ratio
- `meaning`: Fraction of wave-state duration where the wave is resident but not doing useful work.

### `STALL share`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: ratio
- `meaning`: Fraction of wave-state duration blocked by resource/dependency hazards outside explicit wait instructions.

## Runtime Proxy Metrics

### `sync_wait_share`

- `kind`: derived
- `source`: WAIT wave-state duration
- `units`: ratio
- `meaning`: Share of capture runtime accumulated in WAIT state.

### `sync_wait_cycles / sync_wait_cycles_per_inst`

- `kind`: derived
- `source`: WAIT wave-state duration
- `units`: cycles / cycles per instruction
- `meaning`: Absolute and normalized WAIT-state cost.

### `immed_duration_share / immed_stall_share / immed_stall_per_inst`

- `kind`: derived
- `source`: IMMED instruction category
- `units`: ratio / cycles per instruction
- `meaning`: Proxy for sync-style scalar instructions such as waitcnt/barrier dominating runtime.

### `global_memory_duration_share / global_memory_stall_share`

- `kind`: derived
- `source`: VMEM + SMEM categories
- `units`: ratio
- `meaning`: Capture-global proxy for global/scalar memory contribution.

### `lds_duration_share / lds_stall_share / lds_stall_per_inst`

- `kind`: derived
- `source`: LDS category
- `units`: ratio / cycles per instruction
- `meaning`: Proxy for LDS pressure, barrier interaction, or LDS dependency cost.

## Hotspot Metrics

### `runtime_top_hotspot.address`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: PC address
- `meaning`: Top aggregated runtime hotspot bucket. `0x0` commonly means kernel-entry bucket, not necessarily a single slow instruction.

### `duration / stall / hitcount`

- `kind`: dynamic
- `source`: ROCm thread-trace decoder
- `units`: cycles / count
- `meaning`: Accumulated cost and sample count for a hotspot bucket.

### `avg_duration_per_hit / avg_stall_per_hit`

- `kind`: derived
- `source`: ROCm thread-trace decoder
- `units`: cycles per hotspot hit
- `meaning`: Normalized hotspot cost useful for A/B comparisons.

### `dispatch_share / dispatch_isa_share`

- `kind`: derived
- `source`: stitch model + dispatch ISA mapping
- `units`: ratio
- `meaning`: How much of the focused hotspot aligns with the focused dispatch assignment and dispatch-ISA evidence.

## Event and Barrier Context

### `dispatch_spans / dispatch_assignments`

- `kind`: stitch
- `source`: RGP markers + stitch model
- `units`: count
- `meaning`: Recovered dispatch spans and assignment count in the capture.

### `bind_markers / cb_spans`

- `kind`: stitch
- `source`: RGP markers + stitch model
- `units`: count
- `meaning`: Recovered bind-pipeline markers and command-buffer spans.

### `barrier_markers / barrier_spans / unmatched_barrier_begins`

- `kind`: stitch
- `source`: RGP markers + stitch model
- `units`: count
- `meaning`: Recovered barrier structure and unmatched begin markers.

### `dispatches_per_cb / barriers_per_dispatch`

- `kind`: derived
- `source`: stitch model
- `units`: ratio
- `meaning`: Capture-structure density proxies for submit and synchronization organization.

## Instruction and Source Correlation

### `top_pcs / instruction_ranking`

- `kind`: stitch + static ISA
- `source`: dispatch ISA mapping + disassembly
- `units`: ranked PC list
- `meaning`: Most relevant PCs for the focused shader, with static ISA text.

### `instruction_ranking_delta`

- `kind`: compare
- `source`: hotspot score + dispatch ISA mapping
- `units`: delta
- `meaning`: A/B change in instruction score, hotspot duration, and hotspot stall for the same PC bucket.

### `source_hints / source_delta_hints`

- `kind`: source correlation
- `source`: heuristic keyword match against provided shader source
- `units`: line numbers
- `meaning`: Source lines most likely related to the current runtime bottleneck or delta.

## Current Public-Path Limits

### `active_lane_count`

- `kind`: missing
- `source`: not available on current RGP thread-trace path
- `units`: boolean capability
- `meaning`: Per-instruction active lane / exec mask is not present in the current RADV + RGP thread-trace path.

### `not_issued_reason`

- `kind`: missing
- `source`: not available on current RGP thread-trace path
- `units`: boolean capability
- `meaning`: Fine-grained not-issued reasons are not present in the current RADV + RGP thread-trace path.

### `memory_stride / cacheline_efficiency`

- `kind`: missing
- `source`: not available on current RGP thread-trace path
- `units`: boolean capability
- `meaning`: Stride, cacheline efficiency, and unaligned-transfer style metrics are not present in the current RADV + RGP thread-trace path.
