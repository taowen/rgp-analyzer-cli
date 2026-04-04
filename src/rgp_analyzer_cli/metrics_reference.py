from __future__ import annotations

from typing import Any


def metrics_reference_payload() -> dict[str, Any]:
    return {
        "title": "RGP Analyzer Metrics Reference",
        "sections": [
            {
                "name": "Capture Scope",
                "metrics": [
                    {
                        "name": "trace_quality.level",
                        "kind": "enum",
                        "source": "RGP container + thread-trace decode + dispatch/ISA mapping",
                        "units": "level",
                        "meaning": "Overall evidence depth available in the capture.",
                        "values": "resource_only, dispatch_isa",
                    },
                    {
                        "name": "profiling_constraints.submit_dilution_suspected",
                        "kind": "bool",
                        "source": "thread-trace diagnostics",
                        "units": "boolean",
                        "meaning": "Whether SQTT exists but dispatch/instruction evidence is too diluted to support shader tuning.",
                    },
                    {
                        "name": "capture_capabilities.*",
                        "kind": "bool-set",
                        "source": "tool capability model",
                        "units": "boolean",
                        "meaning": "Which metric families are actually available on the current capture path.",
                    },
                ],
            },
            {
                "name": "Static Resource Metrics",
                "metrics": [
                    {
                        "name": "vgpr_count",
                        "kind": "static",
                        "source": "embedded code object metadata",
                        "units": "registers per wave",
                        "meaning": "Vector general-purpose registers declared by the focused shader.",
                    },
                    {
                        "name": "sgpr_count",
                        "kind": "static",
                        "source": "embedded code object metadata",
                        "units": "registers per wave",
                        "meaning": "Scalar general-purpose registers declared by the focused shader.",
                    },
                    {
                        "name": "lds_size",
                        "kind": "static",
                        "source": "embedded code object metadata",
                        "units": "bytes",
                        "meaning": "Static LDS allocation for the focused shader.",
                    },
                    {
                        "name": "scratch_memory_size",
                        "kind": "static",
                        "source": "embedded code object metadata",
                        "units": "bytes",
                        "meaning": "Static scratch allocation for the focused shader.",
                    },
                    {
                        "name": "wavefront_size",
                        "kind": "static",
                        "source": "embedded code object metadata",
                        "units": "lanes",
                        "meaning": "Wavefront width declared by the focused shader.",
                    },
                ],
            },
            {
                "name": "Runtime Totals",
                "metrics": [
                    {
                        "name": "instructions",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "decoded instruction samples",
                        "meaning": "Total decoded dynamic instruction count in the capture-global runtime profile.",
                    },
                    {
                        "name": "avg_stall_per_inst",
                        "kind": "derived",
                        "source": "ROCm thread-trace decoder",
                        "units": "stall cycles per decoded instruction",
                        "meaning": "Average accumulated stall cycles per decoded instruction sample.",
                    },
                    {
                        "name": "stall_share_of_duration",
                        "kind": "derived",
                        "source": "ROCm thread-trace decoder",
                        "units": "ratio",
                        "meaning": "Fraction of total runtime duration accumulated as stall.",
                    },
                    {
                        "name": "stalled_instruction_share",
                        "kind": "derived",
                        "source": "ROCm thread-trace decoder",
                        "units": "ratio",
                        "meaning": "Fraction of decoded instruction samples marked stalled.",
                    },
                    {
                        "name": "avg_wave_lifetime",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "cycles",
                        "meaning": "Average accumulated lifetime per traced wave in the current runtime window.",
                    },
                    {
                        "name": "max_wave_lifetime",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "cycles",
                        "meaning": "Maximum accumulated lifetime among traced waves.",
                    },
                ],
            },
            {
                "name": "Occupancy Metrics",
                "metrics": [
                    {
                        "name": "occupancy_average_active / runtime_average_active",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "active-wave proxy",
                        "meaning": "Capture-window average active wave proxy. Use for A/B comparison, not as literal waves-per-CU.",
                    },
                    {
                        "name": "occupancy_max_active / runtime_max_active",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "active-wave proxy",
                        "meaning": "Maximum active wave proxy observed in the traced window.",
                    },
                ],
            },
            {
                "name": "Instruction Category Metrics",
                "metrics": [
                    {
                        "name": "VALU / SALU / VMEM / SMEM / LDS / IMMED counts",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "decoded instruction samples",
                        "meaning": "Dynamic instruction counts by category.",
                    },
                    {
                        "name": "<category>.duration_total",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "accumulated cycles",
                        "meaning": "Accumulated runtime duration attributed to this category across traced waves.",
                    },
                    {
                        "name": "<category>.stall_total",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "accumulated stall cycles",
                        "meaning": "Accumulated stall cycles attributed to this category across traced waves.",
                    },
                    {
                        "name": "stall_over_duration",
                        "kind": "derived",
                        "source": "ROCm thread-trace decoder",
                        "units": "ratio",
                        "meaning": "Category stall_total divided by duration_total.",
                    },
                ],
            },
            {
                "name": "Wave-State Metrics",
                "metrics": [
                    {
                        "name": "EXEC share",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "ratio",
                        "meaning": "Fraction of wave-state duration spent actively executing instructions.",
                    },
                    {
                        "name": "WAIT share",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "ratio",
                        "meaning": "Fraction of wave-state duration spent waiting on sync-style instructions such as waitcnt/barrier.",
                    },
                    {
                        "name": "IDLE share",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "ratio",
                        "meaning": "Fraction of wave-state duration where the wave is resident but not doing useful work.",
                    },
                    {
                        "name": "STALL share",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "ratio",
                        "meaning": "Fraction of wave-state duration blocked by resource/dependency hazards outside explicit wait instructions.",
                    },
                ],
            },
            {
                "name": "Runtime Proxy Metrics",
                "metrics": [
                    {
                        "name": "sync_wait_share",
                        "kind": "derived",
                        "source": "WAIT wave-state duration",
                        "units": "ratio",
                        "meaning": "Share of capture runtime accumulated in WAIT state.",
                    },
                    {
                        "name": "sync_wait_cycles / sync_wait_cycles_per_inst",
                        "kind": "derived",
                        "source": "WAIT wave-state duration",
                        "units": "cycles / cycles per instruction",
                        "meaning": "Absolute and normalized WAIT-state cost.",
                    },
                    {
                        "name": "immed_duration_share / immed_stall_share / immed_stall_per_inst",
                        "kind": "derived",
                        "source": "IMMED instruction category",
                        "units": "ratio / cycles per instruction",
                        "meaning": "Proxy for sync-style scalar instructions such as waitcnt/barrier dominating runtime.",
                    },
                    {
                        "name": "global_memory_duration_share / global_memory_stall_share",
                        "kind": "derived",
                        "source": "VMEM + SMEM categories",
                        "units": "ratio",
                        "meaning": "Capture-global proxy for global/scalar memory contribution.",
                    },
                    {
                        "name": "lds_duration_share / lds_stall_share / lds_stall_per_inst",
                        "kind": "derived",
                        "source": "LDS category",
                        "units": "ratio / cycles per instruction",
                        "meaning": "Proxy for LDS pressure, barrier interaction, or LDS dependency cost.",
                    },
                ],
            },
            {
                "name": "Hotspot Metrics",
                "metrics": [
                    {
                        "name": "runtime_top_hotspot.address",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "PC address",
                        "meaning": "Top aggregated runtime hotspot bucket. `0x0` commonly means kernel-entry bucket, not necessarily a single slow instruction.",
                    },
                    {
                        "name": "duration / stall / hitcount",
                        "kind": "dynamic",
                        "source": "ROCm thread-trace decoder",
                        "units": "cycles / count",
                        "meaning": "Accumulated cost and sample count for a hotspot bucket.",
                    },
                    {
                        "name": "avg_duration_per_hit / avg_stall_per_hit",
                        "kind": "derived",
                        "source": "ROCm thread-trace decoder",
                        "units": "cycles per hotspot hit",
                        "meaning": "Normalized hotspot cost useful for A/B comparisons.",
                    },
                    {
                        "name": "dispatch_share / dispatch_isa_share",
                        "kind": "derived",
                        "source": "stitch model + dispatch ISA mapping",
                        "units": "ratio",
                        "meaning": "How much of the focused hotspot aligns with the focused dispatch assignment and dispatch-ISA evidence.",
                    },
                ],
            },
            {
                "name": "Event and Barrier Context",
                "metrics": [
                    {
                        "name": "dispatch_spans / dispatch_assignments",
                        "kind": "stitch",
                        "source": "RGP markers + stitch model",
                        "units": "count",
                        "meaning": "Recovered dispatch spans and assignment count in the capture.",
                    },
                    {
                        "name": "bind_markers / cb_spans",
                        "kind": "stitch",
                        "source": "RGP markers + stitch model",
                        "units": "count",
                        "meaning": "Recovered bind-pipeline markers and command-buffer spans.",
                    },
                    {
                        "name": "barrier_markers / barrier_spans / unmatched_barrier_begins",
                        "kind": "stitch",
                        "source": "RGP markers + stitch model",
                        "units": "count",
                        "meaning": "Recovered barrier structure and unmatched begin markers.",
                    },
                    {
                        "name": "dispatches_per_cb / barriers_per_dispatch",
                        "kind": "derived",
                        "source": "stitch model",
                        "units": "ratio",
                        "meaning": "Capture-structure density proxies for submit and synchronization organization.",
                    },
                ],
            },
            {
                "name": "Instruction and Source Correlation",
                "metrics": [
                    {
                        "name": "top_pcs / instruction_ranking",
                        "kind": "stitch + static ISA",
                        "source": "dispatch ISA mapping + disassembly",
                        "units": "ranked PC list",
                        "meaning": "Most relevant PCs for the focused shader, with static ISA text.",
                    },
                    {
                        "name": "instruction_ranking_delta",
                        "kind": "compare",
                        "source": "hotspot score + dispatch ISA mapping",
                        "units": "delta",
                        "meaning": "A/B change in instruction score, hotspot duration, and hotspot stall for the same PC bucket.",
                    },
                    {
                        "name": "source_hints / source_delta_hints",
                        "kind": "source correlation",
                        "source": "heuristic keyword match against provided shader source",
                        "units": "line numbers",
                        "meaning": "Source lines most likely related to the current runtime bottleneck or delta.",
                    },
                ],
            },
            {
                "name": "Current Public-Path Limits",
                "metrics": [
                    {
                        "name": "active_lane_count",
                        "kind": "missing",
                        "source": "not available on current RGP thread-trace path",
                        "units": "boolean capability",
                        "meaning": "Per-instruction active lane / exec mask is not present in the current RADV + RGP thread-trace path.",
                    },
                    {
                        "name": "not_issued_reason",
                        "kind": "missing",
                        "source": "not available on current RGP thread-trace path",
                        "units": "boolean capability",
                        "meaning": "Fine-grained not-issued reasons are not present in the current RADV + RGP thread-trace path.",
                    },
                    {
                        "name": "memory_stride / cacheline_efficiency",
                        "kind": "missing",
                        "source": "not available on current RGP thread-trace path",
                        "units": "boolean capability",
                        "meaning": "Stride, cacheline efficiency, and unaligned-transfer style metrics are not present in the current RADV + RGP thread-trace path.",
                    },
                ],
            },
        ],
    }


def render_metrics_reference_markdown(payload: dict[str, Any], *, title: str | None = None) -> str:
    lines: list[str] = []
    lines.append(f"# {title or payload.get('title') or 'Metrics Reference'}")
    lines.append("")
    for section in payload.get("sections") or []:
        lines.append(f"## {section.get('name')}")
        lines.append("")
        for metric in section.get("metrics") or []:
            lines.append(f"### `{metric.get('name')}`")
            lines.append("")
            lines.append(f"- `kind`: {metric.get('kind')}")
            lines.append(f"- `source`: {metric.get('source')}")
            lines.append(f"- `units`: {metric.get('units')}")
            if metric.get("values"):
                lines.append(f"- `values`: {metric.get('values')}")
            lines.append(f"- `meaning`: {metric.get('meaning')}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_metrics_reference_report_section(payload: dict[str, Any], *, title: str | None = None) -> str:
    lines: list[str] = []
    lines.append(f"{title or payload.get('title') or 'Metrics Reference'}:")
    for section in payload.get("sections") or []:
        lines.append(f"  {section.get('name')}:")
        for metric in section.get("metrics") or []:
            line = (
                f"    - {metric.get('name')}: "
                f"kind={metric.get('kind')} source={metric.get('source')} units={metric.get('units')}"
            )
            if metric.get("values"):
                line += f" values={metric.get('values')}"
            line += f" meaning={metric.get('meaning')}"
            lines.append(line)
    return "\n".join(lines)
