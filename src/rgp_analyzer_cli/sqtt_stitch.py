from __future__ import annotations

from bisect import bisect_right
from typing import Any

from .decode_bridge import build_stitch_model, find_code_object_record
from .dispatch_isa_evidence import build_dispatch_isa_overview, build_stream_dispatch_isa_summary
from .profiling_analysis import build_runtime_hotspot_profiles, build_runtime_profile
from .resource_metadata import extract_resource_metadata


def _symbol_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    elf = record.get("elf", {})
    symbols = elf.get("symbols", [])
    candidates = []
    for symbol in symbols:
        name = symbol.get("name")
        if not name:
            continue
        candidates.append(
            {
                "name": name,
                "value": int(symbol.get("value", 0)),
                "size": int(symbol.get("size", 0)),
            }
        )
    candidates.sort(key=lambda item: (item["value"], item["size"]))
    return candidates


def _section_bounds(record: dict[str, Any], section_name: str) -> tuple[int, int] | None:
    elf = record.get("elf", {})
    for section in elf.get("sections", []):
        if section.get("name") == section_name:
            return int(section.get("addr", 0)), int(section.get("size", 0))
    return None


def _match_symbol(symbols: list[dict[str, Any]], relative_vaddr: int) -> dict[str, Any] | None:
    if not symbols:
        return None
    symbol_values = [item["value"] for item in symbols]
    index = bisect_right(symbol_values, relative_vaddr) - 1
    if index < 0:
        return None
    symbol = symbols[index]
    size = int(symbol.get("size", 0))
    if size > 0 and relative_vaddr >= symbol["value"] + size:
        return None
    return {
        "name": symbol["name"],
        "value": symbol["value"],
        "size": size,
        "offset": relative_vaddr - symbol["value"],
    }


def _bridge_metadata(report: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_records = extract_resource_metadata(report, limit=10000)
    metadata_by_index = {item["index"]: item for item in metadata_records}
    model = build_stitch_model(report)
    return metadata_by_index, model["entries"], model["streams"]


def _decode_diagnostics(report: dict[str, Any], decode_json: dict[str, Any]) -> dict[str, Any]:
    streams = decode_json.get("streams") or []
    shaderdata_records = sum(int(stream.get("shaderdata_records", 0)) for stream in streams)
    total_instructions = sum(int(stream.get("instructions", 0) or 0) for stream in streams)
    total_waves = sum(int(stream.get("waves", 0) or 0) for stream in streams if stream.get("waves") is not None)
    info_counts: dict[str, int] = {}
    zero_code_object_hotspots = 0
    total_hotspots = 0
    for stream in streams:
        for name, count in (stream.get("info_counts") or {}).items():
            info_counts[str(name)] = info_counts.get(str(name), 0) + int(count)
        for hotspot in stream.get("hotspots") or []:
            total_hotspots += 1
            if int(hotspot.get("code_object_id", 0) or 0) == 0:
                zero_code_object_hotspots += 1

    code_object_load_failures = int(decode_json.get("code_object_load_failures", 0) or 0)
    queue_event_count = sum(len(chunk.get("events") or []) for chunk in (report.get("queue_event_chunks") or []))
    total_sqtt_trace_bytes = sum(int(item.get("size", 0) or 0) for item in (report.get("sqtt_data_chunks") or []))
    likely_missing_codeobj_instrumentation = (
        code_object_load_failures > 0 and shaderdata_records == 0 and zero_code_object_hotspots == total_hotspots and total_hotspots > 0
    )
    sparse_runtime_trace = (
        total_sqtt_trace_bytes > 0
        and queue_event_count == 0
        and total_instructions == 0
        and total_hotspots == 0
        and shaderdata_records == 0
    )
    return {
        "shaderdata_records": shaderdata_records,
        "total_instructions": total_instructions,
        "total_waves": total_waves,
        "info_counts": info_counts,
        "zero_code_object_hotspots": zero_code_object_hotspots,
        "total_hotspots": total_hotspots,
        "queue_event_count": queue_event_count,
        "total_sqtt_trace_bytes": total_sqtt_trace_bytes,
        "likely_missing_codeobj_instrumentation": likely_missing_codeobj_instrumentation,
        "sparse_runtime_trace": sparse_runtime_trace,
    }


def stitch_hotspots(report: dict[str, Any], decode_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decode_json or not decode_json.get("streams"):
        return decode_json

    metadata_by_index, bridge_entries, stream_models = _bridge_metadata(report)
    stitched = dict(decode_json)
    stitched["decode_diagnostics"] = _decode_diagnostics(report, decode_json)
    stitched_streams: list[dict[str, Any]] = []
    stream_model_by_index = {int(stream["stream_index"]): stream for stream in stream_models}
    dispatch_isa_by_stream = {
        int(stream.get("index", -1)): build_stream_dispatch_isa_summary(report, int(stream.get("index", -1)))
        for stream in decode_json.get("streams", [])
    }
    dispatch_isa_overview = build_dispatch_isa_overview(dispatch_isa_by_stream)
    stitched["dispatch_isa_overview"] = dispatch_isa_overview

    resolved_entries = [entry for entry in bridge_entries if entry.get("resolved")]
    single_bridge = resolved_entries[0] if len(resolved_entries) == 1 else None

    for stream in decode_json.get("streams", []):
        stitched_stream = dict(stream)
        stream_model = stream_model_by_index.get(int(stream.get("index", -1)), {})
        dispatch_isa = dispatch_isa_by_stream.get(int(stream.get("index", -1)))
        dispatch_assignment_histogram = {
            int(key): int(value) for key, value in (stream_model.get("dispatch_assignment_histogram") or {}).items()
        }
        total_dispatch_assignments = sum(dispatch_assignment_histogram.values())
        stitched_stream["stream_context"] = {
            "resolved_code_object_index": stream_model.get("resolved_code_object_index"),
            "resolved_api_pso_hash": stream_model.get("resolved_api_pso_hash"),
            "resolved_pipeline_hash": stream_model.get("resolved_pipeline_hash"),
            "stream_match_kind": stream_model.get("stream_match_kind"),
            "dispatch_marker_count": len(stream_model.get("dispatch_event_markers", [])),
            "bind_marker_count": len(stream_model.get("bind_pipeline_markers", [])),
            "assigned_dispatch_span_count": int(stream_model.get("assigned_dispatch_span_count", 0)),
            "distinct_assigned_code_objects": stream_model.get("distinct_assigned_code_objects", []),
            "dispatch_isa_mapped_dispatch_count": int((dispatch_isa or {}).get("mapped_dispatch_count", 0) or 0),
            "dispatch_isa_dispatch_count": int((dispatch_isa or {}).get("dispatch_count", 0) or 0),
        }
        stitched_stream["runtime_profile"] = build_runtime_profile(stream)
        if dispatch_isa is not None:
            stitched_stream["dispatch_isa_evidence"] = dispatch_isa
        annotated_hotspots = []
        for hotspot in stream.get("hotspots", []):
            annotated = dict(hotspot)
            candidates: list[dict[str, Any]] = []

            candidate_entries: list[dict[str, Any]] = []
            if hotspot.get("code_object_id"):
                candidate_entries = [item for item in bridge_entries if item.get("load_id") == hotspot["code_object_id"]]
            elif stream_model.get("resolved_code_object_index") is not None:
                candidate_entries = [
                    item
                    for item in resolved_entries
                    if int(item["code_object_index"]) == int(stream_model["resolved_code_object_index"])
                ]
            elif single_bridge is not None and not stream_model:
                candidate_entries = [single_bridge]
            elif dispatch_assignment_histogram:
                candidate_entries = [
                    item
                    for item in resolved_entries
                    if int(item["code_object_index"]) in dispatch_assignment_histogram
                ]
                candidate_entries.sort(
                    key=lambda item: (
                        -dispatch_assignment_histogram.get(int(item["code_object_index"]), 0),
                        int(item["code_object_index"]),
                    )
                )

            for entry in candidate_entries:
                record = find_code_object_record(report, int(entry["code_object_index"]))
                if record is None:
                    continue
                metadata = metadata_by_index.get(int(entry["code_object_index"]), {})
                text_bounds = _section_bounds(record, ".text")
                text_start, text_size = text_bounds if text_bounds else (0, 0)
                load_addr = int(entry.get("load_addr", 0) or 0)
                load_size = int(entry.get("load_size", 0) or 0)
                address = int(hotspot.get("address", 0) or 0)

                relative_vaddr = None
                address_kind = None
                if address and load_addr and load_size and load_addr <= address < load_addr + load_size:
                    relative_vaddr = address - load_addr
                    address_kind = "absolute_gpu_va"
                elif address == 0 and text_size:
                    relative_vaddr = 0
                    if dispatch_assignment_histogram:
                        address_kind = "dispatch_span_assignment_guess"
                    else:
                        address_kind = "entry_point_guess"
                elif text_size and 0 <= address < text_size:
                    relative_vaddr = address
                    address_kind = "relative_vaddr_guess"

                if relative_vaddr is None:
                    continue

                symbol = _match_symbol(_symbol_candidates(record), relative_vaddr)
                assignment_count = dispatch_assignment_histogram.get(int(entry["code_object_index"]), 0)
                dispatch_isa_entry = ((dispatch_isa or {}).get("code_objects") or {}).get(int(entry["code_object_index"]), {})
                global_dispatch_isa_entry = (dispatch_isa_overview.get("code_objects") or {}).get(int(entry["code_object_index"]), {})
                candidate = {
                    "match_kind": address_kind,
                    "code_object_index": entry["code_object_index"],
                    "bridge_index": entry["bridge_index"],
                    "stream_match_kind": stream_model.get("stream_match_kind"),
                    "entry_point": metadata.get("entry_point"),
                    "api_pso_hash": entry.get("api_pso_hash"),
                    "pipeline_hash": entry.get("pipeline_hash"),
                    "load_addr": load_addr,
                    "load_size": load_size,
                    "relative_vaddr": relative_vaddr,
                    "text_offset": relative_vaddr - text_start if text_size else None,
                    "symbol": symbol,
                    "dispatch_assignment_count": assignment_count,
                    "dispatch_assignment_share": (
                        assignment_count / total_dispatch_assignments if total_dispatch_assignments else None
                    ),
                    "dispatch_isa_mapped_dispatch_count": int(dispatch_isa_entry.get("mapped_dispatch_count", 0) or 0),
                    "dispatch_isa_dispatch_count": int(dispatch_isa_entry.get("dispatch_count", 0) or 0),
                    "dispatch_isa_mapped_packet_count": int(dispatch_isa_entry.get("mapped_packet_count", 0) or 0),
                    "dispatch_isa_top_pcs": dispatch_isa_entry.get("top_pcs", []),
                    "dispatch_isa_mapped_dispatch_share": dispatch_isa_entry.get("mapped_dispatch_share"),
                    "global_dispatch_isa_mapped_dispatch_count": int(
                        global_dispatch_isa_entry.get("mapped_dispatch_count", 0) or 0
                    ),
                    "global_dispatch_isa_dispatch_count": int(global_dispatch_isa_entry.get("dispatch_count", 0) or 0),
                    "global_dispatch_isa_mapped_packet_count": int(
                        global_dispatch_isa_entry.get("mapped_packet_count", 0) or 0
                    ),
                    "global_dispatch_isa_top_pcs": global_dispatch_isa_entry.get("top_pcs", []),
                    "global_dispatch_isa_mapped_dispatch_share": global_dispatch_isa_entry.get("mapped_dispatch_share"),
                }
                candidates.append(candidate)

            candidates.sort(
                key=lambda item: (
                    -int(item.get("dispatch_isa_mapped_dispatch_count", 0) or 0),
                    -int(item.get("global_dispatch_isa_mapped_dispatch_count", 0) or 0),
                    -int(item.get("dispatch_isa_mapped_packet_count", 0) or 0),
                    -int(item.get("global_dispatch_isa_mapped_packet_count", 0) or 0),
                    -int(item.get("dispatch_assignment_count", 0) or 0),
                    int(item.get("code_object_index", 0) or 0),
                )
            )
            annotated["stitched_candidates"] = candidates
            if candidates:
                primary = candidates[0]
                share = primary.get("dispatch_assignment_share")
                dispatch_isa_share = primary.get("dispatch_isa_mapped_dispatch_share")
                top_pc = (primary.get("dispatch_isa_top_pcs") or [None])[0]
                annotated["stitch_summary"] = {
                    "candidate_count": len(candidates),
                    "primary_code_object_index": primary.get("code_object_index"),
                    "primary_match_kind": primary.get("match_kind"),
                    "primary_symbol": (primary.get("symbol") or {}).get("name") or primary.get("entry_point"),
                    "primary_symbol_offset": (primary.get("symbol") or {}).get("offset"),
                    "primary_dispatch_assignment_share": share,
                    "primary_dispatch_isa_mapped_dispatch_count": primary.get("dispatch_isa_mapped_dispatch_count"),
                    "primary_dispatch_isa_dispatch_count": primary.get("dispatch_isa_dispatch_count"),
                    "primary_dispatch_isa_mapped_dispatch_share": dispatch_isa_share,
                    "primary_dispatch_isa_top_pc": top_pc,
                    "primary_global_dispatch_isa_mapped_dispatch_count": primary.get("global_dispatch_isa_mapped_dispatch_count"),
                    "primary_global_dispatch_isa_dispatch_count": primary.get("global_dispatch_isa_dispatch_count"),
                    "primary_global_dispatch_isa_mapped_dispatch_share": primary.get("global_dispatch_isa_mapped_dispatch_share"),
                    "primary_global_dispatch_isa_top_pc": (primary.get("global_dispatch_isa_top_pcs") or [None])[0],
                    "is_ambiguous": len(candidates) > 1,
                    "ambiguity_kind": "multi_code_object_stream" if len(candidates) > 1 else "single_candidate",
                }
            else:
                annotated["stitch_summary"] = {
                    "candidate_count": 0,
                    "primary_code_object_index": None,
                    "primary_match_kind": None,
                    "primary_symbol": None,
                    "primary_symbol_offset": None,
                    "primary_dispatch_assignment_share": None,
                    "primary_dispatch_isa_mapped_dispatch_count": 0,
                    "primary_dispatch_isa_dispatch_count": 0,
                    "primary_dispatch_isa_mapped_dispatch_share": None,
                    "primary_dispatch_isa_top_pc": None,
                    "primary_global_dispatch_isa_mapped_dispatch_count": 0,
                    "primary_global_dispatch_isa_dispatch_count": 0,
                    "primary_global_dispatch_isa_mapped_dispatch_share": None,
                    "primary_global_dispatch_isa_top_pc": None,
                    "is_ambiguous": False,
                    "ambiguity_kind": "no_candidates",
                }
            annotated_hotspots.append(annotated)

        stitched_stream["annotated_hotspots"] = annotated_hotspots
        stitched_stream["runtime_hotspot_profiles"] = build_runtime_hotspot_profiles(annotated_hotspots)
        stitched_streams.append(stitched_stream)

    stitched["streams"] = stitched_streams
    return stitched


def render_stitched_hotspots(stitched_decode: dict[str, Any]) -> str:
    lines = ["stitched_hotspots:"]
    diagnostics = stitched_decode.get("decode_diagnostics") or {}
    if diagnostics:
        lines.append(
            "  diagnostics:"
            + f" shaderdata_records={diagnostics.get('shaderdata_records')}"
            + f" queue_events={diagnostics.get('queue_event_count')}"
            + f" instructions={diagnostics.get('total_instructions')}"
            + f" zero_code_object_hotspots={diagnostics.get('zero_code_object_hotspots')}/{diagnostics.get('total_hotspots')}"
            + f" likely_missing_codeobj_instrumentation={diagnostics.get('likely_missing_codeobj_instrumentation')}"
            + f" sparse_runtime_trace={diagnostics.get('sparse_runtime_trace')}"
        )
        if diagnostics.get("info_counts"):
            lines.append(f"    info_counts={diagnostics.get('info_counts')}")
    dispatch_isa_overview = stitched_decode.get("dispatch_isa_overview") or {}
    if dispatch_isa_overview:
        lines.append(
            "  dispatch_isa_overview:"
            + f" mapped={int(dispatch_isa_overview.get('mapped_dispatch_count', 0) or 0)}"
            + f"/{int(dispatch_isa_overview.get('dispatch_count', 0) or 0)}"
        )
        for item in (dispatch_isa_overview.get("ordered") or [])[:4]:
            top_pc = item.get("top_pc") or {}
            top_pc_text = ""
            if top_pc:
                top_pc_text = (
                    f" top_pc=0x{int(top_pc.get('pc', 0)):x}"
                    f" {top_pc.get('mnemonic')} {top_pc.get('operands')}".rstrip()
                )
            share = item.get("mapped_dispatch_share")
            share_text = f" share={share:.2f}" if isinstance(share, (int, float)) else ""
            lines.append(
                f"    code_object[{item['code_object_index']}] "
                f"mapped_dispatches={int(item.get('mapped_dispatch_count', 0))}/{int(item.get('dispatch_count', 0))}"
                f"{share_text}{top_pc_text}"
            )
    for stream in stitched_decode.get("streams", []):
        context = stream.get("stream_context") or {}
        runtime_profile = stream.get("runtime_profile") or {}
        lines.append(
            f"  stream[{stream['index']}] waves={stream.get('wave_records', 0)} instructions={stream.get('instructions', 0)} "
            f"resolved_code_object={context.get('resolved_code_object_index')} stream_match={context.get('stream_match_kind')} "
            f"dispatch_assigned={context.get('assigned_dispatch_span_count', 0)} "
            f"dispatch_code_objects={context.get('distinct_assigned_code_objects', [])} "
            f"dispatch_isa={context.get('dispatch_isa_mapped_dispatch_count', 0)}/{context.get('dispatch_isa_dispatch_count', 0)}"
        )
        if runtime_profile:
            avg_stall = runtime_profile.get("avg_stall_per_inst")
            stall_share = runtime_profile.get("stall_share_of_duration")
            occupancy_avg = runtime_profile.get("occupancy_average_active")
            parts = ["    runtime_profile="]
            if isinstance(avg_stall, (int, float)):
                parts.append(f"avg_stall_per_inst={avg_stall:.2f}")
            if isinstance(stall_share, (int, float)):
                parts.append(f"stall_share={stall_share:.2f}")
            if isinstance(occupancy_avg, (int, float)):
                parts.append(f"occupancy_avg={occupancy_avg:.2f}")
            parts.append(f"occupancy_max={int(runtime_profile.get('occupancy_max_active', 0) or 0)}")
            lines.append(" ".join(parts))
            top_categories = runtime_profile.get("category_profiles") or []
            if top_categories:
                top_category = top_categories[0]
                lines.append(
                    "    category_profile="
                    f"{top_category.get('category')} count={int(top_category.get('count', 0) or 0)} "
                    f"stall_total={int(top_category.get('stall_total', 0) or 0)} "
                    f"duration_total={int(top_category.get('duration_total', 0) or 0)}"
                )
            top_states = runtime_profile.get("wave_state_profiles") or []
            if top_states:
                state = top_states[0]
                share = state.get("share")
                share_text = f" share={share:.2f}" if isinstance(share, (int, float)) else ""
                lines.append(
                    "    wave_state="
                    f"{state.get('state')} duration={int(state.get('duration', 0) or 0)}{share_text}"
                )
        for index, hotspot in enumerate(stream.get("annotated_hotspots", [])):
            summary = hotspot.get("stitch_summary") or {}
            summary_text = ""
            if summary.get("candidate_count"):
                share = summary.get("primary_dispatch_assignment_share")
                share_text = f" primary_share={share:.2f}" if isinstance(share, (int, float)) else ""
                dispatch_isa_share = summary.get("primary_dispatch_isa_mapped_dispatch_share")
                dispatch_isa_share_text = (
                    f" isa_share={dispatch_isa_share:.2f}" if isinstance(dispatch_isa_share, (int, float)) else ""
                )
                summary_text = (
                    f" candidates={summary.get('candidate_count')}"
                    f" ambiguous={summary.get('is_ambiguous')}{share_text}{dispatch_isa_share_text}"
                )
            lines.append(
                f"    hotspot[{index}] code_object_id={hotspot.get('code_object_id')} "
                f"address=0x{int(hotspot.get('address', 0)):x} "
                f"hitcount={hotspot.get('hitcount')} total_duration={hotspot.get('total_duration')}{summary_text}"
            )
            for candidate in hotspot.get("stitched_candidates", [])[:3]:
                symbol = candidate.get("symbol") or {}
                symbol_desc = symbol.get("name")
                if symbol_desc and symbol.get("offset") is not None:
                    symbol_desc = f"{symbol_desc}+0x{int(symbol['offset']):x}"
                assignment_desc = ""
                if candidate.get("dispatch_assignment_count"):
                    share = candidate.get("dispatch_assignment_share")
                    if share is not None:
                        assignment_desc = (
                            f" dispatch_spans={candidate['dispatch_assignment_count']}"
                            f" share={share:.2f}"
                        )
                dispatch_isa_desc = ""
                if candidate.get("dispatch_isa_dispatch_count"):
                    mapped_share = candidate.get("dispatch_isa_mapped_dispatch_share")
                    mapped_share_text = f" share={mapped_share:.2f}" if isinstance(mapped_share, (int, float)) else ""
                    dispatch_isa_desc = (
                        f" dispatch_isa={candidate['dispatch_isa_mapped_dispatch_count']}"
                        f"/{candidate['dispatch_isa_dispatch_count']}{mapped_share_text}"
                    )
                global_dispatch_isa_desc = ""
                if candidate.get("global_dispatch_isa_dispatch_count"):
                    mapped_share = candidate.get("global_dispatch_isa_mapped_dispatch_share")
                    mapped_share_text = f" share={mapped_share:.2f}" if isinstance(mapped_share, (int, float)) else ""
                    global_dispatch_isa_desc = (
                        f" global_dispatch_isa={candidate['global_dispatch_isa_mapped_dispatch_count']}"
                        f"/{candidate['global_dispatch_isa_dispatch_count']}{mapped_share_text}"
                    )
                lines.append(
                    f"      -> code_object[{candidate['code_object_index']}] "
                    f"match={candidate['match_kind']} relative_vaddr=0x{int(candidate['relative_vaddr']):x} "
                    f"entry_point={candidate.get('entry_point')} symbol={symbol_desc}"
                    f"{assignment_desc}{dispatch_isa_desc}{global_dispatch_isa_desc}"
                )
                top_pc = (candidate.get("dispatch_isa_top_pcs") or [None])[0]
                if top_pc:
                    lines.append(
                        "        isa_top_pc="
                        f"0x{int(top_pc.get('pc', 0)):x} "
                        f"count={int(top_pc.get('count', 0))} "
                        f"{top_pc.get('mnemonic')} {top_pc.get('operands')}".rstrip()
                    )
                elif candidate.get("global_dispatch_isa_top_pcs"):
                    top_pc = (candidate.get("global_dispatch_isa_top_pcs") or [None])[0]
                    if top_pc:
                        lines.append(
                            "        global_isa_top_pc="
                            f"0x{int(top_pc.get('pc', 0)):x} "
                            f"count={int(top_pc.get('count', 0))} "
                            f"{top_pc.get('mnemonic')} {top_pc.get('operands')}".rstrip()
                        )
        hotspot_profiles = stream.get("runtime_hotspot_profiles") or []
        if hotspot_profiles:
            top = hotspot_profiles[0]
            symbol = top.get("symbol") or {}
            symbol_text = ""
            if symbol.get("name"):
                symbol_text = f" symbol={symbol.get('name')}+0x{int(symbol.get('offset', 0) or 0):x}"
            avg_stall = top.get("avg_stall_per_hit")
            avg_duration = top.get("avg_duration_per_hit")
            stall_share = top.get("stall_share_of_duration")
            lines.append(
                "    hotspot_profile="
                f"address=0x{int(top.get('address', 0) or 0):x}"
                f" duration={int(top.get('total_duration', 0) or 0)}"
                f" stall={int(top.get('total_stall', 0) or 0)}"
                f" hitcount={int(top.get('hitcount', 0) or 0)}"
                + (f" avg_duration={avg_duration:.2f}" if isinstance(avg_duration, (int, float)) else "")
                + (f" avg_stall={avg_stall:.2f}" if isinstance(avg_stall, (int, float)) else "")
                + (f" stall_share={stall_share:.2f}" if isinstance(stall_share, (int, float)) else "")
                + symbol_text
            )
    return "\n".join(lines)
