from __future__ import annotations

from typing import Any

from .resource_metadata import extract_resource_metadata
from .rgp_records import (
    collect_code_object_records,
    collect_loader_load_records,
    collect_pso_records,
    hash_pair,
    hash_text,
)


def build_stitch_entries(report: dict[str, Any]) -> dict[str, Any]:
    code_records = collect_code_object_records(report)
    loader_records = collect_loader_load_records(report)
    pso_records = collect_pso_records(report)
    resource_metadata = extract_resource_metadata(report, limit=10000)
    metadata_by_index = {int(item["index"]): item for item in resource_metadata}

    pso_by_api_hash = {int(item["api_pso_hash"]): item for item in pso_records}
    pso_by_pipeline_hash = {
        hash_pair(item["pipeline_hash"]): item for item in pso_records if hash_pair(item["pipeline_hash"]) is not None
    }
    loader_by_hash: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in loader_records:
        key = hash_pair(record["code_object_hash"])
        if key is not None:
            loader_by_hash.setdefault(key, []).append(record)

    warnings: list[str] = []
    if not code_records:
        warnings.append("No code object records were found in the capture.")
    if not loader_records:
        warnings.append("No LOAD_TO_GPU_MEMORY loader events were found in the capture.")
    if not pso_records:
        warnings.append("No PSO correlation records were found in the capture.")

    entries: list[dict[str, Any]] = []
    resolved_count = 0
    for bridge_index, record in enumerate(code_records):
        metadata = metadata_by_index.get(int(record["index"]), {})
        internal_pipeline_hash = hash_pair(metadata.get("internal_pipeline_hash"))
        pso = None
        match_steps: list[str] = []
        unresolved_reasons: list[str] = []

        if internal_pipeline_hash is not None:
            pso = pso_by_api_hash.get(internal_pipeline_hash[0])
            if pso is not None:
                match_steps.append("metadata.internal_pipeline_hash -> pso.api_pso_hash")
            else:
                pso = pso_by_pipeline_hash.get(internal_pipeline_hash)
                if pso is not None:
                    match_steps.append("metadata.internal_pipeline_hash -> pso.pipeline_hash")
                else:
                    unresolved_reasons.append("no_pso_for_internal_pipeline_hash")
        else:
            unresolved_reasons.append("missing_internal_pipeline_hash")

        resolved_pipeline_hash = hash_pair(pso["pipeline_hash"]) if pso else internal_pipeline_hash
        loader_matches = loader_by_hash.get(resolved_pipeline_hash, []) if resolved_pipeline_hash is not None else []
        if loader_matches:
            match_steps.append("pipeline_hash -> loader.code_object_hash")
        else:
            unresolved_reasons.append("no_loader_for_pipeline_hash")
        if len(loader_matches) > 1:
            unresolved_reasons.append("multiple_loader_matches")

        resolved_loader = loader_matches[0] if len(loader_matches) == 1 else None
        if resolved_loader is not None:
            resolved_count += 1

        entries.append(
            {
                "bridge_index": bridge_index,
                "match_kind": "co_pso_loader" if resolved_loader is not None else "partial_co_pso_loader",
                "resolution_steps": match_steps,
                "unresolved_reasons": unresolved_reasons,
                "resolved": resolved_loader is not None,
                "code_object_index": int(record["index"]),
                "payload_size": int(record["payload_size"]),
                "load_id": bridge_index + 1,
                "load_addr": int(resolved_loader["base_address"]) if resolved_loader else 0,
                "load_size": int(record["payload_size"]),
                "entry_point": metadata.get("entry_point"),
                "internal_pipeline_hash": list(internal_pipeline_hash) if internal_pipeline_hash else None,
                "internal_pipeline_hash_text": hash_text(internal_pipeline_hash),
                "api_shader_hash": metadata.get("api_shader_hash"),
                "pso_record_index": pso.get("index") if pso else None,
                "api_pso_hash": int(pso["api_pso_hash"]) if pso else (internal_pipeline_hash[0] if internal_pipeline_hash else None),
                "pipeline_hash": list(resolved_pipeline_hash) if resolved_pipeline_hash else None,
                "pipeline_hash_text": hash_text(resolved_pipeline_hash),
                "api_level_obj_name": pso.get("api_level_obj_name") if pso else None,
                "loader_match_count": len(loader_matches),
                "loader_records": [
                    {
                        "index": int(item["index"]),
                        "base_address": int(item["base_address"]),
                        "time_stamp": int(item["time_stamp"]),
                        "code_object_hash": list(item["code_object_hash"]),
                    }
                    for item in loader_matches
                ],
                "loader_event_index": resolved_loader.get("index") if resolved_loader else None,
                "loader_event_hash": resolved_loader.get("code_object_hash") if resolved_loader else None,
                "matched_loader_event": resolved_loader is not None,
                "valid_elf": bool(record.get("elf", {}).get("valid_elf")),
            }
        )

    return {
        "code_records": code_records,
        "loader_records": loader_records,
        "pso_records": pso_records,
        "resource_metadata": resource_metadata,
        "warnings": warnings,
        "entries": entries,
        "resolved_entry_count": resolved_count,
    }
