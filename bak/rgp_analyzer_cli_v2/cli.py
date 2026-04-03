from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare_v1_v2
from .models import DecodeOptions, DispatchIsaOptions, TriageOptions
from .render import (
    render_decode_payload,
    render_dispatch_payload,
    render_inspect,
    render_resource_payload,
    render_stitch_report,
    render_triage_payload,
)
from .services import (
    decode_payload,
    dispatch_isa_payload,
    inspect_payload,
    load_capture,
    resource_payload,
    stitch_payload,
    triage_payload,
)


def _json_output(payload: object) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rgp-analyzer-v2", description="Layered v2 CLI for RGP analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_capture(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("rgp_file", type=Path)
        cmd.add_argument("--json", action="store_true", dest="as_json")

    inspect_parser = subparsers.add_parser("inspect")
    add_capture(inspect_parser)
    inspect_parser.add_argument("--limit", type=int, default=10)

    stitch_parser = subparsers.add_parser("stitch-report")
    add_capture(stitch_parser)
    stitch_parser.add_argument("--limit", type=int, default=10)

    decode_parser = subparsers.add_parser("decode-sqtt")
    add_capture(decode_parser)
    decode_parser.add_argument("--build-helper", action="store_true")
    decode_parser.add_argument("--helper", type=Path, default=None)
    decode_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    decode_parser.add_argument("--stream-limit", type=int, default=0)
    decode_parser.add_argument("--hotspot-limit", type=int, default=8)

    resource_parser = subparsers.add_parser("resource-summary")
    add_capture(resource_parser)
    resource_parser.add_argument("--limit", type=int, default=10)
    resource_parser.add_argument("--tool", default=None)

    dispatch_parser = subparsers.add_parser("dispatch-isa-map")
    add_capture(dispatch_parser)
    dispatch_parser.add_argument("--stream-index", type=int, default=0)
    dispatch_parser.add_argument("--dispatch-limit", type=int, default=8)
    dispatch_parser.add_argument("--context-packets", type=int, default=64)
    dispatch_parser.add_argument("--tail-packets", type=int, default=512)
    dispatch_parser.add_argument("--packet-limit", type=int, default=20000)
    dispatch_parser.add_argument("--mapped-limit", type=int, default=32)
    dispatch_parser.add_argument("--tool", default=None)
    dispatch_parser.add_argument("--limit", type=int, default=16)

    triage_parser = subparsers.add_parser("shader-triage")
    add_capture(triage_parser)
    triage_parser.add_argument("--build-helper", action="store_true")
    triage_parser.add_argument("--helper", type=Path, default=None)
    triage_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    triage_parser.add_argument("--isa-tool", default=None)
    triage_parser.add_argument("--readelf-tool", default=None)
    triage_parser.add_argument("--limit", type=int, default=10)
    triage_parser.add_argument("--hotspot-limit", type=int, default=8)

    compare_parser = subparsers.add_parser("compare-v1-v2")
    compare_parser.add_argument("rgp_file", type=Path)
    compare_parser.add_argument("--json", action="store_true", dest="as_json")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "compare-v1-v2":
        payload = compare_v1_v2(args.rgp_file)
        if args.as_json:
            return _json_output(payload)
        print("compare_v1_v2:")
        print(f"  capture: {payload['capture']}")
        print(f"  all_match: {payload['all_match']}")
        for name, ok in payload["checks"].items():
            print(f"  {name}: {ok}")
        return 0

    session = load_capture(args.rgp_file)

    if args.command == "inspect":
        payload = inspect_payload(session)
        if args.as_json:
            return _json_output(payload)
        print(render_inspect(session.report, limit=args.limit))
        return 0

    if args.command == "stitch-report":
        payload = stitch_payload(session)
        if args.as_json:
            return _json_output(payload)
        print(render_stitch_report(session.report, limit=args.limit))
        return 0

    if args.command == "decode-sqtt":
        payload = decode_payload(
            session,
            DecodeOptions(
                stream_limit=args.stream_limit,
                hotspot_limit=args.hotspot_limit,
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_decode_payload(payload))
        return 0

    if args.command == "resource-summary":
        payload = resource_payload(session, limit=args.limit, tool=args.tool)
        if args.as_json:
            return _json_output(payload)
        print(render_resource_payload(payload))
        return 0

    if args.command == "dispatch-isa-map":
        payload = dispatch_isa_payload(
            session,
            DispatchIsaOptions(
                stream_index=args.stream_index,
                dispatch_limit=args.dispatch_limit,
                context_packets=args.context_packets,
                tail_packets=args.tail_packets,
                packet_limit=args.packet_limit,
                mapped_limit=args.mapped_limit,
                tool=args.tool,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_dispatch_payload(payload, limit=args.limit))
        return 0

    if args.command == "shader-triage":
        payload = triage_payload(
            session,
            TriageOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_triage_payload(payload))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
