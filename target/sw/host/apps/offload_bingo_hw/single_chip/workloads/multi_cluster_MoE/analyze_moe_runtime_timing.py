#!/usr/bin/env python3
"""Analyze compact end-of-DFG MoE timing records from a UART log."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


RECORD_RE = re.compile(
    r"\[MOE_TIMING_RECORD\]\s+"
    r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([0-9a-fA-F]+)\s+(\d+)"
)
VERSION_RE = re.compile(r"\[MOE_TIMING_BEGIN\]\s+version=(\d+)")
END_RE = re.compile(r"\[MOE_TIMING_END\]\s+records=(\d+)")
PHASE_RE = re.compile(
    r"\[MOE_PHASE_MCYCLE\]\s+(\S+)\s+count=(\d+)\s+last=(\d+)\s+total=(\d+)"
)

STAGE_NAMES = {
    1: "gather_s1",
    2: "load_s1",
    3: "compute_s1",
    4: "prefetch_s2",
    5: "compute_s2",
    6: "load_s3",
    7: "compute_s3",
    8: "prefetch_s4",
    9: "compute_s4",
    10: "store",
}
RESOURCE_NAMES = {0: "none", 1: "idma", 2: "xdma", 3: "dma_both", 4: "versacore"}
FLAG_NAMES = {
    0: "active",
    1: "skipped",
    2: "ctrl_skip",
    3: "invalid_call",
    4: "no_prefetch",
    5: "no_store",
}


def delta(end: int, start: int) -> int:
    return (end - start) & 0xFFFFFFFF


def decode_record(match: re.Match[str], version: int) -> dict[str, int]:
    meta, task = int(match[1], 16), int(match[2], 16)
    start = int(match[3])
    if version == 2:
        total = int(match[4])
        resource_offset = int(match[5])
        body = int(match[6])
        end = (start + total) & 0xFFFFFFFF
        resource_start = (start + resource_offset) & 0xFFFFFFFF
        resource_end = (resource_start + body) & 0xFFFFFFFF
    elif version == 1:
        end = int(match[4])
        resource_start = int(match[5])
        resource_end = int(match[6])
        total = delta(end, start)
        body = delta(resource_end, resource_start)
    else:
        raise SystemExit(f"unsupported MOE timing schema version {version}")

    record = {
        "stage": meta & 0xFF,
        "resource": (meta >> 8) & 0xF,
        "cluster": (meta >> 12) & 0xF,
        "core": (meta >> 16) & 0xF,
        "block": (meta >> 20) & 0xFFF,
        "slot": task & 0x3F,
        "expert": (task >> 6) & 0xFF,
        "ntokens": (task >> 14) & 0x3FF,
        "start": start,
        "end": end,
        "resource_start": resource_start,
        "resource_end": resource_end,
        "peer_wait": int(match[7]),
        "units": int(match[8]),
        "flags": int(match[9], 16),
        "result": int(match[10]),
    }
    record["total"] = total
    record["body"] = body
    pre = delta(record["resource_start"], record["start"])
    record["setup"] = max(0, pre - record["peer_wait"])
    record["post"] = delta(record["end"], record["resource_end"])
    return record


def union_cycles(intervals: list[tuple[int, int]]) -> tuple[int, int, int]:
    if not intervals:
        return 0, 0, 0
    intervals.sort()
    first, current_end = intervals[0]
    last = current_end
    busy = 0
    for start, end in intervals[1:]:
        if start > current_end:
            busy += current_end - first
            first, current_end = start, end
        elif end > current_end:
            current_end = end
        if end > last:
            last = end
    busy += current_end - first
    window_start = intervals[0][0]
    return last - window_start, busy, last - window_start - busy


def format_flags(flags: int) -> str:
    names = [name for bit, name in FLAG_NAMES.items() if flags & (1 << bit)]
    return ",".join(names) if names else "none"


def print_report(records: list[dict[str, int]], phases: list[tuple[str, int, int, int]], details: bool) -> None:
    if phases:
        print("HOST PHASE MCYCLE")
        for name, count, last, total in phases:
            print(f"  {name:16s} count={count:3d} last={last:10d} total={total:10d}")

    slots: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    clusters: dict[int, list[dict[str, int]]] = defaultdict(list)
    resources: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        slots[(record["cluster"], record["slot"])].append(record)
        clusters[record["cluster"]].append(record)
        resource = record["resource"]
        if resource in (1, 3):
            resources[(record["cluster"], "idma")].append(
                (record["resource_start"], record["resource_end"])
            )
        if resource in (2, 3):
            resources[(record["cluster"], "xdma")].append(
                (record["resource_start"], record["resource_end"])
            )
        if resource == 4:
            resources[(record["cluster"], "versacore")].append(
                (record["resource_start"], record["resource_end"])
            )

    print("\nSLOT WINDOWS")
    for key in sorted(slots):
        group = slots[key]
        start, end = min(r["start"] for r in group), max(r["end"] for r in group)
        dma = sum(r["body"] for r in group if r["resource"] in (1, 2, 3))
        compute = sum(r["body"] for r in group if r["resource"] == 4)
        wait = sum(r["peer_wait"] for r in group)
        dma_bytes = sum(r["units"] for r in group if r["resource"] in (1, 2, 3))
        first = group[0]
        print(
            f"  C{key[0]} slot={key[1]:2d} eid={first['expert']:2d} ntok={first['ntokens']:3d} "
            f"window={delta(end, start):9d} dma_api={dma:9d} vc_api={compute:9d} "
            f"peer_wait={wait:7d} bytes={dma_bytes}"
        )

    cluster_windows: dict[int, tuple[int, int]] = {}
    print("\nCLUSTER WINDOWS")
    for cluster in sorted(clusters):
        group = clusters[cluster]
        start, end = min(r["start"] for r in group), max(r["end"] for r in group)
        cluster_windows[cluster] = (start, end)
        print(
            f"  C{cluster} slots={len({r['slot'] for r in group}):2d} "
            f"records={len(group):3d} start={start} end={end} window={delta(end, start)}"
        )

    print("\nRESOURCE UTILIZATION IN CLUSTER WINDOW")
    for (cluster, resource), intervals in sorted(resources.items()):
        _, busy, _ = union_cycles(intervals)
        cluster_start, cluster_end = cluster_windows[cluster]
        window = delta(cluster_end, cluster_start)
        idle = window - busy
        utilization = 100.0 * busy / window if window else 0.0
        print(
            f"  C{cluster} {resource:10s} records={len(intervals):3d} "
            f"cluster_window={window:9d} busy={busy:9d} idle={idle:9d} "
            f"util={utilization:6.2f}%"
        )

    if details:
        print("\nSLOT NODE TIMELINES")
        for key in sorted(slots):
            group = sorted(slots[key], key=lambda item: (item["start"], item["stage"]))
            slot_start = min(r["start"] for r in group)
            slot_end = max(r["end"] for r in group)
            first = group[0]
            print(
                f"  C{key[0]} slot={key[1]} eid={first['expert']} "
                f"ntok={first['ntokens']} start={slot_start} end={slot_end}"
            )
            for r in group:
                stage = STAGE_NAMES.get(r["stage"], "unknown")
                resource = RESOURCE_NAMES.get(r["resource"], "unknown")
                if r["body"]:
                    resource_window = (
                        f"resource_start={r['resource_start']} "
                        f"resource_end={r['resource_end']}"
                    )
                else:
                    resource_window = "resource_start=- resource_end=-"
                print(
                    f"    {stage:12s} block={r['block']:2d} core={r['core']} "
                    f"start={r['start']} end={r['end']} resource={resource:10s} "
                    f"{resource_window} peer_wait={r['peer_wait']} "
                    f"units={r['units']} status={format_flags(r['flags'])} "
                    f"rc={r['result']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="UART or simulator log")
    parser.add_argument(
        "--details",
        action="store_true",
        help="print every slot node with node and resource start/end timestamps",
    )
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    version_match = VERSION_RE.search(text)
    if version_match is None:
        raise SystemExit("no [MOE_TIMING_BEGIN] schema version found")
    version = int(version_match[1])
    records = [decode_record(match, version) for match in RECORD_RE.finditer(text)]
    phases = [
        (match[1], int(match[2]), int(match[3]), int(match[4]))
        for match in PHASE_RE.finditer(text)
    ]
    if not records:
        raise SystemExit("no [MOE_TIMING_RECORD] entries found")
    end_match = END_RE.search(text)
    expected = int(end_match[1]) if end_match else len(records)
    print(f"TIMING RECORDS schema=v{version} parsed={len(records)} expected={expected}")
    if len(records) != expected:
        print("  WARNING: incomplete records detected; the capture likely truncated long UART lines")
    print_report(records, phases, args.details)


if __name__ == "__main__":
    main()
