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
DEFAULT_PARAMS = Path(__file__).with_name("params.hjson")


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


def load_model_dimensions(params_path: Path) -> tuple[int, int]:
    required = ("hidden_size", "intermediate_size")
    field_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*(\d+)\b")
    values: dict[str, int] = {}
    for line in params_path.read_text().splitlines():
        match = field_re.match(line)
        if match and match[1] in required:
            values[match[1]] = int(match[2])
    missing = [name for name in required if name not in values]
    if missing:
        raise SystemExit(
            f"{params_path} missing compute-efficiency fields: {', '.join(missing)}"
        )
    return values["hidden_size"], values["intermediate_size"]


def print_report(
    records: list[dict[str, int]],
    phases: list[tuple[str, int, int, int]],
    details: bool,
    hidden_size: int,
    intermediate_size: int,
    peak_mac_per_cluster_cc: float,
    individual_cluster_count: int,
) -> None:
    if phases:
        print("HOST PHASE MCYCLE")
        for name, count, last, total in phases:
            print(f"  {name:16s} count={count:3d} last={last:10d} total={total:10d}")

    # Inactive static slots commit task metadata as zero. Grouping those records
    # by the encoded slot would fold every inactive node into C2/C3 slot 0 and
    # incorrectly extend the active slot and cluster windows to the end of the
    # whole static DFG.
    active_records = [record for record in records if record["ntokens"] > 0]
    inactive_records = len(records) - len(active_records)
    if inactive_records:
        print(
            f"\nINACTIVE STATIC RECORDS ignored={inactive_records} "
            f"active={len(active_records)}"
        )

    slots: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    clusters: dict[int, list[dict[str, int]]] = defaultdict(list)
    resources: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for record in active_records:
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
        vc_hw = sum(r["units"] for r in group if r["resource"] == 4)
        wait = sum(r["peer_wait"] for r in group)
        dma_bytes = sum(r["units"] for r in group if r["resource"] in (1, 2, 3))
        first = group[0]
        print(
            f"  C{key[0]} slot={key[1]:2d} eid={first['expert']:2d} ntok={first['ntokens']:3d} "
            f"window={delta(end, start):9d} dma_api={dma:9d} vc_api={compute:9d} "
            f"vc_hw={vc_hw:9d} peer_wait={wait:7d} bytes={dma_bytes}"
        )

    active_slots = slots
    mac_per_pair = 3 * hidden_size * intermediate_size
    active_clusters = sorted({key[0] for key in active_slots})
    if active_slots and active_clusters:
        first_slot_start = min(
            min(record["start"] for record in group)
            for group in active_slots.values()
        )
        last_slot_end = max(
            max(record["end"] for record in group)
            for group in active_slots.values()
        )
        timespan = delta(last_slot_end, first_slot_start)
        routed_token_expert_pairs = sum(
            max(record["ntokens"] for record in group)
            for group in active_slots.values()
        )
        useful_mac = routed_token_expert_pairs * mac_per_pair
        total_peak = individual_cluster_count * peak_mac_per_cluster_cc
        ideal_cycles = useful_mac / total_peak
        efficiency = 100.0 * ideal_cycles / timespan if timespan else 0.0

        print("\nINDIVIDUAL EXPERT COMPUTE EFFICIENCY")
        print(
            "  definition=ideal_compute_cycles/first_active_slot_to_last_active_slot"
        )
        print(
            f"  active_slots={len(active_slots)} clusters={active_clusters} "
            f"routed_token_expert_pairs={routed_token_expert_pairs}"
        )
        print(
            f"  useful_mac={useful_mac} "
            f"({routed_token_expert_pairs} * 3 * {hidden_size} * {intermediate_size})"
        )
        print(
            f"  peak={individual_cluster_count} * {peak_mac_per_cluster_cc:g} "
            f"= {total_peak:g} MAC/cc ideal={ideal_cycles:.0f} cc"
        )
        print(
            f"  first_slot_start={first_slot_start} last_slot_end={last_slot_end} "
            f"timespan={timespan} cc"
        )
        print(f"  compute_efficiency={efficiency:.2f}%")
        print("  note=vc_api/vc_hw busy cycles are diagnostics, not the numerator")

    cluster_windows: dict[int, tuple[int, int]] = {}
    print("\nCLUSTER WINDOWS")
    for cluster in sorted(clusters):
        group = clusters[cluster]
        start, end = min(r["start"] for r in group), max(r["end"] for r in group)
        cluster_windows[cluster] = (start, end)
        cluster_slot_ids = sorted({r["slot"] for r in group})
        routed_pairs = sum(
            max(r["ntokens"] for r in group if r["slot"] == slot)
            for slot in cluster_slot_ids
        )
        ideal_cycles = routed_pairs * mac_per_pair / peak_mac_per_cluster_cc
        window = delta(end, start)
        efficiency = 100.0 * ideal_cycles / window if window else 0.0
        print(
            f"  C{cluster} slots={len(cluster_slot_ids):2d} pairs={routed_pairs:3d} "
            f"records={len(group):3d} start={start} end={end} window={delta(end, start)}"
            f" ideal={ideal_cycles:.0f} efficiency={efficiency:.2f}%"
        )

    print("\nINTER-SLOT GAPS")
    for cluster in sorted(clusters):
        group = clusters[cluster]
        windows = []
        for slot in sorted({r["slot"] for r in group}):
            slot_group = [r for r in group if r["slot"] == slot]
            windows.append(
                (
                    min(r["start"] for r in slot_group),
                    max(r["end"] for r in slot_group),
                )
            )
        slot_window_sum = sum(delta(end, start) for start, end in windows)
        inter_slot_gap = sum(
            max(0, windows[idx][0] - windows[idx - 1][1])
            for idx in range(1, len(windows))
        )
        cluster_start, cluster_end = cluster_windows[cluster]
        cluster_window = delta(cluster_end, cluster_start)
        gap_share = 100.0 * inter_slot_gap / cluster_window if cluster_window else 0.0
        print(
            f"  C{cluster} slot_windows={slot_window_sum:9d} "
            f"inter_slot_gap={inter_slot_gap:9d} gap_share={gap_share:6.2f}%"
        )

    print("\nRESOURCE UTILIZATION IN CLUSTER WINDOW")
    for (cluster, resource), intervals in sorted(resources.items()):
        _, api_busy, _ = union_cycles(intervals)
        cluster_start, cluster_end = cluster_windows[cluster]
        window = delta(cluster_end, cluster_start)
        if resource == "versacore":
            hw_busy = sum(
                r["units"]
                for r in clusters[cluster]
                if r["resource"] == 4
            )
            busy = hw_busy if hw_busy else api_busy
            source = "hw" if hw_busy else "api-fallback"
            idle = max(0, window - busy)
            utilization = 100.0 * busy / window if window else 0.0
            print(
                f"  C{cluster} {resource:10s} records={len(intervals):3d} "
                f"cluster_window={window:9d} api_busy={api_busy:9d} "
                f"hw_busy={hw_busy:9d} idle={idle:9d} util={utilization:6.2f}% "
                f"source={source}"
            )
        else:
            idle = max(0, window - api_busy)
            utilization = 100.0 * api_busy / window if window else 0.0
            print(
                f"  C{cluster} {resource:10s} records={len(intervals):3d} "
                f"cluster_window={window:9d} busy={api_busy:9d} idle={idle:9d} "
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
                if r["resource"] in (1, 2, 3):
                    units = f"bytes={r['units']}"
                elif r["resource"] == 4:
                    units = f"vc_hw_cycles={r['units']}"
                else:
                    units = f"units={r['units']}"
                print(
                    f"    {stage:12s} block={r['block']:2d} core={r['core']} "
                    f"start={r['start']} end={r['end']} resource={resource:10s} "
                    f"{resource_window} peer_wait={r['peer_wait']} "
                    f"{units} status={format_flags(r['flags'])} "
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
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS,
        help="workload params.hjson used for hidden/intermediate dimensions",
    )
    parser.add_argument(
        "--peak-mac-per-cluster-cc",
        type=float,
        default=512.0,
        help="theoretical peak of one individual dual-VersaCore cluster",
    )
    parser.add_argument(
        "--individual-cluster-count",
        type=int,
        default=2,
        help="number of available individual expert clusters (default: C2 and C3)",
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
    hidden_size, intermediate_size = load_model_dimensions(args.params)
    print_report(
        records,
        phases,
        args.details,
        hidden_size,
        intermediate_size,
        args.peak_mac_per_cluster_cc,
        args.individual_cluster_count,
    )


if __name__ == "__main__":
    main()
