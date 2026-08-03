#!/usr/bin/env python3
"""Focused tests for the schema-v3 MoE runtime timing analyzer."""

from __future__ import annotations

import contextlib
import io
import unittest

import analyze_moe_runtime_timing as timing


def make_record(
    node: int,
    cluster: int,
    stage: int,
    start: int,
    total: int,
    *,
    slot: int = 0,
    expert: int = 1,
    ntokens: int = 8,
) -> dict[str, int | None]:
    meta = stage | (cluster << 12)
    task = slot | (expert << 6) | (ntokens << 14)
    return timing.decode_common(
        node=node,
        meta=meta,
        task=task,
        start=start,
        total=total,
        resource_offset=0,
        resource_cycles=0,
        peer_wait=0,
        units=0,
        flags=0,
        result=0,
    )


def make_cluster(
    cluster: int, epoch: int, duration: int, first_node: int
) -> list[dict[str, int | None]]:
    records = [
        make_record(
            first_node,
            cluster,
            timing.SCOPE_BEGIN,
            (epoch - 3) & timing.UINT32_MASK,
            3,
        )
    ]
    for offset, stage in enumerate((1, 2, 11, 3, 4, 5, 6, 12, 7, 8, 9, 10), 1):
        records.append(
            make_record(
                first_node + offset,
                cluster,
                stage,
                (epoch + offset) & timing.UINT32_MASK,
                1,
            )
        )
    records.append(
        make_record(
            first_node + 13,
            cluster,
            timing.SCOPE_END,
            (epoch + duration) & timing.UINT32_MASK,
            2,
        )
    )
    return records


class TimingAnalyzerTest(unittest.TestCase):
    def test_timing_level_is_parsed_from_begin_marker(self) -> None:
        match = timing.LEVEL_RE.search(
            "[MOE_TIMING_BEGIN] version=3 level=1"
        )
        self.assertIsNotNone(match)
        self.assertEqual("1", match[1])

    def test_v3_end_marker_reports_printed_and_expected_counts(self) -> None:
        match = timing.END_RE.search(
            "[MOE_TIMING_END] records=479 expected=479"
        )
        self.assertIsNotNone(match)
        self.assertEqual("479", match[1])
        self.assertEqual("479", match[2])

    def test_global_uses_max_cluster_local_span_with_wrap(self) -> None:
        records = make_cluster(2, 0xFFFFFFF8, 56, 100)
        records += make_cluster(3, 1000, 45, 200)

        scopes, errors = timing.find_scopes(records)
        errors += timing.add_scope_offsets(records, scopes)

        self.assertEqual([], errors)
        self.assertEqual(56, scopes[2]["duration"])
        self.assertEqual(45, scopes[3]["duration"])
        self.assertEqual(56, max(scope["duration"] for scope in scopes.values()))

    def test_level_one_accepts_only_four_scope_records(self) -> None:
        records = [
            make_record(1, 2, timing.SCOPE_BEGIN, 100, 2),
            make_record(2, 2, timing.SCOPE_END, 202, 2),
            make_record(3, 3, timing.SCOPE_BEGIN, 500, 2),
            make_record(4, 3, timing.SCOPE_END, 592, 2),
        ]
        self.assertEqual([], timing.validate_timing_level(records, 1))
        scopes, errors = timing.find_scopes(records)
        self.assertEqual([], errors)
        self.assertEqual(100, scopes[2]["duration"])
        self.assertEqual(90, scopes[3]["duration"])
        self.assertEqual({}, timing.active_slots(records, scopes))

    def test_level_one_rejects_detail_records(self) -> None:
        records = [make_record(1, 2, 2, 100, 2)]
        self.assertEqual(
            ["level 1 capture contains 1 non-scope records"],
            timing.validate_timing_level(records, 1),
        )

    def test_all_slot_stages_and_task_offsets_are_complete(self) -> None:
        records = make_cluster(2, 500, 80, 100)
        records += make_cluster(3, 9000, 70, 200)
        scopes, errors = timing.find_scopes(records)
        errors += timing.add_scope_offsets(records, scopes)
        slots = timing.active_slots(records, scopes)
        errors += timing.validate_slots(slots)

        self.assertEqual([], errors)
        self.assertEqual({(2, 0), (3, 0)}, set(slots))
        self.assertEqual(1, min(record["start_offset"] for record in slots[(2, 0)]))

    def test_valid_report_does_not_infer_gap_or_idle(self) -> None:
        records = make_cluster(2, 100, 80, 100)
        records += make_cluster(3, 5000, 70, 200)
        scopes, errors = timing.find_scopes(records)
        errors += timing.add_scope_offsets(records, scopes)
        slots = timing.active_slots(records, scopes)
        errors += timing.validate_slots(slots)
        self.assertEqual([], errors)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            timing.print_global(scopes, 1.0)
            timing.print_slot_and_stage_report(slots, 1.0)

        report = output.getvalue().lower()
        self.assertNotIn("gap", report)
        self.assertNotIn("idle", report)
        self.assertIn("global=max(c-local) c2 cycles=80", report)


if __name__ == "__main__":
    unittest.main()
