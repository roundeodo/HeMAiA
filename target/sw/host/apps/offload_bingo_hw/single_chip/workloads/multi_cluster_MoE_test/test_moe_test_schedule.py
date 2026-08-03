#!/usr/bin/env python3
"""Focused schedule-contract tests for the FPGA showcase profiles."""

import unittest

import moe_test_schedule as schedule


class MoeTestScheduleTest(unittest.TestCase):
    def test_s2pf_both_profile_is_available(self) -> None:
        self.assertIn(schedule.S2PF_BOTH_PROFILE, schedule.SCHEDULE_PROFILES)

    def test_s2pf_both_changes_only_c0_slot0(self) -> None:
        baseline = {
            ("c0", 0): schedule.DMA_IDMA,
            ("c1", 0): schedule.DMA_NONE,
            ("c0", 1): schedule.DMA_IDMA,
            ("c1", 1): schedule.DMA_XDMA,
        }
        selected = {
            key: schedule.select_two_slot_s2pf_binding(
                schedule.S2PF_BOTH_PROFILE, *key, binding
            )
            for key, binding in baseline.items()
        }
        self.assertEqual(schedule.DMA_BOTH, selected[("c0", 0)])
        self.assertEqual(schedule.DMA_NONE, selected[("c1", 0)])
        self.assertEqual(schedule.DMA_IDMA, selected[("c0", 1)])
        self.assertEqual(schedule.DMA_XDMA, selected[("c1", 1)])

    def test_static_desc_matches_case0_policy0(self) -> None:
        queues = schedule.build_static_desc_schedule()
        audit = schedule.audit_static_desc_schedule(queues)

        self.assertEqual({"c0": 159, "c1": 162}, audit["queue_ticks"])
        self.assertEqual({"c0": 22, "c1": 21}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual((), audit["dma_release_edges"])

    def test_static_desc_has_only_fixed_physical_parameters(self) -> None:
        queues = schedule.build_static_desc_schedule()
        for cluster_name, slots in queues.items():
            expected_dma = (
                schedule.DMA_IDMA
                if cluster_name == "c0"
                else schedule.DMA_XDMA
            )
            for slot in slots:
                self.assertEqual(schedule.SHAPE_B, slot.s1_shape)
                self.assertEqual(schedule.SHAPE_B, slot.s3_shape)
                self.assertEqual(expected_dma, slot.s1_dma)
                self.assertEqual(expected_dma, slot.s3_dma)
                self.assertEqual(schedule.DMA_NONE, slot.s2_prefetch_dma)
                self.assertEqual(schedule.DMA_NONE, slot.s4_prefetch_dma)

    def test_static_desc_uses_exported_token_routing(self) -> None:
        routing = schedule.STATIC_DESC_TOKEN_IDS_BY_EXPERT
        self.assertEqual((0, 1, 2, 3, 4), routing[0][:5])
        self.assertEqual((24, 35, 50), routing[3])
        self.assertEqual((69,), routing[42])
        owners = [[] for _ in range(70)]
        for expert_id, token_ids in enumerate(routing):
            for token_id in token_ids:
                owners[token_id].append(expert_id)
        self.assertTrue(all(len(token_owners) == 2 for token_owners in owners))

    def test_static_desc_structural_lower_bound_is_177_75_ticks(self) -> None:
        queues = schedule.build_static_desc_schedule()
        audit = schedule.audit_static_desc_schedule(queues)

        self.assertEqual(
            {"c0": 702, "c1": 711},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            177.75,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_dynamic_desc_matches_case0_policy1(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        audit = schedule.audit_dynamic_desc_schedule(queues)

        self.assertEqual({"c0": 159, "c1": 159}, audit["queue_ticks"])
        self.assertEqual({"c0": 22, "c1": 21}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(21, len(audit["dma_release_edges"]))

    def test_dynamic_desc_preserves_preload_contract(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            (schedule.DMA_XDMA, 2),
            (slots[1].s4_prefetch_dma, slots[1].s4_prefetch_target_eid),
        )
        self.assertEqual(
            (schedule.DMA_BOTH, 4),
            (slots[3].s4_prefetch_dma, slots[3].s4_prefetch_target_eid),
        )
        self.assertTrue(slots[2].skip_s1)
        self.assertTrue(slots[4].skip_s1)
        self.assertEqual(schedule.DMA_NONE, slots[2].s2_prefetch_dma)
        self.assertEqual(schedule.DMA_BOTH, slots[2].s3_dma)
        self.assertEqual(
            (
                schedule.SHAPE_C,
                schedule.SHAPE_C,
                schedule.DMA_BOTH,
                schedule.DMA_BOTH,
            ),
            (
                slots[22].s1_shape,
                slots[22].s3_shape,
                slots[22].s1_dma,
                slots[22].s3_dma,
            ),
        )

    def test_dynamic_desc_s4pf_is_in_dma_release_graph(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        edges = set(schedule.cross_cluster_dma_release_edges(queues))

        self.assertIn(
            (("c1", 0, 1, "S4PF"), ("c0", 1, 3, "S1")),
            edges,
        )
        self.assertIn(
            (("c0", 1, 3, "S4PF"), ("c1", 1, 2, "S3")),
            edges,
        )

    def test_dynamic_desc_structural_lower_bound_is_175_5_ticks(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        audit = schedule.audit_dynamic_desc_schedule(queues)

        self.assertEqual(
            {"c0": 702, "c1": 699},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            175.5,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_existing_descending_profile_is_unchanged(self) -> None:
        queues = schedule.build_high_to_low_schedule()
        audit = schedule.audit_high_to_low_schedule(queues)

        self.assertEqual({"c0": 163, "c1": 160}, audit["queue_ticks"])
        self.assertEqual(43, audit["active_experts"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(
            179.5,
            schedule.HIGH_TO_LOW_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS / 4,
        )

    def test_ascending_profile_matches_handoff_shape(self) -> None:
        queues = schedule.build_low_to_high_schedule()
        audit = schedule.audit_low_to_high_schedule(queues)

        self.assertEqual({"c0": 165, "c1": 165}, audit["queue_ticks"])
        self.assertEqual(43, audit["active_experts"])
        self.assertEqual(44, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(22, len(queues["c0"]))
        self.assertEqual(22, len(queues["c1"]))

    def test_ascending_e0_split_is_disjoint_and_complete(self) -> None:
        queues = schedule.build_low_to_high_schedule()
        slices = sorted(
            (slot.token_start_rank, slot.ntokens, slot.cluster_name)
            for slots in queues.values()
            for slot in slots
            if slot.expert_id == 0
        )
        self.assertEqual([(0, 11, "c0"), (11, 11, "c1")], slices)

    def test_ascending_e2_preserves_explicit_dma_bindings(self) -> None:
        queues = schedule.build_low_to_high_schedule()
        e2 = next(slot for slot in queues["c1"] if slot.expert_id == 2)
        timeline = schedule._task_timeline(e2, e2.reference_start_tick)

        self.assertEqual(schedule.SHAPE_C, e2.s1_shape)
        self.assertEqual(schedule.DMA_XDMA, e2.s1_dma)
        self.assertEqual(schedule.DMA_IDMA, e2.s3_dma)
        self.assertEqual((120, 144), (timeline.start, timeline.task_end))

    def test_ascending_structural_lower_bound_is_181_5_ticks(self) -> None:
        quarters = schedule.LOW_TO_HIGH_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
        self.assertEqual(
            4 * 165 + 22 * schedule.STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
            quarters,
        )
        self.assertEqual(181.5, quarters / 4)

    def test_ends_inward_profile_matches_handoff_shape(self) -> None:
        queues = schedule.build_ends_inward_schedule()
        audit = schedule.audit_ends_inward_schedule(queues)

        self.assertEqual({"c0": 166, "c1": 166}, audit["queue_ticks"])
        self.assertEqual({"c0": 21, "c1": 22}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(7, len(audit["dma_release_edges"]))

    def test_ends_inward_preserves_special_stage_profiles(self) -> None:
        queues = schedule.build_ends_inward_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            (schedule.SHAPE_C, schedule.SHAPE_B, schedule.DMA_BOTH),
            (slots[41].s1_shape, slots[41].s3_shape, slots[41].s1_dma),
        )
        self.assertEqual(schedule.DMA_BOTH, slots[42].s3_dma)
        self.assertEqual(
            (schedule.SHAPE_A, schedule.SHAPE_C, schedule.DMA_IDMA),
            (slots[3].s1_shape, slots[3].s3_shape, slots[3].s3_dma),
        )

    def test_ends_inward_structural_lower_bound_is_182_5_ticks(self) -> None:
        queues = schedule.build_ends_inward_schedule()
        audit = schedule.audit_ends_inward_schedule(queues)

        self.assertEqual(
            {"c0": 727, "c1": 730},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(182.5, audit["structural_lower_bound_quarter_ticks"] / 4)


if __name__ == "__main__":
    unittest.main()
