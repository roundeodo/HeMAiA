import sys
import unittest
from pathlib import Path


WORKLOADS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOADS_DIR))
SW_DIR = next(path for path in Path(__file__).resolve().parents if path.name == "sw")
KERNEL_DIR = (
    SW_DIR
    / "device/apps/snax/snax-bingo-offload/libsnaxkernel/offload_hw_kernels"
)

from moe_dynamic_slot_dfg import build_dynamic_expert_slot_chain  # noqa: E402


class MoeDynamicSlotDfgTest(unittest.TestCase):
    def test_s2pf_run_map_covers_every_slice(self) -> None:
        for binding in (1, 2, 3):
            for block_count in range(1, 9):
                for transfers_per_step in (1, 2, 4):
                    slices_per_block = 1 if binding == 3 else 2
                    phase0_runs = slices_per_block * ((block_count + 1) // 2)
                    phase1_runs = slices_per_block * (block_count // 2)
                    phase0_groups = (
                        phase0_runs + transfers_per_step - 1
                    ) // transfers_per_step
                    phase1_groups = (
                        phase1_runs + transfers_per_step - 1
                    ) // transfers_per_step
                    transfer_count = max(
                        0 if phase0_groups == 0 else 2 * phase0_groups - 1,
                        2 * phase1_groups,
                    )

                    actual = []
                    for step in range(transfer_count):
                        phase = step & 1
                        phase_group = step >> 1
                        phase_blocks = (block_count + 1 - phase) // 2
                        for sub in range(transfers_per_step):
                            ordinal = phase_group * transfers_per_step + sub
                            if ordinal >= slices_per_block * phase_blocks:
                                continue
                            side = ordinal // phase_blocks
                            block = phase + 2 * (ordinal % phase_blocks)
                            actual.append((block, side))

                    expected = (
                        [(block, 0) for block in range(block_count)]
                        if binding == 3
                        else [
                            (block, side)
                            for block in range(block_count)
                            for side in range(2)
                        ]
                    )
                    self.assertEqual(sorted(expected), sorted(actual))

    def test_s2_compute_and_prefetch_start_after_s1_compute(self) -> None:
        completion_edges = set()
        descriptor_sequences = set()

        def add_node(_core, _kernel, _args, label):
            return label

        chain = build_dynamic_expert_slot_chain(
            add_node=add_node,
            add_edge=lambda source, target: completion_edges.add(
                (source, target)
            ),
            add_descriptor_sequence=lambda source, target: (
                descriptor_sequences.add((source, target))
            ),
            make_block_args=lambda block: block,
            input_ready="INPUT_READY",
            s1_block_count=8,
            s3_block_count=4,
            dma_core_id=1,
            gemm_core_id=0,
            implementation="optimized",
            label_prefix="SLOT",
        )

        self.assertIn(
            (chain["s1_load"], chain["s2_prefetch"]), completion_edges
        )
        self.assertIn(
            (chain["s1_compute"], chain["s2_prefetch"]), completion_edges
        )
        self.assertIn(
            (chain["s1_load"], chain["s2_compute"]), completion_edges
        )
        self.assertIn(
            (chain["s1_compute"], chain["s2_compute"]), completion_edges
        )
        self.assertIn(
            (chain["s1_load"], chain["s1_compute"]), descriptor_sequences
        )

    def test_s2pf_xdma_pipeline_contract_is_present(self) -> None:
        defs = (KERNEL_DIR / "moe_dynamic_defs.h").read_text()
        protocol = (KERNEL_DIR / "moe_dynamic_protocol.h").read_text()
        dma = (KERNEL_DIR / "moe_dynamic_dma.h").read_text()
        s1 = (KERNEL_DIR / "moe_dynamic_stage_s1.h").read_text()
        s3 = (KERNEL_DIR / "moe_dynamic_stage_s3.h").read_text()
        prefetch = (KERNEL_DIR / "moe_dynamic_stage_prefetch.h").read_text()
        compute = (KERNEL_DIR / "moe_dynamic_stage_compute.h").read_text()
        store = (KERNEL_DIR / "moe_dynamic_stage_store.h").read_text()

        self.assertIn("MOE_XDMA_PREPARED_S2PF", defs)
        self.assertNotIn("__moe_prepare_after_s2pf_xdma", protocol)
        self.assertIn("__moe_dyn_prepare_after_s2pf_xdma", dma)
        self.assertIn("cfg->s3_call[0].valid", dma)
        self.assertIn("__moe_dyn_prepare_s2pf_xdma(blk)", s1)
        self.assertIn("__moe_prepare_slot_entry_xdma", s1)
        self.assertIn(
            "__moe_dyn_prepare_after_s2pf_xdma(blk, cfg, st)", s1
        )
        self.assertEqual(
            store.count(
                "__moe_prepare_slot_entry_xdma(&next_blk, next_cfg, st)"
            ),
            2,
        )
        self.assertNotIn(
            "__moe_dyn_prepare_s1_xdma(&next_blk, next_cfg, st)", store
        )
        self.assertIn("__moe_dyn_prepare_s2pf_both_xdma_address", dma)
        self.assertIn("__moe_dyn_prepare_s2pf_single_xdma_address", dma)
        self.assertIn("__moe_dyn_start_both_2d_preloaded_xdma", prefetch)
        self.assertIn("__moe_dyn_prepare_after_s2pf_xdma(blk, cfg, st)", prefetch)
        both_start = dma.index("__moe_dyn_start_both_2d_preloaded_xdma(")
        both_end = dma.index("__moe_dyn_start_pair_2d_idma(", both_start)
        both_body = dma[both_start:both_end]
        self.assertLess(
            both_body.index("__moe_dyn_xdma_start_remote_begin()"),
            both_body.index("snrt_dma_start_2d_wideptr("),
        )
        self.assertLess(
            both_body.index("snrt_dma_start_2d_wideptr("),
            both_body.index("__moe_dyn_xdma_start_remote_commit(previous)"),
        )
        self.assertIn("__moe_dyn_xdma_start_remote_begin()", prefetch)
        self.assertIn(
            "__moe_dyn_xdma_start_remote_commit(previous)", prefetch
        )
        self.assertNotIn("xdma_start()", prefetch)
        self.assertNotIn("xdma_ready", prefetch)
        self.assertNotIn("transfers_per_step == 0u", prefetch)
        self.assertNotIn("__moe_s2pf_run_xdma_mode", prefetch)
        self.assertNotIn("if (both", prefetch)
        self.assertNotIn("__moe_dyn_copy_pair_2d", s1)
        self.assertNotIn("__moe_dyn_copy_pair_2d", s3)
        self.assertNotIn("__moe_dyn_2d_pair_pending_t", s1)
        self.assertNotIn("__moe_dyn_2d_pair_pending_t", s3)
        self.assertNotIn("__moe_dyn_2d_pair_pending_t", prefetch)
        self.assertNotIn("wait_idma", prefetch)
        self.assertNotIn("store_prepared == 0u", store)
        self.assertIn(
            "MOE_S4_CSR_LAYOUT_PHASE_BATCHED, 0u", compute
        )
        self.assertNotIn(
            "!__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S4)",
            compute,
        )


if __name__ == "__main__":
    unittest.main()
