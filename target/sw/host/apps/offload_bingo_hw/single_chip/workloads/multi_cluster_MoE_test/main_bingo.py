#!/usr/bin/env python3

import argparse
import os
import pathlib
import sys

import hjson

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[7]
sys.path.insert(0, str(ROOT_DIR / "target/sw/host/runtime/libbingo/mini_compiler"))
sys.path.insert(0, str(CURRENT_DIR.parent))

from bingo_dfg import BingoDFG  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelCheckResultArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelMoeDynamicExpertBlockArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from moe_dynamic_slot_dfg import (  # noqa: E402
    DEFAULT_SLOT_IMPLEMENTATION,
    build_dynamic_expert_slot_chain,
    dynamic_expert_gather_kernel,
)
from moe_high_to_low_workload import (  # noqa: E402
    add_high_to_low_schedule,
    add_s1_stage_smoke_schedule,
    define_high_to_low_memory,
)
from moe_test_layout import derive_params  # noqa: E402
from moe_test_schedule import (  # noqa: E402
    BASELINE_PROFILE,
    C_TAIL_SMOKE_PROFILE,
    DMA_BOTH,
    DMA_NONE,
    DYNAMIC_DESC_PROFILE,
    ENDS_INWARD_PROFILE,
    HIGH_TO_LOW_PROFILE,
    LOW_TO_HIGH_PROFILE,
    S2PF_BOTH_PROFILE,
    S1_STAGE_SMOKE_PROFILE,
    SCHEDULE_PROFILES,
    STATIC_DESC_PROFILE,
    build_schedule_profile,
    format_schedule_manifest,
    select_two_slot_s2pf_binding,
)

GEMM_CORE = 0
DMA_CORE = 1
HOST_CORE = 2
PROD_CLUSTERS = (("c0", 0), ("c1", 1))
SLOT_IMPLEMENTATION = DEFAULT_SLOT_IMPLEMENTATION
BENCHMARK_CLUSTER_CONFIG = {
    "c0": {
        "s1_shape": 1,
        "s2_shape": 2,
        "s3_shape": 1,
        "s4_shape": 2,
        "s1_dma": 1,
        "skip_s3": 1,
        "s3_dma": 0,
        "s2_prefetch_dma": 1,
        "s4_prefetch_dma": 0,
        "s4_token_start": 0,
    },
    "c1": {
        "s1_shape": 1,
        "s2_shape": 2,
        "s3_shape": 1,
        "s4_shape": 2,
        "s1_dma": 2,
        "skip_s3": 0,
        "s3_dma": 2,
        "s2_prefetch_dma": 0,
        "s4_prefetch_dma": 2,
        "s4_token_start": "after_s3",
    },
}
SLOT1_CLUSTER_CONFIG = {
    "c0": {
        "skip_s1": 0,
        "s1_dma": 1,
        "s2_prefetch_dma": 1,
    },
    "c1": {
        "skip_s1": 1,
        "s1_dma": 0,
        "s2_prefetch_dma": 2,
    },
}
MOE_PIPELINE_CTRL_SLOT_BYTES = 1024
MOE_PIPELINE_CTRL_BANK_OFFSET = 448


def offset(handle, byte_offset: int):
    if byte_offset == 0:
        return handle
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, offset=handle.offset + byte_offset)
    return f"{handle.get_c_var_name()} + {byte_offset}"


class ProductionMoeDualClusterSetupArgs(HostBingoKernelCheckResultArgs):
    """Initialize the eleven-token C0/C1 slots and next-expert prefetches."""

    SLOT_BYTES = 384
    HEADER_BYTES = 64

    def __init__(self, p, mh):
        super().__init__(
            golden_data_addr=mh["input"],
            output_data_addr=mh["input"],
            data_size=1,
            name="SETUP_PRODUCTION_SLOT",
        )
        self.p = p
        self.mh = mh
        # These L1 buffers are addressed through the production static ABI,
        # rather than through a generated kernel-argument field. Keep direct
        # references so the mini compiler allocates them and emits addresses.
        for prefix, cluster in PROD_CLUSTERS:
            for name in (
                "layout",
                "gate",
                "up",
                "down",
                "gate_out",
                "pipeline_ctrl",
                "prod_output_l3",
            ):
                setattr(self, f"_abi_memref_{prefix}_{name}", mh[f"{prefix}_{name}"])

    def _addr(self, value, handle_name_map, as_64bit=True):
        assignments = {}
        self._process_addr(
            value,
            "value",
            assignments,
            handle_name_map,
            split_64bit=False,
            as_64bit=as_64bit,
        )
        return assignments["value"]

    def get_post_init_code(self, args_var, handle_name_map):
        del args_var
        p = self.p
        mh = self.mh
        row_stride = p["token_row_stride"]
        s3_row_bytes = p["token_payload_bytes"] // (2 * p["s3_block_count"])
        down_half_bytes = p["down_weight_bytes"] // 2
        down_b_n_stride = [p["down_K"] * (16 << shape) for shape in range(3)]
        down_a_m_stride = [
            (
                (p["down_K"] * 8)
                // (p["base_mesh_col"] << shape)
            )
            * 64
            for shape in range(3)
        ]
        down_d_m_stride = [
            (p["base_mesh_row"] >> shape) * row_stride for shape in range(3)
        ]
        lines = [
            f"_Static_assert(BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES == {self.SLOT_BYTES}u, "
            '"test and production dynamic slot ABI diverged");',
            '_Static_assert(BINGO_MOE_STATIC_ARG_SLOT_BYTES == 192u, '
            '"test and production static slot ABI diverged");',
        ]

        token_count = p["prod_slot0_tokens"]
        next_token_count = p["prod_slot1_tokens"]

        for prefix, cluster in PROD_CLUSTERS:
            bench = BENCHMARK_CLUSTER_CONFIG[prefix]
            s1_shape = bench["s1_shape"]
            s2_shape = bench["s2_shape"]
            s3_shape = bench["s3_shape"]
            s4_shape = bench["s4_shape"]
            s1_rows = p["base_mesh_row"] >> s1_shape
            s2_rows = p["base_mesh_row"] >> s2_shape
            s3_rows = p["base_mesh_row"] >> s3_shape
            s4_rows = p["base_mesh_row"] >> s4_shape
            s2_tokens = max(token_count - s1_rows, 0)
            s2_m_exec = (s2_tokens + s2_rows - 1) // s2_rows
            s1_n = p["indiv_N_per_block"] // (
                p["base_mesh_col"] << s1_shape
            )
            s2_n = (p["s1_block_count"] * p["indiv_N_per_block"]) // (
                p["base_mesh_col"] << s2_shape
            )
            s3_n = p["indiv_down_N_per_block"] // (
                p["base_mesh_col"] << s3_shape
            )
            s4_n = (
                p["s3_block_count"] * p["indiv_down_N_per_block"]
            ) // (
                p["base_mesh_col"] << s4_shape
            )
            s1_dma = bench["s1_dma"]
            runtime_cluster = cluster
            static_l3 = self._addr(mh[f"{prefix}_prod_static_l3"], handle_name_map)
            runtime_l3 = self._addr(mh[f"{prefix}_prod_runtime_l3"], handle_name_map)
            runtime_l1 = self._addr(
                mh[f"{prefix}_prod_runtime_l1"], handle_name_map, as_64bit=False
            )
            token_refs_l1 = self._addr(
                mh[f"{prefix}_token_refs_l1"], handle_name_map, as_64bit=False
            )
            input_l3 = self._addr(mh["input"], handle_name_map)
            gate_l3 = self._addr(mh[f"{prefix}_gate_src"], handle_name_map)
            up_l3 = self._addr(mh[f"{prefix}_up_src"], handle_name_map)
            down_l3 = self._addr(mh[f"{prefix}_down_src"], handle_name_map)
            output_l3 = self._addr(
                mh[f"{prefix}_prod_output_l3"], handle_name_map
            )
            l1_a = self._addr(mh[f"{prefix}_a"], handle_name_map, as_64bit=False)
            l1_gate = self._addr(
                mh[f"{prefix}_gate"], handle_name_map, as_64bit=False
            )
            l1_up = self._addr(
                mh[f"{prefix}_up"], handle_name_map, as_64bit=False
            )
            l1_down = self._addr(
                mh[f"{prefix}_down"], handle_name_map, as_64bit=False
            )
            l1_d = self._addr(
                mh[f"{prefix}_gate_out"], handle_name_map, as_64bit=False
            )
            l1_down_d = self._addr(
                mh[f"{prefix}_out"], handle_name_map, as_64bit=False
            )
            l1_scratch = self._addr(
                mh[f"{prefix}_gate_scratch"], handle_name_map, as_64bit=False
            )

            static_name = f"prod_{prefix}_st"
            runtime_name = f"prod_{prefix}_rt"
            slot0_name = f"prod_{prefix}_slot0"
            slot1_name = f"prod_{prefix}_slot1"
            skip_s3 = bench["skip_s3"]
            s3_dma = bench["s3_dma"]
            ctrl0 = (
                1
                | (skip_s3 << 2)
                | (s1_shape << 5)
                | (s3_shape << 7)
                | (s1_dma << 9)
                | (s3_dma << 11)
                | (runtime_cluster << 13)
            )
            slot1_bench = SLOT1_CLUSTER_CONFIG[prefix]
            slot1_skip_s1 = slot1_bench["skip_s1"]
            slot1_s1_dma = slot1_bench["s1_dma"]
            ctrl1 = (
                1
                | (slot1_skip_s1 << 1)
                | (1 << 2)
                | (s1_shape << 5)
                | (s3_shape << 7)
                | (slot1_s1_dma << 9)
                | (runtime_cluster << 13)
                | (1 << 14)
            )
            s2_dma = select_two_slot_s2pf_binding(
                p["schedule_profile"], prefix, 0, bench["s2_prefetch_dma"]
            )
            s4_dma = bench["s4_prefetch_dma"]
            s2_prefetch_vd = (
                ((1 | (s2_dma << 1)) << (2 * 3)) if s2_dma else 0
            )
            s4_prefetch_vd = (
                ((1 | (s4_dma << 1)) << (3 * 3)) if s4_dma else 0
            )
            s4_token_start = (
                s3_rows
                if bench["s4_token_start"] == "after_s3"
                else bench["s4_token_start"]
            )
            s4_tokens = max(token_count - s4_token_start, 0)
            s4_m_exec = (s4_tokens + s4_rows - 1) // s4_rows
            slot1_s2_token_start = 0 if slot1_skip_s1 else s1_rows
            slot1_s2_tokens = max(next_token_count - slot1_s2_token_start, 0)
            slot1_s2_m_exec = (
                slot1_s2_tokens + s2_rows - 1
            ) // s2_rows
            slot1_s4_m_exec = (
                next_token_count + s4_rows - 1
            ) // s4_rows
            slot1_s2_dma = select_two_slot_s2pf_binding(
                p["schedule_profile"], prefix, 1,
                slot1_bench["s2_prefetch_dma"]
            )
            slot1_s2_prefetch_vd = (
                (1 | (slot1_s2_dma << 1)) << (2 * 3)
            )

            lines += [
                f"__snax_bingo_moe_dynamic_expert_static_args_t *{static_name} = "
                f"(__snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t){static_l3};",
                f"uint8_t *{runtime_name} = (uint8_t *)(uintptr_t){runtime_l3};",
                f"memset({static_name}, 0, sizeof(*{static_name}));",
                f"memset({runtime_name}, 0, {self.HEADER_BYTES + 2 * self.SLOT_BYTES}u);",
                *(
                    [f"memset((void *)(uintptr_t){output_l3}, 0, "
                     f"{p['prod_output_bytes']}u);"]
                    if p["prod_clear_output"]
                    else []
                ),
                f"{static_name}->token_refs_addr = {token_refs_l1};",
                f"{static_name}->input_A_l3_base = {input_l3};",
                f"{static_name}->indiv_gate_B_l3 = {gate_l3};",
                f"{static_name}->indiv_up_B_l3 = {up_l3};",
                f"{static_name}->indiv_down_B_l3 = {down_l3};",
                f"{static_name}->output_l3_base = {output_l3};",
                f"{static_name}->active_state_l1_addr = {runtime_l1};",
                f"{static_name}->l1_a_addr = {l1_a};",
                f"{static_name}->l1_b_gate_addr = {l1_gate};",
                f"{static_name}->l1_b_up_addr = {l1_up};",
                f"{static_name}->l1_b_down_addr = {l1_down};",
                f"{static_name}->l1_d_addr = {l1_d};",
                f"{static_name}->l1_down_d_addr = {l1_down_d};",
                f"{static_name}->l1_d1_scratch_addr = {l1_scratch};",
                f"{static_name}->A_token_bytes = {p['token_payload_bytes']}u;",
                f"{static_name}->indiv_B_expert_stride = {p['gate_weight_bytes']}u;",
                f"{static_name}->indiv_down_B_expert_stride = {p['down_weight_bytes']}u;",
                f"{static_name}->indiv_B_block_stride = {p['gate_block_bytes']}u;",
                f"{static_name}->indiv_down_B_block_stride = {p['down_block_bytes']}u;",
                f"{static_name}->s1_block_count = {p['s1_block_count']}u;",
                f"{static_name}->s3_block_count = {p['s3_block_count']}u;",
                f"{static_name}->indiv_K1 = {p['gate_K']}u;",
                f"{static_name}->indiv_N_per_block = {p['indiv_N_per_block']}u;",
                f"{static_name}->indiv_down_K1 = {p['down_K']}u;",
                f"{static_name}->indiv_down_N_per_block = {p['indiv_down_N_per_block']}u;",
                f"{static_name}->rescale_mult = 1u;",
                f"{static_name}->rescale_shift = 0u;",
                f"{static_name}->output_expert_stride_bytes = {p['prod_output_bytes']}u;",
                f"{static_name}->max_tokens_per_expert = "
                f"{token_count + next_token_count}u;",
                f"{static_name}->A_row_stride = {row_stride}u;",
                f"{static_name}->s3_row_bytes = {s3_row_bytes}u;",
                f"{static_name}->down_half_weight_bytes = {down_half_bytes}u;",
                f"{static_name}->down_b_k_section = {p['down_K'] * 8 * 2}u;",
                f"{static_name}->down_b_n_stride[0] = {down_b_n_stride[0]}u;",
                f"{static_name}->down_b_n_stride[1] = {down_b_n_stride[1]}u;",
                f"{static_name}->down_b_n_stride[2] = {down_b_n_stride[2]}u;",
                f"{static_name}->down_a_m_stride[0] = {down_a_m_stride[0]}u;",
                f"{static_name}->down_a_m_stride[1] = {down_a_m_stride[1]}u;",
                f"{static_name}->down_a_m_stride[2] = {down_a_m_stride[2]}u;",
                f"{static_name}->down_d_m_stride[0] = {down_d_m_stride[0]}u;",
                f"{static_name}->down_d_m_stride[1] = {down_d_m_stride[1]}u;",
                f"{static_name}->down_d_m_stride[2] = {down_d_m_stride[2]}u;",
                f"((uint32_t *){runtime_name})[2] = 2u;",
                f"((uint32_t *){runtime_name})[3] = 2u;",
                f"__snax_bingo_kernel_moe_dynamic_expert_args_t *{slot0_name} = "
                f"(__snax_bingo_kernel_moe_dynamic_expert_args_t *)"
                f"({runtime_name} + {self.HEADER_BYTES}u);",
                f"__snax_bingo_kernel_moe_dynamic_expert_args_t *{slot1_name} = "
                f"(__snax_bingo_kernel_moe_dynamic_expert_args_t *)"
                f"({runtime_name} + {self.HEADER_BYTES + self.SLOT_BYTES}u);",
                f"{slot0_name}->ctrl = {ctrl0}u;",
                f"{slot0_name}->expert_id = 0u;",
                f"{slot0_name}->token_ref_start = 0u;",
                f"{slot0_name}->ntokens = {token_count}u;",
                f"{slot0_name}->m_s2_exec = {s2_m_exec}u;",
                f"{slot0_name}->m_s4_exec = {s4_m_exec}u;",
                f"{slot0_name}->dma_slot_vd = {s2_prefetch_vd | s4_prefetch_vd}u;",
                f"{slot0_name}->dma_slot_eids = 0u;",
                f"{slot1_name}->ctrl = {ctrl1}u;",
                f"{slot1_name}->expert_id = 0u;",
                f"{slot1_name}->token_ref_start = {token_count}u;",
                f"{slot1_name}->ntokens = {next_token_count}u;",
                f"{slot1_name}->m_s2_exec = {slot1_s2_m_exec}u;",
                f"{slot1_name}->m_s4_exec = {slot1_s4_m_exec}u;",
                f"{slot1_name}->dma_slot_vd = {slot1_s2_prefetch_vd}u;",
                f"{slot1_name}->dma_slot_eids = 0u;",
            ]
            for block in range(p["s1_block_count"]):
                output_offset = block * p["bank_mode0_output_block_span"]
                lines += [
                    f"{slot0_name}->s1_call[{block}].valid = 1u;",
                    f"{slot0_name}->s1_call[{block}].output_D0_addr = "
                    f"{l1_d} + {output_offset}u;",
                    f"{slot0_name}->s1_call[{block}].N = {s1_n}u;",
                    f"{slot0_name}->s1_call[{block}].array_shape = {s1_shape}u;",
                    f"{slot1_name}->s1_call[{block}].valid = "
                    f"{1 - slot1_skip_s1}u;",
                    f"{slot1_name}->s1_call[{block}].output_D0_addr = "
                    f"{l1_d} + {output_offset}u;",
                    f"{slot1_name}->s1_call[{block}].N = {s1_n}u;",
                    f"{slot1_name}->s1_call[{block}].array_shape = {s1_shape}u;",
                ]
            for block in range(p["s3_block_count"]):
                lines += [
                    f"{slot0_name}->s3_call[{block}].valid = {1 - skip_s3}u;",
                    f"{slot0_name}->s3_call[{block}].N = {s3_n}u;",
                    f"{slot0_name}->s3_call[{block}].array_shape = {s3_shape}u;",
                    f"{slot0_name}->s3_call[{block}].reserved = 0u;",
                    f"{slot1_name}->s3_call[{block}].valid = 0u;",
                    f"{slot1_name}->s3_call[{block}].N = {s3_n}u;",
                    f"{slot1_name}->s3_call[{block}].array_shape = {s3_shape}u;",
                    f"{slot1_name}->s3_call[{block}].reserved = 0u;",
                ]
            lines += [
                f"{slot0_name}->s2_call.valid = 1u;",
                f"{slot0_name}->s2_call.token_start = {s1_rows}u;",
                f"{slot0_name}->s2_call.reserved = 0u;",
                f"{slot0_name}->s2_call.M = {s2_m_exec}u;",
                f"{slot0_name}->s2_call.N = {s2_n}u;",
                f"{slot0_name}->s2_call.array_shape = {s2_shape}u;",
                f"{slot0_name}->s4_call.valid = 1u;",
                f"{slot0_name}->s4_call.token_start = {s4_token_start}u;",
                f"{slot0_name}->s4_call.reserved0 = 0u;",
                f"{slot0_name}->s4_call.reserved1 = 0u;",
                f"{slot0_name}->s4_call.M = {s4_m_exec}u;",
                f"{slot0_name}->s4_call.N = {s4_n}u;",
                f"{slot0_name}->s4_call.array_shape = {s4_shape}u;",
                f"{slot1_name}->s2_call.valid = 1u;",
                f"{slot1_name}->s2_call.token_start = {slot1_s2_token_start}u;",
                f"{slot1_name}->s2_call.reserved = 0u;",
                f"{slot1_name}->s2_call.M = {slot1_s2_m_exec}u;",
                f"{slot1_name}->s2_call.N = {s2_n}u;",
                f"{slot1_name}->s2_call.array_shape = {s2_shape}u;",
                f"{slot1_name}->s4_call.valid = 1u;",
                f"{slot1_name}->s4_call.token_start = 0u;",
                f"{slot1_name}->s4_call.reserved0 = 0u;",
                f"{slot1_name}->s4_call.reserved1 = 0u;",
                f"{slot1_name}->s4_call.M = {slot1_s4_m_exec}u;",
                f"{slot1_name}->s4_call.N = {s4_n}u;",
                f"{slot1_name}->s4_call.array_shape = {s4_shape}u;",
            ]
        return lines


def get_args():
    parser = argparse.ArgumentParser(description="Static single-slot MoE overlap test")
    parser.add_argument("--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_offload_file_name", type=str, required=True)
    parser.add_argument(
        "--schedule-profile",
        choices=SCHEDULE_PROFILES,
        default=BASELINE_PROFILE,
    )
    return parser.parse_args()


def load_params(args):
    with args.cfg.open() as f:
        cfg = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hwcfg = hjson.loads(f.read())
    return derive_params({**cfg, **hwcfg}, args.schedule_profile)


def define_production_memory(p):
    """Allocate C0/C1 using the same per-cluster arena as production."""
    mh = {
        "input": BingoMemSymbol("moe_test_input_A"),
        "prod_slot_token_refs": BingoMemSymbol("moe_test_prod_slot_token_refs"),
    }
    for prefix, cluster in PROD_CLUSTERS:
        mh[f"{prefix}_slot0_golden"] = BingoMemSymbol(
            f"moe_test_{prefix}_slot0_golden"
        )
        mh[f"{prefix}_slot1_golden"] = BingoMemSymbol(
            f"moe_test_{prefix}_slot1_golden"
        )
        mh[f"{prefix}_gate_src"] = BingoMemSymbol(f"moe_test_{prefix}_gate_B")
        mh[f"{prefix}_up_src"] = BingoMemSymbol(f"moe_test_{prefix}_up_B")
        mh[f"{prefix}_down_src"] = BingoMemSymbol(f"moe_test_{prefix}_down_B")
        layout = BingoMemAlloc(
            f"moe_test_{prefix}_prod_l1_layout",
            size=p["bank_tcdm_size"],
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=p["bank_tcdm_row_bytes"],
        )
        mh[f"{prefix}_layout"] = layout
        mh[f"{prefix}_a"] = offset(layout, p["bank_delta_local_a"])
        mh[f"{prefix}_gate"] = offset(layout, p["bank_delta_local_b0"])
        mh[f"{prefix}_up"] = offset(layout, p["bank_delta_local_b1"])
        mh[f"{prefix}_down"] = offset(
            layout,
            p["bank_mode1_region_offset"] + p["bank_delta_local_b0"],
        )
        mh[f"{prefix}_gate_out"] = offset(layout, p["bank_delta_local_d0"])
        mh[f"{prefix}_gate_scratch"] = mh[f"{prefix}_gate_out"]
        mh[f"{prefix}_out"] = offset(layout, p["bank_delta_local_mode1_d0"])
        mh[f"{prefix}_token_refs_l1"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_token_refs_l1",
            size=p["prod_token_refs_bytes"],
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=64,
        )
        mh[f"{prefix}_prod_static_l3"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_static_l3", size=192, mem_level="L3"
        )
        mh[f"{prefix}_prod_runtime_l3"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_runtime_l3",
            size=64 + 2 * ProductionMoeDualClusterSetupArgs.SLOT_BYTES,
            mem_level="L3",
        )
        mh[f"{prefix}_prod_static_l1"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_static_l1",
            size=192,
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=64,
        )
        mh[f"{prefix}_prod_runtime_l1"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_runtime_l1",
            size=64 + 2 * ProductionMoeDualClusterSetupArgs.SLOT_BYTES,
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=64,
        )
        mh[f"{prefix}_pipeline_ctrl"] = BingoMemAlloc(
            f"moe_test_{prefix}_pipeline_ctrl",
            size=2 * MOE_PIPELINE_CTRL_SLOT_BYTES,
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=p["bank_tcdm_row_bytes"],
        )
        mh[f"{prefix}_prod_output_l3"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_output_l3",
            size=p["prod_output_bytes"],
            mem_level="L3",
        )
    return mh


def add_node(dfg, cluster, core, kernel, args, node_name=""):
    node = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=cluster,
        assigned_core_id=core,
        node_name=node_name,
        kernel_name=kernel,
        kernel_args=args,
    )
    dfg.bingo_add_node(node)
    return node


def add_copy(dfg, cluster, src, dst, size, node_name=""):
    return add_node(
        dfg,
        cluster,
        DMA_CORE,
        "__snax_bingo_kernel_idma_1d_copy",
        SnaxBingoKernelIdma1dCopyArgs(src, dst, size),
        node_name,
    )



def add_production_slot_handoff_test(dfg, p, mh):
    """Execute both production slots on C0 and C1."""
    setup = add_node(
        dfg,
        0,
        HOST_CORE,
        "__host_bingo_kernel_check_result",
        ProductionMoeDualClusterSetupArgs(p, mh),
        "SETUP_PRODUCTION_SLOT",
    )
    slot0_stores = {}
    slot_runtime_ready = []
    slot_gathers = []
    scope_begins = {}
    scope_args = {}

    for prefix, cluster in PROD_CLUSTERS:
        refs_to_l1 = add_copy(
            dfg,
            cluster,
            mh["prod_slot_token_refs"],
            mh[f"{prefix}_token_refs_l1"],
            p["prod_token_refs_bytes"],
            f"{prefix.upper()}_PROD_LOAD_TOKEN_REFS",
        )
        static_to_l1 = add_copy(
            dfg,
            cluster,
            mh[f"{prefix}_prod_static_l3"],
            mh[f"{prefix}_prod_static_l1"],
            192,
            f"{prefix.upper()}_PROD_LOAD_STATIC_ABI",
        )
        runtime_to_l1 = add_copy(
            dfg,
            cluster,
            mh[f"{prefix}_prod_runtime_l3"],
            mh[f"{prefix}_prod_runtime_l1"],
            64 + 2 * ProductionMoeDualClusterSetupArgs.SLOT_BYTES,
            f"{prefix.upper()}_PROD_LOAD_DYNAMIC_ABI",
        )
        dfg.bingo_add_edge(setup, refs_to_l1)
        dfg.bingo_add_edge(refs_to_l1, static_to_l1)
        dfg.bingo_add_edge(static_to_l1, runtime_to_l1)

        slot0_addr = offset(mh[f"{prefix}_prod_runtime_l1"], 64)
        static_addr = mh[f"{prefix}_prod_static_l1"]
        pipeline_ctrl_addr = offset(
            mh[f"{prefix}_pipeline_ctrl"], MOE_PIPELINE_CTRL_BANK_OFFSET
        )
        slot_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
            slot0_addr, static_addr, pipeline_ctrl_addr, 0
        )
        scope_args[prefix] = slot_args
        scope_begins[prefix] = add_node(
            dfg,
            cluster,
            DMA_CORE,
            "__snax_bingo_kernel_moe_dyn_opt_cluster_begin",
            slot_args,
            f"{prefix.upper()}_PROD_SCOPE_BEGIN",
        )
        gather = add_node(
            dfg,
            cluster,
            DMA_CORE,
            dynamic_expert_gather_kernel(SLOT_IMPLEMENTATION),
            slot_args,
            f"{prefix.upper()}_PROD_SLOT0_GATHER_{p['prod_slot0_tokens']}_TOKENS",
        )
        slot_runtime_ready.append(runtime_to_l1)
        slot_gathers.append(gather)

        def make_block_args(block):
            return SnaxBingoKernelMoeDynamicExpertBlockArgs(
                slot0_addr, static_addr, pipeline_ctrl_addr, block
            )

        chain = build_dynamic_expert_slot_chain(
            add_node=lambda core, kernel, args, label: add_node(
                dfg, cluster, core, kernel, args, label
            ),
            add_edge=dfg.bingo_add_edge,
            add_descriptor_sequence=lambda producer, consumer: dfg.add_edge(
                producer, consumer, descriptor_sequence=True
            ),
            make_block_args=make_block_args,
            input_ready=gather,
            s1_block_count=p["s1_block_count"],
            s3_block_count=p["s3_block_count"],
            dma_core_id=DMA_CORE,
            gemm_core_id=GEMM_CORE,
            implementation=SLOT_IMPLEMENTATION,
            label_prefix=f"{prefix.upper()}_PROD",
        )
        slot0_stores[prefix] = chain["store"]

    # Give both cluster slots the same release condition. Each gather becomes
    # ready only after both clusters have finished loading their runtime ABI.
    for runtime_ready in slot_runtime_ready:
        for scope_begin in scope_begins.values():
            dfg.bingo_add_edge(runtime_ready, scope_begin)
    for prefix, gather in zip((prefix for prefix, _ in PROD_CLUSTERS), slot_gathers):
        dfg.bingo_add_edge(scope_begins[prefix], gather)

    slot1_chains = {}
    for prefix, cluster in PROD_CLUSTERS:
        slot1_addr = offset(
            mh[f"{prefix}_prod_runtime_l1"],
            ProductionMoeDualClusterSetupArgs.HEADER_BYTES
            + ProductionMoeDualClusterSetupArgs.SLOT_BYTES,
        )
        static_addr = mh[f"{prefix}_prod_static_l1"]
        pipeline_ctrl_addr = offset(
            mh[f"{prefix}_pipeline_ctrl"],
            MOE_PIPELINE_CTRL_BANK_OFFSET + MOE_PIPELINE_CTRL_SLOT_BYTES,
        )

        def make_slot1_block_args(block):
            return SnaxBingoKernelMoeDynamicExpertBlockArgs(
                slot1_addr, static_addr, pipeline_ctrl_addr, block
            )

        slot1_chains[prefix] = build_dynamic_expert_slot_chain(
            add_node=lambda core, kernel, args, label, cluster=cluster: add_node(
                dfg, cluster, core, kernel, args, label
            ),
            add_edge=dfg.bingo_add_edge,
            add_descriptor_sequence=lambda producer, consumer: dfg.add_edge(
                producer, consumer, descriptor_sequence=True
            ),
            make_block_args=make_slot1_block_args,
            input_ready=slot0_stores[prefix],
            s1_block_count=p["s1_block_count"],
            s3_block_count=p["s3_block_count"],
            dma_core_id=DMA_CORE,
            gemm_core_id=GEMM_CORE,
            implementation=SLOT_IMPLEMENTATION,
            label_prefix=f"{prefix.upper()}_PROD_SLOT1",
        )

    final_stores = []
    for prefix, cluster in PROD_CLUSTERS:
        scope_end = add_node(
            dfg,
            cluster,
            DMA_CORE,
            "__snax_bingo_kernel_moe_dyn_opt_cluster_end",
            scope_args[prefix],
            f"{prefix.upper()}_PROD_SCOPE_END",
        )
        dfg.bingo_add_edge(slot1_chains[prefix]["store"], scope_end)
        final_stores.append((prefix, scope_end))
    done_checks = []
    for prefix, _ in final_stores:
        checks = (
            (
                "SLOT0",
                mh[f"{prefix}_slot0_golden"],
                mh[f"{prefix}_prod_output_l3"],
                p["prod_slot0_tokens"],
            ),
            (
                "SLOT1",
                mh[f"{prefix}_slot1_golden"],
                offset(
                    mh[f"{prefix}_prod_output_l3"],
                    p["prod_slot0_tokens"] * p["token_payload_bytes"],
                ),
                p["prod_slot1_tokens"],
            ),
        )
        for slot_name, golden, output, token_count in checks:
            check_name = f"{prefix.upper()}_PRODUCTION_{slot_name}_OUTPUT"
            done = add_node(
                dfg,
                0,
                HOST_CORE,
                "__host_bingo_kernel_check_result",
                HostBingoKernelCheckResultArgs(
                    golden_data_addr=golden,
                    output_data_addr=output,
                    data_size=token_count * p["token_payload_bytes"],
                    name=check_name,
                ),
                check_name,
            )
            for _, completed_store in final_stores:
                dfg.bingo_add_edge(completed_store, done)
            done_checks.append(done)
    return done_checks


def create_dfg(p, mh):
    dfg = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=4,
        num_cores_per_cluster=2,
        is_host_as_acc=True,
        chiplet_ids=[0x00],
    )
    if p["schedule_profile"] == S1_STAGE_SMOKE_PROFILE:
        queues = build_schedule_profile(p["schedule_profile"])
        print("s1_stage_smoke schedule: C1/E23 -> C0/E24, S1-only 2-token C/C")
        add_s1_stage_smoke_schedule(dfg, p, mh, queues)
    elif p["schedule_profile"] in (
        HIGH_TO_LOW_PROFILE,
        LOW_TO_HIGH_PROFILE,
        ENDS_INWARD_PROFILE,
        STATIC_DESC_PROFILE,
        DYNAMIC_DESC_PROFILE,
        C_TAIL_SMOKE_PROFILE,
    ):
        queues = build_schedule_profile(p["schedule_profile"])
        if p["schedule_profile"] in (
            HIGH_TO_LOW_PROFILE,
            LOW_TO_HIGH_PROFILE,
            ENDS_INWARD_PROFILE,
            STATIC_DESC_PROFILE,
            DYNAMIC_DESC_PROFILE,
        ):
            print(format_schedule_manifest(queues, p["schedule_profile"]))
        else:
            print("c_tail_smoke schedule: C1/E23 -> C0/E24, both 2-token C/C")
        add_high_to_low_schedule(dfg, p, mh, queues, SLOT_IMPLEMENTATION)
    else:
        s2pf_mode = (
            "BOTH" if p["schedule_profile"] == S2PF_BOTH_PROFILE else "single"
        )
        print(
            "Execute concurrent C0/C1 slot0 and slot1 paths with fused slot "
            f"handoff; C0 slot0 S2PF={s2pf_mode}"
        )
        add_production_slot_handoff_test(dfg, p, mh)
    return dfg


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    params = load_params(args)
    if params["schedule_profile"] in (
        HIGH_TO_LOW_PROFILE,
        LOW_TO_HIGH_PROFILE,
        ENDS_INWARD_PROFILE,
        STATIC_DESC_PROFILE,
        DYNAMIC_DESC_PROFILE,
        C_TAIL_SMOKE_PROFILE,
        S1_STAGE_SMOKE_PROFILE,
    ):
        memory = define_high_to_low_memory(
            params, build_schedule_profile(params["schedule_profile"])
        )
    else:
        memory = define_production_memory(params)
    dfg = create_dfg(params, memory)
    dfg.bingo_compile_dfg(
        params["app_name"],
        args.output_dir,
        args.output_offload_file_name,
        extra_include_header_list=[
            "multi_cluster_MoE_test_data.h",
            "../multi_cluster_MoE/moe_runtime_timing.h",
        ],
        profile_kernel_prefix="__snax_bingo_kernel_moe_",
        profile_condition="MOE_RUNTIME_TIMING",
        profile_report_function="__host_bingo_moe_print_runtime_timing_v3",
    )

if __name__ == "__main__":
    main()
