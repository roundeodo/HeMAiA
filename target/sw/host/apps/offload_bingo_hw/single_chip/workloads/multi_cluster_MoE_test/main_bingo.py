#!/usr/bin/env python3

import argparse
import os
import pathlib
import sys

import hjson

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[7]
sys.path.insert(0, str(ROOT_DIR / "target/sw/host/runtime/libbingo/mini_compiler"))

from bingo_dfg import BingoDFG  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelCheckResultArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelMoeDynamicExpertBlockArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from moe_test_layout import derive_params  # noqa: E402

GEMM_CORE = 0
DMA_CORE = 1
HOST_CORE = 2
PROD_CLUSTERS = (("c0", 0, 1), ("c1", 1, 2))
MOE_PIPELINE_CTRL_SLOT_BYTES = 1024
MOE_PIPELINE_CTRL_BANK_OFFSET = 448


def offset(handle, byte_offset: int):
    if byte_offset == 0:
        return handle
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, offset=handle.offset + byte_offset)
    return f"{handle.get_c_var_name()} + {byte_offset}"


class ProductionMoeDualClusterSetupArgs(HostBingoKernelCheckResultArgs):
    """Initialize one computed slot and one gather-only slot per cluster."""

    SLOT_BYTES = 384
    HEADER_BYTES = 64
    NTOKENS = 6

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
        for prefix, _, _ in PROD_CLUSTERS:
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
        s3_row_bytes = p["token_payload_bytes"] // (2 * p["block_count"])
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

        for prefix, _, s1_dma in PROD_CLUSTERS:
            runtime_cluster = 0 if prefix == "c0" else 1
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
            skip_s3 = 1 if prefix == "c0" else 0
            s3_dma = 0 if prefix == "c0" else 2
            ctrl0 = (
                1
                | (skip_s3 << 2)
                | (p["s1_shape"] << 5)
                | (p["s1_shape"] << 7)
                | (s1_dma << 9)
                | (s3_dma << 11)
                | (runtime_cluster << 13)
            )
            ctrl1 = (
                1
                | (runtime_cluster << 13)
                | (1 << 14)
            )
            s2_prefetch_vd = ((1 | (3 << 1)) << (2 * 3)) if prefix == "c0" else 0
            s4_token_start = 0 if prefix == "c0" else p["s1_rows"]
            s4_m_exec = 3 if prefix == "c0" else 1

            lines += [
                f"__snax_bingo_moe_dynamic_expert_static_args_t *{static_name} = "
                f"(__snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t){static_l3};",
                f"uint8_t *{runtime_name} = (uint8_t *)(uintptr_t){runtime_l3};",
                f"memset({static_name}, 0, sizeof(*{static_name}));",
                f"memset({runtime_name}, 0, {self.HEADER_BYTES + 2 * self.SLOT_BYTES}u);",
                f"memset((void *)(uintptr_t){output_l3}, 0, {p['prod_output_bytes']}u);",
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
                f"{static_name}->s1_block_count = {p['block_count']}u;",
                f"{static_name}->s3_block_count = {p['block_count']}u;",
                f"{static_name}->indiv_K1 = {p['gate_K']}u;",
                f"{static_name}->indiv_N_per_block = {p['indiv_N_per_block']}u;",
                f"{static_name}->indiv_down_K1 = {p['down_K']}u;",
                f"{static_name}->indiv_down_N_per_block = {p['indiv_down_N_per_block']}u;",
                f"{static_name}->rescale_mult = 1u;",
                f"{static_name}->rescale_shift = 0u;",
                f"{static_name}->output_expert_stride_bytes = {p['prod_output_bytes']}u;",
                f"{static_name}->max_tokens_per_expert = {2 * self.NTOKENS}u;",
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
                f"{slot0_name}->ntokens = {self.NTOKENS}u;",
                f"{slot0_name}->m_s2_exec = 1u;",
                f"{slot0_name}->m_s4_exec = {s4_m_exec}u;",
                f"{slot0_name}->dma_slot_vd = {s2_prefetch_vd}u;",
                f"{slot0_name}->dma_slot_eids = 0u;",
                f"{slot1_name}->ctrl = {ctrl1}u;",
                f"{slot1_name}->expert_id = 0u;",
                f"{slot1_name}->token_ref_start = {self.NTOKENS}u;",
                f"{slot1_name}->ntokens = {self.NTOKENS}u;",
            ]
            for block in range(p["block_count"]):
                output_offset = block * p["bank_mode0_output_block_span"]
                lines += [
                    f"{slot0_name}->s1_call[{block}].valid = 1u;",
                    f"{slot0_name}->s1_call[{block}].output_D0_addr = "
                    f"{l1_d} + {output_offset}u;",
                    f"{slot0_name}->s1_call[{block}].N = {p['gate_N_s1']}u;",
                    f"{slot0_name}->s1_call[{block}].array_shape = {p['s1_shape']}u;",
                    f"{slot0_name}->s3_call[{block}].valid = {1 - skip_s3}u;",
                    f"{slot0_name}->s3_call[{block}].N = {p['down_N_s3_block']}u;",
                    f"{slot0_name}->s3_call[{block}].array_shape = {p['s1_shape']}u;",
                    f"{slot0_name}->s3_call[{block}].reserved = 0u;",
                ]
            lines += [
                f"{slot0_name}->s2_call.valid = 1u;",
                f"{slot0_name}->s2_call.token_start = {p['s1_rows']}u;",
                f"{slot0_name}->s2_call.reserved = 0u;",
                f"{slot0_name}->s2_call.M = 1u;",
                f"{slot0_name}->s2_call.N = {p['gate_N_s2']}u;",
                f"{slot0_name}->s2_call.array_shape = {p['s2_shape']}u;",
                f"{slot0_name}->s4_call.valid = 1u;",
                f"{slot0_name}->s4_call.token_start = {s4_token_start}u;",
                f"{slot0_name}->s4_call.reserved0 = 0u;",
                f"{slot0_name}->s4_call.reserved1 = 0u;",
                f"{slot0_name}->s4_call.M = {s4_m_exec}u;",
                f"{slot0_name}->s4_call.N = {p['down_N_s4']}u;",
                f"{slot0_name}->s4_call.array_shape = {p['s2_shape']}u;",
            ]
        return lines


def get_args():
    parser = argparse.ArgumentParser(description="Static single-slot MoE overlap test")
    parser.add_argument("--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_offload_file_name", type=str, required=True)
    return parser.parse_args()


def load_params(args):
    with args.cfg.open() as f:
        cfg = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hwcfg = hjson.loads(f.read())
    return derive_params({**cfg, **hwcfg})


def define_production_memory(p):
    """Allocate C0/C1 using the same per-cluster arena as production."""
    mh = {
        "input": BingoMemSymbol("moe_test_input_A"),
        "prod_slot_token_refs": BingoMemSymbol("moe_test_prod_slot_token_refs"),
    }
    for prefix, cluster, _ in PROD_CLUSTERS:
        mh[f"{prefix}_slot0_golden"] = BingoMemSymbol(
            f"moe_test_{prefix}_slot0_golden"
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
    """Execute slot0 fully, then store it while gathering six slot1 tokens."""
    setup = add_node(
        dfg,
        0,
        HOST_CORE,
        "__host_bingo_kernel_check_result",
        ProductionMoeDualClusterSetupArgs(p, mh),
        "SETUP_PRODUCTION_SLOT",
    )
    final_stores = []

    for prefix, cluster, _ in PROD_CLUSTERS:
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
        gather = add_node(
            dfg,
            cluster,
            DMA_CORE,
            "__snax_bingo_kernel_moe_dynamic_expert_gather_s1",
            slot_args,
            f"{prefix.upper()}_PROD_SLOT0_GATHER_6_TOKENS",
        )
        dfg.bingo_add_edge(runtime_to_l1, gather)

        s1_loads = []
        s1_computes = []
        for block in range(p["block_count"]):
            block_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
                slot0_addr, static_addr, pipeline_ctrl_addr, block
            )
            load = add_node(
                dfg,
                cluster,
                DMA_CORE,
                "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block",
                block_args,
                f"{prefix.upper()}_PROD_S1_LOAD_BLOCK_{block}",
            )
            compute_kernel = (
                "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block_pc"
                if block == 0
                else "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block"
            )
            compute = add_node(
                dfg,
                cluster,
                GEMM_CORE,
                compute_kernel,
                block_args,
                f"{prefix.upper()}_PROD_S1_COMPUTE_BLOCK_{block}",
            )
            if block == 0:
                config = add_node(
                    dfg,
                    cluster,
                    GEMM_CORE,
                    "__snax_bingo_kernel_moe_dynamic_expert_configure_gate_up_block0",
                    block_args,
                    f"{prefix.upper()}_PROD_S1_CONFIG_BLOCK0_DURING_LOAD0",
                )
                dfg.bingo_add_edge(gather, load)
                dfg.bingo_add_edge(gather, config)
                dfg.bingo_add_edge(config, compute)
            else:
                dfg.bingo_add_edge(s1_loads[-1], load)
                if block >= 2:
                    dfg.bingo_add_edge(s1_computes[-2], load)
                dfg.bingo_add_edge(s1_computes[-1], compute)
            dfg.bingo_add_edge(load, compute)
            s1_loads.append(load)
            s1_computes.append(compute)

        prefetch = add_node(
            dfg,
            cluster,
            DMA_CORE,
            "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down",
            slot_args,
            f"{prefix.upper()}_PROD_S2_DOWN_PREFETCH",
        )
        s2 = add_node(
            dfg,
            cluster,
            GEMM_CORE,
            "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full",
            slot_args,
            f"{prefix.upper()}_PROD_S2_COMPUTE_LAST_2_TOKENS",
        )
        dfg.bingo_add_edge(s1_computes[-1], prefetch)
        dfg.bingo_add_edge(s1_computes[-1], s2)

        s3_loads = []
        s3_computes = []
        for block in range(p["block_count"]):
            block_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
                slot0_addr, static_addr, pipeline_ctrl_addr, block
            )
            load = add_node(
                dfg,
                cluster,
                DMA_CORE,
                "__snax_bingo_kernel_moe_dynamic_expert_load_down_block",
                block_args,
                f"{prefix.upper()}_PROD_S3_LOAD_BLOCK_{block}",
            )
            compute_kernel = (
                "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block_pc"
                if block == 0
                else "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block"
            )
            compute = add_node(
                dfg,
                cluster,
                GEMM_CORE,
                compute_kernel,
                block_args,
                f"{prefix.upper()}_PROD_S3_COMPUTE_BLOCK_{block}",
            )
            if block == 0:
                config = add_node(
                    dfg,
                    cluster,
                    GEMM_CORE,
                    "__snax_bingo_kernel_moe_dynamic_expert_configure_down_block0",
                    block_args,
                    f"{prefix.upper()}_PROD_S3_CONFIG_BLOCK0_DURING_LOAD0",
                )
                for predecessor in (s2, prefetch):
                    dfg.bingo_add_edge(predecessor, load)
                    dfg.bingo_add_edge(predecessor, config)
                dfg.bingo_add_edge(config, compute)
            else:
                dfg.bingo_add_edge(s3_loads[-1], load)
                if block >= 2:
                    dfg.bingo_add_edge(s3_computes[-2], load)
                dfg.bingo_add_edge(s3_computes[-1], compute)
            dfg.bingo_add_edge(load, compute)
            s3_loads.append(load)
            s3_computes.append(compute)

        s4 = add_node(
            dfg,
            cluster,
            GEMM_CORE,
            "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full",
            slot_args,
            f"{prefix.upper()}_PROD_S4_COMPUTE_REMAINDER",
        )
        dfg.bingo_add_edge(s3_computes[-1], s4)

        prepare = add_node(
            dfg,
            cluster,
            DMA_CORE,
            "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1",
            slot_args,
            f"{prefix.upper()}_PROD_S4_PREFETCH_OR_PREPARE_STORE",
        )
        # Match the production DFG exactly. C0 still executes the skipped-S3
        # control chain before this node; optimizing that chain is a separate
        # hardware/Bingo change.
        dfg.bingo_add_edge(s3_loads[-1], prepare)
        store = add_node(
            dfg,
            cluster,
            DMA_CORE,
            "__snax_bingo_kernel_moe_dynamic_expert_store_and_gather_next",
            slot_args,
            f"{prefix.upper()}_PROD_SLOT0_STORE_GATHER_SLOT1_6_TOKENS",
        )
        dfg.bingo_add_edge(s4, store)
        dfg.bingo_add_edge(prepare, store)
        final_stores.append((prefix, store))

    done_checks = []
    for prefix, store in final_stores:
        check_name = f"{prefix.upper()}_PRODUCTION_SLOT_OUTPUT"
        done = add_node(
            dfg,
            0,
            HOST_CORE,
            "__host_bingo_kernel_check_result",
            HostBingoKernelCheckResultArgs(
                golden_data_addr=mh[f"{prefix}_slot0_golden"],
                output_data_addr=mh[f"{prefix}_prod_output_l3"],
                data_size=p["prod_slot_tokens"] * p["token_payload_bytes"],
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
    print(
        "Execute concurrent C0/C1 slot0 paths, then store slot0 while gathering "
        "six tokens for slot1; slot1 compute is intentionally absent"
    )
    add_production_slot_handoff_test(dfg, p, mh)
    return dfg


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    params = load_params(args)
    memory = define_production_memory(params)
    dfg = create_dfg(params, memory)
    dfg.bingo_compile_dfg(
        params["app_name"],
        args.output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["multi_cluster_MoE_test_data.h"],
    )

if __name__ == "__main__":
    main()
