# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# multi_cluster_MoE Bingo DFG - Dynamic Scheduling with iDMA + xDMA Parallel
#
# Architecture:
#   Cluster 0 (C0): Shared Expert 0 (gate+up SwiGLU + down proj)
#   Cluster 1 (C1): Shared Expert 1 (gate+up SwiGLU + down proj)
#   Cluster 2 (C2): INDIV_A - individual expert slot 0 (dynamic per round)
#   Cluster 3 (C3): INDIV_B - router GEMM + individual expert slot 1 (dynamic)
#
# DMA resource topology (hardware):
#   System iDMA (ONE per chip): base 0x05000000, triggered by CVA6 via sys_dma_memcpy().
#     In DFG: HOST lane (cluster=0, core=HOST_CORE_ID=2), kernel __host_bingo_kernel_idma.
#   Cluster xDMA: triggered by each target cluster DM core via CSR offset 960.
#     In DFG: target-cluster DM lane (core=DMA_CORE_ID=1), kernel __snax_bingo_kernel_xdma_1d_copy.
#   iDMA and xDMA are independent hardware engines and can run truly in parallel.
#   xDMA L3->L1 copies run on the destination cluster so the TCDM endpoint is local.
#
# DFG flow:
#
#   Router critical path — 先完成 router/shared 的全部 L3->L1 输入搬运，
#   再运行 Router GEMM + 写回。写回后 RouterSched 与 shared fused compute 并行，
#   但 RouterSched 不再与静态权重 DMA 争用 L3/互连。
#
#   Phase 0 — Router/shared 输入预加载（iDMA + xDMA 两路硬件并行）：
#     Router_B/A、C0/C1 gate/up/down/config/A 全部到达各自 L1 后，才启动 Router。
#     C2/C3 individual resident weights 延后到 RouterSched 之后搬运。
#
#   Phase 1 — Router 优先计算：
#     C3 Router GEMM → Router D 写回 L3
#     写回后同时启动 RouterSched 和 C0/C1 shared fused compute。
#
#   Phase 2 — CVA6 TopK（router 输出写回 L3 后）
#   Phase 3 — MoEPrepare：读 expert_token_counts 和 CAM 状态。pure HW build
#              直接驱动 RTL scheduler，并将 RTL compact plan 直接 lowered 到 C2/C3
#              的 L3 stage dynamic args；pure SW build 才使用 request/schedule ABI。
#   Phase 4 — MoEExecute：同步 runtime_state，并只把本轮有效 slot 的 dynamic
#              args 从 L3 flush 到 C2/C3 L1；跨 cluster slot wavefront 由 Bingo DFG
#              cross-edge 表达，slot 内 stage skip 由 device kernel 根据本 slot 参数处理。

import os
import sys
import argparse
import pathlib
import hjson
import networkx as nx

from moe_l15_layout import derive_workload_params

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(
    os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
)

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")


def parse_inorder_completion_core_ids() -> set:
    raw = os.environ.get("BINGO_INORDER_CORE_IDS", "0").strip().lower()
    if raw in {"", "none", "off", "false", "no"}:
        return set()
    if raw in {"all", "*"}:
        return {0, 1, 2}
    return {int(c.strip()) for c in raw.split(",") if c.strip()}


INORDER_COMPLETION_CORE_IDS = parse_inorder_completion_core_ids()
print(
    "BINGO_INORDER_CORE_IDS: "
    + (
        ",".join(str(c) for c in sorted(INORDER_COMPLETION_CORE_IDS))
        if INORDER_COMPLETION_CORE_IDS
        else "none"
    )
)

from bingo_dfg import BingoDFG
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import (
    SnaxBingoKernelDummyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelMoeRouterGemmS0Args,
    SnaxBingoKernelDualVcL15MoeFullArgs,
    SnaxBingoKernelMoeInitOutputPaddingArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelMoERouterScheduleArgs,
    HostBingoKernelMoEPrepareRequestArgs as LibHostBingoKernelMoEPrepareRequestArgs,
    HostBingoKernelMoEExecuteArgs as LibHostBingoKernelMoEExecuteArgs,
    SnaxBingoKernelMoeDynamicExpertBlockArgs,
)

# Per-slot runtime record only contains dynamic scheduler output plus compact
# bottom-level offsets. Static node constants live in one L1 static context per
# individual cluster. 192B is 64B-aligned and covers the compact C struct.
MOE_SCHEDULE_BYTES = 32768
MOE_RUNTIME_STATE_BYTES = 64
MOE_RUNTIME_HEADER_BYTES = MOE_RUNTIME_STATE_BYTES
MOE_SW_SCHED_ABI_COND = "!defined(MOE_ENABLE_HW_SCHEDULER)"
ENABLE_PHASE3_PHASE4 = True
# 当 ENABLE_PHASE3_PHASE4=True 时，此开关进一步控制是否展开 individual slot 执行链。
# 设为 False 时：DFG 在 node_execute 之后截止。pure HW build 中，node_prepare
# 已完成 RTL schedule + direct lowering + L3 stage args 写入，node_execute 完成
# runtime_state 同步和 L3->C2/C3 L1 dynamic args flush；只是不触发 C2/C3 上的
# GEMM/DMA slot 任务。
# 设为 True 时恢复完整 individual expert 执行链（默认完整 workload）。
ENABLE_INDIVIDUAL_SLOTS = True


# Use the canonical ABI mirror from libbingo. request/schedule buffers are now
# a pure-SW scheduler ABI; HW scheduler builds consume counts/CAM directly.
HostBingoKernelMoEPrepareRequestArgs = LibHostBingoKernelMoEPrepareRequestArgs
HostBingoKernelMoEExecuteArgs = LibHostBingoKernelMoEExecuteArgs


# =========================================================================
# Cluster / Core ID constants
# =========================================================================
CLUSTER_INDIV_A = 2  # C2: individual expert DFG copy A
CLUSTER_INDIV_B = 3  # C3: router GEMM + individual expert DFG copy B
CLUSTER_SHARED_0 = 0  # C0: Shared Expert 0 (alongside host)
CLUSTER_SHARED_1 = 1  # C1: Shared Expert 1
HOST_CLUSTER_ID = 0  # CVA6 host task queue (cluster 0, core 2)

GEMM_CORE_ID = 0  # VersaCore GEMM engine
DMA_CORE_ID = 1  # Cluster iDMA
HOST_CORE_ID = 2  # CVA6 Host (always cluster 0)

INDIV_CLUSTERS = [CLUSTER_INDIV_A, CLUSTER_INDIV_B]
SHARED_CLUSTERS = [CLUSTER_SHARED_0, CLUSTER_SHARED_1]


# =========================================================================
# Helpers
# =========================================================================


def addr_offset(handle, offset: int):
    """Return handle with added offset (zero-offset returns handle unchanged)."""
    if offset == 0:
        return handle
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, offset=handle.offset + offset)
    return f"{handle.get_c_var_name()} + {offset}"


def enforce_in_order_completion_per_core(bingo_dfg: BingoDFG) -> None:
    if not INORDER_COMPLETION_CORE_IDS:
        return
    topo = list(nx.topological_sort(bingo_dfg))
    normal = [
        n
        for n in topo
        if n.node_type == "normal" and n.assigned_core_id in INORDER_COMPLETION_CORE_IDS
    ]
    prev = {}
    for node in normal:
        lane = (
            node.assigned_chiplet_id,
            node.assigned_cluster_id,
            node.assigned_core_id,
        )
        p = prev.get(lane)
        if p is not None and not nx.has_path(bingo_dfg, p, node):
            bingo_dfg.bingo_add_edge(p, node)
        prev[lane] = node


def write_moe_config_header(header_path: str, params) -> None:
    router_mesh_row = params["meshRow"]
    if router_mesh_row == 0 or (router_mesh_row & (router_mesh_row - 1)) != 0:
        raise ValueError("pure-HW Router path requires a power-of-two meshRow")

    s1_row_bytes = params["indiv_D_tilesize"] // params["max_tokens_per_expert"]
    down_row_bytes = params["indiv_down_D_tilesize"] // params["max_tokens_per_expert"]
    lines = [
        "#pragma once",
        "#ifndef MOE_ENABLE_HW_SCHEDULER",
        '#include "moe_router_host.h"',
        "#endif",
        "#define MOE_ENABLE_MULTI_CLUSTER_MOE",
        f"#define MOE_HW_ROUTER_MESH_ROW {router_mesh_row}u",
        f"#define MOE_HW_ROUTER_MESH_COL {params['meshCol']}u",
        f"#define MOE_HW_ROUTER_M1 {params['router_M1']}u",
        f"#define MOE_HW_ROUTER_N1 {params['router_N1']}u",
        f"#define MOE_HW_ROUTER_MR_SHIFT {router_mesh_row.bit_length() - 1}u",
        f"#define MOE_HW_S1_BLOCKS {params['indiv_N2']}u",
        f"#define MOE_HW_S3_BLOCKS {params['indiv_down_N2']}u",
        f"#define MOE_HW_S1_ROW_BYTES {s1_row_bytes}u",
        f"#define MOE_HW_DOWN_ROW_BYTES {down_row_bytes}u",
        f"#define MOE_HW_A_ROW_STRIDE {params['A_token_padded_bytes']}u",
        f"#define MOE_HW_DOWN_ROW_STRIDE {params['A_token_padded_bytes']}u",
        f"#define MOE_HW_DOWN_HALF_ROW_BYTES {params['A_token_bytes'] // 2}u",
        f"#define MOE_HW_S1_N_BASE {params['indiv_N1'] * params['meshCol']}u",
        f"#define MOE_HW_S3_N_BASE {params['indiv_down_N1'] * params['meshCol']}u",
        "",
    ]
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def get_args():
    parser = argparse.ArgumentParser(description="Multi-Cluster MoE DFG Generator")
    parser.add_argument(
        "--cfg", type=pathlib.Path, default=pathlib.Path(current_dir) / "params.hjson"
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        default=pathlib.Path(ROOT_DIR)
        / "deps"
        / "snitch_cluster"
        / "target"
        / "snitch_cluster"
        / "cfg"
        / "snax_dual_versacore_int16x4_rebalanced_cluster.hjson",
    )
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument(
        "--output_offload_file_name", type=str, default="offload_bingo_hw.h"
    )
    return parser.parse_args()


def load_workload_config(args):
    with args.cfg.open() as f:
        p = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        h = hjson.loads(f.read())
    return {**p, **h}


# =========================================================================
# Memory handles
# =========================================================================


def define_memory_handles(params):
    mh = {}
    E = params["num_indiv_experts"]
    K = params["top_k"]
    M = params["M_total"]
    N2 = params["indiv_N2"]
    chip = 0

    # ------------------------------------------------------------------
    # L3 static data symbols (loaded from DRAM by build system)
    # ------------------------------------------------------------------
    # Router uses the same token-contiguous A as the dynamic path:
    # one logical token row is A_token_bytes payload + 32B padding. The router
    # GEMM reads only the payload and advances by A_token_padded_bytes per row.
    for m in range(params["router_M2"]):
        mh[f"L3_Sym_Router_A_tile_{m}"] = BingoMemSymbol(
            "input_A", offset=m * params["router_A_tile_padded_bytes"]
        )
    for n in range(params["router_N2"]):
        mh[f"L3_Sym_Router_B_tile_{n}"] = BingoMemSymbol(
            "router_B", offset=n * params["router_B_tilesize"]
        )

    # Individual expert weights (gate + up for SwiGLU, down for projection)
    # Address = base + E * expert_stride + n2 * n2_stride
    mh["L3_Sym_Indiv_Gate_B"] = BingoMemSymbol("indiv_gate_B")
    mh["L3_Sym_Indiv_Up_B"] = BingoMemSymbol("indiv_up_B")
    mh["L3_Sym_Indiv_Down_B"] = BingoMemSymbol("indiv_down_B")

    # One canonical padded token buffer is shared by router/shared/individual paths.
    mh["L3_Sym_Input_A"] = BingoMemSymbol("input_A")

    # Shared expert weight symbols (2 shared experts: C0=expert0, C1=expert1)
    mh["L3_Sym_Shared_Gate_B"] = BingoMemSymbol("shared_gate_B")
    mh["L3_Sym_Shared_Up_B"] = BingoMemSymbol("shared_up_B")
    mh["L3_Sym_Shared_Down_B"] = BingoMemSymbol("shared_down_B")
    # The only generated L15 control object is the shared fused-kernel S0 config.
    # The generated L15 configuration is a typed struct, not an array, so its
    # DMA source expression must take the address explicitly.
    mh["L3_Sym_L15_Shared_Dev_Cfg"] = BingoMemSymbol("&l15_dev_shared_s0_cfg")

    # ------------------------------------------------------------------
    # L3 dynamic allocations
    # ------------------------------------------------------------------
    mh["L3_Alloc_Router_Output"] = BingoMemAlloc(
        "l3_router_out",
        size=params["router_M2"] * params["router_D_total_bytes"],
        mem_level="L3",
    )
    mh["L3_Alloc_TopK_Indices"] = BingoMemAlloc(
        "l3_topk_idx",
        size=M * K * 2,  # uint16_t[M][K]
        mem_level="L3",
    )
    mh["L3_Alloc_TopK_Scores"] = BingoMemAlloc(
        "l3_topk_scores",
        size=M * K * 4,  # int32_t[M][K]
        mem_level="L3",
    )
    mh["L3_Alloc_Expert_Counts"] = BingoMemAlloc(
        "l3_expert_counts",
        size=E * 4,  # uint32_t[E]
        mem_level="L3",
    )
    # CAM state: 每个 indiv slot 当前驻留的 expert_id (int32_t, -1 = 无驻留)
    mh["L3_Alloc_CAM_State"] = BingoMemAlloc(
        "l3_cam_state",
        size=2 * 4,  # int32_t[num_indiv_slots=2]
        mem_level="L3",
    )
    # SW scheduler request/schedule buffers. Pure HW builds pass counts/CAM
    # directly to the RTL scheduler and do not allocate these buffers.
    mh["L3_Alloc_MoE_Request"] = BingoMemAlloc(
        "l3_moe_request",
        size=256,
        mem_level="L3",
        condition=MOE_SW_SCHED_ABI_COND,
    )
    mh["L3_Alloc_MoE_Schedule"] = BingoMemAlloc(
        "l3_moe_schedule",
        size=MOE_SCHEDULE_BYTES,
        mem_level="L3",
        condition=MOE_SW_SCHED_ABI_COND,
    )
    mh["L3_Alloc_MoE_Runtime_State"] = BingoMemAlloc(
        "l3_moe_runtime_state",
        size=MOE_RUNTIME_STATE_BYTES,
        mem_level="L3",
    )
    # L3 staging buffers contain only the runtime header and dynamic slot args.
    # Final static device contexts are generated once during host initialization
    # and copied separately to C2/C3 L1. Prepare writes these dynamic records;
    # Execute flushes only the header plus the active records to cluster L1.
    mh["L3_Alloc_C2_Stage"] = BingoMemAlloc(
        "l3_c2_stage",
        size=(
            MOE_RUNTIME_HEADER_BYTES
            + params["dynamic_slot_count"] * params["dynamic_arg_slot_bytes"]
        ),
        mem_level="L3",
    )
    mh["L3_Alloc_C3_Stage"] = BingoMemAlloc(
        "l3_c3_stage",
        size=(
            MOE_RUNTIME_HEADER_BYTES
            + params["dynamic_slot_count"] * params["dynamic_arg_slot_bytes"]
        ),
        mem_level="L3",
    )
    # Per-expert fixed-stride token table. Entry index:
    #   expert_id * max_tokens_per_expert + token_start_rank + local_t
    # This keeps device gather independent of a packed prefix-sum table.
    mh["L3_Alloc_Expert_Token_Ids"] = BingoMemAlloc(
        "l3_expert_token_ids",
        size=E * params["max_tokens_per_expert"] * 2,
        mem_level="L3",
    )
    num_se = params["num_shared_experts"]
    # Down projection outputs
    N2d = params["indiv_down_N2"]
    mh["L3_Alloc_Indiv_Down_Output"] = BingoMemAlloc(
        "l3_indiv_down_out",
        size=E * params["max_tokens_per_expert"] * params["A_token_padded_bytes"],
        mem_level="L3",
    )
    # Shared down output: 2 experts × mode1_padded_bytes each.
    # L15 kernel produces one mode1-padded output block per shared expert.
    mh["L3_Alloc_Shared_Down_Output"] = BingoMemAlloc(
        "l3_shared_down_out",
        size=num_se * params["l15_mode1_padded_bytes"],
        mem_level="L3",
    )

    # ------------------------------------------------------------------
    # L1 buffers
    # Shared/router weights are preloaded before Router compute. C2/C3 resident
    # individual weights are released after RouterSched.
    # SwiGLU uses L1_B_gate + L1_B_up simultaneously; down proj uses L1_B_down.
    # L1_D is reused as scratch for both SwiGLU and down GEMM outputs.
    # L1_down_D is a dedicated output buffer for down projection.
    #
    # Each individual slot has an L1_A padded token tile that stores up to
    # max_tokens_per_expert physical padded rows.
    # ------------------------------------------------------------------

    max_tok = params["max_tokens_per_expert"]

    # C2/C3 individual expert weight + scratch buffers
    for prefix, cid in [("C2_indiv", CLUSTER_INDIV_A), ("C3_indiv", CLUSTER_INDIV_B)]:
        # One contiguous runtime block: [64B active-state header][slot args].
        # MoEExecute can therefore flush both with one DMA per cluster.
        mh[f"{prefix}_Dyn_Args"] = BingoMemAlloc(
            f"{prefix.lower()}_dyn_args",
            size=(
                MOE_RUNTIME_HEADER_BYTES
                + params["dynamic_slot_count"] * params["dynamic_arg_slot_bytes"]
            ),
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_Static_Args"] = BingoMemAlloc(
            f"{prefix.lower()}_static_args",
            size=params["dynamic_arg_slot_bytes"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        # L1_A and L1_down_D preserve the 32B-per-row L15 bank rotation.
        mh[f"{prefix}_L1_A"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_a",
            size=max_tok * params["A_token_padded_bytes"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_L1_B_gate"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_b_gate",
            size=params["indiv_B_expert_stride"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_L1_B_up"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_b_up",
            size=params["indiv_B_expert_stride"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        # L1_B_down: colored left-half blocks followed by colored right-half blocks.
        mh[f"{prefix}_L1_B_down"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_b_down",
            size=params["indiv_down_B_expert_stride"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        # L1_D: holds ALL N2 SwiGLU output tiles contiguously → used as A for down GEMM
        mh[f"{prefix}_L1_D"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_d",
            size=N2 * params["indiv_D_tilesize"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        # L1_D1_scratch: Mode 0 D1 is a hardware-replicated copy of D0 (same SwiGLU result).
        # Use a separate 1-tile scratch buffer so the two writer ports never alias the same
        # TCDM address (avoids bank conflicts). This buffer is never read by SW.
        mh[f"{prefix}_L1_D1_scratch"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_d1_scratch",
            size=params["indiv_D_tilesize"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        # Mode-1 output rows: D0 half, D1 half, then 32 zero padding bytes.
        mh[f"{prefix}_L1_down_D"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_down_d",
            size=max_tok * params["A_token_padded_bytes"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )

    # C3 router buffers (router_B is a single tile; separate from indiv buffers)
    mh["C3_router_L1_A"] = BingoMemAlloc(
        "c3_router_l1_a",
        size=params["router_A_padded_bytes"],
        mem_level="L1",
        chip_id=chip,
        cluster_id=CLUSTER_INDIV_B,
    )
    mh["C3_router_L1_B"] = BingoMemAlloc(
        "c3_router_l1_b",
        size=params["router_B_total_bytes"],
        mem_level="L1",
        chip_id=chip,
        cluster_id=CLUSTER_INDIV_B,
    )
    mh["C3_router_L1_D"] = BingoMemAlloc(
        "c3_router_l1_d",
        size=params["router_D_total_bytes"],
        mem_level="L1",
        chip_id=chip,
        cluster_id=CLUSTER_INDIV_B,
    )

    # C0/C1 shared expert buffers — single contiguous L15 layout region per cluster.
    # The L15 kernel manages all internal tensor placement within this region.
    for prefix, cid in [
        ("C0", CLUSTER_SHARED_0),
        ("C1", CLUSTER_SHARED_1),
    ]:
        mh[f"{prefix}_L1_Layout"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_layout",
            size=params["l15_tcdm_size"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )

    return mh


# =========================================================================
# DFG construction
# =========================================================================


def create_dfg(params, mh):
    bingo_dfg = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=4,
        num_cores_per_cluster=2,
        is_host_as_acc=True,
        chiplet_ids=[0x00],
    )

    E = params["num_indiv_experts"]
    N2 = params["indiv_N2"]
    N2d = params["indiv_down_N2"]

    # Initialize only the 32-byte holes that the Mode-1 writers intentionally
    # skip. These four nodes run once when the static DFG starts; fused/shared
    # and dynamic/individual compute calls never clear output padding again.
    output_padding_init = {}
    for name, cluster_id, output_base in [
        (
            "C0",
            CLUSTER_SHARED_0,
            addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_mode1_d0"]),
        ),
        (
            "C1",
            CLUSTER_SHARED_1,
            addr_offset(mh["C1_L1_Layout"], params["l15_delta_local_mode1_d0"]),
        ),
        ("C2", CLUSTER_INDIV_A, mh["C2_indiv_L1_down_D"]),
        ("C3", CLUSTER_INDIV_B, mh["C3_indiv_L1_down_D"]),
    ]:
        node = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=cluster_id,
            assigned_core_id=DMA_CORE_ID,
            kernel_name="__snax_bingo_kernel_moe_init_output_padding",
            kernel_args=SnaxBingoKernelMoeInitOutputPaddingArgs(
                output_base=output_base,
                row_payload_bytes=params["A_token_bytes"],
                row_stride_bytes=params["A_token_padded_bytes"],
                rows=params["max_tokens_per_expert"],
            ),
        )
        bingo_dfg.bingo_add_node(node)
        output_padding_init[name] = node

    # C2 xDMA gathers only the 2048B payload because XDMA_WIDTH is 64B and the
    # 2080B physical row is not a legal 1D transfer length. Initialize its row
    # tails once; subsequent gathers preserve the 2080B stride and overwrite
    # only payload bytes. C3 iDMA continues to copy complete physical rows.
    node_c2_input_padding_init = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_A,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_moe_init_output_padding",
        kernel_args=SnaxBingoKernelMoeInitOutputPaddingArgs(
            output_base=mh["C2_indiv_L1_A"],
            row_payload_bytes=params["A_token_bytes"],
            row_stride_bytes=params["A_token_padded_bytes"],
            rows=params["max_tokens_per_expert"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c2_input_padding_init)

    # =====================================================================
    # Phase 0: Weight preload — 系统 iDMA (HOST lane) + cluster xDMA 真并行
    #
    # 硬件资源：
    #   系统 iDMA (ONE): 有目标cluster DM core 通过对应api触发。
    #   cluster xDMA: 由目标集群 DM core 通过 CSR 960 触发，目标 L1/TCDM 为本地端点。
    #   两条 lane 分属独立硬件，DFG 中无 cross-edge → 真正并行执行。
    #
    # iDMA path (target DM lane, 串行): Router input + shared gate/config/A。
    # C2/C3 individual gate weights are released only after RouterSched.
    #
    # xDMA path (target DM lane, 串行): 先加载 shared up/down weights；
    # C2/C3 individual up/down weights are released only after RouterSched.
    # =====================================================================

    # ---- iDMA PATH: Router B/A first, then gate_B for all clusters ----
    # 全部使用 __snax_bingo_kernel_idma_1d_copy，在目标 cluster DM core 上触发。

    # C0/C1 load their own shared-expert gate weights into the L15 B0 region.
    node_idma_c0_gate = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Shared_Gate_B"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_b0"]),
            size=params["l15_b_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c0_gate)

    node_idma_c1_gate = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Shared_Gate_B"], params["shared_B_expert_stride"]
            ),
            dst_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_local_b0"]),
            size=params["l15_b_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c1_gate)
    bingo_dfg.bingo_add_edge(node_idma_c0_gate, node_idma_c1_gate)  # 系统 iDMA 串行

    node_idma_c2_gate = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_A,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Indiv_Gate_B"],  # expert0, offset=0
            dst_addr=mh["C2_indiv_L1_B_gate"],
            size=params["indiv_B_expert_stride"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c2_gate)

    node_idma_c3_router = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Router_B_tile_0"],
            dst_addr=mh["C3_router_L1_B"],
            size=params["router_B_total_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c3_router)

    # Load Router B/A first on the system-iDMA chain, then finish both shared
    # gate matrices before the Router/shared preload barrier is released.
    node_c3_load_A = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Router_A_tile_0"],
            dst_addr=mh["C3_router_L1_A"],
            size=params["router_A_padded_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c3_load_A)
    bingo_dfg.bingo_add_edge(node_idma_c3_router, node_c3_load_A)
    bingo_dfg.bingo_add_edge(node_c3_load_A, node_idma_c0_gate)

    node_idma_c3_gate = BingoNode(  # individual iDMA chain tail
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Indiv_Gate_B"], 7 * params["indiv_B_expert_stride"]
            ),  # expert7
            dst_addr=mh["C3_indiv_L1_B_gate"],
            size=params["indiv_B_expert_stride"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c3_gate)
    bingo_dfg.bingo_add_edge(node_idma_c2_gate, node_idma_c3_gate)

    # ---- xDMA PATH (target DM lane): shared up_B + individual up/down_B ----
    # 全部使用 __snax_bingo_kernel_xdma_1d_copy，在目标 cluster DM core 上触发。
    # L3->L1/TCDM 搬运必须让目标 TCDM 作为本地端点，避免双远端 xDMA 事务。

    # C0/C1 load their own shared-expert up weights into the L15 B1 region.
    node_xdma_c0_up = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Shared_Up_B"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_b1"]),
            size=params["l15_b_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c0_up)

    # C1 B1 is serial after C0 B1 on the xDMA chain.
    node_xdma_c1_up = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Shared_Up_B"], params["shared_B_expert_stride"]
            ),
            dst_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_local_b1"]),
            size=params["l15_b_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c1_up)
    bingo_dfg.bingo_add_edge(node_xdma_c0_up, node_xdma_c1_up)

    node_xdma_c2_up = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_A,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Indiv_Up_B"],
            dst_addr=mh["C2_indiv_L1_B_up"],
            size=params["indiv_B_expert_stride"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c2_up)

    node_xdma_c2_down = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_A,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Indiv_Down_B"],
            dst_addr=mh["C2_indiv_L1_B_down"],
            size=params["indiv_down_B_expert_stride"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c2_down)
    bingo_dfg.bingo_add_edge(node_xdma_c2_up, node_xdma_c2_down)

    node_xdma_c3_up = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Indiv_Up_B"], 7 * params["indiv_B_expert_stride"]
            ),
            dst_addr=mh["C3_indiv_L1_B_up"],
            size=params["indiv_B_expert_stride"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c3_up)
    bingo_dfg.bingo_add_edge(node_xdma_c2_down, node_xdma_c3_up)

    node_xdma_c3_down = BingoNode(  # individual xDMA chain tail
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Indiv_Down_B"], 7 * params["indiv_down_B_expert_stride"]
            ),
            dst_addr=mh["C3_indiv_L1_B_down"],
            size=params["indiv_down_B_expert_stride"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c3_down)
    bingo_dfg.bingo_add_edge(node_xdma_c3_up, node_xdma_c3_down)

    # =====================================================================
    # Shared A token loading is part of the Router/shared preload barrier.
    # Its final dependencies are attached after both shared down-weight/config
    # chains have been built below.
    # =====================================================================

    node_c0_load_A = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Input_A"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_a"]),
            size=params["l15_a_data_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c0_load_A)

    node_c1_load_A = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Input_A"],
            dst_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_local_a"]),
            size=params["l15_a_data_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c1_load_A)

    # =====================================================================
    # Phase 1b: shared-expert GEMM.
    # Use the fused L15 kernel: Mode-0 SwiGLU and Mode-1 down projection run
    # inside one device node. Mode-0 is started and completed before Mode-1
    # streamer/VersaCore CSRs are programmed. All shared tensors (A, gate/up,
    # W2l/W2r) must therefore be staged in L1 before this node starts.
    # =====================================================================

    # ---- Phase 1b: C0/C1 compute via fused L15 kernels ----

    # ---- C0 Phase 1 W2 loading (xDMA on C0 DM core, starts after c0_b1) ----
    node_xdma_c0_w2l = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Shared_Down_B"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_w2l"]),
            size=params["l15_w2_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c0_w2l)
    bingo_dfg.bingo_add_edge(node_xdma_c0_up, node_xdma_c0_w2l)

    node_xdma_c0_w2r = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Shared_Down_B"], params["l15_w2_data_length"]
            ),
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_w2r"]),
            size=params["l15_w2_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c0_w2r)
    bingo_dfg.bingo_add_edge(node_xdma_c0_w2l, node_xdma_c0_w2r)

    node_c0_load_l15_cfg = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_L15_Shared_Dev_Cfg"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_cfg"]),
            size=params["l15_cfg_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c0_load_l15_cfg)
    bingo_dfg.bingo_add_edge(node_xdma_c0_w2r, node_c0_load_l15_cfg)

    # ---- C1 Phase 1 W2 loading (xDMA on C1 DM core, starts after c1_b1) ----
    node_xdma_c1_w2l = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Shared_Down_B"],
                params["shared_down_B_expert_stride"],
            ),
            dst_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_local_w2l"]),
            size=params["l15_w2_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c1_w2l)
    bingo_dfg.bingo_add_edge(node_xdma_c1_up, node_xdma_c1_w2l)

    node_xdma_c1_w2r = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Shared_Down_B"],
                params["shared_down_B_expert_stride"] + params["l15_w2_data_length"],
            ),
            dst_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_local_w2r"]),
            size=params["l15_w2_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c1_w2r)
    bingo_dfg.bingo_add_edge(node_xdma_c1_w2l, node_xdma_c1_w2r)

    node_c1_load_l15_cfg = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_L15_Shared_Dev_Cfg"],
            dst_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_cfg"]),
            size=params["l15_cfg_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c1_load_l15_cfg)
    bingo_dfg.bingo_add_edge(node_xdma_c1_w2r, node_c1_load_l15_cfg)

    # Finish every Router/shared L3->L1 transfer before Router compute starts.
    # The explicit chains also prevent the system iDMA and xDMA traffic from
    # overlapping RouterSched later in the graph.
    bingo_dfg.bingo_add_edge(node_xdma_c0_w2r, node_xdma_c1_up)
    bingo_dfg.bingo_add_edge(node_c1_load_l15_cfg, node_c0_load_A)
    bingo_dfg.bingo_add_edge(node_xdma_c1_w2r, node_c1_load_A)

    # Serialize the two small shared config copies on the single system-iDMA
    # path after both shared gate matrices are resident.
    bingo_dfg.bingo_add_edge(node_idma_c1_gate, node_c0_load_l15_cfg)
    bingo_dfg.bingo_add_edge(node_c0_load_l15_cfg, node_c1_load_l15_cfg)

    # --- C0: fused SwiGLU + down projection ---
    node_c0_shared_full = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=GEMM_CORE_ID,
        kernel_name="__snax_bingo_kernel_dual_vc_l15_moe_full",
        kernel_args=SnaxBingoKernelDualVcL15MoeFullArgs(
            shape_cfg_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_cfg"]),
            tcdm_base=mh["C0_L1_Layout"],
            rescale_mult=1,
            rescale_shift=0,
        ),
    )
    bingo_dfg.bingo_add_node(node_c0_shared_full)
    bingo_dfg.bingo_add_edge(node_c0_load_A, node_c0_shared_full)
    bingo_dfg.bingo_add_edge(node_c0_load_l15_cfg, node_c0_shared_full)
    bingo_dfg.bingo_add_edge(output_padding_init["C0"], node_c0_shared_full)

    node_c0_store_out = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=HOST_CLUSTER_ID,
        assigned_core_id=HOST_CORE_ID,
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=addr_offset(
                mh["C0_L1_Layout"], params["l15_delta_local_mode1_d0"]
            ),
            dst_addr=mh["L3_Alloc_Shared_Down_Output"],
            size=params["l15_mode1_padded_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c0_store_out)
    prev_c0 = node_c0_store_out

    # --- C1: fused SwiGLU + down projection ---
    node_c1_shared_full = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=GEMM_CORE_ID,
        kernel_name="__snax_bingo_kernel_dual_vc_l15_moe_full",
        kernel_args=SnaxBingoKernelDualVcL15MoeFullArgs(
            shape_cfg_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_cfg"]),
            tcdm_base=mh["C1_L1_Layout"],
            rescale_mult=1,
            rescale_shift=0,
        ),
    )
    bingo_dfg.bingo_add_node(node_c1_shared_full)
    bingo_dfg.bingo_add_edge(node_c1_load_A, node_c1_shared_full)
    bingo_dfg.bingo_add_edge(node_c1_load_l15_cfg, node_c1_shared_full)
    bingo_dfg.bingo_add_edge(output_padding_init["C1"], node_c1_shared_full)

    node_c1_store_out = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=HOST_CLUSTER_ID,
        assigned_core_id=HOST_CORE_ID,
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=addr_offset(
                mh["C1_L1_Layout"], params["l15_delta_local_mode1_d0"]
            ),
            dst_addr=addr_offset(
                mh["L3_Alloc_Shared_Down_Output"], params["l15_mode1_padded_bytes"]
            ),
            size=params["l15_mode1_padded_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c1_store_out)
    prev_c1 = node_c1_store_out

    # --- C3: router GEMM (Mode 1: split total N groups across two VCs) ---
    router_B_half_stride = params["router_B_vc_stride"]
    router_D_half = params["router_D_vc_stride"]
    node_c3_router_gemm = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=GEMM_CORE_ID,
        kernel_name="__snax_bingo_kernel_moe_router_gemm_s0",
        kernel_args=SnaxBingoKernelMoeRouterGemmS0Args(
            input_A_addr=mh["C3_router_L1_A"],
            input_B0_addr=mh["C3_router_L1_B"],  # N1-tile[0]
            input_B1_addr=addr_offset(
                mh["C3_router_L1_B"], router_B_half_stride
            ),  # N1-tile[1]
            output_D0_addr=mh["C3_router_L1_D"],  # left cols
            output_D1_addr=addr_offset(
                mh["C3_router_L1_D"], router_D_half
            ),  # right cols
            M=params["router_M1"],
            K=params["router_K1"],
            N=params["router_vc_N"],
            rescale_mult=1,
            rescale_shift=0,
        ),
    )
    bingo_dfg.bingo_add_node(node_c3_router_gemm)
    bingo_dfg.bingo_add_edge(node_c3_load_A, node_c3_router_gemm)
    bingo_dfg.bingo_add_edge(node_c0_load_A, node_c3_router_gemm)
    bingo_dfg.bingo_add_edge(node_c1_load_A, node_c3_router_gemm)

    node_c3_store_D = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=HOST_CLUSTER_ID,
        assigned_core_id=HOST_CORE_ID,
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=mh["C3_router_L1_D"],
            dst_addr=mh["L3_Alloc_Router_Output"],
            size=params["router_D_total_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c3_store_D)
    bingo_dfg.bingo_add_edge(node_c3_router_gemm, node_c3_store_D)

    # Shared compute uses only resident L1/TCDM data. Start it only after the
    # Router result has reached L3, so it can overlap RouterSched without adding
    # any L3 weight traffic to the RouterSched window.
    bingo_dfg.bingo_add_edge(node_c3_store_D, node_c0_shared_full)
    bingo_dfg.bingo_add_edge(node_c3_store_D, node_c1_shared_full)

    # =====================================================================
    # Phase 2: Host TopK (depends only on router output; shared expert
    # computation is independent and finishes in parallel)
    # =====================================================================
    node_topk = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=HOST_CLUSTER_ID,
        assigned_core_id=HOST_CORE_ID,
        kernel_name="__host_bingo_kernel_moe_router_schedule",
        kernel_args=HostBingoKernelMoERouterScheduleArgs(
            total_tokens=params["M_total"],
            hardware_output_buffer_addr=mh["L3_Alloc_Router_Output"],
            global_indices_out_addr=mh["L3_Alloc_TopK_Indices"],
            global_scores_out_addr=mh["L3_Alloc_TopK_Scores"],
            expert_token_counts_out_addr=mh["L3_Alloc_Expert_Counts"],
            expert_token_ids_addr=mh["L3_Alloc_Expert_Token_Ids"],
            max_tokens_per_expert=params["max_tokens_per_expert"],
            expert_number_each_layer=params["num_indiv_experts"],
            individual_expert_number_k=params["top_k"],
            mesh_row=params["meshRow"],
            mesh_col=params["meshCol"],
            router_m1=params["router_M1"],
            router_n1=params["router_N1"],
        ),
    )
    bingo_dfg.bingo_add_node(node_topk)
    bingo_dfg.bingo_add_edge(node_c3_store_D, node_topk)

    # Individual resident weights are not on the Router critical path. Release
    # both DMA chains only after RouterSched has consumed the router output.
    bingo_dfg.bingo_add_edge(node_topk, node_idma_c2_gate)
    bingo_dfg.bingo_add_edge(node_topk, node_xdma_c2_up)

    if not ENABLE_PHASE3_PHASE4:
        enforce_in_order_completion_per_core(bingo_dfg)
        return bingo_dfg

    # =====================================================================
    # Phase 3: MoEPrepare
    #
    # Pure HW path: consume expert_token_counts + CAM state, drive RTL
    # scheduler, then direct-lower compact plan entries into C2/C3 L3 stage
    # dynamic args. request/schedule buffers belong only to pure SW builds and
    # are not part of the HW scheduler ABI.
    # =====================================================================
    node_prepare = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=HOST_CLUSTER_ID,
        assigned_core_id=HOST_CORE_ID,
        kernel_name="__host_bingo_kernel_moe_prepare_request",
        kernel_args=HostBingoKernelMoEPrepareRequestArgs(
            expert_token_counts_addr=mh["L3_Alloc_Expert_Counts"],
            cam_state_addr=mh["L3_Alloc_CAM_State"],
            request_out_addr=mh["L3_Alloc_MoE_Request"],
            schedule_out_addr=mh["L3_Alloc_MoE_Schedule"],
            expert_token_ids_addr=mh["L3_Alloc_Expert_Token_Ids"],
            n_experts=params["num_indiv_experts"],
            topk_indices_l3=mh["L3_Alloc_TopK_Indices"],
            M_total=params["M_total"],
            top_k=params["top_k"],
            expert_token_counts_valid=1,
            runtime_state_addr=mh["L3_Alloc_MoE_Runtime_State"],
            c2_stage_base=addr_offset(
                mh["L3_Alloc_C2_Stage"], MOE_RUNTIME_HEADER_BYTES
            ),
            c3_stage_base=addr_offset(
                mh["L3_Alloc_C3_Stage"], MOE_RUNTIME_HEADER_BYTES
            ),
            dynamic_arg_slot_bytes=params["dynamic_arg_slot_bytes"],
            dynamic_num_slots=params["dynamic_slot_count"],
            c2_l1_a=mh["C2_indiv_L1_A"],
            c2_l1_d=mh["C2_indiv_L1_D"],
            c2_l1_down_d=mh["C2_indiv_L1_down_D"],
            c3_l1_a=mh["C3_indiv_L1_A"],
            c3_l1_d=mh["C3_indiv_L1_D"],
            c3_l1_down_d=mh["C3_indiv_L1_down_D"],
            A_token_bytes=params["A_token_bytes"],
            indiv_D_tile_bytes=params["indiv_D_tilesize"],
            indiv_down_D_tile_bytes=params["indiv_down_D_tilesize"],
            indiv_N_per_block=params["indiv_N1"] * params["meshCol"],
            indiv_down_N_per_block=params["indiv_down_N1"] * params["meshCol"],
            s1_block_count=N2,
            s3_block_count=N2d,
            max_tokens_per_expert=params["max_tokens_per_expert"],
        ),
    )
    bingo_dfg.bingo_add_node(node_prepare)
    bingo_dfg.bingo_add_edge(node_topk, node_prepare)

    # =====================================================================
    # Phase 4: MoEExecute + optional static slot execution
    #
    # HW scheduler direct path 下，node_prepare 已经生成 L3 stage dynamic
    # args；node_execute 只同步 runtime_state 并把有效 slot args flush 到
    # cluster L1。非 HW build 才在这里根据 schedule 生成 stage args。
    # ENABLE_INDIVIDUAL_SLOTS=False 时在这里截止，只测完整 scheduler+lowering；
    # True 时后续 static slot graph 才真正执行 S1/S2/S3/S4/store。跨 cluster
    # slot wavefront 由 Bingo DFG cross-edge 表达；slot 内 skip 仍由 device
    # kernel 根据本 slot ctrl 字段处理。
    # =====================================================================
    node_execute = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=HOST_CLUSTER_ID,
        assigned_core_id=HOST_CORE_ID,
        kernel_name="__host_bingo_kernel_moe_execute",
        kernel_args=HostBingoKernelMoEExecuteArgs(
            request_addr=mh["L3_Alloc_MoE_Request"],
            schedule_addr=mh["L3_Alloc_MoE_Schedule"],
            runtime_state_addr=mh["L3_Alloc_MoE_Runtime_State"],
            expert_token_ids_addr=mh["L3_Alloc_Expert_Token_Ids"],
            cam_state_addr=mh["L3_Alloc_CAM_State"],
            input_A_l3_base=mh["L3_Sym_Input_A"],
            indiv_gate_B_l3=mh["L3_Sym_Indiv_Gate_B"],
            indiv_up_B_l3=mh["L3_Sym_Indiv_Up_B"],
            indiv_down_B_l3=mh["L3_Sym_Indiv_Down_B"],
            c2_l1_b_gate=mh["C2_indiv_L1_B_gate"],
            c2_l1_b_up=mh["C2_indiv_L1_B_up"],
            c2_l1_b_down=mh["C2_indiv_L1_B_down"],
            c2_l1_a=mh["C2_indiv_L1_A"],
            c2_l1_d=mh["C2_indiv_L1_D"],
            c2_l1_down_d=mh["C2_indiv_L1_down_D"],
            c2_l1_d1_scratch=mh["C2_indiv_L1_D1_scratch"],
            c3_l1_b_gate=mh["C3_indiv_L1_B_gate"],
            c3_l1_b_up=mh["C3_indiv_L1_B_up"],
            c3_l1_b_down=mh["C3_indiv_L1_B_down"],
            c3_l1_a=mh["C3_indiv_L1_A"],
            c3_l1_d=mh["C3_indiv_L1_D"],
            c3_l1_down_d=mh["C3_indiv_L1_down_D"],
            c3_l1_d1_scratch=mh["C3_indiv_L1_D1_scratch"],
            output_l3_addr=mh["L3_Alloc_Indiv_Down_Output"],
            c2_active_state_l1_addr=mh["C2_indiv_Dyn_Args"],
            c3_active_state_l1_addr=mh["C3_indiv_Dyn_Args"],
            A_token_bytes=params["A_token_bytes"],
            indiv_B_expert_stride=params["indiv_B_expert_stride"],
            indiv_down_B_expert_stride=params["indiv_down_B_expert_stride"],
            indiv_B_tile_bytes=params["indiv_B_tilesize"],
            indiv_B_block_stride=params["indiv_B_block_stride"],
            indiv_D_tile_bytes=params["indiv_D_tilesize"],
            indiv_down_B_tile_bytes=params["indiv_down_B_tilesize"],
            indiv_down_B_block_stride=params["indiv_down_B_block_stride"],
            indiv_down_D_tile_bytes=params["indiv_down_D_tilesize"],
            s1_block_count=N2,
            s3_block_count=N2d,
            indiv_K1=params["indiv_K1"],
            indiv_N_per_block=params["indiv_N1"] * params["meshCol"],
            indiv_down_K1=params["indiv_down_K1"],
            indiv_down_N_per_block=params["indiv_down_N1"] * params["meshCol"],
            rescale_mult=1,
            rescale_shift=0,
            output_expert_stride_bytes=params["max_tokens_per_expert"]
            * params["A_token_padded_bytes"],
            max_tokens_per_expert=params["max_tokens_per_expert"],
            c2_static_args_base=mh["C2_indiv_Static_Args"],
            c3_static_args_base=mh["C3_indiv_Static_Args"],
            c2_dynamic_args_base=addr_offset(
                mh["C2_indiv_Dyn_Args"], MOE_RUNTIME_HEADER_BYTES
            ),
            c3_dynamic_args_base=addr_offset(
                mh["C3_indiv_Dyn_Args"], MOE_RUNTIME_HEADER_BYTES
            ),
            dynamic_arg_slot_bytes=params["dynamic_arg_slot_bytes"],
            dynamic_num_slots=params["dynamic_slot_count"],
            c2_stage_base=mh["L3_Alloc_C2_Stage"],
            c3_stage_base=mh["L3_Alloc_C3_Stage"],
        ),
    )
    bingo_dfg.bingo_add_node(node_execute)
    bingo_dfg.bingo_add_edge(node_prepare, node_execute)
    # MoEPrepare may overlap the post-RouterSched individual weight traffic,
    # but MoEExecute must not release slots before both DMA chains are ready.
    bingo_dfg.bingo_add_edge(node_idma_c3_gate, node_execute)
    bingo_dfg.bingo_add_edge(node_xdma_c3_down, node_execute)

    # Keep the Router -> scheduler -> individual path independent of shared
    # compute. A direct execute -> shared-store join makes the mini compiler
    # insert a blocked dummy-check into the host waiting queue, ahead of the
    # Router store. Join on each shared cluster's idle DM lane instead, then
    # release the corresponding host writeback with one dependency.
    for cluster_id, shared_compute, shared_store in (
        (CLUSTER_SHARED_0, node_c0_shared_full, node_c0_store_out),
        (CLUSTER_SHARED_1, node_c1_shared_full, node_c1_store_out),
    ):
        relay = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=cluster_id,
            assigned_core_id=DMA_CORE_ID,
            kernel_name="__snax_bingo_kernel_completion_relay",
            kernel_args=SnaxBingoKernelDummyArgs(0),
        )
        bingo_dfg.bingo_add_node(relay)
        bingo_dfg.bingo_add_edge(shared_compute, relay)
        bingo_dfg.bingo_add_edge(node_execute, relay)
        bingo_dfg.bingo_add_edge(relay, shared_store)
    bingo_dfg.bingo_add_edge(output_padding_init["C2"], node_execute)
    bingo_dfg.bingo_add_edge(output_padding_init["C3"], node_execute)
    bingo_dfg.bingo_add_edge(node_c2_input_padding_init, node_execute)

    def add_dynamic_slot_chain(prefix: str, cluster_id: int, slot: int, deps):
        dyn_arg_addr = addr_offset(
            mh[f"{prefix}_Dyn_Args"],
            MOE_RUNTIME_HEADER_BYTES + slot * params["dynamic_arg_slot_bytes"],
        )
        static_arg_addr = mh[f"{prefix}_Static_Args"]
        slot_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
            dyn_arg_addr, static_arg_addr, 0
        )

        def add_node(core_id, kernel_name, kernel_args):
            node = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=cluster_id,
                assigned_core_id=core_id,
                kernel_name=kernel_name,
                kernel_args=kernel_args,
            )
            bingo_dfg.bingo_add_node(node)
            return node

        gather_s1 = add_node(
            DMA_CORE_ID,
            "__snax_bingo_kernel_moe_dynamic_expert_gather_s1",
            slot_args,
        )
        for dep in deps:
            bingo_dfg.bingo_add_edge(dep, gather_s1)

        # S1: N2 个 load+compute 节点对，边搬边算流水线
        # skip_s1=1(cache hit) 时：所有 load/compute 节点直接跳过，token 由 compute_gate_up_full(S2) 处理
        s1_loads = []
        s1_computes = []
        for block in range(N2):
            block_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
                dyn_arg_addr, static_arg_addr, block
            )
            load = add_node(
                DMA_CORE_ID,
                "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block",
                block_args,
            )
            compute = add_node(
                GEMM_CORE_ID,
                "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block",
                block_args,
            )
            bingo_dfg.bingo_add_edge(gather_s1, load)
            if block > 0:
                bingo_dfg.bingo_add_edge(s1_loads[block - 1], load)
                bingo_dfg.bingo_add_edge(s1_computes[block - 1], compute)
            bingo_dfg.bingo_add_edge(load, compute)
            s1_loads.append(load)
            s1_computes.append(compute)

        prefetch_s2_down = add_node(
            DMA_CORE_ID,
            "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down",
            slot_args,
        )
        bingo_dfg.bingo_add_edge(s1_loads[-1], prefetch_s2_down)

        # S2: gate+up 全量 GEMM 节点
        # cache hit(skip_s1=1)：处理所有 token；否则处理 ntokens-shape_M 尾部 token
        compute_gate_up_full = add_node(
            GEMM_CORE_ID,
            "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full",
            slot_args,
        )
        bingo_dfg.bingo_add_edge(s1_computes[-1], compute_gate_up_full)

        # S3+S4: N2d 个 load+compute 节点对，边搬边算流水线
        # skip_s3=1(cache hit) 时：load/compute 节点全部跳过，所有 token 由 compute_down_full 处理
        s3_loads = []
        s3_computes = []
        for block in range(N2d):
            block_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
                dyn_arg_addr, static_arg_addr, block
            )
            load = add_node(
                DMA_CORE_ID,
                "__snax_bingo_kernel_moe_dynamic_expert_load_down_block",
                block_args,
            )
            compute = add_node(
                GEMM_CORE_ID,
                "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block",
                block_args,
            )
            bingo_dfg.bingo_add_edge(compute_gate_up_full, load)
            bingo_dfg.bingo_add_edge(prefetch_s2_down, load)
            if block > 0:
                bingo_dfg.bingo_add_edge(s3_loads[block - 1], load)
                bingo_dfg.bingo_add_edge(s3_computes[block - 1], compute)
            bingo_dfg.bingo_add_edge(load, compute)
            s3_loads.append(load)
            s3_computes.append(compute)

        prefetch_s4_next_s1 = add_node(
            DMA_CORE_ID,
            "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1",
            slot_args,
        )
        # This prefetch is intentionally overlapped with the current slot's
        # down-full compute. Do not serialize it after compute_down_full just to
        # make trace node durations look cleaner; that would change the workload.
        bingo_dfg.bingo_add_edge(s3_loads[-1], prefetch_s4_next_s1)

        # S4: down 全量 GEMM 节点
        # cache hit(skip_s3=1)：处理所有 token；否则处理 ntokens-shape_M 尾部 token
        compute_down_full = add_node(
            GEMM_CORE_ID,
            "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full",
            slot_args,
        )
        bingo_dfg.bingo_add_edge(s3_computes[-1], compute_down_full)

        store = add_node(
            DMA_CORE_ID,
            "__snax_bingo_kernel_moe_dynamic_expert_store",
            slot_args,
        )
        bingo_dfg.bingo_add_edge(compute_down_full, store)
        bingo_dfg.bingo_add_edge(prefetch_s4_next_s1, store)
        return store

    if ENABLE_INDIVIDUAL_SLOTS:
        prev_c2_stores = [node_execute]
        prev_c3_stores = [node_execute]
        for slot in range(params["dynamic_slot_count"]):
            c2_store = add_dynamic_slot_chain(
                "C2_indiv", CLUSTER_INDIV_A, slot, prev_c2_stores
            )
            c3_store = add_dynamic_slot_chain(
                "C3_indiv", CLUSTER_INDIV_B, slot, prev_c3_stores
            )
            prev_c2_stores = [c2_store]
            prev_c3_stores = [c3_store]

    enforce_in_order_completion_per_core(bingo_dfg)
    return bingo_dfg


# =========================================================================
# Main
# =========================================================================


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_workload_config(args)
    params = derive_workload_params(cfg)
    config_header_name = f"{params['app_name']}_config.h"
    write_moe_config_header(os.path.join(args.output_dir, config_header_name), params)
    mh = define_memory_handles(params)
    dfg = create_dfg(params, mh)
    data_header_name = f"{params['app_name']}_data.h"
    dfg.bingo_compile_dfg(
        params["app_name"],
        args.output_dir,
        args.output_offload_file_name,
        extra_include_header_list=[data_header_name, "moe_runtime_timing.h"],
        post_execute_code=["__host_bingo_moe_print_phase_timing();"],
        pre_host_include_header_list=[config_header_name],
        profile_kernel_prefix="__snax_bingo_kernel_moe_dynamic_expert_",
        profile_condition="MOE_RUNTIME_TIMING",
        profile_report_function="__host_bingo_moe_print_runtime_timing",
    )


if __name__ == "__main__":
    main()
