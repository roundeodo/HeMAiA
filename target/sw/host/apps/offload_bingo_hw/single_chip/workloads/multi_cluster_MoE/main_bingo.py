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
#   Phase 0 — 权重预加载（iDMA + xDMA 两路硬件并行）：
#     iDMA path (HOST lane, serial): 所有 gate_B + router_B (系统 iDMA 读通道)
#       C0_gate_B → C1_gate_B → C2_gate_B → C3_router_B → C3_gate_B
#     xDMA path (target DM lane, serial): shared up_B + individual up/down_B
#       C0_up_B → C1_up_B → C2_up_B → C2_down_B → C3_up_B → C3_down_B
#     shared down W2L/W2R 在 Phase 1b 中按 cluster 分别加载，随后启动 fused L15 node。
#     两条 lane 无 cross-dependency → DFG 中真正并行（两套独立硬件）。
#
#   Phase 1a — Token 搬运（iDMA + xDMA 两路并行，等两路 Phase 0 全部完成后开始）：
#     C0_shared_A: HOST lane iDMA;  C1_shared_A: C1_DM lane xDMA（两路并行）
#     C3_router_A: HOST lane iDMA，等 C0_load_A 完成后串行
#
#   Phase 1b — 各 cluster GEMM（B 已驻留，A 就绪后立即执行）：
#     C0: shared_expert0 gate+up SwiGLU → down proj
#     C1: shared_expert1 gate+up SwiGLU → down proj  [与 C0 并行]
#     C3: router GEMM                                  [与 C0/C1 并行]
#
#   Phase 2 — CVA6 TopK（router 输出写回 L3 后）
#   Phase 3 — MoEPrepare：读 expert_token_counts 和 CAM 状态。pure HW fast build
#              直接驱动 RTL scheduler，并将 RTL compact plan 直接 lowered 到 C2/C3
#              的 L3 stage dynamic args；SW/check build 才使用 request/schedule ABI。
#   Phase 4 — MoEExecute：同步 runtime_state，并只把本轮有效 slot 的 dynamic
#              args 从 L3 flush 到 C2/C3 L1；跨 cluster slot wavefront 由 Bingo DFG
#              cross-edge 表达，slot 内 stage skip 由 device kernel 根据本 slot 参数处理。

import os
import sys
import argparse
import pathlib
import hjson
import re
import struct
import networkx as nx

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
if os.environ.get("BINGO_ENFORCE_INORDER_PER_CORE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    if "BINGO_INORDER_CORE_IDS" not in os.environ:
        INORDER_COMPLETION_CORE_IDS = {0, 1, 2}

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
    BingoKernelArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdmaBroadcastArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelDualVcGemmFullArgs,
    SnaxBingoKernelDualVcSwigluFullArgs,
    SnaxBingoKernelDualVcL15MoeFullArgs,
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelGemmMinimalArgs,
    HostBingoKernelMoERouterScheduleArgs,
    HostBingoKernelMoEPrepareRequestArgs as LibHostBingoKernelMoEPrepareRequestArgs,
    HostBingoKernelMoEExecuteArgs as LibHostBingoKernelMoEExecuteArgs,
    SnaxBingoKernelMoeDynamicExpertBlockArgs,
)
# MOE_DYNAMIC_SLOT_COUNT: max tasks per cluster side.
# With MOE_MAX_EXPERTS=64 and greedy SPLIT: max tasks/cluster = 64 (1 per expert).
# Set to 32 conservatively (ceil(64/2) tasks per cluster under balanced assignment).
MOE_DYNAMIC_SLOT_COUNT = 32
# Per-slot runtime record only contains dynamic scheduler output plus compact
# bottom-level offsets. Static node constants live in one L1 static context per
# individual cluster. 192B is 64B-aligned and covers the compact C struct.
MOE_DYNAMIC_ARG_SLOT_BYTES = 192
MOE_DYNAMIC_STATIC_ARG_BYTES = 192
MOE_SCHEDULE_BYTES = 32768
MOE_RUNTIME_STATE_BYTES = 64
MOE_LEGACY_SCHED_ABI_COND = (
    "!defined(MOE_ENABLE_HW_SCHEDULER) || defined(MOE_ENABLE_HW_SCHEDULER_CHECK)"
)
ENABLE_PHASE3_PHASE4 = True
# 当 ENABLE_PHASE3_PHASE4=True 时，此开关进一步控制是否展开 individual slot 执行链。
# 设为 False 时：DFG 在 node_execute 之后截止。pure HW fast build 中，node_prepare
# 已完成 RTL schedule + direct lowering + L3 stage args 写入，node_execute 完成
# runtime_state 同步和 L3->C2/C3 L1 dynamic args flush；只是不触发 C2/C3 上的
# GEMM/DMA slot 任务。
# 设为 True 时恢复完整 individual expert 执行链（默认完整 workload）。
ENABLE_INDIVIDUAL_SLOTS = False


# Use the canonical ABI mirror from libbingo. Pure HW fast mode ignores the
# legacy request/schedule fields at runtime; they remain in the struct only for
# SW scheduler and HW-check builds.
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


def float_to_uint32_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


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


def patch_moe_header_preamble(header_path: str) -> None:
    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()

    required = (
        '#include "libbingo/bingo_api.h"\n'
        '#include "MoE_operator.h"\n'
        "#define MOE_OPERATOR_CUSTOM\n"
        "#define MOE_ENABLE_DYNAMIC_BASELINE\n"
        '#include "host.h"\n'
    )
    generated = '#include "libbingo/bingo_api.h"\n#include "host.h"\n'
    if required not in content and generated not in content:
        raise RuntimeError("Cannot locate generated host include preamble")
    if required not in content:
        content = content.replace(generated, required, 1)

    init_call_marker = "__host_moe_init_stage_templates("
    if init_call_marker not in content:
        pattern = re.compile(
            r"(?P<indent>\s*)(?P<var>args_host_chip00_\d+)"
            r"->scratchpad_ptr = \(uint64_t\)\(uintptr_t\)sp_host_\d+;\n"
            r"(?P=indent)host_arg_list_chip_00\[\d+\] = "
            r"\(uint64_t\)\(uintptr_t\)(?P=var);\n"
            r"(?P=indent)host_kernel_list_chip_00\[\d+\] = "
            r"\(uint64_t\)\(uintptr_t\)&__host_bingo_kernel_moe_execute;\n"
        )

        def _insert_stage_template_init(match: re.Match) -> str:
            indent = match.group("indent")
            var = match.group("var")
            return (
                match.group(0)
                + f"{indent}// One-time initialization of C2/C3 dynamic slot templates.\n"
                + f"{indent}if (__host_moe_init_stage_templates({var}) != BINGO_RET_SUCC) return BINGO_RET_FAIL;\n"
            )

        content, n = pattern.subn(_insert_stage_template_init, content, count=1)
        if n != 1:
            raise RuntimeError("Cannot locate MoEExecute args block for stage template init")

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================================================================
# Config parsing
# =========================================================================
head_path = os.path.join(current_dir, "MoE_common_variable.h")


def parse_header_config(path):
    cfg = {}
    pat = re.compile(r"#define\s+(\w+)\s+([-+]?[\d\.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    cfg[m.group(1)] = eval(m.group(2).strip())
                except Exception:
                    pass
    return cfg


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
    parser.add_argument("--emit_mini_golden", action="store_true")
    return parser.parse_args()


def load_workload_config(args):
    with args.cfg.open() as f:
        p = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        h = hjson.loads(f.read())
    return {**p, **h, "emit_mini_golden": args.emit_mini_golden}


# =========================================================================
# Workload parameters
# =========================================================================


def define_workload_params(**kw):
    moe = parse_header_config(head_path)
    data_type = 0
    # Support both old (snax_versacore_core_template) and new (snax_dual_versacore_int16x4_core_template) key
    core_tmpl = kw.get("snax_versacore_core_template") or kw.get(
        "snax_dual_versacore_int16x4_core_template"
    )
    if core_tmpl is None:
        raise KeyError("No versacore core template found in hwcfg")
    acc = core_tmpl["snax_acc_cfg"][0]
    ashape = kw["array_shape"]
    meshRow = acc["snax_versacore_spatial_unrolling"][data_type][ashape][0]
    tileSize = acc["snax_versacore_spatial_unrolling"][data_type][ashape][1]
    meshCol = acc["snax_versacore_spatial_unrolling"][data_type][ashape][2]
    meshRow_A = int(kw.get("A_meshRow", meshRow))
    tileSize_A = int(kw.get("A_tileSize", tileSize))

    p = {
        "app_name": "multi_cluster_MoE",
        "array_shape": ashape,
        "meshRow": meshRow,
        "tileSize": tileSize,
        "meshCol": meshCol,
        # Router GEMM
        "router_M1": kw["router_M1"],
        "router_N1": kw["router_N1"],
        "router_K1": kw["router_K1"],
        "router_M2": kw["router_M2"],
        "router_N2": kw["router_N2"],
        "router_K2": kw["router_K2"],
        # Individual expert gate+up SwiGLU (Mode 0)
        "indiv_M1": kw["indiv_M1"],
        "indiv_N1": kw["indiv_N1"],
        "indiv_K1": kw["indiv_K1"],
        "indiv_M2": kw["indiv_M2"],
        "indiv_N2": kw["indiv_N2"],
        "indiv_K2": kw["indiv_K2"],
        # Individual expert down projection (Mode 1)
        "indiv_down_M1": kw.get("indiv_down_M1", kw["indiv_M1"]),
        "indiv_down_N1": kw.get("indiv_down_N1", kw["indiv_N1"]),
        "indiv_down_K1": kw.get("indiv_down_K1", 4),
        "indiv_down_M2": kw.get("indiv_down_M2", kw["indiv_M2"]),
        "indiv_down_N2": kw.get("indiv_down_N2", kw["indiv_N2"]),
        "indiv_down_K2": kw.get("indiv_down_K2", 1),
        # MoE config from header
        "num_indiv_experts": int(moe["expert_number_each_layer"]),
        "top_k": int(moe["individual_expert_number_k"]),
        "M_total": kw["router_M2"] * kw["router_M1"] * meshRow_A,
        "max_dispatch_rounds": kw.get("max_dispatch_rounds", 4),
        "dynamic_slot_count": int(
            kw.get("moe_dynamic_slot_count", MOE_DYNAMIC_SLOT_COUNT)
        ),
        "dynamic_arg_slot_bytes": int(
            kw.get("moe_dynamic_arg_slot_bytes", MOE_DYNAMIC_ARG_SLOT_BYTES)
        ),
        # Shared expert gate+up SwiGLU (C0=expert0, C1=expert1)
        "shared_M1": kw["shared_M1"],
        "shared_N1": kw["shared_N1"],
        "shared_K1": kw["shared_K1"],
        "shared_M2": kw["shared_M2"],
        "shared_N2": kw["shared_N2"],
        "shared_K2": kw["shared_K2"],
        # Shared expert down projection
        "shared_down_M1": kw.get("shared_down_M1", kw["shared_M1"]),
        "shared_down_N1": kw.get("shared_down_N1", kw["shared_N1"]),
        "shared_down_K1": kw.get("shared_down_K1", 4),
        "shared_down_M2": kw.get("shared_down_M2", kw["shared_M2"]),
        "shared_down_N2": kw.get("shared_down_N2", kw["shared_N2"]),
        "shared_down_K2": kw.get("shared_down_K2", 1),
        "num_shared_experts": int(moe["shared_expert_number_k"]),
        # GEMM mode
        "addNonZeroC": kw["addNonZeroC"],
        "addZeroC": kw["addZeroC"],
        "accumPrevC": kw["accumPrevC"],
    }

    # Tile byte sizes
    # A: INT16 → 2 bytes/element
    # B: INT4 packed → tileSize*meshCol/2 bytes
    # D: INT32 → 4 bytes/element
    # Note: A tiles use meshRow_A/tileSize_A (may differ from hardware defaults);
    #       D tiles also use meshRow_A since A's row count determines D's row count.
    p["router_A_tilesize"] = (
        p["router_M1"] * p["router_K1"] * meshRow_A * tileSize_A * 2
    )
    p["router_B_tilesize"] = p["router_K1"] * p["router_N1"] * tileSize * meshCol // 2
    p["router_D_tilesize"] = (
        p["router_M1"] * p["router_N1"] * meshRow_A * meshCol * 2
    )  # INT16 output (2 bytes/element)
    p["indiv_A_tilesize"] = p["indiv_M1"] * p["indiv_K1"] * meshRow_A * tileSize_A * 2
    p["indiv_B_tilesize"] = p["indiv_K1"] * p["indiv_N1"] * tileSize * meshCol // 2
    p["indiv_D_tilesize"] = (
        p["indiv_M1"] * p["indiv_N1"] * meshRow_A * meshCol * 2
    )  # INT16 output (2 bytes/element)
    k_total_input = p["router_K2"] * p["router_K1"] * tileSize
    if k_total_input % tileSize_A != 0:
        raise ValueError("input K_total must be divisible by A_tileSize")
    k1_a = k_total_input // tileSize_A
    # A_token_bytes is one logical token vector (K_total INT16 elements).
    # input_A is token-contiguous in L3; gather_s1 packs selected token rows
    # into the K-block-major L1_A layout required by the streamer.
    meshCol_down = int(kw.get("down_meshCol", meshCol))
    down_vc_meshCol = meshCol
    if meshCol_down != 2 * down_vc_meshCol:
        raise ValueError(
            "dual-VC down projection expects down_meshCol == 2 * hw meshCol"
        )
    p["A_token_bytes"] = k1_a * tileSize_A * 2
    p["A_total_bytes"] = p["M_total"] * p["A_token_bytes"]
    p["A_token_padded_bytes"] = p["A_token_bytes"] + 32
    p["A_total_padded_bytes"] = p["M_total"] * p["A_token_padded_bytes"]
    p["router_A_tile_padded_bytes"] = (
        p["router_M1"] * meshRow_A * p["A_token_padded_bytes"]
    )
    p["router_A_padded_bytes"] = p["router_M2"] * p["router_A_tile_padded_bytes"]
    # Down projection tile sizes are per VC half. The logical output width is
    # meshCol_down, but B0/B1 are stored as two contiguous hw-meshCol halves.
    p["indiv_down_A_tilesize"] = (
        p["indiv_down_M1"] * p["indiv_down_K1"] * meshRow_A * tileSize_A * 2
    )
    p["indiv_down_B_tilesize"] = (
        p["indiv_down_K1"] * p["indiv_down_N1"] * tileSize * down_vc_meshCol // 2
    )
    p["indiv_down_D_tilesize"] = (
        p["indiv_down_M1"]
        * p["indiv_down_N1"]
        * meshRow_A
        * down_vc_meshCol
        * 2  # INT16 output (2 bytes/element)
    )
    p["shared_down_A_tilesize"] = (
        p["shared_down_M1"] * p["shared_down_K1"] * meshRow_A * tileSize_A * 2
    )
    p["shared_down_B_tilesize"] = (
        p["shared_down_K1"] * p["shared_down_N1"] * tileSize * down_vc_meshCol // 2
    )
    p["shared_down_D_tilesize"] = (
        p["shared_down_M1"]
        * p["shared_down_N1"]
        * meshRow_A
        * down_vc_meshCol
        * 2  # INT16 output (2 bytes/element)
    )

    # L3 stride constants (INT4 packed → ÷2)
    p["indiv_B_expert_stride"] = (
        p["indiv_N2"]
        * p["indiv_K2"]
        * p["indiv_K1"]
        * p["indiv_N1"]
        * tileSize
        * meshCol
        // 2
    )
    p["indiv_B_n2_stride"] = (
        p["indiv_K2"] * p["indiv_K1"] * p["indiv_N1"] * tileSize * meshCol // 2
    )
    p["indiv_down_B_expert_stride"] = (
        2
        * p["indiv_down_N2"]
        * p["indiv_down_K2"]
        * p["indiv_down_K1"]
        * p["indiv_down_N1"]
        * tileSize
        * down_vc_meshCol
        // 2
    )
    p["indiv_down_B_n2_stride"] = (
        p["indiv_down_K2"]
        * p["indiv_down_K1"]
        * p["indiv_down_N1"]
        * tileSize
        * down_vc_meshCol
        // 2
    )
    p["shared_B_tilesize"] = p["shared_K1"] * p["shared_N1"] * tileSize * meshCol // 2
    p["shared_D_tilesize"] = (
        p["shared_M1"] * p["shared_N1"] * meshRow_A * meshCol * 2
    )  # INT16 output (2 bytes/element)
    p["shared_B_expert_stride"] = (
        p["shared_N2"]
        * p["shared_K2"]
        * p["shared_K1"]
        * p["shared_N1"]
        * tileSize
        * meshCol
        // 2
    )
    p["shared_B_n2_stride"] = (
        p["shared_K2"] * p["shared_K1"] * p["shared_N1"] * tileSize * meshCol // 2
    )
    p["shared_down_B_expert_stride"] = (
        2
        * p["shared_down_N2"]
        * p["shared_down_K2"]
        * p["shared_down_K1"]
        * p["shared_down_N1"]
        * tileSize
        * down_vc_meshCol
        // 2
    )
    p["shared_down_B_n2_stride"] = (
        p["shared_down_K2"]
        * p["shared_down_K1"]
        * p["shared_down_N1"]
        * tileSize
        * down_vc_meshCol
        // 2
    )
    router_total_n_groups = p["router_N2"] * p["router_N1"]
    if router_total_n_groups % 2 != 0:
        raise ValueError(
            "dual-VC router split expects router_N2 * router_N1 to be even"
        )
    p["router_vc_N"] = router_total_n_groups // 2
    p["router_B_total_bytes"] = p["router_N2"] * p["router_B_tilesize"]
    p["router_D_total_bytes"] = p["router_N2"] * p["router_D_tilesize"]
    p["router_B_vc_stride"] = (
        p["router_vc_N"] * p["router_K2"] * p["router_K1"] * tileSize * meshCol // 2
    )
    p["router_D_vc_stride"] = (
        p["router_vc_N"]
        * p["router_M1"]
        * meshRow_A
        * meshCol
        * 2  # INT16 output (2 bytes/element)
    )
    hidden_indiv = p["indiv_N2"] * p["indiv_N1"] * meshCol
    hidden_shared = p["shared_N2"] * p["shared_N1"] * meshCol
    k_indiv_down = p["indiv_down_K2"] * p["indiv_down_K1"] * tileSize
    k_shared_down = p["shared_down_K2"] * p["shared_down_K1"] * tileSize
    if k_indiv_down != hidden_indiv:
        raise ValueError("individual down K must match individual gate/up hidden width")
    if k_shared_down != hidden_shared:
        raise ValueError("shared down K must match shared gate/up hidden width")
    # Max tokens any single expert can receive = M_total (worst case all tokens to one expert)
    p["max_tokens_per_expert"] = p["M_total"]

    # ------------------------------------------------------------------
    # L15 layout parameters (must match multi_cluster_MoE_datagen.py).
    # L15 uses the full input K dimension (same as router K), not shared expert K.
    # ------------------------------------------------------------------
    _l15_a_pad = 32  # L15_LAYOUT["a_pad"]
    _k0_total = p["router_K2"] * p["router_K1"] * tileSize  # = K_total input
    _k0_bytes = _k0_total * 2  # int16 → 2 B/elem
    _a_row_stride = _k0_bytes + _l15_a_pad  # = 2080 B
    _n0_total_s0 = p["shared_N2"] * p["shared_N1"] * 4  # s0_meshCol=4
    # Logical down output width after concatenating the two VC output halves.
    _n1_total = p["shared_down_N2"] * p["shared_down_N1"] * meshCol_down
    if _n1_total % 2 != 0:
        raise ValueError(
            "L15 dual-VC down output width must split evenly across two VCs"
        )
    _n1_per_vc = _n1_total // 2
    _k0_s0_tiles = _k0_total // 8
    _k1_s0_tiles = _n0_total_s0 // 8  # k1=n0 for mode1 input
    _n0_s0_tiles = _n0_total_s0 // 4
    _n1_s0_tiles_per_vc = _n1_per_vc // 4
    p["l15_a_row_stride"] = _a_row_stride
    p["l15_b_data_length"] = _k0_s0_tiles * _n0_s0_tiles * 16  # bytes (uint8)
    p["l15_w2_data_length"] = _k1_s0_tiles * _n1_s0_tiles_per_vc * 16
    p["l15_a_data_bytes"] = p["M_total"] * _a_row_stride  # M_total * a_row_stride bytes

    # Dynamic L15 TCDM layout offsets: matches place_tensors in multi_cluster_MoE_datagen.py.
    # L15_LAYOUT: b1_color=272, w2l_color=128, m1d0_color=256; all others 0, align=1024.
    def _l15_col(offset, color=0, align=1024):
        return ((int(offset) + align - 1) // align) * align + int(color)

    _w_bytes = p["l15_b_data_length"]  # k0_s0_tiles * n0_s0_tiles * 16
    _w2_bytes = p["l15_w2_data_length"]  # k1_s0_tiles * n1_s0_tiles_per_vc * 16
    _a_bytes = p["l15_a_data_bytes"]  # M_total * a_row_stride
    _mode0_d_bytes = (
        p["M_total"] * _n0_total_s0 * 2
    )  # M_total * n0_total * sizeof(int16)
    p["l15_mode0_output_bytes"] = _mode0_d_bytes
    p["l15_mode0_row_bytes"] = _n0_total_s0 * 2
    p["l15_delta_local_b0"] = 0
    p["l15_delta_local_b1"] = _l15_col(p["l15_delta_local_b0"] + _w_bytes, 272)
    p["l15_delta_local_w2l"] = _l15_col(p["l15_delta_local_b1"] + _w_bytes, 128)
    p["l15_delta_local_w2r"] = _l15_col(p["l15_delta_local_w2l"] + _w2_bytes)
    p["l15_delta_local_a"] = _l15_col(p["l15_delta_local_w2r"] + _w2_bytes)
    _delta_d0 = _l15_col(p["l15_delta_local_a"] + _a_bytes)
    p["l15_delta_local_d0"] = _delta_d0
    p["l15_delta_local_mode1_d0"] = _l15_col(_delta_d0 + _mode0_d_bytes, 256)
    # Mode-1 output follows the standalone L15 reference layout: each token row
    # uses the same padded stride as Mode-0 input A.
    _mode1_row_stride = _a_row_stride
    if _mode1_row_stride < _n1_total * 2:
        raise ValueError("L15 Mode-1 output row stride is smaller than logical output")
    p["l15_mode1_payload_bytes_per_row"] = _n1_total * 2
    p["l15_mode1_padded_bytes"] = p["M_total"] * _mode1_row_stride
    if p["l15_mode1_payload_bytes_per_row"] % 64 != 0:
        raise ValueError("L15 Mode-1 payload row bytes must be 64-byte aligned")
    p["l15_delta_cfg"] = p["l15_delta_local_mode1_d0"] + p["l15_mode1_padded_bytes"]
    p["l15_cfg_bytes"] = 91 * 4  # moe_l15_shape_cfg_t = 91 int32_t values
    p["l15_tcdm_size"] = p["l15_delta_cfg"] + p["l15_cfg_bytes"]

    return p


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
    # one logical token row is 2048B payload + 32B padding. The router GEMM
    # kernel reads only the payload with a 2080B row stride, so its input and
    # datagen golden are derived from the same A_phys array.
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

    # input_A: token-contiguous physical layout. Dynamic gather_s1 maps token ids
    # to contiguous logical vectors, then packs them into K-block-major L1_A.
    mh["L3_Sym_Input_A"] = BingoMemSymbol("input_A")

    # Shared expert weight symbols (2 shared experts: C0=expert0, C1=expert1)
    mh["L3_Sym_Shared_Gate_B"] = BingoMemSymbol("shared_gate_B")
    mh["L3_Sym_Shared_Up_B"] = BingoMemSymbol("shared_up_B")
    mh["L3_Sym_Shared_Down_B"] = BingoMemSymbol("shared_down_B")
    # L15 layout data symbols (generated by multi_cluster_MoE_datagen.py).
    # These are static arrays in L3 consumed directly by the L15 kernel.
    _a_row = params["l15_a_row_stride"]
    mh["L3_Sym_Layout_W"] = BingoMemSymbol("layout_W")
    mh["L3_Sym_Layout_V"] = BingoMemSymbol("layout_V")
    mh["L3_Sym_Layout_W2L"] = BingoMemSymbol("layout_W2_left")
    mh["L3_Sym_Layout_W2R"] = BingoMemSymbol("layout_W2_right")
    mh["L3_Sym_Layout_A"] = BingoMemSymbol(f"layout_A_row_stride_{_a_row}")
    # Flat int32_t shape-config array for S0 (array_shape=0, matches moe_l15_shape_cfg_t)
    mh["L3_Sym_L15_S0_Dev_Cfg"] = BingoMemSymbol("l15_dev_s0_cfg")
    # Shared-expert variant: m_tiles=4 so kernel covers all 32 tokens per call
    mh["L3_Sym_L15_Shared_Dev_Cfg"] = BingoMemSymbol("l15_dev_shared_s0_cfg")

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
    # Legacy request buffer: only used by SW scheduler / HW-check builds.
    # Pure HW fast build passes counts/CAM directly to the RTL scheduler.
    mh["L3_Alloc_MoE_Request"] = BingoMemAlloc(
        "l3_moe_request",
        size=256,
        mem_level="L3",
        condition=MOE_LEGACY_SCHED_ABI_COND,
    )
    mh["L3_Alloc_MoE_Schedule"] = BingoMemAlloc(
        "l3_moe_schedule",
        size=MOE_SCHEDULE_BYTES,
        mem_level="L3",
        condition=MOE_LEGACY_SCHED_ABI_COND,
    )
    mh["L3_Alloc_MoE_Runtime_State"] = BingoMemAlloc(
        "l3_moe_runtime_state",
        size=MOE_RUNTIME_STATE_BYTES,
        mem_level="L3",
    )
    # L3 staging buffers for dynamic slot args.
    # The static template is initialized here first and DMA-copied to cluster L1.
    # Phase 4 then rewrites the dynamic fields in L3 and flushes the slot records
    # to the L1 runtime args consumed by the device kernels.
    mh["L3_Alloc_C2_Stage"] = BingoMemAlloc(
        "l3_c2_stage",
        size=MOE_DYNAMIC_SLOT_COUNT * MOE_DYNAMIC_ARG_SLOT_BYTES,
        mem_level="L3",
    )
    mh["L3_Alloc_C3_Stage"] = BingoMemAlloc(
        "l3_c3_stage",
        size=MOE_DYNAMIC_SLOT_COUNT * MOE_DYNAMIC_ARG_SLOT_BYTES,
        mem_level="L3",
    )
    mh["L3_Alloc_Expert_Token_Offsets"] = BingoMemAlloc(
        "l3_expert_token_offsets",
        size=(E + 1) * 4,
        mem_level="L3",
    )
    mh["L3_Alloc_Expert_Token_Ids"] = BingoMemAlloc(
        "l3_expert_token_ids",
        size=M * K * 2,
        mem_level="L3",
    )
    mh["L3_Alloc_Expert_Token_Kpos"] = BingoMemAlloc(
        "l3_expert_token_kpos",
        size=M * K * 2,
        mem_level="L3",
    )
    # SwiGLU output: [E][N2] tiles, INT16 each (replaces separate gate/up outputs)
    mh["L3_Alloc_SwiGLU_Output"] = BingoMemAlloc(
        "l3_swiglu_out",
        size=E * N2 * params["indiv_D_tilesize"],
        mem_level="L3",
    )
    # Shared expert SwiGLU output (indexed [se_id][n2_idx])
    num_se = params["num_shared_experts"]
    N2s = params["shared_N2"]
    mh["L3_Alloc_Shared_SwiGLU_Output"] = BingoMemAlloc(
        "l3_shared_swiglu_out",
        size=num_se * N2s * params["shared_D_tilesize"],
        mem_level="L3",
    )
    # Down projection outputs
    N2d = params["indiv_down_N2"]
    N2sd = params["shared_down_N2"]
    mh["L3_Alloc_Indiv_Down_Output"] = BingoMemAlloc(
        "l3_indiv_down_out",
        size=2 * E * N2d * params["indiv_down_D_tilesize"],
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
    # Phase 0 preloads all gate+up+down weight tiles for each resident expert.
    # SwiGLU uses L1_B_gate + L1_B_up simultaneously; down proj uses L1_B_down.
    # L1_D is reused as scratch for both SwiGLU and down GEMM outputs.
    # L1_down_D is a dedicated output buffer for down projection.
    #
    # Each individual slot (C2=slot0, C3=slot1) also has:
    #   L1_EDT:  Expert Dispatch Table written by host at runtime.
    #            Contains expert_id, token_count, and per-token L3 offsets.
    #            Size = sizeof(slot_edt_t) = 4 + 4 + MAX_TOKENS*4 bytes.
    #   L1_A:    Compact A tile for dynamic GEMM. It stores up to
    #            max_tokens_per_expert logical token vectors.
    # ------------------------------------------------------------------

    # EDT size: expert_id(4) + token_count(4) + token_l3_offsets[max_tokens](4 each)
    max_tok = params["max_tokens_per_expert"]
    edt_size = 4 + 4 + max_tok * 4  # bytes

    # C2/C3 individual expert weight + scratch buffers
    for prefix, cid in [("C2_indiv", CLUSTER_INDIV_A), ("C3_indiv", CLUSTER_INDIV_B)]:
        # EDT: written by host ExpertDispatch kernel at runtime
        mh[f"{prefix}_L1_EDT"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_edt",
            size=edt_size,
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_Dyn_Args"] = BingoMemAlloc(
            f"{prefix.lower()}_dyn_args",
            size=params["dynamic_slot_count"] * params["dynamic_arg_slot_bytes"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_Static_Args"] = BingoMemAlloc(
            f"{prefix.lower()}_static_args",
            size=MOE_DYNAMIC_STATIC_ARG_BYTES,
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_Active_State"] = BingoMemAlloc(
            f"{prefix.lower()}_active_state",
            size=MOE_RUNTIME_STATE_BYTES,
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        # L1_A: sized for max_tokens tokens (runtime fetch fills 1..max_tok tokens).
        # S3/S4 down outputs are written to L1_down_D in one unified full-N row layout.
        mh[f"{prefix}_L1_A"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_a",
            size=max_tok * params["A_token_bytes"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_L1_B_gate"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_b_gate",
            size=N2 * params["indiv_B_tilesize"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        mh[f"{prefix}_L1_B_up"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_b_up",
            size=N2 * params["indiv_B_tilesize"],
            mem_level="L1",
            chip_id=chip,
            cluster_id=cid,
        )
        # L1_B_down: left-half tiles (N2d) + right-half tiles (N2d) contiguous
        mh[f"{prefix}_L1_B_down"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_b_down",
            size=2 * N2d * params["indiv_down_B_tilesize"],
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
        # L1_down_D: D0 (VC0, left N cols) + D1 (VC1, right N cols) side by side
        mh[f"{prefix}_L1_down_D"] = BingoMemAlloc(
            f"{prefix.lower()}_l1_down_d",
            size=2 * N2d * params["indiv_down_D_tilesize"],
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

    E = params["num_indiv_experts"]  # 8
    N2 = params["indiv_N2"]  # 2
    N2s = params["shared_N2"]  # 2
    N2d = params["indiv_down_N2"]  # down projection N2
    N2sd = params["shared_down_N2"]  # shared down projection N2

    # =====================================================================
    # Phase 0: Weight preload — 系统 iDMA (HOST lane) + cluster xDMA 真并行
    #
    # 硬件资源：
    #   系统 iDMA (ONE): 有目标cluster DM core 通过对应api触发。
    #   cluster xDMA: 由目标集群 DM core 通过 CSR 960 触发，目标 L1/TCDM 为本地端点。
    #   两条 lane 分属独立硬件，DFG 中无 cross-edge → 真正并行执行。
    #
    # iDMA path (target DM lane, 串行): 加载所有 gate_B 权重 + router_B
    #   C0_gate_B → C1_gate_B → C2_gate_B → C3_router_B → C3_gate_B
    #
    # xDMA path (target DM lane, 串行): 加载 shared up_B + individual up/down_B 权重
    #   C0_up_B → C1_up_B → C2_up_B → C2_down_B → C3_up_B → C3_down_B
    # shared down W2L/W2R 不在 Phase 0 这条链里；它们在 Phase 1b 中按
    # cluster 分别加载，并作为 fused shared L15 compute node 的输入依赖。
    # =====================================================================

    # ---- iDMA PATH: gate_B for all clusters + router_B ----
    # 全部使用 __snax_bingo_kernel_idma_1d_copy，在目标 cluster DM core 上触发。

    # C0: load B0 (layout_W / gate weights) into L15 layout region
    node_idma_c0_gate = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Layout_W"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_b0"]),
            size=params["l15_b_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c0_gate)

    # C1: load B0 (layout_W / gate weights) into L15 layout region
    node_idma_c1_gate = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Layout_W"],
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
            size=N2 * params["indiv_B_tilesize"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c2_gate)
    bingo_dfg.bingo_add_edge(node_idma_c1_gate, node_idma_c2_gate)

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
    bingo_dfg.bingo_add_edge(node_idma_c2_gate, node_idma_c3_router)

    node_idma_c3_gate = BingoNode(  # iDMA Phase 0 最后一个节点
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Indiv_Gate_B"], 7 * params["indiv_B_expert_stride"]
            ),  # expert7
            dst_addr=mh["C3_indiv_L1_B_gate"],
            size=N2 * params["indiv_B_tilesize"],
        ),
    )
    bingo_dfg.bingo_add_node(node_idma_c3_gate)
    bingo_dfg.bingo_add_edge(node_idma_c3_router, node_idma_c3_gate)

    # ---- xDMA PATH (target DM lane): shared up_B + individual up/down_B ----
    # 全部使用 __snax_bingo_kernel_xdma_1d_copy，在目标 cluster DM core 上触发。
    # L3->L1/TCDM 搬运必须让目标 TCDM 作为本地端点，避免双远端 xDMA 事务。

    # C0: B1 (layout_V / up weights)
    node_xdma_c0_up = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Layout_V"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_b1"]),
            size=params["l15_b_data_length"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c0_up)

    # C1: B1 (layout_V / up weights) — serial after c0_up in the xDMA chain
    node_xdma_c1_up = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Layout_V"],
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
            size=N2 * params["indiv_B_tilesize"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c2_up)
    bingo_dfg.bingo_add_edge(node_xdma_c1_up, node_xdma_c2_up)

    node_xdma_c2_down = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_A,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Indiv_Down_B"],
            dst_addr=mh["C2_indiv_L1_B_down"],
            size=2 * N2d * params["indiv_down_B_tilesize"],
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
            size=N2 * params["indiv_B_tilesize"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c3_up)
    bingo_dfg.bingo_add_edge(node_xdma_c2_down, node_xdma_c3_up)

    node_xdma_c3_down = BingoNode(  # xDMA Phase 0 最后一个节点
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=addr_offset(
                mh["L3_Sym_Indiv_Down_B"], 7 * params["indiv_down_B_expert_stride"]
            ),
            dst_addr=mh["C3_indiv_L1_B_down"],
            size=2 * N2d * params["indiv_down_B_tilesize"],
        ),
    )
    bingo_dfg.bingo_add_node(node_xdma_c3_down)
    bingo_dfg.bingo_add_edge(node_xdma_c3_up, node_xdma_c3_down)

    # =====================================================================
    # Phase 1a: A token 加载（iDMA + xDMA 两路并行）
    # 必须等待两路 Phase 0 全部完成后才开始：
    #   node_idma_c3_gate (iDMA path 最后) AND node_xdma_c3_down (xDMA path 最后)
    # 三个 A 加载串行在系统 iDMA 上（只有一个物理引擎）。
    # =====================================================================

    node_c0_load_A = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mh["L3_Sym_Layout_A"],
            dst_addr=addr_offset(mh["C0_L1_Layout"], params["l15_delta_local_a"]),
            size=params["l15_a_data_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c0_load_A)
    bingo_dfg.bingo_add_edge(node_idma_c3_gate, node_c0_load_A)  # iDMA path 全部完成
    bingo_dfg.bingo_add_edge(node_xdma_c3_down, node_c0_load_A)  # xDMA path 全部完成

    node_c1_load_A = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_1,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Layout_A"],
            dst_addr=addr_offset(mh["C1_L1_Layout"], params["l15_delta_local_a"]),
            size=params["l15_a_data_bytes"],
        ),
    )
    bingo_dfg.bingo_add_node(node_c1_load_A)
    # xDMA 与 Phase 0 xDMA 串行（同一引擎），iDMA 无关，故只等 Phase 0 两路都完成
    bingo_dfg.bingo_add_edge(node_xdma_c3_down, node_c1_load_A)  # xDMA Phase 0 结束
    bingo_dfg.bingo_add_edge(node_idma_c3_gate, node_c1_load_A)  # iDMA Phase 0 结束

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
    bingo_dfg.bingo_add_edge(node_c0_load_A, node_c3_load_A)  # 系统 iDMA 串行

    # =====================================================================
    # Phase 1b: shared-expert GEMM.
    # Use the fused L15 kernel: Mode-0 SwiGLU and Mode-1 down projection run
    # inside one device node.  The fused kernel was validated separately and
    # keeps the intended Mode-1 CSR preload optimization.  The node waits for
    # all required shared tensors (A, gate/up, W2l/W2r) to be staged in L1
    # before it starts; this removes the split-node D0 producer/consumer edge.
    # =====================================================================

    # ---- Phase 1b: C0/C1 compute via fused L15 kernels ----

    # ---- C0 Phase 1 W2 loading (xDMA on C0 DM core, starts after c0_b1) ----
    node_xdma_c0_w2l = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_SHARED_0,
        assigned_core_id=DMA_CORE_ID,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mh["L3_Sym_Layout_W2L"],
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
            src_addr=mh["L3_Sym_Layout_W2R"],
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
            src_addr=mh["L3_Sym_Layout_W2L"],
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
            src_addr=mh["L3_Sym_Layout_W2R"],
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
    bingo_dfg.bingo_add_edge(node_c0_shared_full, node_c0_store_out)
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
    bingo_dfg.bingo_add_edge(node_c1_shared_full, node_c1_store_out)
    prev_c1 = node_c1_store_out

    # --- C3: router GEMM (Mode 1: split total N groups across two VCs) ---
    router_B_half_stride = params["router_B_vc_stride"]
    router_D_half = params["router_D_vc_stride"]
    node_c3_router_gemm = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=CLUSTER_INDIV_B,
        assigned_core_id=GEMM_CORE_ID,
        kernel_name="__snax_bingo_kernel_dual_vc_gemm_full",
        kernel_args=SnaxBingoKernelDualVcGemmFullArgs(
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
            array_shape=params["array_shape"],
            rescale_mult=1,
            rescale_shift=0,
        ),
    )
    bingo_dfg.bingo_add_node(node_c3_router_gemm)
    bingo_dfg.bingo_add_edge(node_c3_load_A, node_c3_router_gemm)

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

    if not ENABLE_PHASE3_PHASE4:
        enforce_in_order_completion_per_core(bingo_dfg)
        return bingo_dfg

    # =====================================================================
    # Phase 3: MoEPrepare
    #
    # Pure HW fast path: consume expert_token_counts + CAM state, drive RTL
    # scheduler, then direct-lower compact plan entries into C2/C3 L3 stage
    # dynamic args. request/schedule buffers are legacy fields for SW/check
    # builds and are not touched by the fast runtime path.
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
            expert_token_offsets_addr=mh["L3_Alloc_Expert_Token_Offsets"],
            expert_token_ids_addr=mh["L3_Alloc_Expert_Token_Ids"],
            expert_token_kpos_addr=mh["L3_Alloc_Expert_Token_Kpos"],
            n_experts=params["num_indiv_experts"],
            topk_indices_l3=mh["L3_Alloc_TopK_Indices"],
            M_total=params["M_total"],
            top_k=params["top_k"],
            expert_token_counts_valid=1,
            runtime_state_addr=mh["L3_Alloc_MoE_Runtime_State"],
            c2_stage_base=mh["L3_Alloc_C2_Stage"],
            c3_stage_base=mh["L3_Alloc_C3_Stage"],
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
            expert_token_offsets_addr=mh["L3_Alloc_Expert_Token_Offsets"],
            expert_token_ids_addr=mh["L3_Alloc_Expert_Token_Ids"],
            expert_token_kpos_addr=mh["L3_Alloc_Expert_Token_Kpos"],
            cam_state_addr=mh["L3_Alloc_CAM_State"],
            input_A_l3_base=mh["L3_Sym_Input_A"],
            topk_indices_l3=mh["L3_Alloc_TopK_Indices"],
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
            c2_active_state_l1_addr=mh["C2_indiv_Active_State"],
            c3_active_state_l1_addr=mh["C3_indiv_Active_State"],
            A_token_bytes=params["A_token_bytes"],
            indiv_B_expert_stride=params["indiv_B_expert_stride"],
            indiv_down_B_expert_stride=params["indiv_down_B_expert_stride"],
            down_D_bytes_per_expert=2 * N2d * params["indiv_down_D_tilesize"],
            M_total=params["M_total"],
            top_k=params["top_k"],
            indiv_B_tile_bytes=params["indiv_B_tilesize"],
            indiv_D_tile_bytes=params["indiv_D_tilesize"],
            indiv_down_B_tile_bytes=params["indiv_down_B_tilesize"],
            indiv_down_D_tile_bytes=params["indiv_down_D_tilesize"],
            indiv_N2=N2,
            indiv_down_N2=N2d,
            s1_block_count=N2,
            s3_block_count=N2d,
            indiv_K1=params["indiv_K1"],
            indiv_N_per_block=params["indiv_N1"] * params["meshCol"],
            indiv_down_K1=params["indiv_down_K1"],
            indiv_down_N_per_block=params["indiv_down_N1"] * params["meshCol"],
            rescale_mult=1,
            rescale_shift=0,
            output_expert_stride_bytes=2 * N2d * params["indiv_down_D_tilesize"],
            max_tokens_per_expert=params["max_tokens_per_expert"],
            c2_static_args_base=mh["C2_indiv_Static_Args"],
            c3_static_args_base=mh["C3_indiv_Static_Args"],
            c2_dynamic_args_base=mh["C2_indiv_Dyn_Args"],
            c3_dynamic_args_base=mh["C3_indiv_Dyn_Args"],
            dynamic_arg_slot_bytes=params["dynamic_arg_slot_bytes"],
            dynamic_num_slots=params["dynamic_slot_count"],
            c2_stage_base=mh["L3_Alloc_C2_Stage"],
            c3_stage_base=mh["L3_Alloc_C3_Stage"],
        ),
    )
    bingo_dfg.bingo_add_node(node_execute)
    bingo_dfg.bingo_add_edge(node_prepare, node_execute)

    def add_dynamic_slot_chain(prefix: str, cluster_id: int, slot: int, deps):
        dyn_arg_addr = addr_offset(
            mh[f"{prefix}_Dyn_Args"], slot * params["dynamic_arg_slot_bytes"]
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
        s2_computes = []
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
                bingo_dfg.bingo_add_edge(s2_computes[block - 1], compute)
            bingo_dfg.bingo_add_edge(load, compute)
            s1_loads.append(load)
            s2_computes.append(compute)

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
        bingo_dfg.bingo_add_edge(s2_computes[-1], compute_gate_up_full)

        # S3+S4: N2d 个 load+compute 节点对，边搬边算流水线
        # skip_s3=1(cache hit) 时：load/compute 节点全部跳过，所有 token 由 compute_down_full 处理
        s3_loads = []
        s4_computes = []
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
                bingo_dfg.bingo_add_edge(s4_computes[block - 1], compute)
            bingo_dfg.bingo_add_edge(load, compute)
            s3_loads.append(load)
            s4_computes.append(compute)

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
        bingo_dfg.bingo_add_edge(s4_computes[-1], compute_down_full)

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
    params = define_workload_params(**cfg)
    mh = define_memory_handles(params)
    dfg = create_dfg(params, mh)
    data_header_name = f"{params['app_name']}_data.h"
    dfg.bingo_compile_dfg(
        params["app_name"],
        args.output_dir,
        args.output_offload_file_name,
        extra_include_header_list=[data_header_name],
    )
    patch_moe_header_preamble(
        os.path.join(args.output_dir, args.output_offload_file_name)
    )


if __name__ == "__main__":
    main()
