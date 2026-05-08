# Fanchen Kong <fanchen.kong@kuleuven.be>

# This file is the main entry point for the bingo offload application
# Users will create the dfg in this file
# And then the mini-compiler will emit the WORKLOAD.h file
import os
import sys
import argparse
import numpy as np
import pathlib
import hjson
import re
import struct
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")

NODE37_DIAG_MODE = os.environ.get("BINGO_NODE37_DIAG", "none").strip().lower()
VALID_NODE37_DIAG_MODES = {
    "none",
    "mirror_node36_src",
    "bypass_node37_dep",
    "mirror_and_bypass",
}

if NODE37_DIAG_MODE not in VALID_NODE37_DIAG_MODES:
    raise ValueError(
        f"Unsupported BINGO_NODE37_DIAG mode: {NODE37_DIAG_MODE}. "
        f"Expected one of {sorted(VALID_NODE37_DIAG_MODES)}"
    )

print(f"BINGO_NODE37_DIAG: {NODE37_DIAG_MODE}")


def parse_inorder_completion_core_ids() -> set[int]:
    raw_core_ids = os.environ.get("BINGO_INORDER_CORE_IDS", "0").strip().lower()
    if raw_core_ids in {"", "none", "off", "false", "no"}:
        return set()
    if raw_core_ids in {"all", "*"}:
        return {0, 1, 2}
    return {
        int(core_id.strip()) for core_id in raw_core_ids.split(",") if core_id.strip()
    }


INORDER_COMPLETION_CORE_IDS = parse_inorder_completion_core_ids()

LEGACY_ENFORCE_INORDER_PER_CORE = os.environ.get(
    "BINGO_ENFORCE_INORDER_PER_CORE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
if LEGACY_ENFORCE_INORDER_PER_CORE and "BINGO_INORDER_CORE_IDS" not in os.environ:
    INORDER_COMPLETION_CORE_IDS = {0, 1, 2}

print(
    "BINGO_INORDER_CORE_IDS: "
    + (
        ",".join(str(core_id) for core_id in sorted(INORDER_COMPLETION_CORE_IDS))
        if INORDER_COMPLETION_CORE_IDS
        else "none"
    )
)

from bingo_dfg import BingoDFG
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import (
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelGemmMinimalArgs,
    SnaxBingoKernelDualVcGemmFullArgs,
    SnaxBingoKernelDualVcSwigluFullArgs,
    HostBingoKernelMoERouterScheduleArgs,  # 补充
    HostBingoKernelComputeDelayedSoftmaxArgs,  # 补充
    HostBingoKernelBuildScatterMetadataArgs,  # 补充
    HostBingoKernelComputeSwishActivationArgs,  # 补充
    HostBingoKernelComputeGluMultiplicationArgs,  # 补充
    HostBingoKernelExpertsResultAccumulateArgs,  # 补充
    HostBingoKernelScatterAndPadArgs,
)


def get_args():
    parser = argparse.ArgumentParser(description="Bingo HW Manager")
    parser.add_argument(
        "--cfg",
        type=pathlib.Path,
        default=pathlib.Path(current_dir) / "params.hjson",
        help="Workload parameter config file",
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
        / "snax_versacore_to_cluster.hjson",
        help="Hardware config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--output_offload_file_name",
        type=str,
        default="offload_bingo_hw.h",
        help="Output filename for the offload header file",
    )
    parser.add_argument(
        "--emit_mini_golden",
        action="store_true",
        help="Emit mini golden data for verification",
    )
    return parser.parse_args()


head_path = os.path.join(current_dir, "MoE_common_variable.h")


def parse_header_config(head_path):
    MoE_config = {}
    pattern = re.compile(r"#define\s+(\w+)\s+([-+]?[\d\.]+)")
    with open(head_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                key = match.group(1)
                value_str = match.group(2).strip()
                MoE_config[key] = eval(value_str)
    return MoE_config


def load_workload_config(args):
    with args.cfg.open() as cfg_file:
        param_cfg = hjson.loads(cfg_file.read())

    with args.hwcfg.open() as hw_file:
        hw_cfg = hjson.loads(hw_file.read())

    merged_config = {**param_cfg, **hw_cfg}
    merged_config["emit_mini_golden"] = args.emit_mini_golden
    return merged_config


def define_workload_params(**kwargs):
    """Defines the GeMM workload parameters (dual-VC INT16×INT4)."""
    MoE_config = parse_header_config(head_path)
    num_double_buffers = 2
    data_type = 0  # INT16×INT4 data type index
    # Support both legacy single-VC and new dual-VC template keys
    core_tmpl = kwargs.get("snax_versacore_core_template") or kwargs.get(
        "snax_dual_versacore_int16x4_core_template"
    )
    if core_tmpl is None:
        raise KeyError(
            "No versacore template key found in hwcfg. "
            "Expected 'snax_versacore_core_template' or "
            "'snax_dual_versacore_int16x4_core_template'."
        )
    snax_acc_cfg = core_tmpl["snax_acc_cfg"][0]
    array_shape = kwargs["array_shape"]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        0
    ]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        1
    ]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        2
    ]
    # Per-matrix A overrides (INT16 A may use different meshRow/tileSize)
    meshRow_A = int(kwargs.get("A_meshRow", meshRow))
    tileSize_A = int(kwargs.get("A_tileSize", tileSize))
    # Dual-VC down projection uses concatenated output width
    meshCol_down = int(kwargs.get("down_meshCol", meshCol))
    vc_meshCol_down = meshCol  # each VC handles half the down output width

    params = {
        "app_name": "single_cluster_MoE",
        "num_double_buffers": num_double_buffers,
        "meshRow_A": meshRow_A,
        "tileSize_A": tileSize_A,
        "meshCol_down": meshCol_down,
        "vc_meshCol_down": vc_meshCol_down,
        "router_M1": kwargs["router_M1"],
        "router_N1": kwargs["router_N1"],
        "router_K1": kwargs["router_K1"],
        "router_M2": kwargs["router_M2"],
        "router_N2": kwargs["router_N2"],
        "router_K2": kwargs["router_K2"],
        "shared_swish_glu_M1": kwargs["shared_swish_glu_M1"],
        "shared_swish_glu_N1": kwargs["shared_swish_glu_N1"],
        "shared_swish_glu_K1": kwargs["shared_swish_glu_K1"],
        "shared_swish_glu_M2": kwargs["shared_swish_glu_M2"],
        "shared_swish_glu_N2": kwargs["shared_swish_glu_N2"],
        "shared_swish_glu_K2": kwargs["shared_swish_glu_K2"],
        "shared_down_projection_M1": kwargs["shared_down_projection_M1"],
        "shared_down_projection_N1": kwargs["shared_down_projection_N1"],
        "shared_down_projection_K1": kwargs["shared_down_projection_K1"],
        "shared_down_projection_M2": kwargs["shared_down_projection_M2"],
        "shared_down_projection_N2": kwargs["shared_down_projection_N2"],
        "shared_down_projection_K2": kwargs["shared_down_projection_K2"],
        "individual_swish_glu_M1": kwargs["individual_swish_glu_M1"],
        "individual_swish_glu_N1": kwargs["individual_swish_glu_N1"],
        "individual_swish_glu_K1": kwargs["individual_swish_glu_K1"],
        "individual_swish_glu_M2": kwargs["individual_swish_glu_M2"],
        "individual_swish_glu_N2": kwargs["individual_swish_glu_N2"],
        "individual_swish_glu_K2": kwargs["individual_swish_glu_K2"],
        "individual_down_projection_M1": kwargs["individual_down_projection_M1"],
        "individual_down_projection_N1": kwargs["individual_down_projection_N1"],
        "individual_down_projection_K1": kwargs["individual_down_projection_K1"],
        "individual_down_projection_M2": kwargs["individual_down_projection_M2"],
        "individual_down_projection_N2": kwargs["individual_down_projection_N2"],
        "individual_down_projection_K2": kwargs["individual_down_projection_K2"],
        "meshRow": meshRow,
        "tileSize": tileSize,
        "meshCol": meshCol,
        "array_shape": array_shape,
        "snax_acc_cfg": snax_acc_cfg,
    }
    # -------------------------------------------------------------------------
    # Tile byte sizes for dual-VC INT16×INT4 hardware:
    #   A: INT16  → 2 bytes/element
    #   B: INT4 nibble-packed → tileSize × meshCol / 2 bytes per tile
    #   D: INT16  → 2 bytes/element (dual-VC output precision)
    # For down projection B/D, use vc_meshCol_down (= meshCol, per-VC half)
    #   and the combined output width is meshCol_down = 2 × vc_meshCol_down.
    # -------------------------------------------------------------------------

    # Router stage (dual-VC GEMM mode; B0/B1 both point to same router tile)
    params["router_A_tileSize"] = (
        params["router_M1"] * params["router_K1"] * meshRow_A * tileSize_A * 2
    )
    params["router_B_tileSize"] = (
        params["router_K1"] * params["router_N1"] * tileSize * meshCol // 2
    )
    params["router_D_tileSize"] = (
        params["router_M1"] * params["router_N1"] * meshRow * meshCol * 2
    )

    # Shared expert gate+up (dual-VC SwiGLU)
    params["shared_swish_glu_A_tileSize"] = (
        params["shared_swish_glu_M1"]
        * params["shared_swish_glu_K1"]
        * meshRow_A
        * tileSize_A
        * 2
    )
    params["shared_swish_glu_B_tileSize"] = (
        params["shared_swish_glu_K1"]
        * params["shared_swish_glu_N1"]
        * tileSize
        * meshCol
        // 2
    )
    # D from SwiGLU is INT16 (one output per mesh element)
    params["shared_swish_glu_D_tileSize"] = (
        params["shared_swish_glu_M1"]
        * params["shared_swish_glu_N1"]
        * meshRow
        * meshCol
        * 2
    )
    # Shared expert down projection (dual-VC GEMM; B per VC-half, D per VC-half)
    params["shared_down_projection_A_tileSize"] = (
        params["shared_down_projection_M1"]
        * params["shared_down_projection_K1"]
        * meshRow_A
        * tileSize_A
        * 2
    )
    # B tile per VC half (vc_meshCol_down columns)
    params["shared_down_projection_B_tileSize"] = (
        params["shared_down_projection_K1"]
        * params["shared_down_projection_N1"]
        * tileSize
        * vc_meshCol_down
        // 2
    )
    # D tile per VC half (vc_meshCol_down output columns, INT16)
    params["shared_down_projection_D_tileSize"] = (
        params["shared_down_projection_M1"]
        * params["shared_down_projection_N1"]
        * meshRow
        * vc_meshCol_down
        * 2
    )

    # Individual expert gate+up (dual-VC SwiGLU)
    params["individual_swish_glu_A_tileSize"] = (
        params["individual_swish_glu_M1"]
        * params["individual_swish_glu_K1"]
        * meshRow_A
        * tileSize_A
        * 2
    )
    params["individual_swish_glu_B_tileSize"] = (
        params["individual_swish_glu_K1"]
        * params["individual_swish_glu_N1"]
        * tileSize
        * meshCol
        // 2
    )
    params["individual_swish_glu_D_tileSize"] = (
        params["individual_swish_glu_M1"]
        * params["individual_swish_glu_N1"]
        * meshRow
        * meshCol
        * 2
    )
    # Individual expert down projection (dual-VC GEMM)
    params["individual_down_projection_A_tileSize"] = (
        params["individual_down_projection_M1"]
        * params["individual_down_projection_K1"]
        * meshRow_A
        * tileSize_A
        * 2
    )
    params["individual_down_projection_B_tileSize"] = (
        params["individual_down_projection_K1"]
        * params["individual_down_projection_N1"]
        * tileSize
        * vc_meshCol_down
        // 2
    )
    params["individual_down_projection_D_tileSize"] = (
        params["individual_down_projection_M1"]
        * params["individual_down_projection_N1"]
        * meshRow
        * vc_meshCol_down
        * 2
    )

    params["input_dimension"] = MoE_config["input_dimension"]
    params["expert_number_each_layer"] = MoE_config["expert_number_each_layer"]
    params["individual_expert_number_k"] = MoE_config["individual_expert_number_k"]
    params["shared_expert_number_k"] = MoE_config["shared_expert_number_k"]
    params["final_shift_step"] = MoE_config["final_shift_step"]
    params["swish_glu_scale_in"] = MoE_config["swish_glu_scale_in"]
    params["swish_glu_scale_out"] = MoE_config["swish_glu_scale_out"]
    params["softmax_scale"] = MoE_config["softmax_scale"]

    return params


def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}
    mem_handles["L3_Sym_Input_A"] = BingoMemSymbol("input_A")

    # Define memory Symbol
    # =========================================================================
    # 1. L3 静态数据符号挂载 (L3_Sym)
    # 物理意义：编译期确定的 C 语言全局数组，利用 M2/N2 静态展开 Tile Offset
    # =========================================================================
    assert (
        params["router_N2"] == 1
    ), "Current Router DFG implementation does not support N2 > 1 dimension tiling."
    # ------------------ Router 阶段 ------------------
    for m in range(params["router_M2"]):
        mem_handles[f"L3_Sym_Router_Input_A_tile_{m}"] = BingoMemSymbol(
            "input_A", offset=m * params["router_A_tileSize"]
        )

    for n in range(params["router_N2"]):
        mem_handles[f"L3_Sym_Router_Weight_B_tile_{n}"] = BingoMemSymbol(
            "router_B", offset=n * params["router_B_tileSize"]
        )

    # ------------------ Shared Expert 阶段 ------------------
    for m in range(params["shared_swish_glu_M2"]):
        # Gate 和 Up 共享同一个物理输入 A
        mem_handles[f"L3_Sym_Shared_Swish_Input_A_tile_{m}"] = BingoMemSymbol(
            "input_A", offset=m * params["shared_swish_glu_A_tileSize"]
        )

    for n in range(params["shared_swish_glu_N2"]):
        mem_handles[f"L3_Sym_Shared_Up_Weight_B_tile_{n}"] = BingoMemSymbol(
            "shared_experts_up_projection_B",
            offset=n * params["shared_swish_glu_B_tileSize"],
        )
        mem_handles[f"L3_Sym_Shared_Gate_Weight_B_tile_{n}"] = BingoMemSymbol(
            "shared_experts_gate_B", offset=n * params["shared_swish_glu_B_tileSize"]
        )

    for n in range(params["shared_down_projection_N2"]):
        mem_handles[f"L3_Sym_Shared_Down_Weight_B_tile_{n}"] = BingoMemSymbol(
            "shared_experts_down_projection_B",
            offset=n * params["shared_down_projection_B_tileSize"],
        )

    # ------------------ Individual Expert 阶段 ------------------
    # A 和 D 依赖 Router 动态选路，不在编译期展开 Tile，留由 L3_Alloc 动态池处理
    for n in range(params["individual_swish_glu_N2"]):
        mem_handles[f"L3_Sym_Indiv_Up_Weight_B_tile_{n}"] = BingoMemSymbol(
            "individual_experts_up_projection_B",
            offset=n * params["individual_swish_glu_B_tileSize"],
        )
        mem_handles[f"L3_Sym_Indiv_Gate_Weight_B_tile_{n}"] = BingoMemSymbol(
            "individual_experts_gate_B",
            offset=n * params["individual_swish_glu_B_tileSize"],
        )

    for n in range(params["individual_down_projection_N2"]):
        mem_handles[f"L3_Sym_Indiv_Down_Weight_B_tile_{n}"] = BingoMemSymbol(
            "individual_experts_down_projection_B",
            offset=n * params["individual_down_projection_B_tileSize"],
        )

    # ------------------ 黄金验证数据 ------------------
    mem_handles["L3_Sym_Golden_Layer_Output"] = BingoMemSymbol("layer_output")

    # =========================================================================
    # 2. L1 统一动态缓冲池 (L1_Buf)
    # 物理意义：TCDM 中的时分复用空间，根据 M2/N2 探针动态决定是否 Ping-Pong
    # =========================================================================
    # 获取全流程对 L1 SRAM 容量的最大极值需求
    max_A_l1_size = max(
        params["router_A_tileSize"],
        params["shared_swish_glu_A_tileSize"],
        params["shared_down_projection_A_tileSize"],
        params["individual_swish_glu_A_tileSize"],
        params["individual_down_projection_A_tileSize"],
    )
    max_B_l1_size = max(
        params["router_B_tileSize"],
        params["shared_swish_glu_B_tileSize"],
        params["shared_down_projection_B_tileSize"],
        params["individual_swish_glu_B_tileSize"],
        params["individual_down_projection_B_tileSize"],
    )
    max_D_l1_size = max(
        params["router_D_tileSize"],
        params["shared_swish_glu_D_tileSize"],
        params["shared_down_projection_D_tileSize"],
        params["individual_swish_glu_D_tileSize"],
        params["individual_down_projection_D_tileSize"],
    )

    # 动态架构探针
    A_needs_pingpong = (
        max(
            params["router_M2"],
            params["shared_swish_glu_M2"],
            params["shared_down_projection_M2"],
            params["individual_swish_glu_M2"],
            params["individual_down_projection_M2"],
        )
        > 1
    )
    B_needs_pingpong = (
        max(
            params["router_N2"],
            params["shared_swish_glu_N2"],
            params["shared_down_projection_N2"],
            params["individual_swish_glu_N2"],
            params["individual_down_projection_N2"],
        )
        > 1
    )
    D_needs_pingpong = A_needs_pingpong or B_needs_pingpong

    chip_id = 0
    cluster_id = 0

    if B_needs_pingpong:
        mem_handles["L1_Buf_Unified_B_ping"] = BingoMemAlloc(
            "l1_buf_B_ping",
            size=max_B_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
        mem_handles["L1_Buf_Unified_B_pong"] = BingoMemAlloc(
            "l1_buf_B_pong",
            size=max_B_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
    else:
        mem_handles["L1_Buf_Unified_B"] = BingoMemAlloc(
            "l1_buf_B",
            size=max_B_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )

    if A_needs_pingpong:
        mem_handles["L1_Buf_Unified_A_ping"] = BingoMemAlloc(
            "l1_buf_A_ping",
            size=max_A_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
        mem_handles["L1_Buf_Unified_A_pong"] = BingoMemAlloc(
            "l1_buf_A_pong",
            size=max_A_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
    else:
        mem_handles["L1_Buf_Unified_A"] = BingoMemAlloc(
            "l1_buf_A",
            size=max_A_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )

    if D_needs_pingpong:
        mem_handles["L1_Buf_Unified_D_ping"] = BingoMemAlloc(
            "l1_buf_D_ping",
            size=max_D_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
        mem_handles["L1_Buf_Unified_D_pong"] = BingoMemAlloc(
            "l1_buf_D_pong",
            size=max_D_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )
    else:
        mem_handles["L1_Buf_Unified_D"] = BingoMemAlloc(
            "l1_buf_D",
            size=max_D_l1_size,
            mem_level="L1",
            chip_id=chip_id,
            cluster_id=cluster_id,
        )

    # =========================================================================
    # Dual-VC 专用 L1 缓冲区
    # SwiGLU 内核同时读取 B_gate 和 B_up，因此需要两个独立的 B 缓冲区。
    # 每个缓冲区容纳 N2 × B_tileSize 字节，保存全部 N2 tile。
    # =========================================================================
    # 最大 N2 个 SwiGLU B tile (gate 和 up 分别占用)
    max_swiglu_n2 = max(
        params["shared_swish_glu_N2"],
        params["individual_swish_glu_N2"],
    )
    max_swiglu_B_tile = max(
        params["shared_swish_glu_B_tileSize"],
        params["individual_swish_glu_B_tileSize"],
    )
    gate_B_n2_size = max_swiglu_n2 * max_swiglu_B_tile
    up_B_n2_size = max_swiglu_n2 * max_swiglu_B_tile
    mem_handles["L1_Buf_Gate_B_N2"] = BingoMemAlloc(
        "l1_gate_b_n2",
        size=gate_B_n2_size,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )
    mem_handles["L1_Buf_Up_B_N2"] = BingoMemAlloc(
        "l1_up_b_n2",
        size=up_B_n2_size,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )

    # SwiGLU D0 输出缓冲区（保存全部 N2 tile），供后续 Down Projection 读取作为 A
    max_swiglu_D_tile = max(
        params["shared_swish_glu_D_tileSize"],
        params["individual_swish_glu_D_tileSize"],
    )
    swiglu_D_n2_size = max_swiglu_n2 * max_swiglu_D_tile
    mem_handles["L1_Buf_Swiglu_D_N2"] = BingoMemAlloc(
        "l1_swiglu_d_n2",
        size=swiglu_D_n2_size,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )

    # D1 scratch（SwiGLU Mode 0 的 D1 输出，与 D0 内容相同，不使用）
    d1_scratch_size = max_swiglu_D_tile
    mem_handles["L1_Buf_D1_Scratch"] = BingoMemAlloc(
        "l1_d1_scratch",
        size=d1_scratch_size,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )

    # Down Projection B0/B1（每次调用 DualVcGemm 时各装一个 VC half 的 B tile）
    max_down_B_tile = max(
        params["shared_down_projection_B_tileSize"],
        params["individual_down_projection_B_tileSize"],
    )
    mem_handles["L1_Buf_Down_B0"] = BingoMemAlloc(
        "l1_down_b0",
        size=max_down_B_tile,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )
    mem_handles["L1_Buf_Down_B1"] = BingoMemAlloc(
        "l1_down_b1",
        size=max_down_B_tile,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )

    # Down Projection D（每次 DualVcGemm 各输出一个 VC half 的 D tile）
    max_down_D_tile = max(
        params["shared_down_projection_D_tileSize"],
        params["individual_down_projection_D_tileSize"],
    )
    mem_handles["L1_Buf_Down_D0"] = BingoMemAlloc(
        "l1_down_d0",
        size=max_down_D_tile,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )
    mem_handles["L1_Buf_Down_D1"] = BingoMemAlloc(
        "l1_down_d1",
        size=max_down_D_tile,
        mem_level="L1",
        chip_id=chip_id,
        cluster_id=cluster_id,
    )

    # =========================================================================
    # 3. L3 运行时动态中间池 (L3_Alloc)
    # 物理意义：CVA6 与硬件之间交接中间特征图 (Feature Maps) 和控制表的内存
    # =========================================================================

    # --- 控制面表格 ---
    metadata_size = (
        params["router_M1"]
        * params["meshRow"]
        * params["router_M2"]
        * params["individual_expert_number_k"]
        * 4
    )
    # store the chosen experts for all tokens
    mem_handles["L3_Alloc_Router_TopK_Indices"] = BingoMemAlloc(
        "l3_router_topk_idx", size=metadata_size, mem_level="L3"
    )
    # 2. 【新增】存选出来的原始分数 (Fashion 1: 按 Token 顺排，给后台 Softmax 用)
    mem_handles["L3_Alloc_Router_TopK_Scores"] = BingoMemAlloc(
        "l3_router_topk_scores", size=metadata_size, mem_level="L3"
    )
    # store the probability of chosen experts for all tokens
    mem_handles["L3_Alloc_Router_Probabilities"] = BingoMemAlloc(
        "l3_router_prob", size=metadata_size, mem_level="L3"
    )
    # record the tokens for each expert
    expert_num = params["expert_number_each_layer"]
    mem_handles["L3_Alloc_Expert_Histogram"] = BingoMemAlloc(
        "l3_exp_hist", size=expert_num * 4, mem_level="L3"
    )
    # record the offset in the input matrix of each expert
    # so that we can separate them with padding data after down projection
    mem_handles["L3_Alloc_Expert_Offsets"] = BingoMemAlloc(
        "l3_exp_offset", size=expert_num * 4, mem_level="L3"
    )

    # 4. 【注释升级】导航票表 (Fashion 2: 按专家物理坑位排)
    # 记录该位置的数据对应的 flat_idx (token_id * K + k_index)
    # 用于归约时精准反推数据该加到哪一行、该乘哪个概率
    mem_handles["L3_Alloc_Reverse_Index"] = BingoMemAlloc(
        "l3_rev_idx", size=metadata_size, mem_level="L3"
    )

    # --- 数据面张量池 (按各阶段的宏观满尺寸分配) ---
    router_D_total = (
        params["router_D_tileSize"] * params["router_M2"] * params["router_N2"]
    )
    mem_handles["L3_Alloc_Router_HW_Output"] = BingoMemAlloc(
        "l3_router_hw_out", size=router_D_total, mem_level="L3"
    )

    # Shared Expert 中间态
    shared_swish_D_total = (
        params["shared_expert_number_k"]  # <--- 补齐专家数
        * params["shared_swish_glu_D_tileSize"]
        * params["shared_swish_glu_M2"]
        * params["shared_swish_glu_N2"]
    )
    mem_handles["L3_Alloc_Shared_Up_HW_Output"] = BingoMemAlloc(
        "l3_shared_up_out", size=shared_swish_D_total, mem_level="L3"
    )
    mem_handles["L3_Alloc_Shared_Gate_HW_Output"] = BingoMemAlloc(
        "l3_shared_gate_out", size=shared_swish_D_total, mem_level="L3"
    )

    # 分配给 Shared Expert 的 Swish 浮点计算暂存池
    shared_swish_float_buf_size = (
        params["shared_expert_number_k"]
        * params["shared_swish_glu_M2"]
        * params["shared_swish_glu_M1"]
        * params["meshRow"]
        * params["shared_swish_glu_N2"]
        * params["shared_swish_glu_N1"]
        * params["meshCol"]
        * 4  # 浮点数占 4 字节
    )
    mem_handles["L3_Temp_Swish_Float_Buf"] = BingoMemAlloc(
        "l3_temp_shared_swish_buf", size=shared_swish_float_buf_size, mem_level="L3"
    )

    # 这个 Buffer 既是 CVA6 算完 Swish 的落盘处，也是后续 Down 的输入起点
    shared_down_A_total = (
        params["shared_expert_number_k"]  # <--- 补齐专家数
        * params["shared_down_projection_A_tileSize"]
        * params["shared_down_projection_M2"]
    )

    mem_handles["L3_Alloc_Shared_Activated_A"] = BingoMemAlloc(
        "l3_shared_act_a", size=shared_down_A_total, mem_level="L3"
    )

    shared_down_D_total = (
        params["shared_expert_number_k"]  # <--- 补齐专家数
        * params["shared_down_projection_D_tileSize"]
        * params["shared_down_projection_M2"]
        * params["shared_down_projection_N2"]
        * 2  # dual-VC: D0 + D1 两路输出
    )

    mem_handles["L3_Alloc_Shared_Down_HW_Output"] = BingoMemAlloc(
        "l3_shared_down_out", size=shared_down_D_total, mem_level="L3"
    )

    # Individual Expert 中间态 (Scatter 池要足够包容所有专家的动态重组)
    indiv_A_total = (
        params["expert_number_each_layer"]  # <--- 修正：必须是总专家数 E，不是 K！
        * params["individual_swish_glu_A_tileSize"]
        * params["individual_swish_glu_M2"]
    )

    mem_handles["L3_Alloc_Indiv_Scatter_Pool"] = BingoMemAlloc(
        "l3_indiv_scatter_pool", size=indiv_A_total, mem_level="L3"
    )

    indiv_swish_D_total = (
        params["expert_number_each_layer"]  # <--- 补齐专家数
        * params["individual_swish_glu_D_tileSize"]
        * params["individual_swish_glu_M2"]
        * params["individual_swish_glu_N2"]
    )

    mem_handles["L3_Alloc_Indiv_Up_HW_Output"] = BingoMemAlloc(
        "l3_indiv_up_out", size=indiv_swish_D_total, mem_level="L3"
    )
    mem_handles["L3_Alloc_Indiv_Gate_HW_Output"] = BingoMemAlloc(
        "l3_indiv_gate_out", size=indiv_swish_D_total, mem_level="L3"
    )

    # 分配给 Individual Expert 的 Swish 浮点计算暂存池
    indiv_swish_float_buf_size = (
        params["expert_number_each_layer"]
        * params["individual_swish_glu_M2"]
        * params["individual_swish_glu_M1"]
        * params["meshRow"]
        * params["individual_swish_glu_N2"]
        * params["individual_swish_glu_N1"]
        * params["meshCol"]
        * 4  # 浮点数占 4 字节
    )
    mem_handles["L3_Temp_Indiv_Swish_Float_Buf"] = BingoMemAlloc(
        "l3_temp_indiv_swish_buf", size=indiv_swish_float_buf_size, mem_level="L3"
    )

    indiv_down_A_total = (
        params["expert_number_each_layer"]  # <--- 补齐专家数
        * params["individual_down_projection_A_tileSize"]
        * params["individual_down_projection_M2"]
    )

    mem_handles["L3_Alloc_Indiv_Activated_A"] = BingoMemAlloc(
        "l3_indiv_act_a", size=indiv_down_A_total, mem_level="L3"
    )

    indiv_down_D_total = (
        params["individual_down_projection_D_tileSize"]
        * params["individual_down_projection_M2"]
        * params["individual_down_projection_N2"]
        * params["expert_number_each_layer"]  # <--- 必须乘以专家总数
        * 2  # dual-VC: D0 + D1 两路输出
    )

    mem_handles["L3_Alloc_Indiv_Down_HW_Output"] = BingoMemAlloc(
        "l3_indiv_down_out", size=indiv_down_D_total, mem_level="L3"
    )

    final_output_total_size = (
        params["router_M1"]
        * params["meshRow"]
        * params["router_M2"]  # 总 Token 数
        * params["individual_down_projection_N1"]
        * params["meshCol"]
        * params["individual_down_projection_N2"]  # 输出特征维度
        * 4  # int32 或 float32，占 4 字节
    )

    # 最终输出累加池
    mem_handles["L3_Alloc_Final_MoE_Output"] = BingoMemAlloc(
        "l3_final_moe_out", size=final_output_total_size, mem_level="L3"
    )

    return mem_handles


def float_to_uint64_bits(f_val: float) -> int:
    # 先将 float 打包为 4 字节的 bytes，再解包为 32 位无符号整数
    packed_f32 = struct.unpack("<I", struct.pack("<f", float(f_val)))[0]
    return packed_f32  # 自动扩展为 Python 的长整型，传递给 C 时会作为 uint64_t 处理


def addr_with_optional_offset(mem_handle, offset: int):
    """Return the handle object for zero offset to keep allocator metadata intact."""
    if offset == 0:
        return mem_handle
    return f"{mem_handle.get_c_var_name()} + {offset}"


def is_shared_expert1_first_up_tile(se_id: int, m: int, n: int) -> bool:
    return se_id == 1 and m == 0 and n == 0


def should_mirror_node36_src(se_id: int, m: int, n: int) -> bool:
    return NODE37_DIAG_MODE in {
        "mirror_node36_src",
        "mirror_and_bypass",
    } and is_shared_expert1_first_up_tile(se_id, m, n)


def should_bypass_node37_dep(se_id: int, m: int, n: int) -> bool:
    return NODE37_DIAG_MODE in {
        "bypass_node37_dep",
        "mirror_and_bypass",
    } and is_shared_expert1_first_up_tile(se_id, m, n)


def enforce_in_order_completion_per_core(bingo_dfg: BingoDFG) -> None:
    """Serialize normal tasks per core to match HW checkout/done semantics."""
    if not INORDER_COMPLETION_CORE_IDS:
        return

    topo_order = list(nx.topological_sort(bingo_dfg))
    normal_nodes = [
        node
        for node in topo_order
        if node.node_type == "normal"
        and node.assigned_core_id in INORDER_COMPLETION_CORE_IDS
    ]
    prev_node_per_lane = {}

    for node in normal_nodes:
        lane = (
            node.assigned_chiplet_id,
            node.assigned_cluster_id,
            node.assigned_core_id,
        )
        prev_node = prev_node_per_lane.get(lane)
        if prev_node is not None:
            # 使用 has_path 而非 has_edge：若两节点间已存在传递路径则跳过，
            # 避免插入冗余直接边。冗余直接边会导致前序节点需同时向多个后继
            # 发送 dep_matrix 完成信号（fan-out），而硬件每次完成只能设置
            # 一个 DepSet 槽位，造成其中一个后继永久等待（死锁）。
            if not nx.has_path(bingo_dfg, prev_node, node):
                bingo_dfg.bingo_add_edge(prev_node, node)
        prev_node_per_lane[lane] = node


def create_dfg(params, mem_handles):
    """Creates the Bingo Data Flow Graph with nodes and dependencies."""
    # 1. Initialize DFG
    num_chiplets = 1
    num_clusters_per_chiplet = 1
    num_cores_per_cluster = 2
    is_host_as_acc = True
    chiplet_ids = [0x00]
    bingo_dfg = BingoDFG(
        num_chiplets,
        num_clusters_per_chiplet,
        num_cores_per_cluster,
        is_host_as_acc,
        chiplet_ids,
    )
    cur_chiplet_id = 0
    cur_cluster_id = 0
    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for Host DMA Store

    # 物理层面：内存池里到底有没有分配 Ping-Pong 空间？
    # 这取决于 define_memory_handles 的分配结果
    has_phys_A_pingpong = "L1_Buf_Unified_A_ping" in mem_handles
    has_phys_B_pingpong = "L1_Buf_Unified_B_ping" in mem_handles
    has_phys_D_pingpong = "L1_Buf_Unified_D_ping" in mem_handles

    # 推导总 Token 数量 (物理行数)
    total_tokens = params["router_M1"] * params["meshRow"] * params["router_M2"]
    scale_in_raw = float_to_uint64_bits(params["swish_glu_scale_in"])
    scale_out_raw = float_to_uint64_bits(params["swish_glu_scale_out"])

    # 2. Define Nodes
    # 2.1 静态加载 Router Weight (B 矩阵)
    l1_B_router = (
        mem_handles["L1_Buf_Unified_B_ping"]
        if has_phys_B_pingpong
        else mem_handles["L1_Buf_Unified_B"]
    )
    node_router_copy_B = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["L3_Sym_Router_Weight_B_tile_0"],
            dst_addr=l1_B_router,
            size=params["router_B_tileSize"],
        ),
    )
    bingo_dfg.bingo_add_node(node_router_copy_B)

    router_copy_A_nodes = []
    router_gemm_nodes = []
    router_copy_D_nodes = []

    for m in range(params["router_M2"]):
        if has_phys_A_pingpong and params["router_M2"] > 1:
            current_l1_A = (
                mem_handles["L1_Buf_Unified_A_ping"]
                if m % 2 == 0
                else mem_handles["L1_Buf_Unified_A_pong"]
            )
        elif has_phys_A_pingpong:
            current_l1_A = mem_handles["L1_Buf_Unified_A_ping"]  # 有空间但不翻转
        else:
            current_l1_A = mem_handles["L1_Buf_Unified_A"]  # 根本没分配 Ping-Pong

        if has_phys_D_pingpong and params["router_M2"] > 1:
            current_l1_D = (
                mem_handles["L1_Buf_Unified_D_ping"]
                if m % 2 == 0
                else mem_handles["L1_Buf_Unified_D_pong"]
            )
        elif has_phys_D_pingpong:
            current_l1_D = mem_handles["L1_Buf_Unified_D_ping"]  # 有空间但不翻转
        else:
            current_l1_D = mem_handles["L1_Buf_Unified_D"]  # 根本没分配 Ping-Pong

        # DMA 搬运 Input A
        node_copy_A = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[f"L3_Sym_Router_Input_A_tile_{m}"],
                dst_addr=current_l1_A,
                size=params["router_A_tileSize"],
            ),
        )
        router_copy_A_nodes.append(node_copy_A)
        bingo_dfg.bingo_add_node(node_copy_A)

        # 硬件 GEMM 计算 — 统一使用 dual-VC GEMM full（B0=B1=router_B; D1=scratch）
        node_gemm = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=gemm_core_id,
            kernel_name="__snax_bingo_kernel_dual_vc_gemm_full",
            kernel_args=SnaxBingoKernelDualVcGemmFullArgs(
                input_A_addr=current_l1_A,
                input_B0_addr=l1_B_router,
                input_B1_addr=l1_B_router,  # 同一个 router tile，D1 结果丢弃
                output_D0_addr=current_l1_D,
                output_D1_addr=mem_handles["L1_Buf_D1_Scratch"],  # 丢弃
                M=params["router_M1"],
                K=params["router_K1"],
                N=params["router_N1"],
                array_shape=params["array_shape"],
                rescale_mult=1,
                rescale_shift=0,
            ),
        )
        router_gemm_nodes.append(node_gemm)
        bingo_dfg.bingo_add_node(node_gemm)

        # DMA 写回 L3 张量池
        node_copy_D = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=current_l1_D,
                dst_addr=addr_with_optional_offset(
                    mem_handles["L3_Alloc_Router_HW_Output"],
                    m * params["router_D_tileSize"],
                ),
                size=params["router_D_tileSize"],
            ),
        )
        router_copy_D_nodes.append(node_copy_D)
        bingo_dfg.bingo_add_node(node_copy_D)

    # -------------------------------------------------------------------------
    # 阶段 2: 异构控制与调度 (CVA6 CPU 处理)
    # -------------------------------------------------------------------------

    # 节点: 提取 Top-K
    node_host_router_topk = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_moe_router_schedule",
        kernel_args=HostBingoKernelMoERouterScheduleArgs(
            total_tokens=total_tokens,
            hardware_output_buffer_addr=mem_handles[
                "L3_Alloc_Router_HW_Output"
            ],  # 是一个大矩阵？
            global_indices_out_addr=mem_handles["L3_Alloc_Router_TopK_Indices"],
            global_scores_out_addr=mem_handles["L3_Alloc_Router_TopK_Scores"],
            expert_number_each_layer=params["expert_number_each_layer"],
            individual_expert_number_k=params["individual_expert_number_k"],
            mesh_row=params["meshRow"],
            mesh_col=params["meshCol"],
            router_m1=params["router_M1"],
            router_n1=params["router_N1"],
        ),
    )
    bingo_dfg.bingo_add_node(node_host_router_topk)

    # 节点: 计算 Softmax
    node_host_compute_softmax = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_compute_delayed_softmax",
        kernel_args=HostBingoKernelComputeDelayedSoftmaxArgs(
            global_top_k_scores_ptr_addr=mem_handles["L3_Alloc_Router_TopK_Scores"],
            global_calculated_probability_ptr_addr=mem_handles[
                "L3_Alloc_Router_Probabilities"
            ],
            actual_total_tokens=total_tokens,
            individual_expert_number_k=params["individual_expert_number_k"],
            softmax_scale_raw=float_to_uint64_bits(params["softmax_scale"]),
        ),
    )
    bingo_dfg.bingo_add_node(node_host_compute_softmax)

    # 节点: 生成 Scatter 元数据
    node_host_build_scatter_meta = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_build_scatter_metadata",
        kernel_args=HostBingoKernelBuildScatterMetadataArgs(
            global_top_k_indices_addr=mem_handles["L3_Alloc_Router_TopK_Indices"],
            actual_total_tokens=total_tokens,
            expert_token_counts_addr=mem_handles["L3_Alloc_Expert_Histogram"],
            expert_memory_offsets_addr=mem_handles["L3_Alloc_Expert_Offsets"],
            reverse_original_token_flat_idx_addr=mem_handles["L3_Alloc_Reverse_Index"],
            expert_number_each_layer=params["expert_number_each_layer"],
            individual_expert_number_k=params["individual_expert_number_k"],
        ),
    )
    bingo_dfg.bingo_add_node(node_host_build_scatter_meta)

    # =========================================================================
    # 辅助变量：算出每个专家占据的权重总字节数 (用于 B 矩阵的全局地址跨步)
    # =========================================================================
    shared_gate_B_expert_stride = (
        params["shared_swish_glu_N2"] * params["shared_swish_glu_B_tileSize"]
    )
    shared_up_B_expert_stride = (
        params["shared_swish_glu_N2"] * params["shared_swish_glu_B_tileSize"]
    )
    # dual-VC down B: each expert stores [vc0_N2_tiles, vc1_N2_tiles] consecutively
    shared_down_B_vc_stride = (
        params["shared_down_projection_N2"]
        * params["shared_down_projection_B_tileSize"]
    )
    shared_down_B_expert_stride = 2 * shared_down_B_vc_stride

    indiv_gate_B_expert_stride = (
        params["individual_swish_glu_N2"] * params["individual_swish_glu_B_tileSize"]
    )
    indiv_up_B_expert_stride = (
        params["individual_swish_glu_N2"] * params["individual_swish_glu_B_tileSize"]
    )
    # dual-VC down B for individual experts
    indiv_down_B_vc_stride = (
        params["individual_down_projection_N2"]
        * params["individual_down_projection_B_tileSize"]
    )
    indiv_down_B_expert_stride = 2 * indiv_down_B_vc_stride

    # =========================================================================
    # 3. 阶段 3: Shared Expert 任务定义 (引入 shared_expert_number_k 循环)
    # =========================================================================
    shared_nodes = {
        "load_A": {},  # Key: (se_id, m)
        "load_gate_B": {},  # Key: (se_id, m, n)  — preload into L1_gate_B
        "load_up_B": {},  # Key: (se_id, m, n)  — preload into L1_up_B
        "swiglu_gemm": {},  # Key: (se_id, m, n)  — DualVcSwiglu per N2 tile
        "load_down_B0": {},  # Key: (se_id, m, n)  — VC0 down B
        "load_down_B1": {},  # Key: (se_id, m, n)  — VC1 down B
        "down_gemm": {},  # Key: (se_id, m, n)  — DualVcGemm
        "store_down_D0": {},  # Key: (se_id, m, n)  — D0 to L3
        "store_down_D1": {},  # Key: (se_id, m, n)  — D1 to L3
    }

    for se_id in range(params["shared_expert_number_k"]):
        for m in range(params["shared_swish_glu_M2"]):
            # Input A 只与 M 有关，每个 M 块只需要搬运一次
            if has_phys_A_pingpong and params["shared_swish_glu_M2"] > 1:
                current_l1_A = (
                    mem_handles["L1_Buf_Unified_A_ping"]
                    if m % 2 == 0
                    else mem_handles["L1_Buf_Unified_A_pong"]
                )
            elif has_phys_A_pingpong:
                current_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
            else:
                current_l1_A = mem_handles["L1_Buf_Unified_A"]

            # 节点：DMA Load Shared Input A
            node_shared_load_A = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=dma_core_id,
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=mem_handles[f"L3_Sym_Shared_Swish_Input_A_tile_{m}"],
                    dst_addr=current_l1_A,
                    size=params["shared_swish_glu_A_tileSize"],
                ),
            )
            shared_nodes["load_A"][(se_id, m)] = node_shared_load_A
            bingo_dfg.bingo_add_node(node_shared_load_A)

            # ------------------ 预加载 gate_B 和 up_B 全部 N2 tile ------------------
            # dual-VC SwiGLU 需要同时提供 gate_B 和 up_B，因此先把全部 N2 tile
            # 分别预加载到 L1_Buf_Gate_B_N2 和 L1_Buf_Up_B_N2。
            for n in range(params["shared_swish_glu_N2"]):
                gate_B_src = BingoMemSymbol(
                    "shared_experts_gate_B",
                    offset=se_id * shared_gate_B_expert_stride
                    + n * params["shared_swish_glu_B_tileSize"],
                )
                node_load_gate_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=gate_B_src,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Gate_B_N2"],
                            n * params["shared_swish_glu_B_tileSize"],
                        ),
                        size=params["shared_swish_glu_B_tileSize"],
                    ),
                )
                shared_nodes["load_gate_B"][(se_id, m, n)] = node_load_gate_B
                bingo_dfg.bingo_add_node(node_load_gate_B)

                up_B_src = BingoMemSymbol(
                    "shared_experts_up_projection_B",
                    offset=se_id * shared_up_B_expert_stride
                    + n * params["shared_swish_glu_B_tileSize"],
                )
                node_load_up_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=up_B_src,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Up_B_N2"],
                            n * params["shared_swish_glu_B_tileSize"],
                        ),
                        size=params["shared_swish_glu_B_tileSize"],
                    ),
                )
                shared_nodes["load_up_B"][(se_id, m, n)] = node_load_up_B
                bingo_dfg.bingo_add_node(node_load_up_B)

            # ------------------ dual-VC SwiGLU 循环 (N2) ------------------
            for n in range(params["shared_swish_glu_N2"]):
                node_swiglu = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name="__snax_bingo_kernel_dual_vc_swiglu_full",
                    kernel_args=SnaxBingoKernelDualVcSwigluFullArgs(
                        input_A_addr=current_l1_A,
                        input_B_gate_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Gate_B_N2"],
                            n * params["shared_swish_glu_B_tileSize"],
                        ),
                        input_B_up_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Up_B_N2"],
                            n * params["shared_swish_glu_B_tileSize"],
                        ),
                        output_D0_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Swiglu_D_N2"],
                            n * params["shared_swish_glu_D_tileSize"],
                        ),
                        output_D1_addr=mem_handles["L1_Buf_D1_Scratch"],  # 丢弃
                        M=params["shared_swish_glu_M1"],
                        K=params["shared_swish_glu_K1"],
                        N=params["shared_swish_glu_N1"],
                        array_shape=params["array_shape"],
                        rescale_mult=1,
                        rescale_shift=0,
                    ),
                )
                shared_nodes["swiglu_gemm"][(se_id, m, n)] = node_swiglu
                bingo_dfg.bingo_add_node(node_swiglu)

            # ------------------ dual-VC Down Projection 循环 (N2) ------------------
            # SwiGLU 输出已在 L1_Buf_Swiglu_D_N2，作为 Down Projection 的 A。
            # 每次 DualVcGemm 调用: B0=vc0 tile, B1=vc1 tile → D0, D1 分别写到 L3。
            for n in range(params["shared_down_projection_N2"]):
                # 下投影 B0 (vc0 half)
                down_B0_src = BingoMemSymbol(
                    "shared_experts_down_projection_B",
                    offset=se_id * shared_down_B_expert_stride
                    + n * params["shared_down_projection_B_tileSize"],
                )
                node_load_down_B0 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=down_B0_src,
                        dst_addr=mem_handles["L1_Buf_Down_B0"],
                        size=params["shared_down_projection_B_tileSize"],
                    ),
                )
                shared_nodes["load_down_B0"][(se_id, m, n)] = node_load_down_B0
                bingo_dfg.bingo_add_node(node_load_down_B0)

                # 下投影 B1 (vc1 half)
                down_B1_src = BingoMemSymbol(
                    "shared_experts_down_projection_B",
                    offset=se_id * shared_down_B_expert_stride
                    + shared_down_B_vc_stride
                    + n * params["shared_down_projection_B_tileSize"],
                )
                node_load_down_B1 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=down_B1_src,
                        dst_addr=mem_handles["L1_Buf_Down_B1"],
                        size=params["shared_down_projection_B_tileSize"],
                    ),
                )
                shared_nodes["load_down_B1"][(se_id, m, n)] = node_load_down_B1
                bingo_dfg.bingo_add_node(node_load_down_B1)

                # dual-VC GEMM Down
                node_down_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name="__snax_bingo_kernel_dual_vc_gemm_full",
                    kernel_args=SnaxBingoKernelDualVcGemmFullArgs(
                        input_A_addr=mem_handles["L1_Buf_Swiglu_D_N2"],
                        input_B0_addr=mem_handles["L1_Buf_Down_B0"],
                        input_B1_addr=mem_handles["L1_Buf_Down_B1"],
                        output_D0_addr=mem_handles["L1_Buf_Down_D0"],
                        output_D1_addr=mem_handles["L1_Buf_Down_D1"],
                        M=params["shared_down_projection_M1"],
                        K=params["shared_down_projection_K1"],
                        N=params["shared_down_projection_N1"],
                        array_shape=params["array_shape"],
                        rescale_mult=1,
                        rescale_shift=0,
                    ),
                )
                shared_nodes["down_gemm"][(se_id, m, n)] = node_down_gemm
                bingo_dfg.bingo_add_node(node_down_gemm)

                # Store D0 to L3
                down_D_base_offset = (
                    se_id
                    * params["shared_down_projection_M2"]
                    * params["shared_down_projection_N2"]
                    * 2  # factor 2: D0+D1
                    * params["shared_down_projection_D_tileSize"]
                    + (m * params["shared_down_projection_N2"] + n)
                    * 2
                    * params["shared_down_projection_D_tileSize"]
                )
                node_store_down_D0 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=mem_handles["L1_Buf_Down_D0"],
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Shared_Down_HW_Output"],
                            down_D_base_offset,
                        ),
                        size=params["shared_down_projection_D_tileSize"],
                    ),
                )
                shared_nodes["store_down_D0"][(se_id, m, n)] = node_store_down_D0
                bingo_dfg.bingo_add_node(node_store_down_D0)

                # Store D1 to L3 (immediately after D0)
                node_store_down_D1 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=mem_handles["L1_Buf_Down_D1"],
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Shared_Down_HW_Output"],
                            down_D_base_offset
                            + params["shared_down_projection_D_tileSize"],
                        ),
                        size=params["shared_down_projection_D_tileSize"],
                    ),
                )
                shared_nodes["store_down_D1"][(se_id, m, n)] = node_store_down_D1
                bingo_dfg.bingo_add_node(node_store_down_D1)

    # =========================================================================
    # 4. 阶段 4: Individual Expert 任务定义 (引入 expert_number_each_layer 循环)
    # =========================================================================
    indiv_nodes = {
        "scatter_pad": {},  # Key: (expert_id,)
        "load_gate_B": {},  # Key: (expert_id, m, n)
        "load_up_B": {},  # Key: (expert_id, m, n)
        "swiglu_gemm": {},  # Key: (expert_id, m, n)
        "load_down_B0": {},  # Key: (expert_id, m, n)
        "load_down_B1": {},  # Key: (expert_id, m, n)
        "down_gemm": {},  # Key: (expert_id, m, n)
        "store_down_D0": {},  # Key: (expert_id, m, n)
        "store_down_D1": {},  # Key: (expert_id, m, n)
    }

    for expert_id in range(params["expert_number_each_layer"]):

        # 确定单个专家输入所需的绝对行数
        max_tokens_per_expert = (
            params["individual_swish_glu_M1"]
            * params["meshRow"]
            * params["individual_swish_glu_M2"]
        )

        # L1 A 缓冲（scatter 直接写入目标）
        if has_phys_A_pingpong and params["individual_swish_glu_M2"] > 1:
            scatter_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
        elif has_phys_A_pingpong:
            scatter_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
        else:
            scatter_l1_A = mem_handles["L1_Buf_Unified_A"]

        # 节点：Host scatter (L3 global input → L1, 一跳)
        node_indiv_scatter = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            kernel_name="__host_bingo_kernel_scatter_and_pad_input",
            kernel_args=HostBingoKernelScatterAndPadArgs(
                expert_id=expert_id,
                global_input_A_addr=mem_handles["L3_Sym_Input_A"],
                padded_scatter_pool_addr=scatter_l1_A,
                expert_token_counts_addr=mem_handles["L3_Alloc_Expert_Histogram"],
                expert_memory_offsets_addr=mem_handles["L3_Alloc_Expert_Offsets"],
                reverse_original_token_flat_idx_addr=mem_handles[
                    "L3_Alloc_Reverse_Index"
                ],
                input_dimension=params["input_dimension"],
                max_padded_tokens=max_tokens_per_expert,
                individual_expert_number_k=params["individual_expert_number_k"],
            ),
        )
        indiv_nodes["scatter_pad"][(expert_id,)] = node_indiv_scatter
        bingo_dfg.bingo_add_node(node_indiv_scatter)

        for m in range(params["individual_swish_glu_M2"]):
            if has_phys_A_pingpong and params["individual_swish_glu_M2"] > 1:
                current_l1_A = (
                    mem_handles["L1_Buf_Unified_A_ping"]
                    if m % 2 == 0
                    else mem_handles["L1_Buf_Unified_A_pong"]
                )
            elif has_phys_A_pingpong:
                current_l1_A = mem_handles["L1_Buf_Unified_A_ping"]
            else:
                current_l1_A = mem_handles["L1_Buf_Unified_A"]

            # ------------------ 预加载 gate_B 和 up_B (dual-VC SwiGLU) ------------------
            for n in range(params["individual_swish_glu_N2"]):
                gate_B_src = BingoMemSymbol(
                    "individual_experts_gate_B",
                    offset=expert_id * indiv_gate_B_expert_stride
                    + n * params["individual_swish_glu_B_tileSize"],
                )
                node_load_gate_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=gate_B_src,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Gate_B_N2"],
                            n * params["individual_swish_glu_B_tileSize"],
                        ),
                        size=params["individual_swish_glu_B_tileSize"],
                    ),
                )
                indiv_nodes["load_gate_B"][(expert_id, m, n)] = node_load_gate_B
                bingo_dfg.bingo_add_node(node_load_gate_B)

                up_B_src = BingoMemSymbol(
                    "individual_experts_up_projection_B",
                    offset=expert_id * indiv_up_B_expert_stride
                    + n * params["individual_swish_glu_B_tileSize"],
                )
                node_load_up_B = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=up_B_src,
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Up_B_N2"],
                            n * params["individual_swish_glu_B_tileSize"],
                        ),
                        size=params["individual_swish_glu_B_tileSize"],
                    ),
                )
                indiv_nodes["load_up_B"][(expert_id, m, n)] = node_load_up_B
                bingo_dfg.bingo_add_node(node_load_up_B)

            # ------------------ dual-VC SwiGLU 循环 (N2) ------------------
            for n in range(params["individual_swish_glu_N2"]):
                node_swiglu = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name="__snax_bingo_kernel_dual_vc_swiglu_full",
                    kernel_args=SnaxBingoKernelDualVcSwigluFullArgs(
                        input_A_addr=current_l1_A,
                        input_B_gate_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Gate_B_N2"],
                            n * params["individual_swish_glu_B_tileSize"],
                        ),
                        input_B_up_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Up_B_N2"],
                            n * params["individual_swish_glu_B_tileSize"],
                        ),
                        output_D0_addr=addr_with_optional_offset(
                            mem_handles["L1_Buf_Swiglu_D_N2"],
                            n * params["individual_swish_glu_D_tileSize"],
                        ),
                        output_D1_addr=mem_handles["L1_Buf_D1_Scratch"],
                        M=params["individual_swish_glu_M1"],
                        K=params["individual_swish_glu_K1"],
                        N=params["individual_swish_glu_N1"],
                        array_shape=params["array_shape"],
                        rescale_mult=1,
                        rescale_shift=0,
                    ),
                )
                indiv_nodes["swiglu_gemm"][(expert_id, m, n)] = node_swiglu
                bingo_dfg.bingo_add_node(node_swiglu)

            # ------------------ dual-VC Down Projection 循环 (N2) ------------------
            indiv_down_B_vc_stride = (
                params["individual_down_projection_N2"]
                * params["individual_down_projection_B_tileSize"]
            )

            for n in range(params["individual_down_projection_N2"]):
                down_B0_src = BingoMemSymbol(
                    "individual_experts_down_projection_B",
                    offset=expert_id * indiv_down_B_expert_stride
                    + n * params["individual_down_projection_B_tileSize"],
                )
                node_load_down_B0 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=down_B0_src,
                        dst_addr=mem_handles["L1_Buf_Down_B0"],
                        size=params["individual_down_projection_B_tileSize"],
                    ),
                )
                indiv_nodes["load_down_B0"][(expert_id, m, n)] = node_load_down_B0
                bingo_dfg.bingo_add_node(node_load_down_B0)

                down_B1_src = BingoMemSymbol(
                    "individual_experts_down_projection_B",
                    offset=expert_id * indiv_down_B_expert_stride
                    + indiv_down_B_vc_stride
                    + n * params["individual_down_projection_B_tileSize"],
                )
                node_load_down_B1 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=dma_core_id,
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=down_B1_src,
                        dst_addr=mem_handles["L1_Buf_Down_B1"],
                        size=params["individual_down_projection_B_tileSize"],
                    ),
                )
                indiv_nodes["load_down_B1"][(expert_id, m, n)] = node_load_down_B1
                bingo_dfg.bingo_add_node(node_load_down_B1)

                node_down_gemm = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=gemm_core_id,
                    kernel_name="__snax_bingo_kernel_dual_vc_gemm_full",
                    kernel_args=SnaxBingoKernelDualVcGemmFullArgs(
                        input_A_addr=mem_handles["L1_Buf_Swiglu_D_N2"],
                        input_B0_addr=mem_handles["L1_Buf_Down_B0"],
                        input_B1_addr=mem_handles["L1_Buf_Down_B1"],
                        output_D0_addr=mem_handles["L1_Buf_Down_D0"],
                        output_D1_addr=mem_handles["L1_Buf_Down_D1"],
                        M=params["individual_down_projection_M1"],
                        K=params["individual_down_projection_K1"],
                        N=params["individual_down_projection_N1"],
                        array_shape=params["array_shape"],
                        rescale_mult=1,
                        rescale_shift=0,
                    ),
                )
                indiv_nodes["down_gemm"][(expert_id, m, n)] = node_down_gemm
                bingo_dfg.bingo_add_node(node_down_gemm)

                # D0 offset: expert * M2 * N2 * 2 * D_tileSize + (m*N2+n) * 2 * D_tileSize
                indiv_down_D_base_offset = (
                    expert_id
                    * params["individual_down_projection_M2"]
                    * params["individual_down_projection_N2"]
                    * 2
                    * params["individual_down_projection_D_tileSize"]
                    + (m * params["individual_down_projection_N2"] + n)
                    * 2
                    * params["individual_down_projection_D_tileSize"]
                )
                node_store_down_D0 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=mem_handles["L1_Buf_Down_D0"],
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Indiv_Down_HW_Output"],
                            indiv_down_D_base_offset,
                        ),
                        size=params["individual_down_projection_D_tileSize"],
                    ),
                )
                indiv_nodes["store_down_D0"][(expert_id, m, n)] = node_store_down_D0
                bingo_dfg.bingo_add_node(node_store_down_D0)

                node_store_down_D1 = BingoNode(
                    assigned_chiplet_id=0,
                    assigned_cluster_id=0,
                    assigned_core_id=host_core_id,
                    kernel_name="__host_bingo_kernel_idma",
                    kernel_args=HostBingoKernelIdmaArgs(
                        src_addr=mem_handles["L1_Buf_Down_D1"],
                        dst_addr=addr_with_optional_offset(
                            mem_handles["L3_Alloc_Indiv_Down_HW_Output"],
                            indiv_down_D_base_offset
                            + params["individual_down_projection_D_tileSize"],
                        ),
                        size=params["individual_down_projection_D_tileSize"],
                    ),
                )
                indiv_nodes["store_down_D1"][(expert_id, m, n)] = node_store_down_D1
                bingo_dfg.bingo_add_node(node_store_down_D1)
    # =========================================================================
    # 5. 阶段 5: Final Accumulate (归约与加权)
    # =========================================================================
    node_accumulate = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,  # 必须由 CVA6 核心执行
        kernel_name="__host_bingo_kernel_experts_result_accumulate",
        kernel_args=HostBingoKernelExpertsResultAccumulateArgs(
            shared_expert_hw_output_addr=mem_handles["L3_Alloc_Shared_Down_HW_Output"],
            individual_experts_hw_output_addr=mem_handles[
                "L3_Alloc_Indiv_Down_HW_Output"
            ],
            reverse_original_token_flat_idx_addr=mem_handles["L3_Alloc_Reverse_Index"],
            global_calculated_probability_addr=mem_handles[
                "L3_Alloc_Router_Probabilities"
            ],
            final_layer_output_addr=mem_handles["L3_Alloc_Final_MoE_Output"],
            # 这里必须使用真实的全局 Token 总数
            actual_total_tokens=total_tokens,
            input_dimension=params["input_dimension"],
            individual_expert_number_k=params["individual_expert_number_k"],
            softmax_scale_raw=float_to_uint64_bits(params["softmax_scale"]),
        ),
    )
    bingo_dfg.bingo_add_node(node_accumulate)

    # define dependency
    # 1. router
    # Keep Router startup single-rooted: B must be loaded before first A tile launch.
    bingo_dfg.bingo_add_edge(node_router_copy_B, router_copy_A_nodes[0])
    bingo_dfg.bingo_add_edge(node_router_copy_B, router_gemm_nodes[0])
    for m in range(params["router_M2"]):
        bingo_dfg.bingo_add_edge(router_copy_A_nodes[m], router_gemm_nodes[m])
        bingo_dfg.bingo_add_edge(router_gemm_nodes[m], router_copy_D_nodes[m])

        if m < params["router_M2"] - 1:
            bingo_dfg.bingo_add_edge(router_copy_A_nodes[m], router_copy_A_nodes[m + 1])
            bingo_dfg.bingo_add_edge(router_gemm_nodes[m], router_gemm_nodes[m + 1])
            bingo_dfg.bingo_add_edge(router_copy_D_nodes[m], router_copy_D_nodes[m + 1])

            bingo_dfg.bingo_add_edge(router_copy_A_nodes[m + 1], router_copy_D_nodes[m])

        if m < params["router_M2"] - 2:
            bingo_dfg.bingo_add_edge(router_gemm_nodes[m], router_copy_A_nodes[m + 2])
            bingo_dfg.bingo_add_edge(router_copy_D_nodes[m], router_gemm_nodes[m + 2])

        if m < params["router_M2"] - 3:
            bingo_dfg.bingo_add_edge(router_copy_D_nodes[m], router_copy_A_nodes[m + 3])

    bingo_dfg.bingo_add_edge(router_copy_D_nodes[-1], node_host_router_topk)
    bingo_dfg.bingo_add_edge(node_host_router_topk, node_host_build_scatter_meta)
    bingo_dfg.bingo_add_edge(node_host_build_scatter_meta, node_host_compute_softmax)

    # Shared 阶段入口门控：仅第一个 shared expert 的 A tile 由 softmax 启动。
    # Gate/Up 的首个 B tile 统一通过阶段内数据流接力，避免把 host 入口
    # 直接扇出到多个 DMA 首节点，触发首块 descriptor 的双前驱膨胀。
    bingo_dfg.bingo_add_edge(node_host_compute_softmax, shared_nodes["load_A"][(0, 0)])

    # Shared expert 串行：后一个 expert 依赖前一个 expert 的 down projection 完成
    last_m_shared = params["shared_swish_glu_M2"] - 1
    last_n_shared = params["shared_down_projection_N2"] - 1
    for se_id in range(1, params["shared_expert_number_k"]):
        bingo_dfg.bingo_add_edge(
            shared_nodes["store_down_D1"][(se_id - 1, last_m_shared, last_n_shared)],
            shared_nodes["load_A"][(se_id, 0)],
        )

    # 2. share expert
    # =========================================================================
    # 定义依赖关系: Shared Expert 模块 (dual-VC 版本)
    # =========================================================================

    M2 = params["shared_swish_glu_M2"]
    N2 = params["shared_swish_glu_N2"]
    N2_down = params["shared_down_projection_N2"]

    for se_id in range(params["shared_expert_number_k"]):
        for m in range(M2):
            idx_m = (se_id, m)

            # -----------------------------------------------------------------
            # 1. SwiGLU 预加载阶段: 并行预取所有 N2 的 gate_B 和 up_B
            #    load_A 完成后，顺序触发 gate_B[0]→gate_B[1]→... 和 up_B[0]→up_B[1]→...
            # -----------------------------------------------------------------
            for n in range(N2):
                idx_mn = (se_id, m, n)

                if n == 0:
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_A"][idx_m],
                        shared_nodes["load_gate_B"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_A"][idx_m],
                        shared_nodes["load_up_B"][idx_mn],
                    )
                else:
                    prev_mn = (se_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_gate_B"][prev_mn],
                        shared_nodes["load_gate_B"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_up_B"][prev_mn],
                        shared_nodes["load_up_B"][idx_mn],
                    )

                # gate_B[n] and up_B[n] both must finish before swiglu_gemm[n]
                bingo_dfg.bingo_add_edge(
                    shared_nodes["load_gate_B"][idx_mn],
                    shared_nodes["swiglu_gemm"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    shared_nodes["load_up_B"][idx_mn],
                    shared_nodes["swiglu_gemm"][idx_mn],
                )

                # swiglu_gemm must run in order (L1 output buffer reuse)
                if n > 0:
                    prev_mn = (se_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["swiglu_gemm"][prev_mn],
                        shared_nodes["swiglu_gemm"][idx_mn],
                    )

            # -----------------------------------------------------------------
            # 2. Down Projection 循环 (dual-VC GEMM)
            #    在所有 SwiGLU 完成后开始；B0/B1 顺序加载
            # -----------------------------------------------------------------
            for n in range(N2_down):
                idx_mn = (se_id, m, n)

                if n == 0:
                    # 等待所有 SwiGLU tile 全部完成后才开始 down projection
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["swiglu_gemm"][(se_id, m, N2 - 1)],
                        shared_nodes["load_down_B0"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["swiglu_gemm"][(se_id, m, N2 - 1)],
                        shared_nodes["load_down_B1"][idx_mn],
                    )
                else:
                    prev_mn = (se_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_down_B0"][prev_mn],
                        shared_nodes["load_down_B0"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["load_down_B1"][prev_mn],
                        shared_nodes["load_down_B1"][idx_mn],
                    )

                bingo_dfg.bingo_add_edge(
                    shared_nodes["load_down_B0"][idx_mn],
                    shared_nodes["down_gemm"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    shared_nodes["load_down_B1"][idx_mn],
                    shared_nodes["down_gemm"][idx_mn],
                )

                if n > 0:
                    prev_mn = (se_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["down_gemm"][prev_mn],
                        shared_nodes["down_gemm"][idx_mn],
                    )

                bingo_dfg.bingo_add_edge(
                    shared_nodes["down_gemm"][idx_mn],
                    shared_nodes["store_down_D0"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    shared_nodes["down_gemm"][idx_mn],
                    shared_nodes["store_down_D1"][idx_mn],
                )

                if n < N2_down - 1:
                    next_mn = (se_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_down_D0"][idx_mn],
                        shared_nodes["store_down_D0"][next_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        shared_nodes["store_down_D1"][idx_mn],
                        shared_nodes["store_down_D1"][next_mn],
                    )

            # -----------------------------------------------------------------
            # 3. M 维度的向前看防御 (M-level Forward-Looking WAR)
            # -----------------------------------------------------------------
            if m < M2 - 1:
                bingo_dfg.bingo_add_edge(
                    shared_nodes["down_gemm"][(se_id, m, N2_down - 1)],
                    shared_nodes["load_A"][(se_id, m + 1)],
                )

    # 3. individual expert
    # ==================================================================================
    E = params["expert_number_each_layer"]
    M2 = params["individual_swish_glu_M2"]
    N2_gate = params["individual_swish_glu_N2"]
    N2_down = params["individual_down_projection_N2"]

    # [全局入口] Shared 严格串行后，只需最后一个 Shared Expert 收工来启动 Individual。
    last_shared_expert = params["shared_expert_number_k"] - 1
    last_m_shared = params["shared_swish_glu_M2"] - 1
    last_n_shared = params["shared_down_projection_N2"] - 1
    last_shared_store_down_D = shared_nodes["store_down_D1"][
        (last_shared_expert, last_m_shared, last_n_shared)
    ]
    bingo_dfg.bingo_add_edge(
        last_shared_store_down_D,
        indiv_nodes["scatter_pad"][(0,)],
    )

    # [元数据入口] build_scatter_meta → scatter_pad[(0,)] 这条直接边已被移除。
    # 原因：build_scatter_meta 已通过 topk→build_scatter_meta→softmax→shared_load_A→
    # ...→last_shared_store_down_D→scatter_pad 这条串行路径，先于 scatter_pad 完成。
    # 若再加一条 build_scatter_meta→scatter_pad 的直接边，scatter_pad 有两个 Core2
    # 前驱（build_scatter_meta 和 last_shared_store_down_D），触发 dummy_check 插入
    # WAIT_NO_SIG dummy 节点，从而消耗 dep_matrix 中的单比特令牌，造成死锁。
    for expert_id in range(E):

        # Inter-expert 串行：下一个 expert scatter 必须等上一个 expert 的 store_down_D1 完成
        if expert_id < E - 1:
            bingo_dfg.bingo_add_edge(
                indiv_nodes["store_down_D1"][(expert_id, M2 - 1, N2_down - 1)],
                indiv_nodes["scatter_pad"][(expert_id + 1,)],
            )

        for m in range(M2):
            idx_m = (expert_id, m)

            # -----------------------------------------------------------------
            # 1. SwiGLU 预加载：scatter_pad 触发 gate_B[0] 和 up_B[0] 顺序加载
            # -----------------------------------------------------------------
            for n in range(N2_gate):
                idx_mn = (expert_id, m, n)

                if n == 0:
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["scatter_pad"][(expert_id,)],
                        indiv_nodes["load_gate_B"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["scatter_pad"][(expert_id,)],
                        indiv_nodes["load_up_B"][idx_mn],
                    )
                else:
                    prev_mn = (expert_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["load_gate_B"][prev_mn],
                        indiv_nodes["load_gate_B"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["load_up_B"][prev_mn],
                        indiv_nodes["load_up_B"][idx_mn],
                    )

                bingo_dfg.bingo_add_edge(
                    indiv_nodes["load_gate_B"][idx_mn],
                    indiv_nodes["swiglu_gemm"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["load_up_B"][idx_mn],
                    indiv_nodes["swiglu_gemm"][idx_mn],
                )

                if n > 0:
                    prev_mn = (expert_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["swiglu_gemm"][prev_mn],
                        indiv_nodes["swiglu_gemm"][idx_mn],
                    )

            # -----------------------------------------------------------------
            # 2. Down Projection 循环 (dual-VC GEMM)
            # -----------------------------------------------------------------
            for n in range(N2_down):
                idx_mn = (expert_id, m, n)

                if n == 0:
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["swiglu_gemm"][(expert_id, m, N2_gate - 1)],
                        indiv_nodes["load_down_B0"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["swiglu_gemm"][(expert_id, m, N2_gate - 1)],
                        indiv_nodes["load_down_B1"][idx_mn],
                    )
                else:
                    prev_mn = (expert_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["load_down_B0"][prev_mn],
                        indiv_nodes["load_down_B0"][idx_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["load_down_B1"][prev_mn],
                        indiv_nodes["load_down_B1"][idx_mn],
                    )

                bingo_dfg.bingo_add_edge(
                    indiv_nodes["load_down_B0"][idx_mn],
                    indiv_nodes["down_gemm"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["load_down_B1"][idx_mn],
                    indiv_nodes["down_gemm"][idx_mn],
                )

                if n > 0:
                    prev_mn = (expert_id, m, n - 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["down_gemm"][prev_mn],
                        indiv_nodes["down_gemm"][idx_mn],
                    )

                bingo_dfg.bingo_add_edge(
                    indiv_nodes["down_gemm"][idx_mn],
                    indiv_nodes["store_down_D0"][idx_mn],
                )
                bingo_dfg.bingo_add_edge(
                    indiv_nodes["down_gemm"][idx_mn],
                    indiv_nodes["store_down_D1"][idx_mn],
                )

                if n < N2_down - 1:
                    next_mn = (expert_id, m, n + 1)
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["store_down_D0"][idx_mn],
                        indiv_nodes["store_down_D0"][next_mn],
                    )
                    bingo_dfg.bingo_add_edge(
                        indiv_nodes["store_down_D1"][idx_mn],
                        indiv_nodes["store_down_D1"][next_mn],
                    )

    # =========================================================================
    # 5. 全局同步屏障：Accumulation (最终结果归约)
    # =========================================================================
    last_indiv_expert = params["expert_number_each_layer"] - 1
    last_m_indiv = params["individual_swish_glu_M2"] - 1
    last_n_indiv = params["individual_down_projection_N2"] - 1
    bingo_dfg.bingo_add_edge(
        indiv_nodes["store_down_D1"][(last_indiv_expert, last_m_indiv, last_n_indiv)],
        node_accumulate,
    )

    enforce_in_order_completion_per_core(bingo_dfg)

    return bingo_dfg


def main():
    args = get_args()
    output_dir = args.output_dir
    output_file_name = args.output_offload_file_name
    print(f"Output DIR: {output_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Execute Pipeline
    merged_config = load_workload_config(args)
    params = define_workload_params(**merged_config)
    mem_handles = define_memory_handles(params)
    dfg = create_dfg(params, mem_handles)
    data_header_name = f"{params['app_name']}_data.h"
    dfg.bingo_compile_dfg(
        params["app_name"],
        output_dir,
        output_file_name,
        extra_include_header_list=[data_header_name],
    )


if __name__ == "__main__":
    main()
