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
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelDualDmaArgs,
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
    """简化 DFG：验证 iDMA + xDMA 并行执行 + dual-VC SwiGLU 的时序。

    只含 3 个节点：
      Node 0 (core=1, iDMA):      加载 input A                  (16KB)
      Node 1 (core=1, dual_dma):  并行加载 gate B (iDMA) + up B (xDMA)  (各 512KB)
      Node 2 (core=0, GEMM):      dual-VC SwiGLU 计算，依赖 Node 0/1

    Node 1 内部先非阻塞发射 xDMA，再非阻塞发射 iDMA，最后统一等待，
    两个独立 DMA 引擎在硬件层面形成传输重叠。
    """
    bingo_dfg = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=4,  # 必须与硬件物理集群数 NrClustersPerQuad=4 一致，否则 cluster_id 位宽不匹配导致 core_id 字段错位
        num_cores_per_cluster=2,
        is_host_as_acc=True,
        chiplet_ids=[0x00],
    )

    gemm_core_id = 0  # VersaCore 计算核
    dma_core_id = 1  # 集群 DM 核（iDMA + xDMA 均由此核发起）

    # ---- 缓冲区选择（M2=1，不需要 Ping-Pong）----
    has_phys_A_pingpong = "L1_Buf_Unified_A_ping" in mem_handles
    l1_A = (
        mem_handles["L1_Buf_Unified_A_ping"]
        if has_phys_A_pingpong
        else mem_handles["L1_Buf_Unified_A"]
    )

    A_size = params["shared_swish_glu_A_tileSize"]  # 16 KB
    B_size = params["shared_swish_glu_B_tileSize"]  # 512 KB（N1=256）

    # ---- Node 0: iDMA — 加载 input A (16KB) ----
    node_load_A = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["L3_Sym_Shared_Swish_Input_A_tile_0"],
            dst_addr=l1_A,
            size=A_size,
        ),
    )
    bingo_dfg.bingo_add_node(node_load_A)

    # ---- Node 1: Dual DMA — 同时加载 gate B (iDMA) + up-proj B (xDMA)，各 512KB ----
    # iDMA 和 xDMA 在同一个 Bingo 节点内依次发射（均非阻塞），再统一等待，
    # 两个 DMA 引擎在硬件层面形成传输重叠。
    # B_size=524288 = 64×8192，满足 xDMA 64 字节对齐要求。
    node_dual_dma = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_dual_dma",
        kernel_args=SnaxBingoKernelDualDmaArgs(
            idma_src_addr=mem_handles["L3_Sym_Shared_Gate_Weight_B_tile_0"],
            idma_dst_addr=mem_handles["L1_Buf_Gate_B_N2"],
            idma_size=B_size,
            xdma_src_addr=mem_handles["L3_Sym_Shared_Up_Weight_B_tile_0"],
            xdma_dst_addr=mem_handles["L1_Buf_Up_B_N2"],
            xdma_size=B_size,
        ),
    )
    bingo_dfg.bingo_add_node(node_dual_dma)

    # ---- Node 3: dual-VC SwiGLU 计算 ----
    # 依赖 Node 0/1/2 全部完成后才开始。
    node_swiglu = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=gemm_core_id,
        kernel_name="__snax_bingo_kernel_dual_vc_swiglu_full",
        kernel_args=SnaxBingoKernelDualVcSwigluFullArgs(
            input_A_addr=l1_A,
            input_B_gate_addr=mem_handles["L1_Buf_Gate_B_N2"],
            input_B_up_addr=mem_handles["L1_Buf_Up_B_N2"],
            output_D0_addr=mem_handles["L1_Buf_Swiglu_D_N2"],
            output_D1_addr=mem_handles["L1_Buf_D1_Scratch"],
            M=params["shared_swish_glu_M1"],
            K=params["shared_swish_glu_K1"],
            N=params["shared_swish_glu_N1"],
            array_shape=params["array_shape"],
            rescale_mult=1,
            rescale_shift=0,
        ),
    )
    bingo_dfg.bingo_add_node(node_swiglu)

    # ---- 依赖边 ----
    # Node 0 → Node 1：确保 input A 加载完成后再发起 B weight 传输
    bingo_dfg.bingo_add_edge(node_load_A, node_dual_dma)
    # Node 1 → Node 2：gate B + up B 全部就绪后才能开始 SwiGLU 计算
    bingo_dfg.bingo_add_edge(node_dual_dma, node_swiglu)

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
