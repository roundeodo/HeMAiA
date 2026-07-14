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
    SnaxBingoKernelDualDmaArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelMoeDownArgs,
    SnaxBingoKernelMoeInitOutputPaddingArgs,
    SnaxBingoKernelMoeSwigluArgs,
    SnaxBingoKernelXdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from moe_test_layout import derive_params  # noqa: E402

GEMM_CORE = 0
DMA_CORE = 1
TEST_CLUSTERS = (("c0", 0),)


def offset(handle, byte_offset: int):
    if byte_offset == 0:
        return handle
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, offset=handle.offset + byte_offset)
    return f"{handle.get_c_var_name()} + {byte_offset}"


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


def define_memory(p):
    mh = {"input": BingoMemSymbol("moe_test_input_A")}
    mh["next_gate_src"] = BingoMemSymbol("moe_test_next_gate_B")
    mh["next_up_src"] = BingoMemSymbol("moe_test_next_up_B")
    for prefix, cluster in TEST_CLUSTERS:
        mh[f"{prefix}_gate_src"] = BingoMemSymbol(f"moe_test_{prefix}_gate_B")
        mh[f"{prefix}_up_src"] = BingoMemSymbol(f"moe_test_{prefix}_up_B")
        mh[f"{prefix}_down_src"] = BingoMemSymbol(f"moe_test_{prefix}_down_B")
        for name, size in (
            ("a", p["token_buffer_bytes"]),
            ("gate", p["gate_weight_bytes"]),
            ("up", p["gate_weight_bytes"]),
            ("down", p["down_weight_bytes"]),
            ("gate_out", p["gate_output_bytes"]),
            ("gate_scratch", p["gate_scratch_bytes"]),
            ("out", p["output_bytes"]),
            ("next_gate", p["gate_weight_bytes"]),
            ("next_up", p["gate_weight_bytes"]),
        ):
            mh[f"{prefix}_{name}"] = BingoMemAlloc(
                f"moe_test_{prefix}_{name}",
                size=size,
                mem_level="L1",
                chip_id=0,
                cluster_id=cluster,
            )
        mh[f"{prefix}_out_l3"] = BingoMemAlloc(
            f"moe_test_{prefix}_out_l3", size=p["output_bytes"], mem_level="L3"
        )

    probe_bytes = p["dma_probe_bytes"]
    mh["probe_c0_arena"] = BingoMemAlloc(
        "moe_test_probe_c0_arena",
        size=2 * probe_bytes,
        mem_level="L1",
        chip_id=0,
        cluster_id=0,
    )
    mh["probe_c1_arena"] = BingoMemAlloc(
        "moe_test_probe_c1_arena",
        size=2 * probe_bytes + 64,
        mem_level="L1",
        chip_id=0,
        cluster_id=1,
    )
    mh["probe_out_l3"] = BingoMemAlloc(
        "moe_test_probe_out_l3", size=probe_bytes, mem_level="L3"
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


def add_slot(dfg, p, mh, prefix, cluster, start_dependencies=()):
    load_a = add_copy(
        dfg,
        cluster,
        mh["input"],
        mh[f"{prefix}_a"],
        p["token_buffer_bytes"],
        f"{prefix.upper()}_LOAD_8_TOKEN_ROWS_IDMA",
    )
    init_output_padding = add_node(
        dfg,
        cluster,
        GEMM_CORE,
        "__snax_bingo_kernel_moe_init_output_padding",
        SnaxBingoKernelMoeInitOutputPaddingArgs(
            mh[f"{prefix}_out"],
            p["token_payload_bytes"],
            p["token_row_stride"],
            p["total_tokens"],
        ),
        f"{prefix.upper()}_INIT_OUTPUT_PADDING",
    )
    for dependency in start_dependencies:
        dfg.bingo_add_edge(dependency, load_a)
        dfg.bingo_add_edge(dependency, init_output_padding)

    s1_compute = []
    previous_weight_load = load_a
    for block in range(p["block_count"]):
        weight_offset = block * p["gate_block_bytes"]
        load_gate = add_copy(
            dfg,
            cluster,
            offset(mh[f"{prefix}_gate_src"], weight_offset),
            offset(mh[f"{prefix}_gate"], weight_offset),
            p["gate_block_bytes"],
            f"{prefix.upper()}_S1_LOAD_GATE_BLOCK_{block}_IDMA",
        )
        load_up = add_copy(
            dfg,
            cluster,
            offset(mh[f"{prefix}_up_src"], weight_offset),
            offset(mh[f"{prefix}_up"], weight_offset),
            p["gate_block_bytes"],
            f"{prefix.upper()}_S1_LOAD_UP_BLOCK_{block}_IDMA",
        )
        dfg.bingo_add_edge(previous_weight_load, load_gate)
        dfg.bingo_add_edge(load_gate, load_up)
        compute = add_node(
            dfg,
            cluster,
            GEMM_CORE,
            "__snax_bingo_kernel_moe_swiglu",
            SnaxBingoKernelMoeSwigluArgs(
                mh[f"{prefix}_a"],
                offset(mh[f"{prefix}_gate"], weight_offset),
                offset(mh[f"{prefix}_up"], weight_offset),
                offset(
                    mh[f"{prefix}_gate_out"],
                    block * p["s1_rows"] * p["gate_block_row_bytes"],
                ),
                mh[f"{prefix}_gate_scratch"],
                M=p["s1_M"],
                K=p["gate_K"],
                N=p["gate_N_s1"],
                b_block_count=1,
                b_block_stride=p["gate_block_bytes"],
                array_shape=p["s1_shape"],
            ),
            f"{prefix.upper()}_S1_COMPUTE_BLOCK_{block}_SHAPE_4x8x16",
        )
        dfg.bingo_add_edge(load_up, compute)
        if s1_compute:
            dfg.bingo_add_edge(s1_compute[-1], compute)
        else:
            dfg.bingo_add_edge(init_output_padding, compute)
        s1_compute.append(compute)
        previous_weight_load = load_up

    s2 = add_node(
        dfg,
        cluster,
        GEMM_CORE,
        "__snax_bingo_kernel_moe_swiglu",
        SnaxBingoKernelMoeSwigluArgs(
            offset(mh[f"{prefix}_a"], p["s1_rows"] * p["token_row_stride"]),
            mh[f"{prefix}_gate"],
            mh[f"{prefix}_up"],
            offset(
                mh[f"{prefix}_gate_out"],
                p["s1_rows"] * p["gate_full_row_bytes"],
            ),
            mh[f"{prefix}_gate_scratch"],
            M=p["s2_M"],
            K=p["gate_K"],
            N=p["gate_N_s2"],
            b_block_count=p["block_count"],
            b_block_stride=p["gate_block_bytes"],
            array_shape=p["s2_shape"],
        ),
        f"{prefix.upper()}_S2_COMPUTE_TAIL",
    )
    dfg.bingo_add_edge(s1_compute[-1], s2)

    down_half_bytes = p["block_count"] * p["down_block_bytes"]
    prefetch_s2_down = add_node(
        dfg,
        cluster,
        DMA_CORE,
        "__snax_bingo_kernel_dual_dma",
        SnaxBingoKernelDualDmaArgs(
            mh[f"{prefix}_down_src"],
            mh[f"{prefix}_down"],
            down_half_bytes,
            offset(mh[f"{prefix}_down_src"], down_half_bytes),
            offset(mh[f"{prefix}_down"], down_half_bytes),
            down_half_bytes,
        ),
        f"{prefix.upper()}_S2_PREFETCH_DOWN_IDMA_XDMA",
    )
    # S2 compute and down prefetch become runnable from the same S1 boundary.
    dfg.bingo_add_edge(s1_compute[-1], prefetch_s2_down)

    s3 = add_node(
        dfg,
        cluster,
        GEMM_CORE,
        "__snax_bingo_kernel_moe_down",
        SnaxBingoKernelMoeDownArgs(
            mh[f"{prefix}_gate_out"],
            mh[f"{prefix}_down"],
            offset(mh[f"{prefix}_down"], down_half_bytes),
            mh[f"{prefix}_out"],
            offset(mh[f"{prefix}_out"], p["down_vc_row_bytes"]),
            M=p["s1_M"],
            K=p["down_K"],
            N=p["down_N_s3_full"],
            b_block_count=1,
            b_block_stride=down_half_bytes,
            array_shape=p["s1_shape"],
            output_row_stride=p["token_row_stride"],
        ),
        f"{prefix.upper()}_S3_COMPUTE_FULL_SHAPE_4x8x16",
    )
    dfg.bingo_add_edge(s2, s3)
    dfg.bingo_add_edge(prefetch_s2_down, s3)

    prefetch_next_swiglu = add_node(
        dfg,
        cluster,
        DMA_CORE,
        "__snax_bingo_kernel_dual_dma",
        SnaxBingoKernelDualDmaArgs(
            mh["next_gate_src"],
            mh[f"{prefix}_next_gate"],
            p["gate_weight_bytes"],
            mh["next_up_src"],
            mh[f"{prefix}_next_up"],
            p["gate_weight_bytes"],
        ),
        f"{prefix.upper()}_S3_PREFETCH_NEXT_GATE_UP_IDMA_XDMA",
    )
    # Start the next-expert prefetch at the same S3 boundary. The first edge
    # also prevents it from contending with the preceding dual-DMA prefetch.
    dfg.bingo_add_edge(prefetch_s2_down, prefetch_next_swiglu)
    dfg.bingo_add_edge(s2, prefetch_next_swiglu)

    tail_output_offset = p["s1_rows"] * p["token_row_stride"]
    s4 = add_node(
        dfg,
        cluster,
        GEMM_CORE,
        "__snax_bingo_kernel_moe_down",
        SnaxBingoKernelMoeDownArgs(
            offset(
                mh[f"{prefix}_gate_out"],
                p["s1_rows"] * p["gate_full_row_bytes"],
            ),
            mh[f"{prefix}_down"],
            offset(
                mh[f"{prefix}_down"],
                p["block_count"] * p["down_block_bytes"],
            ),
            offset(mh[f"{prefix}_out"], tail_output_offset),
            offset(
                mh[f"{prefix}_out"], tail_output_offset + p["down_vc_row_bytes"]
            ),
            M=p["s2_M"],
            K=p["down_K"],
            N=p["down_N_s4"],
            b_block_count=p["block_count"],
            b_block_stride=p["down_block_bytes"],
            array_shape=p["s2_shape"],
            output_row_stride=p["token_row_stride"],
        ),
        f"{prefix.upper()}_S4_COMPUTE_TAIL",
    )
    dfg.bingo_add_edge(s3, s4)

    store = add_copy(
        dfg,
        cluster,
        mh[f"{prefix}_out"],
        mh[f"{prefix}_out_l3"],
        p["output_bytes"],
        f"{prefix.upper()}_STORE_OUTPUT_IDMA",
    )
    dfg.bingo_add_edge(s4, store)
    dfg.bingo_add_edge(prefetch_next_swiglu, store)
    return store


def add_dma_probes(dfg, p, mh):
    size = p["dma_probe_bytes"]
    c0_arena = mh["probe_c0_arena"]

    idma_baseline = add_node(
        dfg,
        0,
        DMA_CORE,
        "__snax_bingo_kernel_idma_1d_copy",
        SnaxBingoKernelIdma1dCopyArgs(mh["c0_gate_src"], c0_arena, size),
        "DMA_PROBE_IDMA_C0_L3_TO_L1",
    )

    xdma_load_baseline = add_node(
        dfg,
        0,
        DMA_CORE,
        "__snax_bingo_kernel_xdma_1d_copy",
        SnaxBingoKernelXdma1dCopyArgs(mh["c0_gate_src"], c0_arena, size),
        "DMA_PROBE_XDMA_C0_L3_TO_L1",
    )
    dfg.bingo_add_edge(idma_baseline, xdma_load_baseline)

    xdma_store_baseline = add_node(
        dfg,
        0,
        DMA_CORE,
        "__snax_bingo_kernel_xdma_1d_copy",
        SnaxBingoKernelXdma1dCopyArgs(c0_arena, mh["probe_out_l3"], size),
        "DMA_PROBE_XDMA_C0_L1_TO_L3",
    )
    dfg.bingo_add_edge(xdma_load_baseline, xdma_store_baseline)

    dual_same_phase = add_node(
        dfg,
        0,
        DMA_CORE,
        "__snax_bingo_kernel_dual_dma",
        SnaxBingoKernelDualDmaArgs(
            mh["c0_gate_src"],
            offset(c0_arena, size),
            size,
            mh["c0_up_src"],
            c0_arena,
            size,
        ),
        "DMA_PROBE_DUAL_C0_SAME_BANK_PHASE",
    )
    dfg.bingo_add_edge(xdma_store_baseline, dual_same_phase)

    dual_shifted_phase = add_node(
        dfg,
        0,
        DMA_CORE,
        "__snax_bingo_kernel_dual_dma",
        SnaxBingoKernelDualDmaArgs(
            mh["c0_gate_src"],
            offset(c0_arena, size + 64),
            size,
            mh["c0_up_src"],
            c0_arena,
            size,
        ),
        "DMA_PROBE_DUAL_C0_SHIFTED_BANK_PHASE",
    )
    dfg.bingo_add_edge(dual_same_phase, dual_shifted_phase)

    cross_xdma_c0 = add_node(
        dfg,
        0,
        DMA_CORE,
        "__snax_bingo_kernel_xdma_1d_copy",
        SnaxBingoKernelXdma1dCopyArgs(
            mh["c0_gate_src"], mh["probe_c0_arena"], size
        ),
        "DMA_PROBE_XDMA_C0_CROSS_CLUSTER_CONCURRENT",
    )
    cross_xdma_c1 = add_node(
        dfg,
        1,
        DMA_CORE,
        "__snax_bingo_kernel_xdma_1d_copy",
        SnaxBingoKernelXdma1dCopyArgs(
            mh["next_gate_src"], mh["probe_c1_arena"], size
        ),
        "DMA_PROBE_XDMA_C1_CROSS_CLUSTER_CONCURRENT",
    )
    dfg.bingo_add_edge(dual_shifted_phase, cross_xdma_c0)
    dfg.bingo_add_edge(dual_shifted_phase, cross_xdma_c1)

    cross_idma_c0 = add_node(
        dfg,
        0,
        DMA_CORE,
        "__snax_bingo_kernel_idma_1d_copy",
        SnaxBingoKernelIdma1dCopyArgs(
            mh["c0_up_src"], offset(mh["probe_c0_arena"], size), size
        ),
        "DMA_PROBE_IDMA_C0_CROSS_CLUSTER_CONCURRENT",
    )
    cross_idma_c1 = add_node(
        dfg,
        1,
        DMA_CORE,
        "__snax_bingo_kernel_idma_1d_copy",
        SnaxBingoKernelIdma1dCopyArgs(
            mh["next_up_src"], offset(mh["probe_c1_arena"], size), size
        ),
        "DMA_PROBE_IDMA_C1_CROSS_CLUSTER_CONCURRENT",
    )
    for cross_idma in (cross_idma_c0, cross_idma_c1):
        dfg.bingo_add_edge(cross_xdma_c0, cross_idma)
        dfg.bingo_add_edge(cross_xdma_c1, cross_idma)

    return (cross_idma_c0, cross_idma_c1)


def create_dfg(p, mh):
    dfg = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=4,
        num_cores_per_cluster=2,
        is_host_as_acc=True,
        chiplet_ids=[0x00],
    )
    probe_completion = add_dma_probes(dfg, p, mh)
    for prefix, cluster in TEST_CLUSTERS:
        add_slot(dfg, p, mh, prefix, cluster, probe_completion)
    return dfg


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    params = load_params(args)
    memory = define_memory(params)
    dfg = create_dfg(params, memory)
    dfg.bingo_compile_dfg(
        params["app_name"],
        args.output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["multi_cluster_MoE_test_data.h"],
    )


if __name__ == "__main__":
    main()
