#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Dual-VersaCore multi_cluster_MoE datagen (mixed shared/individual layouts)
#
# Individual experts consume dense L3 token rows through the bank-aware APIs.
# Shared experts retain the padded-row L1.5 layout and its fused legacy API.
#
# Hardware: snax_dual_versacore_int16x4_4lane_postproc_v2_cluster
#   Shape family (K=8, N-direction expand):
#     S0: meshRow=8, tileSize=8, meshCol=4
#     S1: meshRow=4, tileSize=8, meshCol=8
#     S2: meshRow=2, tileSize=8, meshCol=16
#   Writer: D0(1ch) + D1(1ch), 4-lane postproc (4 int16/beat = 8 bytes/beat)
#   TCDM: 34 ports (A:16 + B0:8 + B1:8 + D0:1 + D1:1), sparse_interconnect=true
#
# Individual-expert TCDM bank layout:
#   banks 0..15  = A / Mode1 D
#   banks 16..47 = B0/B1 ping/pong
#   banks 48..63 = Mode0 D
# Mode0 and Mode1 weights use disjoint row-depth regions (4 MiB + 2 MiB).
#
# 参数说明（params.hjson → 模型全局变量对应关系）:
#   k0_total = hidden_size                          (shared gate/up 输入 K)
#   n0_total = intermediate_size                    (shared gate/up 输出 N)
#   k1_total = intermediate_size                    (down 输入 K)
#   n1_total = hidden_size                          (down 双 VC 合并输出 N)
#   n1_per_vc = hidden_size / 2                     (每个 writer 输出 N)
#   m_total = total_tokens                          (token 总数)
#
# 修改 params.hjson 中的模型维度会同时调整数据、bank layout 和 DFG 参数。

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_PATH = pathlib.Path(__file__).resolve().parent


def find_repo_root(start):
    for parent in (start, *start.parents):
        if (parent / "Bender.yml").is_file() and (parent / "deps").is_dir():
            return parent
    raise RuntimeError(f"Cannot find HeMAiA repo root from {start}")


REPO_ROOT = find_repo_root(CURRENT_PATH)
SNITCH_CLUSTER_DIR = REPO_ROOT / "deps" / "snitch_cluster"
SNITCH_UTIL_SIM = SNITCH_CLUSTER_DIR / "util" / "sim"
# ── data_utils for C array emission ──────────────────────────────────────────
sys.path.insert(0, str(SNITCH_UTIL_SIM))
from data_utils import (  # noqa: E402
    format_vector_definition,
)
from moe_layout import (  # noqa: E402
    L15_CFG_WORDS,
    L15_TOKEN_ROW_GAP_BYTES,
    SHAPE_DIMS,
    derive_mixed_workload_params,
    token_row_stride,
)
from moe_scheduler_bench_cases import (  # noqa: E402
    SCHEDULER_BENCH_CASES,
    get_scheduler_bench_case,
)

np.random.seed(320)

# Legacy L15 config builders below are retained for comparison experiments.
# (name, array_shape, meshRow, tileSize, meshCol)
def log(msg):
    print(f"[datagen] {msg}", file=sys.stderr, flush=True)


def pack_int4(values_flat):
    arr = np.array(values_flat, dtype=np.int8).flatten()
    if len(arr) % 2 != 0:
        arr = np.append(arr, 0)
    lo = arr[0::2].astype(np.uint8) & 0x0F
    hi = arr[1::2].astype(np.uint8) & 0x0F
    return (lo | (hi << 4)).astype(np.uint8)


def build_l15_shape_cfg_fields(shape, globals_, placement, m_tiles=1):
    """Build the named fields of __snax_bingo_moe_l15_shape_cfg_t."""
    _, array_shape, mesh_row, tile_size, mesh_col = shape
    k0_total = globals_["k0_total"]
    n0_total = globals_["n0_total"]
    k1_total = globals_["k1_total"]
    n1_total = globals_["n1_total"]
    n1_per_vc = globals_["n1_per_vc"]
    k0_s0_tiles = globals_["k0_s0_tiles"]
    k1_s0_tiles = globals_["k1_s0_tiles"]
    a_row_stride = token_row_stride(globals_["k0_bytes"])

    k_tiles = k0_total // tile_size
    k1_tiles = k1_total // tile_size
    n0_tiles = n0_total // mesh_col
    n1_tiles = n1_per_vc // mesh_col

    b_k_stride = 16
    mode0_b_n_stride = (mesh_col // 4) * k0_s0_tiles * 16
    mode1_b_n_stride = (mesh_col // 4) * k1_s0_tiles * 16
    mode0_b_sstride = k0_s0_tiles * 16
    mode1_b_sstride = k1_s0_tiles * 16

    a_channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]
    b_channel_en = {0: 0x03, 1: 0x0F, 2: 0xFF}[array_shape]

    d_stride1 = 64
    mode0_d_m_stride = n0_tiles * d_stride1
    d_bound0 = 8
    beats_per_row = mesh_col // 4
    # Mode-1 D output uses the same per-token row stride as Mode-0 A.
    mode1_token_stride = a_row_stride
    if mode1_token_stride < n1_total * 2:
        raise ValueError("L15 Mode-1 output row stride is smaller than logical output")
    mode1_a_sstride = {0: [64, 8], 1: [8, 16], 2: [8, 32]}[array_shape]
    mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]

    fields = {
        "array_shape": array_shape,
        "meshRow": mesh_row,
        "tileSize": tile_size,
        "meshCol": mesh_col,
        "tokens_used": m_tiles * mesh_row,
        "M_tiles": m_tiles,
        "K_tiles": k_tiles,
        "N_tiles": n0_tiles,
        "K1": k1_tiles,
        "N1": n1_tiles,
        "mode0_A_sstride": [8, a_row_stride],
        "mode1_A_sstride": mode1_a_sstride,
        "mode0_B_sstride": [8, mode0_b_sstride],
        "mode1_B_sstride": [8, mode1_b_sstride],
        "D_sstride": [8],
        "mode0_A_tbound": [k_tiles, n0_tiles, m_tiles, 1, 1, 1],
        "mode0_A_tstride": [tile_size * 2, 0, mesh_row * a_row_stride, 0, 0, 0],
        "mode1_A_tbound": [k1_tiles, n1_tiles, m_tiles, 1, 1, 1],
        "mode1_A_tstride": [mode1_a_k_stride, 0, mode0_d_m_stride, 0, 0, 0],
        "mode0_B_tbound": [k_tiles, n0_tiles, m_tiles, 1],
        "mode0_B_tstride": [b_k_stride, mode0_b_n_stride, 0, 0],
        "mode1_B_tbound": [k1_tiles, n1_tiles, m_tiles, 1],
        "mode1_B_tstride": [b_k_stride, mode1_b_n_stride, 0, 0],
        "mode0_D_tbound": [d_bound0, n0_tiles, m_tiles, 1],
        "mode0_D_tstride": [8, d_stride1, mode0_d_m_stride, 0],
        "mode1_D_tbound": [beats_per_row, mesh_row, n1_tiles, m_tiles],
        "mode1_D_tstride": [
            8,
            mode1_token_stride,
            mesh_col * 2,
            mesh_row * mode1_token_stride,
        ],
        "A_channel_en": [a_channel_en],
        "B_channel_en": [b_channel_en],
        "D_channel_en": [0x01],
        "delta_local_a": placement["delta_local_a"],
        "delta_local_b0": placement["delta_local_b0"],
        "delta_local_b1": placement["delta_local_b1"],
        "delta_local_d0": placement["delta_local_d0"],
        "delta_local_w2l": placement["delta_local_w2l"],
        "delta_local_w2r": placement["delta_local_w2r"],
        "delta_local_mode1_d0": placement["delta_local_mode1_d0"],
        "delta_local_mode1_d1": placement["delta_local_mode1_d1"],
        "tcdm_end": placement["tcdm_end"],
        "mode0_output_elems": m_tiles * mesh_row * n0_total,
        "mode1_output_elems": m_tiles * mesh_row * n1_total,
        "mode1_output_row_stride_bytes": mode1_token_stride,
        "mode1_output_span_elems": m_tiles * mesh_row * (mode1_token_stride // 2),
    }
    word_count = sum(len(value) if isinstance(value, list) else 1 for value in fields.values())
    if word_count != L15_CFG_WORDS:
        raise ValueError(f"L15 config has {word_count} words, expected {L15_CFG_WORDS}")
    return fields


def format_l15_cfg_definition(name, fields):
    lines = [
        f"static const __snax_bingo_moe_l15_shape_cfg_t {name} = {{"
    ]
    for field, value in fields.items():
        if isinstance(value, list):
            body = ", ".join(str(v) for v in value)
            lines.append(f"    .{field} = {{{body}}},")
        else:
            lines.append(f"    .{field} = {value},")
    lines += [
        "};",
        f'_Static_assert(sizeof({name}) == BINGO_MOE_L15_CFG_WORDS * 4u,',
        '               "generated L15 config size mismatch");',
    ]
    return "\n".join(lines)


def emit_moe_data(**kw):
    p = derive_mixed_workload_params(kw)
    ashape = p["array_shape"]
    meshRow = p["meshRow"]
    tileSize = p["tileSize"]
    meshCol = p["meshCol"]
    meshRow_A = p["A_meshRow"]
    tileSize_A = p["A_tileSize"]
    down_vc_meshCol = meshCol

    rM1, rN1, rK1 = p["router_M1"], p["router_N1"], p["router_K1"]
    rM2, rN2, rK2 = p["router_M2"], p["router_N2"], p["router_K2"]
    iN1, iK1 = p["indiv_N1"], p["indiv_K1"]
    iN2, iK2 = p["indiv_N2"], p["indiv_K2"]
    idN1, idK1 = p["indiv_down_N1"], p["indiv_down_K1"]
    idN2, idK2 = p["indiv_down_N2"], p["indiv_down_K2"]
    sN1, sK1 = p["shared_N1"], p["shared_K1"]
    sN2, sK2 = p["shared_N2"], p["shared_K2"]
    sdM1 = p["shared_down_M1"]
    sdN1, sdK1 = p["shared_down_N1"], p["shared_down_K1"]
    sdN2, sdK2 = p["shared_down_N2"], p["shared_down_K2"]
    num_shared = p["num_shared_experts"]
    num_indiv_experts = p["num_indiv_experts"]
    num_indiv_weight_backings = p["num_indiv_weight_backings"]
    M_total = p["M_total"]
    K_total = p["input_hidden"]
    K1_A = K_total // tileSize_A
    N_router = p["router_output_width"]
    N_indiv = p["indiv_hidden"]
    N_shared = p["shared_hidden"]
    N_indiv_down = p["indiv_output_width"]
    N_shared_down = p["shared_output_width"]

    log(f"S{ashape}: meshRow={meshRow} tileSize={tileSize} meshCol={meshCol}")
    log(
        f"M_total={M_total} K_total={K_total} N_router={N_router} "
        f"N_indiv={N_indiv} N_indiv_down={N_indiv_down} "
        f"N_shared_down={N_shared_down}"
    )

    k0_total = K_total
    n0_total = N_shared
    k1_total = N_shared
    n1_total = N_shared_down
    n1_per_vc = n1_total // 2
    k0_bytes = p["A_token_bytes"]
    k0_s0_tiles = k0_total // 8
    k1_s0_tiles = k1_total // 8
    n0_s0_tiles = n0_total // 4
    n1_s0_tiles = n1_per_vc // 4
    globals_ = {
        "m_total": M_total,
        "k0_total": k0_total,
        "n0_total": n0_total,
        "k1_total": k1_total,
        "n1_total": n1_total,
        "n1_per_vc": n1_per_vc,
        "k0_s0_tiles": k0_s0_tiles,
        "k1_s0_tiles": k1_s0_tiles,
        "n0_s0_tiles": n0_s0_tiles,
        "n1_s0_tiles": n1_s0_tiles,
        "k0_bytes": k0_bytes,
    }
    s0_shape = SHAPE_DIMS[0]
    data_str = []
    token_payload_bytes = p["A_token_bytes"]
    token_stride_bytes = p["A_token_row_stride_bytes"]

    # Individual L3 input uses ordinary dense row-major storage. Each gather issues one 2D
    # descriptor per token and maps successive 16-byte K tiles to successive
    # 512-byte TCDM rows at that token's two-bank offset.
    log(f"generating dense input_A: {M_total} x {token_payload_bytes}B")
    A_phys = np.random.randint(
        -256, 255, size=(M_total, K1_A, tileSize_A), dtype=np.int16
    )
    A_token_data = A_phys.reshape(M_total, K1_A * tileSize_A).view(np.uint8)
    A_flat = A_token_data.reshape(-1)
    assert A_flat.size == M_total * token_stride_bytes
    pad = (-len(A_flat)) % 64
    if pad:
        A_flat = np.pad(A_flat, (0, pad), constant_values=0)
    data_str += [
        format_vector_definition("uint8_t", "input_A", A_flat, alignment=64)
    ]

    # The Router and shared L1.5 kernels retain the historical +32-byte row
    # stride. Keep separate copies so the individual ABI remains dense.
    router_A = np.zeros(
        (p["padded_tokens"], p["router_legacy_A_row_stride_bytes"]),
        dtype=np.uint8,
    )
    router_A[:M_total, :token_payload_bytes] = A_token_data
    data_str += [
        format_vector_definition(
            "uint8_t", "router_input_A", router_A.reshape(-1), alignment=64
        )
    ]

    shared_A = np.zeros((M_total, p["l15_a_row_stride"]), dtype=np.uint8)
    shared_A[:, :token_payload_bytes] = A_token_data
    data_str += [
        format_vector_definition(
            "uint8_t", "shared_input_A", shared_A.reshape(-1), alignment=64
        )
    ]

    log("generating router_B (INT4 packed)")
    rB_values = np.random.randint(
        -7, 7, size=(rN2, rK2 * rK1, rN1, tileSize, meshCol), dtype=np.int8
    )
    # All B arrays use the streamer's physical order
    # (expert, N2, N1, K, meshCol, tileSize).
    rB_stream = np.ascontiguousarray(rB_values.transpose(0, 2, 1, 4, 3))
    data_str += [format_vector_definition("uint8_t", "router_B", pack_int4(rB_stream))]

    log(
        "generating indiv_gate_B / up_B / down_B "
        f"({num_indiv_weight_backings} physical backings for "
        f"{num_indiv_experts} logical experts)"
    )
    gB_phys = np.random.randint(
        -7,
        7,
        size=(
            num_indiv_weight_backings,
            iN2,
            iN1,
            iK2 * iK1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    gB_packed = pack_int4(
        np.ascontiguousarray(gB_phys.transpose(0, 1, 2, 3, 5, 4))
    )
    data_str += [
        format_vector_definition(
            "uint8_t",
            "indiv_gate_B",
            gB_packed,
            alignment=128,
        )
    ]
    uB_phys = np.random.randint(
        -7,
        7,
        size=(
            num_indiv_weight_backings,
            iN2,
            iN1,
            iK2 * iK1,
            tileSize,
            meshCol,
        ),
        dtype=np.int8,
    )
    uB_packed = pack_int4(
        np.ascontiguousarray(uB_phys.transpose(0, 1, 2, 3, 5, 4))
    )
    data_str += [
        format_vector_definition(
            "uint8_t",
            "indiv_up_B",
            uB_packed,
            alignment=64,
        )
    ]
    dB_phys = np.random.randint(
        -7,
        7,
        size=(
            num_indiv_weight_backings,
            2,
            idN2,
            idN1,
            idK2 * idK1,
            tileSize,
            down_vc_meshCol,
        ),
        dtype=np.int8,
    )
    dB_packed = pack_int4(
        np.ascontiguousarray(dB_phys.transpose(0, 1, 2, 3, 4, 6, 5))
    )
    data_str += [
        format_vector_definition(
            "uint8_t",
            "indiv_down_B",
            dB_packed,
            alignment=64,
        )
    ]

    if num_shared > 0:
        log(f"generating shared expert weights ({num_shared} experts)")
        sgB = np.random.randint(
            -7,
            7,
            size=(num_shared, sN2, sN1, sK2 * sK1, tileSize, meshCol),
            dtype=np.int8,
        )
        data_str += [
            format_vector_definition(
                "uint8_t",
                "shared_gate_B",
                pack_int4(np.ascontiguousarray(sgB.transpose(0, 1, 2, 3, 5, 4))),
                alignment=128,
            )
        ]
        suB = np.random.randint(
            -7,
            7,
            size=(num_shared, sN2, sN1, sK2 * sK1, tileSize, meshCol),
            dtype=np.int8,
        )
        data_str += [
            format_vector_definition(
                "uint8_t",
                "shared_up_B",
                pack_int4(np.ascontiguousarray(suB.transpose(0, 1, 2, 3, 5, 4))),
                alignment=64,
            )
        ]
        sdB = np.random.randint(
            -7,
            7,
            size=(num_shared, 2, sdN2, sdN1, sdK2 * sdK1, tileSize, down_vc_meshCol),
            dtype=np.int8,
        )
        data_str += [
            format_vector_definition(
                "uint8_t",
                "shared_down_B",
                pack_int4(np.ascontiguousarray(sdB.transpose(0, 1, 2, 3, 4, 6, 5))),
                alignment=64,
            )
        ]

    placement = {
        "delta_local_a": p["l15_delta_local_a"],
        "delta_local_b0": p["l15_delta_local_b0"],
        "delta_local_b1": p["l15_delta_local_b1"],
        "delta_local_d0": p["l15_delta_local_d0"],
        "delta_local_w2l": p["l15_delta_local_w2l"],
        "delta_local_w2r": p["l15_delta_local_w2r"],
        "delta_local_mode1_d0": p["l15_delta_local_mode1_d0"],
        "delta_local_mode1_d1": p["l15_delta_local_mode1_d1"],
        "tcdm_end": p["l15_delta_cfg"],
    }
    cfg_fields = build_l15_shape_cfg_fields(
        s0_shape, globals_, placement, m_tiles=sdM1
    )
    data_str += [
        "// Shared-expert fused SwiGLU + down-projection config.\n"
        + format_l15_cfg_definition("l15_dev_shared_s0_cfg", cfg_fields)
    ]

    tcdm_kb = p["bank_tcdm_size"] / 1024
    log(
        f"individual bank layout: dense L3 stride={token_stride_bytes}B, "
        f"resident end={p['bank_tcdm_size']}B ({tcdm_kb:.1f} KiB); "
        f"shared L1.5 stride={p['l15_a_row_stride']}B"
    )
    return data_str


def emit_header_file(**kw):
    lines = [
        "// Auto-generated by multi_cluster_MoE_datagen.py",
        "// Hardware: snax_dual_versacore_int16x4_4lane_postproc_v2_cluster",
        "// Runtime data only; no golden/check tensors.",
        "// Individual token rows are dense; shared tokens retain L1.5 padding.",
        "// Do NOT edit by hand.",
        "#pragma once",
        "#include <stdint.h>",
        "#include <stddef.h>",
        '#include "libbingo/device_kernel_args.h"',
        "",
    ]
    lines += emit_moe_data(**kw)
    return "\n\n".join(lines)


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate runtime data for the multi_cluster_MoE workload"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        default=pathlib.Path(CURRENT_DIR) / "params.hjson",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        default=pathlib.Path(CURRENT_DIR)
        / "../../../../../../../../deps/snitch_cluster/target/"
        "snitch_cluster/cfg/"
        "snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson",
    )
    parser.add_argument(
        "--scheduler-bench-case",
        choices=tuple(SCHEDULER_BENCH_CASES),
        help="emit the minimal data header for a Prepare/Execute benchmark",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.scheduler_bench_case is not None:
        case = get_scheduler_bench_case(args.scheduler_bench_case)
        log(
            f"scheduler bench {case.name}: tokens={case.input_tokens} "
            f"active_experts={case.active_experts}"
        )
        print(
            "// Auto-generated scheduler-only MoE benchmark data.\n"
            "// Counts are emitted by multi_cluster_MoE_config.h.\n"
            "#pragma once"
        )
        return
    with args.cfg.open() as f:
        pcfg = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hcfg = hjson.loads(f.read())
    merged = {**pcfg, **hcfg}
    print(emit_header_file(**merged))


if __name__ == "__main__":
    main()
