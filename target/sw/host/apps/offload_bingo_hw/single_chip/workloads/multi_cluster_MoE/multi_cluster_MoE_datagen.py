#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Dual-VersaCore multi_cluster_MoE datagen  (L15 layout, k8_8x4_4lane hw)
#
# 这个脚本负责生成两类东西：
#   1. workload 运行时要喂给硬件的静态输入/权重数据
#   2. 基于 L15 weights-first TCDM 布局的 layout_cfg_t 结构体
#
# Hardware: snax_dual_versacore_int16x4_4lane_postproc_v2_cluster
#   Shape family (K=8, N-direction expand):
#     S0: meshRow=8, tileSize=8, meshCol=4
#     S1: meshRow=4, tileSize=8, meshCol=8
#     S2: meshRow=2, tileSize=8, meshCol=16
#   Writer: D0(1ch) + D1(1ch), 4-lane postproc (4 int16/beat = 8 bytes/beat)
#   TCDM: 34 ports (A:16 + B0:8 + B1:8 + D0:1 + D1:1), sparse_interconnect=true
#
# L15 TCDM 布局（weights-first, padded A, bank coloring）:
#   B0(W/gate) → B1(V/up) → W2_left → W2_right → A(token buf) → Mode0_D0 → Mode1_D0/D1
#   a_row_stride = k0_bytes + a_pad = K0_total*2 + 32 bytes
#   b1_color=272, w2l_color=128, m1d0_color=256 (bank-rotation offsets)
#
# 参数说明（params.hjson → L15 全局变量对应关系）:
#   k0_total = indiv_K2 * indiv_K1 * tileSize       (gate/up 输入 K 维)
#   n0_total = indiv_N2 * indiv_N1 * 4              (gate/up 输出 N，基于 S0 meshCol=4)
#   k1_total = n0_total                             (down 输入 K = gate/up 输出，自动匹配)
#   n1_total = indiv_down_N2 * indiv_down_N1 * meshCol_down  (down 输出 N)
#   m_total  = router_M2 * router_M1 * A_meshRow    (token 总数)
#
# 修改 params.hjson 中的 indiv_N1/indiv_N2/indiv_down_N1/indiv_down_N2 即可灵活调整布局。
# S2 shape 在 n0_total < 16 时会自动跳过（输出注释说明）。

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── silu_pkg (bit-true SiLU golden) ──────────────────────────────────────────
_silu_pkg = os.path.realpath(
    os.path.join(
        CURRENT_DIR,
        "../../../../../../../../local_mirrors/snitch_cluster/util/silu_pkg",
    )
)
if os.path.isdir(_silu_pkg):
    sys.path.insert(0, _silu_pkg)
from silu_out16_balanced_golden import silu_out16_balanced_eval_q  # noqa: E402

# ── data_utils for legacy scalar/vector emit ─────────────────────────────────
sys.path.append(os.path.join(CURRENT_DIR, "../../../../../../../../util/sim/"))
from data_utils import (  # noqa: E402
    format_scalar_definition,
    format_vector_definition,
)

np.random.seed(320)

# ── L15 shape family (K=8, N-direction expand) ───────────────────────────────
# (name, array_shape, meshRow, tileSize, meshCol)
SHAPE_DIMS_ALL = [
    ("S0", 0, 8, 8, 4),
    ("S1", 1, 4, 8, 8),
    ("S2", 2, 2, 8, 16),
]

# A streamer spatial layout: 2 spatial dims of size [2, 8] → 16 channels
A_SPATIAL_BOUNDS = [2, 8]

# L15 layout descriptor
L15_LAYOUT = {
    "id": 15,
    "name": "l15_weights_first_padded_1024_per_token",
    "a_pad": 32,
    "b1_color": 272,
    "w2l_color": 128,
    "m1d0_color": 256,
}


def log(msg):
    print(f"[datagen] {msg}", file=sys.stderr, flush=True)


def parse_header_config(head_path):
    cfg = {}
    pattern = re.compile(r"#define\s+(\w+)\s+([-+]?[\d\.]+)")
    with open(head_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    cfg[m.group(1)] = eval(m.group(2).strip())
                except Exception:
                    pass
    return cfg


def c_array(values):
    return "{ " + ", ".join(str(int(v)) for v in values) + " }"


def c_u8_array(values):
    return "{ " + ", ".join(f"0x{int(v) & 0xff:02x}" for v in values) + " }"


def emit_i16_array(name, values):
    return f"static const int16_t {name}[{len(values)}] = {c_array(values)};"


def emit_u8_array_lit(name, values):
    return f"static const uint8_t {name}[{len(values)}] = {c_u8_array(values)};"


def find_top_k(scores_2D, k):
    return np.argsort(scores_2D, axis=1)[:, ::-1][:, :k].astype(np.uint16)


def pack_int4(values_flat):
    arr = np.array(values_flat, dtype=np.int8).flatten()
    if len(arr) % 2 != 0:
        arr = np.append(arr, 0)
    lo = arr[0::2].astype(np.uint8) & 0x0F
    hi = arr[1::2].astype(np.uint8) & 0x0F
    return (lo | (hi << 4)).astype(np.uint8)


def packed_int4_constant(num_s0_tiles, value):
    assert 0 <= value <= 7
    packed_byte = (value << 4) | value
    return np.full(num_s0_tiles * 16, packed_byte, dtype=np.uint8)


def rescale_down_32to16(arr_int32, input_zp=0, mult=1, output_zp=0, shift=0):
    result = arr_int32.astype(np.int64) - int(input_zp)
    multiplied = result * np.int64(mult)
    if shift > 0:
        shifted_one = np.int64(1) << (shift - 1)
        shifted_data = multiplied + shifted_one
        scaled_32 = np.where(
            result >= 0,
            shifted_data + np.int64(1 << 30),
            shifted_data - np.int64(1 << 30),
        )
        correct_shift = np.where(shift > 31, scaled_32, shifted_data)
        shifted_value = correct_shift >> shift
    else:
        shifted_value = multiplied
    out = shifted_value.astype(np.int32).astype(np.int64) + int(output_zp)
    return np.clip(out, -32768, 32767).astype(np.int16)


def apply_silu_vectorized(arr_int16):
    flat = arr_int16.flatten()
    result = np.array(
        [silu_out16_balanced_eval_q(int(x)) for x in flat], dtype=np.int16
    )
    return result.reshape(arr_int16.shape)


def block_gemm_int16x4(M, K, N, meshRow, tileSize, meshCol, A_flat, B_flat):
    a = A_flat.astype(np.int32).reshape(M, K, meshRow, tileSize)
    b = B_flat.astype(np.int32).reshape(N, K, meshCol, tileSize)
    d = np.zeros((M, N, meshRow, meshCol), dtype=np.int32)
    for mm in range(M):
        for nn in range(N):
            d[mm, nn] = np.tensordot(a[mm], b[nn], axes=([0, 2], [0, 2]))
    return d.reshape(-1)


def make_logical_a(m_total, k_total):
    data = np.zeros((m_total, k_total), dtype=np.int16)
    for m in range(m_total):
        for k in range(k_total):
            data[m, k] = ((m * 5 + k * 3) % 11) - 5
    return data


def make_padded_a(logical_a, row_stride_bytes):
    row_elems = row_stride_bytes // 2
    out = np.zeros((logical_a.shape[0], row_elems), dtype=np.int16)
    out[:, : logical_a.shape[1]] = logical_a
    return out.reshape(-1)


def spatial_offsets(bounds, strides):
    out = []
    for i in range(np.prod(bounds)):
        rem = i
        off = 0
        for bound, stride in zip(bounds, strides):
            off += (rem % bound) * stride
            rem //= bound
        out.append(off)
    return out


def streamer_i16_flat(
    source_i16,
    m_tiles,
    k_bound,
    spatial_bounds,
    spatial_strides,
    k_stride,
    m_stride,
    channel_en,
):
    flat = source_i16.reshape(-1)
    offsets = [
        off
        for i, off in enumerate(spatial_offsets(spatial_bounds, spatial_strides))
        if (channel_en >> i) & 1
    ]
    out = []
    for mt in range(m_tiles):
        for kt in range(k_bound):
            base = mt * m_stride + kt * k_stride
            for off in offsets:
                byte_addr = base + off
                assert byte_addr % 2 == 0
                elem = byte_addr // 2
                assert 0 <= elem and elem + 4 <= len(flat), (elem, len(flat))
                out.extend(flat[elem : elem + 4])
    return np.array(out, dtype=np.int16)


def align_up(value, alignment):
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def colored_offset(offset, color_bytes=0, alignment=1024):
    return align_up(offset, alignment) + int(color_bytes)


def place_tensors(globals_, layout):
    a_row_stride = globals_["k0_bytes"] + layout["a_pad"]
    a_bytes = globals_["m_total"] * a_row_stride
    w_bytes = globals_["k0_s0_tiles"] * globals_["n0_s0_tiles"] * 16
    mode0_d_bytes = globals_["m_total"] * globals_["n0_total"] * 2
    w2_bytes = globals_["k1_s0_tiles"] * globals_["n1_s0_tiles"] * 16
    # Mode-1 D output: dual-VC, each VC writes n1_total INT16 elements per row.
    # Packed row stride = 2 * n1_total * 2 bytes (VC0 + VC1 side-by-side, no gap).
    mode1_row_stride = globals_["n1_total"] * 4  # = 2 VCs × n1_total elems × 2 B/elem
    mode1_padded_d_bytes = globals_["m_total"] * mode1_row_stride

    delta_local_b0 = colored_offset(0, layout.get("b0_color", 0))
    delta_local_b1 = colored_offset(delta_local_b0 + w_bytes, layout.get("b1_color", 0))
    delta_local_w2l = colored_offset(
        delta_local_b1 + w_bytes, layout.get("w2l_color", 0)
    )
    delta_local_w2r = colored_offset(
        delta_local_w2l + w2_bytes, layout.get("w2r_color", 0)
    )
    delta_local_a = colored_offset(delta_local_w2r + w2_bytes, layout.get("a_color", 0))
    delta_local_d0 = colored_offset(delta_local_a + a_bytes, layout.get("d0_color", 0))
    delta_local_mode1_d0 = colored_offset(
        delta_local_d0 + mode0_d_bytes, layout.get("m1d0_color", 0)
    )
    delta_local_mode1_d1 = delta_local_mode1_d0 + globals_["n1_total"] * 2

    return {
        "delta_local_a": delta_local_a,
        "delta_local_b0": delta_local_b0,
        "delta_local_b1": delta_local_b1,
        "delta_local_d0": delta_local_d0,
        "delta_local_w2l": delta_local_w2l,
        "delta_local_w2r": delta_local_w2r,
        "delta_local_mode1_d0": delta_local_mode1_d0,
        "delta_local_mode1_d1": delta_local_mode1_d1,
        "tcdm_end": delta_local_mode1_d0 + mode1_padded_d_bytes,
    }


def build_shape_cfg(shape, globals_, golden_names, layout, placement, m_tiles=1):
    name, array_shape, mesh_row, tile_size, mesh_col = shape
    k0_total = globals_["k0_total"]
    n0_total = globals_["n0_total"]
    k1_total = globals_["k1_total"]
    n1_total = globals_["n1_total"]
    k0_s0_tiles = globals_["k0_s0_tiles"]
    k1_s0_tiles = globals_["k1_s0_tiles"]
    a_row_stride = globals_["k0_bytes"] + layout["a_pad"]

    k_tiles = k0_total // tile_size
    k1_tiles = k1_total // tile_size
    n0_tiles = n0_total // mesh_col
    n1_tiles = n1_total // mesh_col

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
    # Packed Mode-1 D output row stride: 2 VCs × n1_total × 2 B/elem.
    mode1_token_stride = 2 * n1_total * 2
    mode1_a_sstride = {0: [64, 8], 1: [8, 16], 2: [8, 32]}[array_shape]
    mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]

    fields = [
        f".array_shape               = {array_shape}",
        f".meshRow                   = {mesh_row}",
        f".tileSize                  = {tile_size}",
        f".meshCol                   = {mesh_col}",
        f".tokens_used               = {m_tiles * mesh_row}",
        f".M_tiles                   = {m_tiles}",
        f".K_tiles                   = {k_tiles}",
        f".N_tiles                   = {n0_tiles}",
        f".K1                        = {k1_tiles}",
        f".N1                        = {n1_tiles}",
        f".mode0_A_sstride           = {c_array([8, a_row_stride])}",
        f".mode1_A_sstride           = {c_array(mode1_a_sstride)}",
        f".mode0_B_sstride           = {c_array([8, mode0_b_sstride])}",
        f".mode1_B_sstride           = {c_array([8, mode1_b_sstride])}",
        f".D_sstride                 = {c_array([8])}",
        f".mode0_A_tbound            = {c_array([k_tiles, n0_tiles, m_tiles, 1, 1, 1])}",
        f".mode0_A_tstride           = {c_array([tile_size * 2, 0, mesh_row * a_row_stride, 0, 0, 0])}",
        f".mode1_A_tbound            = {c_array([k1_tiles, n1_tiles, m_tiles, 1, 1, 1])}",
        f".mode1_A_tstride           = {c_array([mode1_a_k_stride, 0, mode0_d_m_stride, 0, 0, 0])}",
        f".mode0_B_tbound            = {c_array([k_tiles, n0_tiles, m_tiles, 1])}",
        f".mode0_B_tstride           = {c_array([b_k_stride, mode0_b_n_stride, 0, 0])}",
        f".mode1_B_tbound            = {c_array([k1_tiles, n1_tiles, m_tiles, 1])}",
        f".mode1_B_tstride           = {c_array([b_k_stride, mode1_b_n_stride, 0, 0])}",
        f".mode0_D_tbound            = {c_array([d_bound0, n0_tiles, m_tiles, 1])}",
        f".mode0_D_tstride           = {c_array([8, d_stride1, mode0_d_m_stride, 0])}",
        f".mode1_D_tbound            = {c_array([beats_per_row, mesh_row, n1_tiles, m_tiles])}",
        f".mode1_D_tstride           = {c_array([8, mode1_token_stride, mesh_col * 2, mesh_row * mode1_token_stride])}",
        f".A_channel_en              = {c_array([a_channel_en])}",
        f".B_channel_en              = {c_array([b_channel_en])}",
        f".D_channel_en              = {c_array([0x01])}",
        f".delta_local_a             = {placement['delta_local_a']}",
        f".delta_local_b0            = {placement['delta_local_b0']}",
        f".delta_local_b1            = {placement['delta_local_b1']}",
        f".delta_local_d0            = {placement['delta_local_d0']}",
        f".delta_local_w2l           = {placement['delta_local_w2l']}",
        f".delta_local_w2r           = {placement['delta_local_w2r']}",
        f".delta_local_mode1_d0      = {placement['delta_local_mode1_d0']}",
        f".delta_local_mode1_d1      = {placement['delta_local_mode1_d1']}",
        f".tcdm_end                  = {placement['tcdm_end']}",
        f".mode0_output_elems        = {m_tiles * mesh_row * n0_total}",
        f".mode1_output_elems        = {m_tiles * mesh_row * n1_total}",
        f".mode1_output_row_stride_bytes = {mode1_token_stride}",
        f".mode1_padded_output_elems = {m_tiles * mesh_row * 2 * n1_total}",
        f".mode0_d0_golden           = {golden_names[name][0]}",
        f".mode1_padded_golden       = {golden_names[name][1]}",
    ]
    return "        {\n            " + ",\n            ".join(fields) + "\n        }"


def build_shape_cfg_flat_vals(shape, globals_, layout, placement, m_tiles=1):
    """Return flat list of int32 values matching moe_l15_shape_cfg_t field order (91 values).

    The order must exactly match the integer prefix of shape_cfg_t / moe_l15_shape_cfg_t:
      uint32_t array_shape, meshRow, tileSize, meshCol, tokens_used;
      uint32_t M_tiles, K_tiles, N_tiles, K1, N1;
      int32_t mode0_A_sstride[2], mode1_A_sstride[2];
      int32_t mode0_B_sstride[2], mode1_B_sstride[2], D_sstride[1];
      int32_t mode0_A_tbound[6], mode0_A_tstride[6];
      int32_t mode1_A_tbound[6], mode1_A_tstride[6];
      int32_t mode0_B_tbound[4], mode0_B_tstride[4];
      int32_t mode1_B_tbound[4], mode1_B_tstride[4];
      int32_t mode0_D_tbound[4], mode0_D_tstride[4];
      int32_t mode1_D_tbound[4], mode1_D_tstride[4];
      int32_t A_channel_en[1], B_channel_en[1], D_channel_en[1];
      int32_t delta_local_a, delta_local_b0, delta_local_b1, delta_local_d0;
      int32_t delta_local_w2l, delta_local_w2r;
      int32_t delta_local_mode1_d0, delta_local_mode1_d1;
      int32_t tcdm_end, mode0_output_elems, mode1_output_elems;
      int32_t mode1_output_row_stride_bytes, mode1_padded_output_elems;
    """
    name, array_shape, mesh_row, tile_size, mesh_col = shape
    k0_total = globals_["k0_total"]
    n0_total = globals_["n0_total"]
    k1_total = globals_["k1_total"]
    n1_total = globals_["n1_total"]
    k0_s0_tiles = globals_["k0_s0_tiles"]
    k1_s0_tiles = globals_["k1_s0_tiles"]
    a_row_stride = globals_["k0_bytes"] + layout["a_pad"]

    k_tiles = k0_total // tile_size
    k1_tiles = k1_total // tile_size
    n0_tiles = n0_total // mesh_col
    n1_tiles = n1_total // mesh_col

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
    # Packed Mode-1 D output row stride: 2 VCs × n1_total × 2 B/elem.
    mode1_token_stride = 2 * n1_total * 2
    mode1_a_sstride = {0: [64, 8], 1: [8, 16], 2: [8, 32]}[array_shape]
    mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]

    vals = [
        # uint32_t array_shape, meshRow, tileSize, meshCol, tokens_used
        array_shape,
        mesh_row,
        tile_size,
        mesh_col,
        m_tiles * mesh_row,  # tokens_used = total tokens processed per kernel call
        # uint32_t M_tiles, K_tiles, N_tiles, K1, N1
        m_tiles,
        k_tiles,
        n0_tiles,
        k1_tiles,
        n1_tiles,
        # int32_t mode0_A_sstride[2]
        8,
        a_row_stride,
        # int32_t mode1_A_sstride[2]
        mode1_a_sstride[0],
        mode1_a_sstride[1],
        # int32_t mode0_B_sstride[2]
        8,
        mode0_b_sstride,
        # int32_t mode1_B_sstride[2]
        8,
        mode1_b_sstride,
        # int32_t D_sstride[1]
        8,
        # int32_t mode0_A_tbound[6]
        k_tiles,
        n0_tiles,
        m_tiles,
        1,
        1,
        1,
        # int32_t mode0_A_tstride[6]
        tile_size * 2,
        0,
        mesh_row * a_row_stride,
        0,
        0,
        0,
        # int32_t mode1_A_tbound[6]
        k1_tiles,
        n1_tiles,
        m_tiles,
        1,
        1,
        1,
        # int32_t mode1_A_tstride[6]
        mode1_a_k_stride,
        0,
        mode0_d_m_stride,
        0,
        0,
        0,
        # int32_t mode0_B_tbound[4]
        k_tiles,
        n0_tiles,
        m_tiles,
        1,
        # int32_t mode0_B_tstride[4]
        b_k_stride,
        mode0_b_n_stride,
        0,
        0,
        # int32_t mode1_B_tbound[4]
        k1_tiles,
        n1_tiles,
        m_tiles,
        1,
        # int32_t mode1_B_tstride[4]
        b_k_stride,
        mode1_b_n_stride,
        0,
        0,
        # int32_t mode0_D_tbound[4]
        d_bound0,
        n0_tiles,
        m_tiles,
        1,
        # int32_t mode0_D_tstride[4]
        8,
        d_stride1,
        mode0_d_m_stride,
        0,
        # int32_t mode1_D_tbound[4]
        beats_per_row,
        mesh_row,
        n1_tiles,
        m_tiles,
        # int32_t mode1_D_tstride[4]
        8,
        mode1_token_stride,
        mesh_col * 2,
        mesh_row
        * mode1_token_stride,  # per-M-tile stride: advance output by meshRow tokens
        # int32_t A_channel_en[1], B_channel_en[1], D_channel_en[1]
        a_channel_en,
        b_channel_en,
        0x01,
        # int32_t delta_local_a/b0/b1/d0/w2l/w2r/mode1_d0/mode1_d1
        placement["delta_local_a"],
        placement["delta_local_b0"],
        placement["delta_local_b1"],
        placement["delta_local_d0"],
        placement["delta_local_w2l"],
        placement["delta_local_w2r"],
        placement["delta_local_mode1_d0"],
        placement["delta_local_mode1_d1"],
        # int32_t tcdm_end, mode0_output_elems, mode1_output_elems
        placement["tcdm_end"],
        m_tiles * mesh_row * n0_total,
        m_tiles * mesh_row * n1_total,
        # int32_t mode1_output_row_stride_bytes, mode1_padded_output_elems
        mode1_token_stride,
        m_tiles * mesh_row * 2 * n1_total,
    ]
    assert len(vals) == 91, f"build_shape_cfg_flat_vals: expected 91, got {len(vals)}"
    return vals


def build_golden_arrays(logical_a, globals_, active_shapes):
    k0_total = globals_["k0_total"]
    n0_total = globals_["n0_total"]
    k1_total = globals_["k1_total"]
    n1_total = globals_["n1_total"]
    k0_bytes = globals_["k0_bytes"]

    arrays = {}
    names_by_shape = {}

    for shape in active_shapes:
        name, array_shape, mesh_row, tile_size, mesh_col = shape
        m_tiles = 1
        k_tiles = k0_total // tile_size
        k1_tiles = k1_total // tile_size
        n0_tiles = n0_total // mesh_col
        n1_tiles = n1_total // mesh_col
        a_channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]

        a_flat = streamer_i16_flat(
            logical_a,
            m_tiles,
            k_tiles,
            A_SPATIAL_BOUNDS,
            [8, k0_bytes],
            tile_size * 2,
            mesh_row * k0_bytes,
            a_channel_en,
        )

        b0_flat = np.full(n0_tiles * k_tiles * mesh_col * tile_size, 1, dtype=np.int8)
        b1_flat = np.full(n0_tiles * k_tiles * mesh_col * tile_size, 2, dtype=np.int8)

        vc0 = block_gemm_int16x4(
            m_tiles, k_tiles, n0_tiles, mesh_row, tile_size, mesh_col, a_flat, b0_flat
        )
        vc1 = block_gemm_int16x4(
            m_tiles, k_tiles, n0_tiles, mesh_row, tile_size, mesh_col, a_flat, b1_flat
        )
        vc0_i16 = rescale_down_32to16(vc0)
        vc0_silu = apply_silu_vectorized(vc0_i16)
        vc1_i16 = rescale_down_32to16(vc1)
        mode0 = rescale_down_32to16(
            vc0_silu.astype(np.int32) * vc1_i16.astype(np.int32)
        )

        d_stride1 = 64
        mode0_d_m_stride = n0_tiles * d_stride1
        mode1_a_sstride = {0: [64, 8], 1: [8, 16], 2: [8, 32]}[array_shape]
        mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]
        mode1_a_flat = streamer_i16_flat(
            mode0,
            m_tiles,
            k1_tiles,
            A_SPATIAL_BOUNDS,
            mode1_a_sstride,
            mode1_a_k_stride,
            mode0_d_m_stride,
            a_channel_en,
        )

        w2_left_flat = np.full(
            n1_tiles * k1_tiles * mesh_col * tile_size, 1, dtype=np.int8
        )
        w2_right_flat = np.full(
            n1_tiles * k1_tiles * mesh_col * tile_size, 2, dtype=np.int8
        )

        mode1_d0 = rescale_down_32to16(
            block_gemm_int16x4(
                m_tiles,
                k1_tiles,
                n1_tiles,
                mesh_row,
                tile_size,
                mesh_col,
                mode1_a_flat,
                w2_left_flat,
            )
        )
        mode1_d1 = rescale_down_32to16(
            block_gemm_int16x4(
                m_tiles,
                k1_tiles,
                n1_tiles,
                mesh_row,
                tile_size,
                mesh_col,
                mode1_a_flat,
                w2_right_flat,
            )
        )

        gname_mode0 = f"{name}_mode0_d0_golden"
        gname_mode1 = f"{name}_mode1_padded_golden"
        names_by_shape[name] = (gname_mode0, gname_mode1)
        arrays[gname_mode0] = emit_i16_array(gname_mode0, mode0)

        a_row_stride = globals_["k0_bytes"] + L15_LAYOUT["a_pad"]
        mode1_d0_pt = (
            mode1_d0.reshape(m_tiles, n1_tiles, mesh_row, mesh_col)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        mode1_d1_pt = (
            mode1_d1.reshape(m_tiles, n1_tiles, mesh_row, mesh_col)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        mode1_combined = np.concatenate(
            [
                mode1_d0_pt.reshape(mesh_row, n1_total),
                mode1_d1_pt.reshape(mesh_row, n1_total),
            ],
            axis=1,
        ).reshape(-1)
        row_elems = max(
            a_row_stride // 2, n1_total * 2
        )  # ensure buffer fits mode1 dual-VC output
        mode1_padded = np.zeros((mesh_row, row_elems), dtype=np.int16)
        mode1_padded[:, : n1_total * 2] = mode1_combined.reshape(mesh_row, n1_total * 2)
        arrays[gname_mode1] = emit_i16_array(gname_mode1, mode1_padded.reshape(-1))

    return arrays, names_by_shape


SHAPE_CFG_TYPEDEF = """\
typedef struct {
    uint32_t array_shape, meshRow, tileSize, meshCol, tokens_used;
    uint32_t M_tiles, K_tiles, N_tiles, K1, N1;
    int32_t mode0_A_sstride[2], mode1_A_sstride[2];
    int32_t mode0_B_sstride[2], mode1_B_sstride[2], D_sstride[1];
    int32_t mode0_A_tbound[6], mode0_A_tstride[6];
    int32_t mode1_A_tbound[6], mode1_A_tstride[6];
    int32_t mode0_B_tbound[4], mode0_B_tstride[4];
    int32_t mode1_B_tbound[4], mode1_B_tstride[4];
    int32_t mode0_D_tbound[4], mode0_D_tstride[4];
    int32_t mode1_D_tbound[4], mode1_D_tstride[4];
    int32_t A_channel_en[1], B_channel_en[1], D_channel_en[1];
    int32_t delta_local_a, delta_local_b0, delta_local_b1, delta_local_d0;
    int32_t delta_local_w2l, delta_local_w2r;
    int32_t delta_local_mode1_d0, delta_local_mode1_d1;
    int32_t tcdm_end, mode0_output_elems, mode1_output_elems;
    int32_t mode1_output_row_stride_bytes, mode1_padded_output_elems;
    const int16_t *mode0_d0_golden, *mode1_padded_golden;
} shape_cfg_t;"""

LAYOUT_CFG_TYPEDEF = """\
typedef struct {
    uint32_t layout_id;
    const char *name;
    int32_t a_row_stride;
    const int16_t *a_data;
    int32_t a_data_length;
    const uint8_t *w_data, *v_data, *w2_left_data, *w2_right_data;
    int32_t b_data_length, w2_data_length;
    shape_cfg_t shapes[NUM_LAYOUT_SHAPES];
} layout_cfg_t;"""


def emit_moe_data(**kw):
    head = os.path.join(CURRENT_DIR, "MoE_common_variable.h")
    cfg = parse_header_config(head)

    core_tmpl = kw.get("snax_versacore_core_template") or kw.get(
        "snax_dual_versacore_int16x4_core_template"
    )
    if core_tmpl is None:
        raise KeyError("No versacore core template found in hwcfg")

    data_type = 0
    acc = core_tmpl["snax_acc_cfg"][0]
    ashape = kw["array_shape"]
    meshRow = acc["snax_versacore_spatial_unrolling"][data_type][ashape][0]
    tileSize = acc["snax_versacore_spatial_unrolling"][data_type][ashape][1]
    meshCol = acc["snax_versacore_spatial_unrolling"][data_type][ashape][2]

    meshRow_A = int(kw.get("A_meshRow", meshRow))
    tileSize_A = int(kw.get("A_tileSize", tileSize))
    meshCol_down = int(kw.get("down_meshCol", meshCol))
    down_vc_meshCol = meshCol
    if meshCol_down != 2 * down_vc_meshCol:
        raise ValueError("dual-VC down projection expects down_meshCol == 2 * meshCol")

    rM1, rN1, rK1 = kw["router_M1"], kw["router_N1"], kw["router_K1"]
    rM2, rN2, rK2 = kw["router_M2"], kw["router_N2"], kw["router_K2"]
    iM1, iN1, iK1 = kw["indiv_M1"], kw["indiv_N1"], kw["indiv_K1"]
    iM2, iN2, iK2 = kw["indiv_M2"], kw["indiv_N2"], kw["indiv_K2"]
    idM1 = kw.get("indiv_down_M1", iM1)
    idN1 = kw.get("indiv_down_N1", iN1)
    idK1 = kw.get("indiv_down_K1", 4)
    idM2 = kw.get("indiv_down_M2", iM2)
    idN2 = kw.get("indiv_down_N2", iN2)
    idK2 = kw.get("indiv_down_K2", 1)
    sM1 = kw.get("shared_M1", iM1)
    sN1 = kw.get("shared_N1", iN1)
    sK1 = kw.get("shared_K1", iK1)
    sM2 = kw.get("shared_M2", iM2)
    sN2 = kw.get("shared_N2", iN2)
    sK2 = kw.get("shared_K2", iK2)
    sdM1 = kw.get("shared_down_M1", sM1)
    sdN1 = kw.get("shared_down_N1", sN1)
    sdK1 = kw.get("shared_down_K1", idK1)
    sdM2 = kw.get("shared_down_M2", sM2)
    sdN2 = kw.get("shared_down_N2", sN2)
    sdK2 = kw.get("shared_down_K2", idK2)
    num_shared = int(cfg.get("shared_expert_number_k", 0))

    num_indiv_experts = int(cfg["expert_number_each_layer"])
    top_k = int(cfg["individual_expert_number_k"])
    M_total = rM2 * rM1 * meshRow_A
    K_total = rK2 * rK1 * tileSize
    assert K_total % tileSize_A == 0
    K1_A = K_total // tileSize_A
    N_router = rN2 * rN1 * meshCol
    N_indiv = iN2 * iN1 * meshCol
    N_shared = sN2 * sN1 * meshCol
    N_indiv_down = idN2 * idN1 * meshCol_down
    K_indiv_down = idK2 * idK1 * tileSize
    K_shared_down = sdK2 * sdK1 * tileSize

    assert (
        N_router == num_indiv_experts
    ), f"router N={N_router} != num_indiv_experts={num_indiv_experts}"
    assert (
        K_indiv_down == N_indiv
    ), f"individual down K={K_indiv_down} != gate/up hidden N={N_indiv}"
    assert (
        K_shared_down == N_shared
    ), f"shared down K={K_shared_down} != gate/up hidden N={N_shared}"

    log(f"S{ashape}: meshRow={meshRow} tileSize={tileSize} meshCol={meshCol}")
    log(
        f"M_total={M_total} K_total={K_total} N_router={N_router} "
        f"N_indiv={N_indiv} N_indiv_down={N_indiv_down}"
    )

    # L15 globals (per-cluster per-expert dimensions)
    k0_total = K_total
    s0_meshCol = 4
    n0_total = iN2 * iN1 * s0_meshCol
    k1_total = n0_total
    n1_total = N_indiv_down
    k0_bytes = k0_total * 2
    k0_s0_tiles = k0_total // 8
    k1_s0_tiles = k1_total // 8
    n0_s0_tiles = n0_total // 4
    n1_s0_tiles = n1_total // 4

    assert (
        k1_total == K_indiv_down
    ), f"L15 k1_total={k1_total} != K_indiv_down={K_indiv_down}"

    globals_ = {
        "m_total": M_total,
        "k0_total": k0_total,
        "n0_total": n0_total,
        "k1_total": k1_total,
        "n1_total": n1_total,
        "k0_s0_tiles": k0_s0_tiles,
        "k1_s0_tiles": k1_s0_tiles,
        "n0_s0_tiles": n0_s0_tiles,
        "n1_s0_tiles": n1_s0_tiles,
        "k0_bytes": k0_bytes,
    }

    active_shapes = [
        s
        for s in SHAPE_DIMS_ALL
        if n0_total % s[4] == 0
        and n1_total % s[4] == 0
        and n0_total // s[4] >= 1
        and n1_total // s[4] >= 1
    ]
    skipped = [s[0] for s in SHAPE_DIMS_ALL if s not in active_shapes]
    if skipped:
        log(
            f"NOTE: shapes {skipped} skipped (n0={n0_total} or n1={n1_total} "
            f"too small for those meshCols). Increase indiv_N1/N2 or "
            f"indiv_down_N1/N2 in params.hjson to enable them."
        )
    log(
        f"L15: k0={k0_total} n0={n0_total} k1={k1_total} n1={n1_total} "
        f"m={M_total} active={[s[0] for s in active_shapes]}"
    )

    data_str = []

    # Scalars (used by legacy dispatch in main_bingo)
    for nm, val in [
        ("array_shape", ashape),
        ("meshRow", meshRow),
        ("tileSize", tileSize),
        ("meshCol", meshCol),
        ("meshRow_A", meshRow_A),
        ("tileSize_A", tileSize_A),
        ("K1_A", K1_A),
        ("meshCol_down", meshCol_down),
        ("down_vc_meshCol", down_vc_meshCol),
        ("router_M1", rM1),
        ("router_N1", rN1),
        ("router_K1", rK1),
        ("router_M2", rM2),
        ("router_N2", rN2),
        ("router_K2", rK2),
        ("indiv_M1", iM1),
        ("indiv_N1", iN1),
        ("indiv_K1", iK1),
        ("indiv_M2", iM2),
        ("indiv_N2", iN2),
        ("indiv_K2", iK2),
        ("indiv_down_M1", idM1),
        ("indiv_down_N1", idN1),
        ("indiv_down_K1", idK1),
        ("indiv_down_M2", idM2),
        ("indiv_down_N2", idN2),
        ("indiv_down_K2", idK2),
        ("shared_M1", sM1),
        ("shared_N1", sN1),
        ("shared_K1", sK1),
        ("shared_M2", sM2),
        ("shared_N2", sN2),
        ("shared_K2", sK2),
        ("shared_down_M1", sdM1),
        ("shared_down_N1", sdN1),
        ("shared_down_K1", sdK1),
        ("shared_down_M2", sdM2),
        ("shared_down_N2", sdN2),
        ("shared_down_K2", sdK2),
        ("num_shared_experts", num_shared),
        ("M_total", M_total),
        ("K_total_input", K_total),
        ("N_router", N_router),
        ("N_indiv", N_indiv),
        ("N_indiv_down", N_indiv_down),
        ("K_indiv_down", K_indiv_down),
        ("num_indiv_experts", num_indiv_experts),
        ("top_k", top_k),
        ("addNonZeroC", kw["addNonZeroC"]),
        ("addZeroC", kw["addZeroC"]),
        ("accumPrevC", kw["accumPrevC"]),
    ]:
        data_str += [format_scalar_definition("uint32_t", nm, val)]

    # input_A (tiled layout with per-token L3 padding for 1 DMA/token gather)
    log("generating input_A (INT16, tiled, with 32B/token L3 padding)")
    A_phys = np.random.randint(
        -256, 255, size=(rM2, rM1, meshRow_A, K1_A, tileSize_A), dtype=np.int16
    )
    M_total_tokens = rM2 * rM1 * meshRow_A
    # 每 token 数据 = K1_A × tileSize_A int16 = 2048 bytes，后跟 32 字节零 padding
    # 使 gather 的 L3 stride = A_token_bytes + 32 = 2080，每 token 只需 1 次 DMA
    A_token_data = A_phys.reshape(M_total_tokens, K1_A * tileSize_A).view(
        np.uint8
    )  # (M_total, 2048)
    A_token_pad = np.zeros((M_total_tokens, 32), dtype=np.uint8)
    A_flat = np.concatenate([A_token_data, A_token_pad], axis=1).reshape(
        -1
    )  # (M_total × 2080,)
    pad = (-len(A_flat)) % 64
    if pad:
        A_flat = np.pad(A_flat, (0, pad), constant_values=0)
    data_str += [format_vector_definition("uint8_t", "input_A", A_flat)]
    A_tiled_phys = np.ascontiguousarray(A_phys.transpose(0, 1, 3, 2, 4))
    A_tiled_flat = A_tiled_phys.view(np.uint8).reshape(-1)
    pad = (-len(A_tiled_flat)) % 64
    if pad:
        A_tiled_flat = np.pad(A_tiled_flat, (0, pad), constant_values=0)
    data_str += [format_vector_definition("uint8_t", "input_A_tiled", A_tiled_flat)]

    # router_B
    log("generating router_B (INT4 packed)")
    rB_values = np.random.randint(
        -7, 7, size=(rN2, rK2 * rK1, rN1, tileSize, meshCol), dtype=np.int8
    )
    data_str += [format_vector_definition("uint8_t", "router_B", pack_int4(rB_values))]

    # TopK golden
    log("computing TopK")
    A_mat = A_phys.reshape(M_total, K_total).astype(np.int32)
    rB_mat = (
        rB_values.transpose(0, 2, 4, 1, 3).reshape(N_router, K_total).T.astype(np.int32)
    )
    scores = A_mat @ rB_mat
    topk_idx = find_top_k(scores, top_k)
    exp_counts = np.zeros(num_indiv_experts, dtype=np.uint32)
    for t in range(M_total):
        for s in range(top_k):
            exp_counts[int(topk_idx[t, s])] += 1
    log(f"expert token counts: {exp_counts.tolist()}")
    data_str += [
        format_vector_definition(
            "uint16_t", "golden_topk_indices", topk_idx.reshape(-1)
        )
    ]
    data_str += [
        format_vector_definition("uint32_t", "golden_expert_token_counts", exp_counts)
    ]

    # indiv_gate_B / up_B / down_B (random, legacy dispatch)
    log("generating indiv_gate_B / up_B / down_B")
    gB_phys = np.random.randint(
        -7,
        7,
        size=(num_indiv_experts, iN2, iN1, iK2 * iK1, tileSize, meshCol),
        dtype=np.int8,
    )
    data_str += [
        format_vector_definition(
            "uint8_t", "indiv_gate_B", pack_int4(gB_phys), alignment=128
        )
    ]
    data_str += [
        format_vector_definition(
            "uint8_t",
            "_spm_bank_align_pad_indiv",
            np.zeros(64, dtype=np.uint8),
            alignment=64,
        )
    ]
    uB_phys = np.random.randint(
        -7,
        7,
        size=(num_indiv_experts, iN2, iN1, iK2 * iK1, tileSize, meshCol),
        dtype=np.int8,
    )
    data_str += [
        format_vector_definition(
            "uint8_t", "indiv_up_B", pack_int4(uB_phys), alignment=64
        )
    ]
    dB_phys = np.random.randint(
        -7,
        7,
        size=(num_indiv_experts, 2, idN2, idN1, idK2 * idK1, tileSize, down_vc_meshCol),
        dtype=np.int8,
    )
    data_str += [
        format_vector_definition(
            "uint8_t", "indiv_down_B", pack_int4(dB_phys), alignment=64
        )
    ]

    # shared expert weights
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
                "uint8_t", "shared_gate_B", pack_int4(sgB), alignment=128
            )
        ]
        data_str += [
            format_vector_definition(
                "uint8_t",
                "_spm_bank_align_pad_shared",
                np.zeros(64, dtype=np.uint8),
                alignment=64,
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
                "uint8_t", "shared_up_B", pack_int4(suB), alignment=64
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
                "uint8_t", "shared_down_B", pack_int4(sdB), alignment=64
            )
        ]

    # Stride / tile-size constants (legacy dispatch)
    A_token_bytes = int(K1_A * tileSize_A * 2)
    A_total_bytes = int(M_total * A_token_bytes)
    rB_tile = int(rK1 * rN1 * tileSize * meshCol // 2)
    iB_tile = int(iK1 * iN1 * tileSize * meshCol // 2)
    idB_tile = int(idK1 * idN1 * tileSize * down_vc_meshCol // 2)
    sB_tile = int(sK1 * sN1 * tileSize * meshCol // 2)
    sdB_tile = int(sdK1 * sdN1 * tileSize * down_vc_meshCol // 2)
    rD_tile = int(rM1 * rN1 * meshRow_A * meshCol * 2)  # INT16 output: 2 bytes/elem
    iD_tile = int(iM1 * iN1 * meshRow_A * meshCol * 2)  # INT16 output: 2 bytes/elem
    idD_tile = int(
        idM1 * idN1 * meshRow_A * down_vc_meshCol * 2
    )  # INT16 output: 2 bytes/elem
    sD_tile = int(sM1 * sN1 * meshRow_A * meshCol * 2)  # INT16 output: 2 bytes/elem
    sdD_tile = int(
        sdM1 * sdN1 * meshRow_A * down_vc_meshCol * 2
    )  # INT16 output: 2 bytes/elem
    iB_expert_stride = int(iN2 * iK2 * iK1 * iN1 * tileSize * meshCol // 2)
    iB_n2_stride = int(iK2 * iK1 * iN1 * tileSize * meshCol // 2)
    idB_expert_stride = int(
        2 * idN2 * idK2 * idK1 * idN1 * tileSize * down_vc_meshCol // 2
    )
    idB_n2_stride = int(idK2 * idK1 * idN1 * tileSize * down_vc_meshCol // 2)
    sB_expert_stride = int(sN2 * sK2 * sK1 * sN1 * tileSize * meshCol // 2)
    sB_n2_stride = int(sK2 * sK1 * sN1 * tileSize * meshCol // 2)
    sdB_expert_stride = int(
        2 * sdN2 * sdK2 * sdK1 * sdN1 * tileSize * down_vc_meshCol // 2
    )
    sdB_n2_stride = int(sdK2 * sdK1 * sdN1 * tileSize * down_vc_meshCol // 2)
    rA_tile = int(rM1 * K1_A * meshRow_A * tileSize_A * 2)
    sA_tile = int(sM1 * K1_A * meshRow_A * tileSize_A * 2)

    for nm, val in [
        ("A_token_bytes", A_token_bytes),
        ("A_total_bytes", A_total_bytes),
        ("indiv_B_expert_stride", iB_expert_stride),
        ("indiv_B_n2_stride", iB_n2_stride),
        ("indiv_down_B_expert_stride", idB_expert_stride),
        ("indiv_down_B_n2_stride", idB_n2_stride),
        ("router_D_tilesize", rD_tile),
        ("indiv_D_tilesize", iD_tile),
        ("indiv_down_D_tilesize", idD_tile),
        ("router_A_tilesize", rA_tile),
        ("router_B_tilesize", rB_tile),
        ("indiv_B_tilesize", iB_tile),
        ("indiv_down_B_tilesize", idB_tile),
        ("shared_A_tilesize", sA_tile),
        ("shared_B_tilesize", sB_tile),
        ("shared_D_tilesize", sD_tile),
        ("shared_down_D_tilesize", sdD_tile),
        ("shared_down_B_tilesize", sdB_tile),
        ("shared_B_expert_stride", sB_expert_stride),
        ("shared_B_n2_stride", sB_n2_stride),
        ("shared_down_B_expert_stride", sdB_expert_stride),
        ("shared_down_B_n2_stride", sdB_n2_stride),
    ]:
        data_str += [format_scalar_definition("uint32_t", nm, val)]

    # ═══════════════════════════════════════════════════════════════════════
    # L15 layout: TCDM placement + per-shape streamer CSR + golden arrays
    # ═══════════════════════════════════════════════════════════════════════
    log("computing L15 TCDM placement and golden arrays")
    layout = L15_LAYOUT
    placement = place_tensors(globals_, layout)
    a_row_stride = k0_bytes + layout["a_pad"]

    logical_a = make_logical_a(M_total, k0_total)
    a_name = f"layout_A_row_stride_{a_row_stride}"
    a_data_length = M_total * a_row_stride // 2
    data_str += [emit_i16_array(a_name, make_padded_a(logical_a, a_row_stride))]
    data_str += [
        emit_u8_array_lit(
            "layout_W", packed_int4_constant(k0_s0_tiles * n0_s0_tiles, 1)
        )
    ]
    data_str += [
        emit_u8_array_lit(
            "layout_V", packed_int4_constant(k0_s0_tiles * n0_s0_tiles, 2)
        )
    ]
    data_str += [
        emit_u8_array_lit(
            "layout_W2_left", packed_int4_constant(k1_s0_tiles * n1_s0_tiles, 1)
        )
    ]
    data_str += [
        emit_u8_array_lit(
            "layout_W2_right", packed_int4_constant(k1_s0_tiles * n1_s0_tiles, 2)
        )
    ]

    golden_map, golden_names = build_golden_arrays(logical_a, globals_, active_shapes)
    for arr_str in golden_map.values():
        data_str += [arr_str]
    for nm, _, _, _, _ in SHAPE_DIMS_ALL:
        if nm not in golden_names:
            golden_names[nm] = ("NULL", "NULL")

    b_data_length = k0_s0_tiles * n0_s0_tiles * 16
    w2_data_length = k1_s0_tiles * n1_s0_tiles * 16
    shape_cfgs_str = ",\n".join(
        build_shape_cfg(s, globals_, golden_names, layout, placement)
        for s in active_shapes
    )

    layout_cfg_str = "\n".join(
        [
            "static const layout_cfg_t layout_cfgs[NUM_L15_LAYOUTS] = {",
            "    {",
            f"        .layout_id       = {layout['id']},",
            f"        .name            = \"{layout['name']}\",",
            f"        .a_row_stride    = {a_row_stride},",
            f"        .a_data          = {a_name},",
            f"        .a_data_length   = {a_data_length},",
            "        .w_data          = layout_W,",
            "        .v_data          = layout_V,",
            "        .w2_left_data    = layout_W2_left,",
            "        .w2_right_data   = layout_W2_right,",
            f"        .b_data_length   = {b_data_length},",
            f"        .w2_data_length  = {w2_data_length},",
            "        .shapes = {",
            shape_cfgs_str,
            "        }",
            "    }",
            "};",
        ]
    )
    data_str += [layout_cfg_str]

    # Emit device-visible flat shape-config arrays (no host pointer fields).
    # Each array matches moe_l15_shape_cfg_t field layout exactly (91 int32_t values).
    # Used by the L15 kernel via the shape_cfg_addr arg[0] pointer.
    for s in active_shapes:
        flat_vals = build_shape_cfg_flat_vals(s, globals_, layout, placement)
        sname = s[0].lower()  # e.g. "s0", "s1", "s2"
        vals_str = ", ".join(str(v) for v in flat_vals)
        data_str += [
            f"// L15 device shape config for {s[0]} — matches moe_l15_shape_cfg_t\n"
            f"static const int32_t l15_dev_{sname}_cfg[] = {{\n    {vals_str}\n}};"
        ]

    # Shared-expert shape config: S0 with m_tiles=sdM1, processes ALL M_total tokens
    # in ONE kernel call (no loop in the DFG node, the streamer loops internally).
    # To change token count: update router_M1/M2 and A_meshRow in params.hjson,
    #   then re-run: python3 multi_cluster_MoE_datagen.py
    # M_tiles derivation: sdM1 = shared_down_M1 = M_total / meshRow = M_total / 8
    _s0_shape = next(s for s in SHAPE_DIMS_ALL if s[0] == "S0")
    _shared_m_tiles = sdM1  # from params.hjson: shared_down_M1
    _mode1_tok_stride = 2 * globals_["n1_total"] * 2  # bytes per output token (2 VCs)
    flat_vals_shared = build_shape_cfg_flat_vals(
        _s0_shape, globals_, layout, placement, m_tiles=_shared_m_tiles
    )
    _shared_vals_str = ", ".join(str(v) for v in flat_vals_shared)
    data_str += [
        f"// L15 shared-expert shape config (S0, m_tiles={_shared_m_tiles}=shared_down_M1).\n"
        f"// Processes {_shared_m_tiles * _s0_shape[2]} tokens per kernel call = M_total={M_total}.\n"
        f"// mode1_D_tstride[3]={_s0_shape[2] * _mode1_tok_stride} bytes/M-tile (meshRow*token_stride).\n"
        f"// Used by offload_bingo_hw.h Nodes 18/19/21/22 (shared expert swiglu+down-proj).\n"
        f"static const int32_t l15_dev_shared_s0_cfg[] = {{\n    {_shared_vals_str}\n}};"
    ]

    # Scalar constants consumed by main_bingo.py for L15 DMA sizes and offsets.
    a_data_bytes = M_total * a_row_stride
    mode1_padded_bytes = placement["tcdm_end"] - placement["delta_local_mode1_d0"]
    b_data_length = k0_s0_tiles * n0_s0_tiles * 16
    w2_data_length = k1_s0_tiles * n1_s0_tiles * 16
    for nm, val in [
        ("l15_b_data_length", b_data_length),
        ("l15_w2_data_length", w2_data_length),
        ("l15_a_data_bytes", a_data_bytes),
        ("l15_tcdm_size", placement["tcdm_end"]),
        ("l15_delta_local_b0", placement["delta_local_b0"]),
        ("l15_delta_local_b1", placement["delta_local_b1"]),
        ("l15_delta_local_w2l", placement["delta_local_w2l"]),
        ("l15_delta_local_w2r", placement["delta_local_w2r"]),
        ("l15_delta_local_a", placement["delta_local_a"]),
        ("l15_delta_local_mode1_d0", placement["delta_local_mode1_d0"]),
        ("l15_mode1_padded_bytes", mode1_padded_bytes),
    ]:
        data_str += [format_scalar_definition("uint32_t", nm, val)]

    tcdm_kb = placement["tcdm_end"] / 1024
    log(f"L15 end: {placement['tcdm_end']}B ({tcdm_kb:.1f} KB)")
    return data_str


def emit_header_file(**kw):
    acc = (
        kw.get("snax_versacore_core_template")
        or kw.get("snax_dual_versacore_int16x4_core_template")
    )["snax_acc_cfg"][0]
    ashape = kw["array_shape"]
    mc = acc["snax_versacore_spatial_unrolling"][0][ashape][2]
    dmc = int(kw.get("down_meshCol", mc))
    iN2, iN1 = kw["indiv_N2"], kw["indiv_N1"]
    idN2, idN1 = kw.get("indiv_down_N2", iN2), kw.get("indiv_down_N1", iN1)
    n0 = iN2 * iN1 * 4
    n1 = idN2 * idN1 * dmc
    active_count = sum(
        1
        for (_, _, _, _, c) in SHAPE_DIMS_ALL
        if n0 % c == 0 and n1 % c == 0 and n0 // c >= 1 and n1 // c >= 1
    )

    lines = [
        "// Auto-generated by multi_cluster_MoE_datagen.py",
        "// Hardware: snax_dual_versacore_int16x4_4lane_postproc_v2_cluster",
        "// Layout: L15 weights-first, padded-A, bank-colored TCDM",
        "// Do NOT edit by hand.",
        "#pragma once",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        "#define NUM_L15_LAYOUTS   1",
        f"#define NUM_LAYOUT_SHAPES {active_count}",
        "#define DATA_TYPE           0",
        "#define RESCALE_INPUT_ZP    0",
        "#define RESCALE_MULTIPLIER  1",
        "#define RESCALE_OUTPUT_ZP   0",
        "#define RESCALE_SHIFT       0",
        "#define SUBTRACTION_A       0",
        "#define SUBTRACTION_B       0",
        "#define TCDM_CAPACITY_BYTES (8192 * 1024)",
        "",
        SHAPE_CFG_TYPEDEF,
        "",
        LAYOUT_CFG_TYPEDEF,
    ]
    lines += emit_moe_data(**kw)
    return "\n\n".join(lines)


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate L15-layout data and golden tensors for multi_cluster_MoE"
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
        / "../../../../../../../../local_mirrors/snitch_cluster/target/"
        "snitch_cluster/cfg/"
        "snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson",
    )
    parser.add_argument("--emit_mini_golden", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    with args.cfg.open() as f:
        pcfg = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hcfg = hjson.loads(f.read())
    merged = {**pcfg, **hcfg, "emit_mini_golden": args.emit_mini_golden}
    print(emit_header_file(**merged))


if __name__ == "__main__":
    main()
