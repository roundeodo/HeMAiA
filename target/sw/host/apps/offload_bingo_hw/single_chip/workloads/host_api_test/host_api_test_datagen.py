#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Host API test data generator.
# Generates test vectors and golden outputs for MoE operator C functions.
# Usage: python3 host_api_test_datagen.py > host_api_test_data.h

import numpy as np
import struct

np.random.seed(42)

# ============================================================
# Test Configuration (small dimensions for fast testing)
# ============================================================
MESH_ROW = 8
MESH_COL = 8
TILE_SIZE = 8
ROUTER_M1 = 1
ROUTER_N1 = 1
EXPERT_NUM = 2
TOP_K = 2
TOTAL_TOKENS = ROUTER_M1 * MESH_ROW  # 8
INPUT_DIM = 16
SWISH_SCALE_IN = np.float32(0.000015258789)
SWISH_SCALE_OUT = np.float32(32.0)
SOFTMAX_SCALE = np.float32(65536.0)
NUM_SWISH_ELEMS = 16
MAX_PADDED_TOKENS = TOTAL_TOKENS
TEST_EXPERT_ID = 0
SRAM_BUF_SIZE = ROUTER_M1 * ROUTER_N1 * MESH_ROW * MESH_COL  # 64


# ============================================================
# Helpers
# ============================================================
def f32_hex(val):
    """Float32 → uint32 IEEE 754 hex."""
    return struct.unpack("<I", struct.pack("<f", np.float32(val)))[0]


def sram_idx(r, c):
    """Match C moe_get_sram_block_index with test params."""
    return (
        (r // MESH_ROW) * (ROUTER_N1 * MESH_ROW * MESH_COL)
        + (c // MESH_COL) * (MESH_ROW * MESH_COL)
        + (r % MESH_ROW) * MESH_COL
        + (c % MESH_COL)
    )


def emit_array(c_type, name, data):
    flat = data.flatten().tolist()
    vals = ", ".join(str(int(x)) for x in flat)
    return f"{c_type} {name}[] = {{ {vals} }};"


def emit_hex_array(name, data):
    flat = data.flatten().tolist()
    vals = ", ".join(f"0x{int(x):08X}U" for x in flat)
    return f"uint32_t {name}[] = {{ {vals} }};"


# ============================================================
# Test 1: moe_router_global_schedule (find_top_k + extract)
# ============================================================
# Random scores with small range for meaningful softmax probabilities
scores_2D = np.random.randint(-3, 4, size=(TOTAL_TOKENS, EXPERT_NUM), dtype=np.int32)
# Break ties to ensure deterministic top-k ordering (C bubble sort vs numpy)
for r in range(TOTAL_TOKENS):
    if scores_2D[r, 0] == scores_2D[r, 1]:
        scores_2D[r, 0] += 1

# Pack into SRAM block layout
sram_raw_scores = np.zeros(SRAM_BUF_SIZE, dtype=np.int32)
for r in range(TOTAL_TOKENS):
    for c in range(EXPERT_NUM):
        sram_raw_scores[sram_idx(r, c)] = int(scores_2D[r, c])

# Golden: descending sort → top-k indices
golden_topk_idx = np.argsort(scores_2D, axis=1)[:, ::-1][:, :TOP_K].astype(np.uint16)

# Golden: corresponding scores
golden_topk_scores = np.zeros((TOTAL_TOKENS, TOP_K), dtype=np.int32)
for r in range(TOTAL_TOKENS):
    for k in range(TOP_K):
        golden_topk_scores[r, k] = scores_2D[r, golden_topk_idx[r, k]]


# ============================================================
# Test 2: compute_delayed_softmax
# ============================================================
golden_softmax = np.zeros((TOTAL_TOKENS, TOP_K), dtype=np.uint32)
for r in range(TOTAL_TOKENS):
    s = golden_topk_scores[r].astype(np.float32)
    max_s = np.float32(s[0])
    ev = np.exp((s - max_s).astype(np.float32)).astype(np.float32)
    es = np.float32(np.sum(ev))
    golden_softmax[r] = ((ev / es) * SOFTMAX_SCALE).astype(np.uint32)


# ============================================================
# Test 3: build_scatter_metadata
# ============================================================
idx_flat = golden_topk_idx.flatten()

golden_token_counts = np.zeros(EXPERT_NUM, dtype=np.uint32)
for eid in idx_flat:
    golden_token_counts[eid] += 1

golden_mem_offsets = np.zeros(EXPERT_NUM, dtype=np.uint32)
for i in range(1, EXPERT_NUM):
    golden_mem_offsets[i] = golden_mem_offsets[i - 1] + golden_token_counts[i - 1]

golden_rev_flat = np.zeros(TOTAL_TOKENS * TOP_K, dtype=np.uint32)
wp = golden_mem_offsets.copy()
for t in range(TOTAL_TOKENS):
    for k in range(TOP_K):
        fi = t * TOP_K + k
        eid = golden_topk_idx[t, k]
        golden_rev_flat[wp[eid]] = fi
        wp[eid] += 1


# ============================================================
# Test 4: compute_swish_activation_tile
# ============================================================
gate_data = np.random.randint(-50000, 50000, size=NUM_SWISH_ELEMS, dtype=np.int32)

gate_f = gate_data.astype(np.float32) * SWISH_SCALE_IN
sigmoid_g = np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_f).astype(np.float32))
golden_swish = (gate_f * sigmoid_g).astype(np.float32)

golden_swish_hex = np.array([f32_hex(float(x)) for x in golden_swish], dtype=np.uint32)


# ============================================================
# Test 5: compute_glu_multiplication_tile
# ============================================================
up_data = np.random.randint(-50000, 50000, size=NUM_SWISH_ELEMS, dtype=np.int32)

up_f = up_data.astype(np.float32) * SWISH_SCALE_IN
result_f = golden_swish * up_f
result_i32 = (result_f * SWISH_SCALE_OUT).astype(np.int32)
golden_glu = np.clip(result_i32, -128, 127).astype(np.int8)


# ============================================================
# Test 6: scatter_and_pad_input_for_expert
# ============================================================
global_input_A = np.random.randint(
    -128, 127, size=(TOTAL_TOKENS, INPUT_DIM), dtype=np.int8
)

actual_count = int(golden_token_counts[TEST_EXPERT_ID])
offset = int(golden_mem_offsets[TEST_EXPERT_ID])

golden_scatter = np.zeros((MAX_PADDED_TOKENS, INPUT_DIM), dtype=np.int8)
for i in range(min(actual_count, MAX_PADDED_TOKENS)):
    fi = int(golden_rev_flat[offset + i])
    orig_t = fi // TOP_K
    golden_scatter[i] = global_input_A[orig_t]


# ============================================================
# Emit Header
# ============================================================
print("// Auto-generated by host_api_test_datagen.py — do not edit")
print("#pragma once")
print("#include <stdint.h>")
print()

# Defines
print(f"#define TEST_MESH_ROW       {MESH_ROW}")
print(f"#define TEST_MESH_COL       {MESH_COL}")
print(f"#define TEST_TILE_SIZE      {TILE_SIZE}")
print(f"#define TEST_ROUTER_M1      {ROUTER_M1}")
print(f"#define TEST_ROUTER_N1      {ROUTER_N1}")
print(f"#define TEST_EXPERT_NUM     {EXPERT_NUM}")
print(f"#define TEST_TOP_K          {TOP_K}")
print(f"#define TEST_TOTAL_TOKENS   {TOTAL_TOKENS}")
print(f"#define TEST_INPUT_DIM      {INPUT_DIM}")
print(f"#define TEST_NUM_SWISH_ELEMS {NUM_SWISH_ELEMS}")
print(f"#define TEST_MAX_PAD_TOKENS {MAX_PADDED_TOKENS}")
print(f"#define TEST_EXPERT_ID      {TEST_EXPERT_ID}")
print(f"#define TEST_SRAM_BUF_SIZE  {SRAM_BUF_SIZE}")
print(f"#define TEST_SCALE_IN_HEX   0x{f32_hex(SWISH_SCALE_IN):08X}U")
print(f"#define TEST_SCALE_OUT_HEX  0x{f32_hex(SWISH_SCALE_OUT):08X}U")
print(f"#define TEST_SOFTMAX_HEX    0x{f32_hex(SOFTMAX_SCALE):08X}U")
print()

# Test 1 inputs
print("// ---- Test 1: router schedule inputs ----")
print(emit_array("int32_t", "test_sram_scores", sram_raw_scores))
print()

# Test 1 golden
print("// ---- Test 1: router schedule golden ----")
print(emit_array("uint16_t", "golden_topk_idx", golden_topk_idx.flatten()))
print(emit_array("int32_t", "golden_topk_scores", golden_topk_scores.flatten()))
print()

# Test 2 golden
print("// ---- Test 2: softmax golden ----")
print(emit_array("uint32_t", "golden_softmax", golden_softmax.flatten()))
print()

# Test 3 golden
print("// ---- Test 3: scatter metadata golden ----")
print(emit_array("uint32_t", "golden_token_counts", golden_token_counts))
print(emit_array("uint32_t", "golden_mem_offsets", golden_mem_offsets))
print(emit_array("uint32_t", "golden_rev_flat", golden_rev_flat))
print()

# Test 4 inputs + golden
print("// ---- Test 4: swish activation ----")
print(emit_array("int32_t", "test_gate_data", gate_data))
print(emit_hex_array("golden_swish_hex", golden_swish_hex))
print()

# Test 5 inputs + golden
print("// ---- Test 5: GLU multiplication ----")
print(emit_array("int32_t", "test_up_data", up_data))
print(emit_array("int8_t", "golden_glu", golden_glu))
print()

# Test 6 inputs + golden
print("// ---- Test 6: scatter and pad ----")
print(emit_array("int8_t", "test_global_input_A", global_input_A.flatten()))
print(emit_array("int8_t", "golden_scatter", golden_scatter.flatten()))
