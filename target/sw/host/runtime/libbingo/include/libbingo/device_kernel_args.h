// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

// Physical L15 token row: model-sized INT16 payload followed by one 32-byte
// address gap. DMA and streamer payload accesses skip this gap; its value is
// intentionally undefined at runtime.
#define BINGO_MOE_L15_ROW_GAP_BYTES 32u

/* Expert-major routing reference. The low 15 bits select the original token;
 * bit 15 records its TopK rank. Individual gather masks the rank bit, while a
 * later weighted-accumulation node uses it to select probability[token][rank].
 * Keeping both values in the existing uint16 table avoids a separate kpos ABI. */
#define BINGO_MOE_TOKEN_REF_TOKEN_MASK 0x7fffu
#define BINGO_MOE_TOKEN_REF_RANK_SHIFT 15u
#define BINGO_MOE_TOKEN_REF_PACK(token_id, rank) \
  ((uint16_t)(((uint32_t)(token_id) & BINGO_MOE_TOKEN_REF_TOKEN_MASK) | \
              (((uint32_t)(rank) & 1u) << BINGO_MOE_TOKEN_REF_RANK_SHIFT)))
#define BINGO_MOE_TOKEN_REF_TOKEN(ref) \
  ((uint32_t)(ref) & BINGO_MOE_TOKEN_REF_TOKEN_MASK)
#define BINGO_MOE_TOKEN_REF_RANK(ref) \
  (((uint32_t)(ref) >> BINGO_MOE_TOKEN_REF_RANK_SHIFT) & 1u)
#define BINGO_MOE_L15_CFG_WORDS 91u
#include <stdint.h>

#define __SNAX_KERNEL_ARGS_DEFINE typedef struct __attribute__((packed, aligned(4)))

// Every dispatched BINGO kernel-argument struct ends with this 3-field trailer:
//   - gating_sp_addr  : SW guard / CERF group sharing (0 = no guard)
//   - cond_node_index : this node's index in the activation array
//   - scratchpad_ptr  : pointer to this kernel's bingo_kernel_scratchpad_t
//
// The trailer is consumed by BINGO_SW_GUARD_CHECK / BINGO_GET_SP on the
// device side. Append it to every dispatched BINGO args struct as the last entry —
// the user's `;` after the macro invocation supplies the `;` for the
// last field (standard preprocessor-list idiom).
//
// gating_sp_addr + cond_node_index in detail
// ------------------------------------------
// These two fields implement the per-task SW-side gate that pairs with
// the HW CERF gating (Tier 1) for fine-grained conditional execution
// inside a fired CERF group: if `gating_sp_addr` is non-zero the device
// kernel reads the upstream gating task's scratchpad to find a uint8_t
// activation[] array, indexes activation[cond_node_index], and
// early-returns BINGO_RET_SUCC when that slot is 0. The full two-tier
// (HW CERF + SW guard) protocol, the activation-array contract, and a
// worked routing example all live next to the macros that consume
// these fields:
//   target/sw/device/apps/snax/snax-bingo-offload/libsnaxkernel/macros.h
// (search for "SW Guard"). Set `gating_sp_addr = 0` on a kernel arg
// struct to disable the guard for that task; the device-side check then
// short-circuits to a single load + branch-not-taken.
//
// scratchpad_ptr in detail
// ------------------------
// Each task is given a 16-word (64-byte) per-task scratchpad allocated by
// the host runtime before dispatch; this field is the low 32 bits of its
// TCDM-local address (kernel runs on 32-bit snitch). The struct layout,
// the BINGO_GET_SCRATCHPAD accessor, and BINGO_SP_PROFILE live in
// shared/runtime/heterogeneous_runtime.h — see that header for the
// canonical definition.
//
// Three roles the scratchpad plays at runtime:
//   1. Result publication: the kernel writes return_value /
//      num_return_values before returning BINGO_RET_SUCC; downstream
//      tasks and the host post-process read them directly.
//   2. SW-guard activation hand-off: a gating task stashes the address
//      of its uint8_t activation[] array into its own return_value;
//      guarded downstream tasks reach it via gating_sp_addr (see SW
//      guard description in libsnaxkernel/macros.h).
//   3. Per-task profiling: BINGO_SP_PROFILE(sp, field, mcycle) is a
//      no-op unless -DBINGO_SCRATCHPAD_PROFILING is set.
#define BINGO_KERNEL_ARGS_TRAILER \
    uint32_t gating_sp_addr;   \
    uint32_t cond_node_index;  \
    uint32_t scratchpad_ptr


// Define the argument structures for the device kernels
// Each structure is packed and aligned to 4 bytes
// The definition should match the kernel function argument parsing in snax_kernel_lib.h


////////////////////////////////////////////////////////////////////////
///////////////////////// Cluster-level Kernels ////////////////////////
////////////////////////////////////////////////////////////////////////

// Note: name start with __snax_kernel_ 

// Dummy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_dummy_args {
  uint32_t dummy_input;    
} __snax_kernel_dummy_args_t;

// CSR kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_csr_args {
  uint32_t csr_addr;            
  uint32_t csr_value;            
} __snax_kernel_csr_args_t;

// Check Results kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_check_results_args {
  uint32_t golden_data_addr;            
  uint32_t output_data_addr;            
  uint32_t data_size;        // in Bytes
} __snax_kernel_check_results_args_t;

// Check Results Full kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_check_results_full_args {
  uint32_t golden_data_addr_hi;            
  uint32_t golden_data_addr_lo;            
  uint32_t output_data_addr_hi;            
  uint32_t output_data_addr_lo;            
  uint32_t data_size;        // in Bytes
} __snax_kernel_check_results_full_args_t;

// Load-Compute-Store kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_load_compute_store_args {
  uint32_t input_data_addr;            
  uint32_t input_data_size;        // in Bytes
  uint32_t output_data_addr;            
  uint32_t output_data_size;       // in Bytes
} __snax_kernel_load_compute_store_args_t;

// Double Buffer kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_double_buffer_args {
  uint32_t input_data_addr;            
  uint32_t output_data_addr;            
  uint32_t data_size;       // in Bytes
  uint32_t num_tiles;      // Number of tiles >=3
} __snax_kernel_double_buffer_args_t;

// XDMA 1D Copy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_xdma_1d_copy_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
} __snax_kernel_xdma_1d_copy_args_t;

// IDMA 1D Copy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_idma_1d_copy_args_t {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
} __snax_kernel_idma_1d_copy_args_t;

// ---------------------------------------------------------
// ---------------------VERSACORE---------------------------
// ---------------------------------------------------------

// Cluster-level GEMM kernel args. Layout MUST match the parsing in
// offload_sw_kernels/gemm.h (__snax_kernel_versacore_load_compute_store):
// arg0..14, all uint32_t, packed/aligned(4). Mesh dims are intentionally
// absent — the device looks them up from
// runtime/snax/versacore/gemm_shapes.h via array_shape.
//
// Compute: D = A*B + C
//   A: int8, B: int8, C: int32, D: int32
__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_versacore_load_compute_store_args {
  uint32_t input_A_addr_hi;
  uint32_t input_A_addr_lo;
  uint32_t input_B_addr_hi;
  uint32_t input_B_addr_lo;
  uint32_t input_C_addr_hi;
  uint32_t input_C_addr_lo;
  uint32_t output_D_addr_hi;
  uint32_t output_D_addr_lo;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t array_shape;
  uint32_t transpose_A;
  uint32_t transpose_B;
  uint32_t accumPrevC;
} __snax_kernel_versacore_load_compute_store_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_kernel_minimal_cfg_start_gemm_and_wait_args{
  uint32_t input_A_addr_lo;
  uint32_t input_B_addr_lo;
  uint32_t input_C_addr_lo;
  uint32_t output_D_addr_lo;
} __snax_kernel_minimal_cfg_start_gemm_and_wait_args_t;

////////////////////////////////////////////////////////////////////////
//////////////////////// BINGO Core-level Kernels ////////////////////////
////////////////////////////////////////////////////////////////////////

// Note: name start with __snax_bingo_kernel_

// BINGO Dummy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dummy_args {
  uint32_t dummy_input;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dummy_args_t;
// BINGO Entry kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_entry_args {
  uint32_t start_cc_reg_addr;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_entry_args_t;
// BINGO Exit kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_exit_args {
  uint32_t exit_code;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_exit_args_t;

// BINGO IDMA 1D Copy kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_idma_1d_copy_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_idma_1d_copy_args_t;

// BINGO IDMA Broadcast Kernel Args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_idma_broadcast_args {
  uint32_t src_addr_hi;    
  uint32_t src_addr_lo;            
  uint32_t dst_addr_hi;            
  uint32_t dst_addr_lo;            
  uint32_t size;        // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_idma_broadcast_args_t;

// BINGO GEMM Full kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_full_args {
  uint32_t input_A_addr;            
  uint32_t input_B_addr;            
  uint32_t input_C_addr;            
  uint32_t output_D_addr;            
  uint32_t M;            
  uint32_t K;            
  uint32_t N;            
  uint32_t array_shape_idx;            
  uint32_t transpose_A;            
  uint32_t transpose_B;            
  uint32_t accumPrevC;            
  uint32_t quantization_enable;
  uint32_t shift_i;
  uint32_t multiplier_i;
  int32_t input_zp_i;
  int32_t output_zp_i;
  int32_t int32tofp16_enable;
  int32_t int4_a_enable;
  int32_t int4_b_enable;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_full_args_t;

// BINGO XDMA 1D Copy kernel args (same layout as cluster-level)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_1d_copy_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t size;        // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_1d_copy_args_t;

// BINGO Dual DMA (iDMA + xDMA concurrent) kernel args
// iDMA and xDMA are launched simultaneously, then waited in parallel.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dual_dma_args {
  uint32_t idma_src_addr_hi;
  uint32_t idma_src_addr_lo;
  uint32_t idma_dst_addr_hi;
  uint32_t idma_dst_addr_lo;
  uint32_t idma_size;       // in Bytes
  uint32_t xdma_src_addr_hi;
  uint32_t xdma_src_addr_lo;
  uint32_t xdma_dst_addr_hi;
  uint32_t xdma_dst_addr_lo;
  uint32_t xdma_size;       // in Bytes
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dual_dma_args_t;

// BINGO XDMA 6D kernel args (fixed-size, max 5 temporal dims = 6 total dims)
// Exposes full AGU strides/bounds to the user. Unused dims: stride=0, bound=1.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_6d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t spatial_stride_src;
  uint32_t spatial_stride_dst;
  uint32_t num_temporal_dims;        // 1..5
  uint32_t temporal_strides_src[5];  // unused dims = 0
  uint32_t temporal_bounds_src[5];   // unused dims = 1
  uint32_t temporal_strides_dst[5];  // unused dims = 0
  uint32_t temporal_bounds_dst[5];   // unused dims = 1
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_6d_args_t;

// BINGO XDMA Transpose 2D (high-level: user provides shape, kernel computes strides)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_transpose_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t M;              // source rows
  uint32_t N;              // source cols
  uint32_t elem_bytes;     // element size (1=int8, 2=int16, 4=int32)
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_transpose_2d_args_t;

// BINGO XDMA Submatrix 2D (high-level: user provides shape + slice range)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_submatrix_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;       // source matrix rows
  uint32_t src_cols;       // source matrix cols
  uint32_t row_start;      // slice start row (inclusive)
  uint32_t row_end;        // slice end row (exclusive)
  uint32_t col_start;      // slice start col (inclusive)
  uint32_t col_end;        // slice end col (exclusive)
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_submatrix_2d_args_t;

// BINGO XDMA Expand 2D (high-level: broadcast [1, N] -> [M, N])
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_expand_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t M;              // number of output rows (broadcast factor)
  uint32_t N;              // row width (shared by src and dst)
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_expand_2d_args_t;

// BINGO XDMA Concat 2D (high-level: copy one chunk to offset in larger output)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_concat_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;       // rows of THIS input chunk
  uint32_t src_cols;       // cols of THIS input chunk
  uint32_t dst_rows;       // rows of FULL output tensor
  uint32_t dst_cols;       // cols of FULL output tensor
  uint32_t axis;           // 0 = row-concat, 1 = col-concat
  uint32_t offset;         // element offset along concat axis
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_concat_2d_args_t;

// BINGO XDMA Pad 2D (high-level: zero-fill + strided copy into padded output)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_pad_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;
  uint32_t src_cols;
  uint32_t pad_top;        // padding rows before
  uint32_t pad_bottom;     // padding rows after
  uint32_t pad_left;       // padding cols before
  uint32_t pad_right;      // padding cols after
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_pad_2d_args_t;

// BINGO XDMA Gather 2D (high-level: select rows by arithmetic stride)
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_gather_2d_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t src_rows;       // total rows in source tensor
  uint32_t src_cols;       // cols per row
  uint32_t num_indices;    // number of rows to gather
  uint32_t index_start;    // first row index to gather
  uint32_t index_stride;   // stride between indices (1 = contiguous)
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_gather_2d_args_t;

// BINGO XDMA ElementwiseAdd (writer ext: accumulate N int32 operands -> 1).
// Fuses the GEMM K-split partial-sum adds into one streaming xDMA pass.
// num_int32_per_operand must be a multiple of 16 (512b bus / 32b element).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_elementwise_add_args {
  uint32_t src_addr_hi;
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t num_int32_per_operand;
  uint32_t num_operands;
  uint32_t operand_stride;   // bytes between consecutive operand buffers
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_elementwise_add_args_t;

// BINGO XDMA ElementwiseAdd AB (two-operand) (convenience: dst = a + b, int32).
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_elementwise_add_ab_args {
  uint32_t src_a_addr_hi;
  uint32_t src_a_addr_lo;
  uint32_t src_b_addr_hi;
  uint32_t src_b_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t dst_addr_lo;
  uint32_t num_int32;        // multiple of 16
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_elementwise_add_ab_args_t;

// ──────────────────────────────────────────────────────────────────────
// VersaCore blocked-layout conversion kernels
//
// All six kernels convert between row-major (logical 2D) and the three
// VersaCore blocked layouts {A, B, D}. They share a uniform arg layout
// that is parameterized by the scheduler's tile dimensions (M_T, K_T,
// N_T) and the array-shape (meshRow, tileSize, meshCol) so the same
// kernels work for any DSE-chosen tiling.
//
// Layout definitions:
//   A-layout [M_T, K_T, meshRow, tileSize]:
//     A_stored[m, k, r, s] = R_logical[m*meshRow + r, k*tileSize + s]
//   B-layout [N_T, K_T, meshCol, tileSize]:
//     B_stored[n, k, c, s] = R_logical[k*tileSize + s, n*meshCol + c]
//   D-layout [M_T, N_T, meshRow, meshCol]:
//     D_stored[m, n, r, c] = R_logical[m*meshRow + r, n*meshCol + c]
//
// The Python reference implementation lives at
// HeMAiA/util/sim/xdma/layout_convert.py — kernels must produce byte-identical
// output to the reference functions.
// ──────────────────────────────────────────────────────────────────────

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_d_to_row_major_args {
  uint32_t src_addr_hi;   // D-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t M_T;           // VersaCore M-tile count
  uint32_t N_T;           // VersaCore N-tile count
  uint32_t meshRow;
  uint32_t meshCol;
  uint32_t elem_bytes;    // 1 for INT8, 4 for INT32/FP32
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_d_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_a_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // A-layout destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t K_T;
  uint32_t meshRow;
  uint32_t tileSize;
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_row_major_to_a_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_b_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // B-layout destination
  uint32_t dst_addr_lo;
  uint32_t K_T;
  uint32_t N_T;
  uint32_t tileSize;
  uint32_t meshCol;
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_row_major_to_b_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_a_to_row_major_args {
  uint32_t src_addr_hi;   // A-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t K_T;
  uint32_t meshRow;
  uint32_t tileSize;
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_a_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_b_to_row_major_args {
  uint32_t src_addr_hi;   // B-layout source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // row-major destination
  uint32_t dst_addr_lo;
  uint32_t K_T;
  uint32_t N_T;
  uint32_t tileSize;
  uint32_t meshCol;
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_b_to_row_major_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_xdma_row_major_to_d_args {
  uint32_t src_addr_hi;   // row-major source
  uint32_t src_addr_lo;
  uint32_t dst_addr_hi;   // D-layout destination
  uint32_t dst_addr_lo;
  uint32_t M_T;
  uint32_t N_T;
  uint32_t meshRow;
  uint32_t meshCol;
  uint32_t elem_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_xdma_row_major_to_d_args_t;

// BINGO GEMM Minimal kernel args
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_minimal_args {
  uint32_t input_A_addr;            
  uint32_t input_B_addr;            
  uint32_t input_C_addr;            
  uint32_t output_D_addr;            
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_minimal_args_t;

// BINGO GEMM Minimal-SiLU kernel args: like minimal but updates the SiLU CSR.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_gemm_minimal_silu_args {
  uint32_t input_A_addr;
  uint32_t input_B_addr;
  uint32_t input_C_addr;
  uint32_t output_D_addr;
  uint32_t silu_enable;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_gemm_minimal_silu_args_t;

// BINGO Dual-VersaCore GEMM kernel args.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dual_vc_gemm_full_args {
  uint32_t input_A_addr;
  uint32_t input_B0_addr;
  uint32_t input_B1_addr;
  uint32_t output_D0_addr;
  uint32_t output_D1_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t array_shape;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dual_vc_gemm_full_args_t;

// Fixed-S0 Router GEMM. Model dimensions remain generated parameters; the
// hardware shape and core assignment are part of the workload contract.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_router_gemm_s0_args {
  uint32_t input_A_addr;
  uint32_t input_B0_addr;
  uint32_t input_B1_addr;
  uint32_t output_D0_addr;
  uint32_t output_D1_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_router_gemm_s0_args_t;

// Typed L15 configuration record consumed directly by the fused SwiGLU +
// down-projection kernel. It contains dimensions, streamer fields and offsets.
// Datagen emits a designated initializer of this shared type, so field changes
// fail at compile time instead of silently shifting a positional int32 array.
typedef struct __attribute__((packed, aligned(4))) {
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
  int32_t mode1_output_row_stride_bytes, mode1_output_span_elems;
} __snax_bingo_moe_l15_shape_cfg_t;

_Static_assert(
    sizeof(__snax_bingo_moe_l15_shape_cfg_t) ==
        BINGO_MOE_L15_CFG_WORDS * sizeof(uint32_t),
    "L15 config ABI size mismatch");

// BINGO Dual-VersaCore L15 MoE full kernel args (SwiGLU + down-proj in one pass).
// arg[0] shape_cfg_addr: uint32_t TCDM address of
//                       __snax_bingo_moe_l15_shape_cfg_t.
// arg[1] tcdm_base:      uint32_t TCDM base of the L15 layout region.
// arg[2] rescale_mult, arg[3] rescale_shift: post-scale factors.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dual_vc_l15_moe_full_args {
  uint32_t shape_cfg_addr;
  uint32_t tcdm_base;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dual_vc_l15_moe_full_args_t;

// Dense L3 token rows -> bank-partitioned L1 A.  One token occupies two banks;
// each 16-byte K tile advances one complete 512-byte TCDM row.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_stage_tokens_2d_args {
  uint64_t src_addr;
  uint32_t dst_addr;
  uint32_t token_count;
  uint32_t token_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_stage_tokens_2d_args_t;

// Canonical S0 4-column panels -> one bank-partitioned weight chunk pair.
// binding uses the dynamic MoE encoding: 1=iDMA, 2=xDMA, 3=one tensor each.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_stage_weight_pair_2d_args {
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint32_t dst0_addr;
  uint32_t dst1_addr;
  uint32_t bytes_per_block;
  uint32_t block_count;
  uint32_t binding;
  uint32_t phase_xor;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_stage_weight_pair_2d_args_t;

// Bank-partitioned Mode1 D0/D1 -> dense L3 token rows.  xDMA spatial channels
// select tokens while the temporal AGU walks 4-int16 beats down TCDM rows.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_store_tokens_2d_args {
  uint32_t src_d0_addr;
  uint32_t src_d1_addr;
  uint64_t dst_addr;
  uint32_t token_count;
  uint32_t token_bytes;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_store_tokens_2d_args_t;

// Static shared-expert fused execution over the bank-partitioned resident
// layout. Each logical block is one physical chunk. Both block counts are
// generated host-side so the device hot path performs no dimension division.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dual_vc_bank_moe_full_args {
  uint32_t tcdm_base;
  uint32_t token_count;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t s1_block_count;
  uint32_t s3_block_count;
  uint32_t chunk_cols;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dual_vc_bank_moe_full_args_t;

// Direct static-stage SwiGLU launch. N is the total number of temporal
// N-groups across all B blocks; b_block_count selects one S1 block or all
// blocks for the S2 tail.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_swiglu_args {
  uint32_t input_A_addr;
  uint32_t input_gate_B_addr;
  uint32_t input_up_B_addr;
  uint32_t output_D0_addr;
  uint32_t output_D1_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t b_block_count;
  uint32_t b_block_stride;
  uint32_t array_shape;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_swiglu_args_t;

// Direct static-stage down-projection launch. Shape S0/S1 accepts one B
// block; shape S2 may consume all contiguous blocks in one S4 launch.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_down_args {
  uint32_t input_A_addr;
  uint32_t input_B0_addr;
  uint32_t input_B1_addr;
  uint32_t output_D0_addr;
  uint32_t output_D1_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t b_block_count;
  uint32_t b_block_stride;
  uint32_t array_shape;
  uint32_t output_row_stride;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_down_args_t;

// Regression-only arguments for the production last-S1 -> S2 -> S3 chain.
// Three kernels share this image so each task consumes the CSR configuration
// staged by its predecessor without an intervening reconfiguration.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_active_cfg_preload_test_args {
  uint32_t input_s1_A_addr;
  uint32_t input_s2_A_addr;
  uint32_t input_gate_B_addr;
  uint32_t input_up_B_addr;
  uint32_t s1_output_D0_addr;
  uint32_t s2_output_D0_addr;
  uint32_t swiglu_output_D1_addr;
  uint32_t down_input_A_addr;
  uint32_t input_down_B0_addr;
  uint32_t input_down_B1_addr;
  uint32_t down_output_D0_addr;
  uint32_t down_output_D1_addr;
  uint32_t s1_M;
  uint32_t s1_N;
  uint32_t s1_array_shape;
  uint32_t s2_M;
  uint32_t swiglu_K;
  uint32_t s2_N;
  uint32_t s2_b_block_count;
  uint32_t swiglu_b_block_stride;
  uint32_t s2_array_shape;
  uint32_t down_M;
  uint32_t down_K;
  uint32_t down_N;
  uint32_t down_array_shape;
  uint32_t down_output_row_stride;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_active_cfg_preload_test_args_t;

// One-time initialization of L15 row gaps skipped by Mode-1 writers. This is
// used only for shared outputs whose host-iDMA writeback copies the full span.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_init_output_gaps_args {
  uint32_t output_base;
  uint32_t row_payload_bytes;
  uint32_t row_stride_bytes;
  uint32_t rows;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_init_output_gaps_args_t;

typedef struct __attribute__((packed, aligned(4))) {
  uint64_t token_refs_addr;
  uint64_t input_A_l3_base;
  uint64_t indiv_gate_B_l3;
  uint64_t indiv_up_B_l3;
  uint64_t indiv_down_B_l3;
  uint64_t output_l3_base;
  uint32_t active_state_l1_addr;
  uint32_t l1_a_addr;
  uint32_t l1_b_gate_addr;
  uint32_t l1_b_up_addr;
  uint32_t l1_b_down_addr;
  uint32_t l1_d_addr;
  uint32_t l1_down_d_addr;
  uint32_t l1_d1_scratch_addr;
  uint32_t A_token_bytes;
  uint32_t indiv_B_expert_stride;
  uint32_t indiv_down_B_expert_stride;
  uint32_t indiv_B_block_stride;
  uint32_t indiv_down_B_block_stride;
  uint32_t s1_block_count;
  uint32_t s3_block_count;
  uint32_t indiv_K1;
  uint32_t indiv_N_per_block;
  uint32_t indiv_down_K1;
  uint32_t indiv_down_N_per_block;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  uint32_t output_expert_stride_bytes;
  uint32_t max_tokens_per_expert;
  /* Model- and hardware-static values used by every slot. They are generated
   * once with the static context instead of being recomputed on Snitch. */
  uint32_t A_row_stride;
  uint32_t s3_row_bytes;
  uint32_t down_half_weight_bytes;
  uint32_t down_b_k_section;
  uint32_t down_b_n_stride[3];
  uint32_t down_a_m_stride[3];
  uint32_t down_d_m_stride[3];
} __snax_bingo_moe_dynamic_expert_static_args_t;

#define BINGO_MOE_MAX_BLOCKS 8u

typedef struct __attribute__((packed, aligned(8))) {
  union {
    struct { uint32_t valid; uint32_t output_D0_addr; };
    uint64_t valid_output_word;
  };
  union {
    struct { uint32_t N; uint32_t array_shape; };
    uint64_t n_shape_word;
  };
} __snax_bingo_moe_dyn_s1_call_args_t;

typedef struct __attribute__((packed, aligned(8))) {
  union {
    /* token_start is task-local and logical, not a physical L1 address. */
    struct { uint32_t valid; uint32_t token_start; };
    uint64_t valid_input_word;
  };
  union {
    struct { uint32_t reserved; uint32_t M; };
    uint64_t output_m_word;
  };
  union {
    struct { uint32_t N; uint32_t array_shape; };
    uint64_t n_shape_word;
  };
} __snax_bingo_moe_dyn_s2_call_args_t;

typedef struct __attribute__((packed, aligned(8))) {
  union {
    struct { uint32_t valid; uint32_t N; };
    uint64_t valid_n_word;
  };
  union {
    struct { uint32_t array_shape; uint32_t reserved; };
    uint64_t shape_reserved_word;
  };
} __snax_bingo_moe_dyn_s3_call_args_t;

typedef struct __attribute__((packed, aligned(8))) {
  union {
    /* token_start is task-local and logical, not a physical L1 address. */
    struct { uint32_t valid; uint32_t token_start; };
    uint64_t valid_input_word;
  };
  union {
    struct { uint32_t reserved0; uint32_t reserved1; };
    uint64_t output_pair_word;
  };
  union {
    struct { uint32_t M; uint32_t N; };
    uint64_t m_n_word;
  };
  union {
    struct { uint32_t array_shape; uint32_t reserved; };
    uint64_t shape_reserved_word;
  };
} __snax_bingo_moe_dyn_s4_call_args_t;

typedef struct __attribute__((packed, aligned(8)))
    __snax_bingo_kernel_moe_dynamic_expert_args {
  /* Final CVA6-lowered task arguments. MoEPrepare resolves the RTL task word
   * into complete S1/S2/S3/S4 call records; MoEExecute only flushes active
   * records into L1. Device compute kernels consume these calls directly.
   * bit  0:      active             (1 = slot valid, Snitch will execute)
   * bit  1:      skip_s1            (1 = S1 load+compute 全跳过, cache hit)
   * bit  2:      skip_s3            (1 = S3 load+compute 全跳过, cache hit)
   * bit  3:      skip_s2            (1 = S2 full GEMM skipped)
   * bit  4:      skip_s4            (1 = S4 full GEMM skipped)
   * bits [6:5]:  shape_s1           (0=M8/ShapeA, 1=M4/ShapeB, 2=M2/ShapeC)
   * bits [8:7]:  shape_s3
   * bits [10:9]: dma_s1             (0=NONE, 1=IDMA, 2=XDMA, 3=BOTH)
   * bits [12:11]:dma_s3
   * bit  13:     runtime_cluster_idx (0=C2, 1=C3)
   * bits [19:14]:slot_id            (0-63, local slot index)
   * ──────────────────────────────────────────────────────────────────────────── */
  union {
    struct {
      uint32_t ctrl;
      uint32_t expert_id;
    };
    uint64_t ctrl_expert_word;
  } __attribute__((aligned(8)));
  union {
    struct {
      uint32_t token_ref_start;
      uint32_t ntokens;
    };
    uint64_t token_range_word;
  } __attribute__((aligned(8)));
  union {
    struct {
      uint32_t m_s2_exec; /* S2 M-tile count for s2_call.array_shape. */
      uint32_t m_s4_exec; /* S4 M-tile count for s4_call.array_shape. */
    };
    uint64_t m_exec_word;
  } __attribute__((aligned(8)));
  /* ── dma_slot_vd: packed valid + DMA binding for all 4 DMA slots ───────────
   * For slot i (i=0..3), bits at offset i*3:
   *   bit[i*3+0]: valid      (1 = slot carries a DMA operation)
   *   bit[i*3+2:i*3+1]: dma  (0=NONE, 1=IDMA, 2=XDMA, 3=BOTH)
   * Replaces: dma_slot_valid[4] (16B) + dma_slot_dma[4] (16B) → 1 word (4B)
   * ──────────────────────────────────────────────────────────────────────────── */
  union {
    struct {
      uint32_t dma_slot_vd;
      /* Four 6-bit expert IDs. Validity lives in dma_slot_vd. */
      uint32_t dma_slot_eids;
    };
    uint64_t dma_slot_word;
  } __attribute__((aligned(8)));
  __snax_bingo_moe_dyn_s1_call_args_t s1_call[BINGO_MOE_MAX_BLOCKS];
  __snax_bingo_moe_dyn_s2_call_args_t s2_call;
  __snax_bingo_moe_dyn_s3_call_args_t s3_call[BINGO_MOE_MAX_BLOCKS];
  __snax_bingo_moe_dyn_s4_call_args_t s4_call;
} __snax_bingo_kernel_moe_dynamic_expert_args_t;

_Static_assert(
  sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t) == 344u,
  "CVA6-lowered MoE task ABI size changed unexpectedly");

#define BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES 384u
#define BINGO_MOE_STATIC_ARG_SLOT_BYTES 192u
_Static_assert(
  sizeof(__snax_bingo_moe_dynamic_expert_static_args_t) <=
      BINGO_MOE_STATIC_ARG_SLOT_BYTES,
  "MoE static context exceeds its L1 ABI slot");

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_dynamic_expert_block_args {
  uint32_t task_arg_addr;
  uint32_t static_arg_addr;
  uint32_t pipeline_ctrl_addr;
  uint32_t block_idx;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_dynamic_expert_block_args_t;
