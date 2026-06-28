// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include <stdint.h>

#define __SNAX_KERNEL_ARGS_DEFINE typedef struct __attribute__((packed, aligned(4)))

// Every BINGO core-level kernel args struct ends with this 3-field trailer:
//   - gating_sp_addr  : SW guard / CERF group sharing (0 = no guard)
//   - cond_node_index : this node's index in the activation array
//   - scratchpad_ptr  : pointer to this kernel's bingo_kernel_scratchpad_t
//
// The trailer is consumed by BINGO_SW_GUARD_CHECK / BINGO_GET_SP on the
// device side. Append it to every BINGO args struct as the last entry —
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

// BINGO Dual-VersaCore SwiGLU kernel args.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dual_vc_swiglu_full_args {
  uint32_t input_A_addr;
  uint32_t input_B_gate_addr;
  uint32_t input_B_up_addr;
  uint32_t output_D0_addr;
  uint32_t output_D1_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t array_shape;
  uint32_t rescale_mult;
  uint32_t rescale_shift;

  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dual_vc_swiglu_full_args_t;

// BINGO Dual-VersaCore L15 MoE full kernel args (SwiGLU + down-proj in one pass).
// arg[0] shape_cfg_addr: uint32_t L3 address of l15_dev_sX_cfg[] (moe_l15_shape_cfg_t).
// arg[1] tcdm_base:      uint32_t TCDM base of the L15 layout region.
// arg[2] rescale_mult, arg[3] rescale_shift: post-scale factors.
__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_dual_vc_l15_moe_full_args {
  uint32_t shape_cfg_addr;
  uint32_t tcdm_base;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_dual_vc_l15_moe_full_args_t;

// Split-kernel variants: same arg layout as _full.
typedef __snax_bingo_kernel_dual_vc_l15_moe_full_args_t
        __snax_bingo_kernel_dual_vc_l15_moe_swiglu_args_t;
typedef __snax_bingo_kernel_dual_vc_l15_moe_full_args_t
        __snax_bingo_kernel_dual_vc_l15_moe_down_args_t;

typedef struct __attribute__((packed, aligned(4))) {
  uint32_t valid;
  uint32_t input_A_addr;
  uint32_t input_B_gate_addr;
  uint32_t input_B_up_addr;
  uint32_t output_D0_addr;
  uint32_t output_D1_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t array_shape;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
} __snax_bingo_moe_dyn_swiglu_call_args_t;

typedef struct __attribute__((packed, aligned(4))) {
  uint32_t valid;
  uint32_t input_A_addr;
  uint32_t input_B0_addr;
  uint32_t input_B1_addr;
  uint32_t output_D0_addr;
  uint32_t output_D1_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t array_shape;
  uint32_t d_row_stride_override;
  uint32_t rescale_mult;
  uint32_t rescale_shift;
} __snax_bingo_moe_dyn_down_call_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_dynamic_expert_args {
  /* ── ctrl: packed control word (19 bits used, written every round by Phase4) ──────
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
   * bits [18:14]:slot_id            (0-31, local slot index)
   * ──────────────────────────────────────────────────────────────────────────── */
  uint32_t ctrl;
  uint32_t expert_id;
  uint32_t token_start_rank;
  uint32_t ntokens;
  uint32_t m_s2_exec;    /* S2 M-tile 数 = ⌈tail_tokens / 2⌉（shape C 固定 meshRow=2）; 0 when skip_s2=1 */
  uint32_t m_s4_exec;    /* S4 M-tile 数 = ⌈tail_tokens / 2⌉（shape C 固定 meshRow=2）; 0 when skip_s4=1 */
  uint32_t wait_for_peer_slots;
  /* ── dma_slot_vd: packed valid + DMA binding for all 4 DMA slots ───────────
   * For slot i (i=0..3), bits at offset i*3:
   *   bit[i*3+0]: valid      (1 = slot carries a DMA operation)
   *   bit[i*3+2:i*3+1]: dma  (0=NONE, 1=IDMA, 2=XDMA, 3=BOTH)
   * Replaces: dma_slot_valid[4] (16B) + dma_slot_dma[4] (16B) → 1 word (4B)
   * ──────────────────────────────────────────────────────────────────────────── */
  uint32_t dma_slot_vd;
  int32_t  dma_slot_expert_id[4];   /* prefetch target expert, -1 = none */
  uint32_t dma_slot_idma_seq[4];    /* iDMA sequence number for ordering  */
  uint32_t dma_slot_xdma_seq[4];    /* xDMA sequence number for ordering  */
  /* ── static fields: pre-filled at DFG init, never written by Phase4 ─────── */
  uint64_t token_ids_addr;
  uint64_t input_A_l3_base;
  uint64_t indiv_gate_B_l3;
  uint64_t indiv_up_B_l3;
  uint64_t indiv_down_B_l3;
  uint64_t output_l3_base;
  uint64_t runtime_state_addr;
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
  uint32_t indiv_B_tile_bytes;
  uint32_t indiv_D_tile_bytes;
  uint32_t indiv_down_B_tile_bytes;
  uint32_t indiv_down_D_tile_bytes;
  uint32_t indiv_N2;
  uint32_t indiv_down_N2;
  uint32_t indiv_K1;
  uint32_t indiv_N_per_block;       // = indiv_N1 x meshCol(shape); shape-invariant constant
  uint32_t indiv_down_K1;
  uint32_t indiv_down_N_per_block;  // = indiv_down_N1 x vc_meshCol(shape); shape-invariant constant
  // array_shape is now pre-lowered by the host into s1/s2/s3/s4 call args.
  uint32_t rescale_mult;
  uint32_t rescale_shift;
  uint32_t output_expert_stride_bytes;
  uint32_t max_tokens_per_expert;
  uint32_t s1_block_count;
  uint32_t s3_block_count;
  __snax_bingo_moe_dyn_swiglu_call_args_t s1_call[2];
  __snax_bingo_moe_dyn_swiglu_call_args_t s2_call;
  __snax_bingo_moe_dyn_down_call_args_t s3_call[2];
  __snax_bingo_moe_dyn_down_call_args_t s4_call;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_dynamic_expert_args_t;

__SNAX_KERNEL_ARGS_DEFINE __snax_bingo_kernel_moe_dynamic_expert_block_args {
  uint64_t task_arg_addr;
  uint32_t block_idx;
  BINGO_KERNEL_ARGS_TRAILER;
} __snax_bingo_kernel_moe_dynamic_expert_block_args_t;
