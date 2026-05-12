// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Dual VersaCore SwiGLU library — replaces old single-versacore API.
// New hardware: 3 readers (A=INT16, B0=INT4, B1=INT4) + 2 writers (D0=INT16, D1=INT16)
// Mode 0: SwiGLU (A@B0 → SiLU → elemMul(A@B1)) → D0/D1
// Mode 1: GEMM   (A@B0 → D0)  [used for down projection, router, etc.]

#pragma once

#include <stdbool.h>
#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

// Reading guide:
//   1. CSR macros below describe the dual-VersaCore private register block.
//   2. set_*streamer* helpers program the 3-reader/2-writer streamer frontend.
//   3. set_*csr/mode/rescale/start/wait helpers control the compute datapath.
//   4. The compatibility aliases at the bottom exist only so old kernels still
//      compile while the software stack migrates to the dual-VersaCore API.

// ============================================================
// Dual VersaCore Accelerator CSR definitions
// ============================================================
#define DUAL_VC_CSR_ADDR_BASE         (STREAMER_WRITER1_BUSY_CSR + 1)  // 1034
#define DUAL_VC_OVERWRITE_ACCUM       (DUAL_VC_CSR_ADDR_BASE)          // [0]
#define DUAL_VC_ACCUM_BOUND           (DUAL_VC_OVERWRITE_ACCUM + 1)    // [1]
#define DUAL_VC_OUTPUT_BOUND          (DUAL_VC_ACCUM_BOUND + 1)        // [2]
#define DUAL_VC_SUBTRACTIONS          (DUAL_VC_OUTPUT_BOUND + 1)       // [3]
#define DUAL_VC_ARRAY_SHAPE_CFG       (DUAL_VC_SUBTRACTIONS + 1)       // [4]
#define DUAL_VC_DATA_TYPE_CFG         (DUAL_VC_ARRAY_SHAPE_CFG + 1)    // [5]
#define DUAL_VC_MODE                  (DUAL_VC_DATA_TYPE_CFG + 1)      // [6]
// Rescale0 (VC0/gate path):
#define DUAL_VC_RESCALE0_INPUT_ZP     (DUAL_VC_MODE + 1)               // [7]
#define DUAL_VC_RESCALE0_MULTIPLIER   (DUAL_VC_RESCALE0_INPUT_ZP + 1)  // [8]
#define DUAL_VC_RESCALE0_OUTPUT_ZP    (DUAL_VC_RESCALE0_MULTIPLIER + 1)// [9]
#define DUAL_VC_RESCALE0_SHIFT        (DUAL_VC_RESCALE0_OUTPUT_ZP + 1) // [10]
// Rescale1 (VC1/up path):
#define DUAL_VC_RESCALE1_INPUT_ZP     (DUAL_VC_RESCALE0_SHIFT + 1)     // [11]
#define DUAL_VC_RESCALE1_MULTIPLIER   (DUAL_VC_RESCALE1_INPUT_ZP + 1)  // [12]
#define DUAL_VC_RESCALE1_OUTPUT_ZP    (DUAL_VC_RESCALE1_MULTIPLIER + 1)// [13]
#define DUAL_VC_RESCALE1_SHIFT        (DUAL_VC_RESCALE1_OUTPUT_ZP + 1) // [14]
// Rescale_mul (after elem_mul, Mode 0 only):
#define DUAL_VC_RESCALE_MUL_INPUT_ZP  (DUAL_VC_RESCALE1_SHIFT + 1)    // [15]
#define DUAL_VC_RESCALE_MUL_MULTIPLIER (DUAL_VC_RESCALE_MUL_INPUT_ZP + 1)  // [16]
#define DUAL_VC_RESCALE_MUL_OUTPUT_ZP (DUAL_VC_RESCALE_MUL_MULTIPLIER + 1) // [17]
#define DUAL_VC_RESCALE_MUL_SHIFT     (DUAL_VC_RESCALE_MUL_OUTPUT_ZP + 1)  // [18]
// Control:
#define DUAL_VC_START                 (DUAL_VC_RESCALE_MUL_SHIFT + 1)  // [19]
// Read-only:
#define DUAL_VC_BUSY                  (DUAL_VC_START + 1)
#define DUAL_VC_PERFORMANCE_COUNTER   (DUAL_VC_BUSY + 1)

// Writer-only busy CSRs (for pipeline blocking)
#define WRITER_BUSY_CSR               STREAMER_WRITER_BUSY_CSR
#define WRITER1_BUSY_CSR              STREAMER_WRITER1_BUSY_CSR

// ============================================================
// Channel enable values for dual-versacore (S0/S1/S2 shapes)
// All shapes give same channel enables due to max() rounding:
//   A reader: 16 channels enabled (INT16, 1024-bit wide)
//   B0/B1 reader: 8 channels enabled (INT4, up to 512-bit)
//   D0/D1 writer: 8 channels enabled (INT16 output, 512-bit)
// ============================================================
#define DUAL_VC_CHAN_EN_A   0x0000FFFFu   // 16 channels
#define DUAL_VC_CHAN_EN_B   0x000000FFu   // 8 channels
#define DUAL_VC_CHAN_EN_D   0x000000FFu   // 8 channels

// ============================================================
// New Dual VersaCore Functions
// ============================================================

// Packs the two subtraction / zero-point operands into the format expected by
// the shared subtraction CSR used by the quantized datapath.
static inline int32_t gen_dual_vc_subtraction_config(int8_t sub_a, int8_t sub_b) {
    return (int32_t)(((uint8_t)sub_b << 8) | (uint8_t)sub_a);
}

// ============================================================
// Streamer programming helpers
// ============================================================

// Full streamer configuration for dual-versacore (3R+2W)
// All addresses are absolute TCDM addresses (uint32_t low 32 bits).
// The stride/bound arrays already follow the layout generated in
// streamer_csr_addr_map.h; this helper simply writes them in the fixed port
// order A / B0 / B1 / D0 / D1.
static inline void set_dual_versacore_streamer_csr(
    uint32_t A_addr,  int32_t* Aslstride,  int32_t* Atlbound,  int32_t* Atlstride,
    int32_t remap_A,  int32_t* chan_en_A,
    uint32_t B0_addr, int32_t* B0slstride, int32_t* B0tlbound, int32_t* B0tlstride,
    int32_t remap_B0, int32_t* chan_en_B0,
    uint32_t B1_addr, int32_t* B1slstride, int32_t* B1tlbound, int32_t* B1tlstride,
    int32_t remap_B1, int32_t* chan_en_B1,
    uint32_t D0_addr, int32_t* D0slstride, int32_t* D0tlbound, int32_t* D0tlstride,
    int32_t remap_D0, int32_t* chan_en_D0,
    uint32_t D1_addr, int32_t* D1slstride, int32_t* D1tlbound, int32_t* D1tlstride,
    int32_t remap_D1, int32_t* chan_en_D1) {

    // ------ Reader 0: A (INT16) ------
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(S_STRIDE_BASE_READER_0 + 0, Aslstride[0]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++)  csrw_ss(T_BOUND_BASE_READER_0  + i, Atlbound[i]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) csrw_ss(T_STRIDE_BASE_READER_0 + i, Atlstride[i]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, remap_A);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0 + 0, chan_en_A[0]);

    // ------ Reader 1: B0/gate (INT4) ------
    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(S_STRIDE_BASE_READER_1 + 0, B0slstride[0]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++)  csrw_ss(T_BOUND_BASE_READER_1  + i, B0tlbound[i]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) csrw_ss(T_STRIDE_BASE_READER_1 + i, B0tlstride[i]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, remap_B0);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1 + 0, chan_en_B0[0]);

    // ------ Reader 2: B1/up (INT4) ------
    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2 + 0, B1slstride[0]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_2; i++)  csrw_ss(T_BOUND_BASE_READER_2  + i, B1tlbound[i]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_2; i++) csrw_ss(T_STRIDE_BASE_READER_2 + i, B1tlstride[i]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, remap_B1);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2 + 0, chan_en_B1[0]);

    // ------ Writer 0: D0 (INT16) ------
    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0 + 0, D0slstride[0]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++)  csrw_ss(T_BOUND_BASE_WRITER_0  + i, D0tlbound[i]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) csrw_ss(T_STRIDE_BASE_WRITER_0 + i, D0tlstride[i]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, remap_D0);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0 + 0, chan_en_D0[0]);

    // ------ Writer 1: D1 (INT16) ------
    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1 + 0, D1slstride[0]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_WRITER_1; i++)  csrw_ss(T_BOUND_BASE_WRITER_1  + i, D1tlbound[i]);
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_WRITER_1; i++) csrw_ss(T_STRIDE_BASE_WRITER_1 + i, D1tlstride[i]);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, remap_D1);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1 + 0, chan_en_D1[0]);
}

// Minimal streamer config: only update base pointers (5 CSR writes).
// Used on the hot path when only tensor locations change but tensor shape and
// channel topology stay fixed across launches.
static inline void set_minimal_dual_vc_streamer_cfg(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr) {
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
}

// ============================================================
// Compute-side CSR helpers
// ============================================================

// Configure accelerator CSRs (shared by both VersaCores)
static inline void set_dual_versacore_csr(
    uint32_t take_in_new_c,
    uint32_t a_b_input_times_one_output,
    uint32_t output_times,
    uint32_t subtractions,
    uint32_t array_shape,
    uint32_t data_type) {
    csrw_ss(DUAL_VC_OVERWRITE_ACCUM, take_in_new_c);
    csrw_ss(DUAL_VC_ACCUM_BOUND,     a_b_input_times_one_output);
    csrw_ss(DUAL_VC_OUTPUT_BOUND,    output_times);
    csrw_ss(DUAL_VC_SUBTRACTIONS,    subtractions);
    csrw_ss(DUAL_VC_ARRAY_SHAPE_CFG, array_shape);
    csrw_ss(DUAL_VC_DATA_TYPE_CFG,   data_type);
}

// Mode 0 = SwiGLU, Mode 1 = GEMM (down projection / router)
static inline void set_dual_versacore_mode(uint32_t mode) {
    csrw_ss(DUAL_VC_MODE, mode);
}

// Rescale parameters for VC0 (gate path / Mode 1 output)
static inline void set_dual_versacore_rescale0(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift) {
    csrw_ss(DUAL_VC_RESCALE0_INPUT_ZP,   (uint32_t)input_zp);
    csrw_ss(DUAL_VC_RESCALE0_MULTIPLIER, multiplier);
    csrw_ss(DUAL_VC_RESCALE0_OUTPUT_ZP,  (uint32_t)output_zp);
    csrw_ss(DUAL_VC_RESCALE0_SHIFT,      shift);
}

// Rescale parameters for VC1 (up path)
static inline void set_dual_versacore_rescale1(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift) {
    csrw_ss(DUAL_VC_RESCALE1_INPUT_ZP,   (uint32_t)input_zp);
    csrw_ss(DUAL_VC_RESCALE1_MULTIPLIER, multiplier);
    csrw_ss(DUAL_VC_RESCALE1_OUTPUT_ZP,  (uint32_t)output_zp);
    csrw_ss(DUAL_VC_RESCALE1_SHIFT,      shift);
}

// Rescale parameters after element-wise multiply (Mode 0 only)
static inline void set_dual_versacore_rescale_mul(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift) {
    csrw_ss(DUAL_VC_RESCALE_MUL_INPUT_ZP,   (uint32_t)input_zp);
    csrw_ss(DUAL_VC_RESCALE_MUL_MULTIPLIER, multiplier);
    csrw_ss(DUAL_VC_RESCALE_MUL_OUTPUT_ZP,  (uint32_t)output_zp);
    csrw_ss(DUAL_VC_RESCALE_MUL_SHIFT,      shift);
}

// ============================================================
// Run-control helpers
// ============================================================

static inline void start_dual_vc_streamer() { csrw_ss(STREAMER_START_CSR, 1); }
static inline void start_dual_vc()          { csrw_ss(DUAL_VC_START, 1); }

static inline void start_dual_vc_and_streamer() {
    start_dual_vc_streamer();   // STREAMER must be started first so it is ready
    start_dual_vc();            // then fire the accelerator
}

static inline void wait_dual_vc() {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    while (csrr_ss(DUAL_VC_BUSY)) {}
}

static inline void wait_dual_vc_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    while (csrr_ss(STREAMER_BUSY_CSR)) {}
}

static inline void wait_dual_vc_and_streamer() {
    wait_dual_vc();
    wait_dual_vc_streamer();
}

// Wait only for the writer side to drain. This is useful when Mode 0 pipelines
// the multiply/output path and the caller needs a narrower fence than a full
// compute+streamer stop.
static inline void wait_dual_vc_writer() {
    while (csrr_ss(WRITER_BUSY_CSR) || csrr_ss(WRITER1_BUSY_CSR)) {}
}

static inline uint32_t read_dual_vc_perf_counter() {
    return csrr_ss(DUAL_VC_PERFORMANCE_COUNTER);
}

static inline uint32_t read_dual_vc_streamer_perf_counter() {
    return csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
}

// ============================================================
// Backward compatibility aliases (old single-versacore names)
// ============================================================
#define VERSACORE_CSR_ADDR_BASE       DUAL_VC_CSR_ADDR_BASE
#define OVERWRITE_ACCUM               DUAL_VC_OVERWRITE_ACCUM
#define ACCUM_BOUND                   DUAL_VC_ACCUM_BOUND
#define OUTPUT_BOUND                  DUAL_VC_OUTPUT_BOUND
#define SUBTRACTIONS                  DUAL_VC_SUBTRACTIONS
#define ARRAY_SHAPE_CFG               DUAL_VC_ARRAY_SHAPE_CFG
#define DATA_TYPE_CFG                 DUAL_VC_DATA_TYPE_CFG
#define VERSACORE_START_CSR           DUAL_VC_START
#define VERSACORE_BUSY                DUAL_VC_BUSY
#define VERSACORE_PERFORMANCE_COUNTER DUAL_VC_PERFORMANCE_COUNTER

static inline uint32_t gen_subtraction_config(int8_t sub_a, int8_t sub_b) {
    return (uint32_t)gen_dual_vc_subtraction_config(sub_a, sub_b);
}

static inline uint32_t versacore_compat_l1_addr(uint32_t addr) {
    uint32_t l1_base = snrt_l1_start_addr();
    return (addr < l1_base) ? (addr + l1_base) : addr;
}

static inline void __set_versacore_streamer_csr_common(
    uint32_t A_addr,  void* Asl_raw,  void* Atb_raw,  void* Ats_raw,
    uint32_t rimA, uint32_t trA, void* ceA_raw,
    uint32_t B_addr,  void* Bsl_raw,  void* Btb_raw,  void* Bts_raw,
    uint32_t rimB, uint32_t trB, void* ceB_raw,
    uint32_t C_addr,  void* Csl_raw,  void* Ctb_raw,  void* Cts_raw,
    uint32_t rimC, void* ceC_raw,
    uint32_t D_addr,  void* Dsl_raw,  void* Dtb_raw,  void* Dts_raw,
    uint32_t rimD, void* ceD_raw,
    int32_t shape, uint32_t qen, uint32_t shift, uint32_t mult,
    int32_t izp, int32_t ozp, int32_t fp16en, int32_t int4a, int32_t int4b,
    int32_t silu) {
    int32_t* Asl = (int32_t*)Asl_raw;
    int32_t* Atb = (int32_t*)Atb_raw;
    int32_t* Ats = (int32_t*)Ats_raw;
    int32_t* Bsl = (int32_t*)Bsl_raw;
    int32_t* Btb = (int32_t*)Btb_raw;
    int32_t* Bts = (int32_t*)Bts_raw;
    int32_t* Csl = (int32_t*)Csl_raw;
    int32_t* Ctb = (int32_t*)Ctb_raw;
    int32_t* Cts = (int32_t*)Cts_raw;
    int32_t* Dsl = (int32_t*)Dsl_raw;
    int32_t* Dtb = (int32_t*)Dtb_raw;
    int32_t* Dts = (int32_t*)Dts_raw;
    int32_t* ceA = (int32_t*)ceA_raw;
    int32_t* ceB = (int32_t*)ceB_raw;
    int32_t* ceC = (int32_t*)ceC_raw;
    int32_t* ceD = (int32_t*)ceD_raw;

#if defined(BASE_PTR_READER_WRITER_0_LOW) && defined(BASE_PTR_READER_WRITER_1_LOW)
    csrw_ss(BASE_PTR_READER_0_LOW, versacore_compat_l1_addr(A_addr));
    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_0 + i, Asl[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atb[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Ats[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, rimA);
#endif
#ifdef ENABLED_CHANNEL_READER_0
    for (int i = 0; i < ENABLED_CHANNEL_READER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_0 + i, ceA[i]);
    }
#endif

    csrw_ss(BASE_PTR_READER_1_LOW, versacore_compat_l1_addr(B_addr));
    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_1 + i, Bsl[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_1 + i, Btb[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_1 + i, Bts[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, rimB);
#endif
#ifdef ENABLED_CHANNEL_READER_1
    for (int i = 0; i < ENABLED_CHANNEL_READER_1_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_1 + i, ceB[i]);
    }
#endif

    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, versacore_compat_l1_addr(C_addr));
    for (int i = 0; i < S_STRIDE_NUM_READER_WRITER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_WRITER_0 + i, Csl[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_READER_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_WRITER_0 + i, Ctb[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_READER_WRITER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_WRITER_0 + i, Cts[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, rimC);
#endif
#ifdef ENABLED_CHANNEL_READER_WRITER_0
    for (int i = 0; i < ENABLED_CHANNEL_READER_WRITER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_WRITER_0 + i, ceC[i]);
    }
#endif

    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, versacore_compat_l1_addr(D_addr));
    for (int i = 0; i < S_STRIDE_NUM_READER_WRITER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_WRITER_1 + i, Dsl[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_READER_WRITER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_WRITER_1 + i, Dtb[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_READER_WRITER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_WRITER_1 + i, Dts[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, rimD);
#endif
#ifdef ENABLED_CHANNEL_READER_WRITER_1
    for (int i = 0; i < ENABLED_CHANNEL_READER_WRITER_1_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_WRITER_1 + i, ceD[i]);
    }
#endif

#ifdef READER_EXTENSION_0_CSR_BASE
    uint32_t cfgA = ((trA & 0x1) << 1) | (int4a & 0x1);
    csrw_ss(READER_EXTENSION_0_CSR_BASE, cfgA);
    csrw_ss(READER_EXTENSION_0_CSR_BASE + 1, shape);
#endif
#ifdef READER_EXTENSION_1_CSR_BASE
    uint32_t cfgB = ((trB & 0x1) << 1) | (int4b & 0x1);
    csrw_ss(READER_EXTENSION_1_CSR_BASE, cfgB);
    csrw_ss(READER_EXTENSION_1_CSR_BASE + 1, shape);
#endif
#ifdef READER_WRITER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE,
            (fp16en << 2) | (silu << 1) | qen);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, izp);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 2, mult);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 3, ozp);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 4, shift);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 5, (shape == 4) ? 1 : 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 6, 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 7, (shape == 4) ? 1 : 0);
#endif
#elif defined(BASE_PTR_READER_2_LOW) && defined(BASE_PTR_WRITER_0_LOW)
    (void)qen;
    (void)fp16en;
    (void)silu;
    csrw_ss(BASE_PTR_READER_0_LOW, versacore_compat_l1_addr(A_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_0 + i, Asl[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atb[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Ats[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, rimA);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0 + 0, ceA[0]);

    csrw_ss(BASE_PTR_READER_1_LOW, versacore_compat_l1_addr(B_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_1 + i, Bsl[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_1 + i, (i < 3) ? Btb[i] : 1);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_1 + i, (i < 3) ? Bts[i] : 0);
    }
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, rimB);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1 + 0, ceB[0]);

    csrw_ss(BASE_PTR_READER_2_LOW, versacore_compat_l1_addr(C_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_READER_2; i++) {
        csrw_ss(S_STRIDE_BASE_READER_2 + i, Csl[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_2; i++) {
        csrw_ss(T_BOUND_BASE_READER_2 + i, Ctb[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_2; i++) {
        csrw_ss(T_STRIDE_BASE_READER_2 + i, Cts[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, rimC);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2 + 0, 0);

    csrw_ss(BASE_PTR_WRITER_0_LOW, versacore_compat_l1_addr(D_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_0 + i, Dsl[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_0 + i, Dtb[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_0 + i, Dts[i]);
    }
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, rimD);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0 + 0, ceD[0]);

    csrw_ss(BASE_PTR_WRITER_1_LOW, versacore_compat_l1_addr(D_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_WRITER_1; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_1 + i, Dsl[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_WRITER_1; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_1 + i, Dtb[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_WRITER_1; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_1 + i, Dts[i]);
    }
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, rimD);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1 + 0, 0);

    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(izp, mult, ozp, shift);
    (void)trA;
    (void)trB;
    (void)int4a;
    (void)int4b;
    (void)ceC;
#else
    (void)A_addr; (void)Asl; (void)Atb; (void)Ats; (void)rimA; (void)trA; (void)ceA;
    (void)B_addr; (void)Bsl; (void)Btb; (void)Bts; (void)rimB; (void)trB; (void)ceB;
    (void)C_addr; (void)Csl; (void)Ctb; (void)Cts; (void)rimC; (void)ceC;
    (void)D_addr; (void)Dsl; (void)Dtb; (void)Dts; (void)rimD; (void)ceD;
    (void)shape; (void)qen; (void)shift; (void)mult; (void)izp; (void)ozp;
    (void)fp16en; (void)int4a; (void)int4b; (void)silu;
#endif
}

static inline void set_versacore_streamer_csr(
    uint32_t A_addr,  void* Asl,  void* Atb,  void* Ats,
    uint32_t rimA, uint32_t trA, void* ceA,
    uint32_t B_addr,  void* Bsl,  void* Btb,  void* Bts,
    uint32_t rimB, uint32_t trB, void* ceB,
    uint32_t C_addr,  void* Csl,  void* Ctb,  void* Cts,
    uint32_t rimC, void* ceC,
    uint32_t D_addr,  void* Dsl,  void* Dtb,  void* Dts,
    uint32_t rimD, void* ceD,
    int32_t shape, uint32_t qen, uint32_t shift, uint32_t mult,
    int32_t izp, int32_t ozp, int32_t fp16en, int32_t int4a, int32_t int4b) {
    int32_t* Asl_i = (int32_t*)Asl;
    int32_t* Atb_i = (int32_t*)Atb;
    int32_t* Ats_i = (int32_t*)Ats;
    int32_t* Bsl_i = (int32_t*)Bsl;
    int32_t* Btb_i = (int32_t*)Btb;
    int32_t* Bts_i = (int32_t*)Bts;
    int32_t* Csl_i = (int32_t*)Csl;
    int32_t* Ctb_i = (int32_t*)Ctb;
    int32_t* Cts_i = (int32_t*)Cts;
    int32_t* Dsl_i = (int32_t*)Dsl;
    int32_t* Dtb_i = (int32_t*)Dtb;
    int32_t* Dts_i = (int32_t*)Dts;
    int32_t* ceA_i = (int32_t*)ceA;
    int32_t* ceB_i = (int32_t*)ceB;
    int32_t* ceC_i = (int32_t*)ceC;
    int32_t* ceD_i = (int32_t*)ceD;

#if defined(BASE_PTR_READER_WRITER_0_LOW) && defined(BASE_PTR_READER_WRITER_1_LOW)
    csrw_ss(BASE_PTR_READER_0_LOW, versacore_compat_l1_addr(A_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_0 + i, Asl_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atb_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Ats_i[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, rimA);
#endif
#ifdef ENABLED_CHANNEL_READER_0
    for (int i = 0; i < ENABLED_CHANNEL_READER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_0 + i, ceA_i[i]);
    }
#endif

    csrw_ss(BASE_PTR_READER_1_LOW, versacore_compat_l1_addr(B_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_1 + i, Bsl_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_1 + i, Btb_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_1 + i, Bts_i[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, rimB);
#endif
#ifdef ENABLED_CHANNEL_READER_1
    for (int i = 0; i < ENABLED_CHANNEL_READER_1_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_1 + i, ceB_i[i]);
    }
#endif

    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, versacore_compat_l1_addr(C_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_READER_WRITER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_WRITER_0 + i, Csl_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_WRITER_0 + i, Ctb_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_WRITER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_WRITER_0 + i, Cts_i[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, rimC);
#endif
#ifdef ENABLED_CHANNEL_READER_WRITER_0
    for (int i = 0; i < ENABLED_CHANNEL_READER_WRITER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_WRITER_0 + i, ceC_i[i]);
    }
#endif

    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, versacore_compat_l1_addr(D_addr));
    #pragma GCC unroll 8
    for (int i = 0; i < S_STRIDE_NUM_READER_WRITER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_WRITER_1 + i, Dsl_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_BOUND_NUM_READER_WRITER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_WRITER_1 + i, Dtb_i[i]);
    }
    #pragma GCC unroll 8
    for (int i = 0; i < T_STRIDE_NUM_READER_WRITER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_WRITER_1 + i, Dts_i[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, rimD);
#endif
#ifdef ENABLED_CHANNEL_READER_WRITER_1
    for (int i = 0; i < ENABLED_CHANNEL_READER_WRITER_1_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_WRITER_1 + i, ceD_i[i]);
    }
#endif

#ifdef READER_EXTENSION_0_CSR_BASE
    uint32_t cfgA = ((trA & 0x1) << 1) | (int4a & 0x1);
    csrw_ss(READER_EXTENSION_0_CSR_BASE, cfgA);
    csrw_ss(READER_EXTENSION_0_CSR_BASE + 1, shape);
#endif
#ifdef READER_EXTENSION_1_CSR_BASE
    uint32_t cfgB = ((trB & 0x1) << 1) | (int4b & 0x1);
    csrw_ss(READER_EXTENSION_1_CSR_BASE, cfgB);
    csrw_ss(READER_EXTENSION_1_CSR_BASE + 1, shape);
#endif
#ifdef READER_WRITER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE, (fp16en << 1) | qen);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, izp);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 2, mult);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 3, ozp);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 4, shift);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 5, shape);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 6, shape);
#endif
#else
    __set_versacore_streamer_csr_common(
        A_addr, Asl, Atb, Ats, rimA, trA, ceA,
        B_addr, Bsl, Btb, Bts, rimB, trB, ceB,
        C_addr, Csl, Ctb, Cts, rimC, ceC,
        D_addr, Dsl, Dtb, Dts, rimD, ceD,
        shape, qen, shift, mult, izp, ozp, fp16en, int4a, int4b, 0);
#endif
}

static inline void set_versacore_streamer_csr_with_silu(
    uint32_t A_addr,  void* Asl,  void* Atb,  void* Ats,
    uint32_t rimA, uint32_t trA, void* ceA,
    uint32_t B_addr,  void* Bsl,  void* Btb,  void* Bts,
    uint32_t rimB, uint32_t trB, void* ceB,
    uint32_t C_addr,  void* Csl,  void* Ctb,  void* Cts,
    uint32_t rimC, void* ceC,
    uint32_t D_addr,  void* Dsl,  void* Dtb,  void* Dts,
    uint32_t rimD, void* ceD,
    int32_t shape, uint32_t qen, uint32_t shift, uint32_t mult,
    int32_t izp, int32_t ozp, int32_t fp16en, int32_t int4a, int32_t int4b,
    int32_t silu) {
    __set_versacore_streamer_csr_common(
        A_addr, Asl, Atb, Ats, rimA, trA, ceA,
        B_addr, Bsl, Btb, Bts, rimB, trB, ceB,
        C_addr, Csl, Ctb, Cts, rimC, ceC,
        D_addr, Dsl, Dtb, Dts, rimD, ceD,
        shape, qen, shift, mult, izp, ozp, fp16en, int4a, int4b, silu);
}

static inline void set_versacore_csr(uint32_t take_in_new_c,
                                     uint32_t a_b_input_times_one_output,
                                     uint32_t output_times,
                                     uint32_t subtractions,
                                     uint32_t array_shape,
                                     uint32_t data_type) {
    csrw_ss(OVERWRITE_ACCUM, take_in_new_c);
    csrw_ss(ACCUM_BOUND, a_b_input_times_one_output);
    csrw_ss(OUTPUT_BOUND, output_times);
    csrw_ss(SUBTRACTIONS, subtractions);
    csrw_ss(ARRAY_SHAPE_CFG, array_shape);
    csrw_ss(DATA_TYPE_CFG, data_type);
}

static inline void start_streamer()               { start_dual_vc_streamer(); }
static inline void start_versacore()              { start_dual_vc(); }
static inline void start_versacore_and_streamer() { start_dual_vc_and_streamer(); }

static inline void wait_versacore_and_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(VERSACORE_START_CSR, 0);
    while (csrr_ss(VERSACORE_BUSY)) {}
    while (csrr_ss(STREAMER_BUSY_CSR)) {}
}

static inline void wait_versacore() {
    csrw_ss(VERSACORE_START_CSR, 0);
    csrw_ss(VERSACORE_START_CSR, 0);
    while (csrr_ss(VERSACORE_BUSY)) {}
}

static inline void wait_versacore_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    while (csrr_ss(STREAMER_BUSY_CSR)) {}
}

static inline void set_minimal_streamer_cfg(uint32_t A_addr, uint32_t B_addr,
                                            uint32_t C_addr, uint32_t D_addr) {
#if defined(BASE_PTR_READER_WRITER_0_LOW) && defined(BASE_PTR_READER_WRITER_1_LOW)
    csrw_ss(BASE_PTR_READER_0_LOW, versacore_compat_l1_addr(A_addr));
    csrw_ss(BASE_PTR_READER_1_LOW, versacore_compat_l1_addr(B_addr));
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, versacore_compat_l1_addr(C_addr));
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, versacore_compat_l1_addr(D_addr));
#else
    csrw_ss(BASE_PTR_READER_0_LOW, versacore_compat_l1_addr(A_addr));
    csrw_ss(BASE_PTR_READER_1_LOW, versacore_compat_l1_addr(B_addr));
    csrw_ss(BASE_PTR_READER_2_LOW, versacore_compat_l1_addr(C_addr));
    csrw_ss(BASE_PTR_WRITER_0_LOW, versacore_compat_l1_addr(D_addr));
    csrw_ss(BASE_PTR_WRITER_1_LOW, versacore_compat_l1_addr(D_addr));
    csrw_ss(ENABLED_CHANNEL_READER_2 + 0, 0);
    csrw_ss(ENABLED_CHANNEL_WRITER_1 + 0, 0);
#endif
}

static inline void set_minimal_streamer_cfg_with_silu(uint32_t A_addr,
                                                       uint32_t B_addr,
                                                       uint32_t C_addr,
                                                       uint32_t D_addr,
                                                       uint32_t silu) {
    set_minimal_streamer_cfg(A_addr, B_addr, C_addr, D_addr);
#ifdef READER_WRITER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE, (silu << 1));
#else
    (void)silu;
#endif
}

static inline uint32_t read_versacore_streamer_perf_counter() {
    uint32_t perf_counter = csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
    return perf_counter;
}

static inline uint32_t read_versacore_perf_counter() {
    uint32_t perf_counter = csrr_ss(VERSACORE_PERFORMANCE_COUNTER);
    return perf_counter;
}

static inline uint32_t check_versacore_result_D32(void* output_raw,
                                                   void* output_golden_raw,
                                                   int32_t data_length,
                                                   bool banked_data_layout) {
    int8_t* output = (int8_t*)output_raw;
    int8_t* output_golden = (int8_t*)output_golden_raw;
    uint32_t err = 0;

    if (banked_data_layout) {
        for (int i = 0; i < data_length / 16; i += 1) {
            for (int j = 0; j < 16; j++) {
                if (*(output + i * (256 / 4) + j) != output_golden[i * 16 + j]) {
                    err++;
                }
            }
        }
    } else {
        for (int i = 0; i < data_length; i++) {
            if (output[i] != output_golden[i]) {
                err++;
                printf("Unequals. output[%d] = %d, output_golden[%d] = %d\n",
                       i, output[i], i, output_golden[i]);
            }
        }
    }

    return err;
}
