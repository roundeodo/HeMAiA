// Local dual-VersaCore and dynamic MoE kernels ported from the pre-rebase monolithic libsnaxkernel.
#pragma once

#include "../macros.h"
#include <snax_versacore_lib.h>
#include <snax_xdma_lib.h>

#define MOE_DUAL_VC_BANK_WIDTH 64u

#define MOE_DUAL_VC_MESH_ROW_0 8u
#define MOE_DUAL_VC_TILE_SIZE_0 8u
#define MOE_DUAL_VC_MESH_COL_0 4u

#define MOE_DUAL_VC_MESH_ROW_1 4u
#define MOE_DUAL_VC_TILE_SIZE_1 8u    // k8_8x4_4lane: N-expand, tileSize fixed=8
#define MOE_DUAL_VC_MESH_COL_1 8u    // k8_8x4_4lane: S1 meshCol=8

#define MOE_DUAL_VC_MESH_ROW_2 2u
#define MOE_DUAL_VC_TILE_SIZE_2 8u    // k8_8x4_4lane: N-expand, tileSize fixed=8
#define MOE_DUAL_VC_MESH_COL_2 16u   // k8_8x4_4lane: S2 meshCol=16

#define MOE_DUAL_VC_CSR_ADDR_BASE          (STREAMER_WRITER1_BUSY_CSR + 1)
#define MOE_DUAL_VC_OVERWRITE_ACCUM        (MOE_DUAL_VC_CSR_ADDR_BASE)
#define MOE_DUAL_VC_ACCUM_BOUND            (MOE_DUAL_VC_OVERWRITE_ACCUM + 1)
#define MOE_DUAL_VC_OUTPUT_BOUND           (MOE_DUAL_VC_ACCUM_BOUND + 1)
#define MOE_DUAL_VC_SUBTRACTIONS           (MOE_DUAL_VC_OUTPUT_BOUND + 1)
#define MOE_DUAL_VC_ARRAY_SHAPE_CFG        (MOE_DUAL_VC_SUBTRACTIONS + 1)
#define MOE_DUAL_VC_DATA_TYPE_CFG          (MOE_DUAL_VC_ARRAY_SHAPE_CFG + 1)
#define MOE_DUAL_VC_MODE                   (MOE_DUAL_VC_DATA_TYPE_CFG + 1)
#define MOE_DUAL_VC_RESCALE0_INPUT_ZP      (MOE_DUAL_VC_MODE + 1)
#define MOE_DUAL_VC_RESCALE0_MULTIPLIER    (MOE_DUAL_VC_RESCALE0_INPUT_ZP + 1)
#define MOE_DUAL_VC_RESCALE0_OUTPUT_ZP     (MOE_DUAL_VC_RESCALE0_MULTIPLIER + 1)
#define MOE_DUAL_VC_RESCALE0_SHIFT         (MOE_DUAL_VC_RESCALE0_OUTPUT_ZP + 1)
#define MOE_DUAL_VC_RESCALE1_INPUT_ZP      (MOE_DUAL_VC_RESCALE0_SHIFT + 1)
#define MOE_DUAL_VC_RESCALE1_MULTIPLIER    (MOE_DUAL_VC_RESCALE1_INPUT_ZP + 1)
#define MOE_DUAL_VC_RESCALE1_OUTPUT_ZP     (MOE_DUAL_VC_RESCALE1_MULTIPLIER + 1)
#define MOE_DUAL_VC_RESCALE1_SHIFT         (MOE_DUAL_VC_RESCALE1_OUTPUT_ZP + 1)
#define MOE_DUAL_VC_RESCALE_MUL_INPUT_ZP   (MOE_DUAL_VC_RESCALE1_SHIFT + 1)
#define MOE_DUAL_VC_RESCALE_MUL_MULTIPLIER (MOE_DUAL_VC_RESCALE_MUL_INPUT_ZP + 1)
#define MOE_DUAL_VC_RESCALE_MUL_OUTPUT_ZP  (MOE_DUAL_VC_RESCALE_MUL_MULTIPLIER + 1)
#define MOE_DUAL_VC_RESCALE_MUL_SHIFT      (MOE_DUAL_VC_RESCALE_MUL_OUTPUT_ZP + 1)
#define MOE_DUAL_VC_NUM_RW_CSR             23u  // snax_num_rw_csr for k8_8x4_4lane
#define MOE_DUAL_VC_START                  (MOE_DUAL_VC_CSR_ADDR_BASE + MOE_DUAL_VC_NUM_RW_CSR - 1u)
#define MOE_DUAL_VC_BUSY                   (MOE_DUAL_VC_CSR_ADDR_BASE + MOE_DUAL_VC_NUM_RW_CSR)
// Channel enables: shape-dependent (N-direction expand, k8_8x4_4lane hw)
// A reader: S0=16ch, S1=8ch, S2=4ch
#define MOE_DUAL_VC_CHAN_EN_A(s) ((s)==0u ? 0x0000FFFFu : (s)==1u ? 0x000000FFu : 0x0000000Fu)
// B reader: S0=2ch, S1=4ch, S2=8ch
#define MOE_DUAL_VC_CHAN_EN_B(s) ((s)==0u ? 0x00000003u : (s)==1u ? 0x0000000Fu : 0x000000FFu)
// D writer: 4-lane postproc, single-port output = always 1 channel
#define MOE_DUAL_VC_CHAN_EN_D              0x00000001u

#ifndef BINGO_DEBUG_LEVEL
#define BINGO_DEBUG_LEVEL 0
#endif

// Zero a TCDM byte range on the current core.  Use this only for bytes that
// are semantically padding or intentionally invalid; clearing valid output
// payload would hide writer coverage bugs.
// base/bytes must be 8-byte aligned.
__attribute__((always_inline)) static inline void
__moe_zero_tcdm(uint32_t base, uint32_t bytes)
{
    volatile uint64_t *p = (volatile uint64_t *)(uintptr_t)base;
    uint32_t n = bytes >> 3;
    for (uint32_t i = 0; i < n; i++) {
        p[i] = 0;
    }
}

static inline void moe_set_minimal_streamer_cfg_with_silu(
    uint32_t A_addr, uint32_t B_addr, uint32_t C_addr,
    uint32_t D_addr, uint32_t silu)
{
    set_minimal_streamer_cfg(A_addr, B_addr, C_addr, D_addr);
#ifdef READER_WRITER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE, (silu << 1));
#else
    (void)silu;
#endif
}

static inline void moe_set_dual_versacore_streamer_csr(
    uint32_t A_addr,  int32_t *Aslstride,  int32_t *Atlbound,  int32_t *Atlstride,
    int32_t remap_A,  int32_t *chan_en_A,
    uint32_t B0_addr, int32_t *B0slstride, int32_t *B0tlbound, int32_t *B0tlstride,
    int32_t remap_B0, int32_t *chan_en_B0,
    uint32_t B1_addr, int32_t *B1slstride, int32_t *B1tlbound, int32_t *B1tlstride,
    int32_t remap_B1, int32_t *chan_en_B1,
    uint32_t D0_addr, int32_t *D0slstride, int32_t *D0tlbound, int32_t *D0tlstride,
    int32_t remap_D0, int32_t *chan_en_D0,
    uint32_t D1_addr, int32_t *D1slstride, int32_t *D1tlbound, int32_t *D1tlstride,
    int32_t remap_D1, int32_t *chan_en_D1)
{
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) csrw_ss(S_STRIDE_BASE_READER_0 + i, Aslstride[i]);
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) csrw_ss(T_BOUND_BASE_READER_0 + i, Atlbound[i]);
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) csrw_ss(T_STRIDE_BASE_READER_0 + i, Atlstride[i]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, remap_A);
#else
    (void)remap_A;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0 + 0, chan_en_A[0]);

    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) csrw_ss(S_STRIDE_BASE_READER_1 + i, B0slstride[i]);
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) csrw_ss(T_BOUND_BASE_READER_1 + i, B0tlbound[i]);
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) csrw_ss(T_STRIDE_BASE_READER_1 + i, B0tlstride[i]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, remap_B0);
#else
    (void)remap_B0;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1 + 0, chan_en_B0[0]);

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    for (int i = 0; i < S_STRIDE_NUM_READER_2; i++) csrw_ss(S_STRIDE_BASE_READER_2 + i, B1slstride[i]);
    for (int i = 0; i < T_BOUND_NUM_READER_2; i++) csrw_ss(T_BOUND_BASE_READER_2 + i, B1tlbound[i]);
    for (int i = 0; i < T_STRIDE_NUM_READER_2; i++) csrw_ss(T_STRIDE_BASE_READER_2 + i, B1tlstride[i]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, remap_B1);
#else
    (void)remap_B1;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2 + 0, chan_en_B1[0]);

    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0 + 0, D0slstride[0]);
    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) csrw_ss(T_BOUND_BASE_WRITER_0 + i, D0tlbound[i]);
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) csrw_ss(T_STRIDE_BASE_WRITER_0 + i, D0tlstride[i]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, remap_D0);
#else
    (void)remap_D0;
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0 + 0, chan_en_D0[0]);

    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1 + 0, D1slstride[0]);
    for (int i = 0; i < T_BOUND_NUM_WRITER_1; i++) csrw_ss(T_BOUND_BASE_WRITER_1 + i, D1tlbound[i]);
    for (int i = 0; i < T_STRIDE_NUM_WRITER_1; i++) csrw_ss(T_STRIDE_BASE_WRITER_1 + i, D1tlstride[i]);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, remap_D1);
#else
    (void)remap_D1;
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1 + 0, chan_en_D1[0]);
}

static inline void moe_set_dual_versacore_csr(
    uint32_t take_in_new_c, uint32_t a_b_input_times_one_output,
    uint32_t output_times, uint32_t subtractions,
    uint32_t array_shape, uint32_t data_type)
{
    csrw_ss(MOE_DUAL_VC_OVERWRITE_ACCUM, take_in_new_c);
    csrw_ss(MOE_DUAL_VC_ACCUM_BOUND, a_b_input_times_one_output);
    csrw_ss(MOE_DUAL_VC_OUTPUT_BOUND, output_times);
    csrw_ss(MOE_DUAL_VC_SUBTRACTIONS, subtractions);
    csrw_ss(MOE_DUAL_VC_ARRAY_SHAPE_CFG, array_shape);
    csrw_ss(MOE_DUAL_VC_DATA_TYPE_CFG, data_type);
}

static inline void moe_set_dual_versacore_mode(uint32_t mode)
{
    csrw_ss(MOE_DUAL_VC_MODE, mode);
}

static inline void moe_set_dual_versacore_rescale0(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift)
{
    csrw_ss(MOE_DUAL_VC_RESCALE0_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE0_MULTIPLIER, multiplier);
    csrw_ss(MOE_DUAL_VC_RESCALE0_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE0_SHIFT, shift);
}

static inline void moe_set_dual_versacore_rescale1(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift)
{
    csrw_ss(MOE_DUAL_VC_RESCALE1_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE1_MULTIPLIER, multiplier);
    csrw_ss(MOE_DUAL_VC_RESCALE1_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE1_SHIFT, shift);
}

static inline void moe_set_dual_versacore_rescale_mul(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift)
{
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_MULTIPLIER, multiplier);
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_SHIFT, shift);
}

static inline void moe_start_dual_vc_and_streamer(void)
{
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(MOE_DUAL_VC_START, 1);
}

static inline void moe_wait_dual_vc_and_streamer(void)
{
    csrw_ss(MOE_DUAL_VC_START, 0);
    csrw_ss(MOE_DUAL_VC_START, 0);
    while (csrr_ss(MOE_DUAL_VC_BUSY)) {}
    while (csrr_ss(STREAMER_WRITER_BUSY_CSR) ||
           csrr_ss(STREAMER_WRITER1_BUSY_CSR)) {}
}

// gemm_minimal_silu: like gemm_minimal but also updates the SiLU extension CSR.
// Saves ~45 CSR writes compared to gemm_full when shape (M,K,N,strides) is unchanged.
// Use when: same shape as a preceding gemm_full, only addresses + silu_enable differ.
// arg[0]=A_addr, arg[1]=B_addr, arg[2]=C_addr, arg[3]=D_addr, arg[4]=silu_enable
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_minimal_silu(void *arg)
{
    if (snrt_cluster_core_idx() != 0){
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM minimal_silu should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr    = ((uint32_t *)arg)[0];
    uint32_t B_addr    = ((uint32_t *)arg)[1];
    uint32_t C_addr    = ((uint32_t *)arg)[2];
    uint32_t D_addr    = ((uint32_t *)arg)[3];
    uint32_t silu_enable = ((uint32_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_START);
    moe_set_minimal_streamer_cfg_with_silu(A_addr, B_addr, C_addr, D_addr, silu_enable);
    start_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_START);
    wait_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_END);
    VERSACORE_DEBUG_PRINT("Bingo GEMM Minimal SiLU Kernel Compute Done!\r\n");
    return BINGO_RET_SUCC;
}

// ============================================================
// Dual-VersaCore GEMM kernel (Mode 1: A@B0/B1 -> D0/D1)
// INT16 A x INT4 packed B -> INT16 D
// Args (uint32_t array, 11 fields):
//   [0] A_addr  [1] B0_addr  [2] B1_addr  [3] D0_addr  [4] D1_addr
//   [5] M  [6] K  [7] N
//   [8] array_shape  [9] rescale_mult  [10] rescale_shift
// ============================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_gemm_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: dual_vc_gemm_full must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr      = ((uint32_t *)arg)[0];
    uint32_t B0_addr     = ((uint32_t *)arg)[1];
    uint32_t B1_addr     = ((uint32_t *)arg)[2];  // VC1 weight (right N columns)
    uint32_t D0_addr     = ((uint32_t *)arg)[3];
    uint32_t D1_addr     = ((uint32_t *)arg)[4];  // VC1 output (right N columns)
    uint32_t M           = ((uint32_t *)arg)[5];
    uint32_t K           = ((uint32_t *)arg)[6];
    uint32_t N           = ((uint32_t *)arg)[7];
    uint32_t array_shape = ((uint32_t *)arg)[8];
    uint32_t rscl_mult   = ((uint32_t *)arg)[9];
    uint32_t rscl_shift  = ((uint32_t *)arg)[10];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t meshRow, tileSize, meshCol;
    switch (array_shape) {
    case 0: meshRow = MOE_DUAL_VC_MESH_ROW_0; tileSize = MOE_DUAL_VC_TILE_SIZE_0; meshCol = MOE_DUAL_VC_MESH_COL_0; break;
    case 1: meshRow = MOE_DUAL_VC_MESH_ROW_1; tileSize = MOE_DUAL_VC_TILE_SIZE_1; meshCol = MOE_DUAL_VC_MESH_COL_1; break;
    case 2: meshRow = MOE_DUAL_VC_MESH_ROW_2; tileSize = MOE_DUAL_VC_TILE_SIZE_2; meshCol = MOE_DUAL_VC_MESH_COL_2; break;
    default:
        printf_safe("[C%d c%d]: dual_vc_gemm_full: invalid array_shape %d!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), array_shape);
        return BINGO_RET_FAIL;
    }

    // A: INT16 token-contiguous rows.  multi_cluster_MoE routes the same
    // input_A buffer used by the dynamic path: each token has K*tileSize*2
    // payload bytes followed by 32B L3 padding.  The GEMM reader skips the
    // padding by using a padded row stride while only streaming payload bytes.
    uint32_t a_row_stride = K * tileSize * 2u + 32u;
    int32_t Asl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)a_row_stride };
    int32_t Atb[6]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1, 1, 1 };
    int32_t Ats[6]       = { (int32_t)(tileSize * 2), 0,
                              (int32_t)(meshRow * a_row_stride), 0, 0, 0 };
    int32_t chan_en_A[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape) };

    // B0: INT4 packed, 4-dim temporal, 2-dim spatial.  Use the same canonical
    // mode-1 B reader layout as the L15/reference kernels: per K tile the
    // stream advances by 16B, while the second spatial stride covers the full
    // K section.  A zero second spatial stride aliases active B channels and
    // produces router scores that are self-consistent for TopK but not equal
    // to the datagen GEMM.
    uint32_t n_b_chan      = (array_shape == 0u) ? 2u : (array_shape == 1u) ? 4u : 8u;
    int32_t B_stream_bytes = (int32_t)(n_b_chan * (MOE_DUAL_VC_BANK_WIDTH / 8));
    uint32_t b_k_section   = K * tileSize * 2u;
    int32_t B0sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B0tb[4]        = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B0ts[4]        = { (int32_t)(tileSize * 2u), (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B0[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // B1: INT4 packed, same canonical mode-1 layout as B0.
    int32_t B1sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B1tb[4]        = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B1ts[4]        = { (int32_t)(tileSize * 2u), (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B1[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // D0: INT16, 4-dim temporal (VC0 output, 4-lane postproc: 1ch, 8 bytes/beat)
    // NOTE: For L15 layout mode-1 down-proj, use __snax_bingo_kernel_dual_vc_l15_moe_full.
    int32_t D0sl[1]       = { 8 };
    int32_t D0tb[4]       = { 8, (int32_t)N, (int32_t)M, 1 };
    int32_t D0ts[4]       = { 8, 64, (int32_t)(N * 64), 0 };
    int32_t chan_en_D0[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_D };

    // D1: INT16, 4-dim temporal (VC1 output, 4-lane postproc: 1ch, 8 bytes/beat)
    int32_t D1sl[1]       = { 8 };
    int32_t D1tb[4]       = { 8, (int32_t)N, (int32_t)M, 1 };
    int32_t D1ts[4]       = { 8, 64, (int32_t)(N * 64), 0 };
    int32_t chan_en_D1[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_D };

#if BINGO_DEBUG_LEVEL >= 1
    if (M == 4u && K == 128u && N == 1u && array_shape == 0u) {
        volatile int16_t *A_dbg = (volatile int16_t *)(uintptr_t)A_addr;
        volatile uint8_t *B0_dbg = (volatile uint8_t *)(uintptr_t)B0_addr;
        volatile uint8_t *B1_dbg = (volatile uint8_t *)(uintptr_t)B1_addr;
        printf_safe("[ROUTER_GEMM_DBG] args A=0x%08x B0=0x%08x B1=0x%08x D0=0x%08x D1=0x%08x M=%u K=%u N=%u shape=%u\r\n",
                    A_addr, B0_addr, B1_addr, D0_addr, D1_addr, M, K, N, array_shape);
        printf_safe("[ROUTER_GEMM_DBG] D_layout D0_row_stride=64 D1_minus_D0=%u D_tile_bytes=%u\r\n",
                    (uint32_t)(D1_addr - D0_addr), (uint32_t)(M * N * 64u));
        printf_safe("[ROUTER_GEMM_DBG] A16[0..7]=%d %d %d %d %d %d %d %d\r\n",
                    A_dbg[0], A_dbg[1], A_dbg[2], A_dbg[3],
                    A_dbg[4], A_dbg[5], A_dbg[6], A_dbg[7]);
        printf_safe("[ROUTER_GEMM_DBG] B0u8[0..15]=%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\r\n",
                    B0_dbg[0], B0_dbg[1], B0_dbg[2], B0_dbg[3],
                    B0_dbg[4], B0_dbg[5], B0_dbg[6], B0_dbg[7],
                    B0_dbg[8], B0_dbg[9], B0_dbg[10], B0_dbg[11],
                    B0_dbg[12], B0_dbg[13], B0_dbg[14], B0_dbg[15]);
        printf_safe("[ROUTER_GEMM_DBG] B1u8[0..15]=%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\r\n",
                    B1_dbg[0], B1_dbg[1], B1_dbg[2], B1_dbg[3],
                    B1_dbg[4], B1_dbg[5], B1_dbg[6], B1_dbg[7],
                    B1_dbg[8], B1_dbg[9], B1_dbg[10], B1_dbg[11],
                    B1_dbg[12], B1_dbg[13], B1_dbg[14], B1_dbg[15]);
    }
#endif

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    moe_set_dual_versacore_streamer_csr(
        A_addr,  Asl,  Atb,  Ats,  0, chan_en_A,
        B0_addr, B0sl, B0tb, B0ts, 0, chan_en_B0,
        B1_addr, B1sl, B1tb, B1ts, 0, chan_en_B1,
        D0_addr, D0sl, D0tb, D0ts, 0, chan_en_D0,
        D1_addr, D1sl, D1tb, D1ts, 0, chan_en_D1);

    moe_set_dual_versacore_mode(1);   // Mode 1 = GEMM

    moe_set_dual_versacore_csr(
        1,       // take_in_new_c = start fresh
        K,       // accum_bound
        N * M,   // output_times
        0,       // subtractions
        array_shape,
        0);      // data_type

    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
#if BINGO_DEBUG_LEVEL >= 1
    if (M == 4u && K == 128u && N == 1u && array_shape == 0u) {
        volatile int16_t *D0_dbg = (volatile int16_t *)(uintptr_t)D0_addr;
        volatile int16_t *D1_dbg = (volatile int16_t *)(uintptr_t)D1_addr;
        printf_safe("[ROUTER_GEMM_DBG] D0_16[0..7]=%d %d %d %d %d %d %d %d\r\n",
                    D0_dbg[0], D0_dbg[1], D0_dbg[2], D0_dbg[3],
                    D0_dbg[4], D0_dbg[5], D0_dbg[6], D0_dbg[7]);
        printf_safe("[ROUTER_GEMM_DBG] D1_16[0..7]=%d %d %d %d %d %d %d %d\r\n",
                    D1_dbg[0], D1_dbg[1], D1_dbg[2], D1_dbg[3],
                    D1_dbg[4], D1_dbg[5], D1_dbg[6], D1_dbg[7]);
    }
#endif
    return BINGO_RET_SUCC;
}

// ============================================================
// Dual-VersaCore SwiGLU kernel (Mode 0: gate+up -> SiLU -> elemMul -> D0/D1)
// INT16 A x INT4 packed B_gate, B_up -> INT16 D0, D1
// Args (uint32_t array, 11 fields):
//   [0] A_addr  [1] B_gate_addr  [2] B_up_addr  [3] D0_addr  [4] D1_addr
//   [5] M  [6] K  [7] N
//   [8] array_shape  [9] rescale_mult  [10] rescale_shift
// ============================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_swiglu_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: dual_vc_swiglu_full must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr      = ((uint32_t *)arg)[0];
    uint32_t Bg_addr     = ((uint32_t *)arg)[1];
    uint32_t Bu_addr     = ((uint32_t *)arg)[2];
    uint32_t D0_addr     = ((uint32_t *)arg)[3];
    uint32_t D1_addr     = ((uint32_t *)arg)[4];
    uint32_t M           = ((uint32_t *)arg)[5];
    uint32_t K           = ((uint32_t *)arg)[6];
    uint32_t N           = ((uint32_t *)arg)[7];
    uint32_t array_shape = ((uint32_t *)arg)[8];
    uint32_t rscl_mult           = ((uint32_t *)arg)[9];
    uint32_t rscl_shift          = ((uint32_t *)arg)[10];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    uint32_t meshRow, tileSize, meshCol;
    switch (array_shape) {
    case 0: meshRow = MOE_DUAL_VC_MESH_ROW_0; tileSize = MOE_DUAL_VC_TILE_SIZE_0; meshCol = MOE_DUAL_VC_MESH_COL_0; break;
    case 1: meshRow = MOE_DUAL_VC_MESH_ROW_1; tileSize = MOE_DUAL_VC_TILE_SIZE_1; meshCol = MOE_DUAL_VC_MESH_COL_1; break;
    case 2: meshRow = MOE_DUAL_VC_MESH_ROW_2; tileSize = MOE_DUAL_VC_TILE_SIZE_2; meshCol = MOE_DUAL_VC_MESH_COL_2; break;
    default:
        printf_safe("[C%d c%d]: dual_vc_swiglu_full: invalid array_shape %d!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), array_shape);
        return BINGO_RET_FAIL;
    }

    // A: INT16, 6-dim temporal (shared for gate and up), 2-dim spatial [2, 8] (hw bounds).
    // Dynamic MoE: gather stores tokens in L1 without padding (stride = K*tileSize*2 = 2048).
    // Asl[0]=8 (inner stride, 1 bank); Asl[1]=a_row_stride (outer stride = packed token row).
    // Ats[0] = tileSize*2 (K stride: advance one K-tile within token row).
    // Ats[2] = meshRow * a_row_stride (M stride: skip meshRow packed rows).
    uint32_t a_row_stride = (uint32_t)K * (uint32_t)tileSize * 2u;  /* = 2048, packed (no pad) */
    int32_t Asl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)a_row_stride };
    int32_t Atb[6]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1, 1, 1 };
    int32_t Ats[6]       = { (int32_t)(tileSize * 2), 0,
                              (int32_t)(meshRow * a_row_stride), 0, 0, 0 };
    int32_t chan_en_A[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape) };

    // B0 (gate): INT4 packed, 4-dim temporal, 2-dim spatial (hw req.)
    // L15 B canonical layout: [N_tiles_S0][K_tiles][16 bytes].
    // B_sl[1] = K*16 = 2048 (K-section stride between outer spatial j-groups; fixed).
    // B_ts[0] = 16 (fixed k-stride for all shapes; NOT B_stream_bytes).
    // B_ts[1] = K * B_stream_bytes (correct N-tile stride, same for all shapes).
    uint32_t n_b_chan      = (array_shape == 0u) ? 2u : (array_shape == 1u) ? 4u : 8u;
    int32_t B_stream_bytes = (int32_t)(n_b_chan * (MOE_DUAL_VC_BANK_WIDTH / 8));
    uint32_t b_k_section   = (uint32_t)K * (uint32_t)tileSize * 2u;  /* = K*16 = 2048 */
    int32_t B0sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B0tb[4]        = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B0ts[4]        = { (int32_t)(tileSize * 2u), (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B0[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // B1 (up): INT4 packed, same layout as B0, 2-dim spatial (hw req.)
    int32_t B1sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B1tb[4]        = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B1ts[4]        = { (int32_t)(tileSize * 2u), (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B1[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // D0: INT16, 4-dim temporal (4-lane postproc: 1ch, 4×int16/beat=8 bytes/beat)
    // 8 beats per N-tile (constant for all shapes: meshRow*meshCol/4=8)
    int32_t D0sl[1]       = { 8 };
    int32_t D0tb[4]       = { 8, (int32_t)N, (int32_t)M, 1 };
    int32_t D0ts[4]       = { 8, 64, (int32_t)(N * 64), 0 };
    int32_t chan_en_D0[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_D };

    // D1: NOT used in Mode 0 (SwiGLU). The hardware NEVER sends data to Writer 1
    // in Mode 0 (oa1_in_valid = 1'b0 in DualVersaCoreSwigluGen). We must zero
    // the temporal bounds and channel enable so Streamer Writer 1 completes
    // immediately (0 iterations) instead of waiting for data that never arrives.
    int32_t D1sl[1]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8) };
    int32_t D1tb[4]       = { 0, 0, 0, 0 };   // 0 iterations → Writer 1 done instantly
    int32_t D1ts[4]       = { 0, 0, 0, 0 };   // irrelevant (no iterations)
    int32_t chan_en_D1[1] = { 0 };             // no channels in Mode 0

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    moe_set_dual_versacore_streamer_csr(
        A_addr,  Asl,  Atb,  Ats,  0, chan_en_A,
        Bg_addr, B0sl, B0tb, B0ts, 0, chan_en_B0,
        Bu_addr, B1sl, B1tb, B1ts, 0, chan_en_B1,
        D0_addr, D0sl, D0tb, D0ts, 0, chan_en_D0,
        D1_addr, D1sl, D1tb, D1ts, 0, chan_en_D1);

    moe_set_dual_versacore_mode(0);   // Mode 0 = SwiGLU

    moe_set_dual_versacore_csr(
        1,       // take_in_new_c
        K,       // accum_bound
        N * M,   // output_times
        0,       // subtractions
        array_shape,
        0);      // data_type

    // Rescale for all 3 paths (gate, up, and element-wise multiply)
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale_mul(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);

    // NOTE (k8_8x4_4lane): Bank replication removed.
    // 4-lane postproc D writer uses 1 channel with sequential bank rotation;
    // Mode-1 A reader directly reads the Mode-0 D0 output without any copy.
    return BINGO_RET_SUCC;
}
/* ── Schedule verification debug: controlled by BINGO_DEBUG_LEVEL compile flag ──
 * Pass DEBUG_LEVEL=1 to make (→ -DBINGO_DEBUG_LEVEL=1) to enable slot prints. ── */
#define MOE_DYN_DEBUG_SCHED_VERIFY (BINGO_DEBUG_LEVEL >= 1)

/* ── ctrl-word field extractors ─────────────────────────────────────────────
 * Bit layout written by __host_moe_program_task_arg each scheduling round:
 *   bit  0:       active            (1 = this slot is live)
 *   bit  1:       skip_s1           ditto for skip_s3/s2/s4 below
 *   bit  2:       skip_s3
 *   bit  3:       skip_s2
 *   bit  4:       skip_s4
 *   bits [6:5]:   shape_s1          (0=M8, 1=M4, 2=M2)
 *   bits [8:7]:   shape_s3
 *   bits [10:9]:  dma_s1            (0=NONE, 1=IDMA, 2=XDMA, 3=BOTH)
 *   bits [12:11]: dma_s3
 *   bit  13:      runtime_cluster_idx  (0=C2, 1=C3)
 *   bits [18:14]: slot_id           (0-31)
 * ──────────────────────────────────────────────────────────────────────────── */
#define MOE_DYN_CTRL_ACTIVE(c)   ((c) & 1u)
#define MOE_DYN_CTRL_SKIP_S1(c)  (((c) >> 1u) & 1u)
#define MOE_DYN_CTRL_SKIP_S3(c)  (((c) >> 2u) & 1u)
#define MOE_DYN_CTRL_SKIP_S2(c)  (((c) >> 3u) & 1u)
#define MOE_DYN_CTRL_SKIP_S4(c)  (((c) >> 4u) & 1u)
#define MOE_DYN_CTRL_SHAPE_S1(c) (((c) >> 5u) & 3u)
#define MOE_DYN_CTRL_SHAPE_S3(c) (((c) >> 7u) & 3u)
#define MOE_DYN_CTRL_DMA_S1(c)   (((c) >> 9u) & 3u)
#define MOE_DYN_CTRL_DMA_S3(c)   (((c) >> 11u) & 3u)
#define MOE_DYN_CTRL_CLUSTER(c)  (((c) >> 13u) & 1u)
#define MOE_DYN_CTRL_SLOT_ID(c)  (((c) >> 14u) & 31u) /* bits [18:14]: 5-bit slot_id (0..31, supports 64 experts/2) */
/* ── dma_slot_vd field extractors (3 bits per slot: valid | dma[1:0]) ───────
 * For slot i: bit[i*3] = valid, bits[i*3+2:i*3+1] = dma binding
 * ──────────────────────────────────────────────────────────────────────────── */
#define MOE_DYN_VD_VALID(vd, s)  (((vd) >> ((s)*3u)) & 1u)
#define MOE_DYN_VD_DMA(vd, s)    (((vd) >> ((s)*3u + 1u)) & 3u)

#define MOE_DYN_DMA_SLOT_S1          0u
#define MOE_DYN_DMA_SLOT_S3          1u
#define MOE_DYN_DMA_SLOT_S2_PREFETCH 2u
#define MOE_DYN_DMA_SLOT_S4_PREFETCH 3u
#define MOE_DYN_DMA_IDMA             1u
#define MOE_DYN_DMA_XDMA             2u
#define MOE_DYN_DMA_BOTH             3u
#define MOE_DYN_RT_C2_DONE           0u
#define MOE_DYN_RT_C3_DONE           1u
#define MOE_DYN_RT_C2_ACTIVE_SLOTS   2u
#define MOE_DYN_RT_C3_ACTIVE_SLOTS   3u

static inline uint32_t __moe_dyn_shape_m(uint32_t shape)
{
    if (shape == 0u) return 8u;
    if (shape == 1u) return 4u;
    return 2u;
}

/* Return per-VC meshCol for the given array_shape index.
 * S0: meshCol=4, S1: meshCol=8, S2: meshCol=16 (multidim_spatial_k8 hardware).
 * Dynamic MoE compute call lowering now uses this on the host side before the
 * L3->L1 args flush; device compute nodes consume the pre-lowered N directly. */
static inline uint32_t __moe_dyn_meshcol(uint32_t shape)
{
    if (shape == 0u) return MOE_DUAL_VC_MESH_COL_0;
    if (shape == 1u) return MOE_DUAL_VC_MESH_COL_1;
    return MOE_DUAL_VC_MESH_COL_2;
}

static inline uint32_t __moe_dyn_mode1_a_sstride0(uint32_t shape)
{
    if (shape == 0u) return 64u;
    return 8u;
}

static inline uint32_t __moe_dyn_mode1_a_sstride1(uint32_t shape)
{
    if (shape == 0u) return 8u;
    if (shape == 1u) return 16u;
    return 32u;
}

static inline uint32_t __moe_dyn_mode1_a_k_stride(uint32_t shape)
{
    if (shape == 0u) return 128u;
    if (shape == 1u) return 64u;
    return 16u;
}

static inline uint32_t __moe_dyn_b_channels(uint32_t shape)
{
    if (shape == 0u) return 2u;
    if (shape == 1u) return 4u;
    return 8u;
}

static inline uint32_t __moe_dyn_s1_block_count(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    return (cfg->s1_block_count != 0u) ? cfg->s1_block_count : cfg->indiv_N2;
}

static inline uint32_t __moe_dyn_s3_block_count(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    return (cfg->s3_block_count != 0u) ? cfg->s3_block_count : cfg->indiv_down_N2;
}

static inline uint64_t __moe_dyn_l1_wide(uint32_t local_addr)
{
    return chiplet_addr_transform((uint64_t)local_addr);
}

__attribute__((always_inline)) static inline void
__moe_dyn_cfg_down_from_swiglu_streamer(uint32_t A_addr,
                                        uint32_t B0_addr,
                                        uint32_t B1_addr,
                                        uint32_t D0_addr,
                                        uint32_t D1_addr,
                                        uint32_t M,
                                        uint32_t K,
                                        uint32_t N,
                                        uint32_t array_shape,
                                        uint32_t d_row_stride_override)
{
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t meshCol = __moe_dyn_meshcol(array_shape);
    uint32_t b_stream_bytes =
        __moe_dyn_b_channels(array_shape) * (MOE_DUAL_VC_BANK_WIDTH / 8u);
    uint32_t b_k_section = K * 16u;
    uint32_t a_m_stride = ((K * MOE_DUAL_VC_TILE_SIZE_0) / meshCol) * 64u;
    uint32_t d_row_stride = (d_row_stride_override != 0u) ?
        d_row_stride_override : (N * meshCol * 2u);
    uint32_t d_m_stride = meshRow * d_row_stride;
    uint32_t d_beats_per_row = meshCol / 4u;

    uint32_t a_sstride0 = __moe_dyn_mode1_a_sstride0(array_shape);
    uint32_t a_sstride1 = __moe_dyn_mode1_a_sstride1(array_shape);
    uint32_t a_k_stride = __moe_dyn_mode1_a_k_stride(array_shape);
    uint32_t chan_en_A  = MOE_DUAL_VC_CHAN_EN_A(array_shape);
    uint32_t chan_en_B  = MOE_DUAL_VC_CHAN_EN_B(array_shape);

    /* Dynamic down reads SwiGLU's Mode-0 D0 layout.  Keep this path expanded
     * like the L15 mode1 helper: no local CSR arrays, no loop-based streamer
     * setup, and only runtime-dependent values remain as register operands. */
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(S_STRIDE_BASE_READER_0+0, a_sstride0);
    csrw_ss(S_STRIDE_BASE_READER_0+1, a_sstride1);
    csrw_ss(T_BOUND_BASE_READER_0+0, K);
    csrw_ss(T_BOUND_BASE_READER_0+1, N);
    csrw_ss(T_BOUND_BASE_READER_0+2, M);
    csrw_ss(T_BOUND_BASE_READER_0+3, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+4, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+5, 1u);
    csrw_ss(T_STRIDE_BASE_READER_0+0, a_k_stride);
    csrw_ss(T_STRIDE_BASE_READER_0+1, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+2, a_m_stride);
    csrw_ss(T_STRIDE_BASE_READER_0+3, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+4, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+5, 0u);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, chan_en_A);

    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(S_STRIDE_BASE_READER_1+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_1+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_1+0, K);
    csrw_ss(T_BOUND_BASE_READER_1+1, N);
    csrw_ss(T_BOUND_BASE_READER_1+2, M);
    csrw_ss(T_BOUND_BASE_READER_1+3, 1u);
    csrw_ss(T_STRIDE_BASE_READER_1+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_1+1, K * b_stream_bytes);
    csrw_ss(T_STRIDE_BASE_READER_1+2, 0u);
    csrw_ss(T_STRIDE_BASE_READER_1+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, chan_en_B);

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_2+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_2+0, K);
    csrw_ss(T_BOUND_BASE_READER_2+1, N);
    csrw_ss(T_BOUND_BASE_READER_2+2, M);
    csrw_ss(T_BOUND_BASE_READER_2+3, 1u);
    csrw_ss(T_STRIDE_BASE_READER_2+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_2+1, K * b_stream_bytes);
    csrw_ss(T_STRIDE_BASE_READER_2+2, 0u);
    csrw_ss(T_STRIDE_BASE_READER_2+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, chan_en_B);

    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_0+0, d_beats_per_row);
    csrw_ss(T_BOUND_BASE_WRITER_0+1, meshRow);
    csrw_ss(T_BOUND_BASE_WRITER_0+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_0+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, meshCol * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, MOE_DUAL_VC_CHAN_EN_D);

    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_1+0, d_beats_per_row);
    csrw_ss(T_BOUND_BASE_WRITER_1+1, meshRow);
    csrw_ss(T_BOUND_BASE_WRITER_1+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_1+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, meshCol * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, MOE_DUAL_VC_CHAN_EN_D);
}

__attribute__((always_inline)) static inline uint32_t
__moe_dyn_dual_vc_down_gemm_from_swiglu_params(uint32_t A_addr,
                                               uint32_t B0_addr,
                                               uint32_t B1_addr,
                                               uint32_t D0_addr,
                                               uint32_t D1_addr,
                                               uint32_t M,
                                               uint32_t K,
                                               uint32_t N,
                                               uint32_t array_shape,
                                               uint32_t d_row_stride_override,
                                               uint32_t rscl_mult,
                                               uint32_t rscl_shift)
{
    if (array_shape > 2u) {
        printf_safe("[C%d c%d]: dynamic down: invalid array_shape %u\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx(), array_shape);
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    __moe_dyn_cfg_down_from_swiglu_streamer(
        A_addr, B0_addr, B1_addr, D0_addr, D1_addr,
        M, K, N, array_shape, d_row_stride_override);

    moe_set_dual_versacore_mode(1);
    moe_set_dual_versacore_csr(1, K, N * M, 0, array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}

__attribute__((always_inline)) static inline void
__moe_dyn_cfg_down_shape_c_from_swiglu_streamer(uint32_t A_addr,
                                                uint32_t B0_addr,
                                                uint32_t B1_addr,
                                                uint32_t D0_addr,
                                                uint32_t D1_addr,
                                                uint32_t M,
                                                uint32_t K,
                                                uint32_t N)
{
    uint32_t b_k_section = K * 16u;
    uint32_t b_n_stride = K * 64u;
    uint32_t a_m_stride = ((K * MOE_DUAL_VC_TILE_SIZE_2) /
                           MOE_DUAL_VC_MESH_COL_2) * 64u;
    uint32_t d_row_stride = N * MOE_DUAL_VC_MESH_COL_2 * 2u;
    uint32_t d_m_stride = MOE_DUAL_VC_MESH_ROW_2 * d_row_stride;

    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(S_STRIDE_BASE_READER_0+0, 8u);
    csrw_ss(S_STRIDE_BASE_READER_0+1, 32u);
    csrw_ss(T_BOUND_BASE_READER_0+0, K);
    csrw_ss(T_BOUND_BASE_READER_0+1, N);
    csrw_ss(T_BOUND_BASE_READER_0+2, M);
    csrw_ss(T_BOUND_BASE_READER_0+3, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+4, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+5, 1u);
    csrw_ss(T_STRIDE_BASE_READER_0+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_0+1, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+2, a_m_stride);
    csrw_ss(T_STRIDE_BASE_READER_0+3, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+4, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+5, 0u);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, MOE_DUAL_VC_CHAN_EN_A(2u));

    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(S_STRIDE_BASE_READER_1+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_1+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_1+0, K);
    csrw_ss(T_BOUND_BASE_READER_1+1, N);
    csrw_ss(T_BOUND_BASE_READER_1+2, M);
    csrw_ss(T_BOUND_BASE_READER_1+3, 1u);
    csrw_ss(T_STRIDE_BASE_READER_1+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_1+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_1+2, 0u);
    csrw_ss(T_STRIDE_BASE_READER_1+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, MOE_DUAL_VC_CHAN_EN_B(2u));

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_2+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_2+0, K);
    csrw_ss(T_BOUND_BASE_READER_2+1, N);
    csrw_ss(T_BOUND_BASE_READER_2+2, M);
    csrw_ss(T_BOUND_BASE_READER_2+3, 1u);
    csrw_ss(T_STRIDE_BASE_READER_2+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_2+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_2+2, 0u);
    csrw_ss(T_STRIDE_BASE_READER_2+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, MOE_DUAL_VC_CHAN_EN_B(2u));

    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_0+0, MOE_DUAL_VC_MESH_COL_2 / 4u);
    csrw_ss(T_BOUND_BASE_WRITER_0+1, MOE_DUAL_VC_MESH_ROW_2);
    csrw_ss(T_BOUND_BASE_WRITER_0+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_0+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, MOE_DUAL_VC_MESH_COL_2 * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, MOE_DUAL_VC_CHAN_EN_D);

    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_1+0, MOE_DUAL_VC_MESH_COL_2 / 4u);
    csrw_ss(T_BOUND_BASE_WRITER_1+1, MOE_DUAL_VC_MESH_ROW_2);
    csrw_ss(T_BOUND_BASE_WRITER_1+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_1+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, MOE_DUAL_VC_MESH_COL_2 * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, MOE_DUAL_VC_CHAN_EN_D);
}

__attribute__((always_inline)) static inline uint32_t
__moe_dyn_dual_vc_down_gemm_shape_c_from_swiglu_params(uint32_t A_addr,
                                                       uint32_t B0_addr,
                                                       uint32_t B1_addr,
                                                       uint32_t D0_addr,
                                                       uint32_t D1_addr,
                                                       uint32_t M,
                                                       uint32_t K,
                                                       uint32_t N,
                                                       uint32_t rscl_mult,
                                                       uint32_t rscl_shift)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    __moe_dyn_cfg_down_shape_c_from_swiglu_streamer(
        A_addr, B0_addr, B1_addr, D0_addr, D1_addr, M, K, N);

    moe_set_dual_versacore_mode(1);
    moe_set_dual_versacore_csr(1, K, N * M, 0, 2u, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}

static inline void __moe_dyn_idma_copy(uint64_t dst_addr, uint64_t src_addr, uint32_t bytes)
{
    if (bytes == 0u) return;
    snrt_dma_start_1d_wideptr(dst_addr, src_addr, bytes);
}

static inline void __moe_dyn_idma_copy_2d(uint64_t dst_addr,
                                          uint64_t src_addr,
                                          uint32_t row_bytes,
                                          uint32_t dst_stride,
                                          uint32_t src_stride,
                                          uint32_t rows)
{
    if (row_bytes == 0u || rows == 0u) return;
    snrt_dma_start_2d_wideptr(dst_addr, src_addr, row_bytes,
                              dst_stride, src_stride, rows);
}

static inline int32_t __moe_dyn_xdma_start_copy(uint64_t dst_addr, uint64_t src_addr, uint32_t bytes)
{
    if (bytes == 0u) return -1;
    // xdma_memcpy_1d_fast_full_addr: 30 CSR writes (vs 60 for
    // xdma_disable_all_extensions+xdma_memcpy_1d_full_addr).
    // Skips clearing 15 unused multicast dst slots (saves 30 writes).
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
    xdma_memcpy_1d_fast_full_addr(src_addr, dst_addr, bytes);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    return xdma_start();
}

static inline int32_t __moe_dyn_xdma_start_copy_2d(uint64_t dst_addr,
                                                   uint64_t src_addr,
                                                   uint32_t row_bytes,
                                                   uint32_t dst_stride,
                                                   uint32_t src_stride,
                                                   uint32_t rows)
{
    if (row_bytes == 0u || rows == 0u) return -1;
    if ((row_bytes % XDMA_WIDTH) != 0u) return -1;

    uint32_t strides_src[2] = { XDMA_WIDTH, src_stride };
    uint32_t strides_dst[2] = { XDMA_WIDTH, dst_stride };
    uint32_t bounds[2]      = { row_bytes / XDMA_WIDTH, rows };

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
    xdma_disable_all_extensions();
    xdma_memcpy_nd_full_addr(src_addr, dst_addr,
                             XDMA_WIDTH / XDMA_SPATIAL_CHAN,
                             XDMA_WIDTH / XDMA_SPATIAL_CHAN,
                             2, strides_src, bounds,
                             2, strides_dst, bounds,
                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    return xdma_start();
}

static inline void __moe_dyn_wait_xdma(uint64_t dst_addr, uint64_t src_addr, int32_t task_id)
{
    if (task_id >= 0) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_wait_task(src_addr, dst_addr, (uint32_t)task_id);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
    }
}

static inline int __moe_dyn_dma_is_valid(uint32_t binding)
{
    return binding <= 3u;
}

static inline uint32_t __moe_dyn_copy_pair(uint32_t binding,
                                           uint64_t dst0_addr,
                                           uint64_t src0_addr,
                                           uint64_t dst1_addr,
                                           uint64_t src1_addr,
                                           uint32_t bytes);

static inline uint32_t __moe_dyn_copy_pair_2d(uint32_t binding,
                                              uint64_t dst0_addr,
                                              uint64_t src0_addr,
                                              uint64_t dst1_addr,
                                              uint64_t src1_addr,
                                              uint32_t row_bytes,
                                              uint32_t dst_stride,
                                              uint32_t src_stride,
                                              uint32_t rows);

static inline volatile uint32_t *__moe_dyn_runtime_state(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    return (volatile uint32_t *)(uintptr_t)cfg->runtime_state_addr;
}

static inline uint32_t __moe_dyn_slot_active_this_round(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    uint32_t ctrl = cfg->ctrl;
    if (MOE_DYN_CTRL_ACTIVE(ctrl) == 0u || cfg->ntokens == 0u) return 0u;

    volatile uint32_t *state =
        (volatile uint32_t *)(uintptr_t)cfg->active_state_l1_addr;
    uint32_t active_idx = (MOE_DYN_CTRL_CLUSTER(ctrl) == 0u) ?
        MOE_DYN_RT_C2_ACTIVE_SLOTS : MOE_DYN_RT_C3_ACTIVE_SLOTS;
    return (MOE_DYN_CTRL_SLOT_ID(ctrl) < state[active_idx]) ? 1u : 0u;
}

static inline void __moe_dyn_wait_task_start(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    uint32_t wait_for_peer_slots = cfg->wait_for_peer_slots;
    if (wait_for_peer_slots == 0u) return;

    volatile uint32_t *state = __moe_dyn_runtime_state(cfg);
    uint32_t peer_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C3_DONE : MOE_DYN_RT_C2_DONE;
    while (state[peer_idx] < wait_for_peer_slots) {
        asm volatile("" ::: "memory");
    }
    asm volatile("fence r, rw" ::: "memory");
}

static inline void __moe_dyn_mark_task_complete(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    volatile uint32_t *state = __moe_dyn_runtime_state(cfg);
    uint32_t self_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C2_DONE : MOE_DYN_RT_C3_DONE;
    asm volatile("fence rw, w" ::: "memory");
    state[self_idx] = MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) + 1u;
    asm volatile("fence w, w" ::: "memory");
}

static inline void __moe_dyn_wait_dma_slot(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t slot)
{
    (void)cfg;
    (void)slot;
}

static inline void __moe_dyn_mark_dma_slot(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t slot)
{
    (void)cfg;
    (void)slot;
}

static inline uint32_t __moe_dyn_copy_gate_up_weight(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t expert_id,
    uint32_t binding)
{
    if (binding == 0u) return BINGO_RET_FAIL;
    uint64_t gate_src = cfg->indiv_gate_B_l3 +
        (uint64_t)expert_id * (uint64_t)cfg->indiv_B_expert_stride;
    uint64_t up_src = cfg->indiv_up_B_l3 +
        (uint64_t)expert_id * (uint64_t)cfg->indiv_B_expert_stride;
    uint32_t s1_weight_bytes = __moe_dyn_s1_block_count(cfg) * cfg->indiv_B_tile_bytes;
    return __moe_dyn_copy_pair(
        binding,
        __moe_dyn_l1_wide(cfg->l1_b_gate_addr), gate_src,
        __moe_dyn_l1_wide(cfg->l1_b_up_addr), up_src,
        s1_weight_bytes);
}

static inline uint32_t __moe_dyn_copy_down_weight(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t expert_id,
    uint32_t binding)
{
    if (binding == 0u) return BINGO_RET_FAIL;
    uint64_t down_src = cfg->indiv_down_B_l3 +
        (uint64_t)expert_id * (uint64_t)cfg->indiv_down_B_expert_stride;
    uint32_t down_weight_bytes = __moe_dyn_s3_block_count(cfg) * cfg->indiv_down_B_tile_bytes;
    if ((uint64_t)down_weight_bytes * 2u >
        (uint64_t)cfg->indiv_down_B_expert_stride) return BINGO_RET_FAIL;
    return __moe_dyn_copy_pair(
        binding,
        __moe_dyn_l1_wide(cfg->l1_b_down_addr), down_src,
        __moe_dyn_l1_wide(cfg->l1_b_down_addr + down_weight_bytes),
        down_src + down_weight_bytes,
        down_weight_bytes);
}

static inline uint32_t __moe_dyn_copy_gate_up_weight_block(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t expert_id,
    uint32_t binding,
    uint32_t block_idx)
{
    if (binding == 0u) return BINGO_RET_FAIL;
    uint32_t s1_blocks = __moe_dyn_s1_block_count(cfg);
    if (block_idx >= s1_blocks) return BINGO_RET_SUCC;
    uint64_t gate_src = cfg->indiv_gate_B_l3 +
        (uint64_t)expert_id * (uint64_t)cfg->indiv_B_expert_stride +
        (uint64_t)block_idx * (uint64_t)cfg->indiv_B_tile_bytes;
    uint64_t up_src = cfg->indiv_up_B_l3 +
        (uint64_t)expert_id * (uint64_t)cfg->indiv_B_expert_stride +
        (uint64_t)block_idx * (uint64_t)cfg->indiv_B_tile_bytes;
    uint32_t dst_off = block_idx * cfg->indiv_B_tile_bytes;
    return __moe_dyn_copy_pair(
        binding,
        __moe_dyn_l1_wide(cfg->l1_b_gate_addr + dst_off), gate_src,
        __moe_dyn_l1_wide(cfg->l1_b_up_addr + dst_off), up_src,
        cfg->indiv_B_tile_bytes);
}

static inline uint32_t __moe_dyn_copy_down_weight_block(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t expert_id,
    uint32_t binding,
    uint32_t block_idx)
{
    if (binding == 0u) return BINGO_RET_FAIL;
    uint32_t s3_blocks = __moe_dyn_s3_block_count(cfg);
    if (block_idx >= s3_blocks) return BINGO_RET_SUCC;
    uint64_t down_src = cfg->indiv_down_B_l3 +
        (uint64_t)expert_id * (uint64_t)cfg->indiv_down_B_expert_stride;
    uint64_t left_src = down_src +
        (uint64_t)block_idx * (uint64_t)cfg->indiv_down_B_tile_bytes;
    uint64_t right_src = down_src +
        (uint64_t)(s3_blocks + block_idx) *
            (uint64_t)cfg->indiv_down_B_tile_bytes;
    uint32_t left_off = block_idx * cfg->indiv_down_B_tile_bytes;
    uint32_t right_off = (s3_blocks + block_idx) *
        cfg->indiv_down_B_tile_bytes;
    return __moe_dyn_copy_pair(
        binding,
        __moe_dyn_l1_wide(cfg->l1_b_down_addr + left_off), left_src,
        __moe_dyn_l1_wide(cfg->l1_b_down_addr + right_off), right_src,
        cfg->indiv_down_B_tile_bytes);
}

static inline uint32_t __moe_dyn_copy_pair(uint32_t binding,
                                           uint64_t dst0_addr,
                                           uint64_t src0_addr,
                                           uint64_t dst1_addr,
                                           uint64_t src1_addr,
                                           uint32_t bytes)
{
    if (bytes == 0u || binding == 0u) return BINGO_RET_SUCC;
    if (!__moe_dyn_dma_is_valid(binding)) return BINGO_RET_FAIL;

    if (binding == 1u) {
        __moe_dyn_idma_copy(dst0_addr, src0_addr, bytes);
        __moe_dyn_idma_copy(dst1_addr, src1_addr, bytes);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
        return BINGO_RET_SUCC;
    }

    if (binding == 2u) {
        // xDMA only: 两路串行（各有 CFG 和 WAIT 子事件，来自 xdma_start_copy / wait_xdma）
        int32_t xdma_task = __moe_dyn_xdma_start_copy(dst0_addr, src0_addr, bytes);
        __moe_dyn_wait_xdma(dst0_addr, src0_addr, xdma_task);
        xdma_task = __moe_dyn_xdma_start_copy(dst1_addr, src1_addr, bytes);
        __moe_dyn_wait_xdma(dst1_addr, src1_addr, xdma_task);
        return BINGO_RET_SUCC;
    }

    // binding == 3 (BOTH): xDMA dst1 + iDMA dst0 并行
    int32_t xdma_task = __moe_dyn_xdma_start_copy(dst1_addr, src1_addr, bytes);
    __moe_dyn_idma_copy(dst0_addr, src0_addr, bytes);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
    __moe_dyn_wait_xdma(dst1_addr, src1_addr, xdma_task);
    return BINGO_RET_SUCC;
}

static inline uint32_t __moe_dyn_copy_pair_2d(uint32_t binding,
                                              uint64_t dst0_addr,
                                              uint64_t src0_addr,
                                              uint64_t dst1_addr,
                                              uint64_t src1_addr,
                                              uint32_t row_bytes,
                                              uint32_t dst_stride,
                                              uint32_t src_stride,
                                              uint32_t rows)
{
    if (row_bytes == 0u || rows == 0u || binding == 0u) return BINGO_RET_SUCC;
    if (!__moe_dyn_dma_is_valid(binding)) return BINGO_RET_FAIL;

    if (binding == 1u) {
        __moe_dyn_idma_copy_2d(dst0_addr, src0_addr, row_bytes, dst_stride, src_stride, rows);
        __moe_dyn_idma_copy_2d(dst1_addr, src1_addr, row_bytes, dst_stride, src_stride, rows);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
        return BINGO_RET_SUCC;
    }

    if (binding == 2u) {
        int32_t xdma_task = __moe_dyn_xdma_start_copy_2d(dst0_addr, src0_addr,
                                                        row_bytes, dst_stride,
                                                        src_stride, rows);
        if (xdma_task < 0) return BINGO_RET_FAIL;
        __moe_dyn_wait_xdma(dst0_addr, src0_addr, xdma_task);
        xdma_task = __moe_dyn_xdma_start_copy_2d(dst1_addr, src1_addr,
                                                row_bytes, dst_stride,
                                                src_stride, rows);
        if (xdma_task < 0) return BINGO_RET_FAIL;
        __moe_dyn_wait_xdma(dst1_addr, src1_addr, xdma_task);
        return BINGO_RET_SUCC;
    }

    int32_t xdma_task = __moe_dyn_xdma_start_copy_2d(dst1_addr, src1_addr,
                                                    row_bytes, dst_stride,
                                                    src_stride, rows);
    if (xdma_task < 0) return BINGO_RET_FAIL;
    __moe_dyn_idma_copy_2d(dst0_addr, src0_addr, row_bytes, dst_stride, src_stride, rows);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
    __moe_dyn_wait_xdma(dst1_addr, src1_addr, xdma_task);
    return BINGO_RET_SUCC;
}

static inline void __moe_dyn_zero_bytes(uint64_t dst_addr, uint32_t bytes)
{
    volatile uint8_t *dst = (volatile uint8_t *)(uintptr_t)dst_addr;
    for (uint32_t i = 0; i < bytes; i++) dst[i] = 0u;
}

static inline uint32_t __moe_dyn_m_exec(uint32_t ntokens, uint32_t shape)
{
    uint32_t shape_m = __moe_dyn_shape_m(shape);
    return ntokens > shape_m ? ntokens : shape_m;
}

static inline uint32_t __moe_dyn_gather_s1_tokens(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t m_s1_exec)
{
    if (m_s1_exec > cfg->max_tokens_per_expert) {
        printf_safe("[MoEDyn C%d] slot=%u ntokens=%u exceeds max=%u\r\n",
                    snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->ntokens,
                    cfg->max_tokens_per_expert);
        return BINGO_RET_FAIL;
    }

    if (cfg->A_token_bytes == 0u) {
        printf_safe("[MoEDyn C%d] invalid A layout: A_token_bytes=%u\r\n",
                    snrt_cluster_idx(), cfg->A_token_bytes);
        return BINGO_RET_FAIL;
    }

    /* L3 每 token 后有 32 字节 padding（datagen 预加），所以 L3 stride = A_token_bytes + 32。
     * L1 中 VersaCore streamer 以固定 stride = A_token_bytes 读取，不需要 padding，tokens 紧密排列。
     *
     * 双通道无竞争方案：
     *   C3 (cluster_idx==3) 使用 iDMA：批量非阻塞提交所有 token，单次 wait_all。
     *   C2 (cluster_idx==2) 使用 xDMA：流水线配置——在第 t 笔传输运行期间提前写入
     *     第 t+1 笔的 CSR（36 cc），完全 overlap 传输时间（~41 cc），最后统一 wait。
     * iDMA 和 xDMA 走不同的物理 AXI 通道，两 cluster 同时 gather 时不争用同一条总线。 */
    uint32_t l3_a_row_stride = cfg->A_token_bytes + 32u;  /* L3 stride: includes 32-byte padding */
    uint32_t l1_a_row_stride = cfg->A_token_bytes;          /* L1 stride: packed, no padding */
    uint16_t *token_ids = (uint16_t *)(uintptr_t)cfg->token_ids_addr;

    /* 先做边界检查，再 DMA 提交，避免提交到一半出错留下悬挂传输 */
    for (uint32_t local_t = 0; local_t < cfg->ntokens; local_t++) {
        uint32_t token_id = token_ids[cfg->token_start_rank + local_t];
        if (token_id >= cfg->max_tokens_per_expert) {
            printf_safe("[MoEDyn C%d] token_id=%u exceeds max=%u\r\n",
                        snrt_cluster_idx(), token_id, cfg->max_tokens_per_expert);
            return BINGO_RET_FAIL;
        }
    }

    if (snrt_cluster_idx() == 2u) {
        /* ── C2：xDMA 流水线 gather ──
         * Token 0：全量初始化 CSR（~1701 cc），启动传输（非阻塞）。
         * Token t（t>=1）：在第 t-1 笔传输运行期间预写 CSR（~36 cc，overlap），
         *   再 wait 第 t-1 笔（此时传输已接近完成，等待约 0–5 cc），
         *   最后立即 start 第 t 笔。
         * xDMA 的 CSR 配置路径（narrow bus → xDMA 寄存器）与数据传输路径
         * （xDMA engine → cluster crossbar → L3 wide port）相互独立，
         * 因此在上一笔传输运行时预写 CSR 是安全的。 */
        if (cfg->ntokens > 0u) {
            uint32_t tid0 = token_ids[cfg->token_start_rank];
            uint64_t src0 = cfg->input_A_l3_base + (uint64_t)tid0 * (uint64_t)l3_a_row_stride;
            uint64_t dst0 = __moe_dyn_l1_wide(cfg->l1_a_addr);
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
            xdma_memcpy_1d_fast_full_addr(src0, dst0, cfg->A_token_bytes);
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
            int32_t prev_task = xdma_start();
            uint64_t prev_src = src0, prev_dst = dst0;

            for (uint32_t local_t = 1; local_t < cfg->ntokens; local_t++) {
                uint32_t token_id = token_ids[cfg->token_start_rank + local_t];
                uint64_t src = cfg->input_A_l3_base +
                    (uint64_t)token_id * (uint64_t)l3_a_row_stride;
                uint64_t dst = __moe_dyn_l1_wide(cfg->l1_a_addr + local_t * l1_a_row_stride);
                /* 预写下一笔 CSR（~36 cc），此时上一笔传输仍在运行（~41 cc） */
                xdma_memcpy_1d_fast_full_addr(src, dst, cfg->A_token_bytes);
                /* 等待上一笔完成（CSR 写完后通常只剩 ~5 cc） */
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
                xdma_wait_task(prev_src, prev_dst, (uint32_t)prev_task);
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
                /* 立即启动已配置好的本笔传输 */
                prev_task = xdma_start();
                prev_src = src;
                prev_dst = dst;
            }

            /* 等待最后一笔 */
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
            xdma_wait_task(prev_src, prev_dst, (uint32_t)prev_task);
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
        }
    } else {
        /* ── C3：iDMA 批量 gather ──
         * 批量提交所有 token 的非阻塞 iDMA 请求，单次 wait_all。
         * iDMA 与 C2 的 xDMA 走不同的 AXI 通道，互不竞争。 */
        for (uint32_t local_t = 0; local_t < cfg->ntokens; local_t++) {
            uint32_t token_id = token_ids[cfg->token_start_rank + local_t];
            uint64_t src = cfg->input_A_l3_base +
                (uint64_t)token_id * (uint64_t)l3_a_row_stride;
            uint64_t dst = __moe_dyn_l1_wide(cfg->l1_a_addr + local_t * l1_a_row_stride);
            snrt_dma_start_1d_wideptr(dst, src, cfg->A_token_bytes);
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
    }

    if (m_s1_exec > cfg->ntokens) {
        /* Zero padding token rows in one contiguous range (L1 row = l1_a_row_stride bytes). */
        __moe_dyn_zero_bytes((uint64_t)cfg->l1_a_addr +
            (uint64_t)cfg->ntokens * (uint64_t)l1_a_row_stride,
            (uint64_t)(m_s1_exec - cfg->ntokens) * (uint64_t)l1_a_row_stride);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_gather_s1(void *arg)
{
    if (!snrt_is_dm_core()) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_gather_s1 must run on DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
#if MOE_DYN_DEBUG_SCHED_VERIFY
    /* PROBE gather_s1: 在 ACTIVE check 前 — 确认 DM core 进入了此函数 */
    printf_safe("[PROBE gs C%d] ctrl=0x%08x ntok=%u\r\n",
                snrt_cluster_idx(), cfg->ctrl, cfg->ntokens);
#endif
    if (!__moe_dyn_slot_active_this_round(cfg)) return BINGO_RET_SUCC;
    __moe_dyn_wait_task_start(cfg);

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_START);
    uint32_t rc = __moe_dyn_gather_s1_tokens(
        cfg, __moe_dyn_m_exec(cfg->ntokens, MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl)));
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_END);
    return rc;
}

static inline __snax_bingo_kernel_moe_dynamic_expert_args_t *__moe_dyn_block_cfg(
    void *arg,
    uint32_t *block_idx)
{
    __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (__snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    *block_idx = blk->block_idx;
    return (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
}

static inline __snax_bingo_kernel_dual_vc_swiglu_full_args_t
__moe_dyn_swiglu_args_from_pre(
    const __snax_bingo_moe_dyn_swiglu_call_args_t *pre)
{
    return (__snax_bingo_kernel_dual_vc_swiglu_full_args_t){
        .input_A_addr      = pre->input_A_addr,
        .input_B_gate_addr = pre->input_B_gate_addr,
        .input_B_up_addr   = pre->input_B_up_addr,
        .output_D0_addr    = pre->output_D0_addr,
        .output_D1_addr    = pre->output_D1_addr,
        .M                 = pre->M,
        .K                 = pre->K,
        .N                 = pre->N,
        .array_shape       = pre->array_shape,
        .rescale_mult      = pre->rescale_mult,
        .rescale_shift     = pre->rescale_shift,
    };
}

static inline uint32_t __moe_dyn_run_down_from_pre(
    const __snax_bingo_moe_dyn_down_call_args_t *pre)
{
    if (pre->array_shape == 2u && pre->d_row_stride_override == 0u) {
        return __moe_dyn_dual_vc_down_gemm_shape_c_from_swiglu_params(
            pre->input_A_addr,
            pre->input_B0_addr,
            pre->input_B1_addr,
            pre->output_D0_addr,
            pre->output_D1_addr,
            pre->M,
            pre->K,
            pre->N,
            pre->rescale_mult,
            pre->rescale_shift);
    }

    return __moe_dyn_dual_vc_down_gemm_from_swiglu_params(
        pre->input_A_addr,
        pre->input_B0_addr,
        pre->input_B1_addr,
        pre->output_D0_addr,
        pre->output_D1_addr,
        pre->M,
        pre->K,
        pre->N,
        pre->array_shape,
        pre->d_row_stride_override,
        pre->rescale_mult,
        pre->rescale_shift);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block(void *arg)
{
    if (!snrt_is_dm_core()) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_load_gate_up_block must run on DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    uint32_t n;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        __moe_dyn_block_cfg(arg, &n);
    uint32_t s1_blocks = __moe_dyn_s1_block_count(cfg);
#if MOE_DYN_DEBUG_SCHED_VERIFY
    if (n == 0u) {
        /* RAW probe: before ACTIVE check — shows ctrl even if ACTIVE=0 */
        printf_safe("[DEV C%d] n=%u ctrl=0x%08x ntok=%u s1blk=%u\r\n",
                    snrt_cluster_idx(), n, cfg->ctrl, cfg->ntokens, s1_blocks);
    }
#endif
    if (!__moe_dyn_slot_active_this_round(cfg) || n >= s1_blocks) {
        return BINGO_RET_SUCC;
    }
#if MOE_DYN_DEBUG_SCHED_VERIFY
    if (n == 0u) {
        /* sk: skip_s1 skip_s3 skip_s2 skip_s4 | ms2/ms4: shape-C M-tile counts */
        printf_safe("[DEV C%d] slot=%u e%u n%u sh%u/%u dm%u/%u sk%u%u%u%u ms2=%u ms4=%u\r\n",
                    snrt_cluster_idx(),
                    MOE_DYN_CTRL_SLOT_ID(cfg->ctrl),
                    (unsigned)cfg->expert_id,
                    (unsigned)cfg->ntokens,
                    MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl),
                    MOE_DYN_CTRL_SHAPE_S3(cfg->ctrl),
                    MOE_DYN_CTRL_DMA_S1(cfg->ctrl),
                    MOE_DYN_CTRL_DMA_S3(cfg->ctrl),
                    MOE_DYN_CTRL_SKIP_S1(cfg->ctrl),
                    MOE_DYN_CTRL_SKIP_S3(cfg->ctrl),
                    MOE_DYN_CTRL_SKIP_S2(cfg->ctrl),
                    MOE_DYN_CTRL_SKIP_S4(cfg->ctrl),
                    (unsigned)cfg->m_s2_exec,
                    (unsigned)cfg->m_s4_exec);
    }
#endif
    if (MOE_DYN_CTRL_SKIP_S1(cfg->ctrl) != 0u) return BINGO_RET_SUCC;
    if (MOE_DYN_CTRL_DMA_S1(cfg->ctrl) == 0u) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_START);
    if (n == 0u) __moe_dyn_wait_dma_slot(cfg, MOE_DYN_DMA_SLOT_S1);
    uint32_t rc = __moe_dyn_copy_gate_up_weight_block(
        cfg, cfg->expert_id, MOE_DYN_CTRL_DMA_S1(cfg->ctrl), n);
    if (rc == BINGO_RET_SUCC && n + 1u == s1_blocks) {
        __moe_dyn_mark_dma_slot(cfg, MOE_DYN_DMA_SLOT_S1);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_END);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_gate_up_block must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t n;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        __moe_dyn_block_cfg(arg, &n);
#if MOE_DYN_DEBUG_SCHED_VERIFY
    if (n == 0u) {
        /* PROBE compute_gate_up: 在 ACTIVE check 前 — 确认 core 0 进入了 GEMM 函数 */
        printf_safe("[PROBE cgu C%d n%u] ctrl=0x%08x ntok=%u\r\n",
                    snrt_cluster_idx(), n, cfg->ctrl, cfg->ntokens);
    }
#endif
    if (!__moe_dyn_slot_active_this_round(cfg) || n >= 2u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_swiglu_call_args_t *pre = &cfg->s1_call[n];
    if (pre->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    __snax_bingo_kernel_dual_vc_swiglu_full_args_t swiglu_args =
        __moe_dyn_swiglu_args_from_pre(pre);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_START);
    uint32_t __gu_block_rc = __snax_bingo_kernel_dual_vc_swiglu_full(&swiglu_args);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_END);
    return __gu_block_rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_down_block(void *arg)
{
    if (!snrt_is_dm_core()) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_load_down_block must run on DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    uint32_t n;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        __moe_dyn_block_cfg(arg, &n);
    uint32_t ctrl = cfg->ctrl;
    if (!__moe_dyn_slot_active_this_round(cfg)) {
        return BINGO_RET_SUCC;
    }
    if (MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u) return BINGO_RET_SUCC;

    uint32_t s3_blocks = __moe_dyn_s3_block_count(cfg);
    if (n >= s3_blocks) {
        return BINGO_RET_SUCC;
    }
    if (MOE_DYN_CTRL_DMA_S3(ctrl) == 0u) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_START);
    if (n == 0u) __moe_dyn_wait_dma_slot(cfg, MOE_DYN_DMA_SLOT_S3);
    uint32_t rc = __moe_dyn_copy_down_weight_block(
        cfg, cfg->expert_id, MOE_DYN_CTRL_DMA_S3(ctrl), n);
    if (rc == BINGO_RET_SUCC && n + 1u == s3_blocks) {
        __moe_dyn_mark_dma_slot(cfg, MOE_DYN_DMA_SLOT_S3);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_END);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_block(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_down_block must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t n;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        __moe_dyn_block_cfg(arg, &n);
    uint32_t ctrl = cfg->ctrl;
    if (!__moe_dyn_slot_active_this_round(cfg) || n >= 2u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_down_call_args_t *pre = &cfg->s3_call[n];
    if (pre->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    (void)ctrl;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_START);
    uint32_t __down_block_rc = __moe_dyn_run_down_from_pre(pre);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_END);
    return __down_block_rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down(void *arg)
{
    if (!snrt_is_dm_core()) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_prefetch_s2_down must run on DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
    uint32_t slot = MOE_DYN_DMA_SLOT_S2_PREFETCH;
    if (!__moe_dyn_slot_active_this_round(cfg) || MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        return BINGO_RET_SUCC;
    }
    if (cfg->dma_slot_expert_id[slot] < 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_START);
    __moe_dyn_wait_dma_slot(cfg, slot);
    uint32_t rc = __moe_dyn_copy_down_weight(
        cfg, (uint32_t)cfg->dma_slot_expert_id[slot], MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot));
    if (rc == BINGO_RET_SUCC) __moe_dyn_mark_dma_slot(cfg, slot);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_END);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1(void *arg)
{
    if (!snrt_is_dm_core()) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_prefetch_s4_next_s1 must run on DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
    uint32_t slot = MOE_DYN_DMA_SLOT_S4_PREFETCH;
    if (!__moe_dyn_slot_active_this_round(cfg) || MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        return BINGO_RET_SUCC;
    }
    if (cfg->dma_slot_expert_id[slot] < 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_START);
    __moe_dyn_wait_dma_slot(cfg, slot);
    uint32_t rc = __moe_dyn_copy_gate_up_weight(
        cfg, (uint32_t)cfg->dma_slot_expert_id[slot], MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot));
    if (rc == BINGO_RET_SUCC) __moe_dyn_mark_dma_slot(cfg, slot);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_END);
    return rc;
}

/* ============================================================
 * S2: gate+up 全量 GEMM（在S1 pipeline 之后处理剩余/全部 token）
 *   - skip_s1=1 (cache hit): host prelower 令 A 从偏移 0 开始
 *   - skip_s1=0 (tail):      host prelower 令 A/D 从 S1 prefix 后开始
 * Device side only consumes s2_call and issues a single full-N shape-C call.
 * ============================================================ */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_gate_up_full must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
    if (!__moe_dyn_slot_active_this_round(cfg)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_swiglu_call_args_t *pre = &cfg->s2_call;
    if (pre->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    __snax_bingo_kernel_dual_vc_swiglu_full_args_t swiglu_args =
        __moe_dyn_swiglu_args_from_pre(pre);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_START);
    uint32_t __gu_full_rc = __snax_bingo_kernel_dual_vc_swiglu_full(&swiglu_args);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_END);
    return __gu_full_rc;
}

/* ============================================================
 * S4: down 全量 GEMM（在 S3 pipeline 之后处理剩余/全部 token）
 *   - skip_s3=1 (cache hit): host prelower 令 A/D 从偏移 0 开始
 *   - skip_s3=0 (tail):      host prelower 令 A/D 从 S3 prefix 后开始
 *
 * skip_s3=1 mirrors SwiGLU full: one hardware GEMM covers all output N-blocks.
 * It writes l1_down_d in token-major full-N order; store() scatters it back to
 * the existing block-major L3 layout.
 *
 * skip_s3=0 tail also uses one full-N GEMM and appends its rows to the same
 * l1_down_d token-major full-N layout created by S3.
 * ============================================================ */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_down_full must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
    uint32_t ctrl = cfg->ctrl;
    if (!__moe_dyn_slot_active_this_round(cfg)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_down_call_args_t *pre = &cfg->s4_call;
    if (pre->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    (void)ctrl;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_START);
    uint32_t __down_full_rc = __moe_dyn_run_down_from_pre(pre);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_END);
    return __down_full_rc;
}

static inline uint32_t __moe_dyn_store_down_rowmajor_full(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    uint32_t store_dma,
    uint64_t expert_out_base,
    uint32_t row_bytes,
    uint32_t s3_blocks)
{
    uint32_t full_row_bytes = s3_blocks * row_bytes;
    uint32_t rows = cfg->ntokens;
    uint32_t d1_l1_base = cfg->l1_down_d_addr +
        s3_blocks * cfg->indiv_down_D_tile_bytes;

    for (uint32_t n = 0; n < s3_blocks; n++) {
        uint64_t d0_dst = expert_out_base +
            (uint64_t)n * (uint64_t)cfg->indiv_down_D_tile_bytes +
            (uint64_t)cfg->token_start_rank * (uint64_t)row_bytes;
        uint64_t d1_dst = expert_out_base +
            (uint64_t)(s3_blocks + n) * (uint64_t)cfg->indiv_down_D_tile_bytes +
            (uint64_t)cfg->token_start_rank * (uint64_t)row_bytes;
        uint64_t d0_src = __moe_dyn_l1_wide(cfg->l1_down_d_addr + n * row_bytes);
        uint64_t d1_src = __moe_dyn_l1_wide(d1_l1_base + n * row_bytes);
        uint32_t rc = __moe_dyn_copy_pair_2d(store_dma,
                                             d0_dst, d0_src,
                                             d1_dst, d1_src,
                                             row_bytes,
                                             row_bytes,
                                             full_row_bytes,
                                             rows);
        if (rc != BINGO_RET_SUCC) return rc;
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_store(void *arg)
{
    if (!snrt_is_dm_core()) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_store must run on DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
    if (!__moe_dyn_slot_active_this_round(cfg)) return BINGO_RET_SUCC;

    uint32_t row_bytes = cfg->indiv_down_D_tile_bytes / cfg->max_tokens_per_expert;
    uint32_t s3_blocks = __moe_dyn_s3_block_count(cfg);
    uint64_t expert_out_base = cfg->output_l3_base +
        (uint64_t)cfg->expert_id * (uint64_t)cfg->output_expert_stride_bytes;
    uint32_t store_dma = (MOE_DYN_CTRL_DMA_S3(cfg->ctrl) != 0u) ? MOE_DYN_CTRL_DMA_S3(cfg->ctrl) : 1u;

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_START);
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u ||
        MOE_DYN_CTRL_SKIP_S4(cfg->ctrl) == 0u) {
        uint32_t rc = __moe_dyn_store_down_rowmajor_full(cfg, store_dma,
                                                         expert_out_base,
                                                         row_bytes,
                                                         s3_blocks);
        if (rc != BINGO_RET_SUCC) return rc;
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_END);

#if MOE_DYN_DEBUG_SCHED_VERIFY
        printf_safe("[MoEDyn C%d] slot=%u expert=%u rank=%u ntok=%u done\r\n",
                    snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
                    cfg->token_start_rank, cfg->ntokens);
#endif
        __moe_dyn_mark_task_complete(cfg);
        return BINGO_RET_SUCC;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_END);

#if MOE_DYN_DEBUG_SCHED_VERIFY
    printf_safe("[MoEDyn C%d] slot=%u expert=%u rank=%u ntok=%u done\r\n",
                snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
                cfg->token_start_rank, cfg->ntokens);
#endif
    __moe_dyn_mark_task_complete(cfg);
    return BINGO_RET_SUCC;
}

// ============================================================
// Device-side shape configuration (integer fields only).
// Layout matches the integer prefix of host-side shape_cfg_t
// generated by multi_cluster_MoE_datagen.py.
// The host fills the complete shape_cfg_t struct and places it in
// device-accessible TCDM; the device casts the pointer to this type.
// NOTE: The trailing const pointer fields in the host struct
//       (mode0_d0_golden, mode1_padded_golden) are NOT accessed here.
// ============================================================
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
    /* Host-side pointer fields follow; NOT accessed by device */
} moe_l15_shape_cfg_t;

__attribute__((always_inline)) static inline void
__l15_zero_mode1_output_padding(const moe_l15_shape_cfg_t *cfg, uint32_t tcdm_base)
{
    uint32_t row_stride = (uint32_t)cfg->mode1_output_row_stride_bytes;
    if (row_stride == 0u) {
        return;
    }

    uint32_t region_bytes =
        (uint32_t)cfg->tcdm_end - (uint32_t)cfg->delta_local_mode1_d0;
    uint32_t rows = region_bytes / row_stride;
    if (rows == 0u) {
        return;
    }

    uint32_t writer1_offset =
        (uint32_t)cfg->delta_local_mode1_d1 - (uint32_t)cfg->delta_local_mode1_d0;
    uint32_t per_vc_payload =
        (uint32_t)cfg->mode1_D_tbound[0] *
        (uint32_t)cfg->mode1_D_tbound[2] *
        (uint32_t)cfg->D_sstride[0];
    uint32_t row_payload = writer1_offset + per_vc_payload;
    if (row_payload > row_stride) {
        row_payload = row_stride;
    }

    if (row_stride > row_payload) {
        uint32_t pad_bytes = row_stride - row_payload;
        uint32_t base = tcdm_base + (uint32_t)cfg->delta_local_mode1_d0;
        for (uint32_t r = 0; r < rows; r++) {
            __moe_zero_tcdm(base + r * row_stride + row_payload, pad_bytes);
        }
    }
}

// ============================================================
// L15 streamer CSR helpers — fully expanded to eliminate the
// generic loop-based moe_set_dual_versacore_streamer_csr call.
// Using always_inline guarantees no jal / stack overhead.
// Each csrw_ss with a compile-time CSR address emits a single
// lw+csrw instruction pair; no loop bookkeeping.
// ============================================================

__attribute__((always_inline)) static inline void
__l15_cfg_mode0_streamer(const moe_l15_shape_cfg_t *cfg, uint32_t tcdm_base)
{
    /* Reader 0: A (INT16, 6-dim temporal, 2-dim spatial) */
    csrw_ss(BASE_PTR_READER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_a);
    csrw_ss(S_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode0_A_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode0_A_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_0+0,  (uint32_t)cfg->mode0_A_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_0+1,  (uint32_t)cfg->mode0_A_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_0+2,  (uint32_t)cfg->mode0_A_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_0+3,  (uint32_t)cfg->mode0_A_tbound[3]);
    csrw_ss(T_BOUND_BASE_READER_0+4,  (uint32_t)cfg->mode0_A_tbound[4]);
    csrw_ss(T_BOUND_BASE_READER_0+5,  (uint32_t)cfg->mode0_A_tbound[5]);
    csrw_ss(T_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode0_A_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode0_A_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_0+2, (uint32_t)cfg->mode0_A_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_0+3, (uint32_t)cfg->mode0_A_tstride[3]);
    csrw_ss(T_STRIDE_BASE_READER_0+4, (uint32_t)cfg->mode0_A_tstride[4]);
    csrw_ss(T_STRIDE_BASE_READER_0+5, (uint32_t)cfg->mode0_A_tstride[5]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, (uint32_t)cfg->A_channel_en[0]);
    /* Reader 1: B0 = gate weight (INT4 packed, 4-dim temporal, 2-dim spatial) */
    csrw_ss(BASE_PTR_READER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_b0);
    csrw_ss(S_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode0_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode0_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_1+0,  (uint32_t)cfg->mode0_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_1+1,  (uint32_t)cfg->mode0_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_1+2,  (uint32_t)cfg->mode0_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_1+3,  (uint32_t)cfg->mode0_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode0_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode0_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_1+2, (uint32_t)cfg->mode0_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_1+3, (uint32_t)cfg->mode0_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, (uint32_t)cfg->B_channel_en[0]);
    /* Reader 2: B1 = up weight (same layout as B0) */
    csrw_ss(BASE_PTR_READER_2_LOW,    tcdm_base + (uint32_t)cfg->delta_local_b1);
    csrw_ss(S_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode0_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode0_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_2+0,  (uint32_t)cfg->mode0_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_2+1,  (uint32_t)cfg->mode0_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_2+2,  (uint32_t)cfg->mode0_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_2+3,  (uint32_t)cfg->mode0_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode0_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode0_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_2+2, (uint32_t)cfg->mode0_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_2+3, (uint32_t)cfg->mode0_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, (uint32_t)cfg->B_channel_en[0]);
    /* Writer 0: D0 = SwiGLU output */
    csrw_ss(BASE_PTR_WRITER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_d0);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->D_sstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+0,  (uint32_t)cfg->mode0_D_tbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+1,  (uint32_t)cfg->mode0_D_tbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_0+2,  (uint32_t)cfg->mode0_D_tbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_0+3,  (uint32_t)cfg->mode0_D_tbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->mode0_D_tstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, (uint32_t)cfg->mode0_D_tstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, (uint32_t)cfg->mode0_D_tstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, (uint32_t)cfg->mode0_D_tstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, (uint32_t)cfg->D_channel_en[0]);
    /* Writer 1: disabled in Mode-0 (SwiGLU hw only drives Writer-0) */
    csrw_ss(BASE_PTR_WRITER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_d0);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_1+0,  0u);
    csrw_ss(T_BOUND_BASE_WRITER_1+1,  0u);
    csrw_ss(T_BOUND_BASE_WRITER_1+2,  0u);
    csrw_ss(T_BOUND_BASE_WRITER_1+3,  0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, 0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, 0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, 0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, 0u);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, 0u);
}

__attribute__((always_inline)) static inline void
__l15_cfg_mode1_streamer(const moe_l15_shape_cfg_t *cfg, uint32_t tcdm_base)
{
    /* Reader 0: A = Mode-0 SwiGLU output D0 (INT16, 6-dim temporal) */
    csrw_ss(BASE_PTR_READER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_d0);
    csrw_ss(S_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode1_A_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode1_A_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_0+0,  (uint32_t)cfg->mode1_A_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_0+1,  (uint32_t)cfg->mode1_A_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_0+2,  (uint32_t)cfg->mode1_A_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_0+3,  (uint32_t)cfg->mode1_A_tbound[3]);
    csrw_ss(T_BOUND_BASE_READER_0+4,  (uint32_t)cfg->mode1_A_tbound[4]);
    csrw_ss(T_BOUND_BASE_READER_0+5,  (uint32_t)cfg->mode1_A_tbound[5]);
    csrw_ss(T_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode1_A_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode1_A_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_0+2, (uint32_t)cfg->mode1_A_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_0+3, (uint32_t)cfg->mode1_A_tstride[3]);
    csrw_ss(T_STRIDE_BASE_READER_0+4, (uint32_t)cfg->mode1_A_tstride[4]);
    csrw_ss(T_STRIDE_BASE_READER_0+5, (uint32_t)cfg->mode1_A_tstride[5]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, (uint32_t)cfg->A_channel_en[0]);
    /* Reader 1: B0 = W2l (left VC partition, INT4 packed, 4-dim temporal) */
    csrw_ss(BASE_PTR_READER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_w2l);
    csrw_ss(S_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode1_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode1_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_1+0,  (uint32_t)cfg->mode1_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_1+1,  (uint32_t)cfg->mode1_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_1+2,  (uint32_t)cfg->mode1_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_1+3,  (uint32_t)cfg->mode1_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode1_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode1_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_1+2, (uint32_t)cfg->mode1_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_1+3, (uint32_t)cfg->mode1_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, (uint32_t)cfg->B_channel_en[0]);
    /* Reader 2: B1 = W2r (right VC partition, same layout as W2l) */
    csrw_ss(BASE_PTR_READER_2_LOW,    tcdm_base + (uint32_t)cfg->delta_local_w2r);
    csrw_ss(S_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode1_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode1_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_2+0,  (uint32_t)cfg->mode1_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_2+1,  (uint32_t)cfg->mode1_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_2+2,  (uint32_t)cfg->mode1_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_2+3,  (uint32_t)cfg->mode1_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode1_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode1_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_2+2, (uint32_t)cfg->mode1_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_2+3, (uint32_t)cfg->mode1_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, (uint32_t)cfg->B_channel_en[0]);
    /* Writer 0: VC0 down-proj output (mode1_d0) */
    csrw_ss(BASE_PTR_WRITER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_mode1_d0);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->D_sstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+0,  (uint32_t)cfg->mode1_D_tbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+1,  (uint32_t)cfg->mode1_D_tbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_0+2,  (uint32_t)cfg->mode1_D_tbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_0+3,  (uint32_t)cfg->mode1_D_tbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->mode1_D_tstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, (uint32_t)cfg->mode1_D_tstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, (uint32_t)cfg->mode1_D_tstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, (uint32_t)cfg->mode1_D_tstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, (uint32_t)cfg->D_channel_en[0]);
    /* Writer 1: VC1 down-proj output (mode1_d1) */
    csrw_ss(BASE_PTR_WRITER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_mode1_d1);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, (uint32_t)cfg->D_sstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+0,  (uint32_t)cfg->mode1_D_tbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+1,  (uint32_t)cfg->mode1_D_tbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_1+2,  (uint32_t)cfg->mode1_D_tbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_1+3,  (uint32_t)cfg->mode1_D_tbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, (uint32_t)cfg->mode1_D_tstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, (uint32_t)cfg->mode1_D_tstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, (uint32_t)cfg->mode1_D_tstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, (uint32_t)cfg->mode1_D_tstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, (uint32_t)cfg->D_channel_en[0]);
}

// ============================================================
// L15 MoE kernel — Mode-0 only: SwiGLU (A×B0_gate + A×B1_up → D0).
// B0/B1 and A must be staged in TCDM before call.
// D0 output remains in TCDM for subsequent _down call.
//
// Args (uint32_t array, 4 fields): same as _full.
// ============================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_l15_moe_swiglu(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: dual_vc_l15_moe_swiglu must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    const moe_l15_shape_cfg_t *cfg =
        (const moe_l15_shape_cfg_t *)(uintptr_t)((uint32_t *)arg)[0];
    uint32_t tcdm_base  = ((uint32_t *)arg)[1];
    uint32_t rscl_mult  = ((uint32_t *)arg)[2];
    uint32_t rscl_shift = ((uint32_t *)arg)[3];

    BINGO_TRACE_MARKER(BINGO_TRACE_L15_SWIGLU_CFG_START);
    __l15_cfg_mode0_streamer(cfg, tcdm_base);

    moe_set_dual_versacore_mode(0);
    moe_set_dual_versacore_csr(
        1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale_mul(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_SWIGLU_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_SWIGLU_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_SWIGLU_RUN_END);

    return BINGO_RET_SUCC;
}

// ============================================================
// L15 MoE kernel — Mode-1 only: down-proj GEMM (D0 as A; W2l/W2r as B0/B1).
// Requires: D0 (SwiGLU output) and W2l/W2r already in TCDM.
// Clears only the padded bytes in the Mode-1 output rows before computing.
// Valid payload is intentionally left untouched before the writer runs, so a
// writer coverage bug still propagates X and remains visible to assertions.
//
// Args (uint32_t array, 4 fields): same as _full.
// ============================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_l15_moe_down(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: dual_vc_l15_moe_down must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    const moe_l15_shape_cfg_t *cfg =
        (const moe_l15_shape_cfg_t *)(uintptr_t)((uint32_t *)arg)[0];
    uint32_t tcdm_base  = ((uint32_t *)arg)[1];
    uint32_t rscl_mult  = ((uint32_t *)arg)[2];
    uint32_t rscl_shift = ((uint32_t *)arg)[3];

    __l15_zero_mode1_output_padding(cfg, tcdm_base);

    BINGO_TRACE_MARKER(BINGO_TRACE_L15_DOWN_CFG_START);
    __l15_cfg_mode1_streamer(cfg, tcdm_base);

    moe_set_dual_versacore_mode(1);
    uint32_t mode1_output_tiles =
        (uint32_t)cfg->mode1_D_tbound[2] * (uint32_t)cfg->mode1_D_tbound[3];
    moe_set_dual_versacore_csr(
        1, cfg->K1, mode1_output_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_DOWN_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_DOWN_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_DOWN_RUN_END);

    return BINGO_RET_SUCC;
}

// ============================================================
// Dual-VersaCore L15 MoE kernel: Mode-0 (SwiGLU) + Mode-1 (down-proj GEMM)
// All tensors must be staged in TCDM (L15 weights-first layout) before call.
//
// Args (uint32_t array, 4 fields):
//   [0] = (uint32_t)(uintptr_t) moe_l15_shape_cfg_t *  (device TCDM ptr)
//   [1] = tcdm_base   (absolute L1 address of tensor region start)
//   [2] = rescale_mult
//   [3] = rescale_shift
// ============================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_l15_moe_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: dual_vc_l15_moe_full must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    const moe_l15_shape_cfg_t *cfg =
        (const moe_l15_shape_cfg_t *)(uintptr_t)((uint32_t *)arg)[0];
    uint32_t tcdm_base  = ((uint32_t *)arg)[1];
    uint32_t rscl_mult  = ((uint32_t *)arg)[2];
    uint32_t rscl_shift = ((uint32_t *)arg)[3];

    __l15_zero_mode1_output_padding(cfg, tcdm_base);

    // ---- Mode-0: SwiGLU (A x B0_gate, A x B1_up -> SiLU -> elemMul -> D0) ----
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG_START);  // Mode-0 CSR config start
    __l15_cfg_mode0_streamer(cfg, tcdm_base);

    moe_set_dual_versacore_mode(0);  // Mode 0 = SwiGLU
    moe_set_dual_versacore_csr(
        1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale_mul(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG_END);    // Mode-0 CSR config end

    // ---- Run Mode-0 to completion before touching Mode-1 CSRs ----
    // The standalone L15 reference uses this ordering. Keeping the two modes
    // fused in one Bingo node is fine, but preloading Mode-1 CSRs while Mode-0
    // is active can leave MOE_DUAL_VC_BUSY stuck on the current hardware.
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_END);

    // ---- Mode-1: down projection (D0 as A, W2L/W2R as B0/B1 -> D0/D1) ----
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_START); // Mode-1 CSR config start
    __l15_cfg_mode1_streamer(cfg, tcdm_base);

    moe_set_dual_versacore_mode(1);  // Mode 1 = GEMM
    uint32_t mode1_output_tiles =
        (uint32_t)cfg->mode1_D_tbound[2] * (uint32_t)cfg->mode1_D_tbound[3];
    moe_set_dual_versacore_csr(
        1, cfg->K1, mode1_output_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_END);   // Mode-1 CSR config end

    // ---- Start Mode-1 after its CSRs have been programmed ----
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_END);

    return BINGO_RET_SUCC;
}
