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

// ============================================================
// Addr-only helpers for block-fashion loops:
// When M/K/N/shape/strides are unchanged, only the base pointers
// differ between blocks.  Write 3 or 4 CSRs instead of all 85/77.
// ============================================================

// SwiGLU block loop: B_gate, B_up, D0 change; A and D1 are constant.
static inline void moe_set_dual_vc_addrs_swiglu(
    uint32_t Bg_addr, uint32_t Bu_addr, uint32_t D0_addr)
{
    csrw_ss(BASE_PTR_READER_1_LOW, Bg_addr);
    csrw_ss(BASE_PTR_READER_2_LOW, Bu_addr);
    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
}

// GEMM block loop: B0, B1, D0, D1 all change; A is constant.
static inline void moe_set_dual_vc_addrs_gemm(
    uint32_t B0_addr, uint32_t B1_addr, uint32_t D0_addr, uint32_t D1_addr)
{
    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
}

// DEPRECATED (k8_8x4_4lane): Bank replication is no longer needed.
// The 4-lane postproc D writer (1 channel, CHAN_EN_D=0x01) writes to all banks
// sequentially via natural bank rotation. The Mode-1 A reader accesses the same
// contiguous output region without any manual copy.
static inline void moe_dual_vc_swiglu_bank_replicate(
    uint32_t M, uint32_t N, uint32_t array_shape, uint32_t D0_addr)
{
    (void)M; (void)N; (void)array_shape; (void)D0_addr;
    // No-op: bank replication not required for k8_8x4_4lane 4-lane postproc hw.
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
// Dual-VersaCore GEMM kernel (Mode 1: A@B0 -> D0)
// INT16 A x INT4 packed B -> INT16 D
// Args (uint32_t array, 9 fields):
//   [0] A_addr  [1] B0_addr  [2] D0_addr
//   [3] M  [4] K  [5] N
//   [6] array_shape  [7] rescale_mult  [8] rescale_shift
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

    // A: INT16, 6-dim temporal, 2-dim spatial (hw req., S_STRIDE_NUM_READER_0=2)
    int32_t Asl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), 0 };
    int32_t Atb[6]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1, 1, 1 };
    int32_t Ats[6]       = { (int32_t)(meshRow * tileSize * 2), 0,
                              (int32_t)(K * meshRow * tileSize * 2), 0, 0, 0 };
    int32_t chan_en_A[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape) };

    // B0: INT4 packed, 4-dim temporal, 2-dim spatial (hw req., S_STRIDE_NUM_READER_1=2)
    // n_b_chan: active B channels per shape (S0=2, S1=4, S2=8)
    uint32_t n_b_chan     = (array_shape == 0u) ? 2u : (array_shape == 1u) ? 4u : 8u;
    int32_t B_stream_bytes = (int32_t)(n_b_chan * (MOE_DUAL_VC_BANK_WIDTH / 8));
    int32_t B0sl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), 0 };
    int32_t B0tb[4]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B0ts[4]       = { B_stream_bytes, (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B0[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // B1: INT4 packed, same layout as B0, 2-dim spatial (hw req., S_STRIDE_NUM_READER_2=2)
    int32_t B1sl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), 0 };
    int32_t B1tb[4]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B1ts[4]       = { B_stream_bytes, (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B1[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

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

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
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
/* ── Schedule verification debug: set to 1 to print slot assignment on UART ── */
#define MOE_DYN_DEBUG_SCHED_VERIFY 1

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
 *   bits [15:14]: slot_id           (0-3)
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

static inline uint32_t __moe_dyn_shape_m(uint32_t shape)
{
    if (shape == 0u) return 8u;
    if (shape == 1u) return 4u;
    return 2u;
}

/* Return per-VC meshCol for the given array_shape index.
 * S0: meshCol=4, S1: meshCol=8, S2: meshCol=16 (multidim_spatial_k8 hardware).
 * N1 = indiv_N_per_block / __moe_dyn_meshcol(shape) at runtime. */
static inline uint32_t __moe_dyn_meshcol(uint32_t shape)
{
    if (shape == 0u) return MOE_DUAL_VC_MESH_COL_0;
    if (shape == 1u) return MOE_DUAL_VC_MESH_COL_1;
    return MOE_DUAL_VC_MESH_COL_2;
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

static inline void __moe_dyn_idma_copy(uint64_t dst_addr, uint64_t src_addr, uint32_t bytes)
{
    if (bytes == 0u) return;
    snrt_dma_start_1d_wideptr(dst_addr, src_addr, bytes);
}

static inline int32_t __moe_dyn_xdma_start_copy(uint64_t dst_addr, uint64_t src_addr, uint32_t bytes)
{
    if (bytes == 0u) return -1;
    // xdma_memcpy_1d_fast_full_addr: 30 CSR writes (vs 60 for
    // xdma_disable_all_extensions+xdma_memcpy_1d_full_addr).
    // Skips clearing 15 unused multicast dst slots (saves 30 writes).
    xdma_memcpy_1d_fast_full_addr(src_addr, dst_addr, bytes);
    return xdma_start();
}

static inline void __moe_dyn_wait_xdma(uint64_t dst_addr, uint64_t src_addr, int32_t task_id)
{
    if (task_id >= 0) xdma_wait_task(src_addr, dst_addr, (uint32_t)task_id);
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

static inline volatile uint32_t *__moe_dyn_runtime_state(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    return (volatile uint32_t *)(uintptr_t)cfg->runtime_state_addr;
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

static inline uint32_t __moe_dyn_copy_one(uint32_t binding,
                                          uint64_t dst_addr,
                                          uint64_t src_addr,
                                          uint32_t bytes)
{
    if (bytes == 0u || binding == 0u) return BINGO_RET_SUCC;
    if (!__moe_dyn_dma_is_valid(binding)) return BINGO_RET_FAIL;

    if (binding == 2u) {
        int32_t xdma_task = __moe_dyn_xdma_start_copy(dst_addr, src_addr, bytes);
        __moe_dyn_wait_xdma(dst_addr, src_addr, xdma_task);
        return BINGO_RET_SUCC;
    }

    if (binding == 3u && bytes > 1u) {
        uint32_t first_bytes = bytes / 2u;
        uint32_t second_bytes = bytes - first_bytes;
        int32_t xdma_task = __moe_dyn_xdma_start_copy(dst_addr + first_bytes,
                                                      src_addr + first_bytes,
                                                      second_bytes);
        __moe_dyn_idma_copy(dst_addr, src_addr, first_bytes);
        snrt_dma_wait_all();
        __moe_dyn_wait_xdma(dst_addr + first_bytes, src_addr + first_bytes, xdma_task);
        return BINGO_RET_SUCC;
    }

    __moe_dyn_idma_copy(dst_addr, src_addr, bytes);
    snrt_dma_wait_all();
    return BINGO_RET_SUCC;
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
        snrt_dma_wait_all();
        return BINGO_RET_SUCC;
    }

    if (binding == 2u) {
        int32_t xdma_task = __moe_dyn_xdma_start_copy(dst0_addr, src0_addr, bytes);
        __moe_dyn_wait_xdma(dst0_addr, src0_addr, xdma_task);
        xdma_task = __moe_dyn_xdma_start_copy(dst1_addr, src1_addr, bytes);
        __moe_dyn_wait_xdma(dst1_addr, src1_addr, xdma_task);
        return BINGO_RET_SUCC;
    }

    int32_t xdma_task = __moe_dyn_xdma_start_copy(dst1_addr, src1_addr, bytes);
    __moe_dyn_idma_copy(dst0_addr, src0_addr, bytes);
    snrt_dma_wait_all();
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
     * L1 中 VersaCore streamer 以固定 stride = K×tileSize×2 = A_token_bytes 读取，
     * 不需要 padding，tokens 紧密排列。每 token 做一次整行 DMA（2048 bytes）。 */
    uint32_t l3_a_row_stride = cfg->A_token_bytes + 32u;  /* L3 stride: includes 32-byte padding */
    uint32_t l1_a_row_stride = cfg->A_token_bytes;          /* L1 stride: packed, no padding */
    uint16_t *token_ids = (uint16_t *)(uintptr_t)cfg->token_ids_addr;
    uint32_t gather_dma = (MOE_DYN_CTRL_DMA_S1(cfg->ctrl) != 0u) ? MOE_DYN_CTRL_DMA_S1(cfg->ctrl) : 1u;
    for (uint32_t local_t = 0; local_t < cfg->ntokens; local_t++) {
        uint32_t token_id = token_ids[cfg->token_start_rank + local_t];
        if (token_id >= cfg->max_tokens_per_expert) {
            printf_safe("[MoEDyn C%d] token_id=%u exceeds max=%u\r\n",
                        snrt_cluster_idx(), token_id, cfg->max_tokens_per_expert);
            return BINGO_RET_FAIL;
        }
        uint64_t src = cfg->input_A_l3_base +
            (uint64_t)token_id * (uint64_t)l3_a_row_stride;
        uint32_t dst = cfg->l1_a_addr + local_t * l1_a_row_stride;
        uint32_t rc = __moe_dyn_copy_one(gather_dma,
                                         __moe_dyn_l1_wide(dst), src,
                                         cfg->A_token_bytes); /* copy only data, no L3 padding */
        if (rc != BINGO_RET_SUCC) return rc;
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
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u) return BINGO_RET_SUCC;
    __moe_dyn_wait_task_start(cfg);

    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    uint32_t rc = __moe_dyn_gather_s1_tokens(
        cfg, __moe_dyn_m_exec(cfg->ntokens, MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl)));
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
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
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || n >= s1_blocks) {
        return BINGO_RET_SUCC;
    }
#if MOE_DYN_DEBUG_SCHED_VERIFY
    if (n == 0u) {
        /* sk: skip_s1 skip_s3 skip_s2 skip_s4 | ms2/ms4: actual exec tokens */
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

    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    if (n == 0u) __moe_dyn_wait_dma_slot(cfg, MOE_DYN_DMA_SLOT_S1);
    uint32_t rc = __moe_dyn_copy_gate_up_weight_block(
        cfg, cfg->expert_id, MOE_DYN_CTRL_DMA_S1(cfg->ctrl), n);
    if (rc == BINGO_RET_SUCC && n + 1u == s1_blocks) {
        __moe_dyn_mark_dma_slot(cfg, MOE_DYN_DMA_SLOT_S1);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_gate_up_block must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    uint32_t n;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        __moe_dyn_block_cfg(arg, &n);
    uint32_t s1_blocks = __moe_dyn_s1_block_count(cfg);
#if MOE_DYN_DEBUG_SCHED_VERIFY
    if (n == 0u) {
        /* PROBE compute_gate_up: 在 ACTIVE check 前 — 确认 core 0 进入了 GEMM 函数 */
        printf_safe("[PROBE cgu C%d n%u] ctrl=0x%08x ntok=%u\r\n",
                    snrt_cluster_idx(), n, cfg->ctrl, cfg->ntokens);
    }
#endif
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || n >= s1_blocks) {
        return BINGO_RET_SUCC;
    }

    uint32_t shape_s1    = MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl);
    uint32_t m_s1_exec  = __moe_dyn_m_exec(cfg->ntokens, shape_s1); /* token count for gather/bounds */
    if (m_s1_exec > cfg->max_tokens_per_expert) {
        printf_safe("[MoEDyn C%d] slot=%u ntokens=%u exceeds max=%u\r\n",
                    snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->ntokens,
                    cfg->max_tokens_per_expert);
        return BINGO_RET_FAIL;
    }
    /* S1 只跑一个 batch (M1=1, M2=1): VersaCore M=1，只处理 meshRow 个 token;
     * 剩余 token 由 compute_gate_up_full (S2) 统一处理，M = m_s2_exec */
    (void)m_s1_exec; /* 仅用于上方 bounds check，不再用于计算 M */

    __snax_bingo_kernel_dual_vc_swiglu_full_args_t swiglu_args;
    if (MOE_DYN_CTRL_SKIP_S1(cfg->ctrl) != 0u) {
        /* cache hit: S1 全部跳过，所有 token 由 compute_gate_up_full (S2) 处理 */
        return BINGO_RET_SUCC;
    } else {
        /* 正常流水线：每次处理 block n，与 load[n] 重叠执行。
         * D 地址连续布局：block n 紧接 block n-1 之后，使 S3 能以连续 K-tiles 读取全 N。
         * 每个 M-tile 中 block n 的字节偏移 = n × (shape_m × d_row_bytes)。
         * shape_m = meshRow of shape_s1；d_row_bytes = indiv_D_tile_bytes / max_tokens。
         * 对 shape C (meshRow=2, meshCol=16): 每 block M-tile = 2×512 = 1024 bytes;
         *   block 0 at [0..1023], block 1 at [1024..2047] → full-N M-tile = 2048 bytes ✓ */
        uint32_t d_row_bytes_s1 = cfg->indiv_D_tile_bytes / cfg->max_tokens_per_expert;
        uint32_t shape_m_s1     = __moe_dyn_shape_m(shape_s1);
        swiglu_args = (__snax_bingo_kernel_dual_vc_swiglu_full_args_t){
            .input_A_addr      = cfg->l1_a_addr,
            .input_B_gate_addr = cfg->l1_b_gate_addr + n * cfg->indiv_B_tile_bytes,
            .input_B_up_addr   = cfg->l1_b_up_addr   + n * cfg->indiv_B_tile_bytes,
            .output_D0_addr    = cfg->l1_d_addr + n * shape_m_s1 * d_row_bytes_s1,
            .output_D1_addr    = cfg->l1_d1_scratch_addr,
            .M           = 1u, /* S1: 固定 1 个 batch (M1=1) */
            .K           = cfg->indiv_K1,
            .N           = cfg->indiv_N_per_block / __moe_dyn_meshcol(shape_s1),
            .array_shape = shape_s1,
            .rescale_mult = cfg->rescale_mult,
            .rescale_shift = cfg->rescale_shift,
        };
    }
    return __snax_bingo_kernel_dual_vc_swiglu_full(&swiglu_args);
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
    uint32_t s3_blocks = __moe_dyn_s3_block_count(cfg);
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || n >= s3_blocks) {
        return BINGO_RET_SUCC;
    }
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) != 0u) return BINGO_RET_SUCC;
    if (MOE_DYN_CTRL_DMA_S3(cfg->ctrl) == 0u) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    if (n == 0u) __moe_dyn_wait_dma_slot(cfg, MOE_DYN_DMA_SLOT_S3);
    uint32_t rc = __moe_dyn_copy_down_weight_block(
        cfg, cfg->expert_id, MOE_DYN_CTRL_DMA_S3(cfg->ctrl), n);
    if (rc == BINGO_RET_SUCC && n + 1u == s3_blocks) {
        __moe_dyn_mark_dma_slot(cfg, MOE_DYN_DMA_SLOT_S3);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_block(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_down_block must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    uint32_t n;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        __moe_dyn_block_cfg(arg, &n);
    uint32_t s3_blocks = __moe_dyn_s3_block_count(cfg);
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || n >= s3_blocks) {
        return BINGO_RET_SUCC;
    }

    uint32_t shape_s3    = MOE_DYN_CTRL_SHAPE_S3(cfg->ctrl);
    uint32_t m_s3_exec  = __moe_dyn_m_exec(cfg->ntokens, shape_s3); /* token count for bounds */
    if (m_s3_exec > cfg->max_tokens_per_expert) {
        printf_safe("[MoEDyn C%d] slot=%u ntokens=%u exceeds max=%u\r\n",
                    snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->ntokens,
                    cfg->max_tokens_per_expert);
        return BINGO_RET_FAIL;
    }
    /* S3 只跑一个 batch (M1=1, M2=1): VersaCore M=1，只处理 meshRow 个 token;
     * 剩余 token 由 compute_down_full (S4) 统一处理，M = m_s4_exec */
    (void)m_s3_exec; /* 仅用于上方 bounds check，不再用于计算 M */

    __snax_bingo_kernel_dual_vc_gemm_full_args_t down_args;
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) != 0u) {
        /* cache hit: S3 全部跳过，所有 token 由 compute_down_full (S4) 处理 */
        return BINGO_RET_SUCC;
    } else {
        /* 正常流水线：每次处理 block n，与 load[n] 重叠执行 */
        down_args = (__snax_bingo_kernel_dual_vc_gemm_full_args_t){
            .input_A_addr   = cfg->l1_d_addr,
            .input_B0_addr  = cfg->l1_b_down_addr + n * cfg->indiv_down_B_tile_bytes,
            .input_B1_addr  = cfg->l1_b_down_addr +
                (s3_blocks + n) * cfg->indiv_down_B_tile_bytes,
            .output_D0_addr = cfg->l1_down_d_addr + n * cfg->indiv_down_D_tile_bytes,
            .output_D1_addr = cfg->l1_down_d_addr +
                (s3_blocks + n) * cfg->indiv_down_D_tile_bytes,
            .M           = 1u, /* S3: 固定 1 个 batch (M1=1) */
            .K           = cfg->indiv_down_K1,
            .N           = cfg->indiv_down_N_per_block / __moe_dyn_meshcol(shape_s3),
            .array_shape = shape_s3,
            .rescale_mult = cfg->rescale_mult,
            .rescale_shift = cfg->rescale_shift,
        };
    }
    return __snax_bingo_kernel_dual_vc_gemm_full(&down_args);
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
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        return BINGO_RET_SUCC;
    }
    if (cfg->dma_slot_expert_id[slot] < 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    __moe_dyn_wait_dma_slot(cfg, slot);
    uint32_t rc = __moe_dyn_copy_down_weight(
        cfg, (uint32_t)cfg->dma_slot_expert_id[slot], MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot));
    if (rc == BINGO_RET_SUCC) __moe_dyn_mark_dma_slot(cfg, slot);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
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
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        return BINGO_RET_SUCC;
    }
    if (cfg->dma_slot_expert_id[slot] < 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    __moe_dyn_wait_dma_slot(cfg, slot);
    uint32_t rc = __moe_dyn_copy_gate_up_weight(
        cfg, (uint32_t)cfg->dma_slot_expert_id[slot], MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot));
    if (rc == BINGO_RET_SUCC) __moe_dyn_mark_dma_slot(cfg, slot);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    return rc;
}

/* ============================================================
 * S2: gate+up 全量 GEMM（在 S1 pipeline 之后处理剩余/全部 token）
 *   - skip_s1=1 (cache hit): A 从偏移 0 开始，N = full_N，single call
 *   - skip_s1=0 (tail):      A 从 shape_M token 处开始，per-block loop
 * ============================================================ */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_gate_up_full must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || MOE_DYN_CTRL_SKIP_S2(cfg->ctrl) != 0u) {
        return BINGO_RET_SUCC;
    }

    uint32_t m_s2 = cfg->m_s2_exec;
    uint32_t shape_s1 = MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl);  // S1's shape: used for tail A-offset only
    uint32_t shape_s2 = 2u;  // S2 固定 shape C (meshRow=2)，与 S1 动态 shape 解耦
    if (m_s2 == 0u) return BINGO_RET_SUCC;
    if (m_s2 > cfg->max_tokens_per_expert / 2u) {  /* m_s2 单位: M-tile；shape C meshRow=2 */
        printf_safe("[MoEDyn C%d] slot=%u m_s2_exec=%u exceeds max=%u\r\n",
                    snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), m_s2,
                    cfg->max_tokens_per_expert);
        return BINGO_RET_FAIL;
    }

    uint32_t gate_up_n2 = __moe_dyn_s1_block_count(cfg);  /* gate/up N 方向 N2 分块数 */

    if (MOE_DYN_CTRL_SKIP_S1(cfg->ctrl) != 0u) {
        /* cache hit: 权重全部驻留 TCDM，single full-N GEMM 处理所有 token */
        __snax_bingo_kernel_dual_vc_swiglu_full_args_t swiglu_args = {
            .input_A_addr      = cfg->l1_a_addr,
            .input_B_gate_addr = cfg->l1_b_gate_addr,
            .input_B_up_addr   = cfg->l1_b_up_addr,
            .output_D0_addr    = cfg->l1_d_addr,
            .output_D1_addr    = cfg->l1_d1_scratch_addr,
            .M                 = m_s2,
            .K                 = cfg->indiv_K1,
            .N                 = gate_up_n2 * cfg->indiv_N_per_block / __moe_dyn_meshcol(shape_s2),
            .array_shape       = shape_s2,
            .rescale_mult      = cfg->rescale_mult,
            .rescale_shift     = cfg->rescale_shift,
        };
        return __snax_bingo_kernel_dual_vc_swiglu_full(&swiglu_args);
    } else {
        /* 正常 tail：single full-N GEMM，与 cache hit 路径对称。
         * d_offset 跳过 S1 已写入的完整 M-tile（全 N-blocks 连续布局）:
         *   d_offset = shape_m × gate_up_n2 × d_row_bytes
         * 其中 d_row_bytes = indiv_D_tile_bytes / max_tokens（每 token 每 N-block 的字节数）。*/
        uint32_t shape_m = __moe_dyn_shape_m(shape_s1);
        uint32_t a_offset    = shape_m * cfg->A_token_bytes;
        uint32_t d_row_bytes = cfg->indiv_D_tile_bytes / cfg->max_tokens_per_expert;
        uint32_t d_offset    = shape_m * gate_up_n2 * d_row_bytes;
        __snax_bingo_kernel_dual_vc_swiglu_full_args_t swiglu_args = {
            .input_A_addr      = cfg->l1_a_addr + a_offset,
            .input_B_gate_addr = cfg->l1_b_gate_addr,
            .input_B_up_addr   = cfg->l1_b_up_addr,
            .output_D0_addr    = cfg->l1_d_addr + d_offset,
            .output_D1_addr    = cfg->l1_d1_scratch_addr,
            .M                 = m_s2,
            .K                 = cfg->indiv_K1,
            .N                 = gate_up_n2 * cfg->indiv_N_per_block / __moe_dyn_meshcol(shape_s2),
            .array_shape       = shape_s2,
            .rescale_mult      = cfg->rescale_mult,
            .rescale_shift     = cfg->rescale_shift,
        };
        return __snax_bingo_kernel_dual_vc_swiglu_full(&swiglu_args);
    }
}

/* ============================================================
 * S4: down 全量 GEMM（在 S3 pipeline 之后处理剩余/全部 token）
 *   - skip_s3=1 (cache hit): A 从偏移 0 开始，N = full_N，single call
 *   - skip_s3=0 (tail):      A 从 shape_m_down token 处开始，single full-N call
 * ============================================================ */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) {
        printf_safe("[C%d c%d]: moe_dynamic_expert_compute_down_full must run on core 0!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }

    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)arg;
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u || MOE_DYN_CTRL_SKIP_S4(cfg->ctrl) != 0u) {
        return BINGO_RET_SUCC;
    }

    uint32_t m_s4 = cfg->m_s4_exec;
    uint32_t shape_s3 = MOE_DYN_CTRL_SHAPE_S3(cfg->ctrl);  // S3's shape: used for tail A-offset only
    uint32_t shape_s4 = 2u;  // S4 固定 shape C (meshRow=2)，与 S3 动态 shape 解耦
    if (m_s4 == 0u) return BINGO_RET_SUCC;
    if (m_s4 > cfg->max_tokens_per_expert / 2u) {  /* m_s4 单位: M-tile；shape C meshRow=2 */
        printf_safe("[MoEDyn C%d] slot=%u m_s4_exec=%u exceeds max=%u\r\n",
                    snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), m_s4,
                    cfg->max_tokens_per_expert);
        return BINGO_RET_FAIL;
    }

    uint32_t down_n2 = __moe_dyn_s3_block_count(cfg);  /* down-proj N 方向 N2 分块数 */

    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) != 0u) {
        /* cache hit: 权重全部驻留 TCDM，single full-N down GEMM 处理所有 token */
        __snax_bingo_kernel_dual_vc_gemm_full_args_t down_args = {
            .input_A_addr   = cfg->l1_d_addr,
            .input_B0_addr  = cfg->l1_b_down_addr,
            .input_B1_addr  = cfg->l1_b_down_addr +
                down_n2 * cfg->indiv_down_B_tile_bytes,
            .output_D0_addr = cfg->l1_down_d_addr,
            .output_D1_addr = cfg->l1_down_d_addr +
                down_n2 * cfg->indiv_down_D_tile_bytes,
            .M              = m_s4,
            .K              = cfg->indiv_down_K1,
            .N              = down_n2 * cfg->indiv_down_N_per_block / __moe_dyn_meshcol(shape_s4),
            .array_shape    = shape_s4,
            .rescale_mult   = cfg->rescale_mult,
            .rescale_shift  = cfg->rescale_shift,
        };
        return __snax_bingo_kernel_dual_vc_gemm_full(&down_args);
    } else {
        /* 正常 tail：single full-N GEMM，与 cache-hit 路径对称。
         * A 跳过 S3 已处理的 shape_m_down 个 token（a_offset），
         * D 输出在各块内从 down_d_offset 处写入（与 S3 per-block 输出对齐）。*/
        uint32_t shape_m_down = __moe_dyn_shape_m(shape_s3);
        uint32_t d_row_bytes      = cfg->indiv_D_tile_bytes / cfg->max_tokens_per_expert;
        uint32_t a_offset         = shape_m_down * down_n2 * d_row_bytes;
        uint32_t down_d_row_bytes = cfg->indiv_down_D_tile_bytes / cfg->max_tokens_per_expert;
        uint32_t down_d_offset    = shape_m_down * down_d_row_bytes;
        __snax_bingo_kernel_dual_vc_gemm_full_args_t down_args = {
            .input_A_addr   = cfg->l1_d_addr + a_offset,
            .input_B0_addr  = cfg->l1_b_down_addr,
            .input_B1_addr  = cfg->l1_b_down_addr +
                down_n2 * cfg->indiv_down_B_tile_bytes,
            .output_D0_addr = cfg->l1_down_d_addr + down_d_offset,
            .output_D1_addr = cfg->l1_down_d_addr +
                down_n2 * cfg->indiv_down_D_tile_bytes + down_d_offset,
            .M              = m_s4,
            .K              = cfg->indiv_down_K1,
            .N              = down_n2 * cfg->indiv_down_N_per_block / __moe_dyn_meshcol(shape_s4),
            .array_shape    = shape_s4,
            .rescale_mult   = cfg->rescale_mult,
            .rescale_shift  = cfg->rescale_shift,
        };
        return __snax_bingo_kernel_dual_vc_gemm_full(&down_args);
    }
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
    if (MOE_DYN_CTRL_ACTIVE(cfg->ctrl) == 0u || cfg->ntokens == 0u) return BINGO_RET_SUCC;

    uint32_t row_bytes = cfg->indiv_down_D_tile_bytes / cfg->max_tokens_per_expert;
    uint32_t s3_blocks = __moe_dyn_s3_block_count(cfg);
    uint64_t expert_out_base = cfg->output_l3_base +
        (uint64_t)cfg->expert_id * (uint64_t)cfg->output_expert_stride_bytes;
    uint32_t slice_bytes = cfg->ntokens * row_bytes;
    uint32_t store_dma = (MOE_DYN_CTRL_DMA_S3(cfg->ctrl) != 0u) ? MOE_DYN_CTRL_DMA_S3(cfg->ctrl) : 1u;

    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    for (uint32_t n = 0; n < s3_blocks; n++) {
        uint64_t d0_dst = expert_out_base +
            (uint64_t)n * (uint64_t)cfg->indiv_down_D_tile_bytes +
            (uint64_t)cfg->token_start_rank * (uint64_t)row_bytes;
        uint64_t d1_dst = expert_out_base +
            (uint64_t)(s3_blocks + n) * (uint64_t)cfg->indiv_down_D_tile_bytes +
            (uint64_t)cfg->token_start_rank * (uint64_t)row_bytes;
        uint64_t d0_src = __moe_dyn_l1_wide(cfg->l1_down_d_addr + n * cfg->indiv_down_D_tile_bytes);
        uint64_t d1_src = __moe_dyn_l1_wide(cfg->l1_down_d_addr + (s3_blocks + n) * cfg->indiv_down_D_tile_bytes);
        uint32_t rc = __moe_dyn_copy_pair(store_dma,
                                          d0_dst, d0_src,
                                          d1_dst, d1_src,
                                          slice_bytes);
        if (rc != BINGO_RET_SUCC) return rc;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);

    printf_safe("[MoEDyn C%d] slot=%u expert=%u rank=%u ntok=%u done\r\n",
                snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
                cfg->token_start_rank, cfg->ntokens);
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
// Zeros the Mode-1 output region before computing.
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

    BINGO_TRACE_MARKER(BINGO_TRACE_L15_DOWN_CFG_START);
    __l15_cfg_mode1_streamer(cfg, tcdm_base);

    moe_set_dual_versacore_mode(1);
    moe_set_dual_versacore_csr(
        1, cfg->K1, cfg->N1 * cfg->M_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
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

    // ---- Start Mode-0, then overlap Mode-1 CSR writes with Mode-0 execution ----
    // Hardware double-buffer: non-START CSR writes go directly into regs_* (never
    // blocked). START CSR write is gated by io_readWriteRegIO_ready = ~cstate |
    // readers_all_done, so Mode-1 START below will hardware-stall until Mode-0
    // finishes and cstate=0, then snaps regs_* into csrCfgReg_* atomically.
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_START);
    moe_start_dual_vc_and_streamer();

    // ---- Mode-1 CSR pre-load: overlaps with Mode-0 execution (~1474 cc hidden) ----
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_START); // Mode-1 CSR config start
    __l15_cfg_mode1_streamer(cfg, tcdm_base);

    moe_set_dual_versacore_mode(1);  // Mode 1 = GEMM
    moe_set_dual_versacore_csr(
        1, cfg->K1, cfg->N1 * cfg->M_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_END);   // Mode-1 CSR pre-load end

    // Wait for Mode-0 to fully complete (poll VersaCore & Streamer writers)
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_END);

    // ---- Start Mode-1: csrCfgReg_* snaps from regs_* at this fire ----
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_END);

    return BINGO_RET_SUCC;
}

