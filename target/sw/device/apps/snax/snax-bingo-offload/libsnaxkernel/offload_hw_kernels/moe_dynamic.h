// Dual-VersaCore compute, DMA and dynamic individual-expert device kernels.
#pragma once

#include "../macros.h"
#include "moe_runtime_timing_record.h"
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

#ifndef MOE_INDIV_BRINGUP
#define MOE_INDIV_BRINGUP 0
#endif

#if MOE_INDIV_BRINGUP
#define MOE_INDIV_PRINT(...) printf_safe(__VA_ARGS__)
#else
#define MOE_INDIV_PRINT(...) do { } while (0)
#endif

#if MOE_RUNTIME_TIMING
typedef struct {
    uint32_t start;
    uint32_t resource_start;
    uint32_t resource_end;
    uint32_t peer_wait;
} __moe_runtime_timing_local_t;

#define MOE_PROFILE_BEGIN(name) \
    __moe_runtime_timing_local_t name = { \
        .start = snrt_mcycle(), .resource_start = 0u, \
        .resource_end = 0u, .peer_wait = 0u }

#define MOE_PROFILE_WAIT(name, statement) \
    do { \
        uint32_t __wait_start = snrt_mcycle(); \
        statement; \
        (name).peer_wait += snrt_mcycle() - __wait_start; \
    } while (0)

#define MOE_PROFILE_RESOURCE_BEGIN(name) \
    do { (name).resource_start = snrt_mcycle(); } while (0)

#define MOE_PROFILE_RESOURCE_END(name) \
    do { (name).resource_end = snrt_mcycle(); } while (0)

static inline void __moe_profile_commit(
    void *arg,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    __moe_runtime_timing_local_t *profile,
    uint32_t stage,
    uint32_t resource,
    uint32_t block,
    uint32_t units,
    uint32_t flags,
    uint32_t result)
{
    if (profile->resource_start == 0u) {
        profile->resource_start = profile->start;
        profile->resource_end = profile->start;
    }

    bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(
        arg, __snax_bingo_kernel_moe_dynamic_expert_block_args_t);
    sp->reserved[MOE_SP_PROFILE_MAGIC_IDX] = 0u;
    sp->start_time = profile->start;
    sp->reserved[MOE_SP_PROFILE_META_IDX] = MOE_PROFILE_META(
        stage, resource, snrt_cluster_idx(), snrt_cluster_core_idx(), block);
    sp->reserved[MOE_SP_PROFILE_TASK_IDX] = MOE_PROFILE_TASK(
        (cfg->ctrl >> 14u) & 0x3fu, cfg->expert_id, cfg->ntokens);
    sp->reserved[MOE_SP_PROFILE_RESOURCE_START_IDX] = profile->resource_start;
    sp->reserved[MOE_SP_PROFILE_RESOURCE_END_IDX] = profile->resource_end;
    sp->reserved[MOE_SP_PROFILE_PEER_WAIT_IDX] = profile->peer_wait;
    sp->reserved[MOE_SP_PROFILE_UNITS_IDX] = units;
    sp->reserved[MOE_SP_PROFILE_RESULT_IDX] = result;
    sp->reserved[MOE_SP_PROFILE_FLAGS_IDX] = flags | MOE_PROFILE_FLAG_ACTIVE;
    sp->end_time = snrt_mcycle();
    sp->reserved[MOE_SP_PROFILE_MAGIC_IDX] = MOE_RUNTIME_TIMING_MAGIC;
}

#define MOE_PROFILE_COMMIT(arg, cfg, name, stage, resource, block, units, flags, result) \
    __moe_profile_commit((arg), (cfg), &(name), (stage), (resource), \
                         (block), (units), (flags), (result))
#else
#define MOE_PROFILE_BEGIN(name)
#define MOE_PROFILE_WAIT(name, statement) do { statement; } while (0)
#define MOE_PROFILE_RESOURCE_BEGIN(name) do { } while (0)
#define MOE_PROFILE_RESOURCE_END(name) do { } while (0)
#define MOE_PROFILE_COMMIT(...) do { } while (0)
#endif

static inline uint32_t __moe_profile_dma_resource(uint32_t binding)
{
    return binding == 1u ? MOE_PROFILE_RESOURCE_IDMA :
        (binding == 2u ? MOE_PROFILE_RESOURCE_XDMA :
                         MOE_PROFILE_RESOURCE_DMA_BOTH);
}

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

__attribute__((always_inline)) static inline void
moe_set_dual_versacore_streamer_csr(
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
    csrw_ss(S_STRIDE_BASE_READER_0+0, Aslstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_0+1, Aslstride[1]);
    csrw_ss(T_BOUND_BASE_READER_0+0, Atlbound[0]);
    csrw_ss(T_BOUND_BASE_READER_0+1, Atlbound[1]);
    csrw_ss(T_BOUND_BASE_READER_0+2, Atlbound[2]);
    csrw_ss(T_BOUND_BASE_READER_0+3, Atlbound[3]);
    csrw_ss(T_BOUND_BASE_READER_0+4, Atlbound[4]);
    csrw_ss(T_BOUND_BASE_READER_0+5, Atlbound[5]);
    csrw_ss(T_STRIDE_BASE_READER_0+0, Atlstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_0+1, Atlstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_0+2, Atlstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_0+3, Atlstride[3]);
    csrw_ss(T_STRIDE_BASE_READER_0+4, Atlstride[4]);
    csrw_ss(T_STRIDE_BASE_READER_0+5, Atlstride[5]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, remap_A);
#else
    (void)remap_A;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, chan_en_A[0]);

    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(S_STRIDE_BASE_READER_1+0, B0slstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_1+1, B0slstride[1]);
    csrw_ss(T_BOUND_BASE_READER_1+0, B0tlbound[0]);
    csrw_ss(T_BOUND_BASE_READER_1+1, B0tlbound[1]);
    csrw_ss(T_BOUND_BASE_READER_1+2, B0tlbound[2]);
    csrw_ss(T_BOUND_BASE_READER_1+3, B0tlbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_1+0, B0tlstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_1+1, B0tlstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_1+2, B0tlstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_1+3, B0tlstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, remap_B0);
#else
    (void)remap_B0;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, chan_en_B0[0]);

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2+0, B1slstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_2+1, B1slstride[1]);
    csrw_ss(T_BOUND_BASE_READER_2+0, B1tlbound[0]);
    csrw_ss(T_BOUND_BASE_READER_2+1, B1tlbound[1]);
    csrw_ss(T_BOUND_BASE_READER_2+2, B1tlbound[2]);
    csrw_ss(T_BOUND_BASE_READER_2+3, B1tlbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_2+0, B1tlstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_2+1, B1tlstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_2+2, B1tlstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_2+3, B1tlstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, remap_B1);
#else
    (void)remap_B1;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, chan_en_B1[0]);

    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0, D0slstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+0, D0tlbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+1, D0tlbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_0+2, D0tlbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_0+3, D0tlbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, D0tlstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, D0tlstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, D0tlstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, D0tlstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, remap_D0);
#else
    (void)remap_D0;
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, chan_en_D0[0]);

    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1, D1slstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+0, D1tlbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+1, D1tlbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_1+2, D1tlbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_1+3, D1tlbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, D1tlstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, D1tlstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, D1tlstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, D1tlstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, remap_D1);
#else
    (void)remap_D1;
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, chan_en_D1[0]);
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
    // Match the standalone VersaCore protocol: drop each start signal before
    // waiting for its engine to become idle.  In particular, Mode-0 must
    // release STREAMER_START_CSR before fused L15 code reprograms and restarts
    // the streamer for Mode-1.
    csrw_ss(MOE_DUAL_VC_START, 0);
    csrw_ss(MOE_DUAL_VC_START, 0);
    while (csrr_ss(MOE_DUAL_VC_BUSY)) {}
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    while (csrr_ss(STREAMER_BUSY_CSR)) {}
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
__attribute__((always_inline)) static inline uint32_t
__moe_dual_vc_gemm_full_params(
    uint32_t A_addr,
    uint32_t B0_addr,
    uint32_t B1_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t array_shape,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    uint32_t meshRow = MOE_DUAL_VC_MESH_ROW_0 >> array_shape;
    uint32_t tileSize = MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t meshCol = MOE_DUAL_VC_MESH_COL_0 << array_shape;

    // A: INT16 token-contiguous rows.  multi_cluster_MoE routes the same
    // input_A buffer used by the dynamic path: each token has K*tileSize*2
    // payload bytes followed by the canonical L15 row padding. The GEMM reader skips the
    // padding by using a padded row stride while only streaming payload bytes.
    uint32_t a_row_stride = K * tileSize * 2u +
        BINGO_MOE_L15_ROW_PADDING_BYTES;
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

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_gemm_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_dual_vc_gemm_full_args_t *cfg =
        (const __snax_bingo_kernel_dual_vc_gemm_full_args_t *)arg;
    if (cfg->array_shape > 2u) return BINGO_RET_FAIL;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __moe_dual_vc_gemm_full_params(
        cfg->input_A_addr, cfg->input_B0_addr, cfg->input_B1_addr,
        cfg->output_D0_addr, cfg->output_D1_addr,
        cfg->M, cfg->K, cfg->N, cfg->array_shape,
        cfg->rescale_mult, cfg->rescale_shift);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_router_gemm_s0(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_router_gemm_s0_args_t *cfg =
        (const __snax_bingo_kernel_moe_router_gemm_s0_args_t *)arg;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __moe_dual_vc_gemm_full_params(
        cfg->input_A_addr, cfg->input_B0_addr, cfg->input_B1_addr,
        cfg->output_D0_addr, cfg->output_D1_addr,
        cfg->M, cfg->K, cfg->N, 0u,
        cfg->rescale_mult, cfg->rescale_shift);
}

__attribute__((always_inline)) static inline uint32_t
__moe_dual_vc_swiglu_full_params(
    uint32_t A_addr,
    uint32_t Bg_addr,
    uint32_t Bu_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t b_block_count,
    uint32_t b_block_stride,
    uint32_t array_shape,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    uint32_t meshRow = MOE_DUAL_VC_MESH_ROW_0 >> array_shape;
    uint32_t tileSize = MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t meshCol = MOE_DUAL_VC_MESH_COL_0 << array_shape;

    // A: INT16, 6-dim temporal (shared for gate and up), 2-dim spatial [2, 8] (hw bounds).
    // Dynamic gather preserves the L15 32-byte tail padding for every token.
    // The streamer reads only K*tileSize payload bytes and skips the tail.
    // Ats[0] = tileSize*2 (K stride: advance one K-tile within token row).
    // Ats[2] = meshRow * a_row_stride (M stride: skip meshRow padded rows).
    uint32_t a_row_stride = (uint32_t)K * (uint32_t)tileSize * 2u +
        BINGO_MOE_L15_ROW_PADDING_BYTES;
    int32_t Asl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)a_row_stride };
    int32_t Atb[6]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1, 1, 1 };
    int32_t Ats[6]       = { (int32_t)(tileSize * 2), 0,
                              (int32_t)(meshRow * a_row_stride), 0, 0, 0 };
    int32_t chan_en_A[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape) };

    // B0 (gate): INT4 packed, 4-dim temporal, 2-dim spatial (hw req.)
    // L15 B canonical layout: [N_tiles_S0][K_tiles][16 bytes].
    // B_sl[1] = K*tileSize*2 (K-section stride between outer spatial j-groups).
    // B_ts[0] = 16 (fixed k-stride for all shapes; NOT B_stream_bytes).
    // B_ts[1] = K * B_stream_bytes (correct N-tile stride, same for all shapes).
    uint32_t n_b_chan      = (array_shape == 0u) ? 2u : (array_shape == 1u) ? 4u : 8u;
    int32_t B_stream_bytes = (int32_t)(n_b_chan * (MOE_DUAL_VC_BANK_WIDTH / 8));
    uint32_t b_k_section = (uint32_t)K * (uint32_t)tileSize * 2u;
    uint32_t b_n_per_block = N / b_block_count;
    int32_t B0sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B0tb[4]        = { (int32_t)K, (int32_t)b_n_per_block,
                               (int32_t)b_block_count, (int32_t)M };
    int32_t B0ts[4]        = { (int32_t)(tileSize * 2u),
                               (int32_t)(K * B_stream_bytes),
                               (int32_t)b_block_stride, 0 };
    int32_t chan_en_B0[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // B1 (up): INT4 packed, same layout as B0, 2-dim spatial (hw req.)
    int32_t B1sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B1tb[4]        = { (int32_t)K, (int32_t)b_n_per_block,
                               (int32_t)b_block_count, (int32_t)M };
    int32_t B1ts[4]        = { (int32_t)(tileSize * 2u),
                               (int32_t)(K * B_stream_bytes),
                               (int32_t)b_block_stride, 0 };
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

/* ── ctrl-word field extractors ─────────────────────────────────────────────
 * Bit layout written into each dynamic slot by HW plan lowering or the SW
 * scheduler path:
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
 *   bits [19:14]: slot_id           (0-63)
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
#define MOE_DYN_CTRL_SLOT_ID(c)  (((c) >> 14u) & 63u) /* bits [19:14]: 6-bit slot_id (0..63) */
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
 * L3->L1 args flush; device compute nodes consume the final call dimensions. */
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

static inline uint64_t __moe_dyn_l1_wide(uint32_t local_addr)
{
    return chiplet_addr_transform((uint64_t)local_addr);
}

__attribute__((always_inline)) static inline uint32_t
__moe_dyn_run_down(uint32_t A_addr,
                   uint32_t B0_addr,
                   uint32_t B1_addr,
                   uint32_t D0_addr,
                   uint32_t D1_addr,
                   uint32_t M,
                   uint32_t K,
                   uint32_t N,
                   uint32_t array_shape,
                   uint32_t d_row_stride,
                   uint32_t rscl_mult,
                   uint32_t rscl_shift)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t meshCol = __moe_dyn_meshcol(array_shape);
    uint32_t b_stream_bytes =
        __moe_dyn_b_channels(array_shape) * (MOE_DUAL_VC_BANK_WIDTH / 8u);
    uint32_t b_k_section = K * 16u;
    uint32_t a_m_stride = ((K * MOE_DUAL_VC_TILE_SIZE_0) / meshCol) * 64u;
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

__attribute__((always_inline)) static inline uint32_t
__moe_dyn_run_down_shape_c(uint32_t A_addr,
                           uint32_t B0_addr,
                           uint32_t B1_addr,
                           uint32_t D0_addr,
                           uint32_t D1_addr,
                           uint32_t M,
                           uint32_t K,
                           uint32_t N,
                           uint32_t b_block_count,
                           uint32_t b_block_stride,
                           uint32_t d_row_stride,
                           uint32_t rscl_mult,
                           uint32_t rscl_shift)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    uint32_t b_k_section = K * 16u;
    uint32_t b_n_stride = K * 64u;
    uint32_t b_n_per_block = N / b_block_count;
    uint32_t a_m_stride = ((K * MOE_DUAL_VC_TILE_SIZE_2) /
                           MOE_DUAL_VC_MESH_COL_2) * 64u;
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
    csrw_ss(T_BOUND_BASE_READER_1+1, b_n_per_block);
    csrw_ss(T_BOUND_BASE_READER_1+2, b_block_count);
    csrw_ss(T_BOUND_BASE_READER_1+3, M);
    csrw_ss(T_STRIDE_BASE_READER_1+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_1+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_1+2, b_block_stride);
    csrw_ss(T_STRIDE_BASE_READER_1+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, MOE_DUAL_VC_CHAN_EN_B(2u));

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_2+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_2+0, K);
    csrw_ss(T_BOUND_BASE_READER_2+1, b_n_per_block);
    csrw_ss(T_BOUND_BASE_READER_2+2, b_block_count);
    csrw_ss(T_BOUND_BASE_READER_2+3, M);
    csrw_ss(T_STRIDE_BASE_READER_2+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_2+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_2+2, b_block_stride);
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

static inline void __moe_dyn_wait_xdma(uint64_t dst_addr, uint64_t src_addr, int32_t task_id)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
    xdma_wait_task(src_addr, dst_addr, (uint32_t)task_id);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
}

static inline volatile uint32_t *__moe_dyn_runtime_state(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (volatile uint32_t *)(uintptr_t)st->runtime_state_addr;
}

static inline uint32_t __moe_dyn_slot_active_this_round(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint32_t ctrl = cfg->ctrl;
    if (MOE_DYN_CTRL_ACTIVE(ctrl) == 0u || cfg->ntokens == 0u) return 0u;

    volatile uint32_t *state =
        (volatile uint32_t *)(uintptr_t)st->active_state_l1_addr;
    uint32_t active_idx = (MOE_DYN_CTRL_CLUSTER(ctrl) == 0u) ?
        MOE_DYN_RT_C2_ACTIVE_SLOTS : MOE_DYN_RT_C3_ACTIVE_SLOTS;
    return (MOE_DYN_CTRL_SLOT_ID(ctrl) < state[active_idx]) ? 1u : 0u;
}

static inline void __moe_dyn_wait_task_start(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint32_t wait_for_peer_slots = cfg->wait_for_peer_slots;
    if (wait_for_peer_slots == 0u) return;

    volatile uint32_t *state = __moe_dyn_runtime_state(st);
    uint32_t peer_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C3_DONE : MOE_DYN_RT_C2_DONE;
    while (state[peer_idx] < wait_for_peer_slots) {
        asm volatile("" ::: "memory");
    }
    asm volatile("fence r, rw" ::: "memory");
}

static inline void __moe_dyn_mark_task_complete(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    volatile uint32_t *state = __moe_dyn_runtime_state(st);
    uint32_t self_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C2_DONE : MOE_DYN_RT_C3_DONE;
    asm volatile("fence rw, w" ::: "memory");
    state[self_idx] = MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) + 1u;
    asm volatile("fence w, w" ::: "memory");
}

static inline uint32_t __moe_dyn_copy_pair(uint32_t binding,
                                           uint64_t dst0_addr,
                                           uint64_t src0_addr,
                                           uint64_t dst1_addr,
                                           uint64_t src1_addr,
                                           uint32_t bytes)
{
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

// main_bingo.py fixes each kernel to either GEMM core 0 or DM core 1. These
// workload-private entry points therefore do not repeat core-id checks at run time.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_init_output_padding(void *arg)
{
    const __snax_bingo_kernel_moe_init_output_padding_args_t *cfg =
        (const __snax_bingo_kernel_moe_init_output_padding_args_t *)arg;
    BINGO_TRACE_MARKER(BINGO_TRACE_MOE_OUTPUT_PADDING_INIT_START);
    uint32_t padding_bytes = cfg->row_stride_bytes - cfg->row_payload_bytes;
    for (uint32_t row = 0; row < cfg->rows; row++) {
        __moe_zero_tcdm(cfg->output_base + row * cfg->row_stride_bytes +
                            cfg->row_payload_bytes,
                        padding_bytes);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_MOE_OUTPUT_PADDING_INIT_END);
    return BINGO_RET_SUCC;
}

static inline uint32_t __moe_dyn_gather_slot_tokens(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t input_rows)
{
    /* Preserve the complete L15 row in L1. VersaCore reads only the payload;
     * the 32-byte tail rotates consecutive rows across TCDM banks. Submit every
     * scattered row to the cluster-local iDMA, then wait once for the batch. */
    uint32_t a_row_stride = st->A_token_bytes + BINGO_MOE_L15_ROW_PADDING_BYTES;
    uint16_t *token_ids = (uint16_t *)(uintptr_t)st->token_ids_addr;
    uint32_t expert_token_offset = cfg->expert_id * st->max_tokens_per_expert;

    for (uint32_t local_t = 0; local_t < cfg->ntokens; local_t++) {
        uint32_t token_id =
            token_ids[expert_token_offset + cfg->token_start_rank + local_t];
        uint64_t src = st->input_A_l3_base +
            (uint64_t)token_id * (uint64_t)a_row_stride;
        uint64_t dst = __moe_dyn_l1_wide(st->l1_a_addr + local_t * a_row_stride);
        MOE_INDIV_PRINT(
            "[INDIV_TOKEN] C%u slot=%u eid=%u rank=%u tid=%u "
            "src=0x%08x_%08x dst=0x%08x_%08x bytes=%u\r\n",
            snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl),
            cfg->expert_id, cfg->token_start_rank + local_t, token_id,
            (uint32_t)(src >> 32u), (uint32_t)src,
            (uint32_t)(dst >> 32u), (uint32_t)dst, a_row_stride);
        snrt_dma_start_1d_wideptr(dst, src, a_row_stride);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);

    if (input_rows > cfg->ntokens) {
        /* Rows introduced by the hardware minimum M have no source token. */
        __moe_zero_tcdm(
            st->l1_a_addr + cfg->ntokens * a_row_stride,
            (input_rows - cfg->ntokens) * a_row_stride);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_gather_s1(void *arg)
{
    __snax_bingo_kernel_moe_dynamic_expert_block_args_t *node =
        (__snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        node->task_arg_addr;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        node->static_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) return BINGO_RET_SUCC;
    MOE_PROFILE_BEGIN(profile);
    MOE_PROFILE_WAIT(profile, __moe_dyn_wait_task_start(cfg, st));

    MOE_INDIV_PRINT(
        "[INDIV_BEGIN] C%u slot=%u eid=%u start=%u ntok=%u "
        "shape_s1=%u shape_s3=%u skip=%u%u%u%u wait_peer=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->token_start_rank, cfg->ntokens,
        MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl), MOE_DYN_CTRL_SHAPE_S3(cfg->ctrl),
        MOE_DYN_CTRL_SKIP_S1(cfg->ctrl), MOE_DYN_CTRL_SKIP_S2(cfg->ctrl),
        MOE_DYN_CTRL_SKIP_S3(cfg->ctrl), MOE_DYN_CTRL_SKIP_S4(cfg->ctrl),
        cfg->wait_for_peer_slots);

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_START);
    uint32_t input_rows = 0u;
    if (MOE_DYN_CTRL_SKIP_S1(cfg->ctrl) == 0u)
        input_rows += __moe_dyn_shape_m(MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl));
    if (MOE_DYN_CTRL_SKIP_S2(cfg->ctrl) == 0u)
        input_rows += cfg->m_s2_exec * MOE_DUAL_VC_MESH_ROW_2;
    if (input_rows < cfg->ntokens) input_rows = cfg->ntokens;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t rc = __moe_dyn_gather_slot_tokens(cfg, st, input_rows);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_GATHER_DONE] C%u slot=%u eid=%u ntok=%u rows=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->ntokens, input_rows, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_GATHER_S1,
        MOE_PROFILE_RESOURCE_IDMA,
        0u,
        cfg->ntokens * (st->A_token_bytes + BINGO_MOE_L15_ROW_PADDING_BYTES),
        0u, rc);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    uint32_t n = blk->block_idx;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t s1_blocks = st->s1_block_count;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    if (n >= s1_blocks || MOE_DYN_CTRL_SKIP_S1(cfg->ctrl) != 0u) {
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S1,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_START);
    uint32_t weight_offset = n * st->indiv_B_block_stride;
    uint32_t dma_binding = MOE_DYN_CTRL_DMA_S1(cfg->ctrl);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t rc = __moe_dyn_copy_pair(
        dma_binding,
        __moe_dyn_l1_wide(st->l1_b_gate_addr + weight_offset),
        st->indiv_gate_B_l3 +
            (uint64_t)cfg->expert_id * st->indiv_B_expert_stride + weight_offset,
        __moe_dyn_l1_wide(st->l1_b_up_addr + weight_offset),
        st->indiv_up_B_l3 +
            (uint64_t)cfg->expert_id * st->indiv_B_expert_stride + weight_offset,
        st->indiv_B_tile_bytes);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_LOAD_S1_DONE] C%u slot=%u eid=%u block=%u dma=%u "
        "bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, MOE_DYN_CTRL_DMA_S1(cfg->ctrl), st->indiv_B_tile_bytes, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S1,
        __moe_profile_dma_resource(dma_binding), n,
        2u * st->indiv_B_tile_bytes, 0u, rc);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    uint32_t n = blk->block_idx;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    if (n >= 2u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S1,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_INVALID_CALL,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_s1_call_args_t *call = &cfg->s1_call[n];
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S1,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_INVALID_CALL,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result = __moe_dual_vc_swiglu_full_params(
        st->l1_a_addr,
        st->l1_b_gate_addr + n * st->indiv_B_block_stride,
        st->l1_b_up_addr + n * st->indiv_B_block_stride,
        call->output_D0_addr,
        st->l1_d1_scratch_addr,
        1u,
        st->indiv_K1,
        call->N,
        1u,
        st->indiv_B_block_stride,
        call->array_shape,
        st->rescale_mult,
        st->rescale_shift);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S1_DONE] C%u slot=%u eid=%u block=%u shape=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, call->array_shape, call->N, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S1,
        MOE_PROFILE_RESOURCE_VERSACORE, n, 0u, 0u, result);
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_down_block(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    uint32_t n = blk->block_idx;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t ctrl = cfg->ctrl;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    uint32_t s3_blocks = st->s3_block_count;
    if (MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u || n >= s3_blocks) {
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S3,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_START);
    uint64_t down_src = st->indiv_down_B_l3 +
        (uint64_t)cfg->expert_id * st->indiv_down_B_expert_stride;
    uint32_t left_offset = n * st->indiv_down_B_block_stride;
    uint32_t right_offset =
        (st->s3_block_count + n) * st->indiv_down_B_block_stride;
    uint32_t dma_binding = MOE_DYN_CTRL_DMA_S3(ctrl);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t rc = __moe_dyn_copy_pair(
        dma_binding,
        __moe_dyn_l1_wide(st->l1_b_down_addr + left_offset),
        down_src + left_offset,
        __moe_dyn_l1_wide(st->l1_b_down_addr + right_offset),
        down_src + right_offset,
        st->indiv_down_B_tile_bytes);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_LOAD_S3_DONE] C%u slot=%u eid=%u block=%u dma=%u "
        "bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, MOE_DYN_CTRL_DMA_S3(ctrl), st->indiv_down_B_tile_bytes, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S3,
        __moe_profile_dma_resource(dma_binding), n,
        2u * st->indiv_down_B_tile_bytes, 0u, rc);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_block(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    uint32_t n = blk->block_idx;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t ctrl = cfg->ctrl;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    if (n >= 2u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_INVALID_CALL,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[n];
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_INVALID_CALL,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    (void)ctrl;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_START);
    uint32_t row_bytes =
        st->indiv_down_D_tile_bytes / st->max_tokens_per_expert;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result =
        __moe_dyn_run_down(
            st->l1_d_addr,
            st->l1_b_down_addr + n * st->indiv_down_B_block_stride,
            st->l1_b_down_addr +
                (st->s3_block_count + n) * st->indiv_down_B_block_stride,
            st->l1_down_d_addr + n * row_bytes,
            st->l1_down_d_addr + st->A_token_bytes / 2u + n * row_bytes,
            1u,
            st->indiv_down_K1,
            call->N,
            call->array_shape,
            st->A_token_bytes + BINGO_MOE_L15_ROW_PADDING_BYTES,
            st->rescale_mult,
            st->rescale_shift);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S3_DONE] C%u slot=%u eid=%u block=%u shape=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, call->array_shape, call->N, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
        MOE_PROFILE_RESOURCE_VERSACORE, n, 0u, 0u, result);
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t slot = MOE_DYN_DMA_SLOT_S2_PREFETCH;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_START);
    uint32_t expert_id = (uint32_t)cfg->dma_slot_expert_id[slot];
    uint32_t half_bytes = st->s3_block_count * st->indiv_down_B_block_stride;
    uint64_t down_src = st->indiv_down_B_l3 +
        (uint64_t)expert_id * st->indiv_down_B_expert_stride;
    uint32_t dma_binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t rc = __moe_dyn_copy_pair(
        dma_binding,
        __moe_dyn_l1_wide(st->l1_b_down_addr), down_src,
        __moe_dyn_l1_wide(st->l1_b_down_addr + half_bytes),
        down_src + half_bytes,
        half_bytes);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S2_DONE] C%u slot=%u eid=%u target_eid=%u "
        "dma=%u bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        expert_id, MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot), half_bytes, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S2,
        __moe_profile_dma_resource(dma_binding), 0u,
        2u * half_bytes, 0u, rc);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t slot = MOE_DYN_DMA_SLOT_S4_PREFETCH;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_START);
    uint32_t expert_id = (uint32_t)cfg->dma_slot_expert_id[slot];
    uint32_t weight_bytes = st->s1_block_count * st->indiv_B_block_stride;
    uint32_t dma_binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t rc = __moe_dyn_copy_pair(
        dma_binding,
        __moe_dyn_l1_wide(st->l1_b_gate_addr),
        st->indiv_gate_B_l3 +
            (uint64_t)expert_id * st->indiv_B_expert_stride,
        __moe_dyn_l1_wide(st->l1_b_up_addr),
        st->indiv_up_B_l3 +
            (uint64_t)expert_id * st->indiv_B_expert_stride,
        weight_bytes);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S4_DONE] C%u slot=%u eid=%u target_eid=%u "
        "dma=%u bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        expert_id, MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot), weight_bytes, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
        __moe_profile_dma_resource(dma_binding), 0u,
        2u * weight_bytes, 0u, rc);
    return rc;
}

/* ============================================================
 * S2: gate+up 全量 GEMM（在S1 pipeline 之后处理剩余/全部 token）
 *   - skip_s1=1 (cache hit): Prepare 令 A 从偏移 0 开始
 *   - skip_s1=0 (tail):      Prepare 令 A/D 从 S1 prefix 后开始
 * Device side only consumes s2_call and issues a single full-N shape-C call.
 * ============================================================ */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    const __snax_bingo_moe_dyn_s2_call_args_t *call = &cfg->s2_call;
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_INVALID_CALL,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_START);
    uint32_t s1_blocks = st->s1_block_count;
    uint32_t n_shape_c = s1_blocks * st->indiv_N_per_block /
        __moe_dyn_meshcol(2u);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result = __moe_dual_vc_swiglu_full_params(
        call->input_A_addr,
        st->l1_b_gate_addr,
        st->l1_b_up_addr,
        call->output_D0_addr,
        st->l1_d1_scratch_addr,
        call->M,
        st->indiv_K1,
        n_shape_c,
        s1_blocks,
        st->indiv_B_block_stride,
        2u,
        st->rescale_mult,
        st->rescale_shift);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S2_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        call->M, n_shape_c, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S2,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u, 0u, 0u, result);
    return result;
}

/* ============================================================
 * S4: down 全量 GEMM（在 S3 pipeline 之后处理剩余/全部 token）
 *   - skip_s3=1 (cache hit): Prepare 令 A/D 从偏移 0 开始
 *   - skip_s3=0 (tail):      Prepare 令 A/D 从 S3 prefix 后开始
 *
 * skip_s3=1 mirrors SwiGLU full: one hardware GEMM covers all output N-blocks.
 * It writes l1_down_d in token-major padded rows; store() preserves those rows
 * in the per-expert L3 output region.
 *
 * skip_s3=0 tail also uses one full-N GEMM and appends its rows to the same
 * l1_down_d token-major full-N layout created by S3.
 * ============================================================ */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_full(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t ctrl = cfg->ctrl;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    const __snax_bingo_moe_dyn_s4_call_args_t *call = &cfg->s4_call;
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_INVALID_CALL,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    (void)ctrl;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_START);
    uint32_t s3_blocks = st->s3_block_count;
    uint32_t n_shape_c = s3_blocks * st->indiv_down_N_per_block /
        __moe_dyn_meshcol(2u);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result = __moe_dyn_run_down_shape_c(
        call->input_A_addr,
        st->l1_b_down_addr,
        st->l1_b_down_addr + s3_blocks * st->indiv_down_B_block_stride,
        call->output_D0_addr,
        call->output_D1_addr,
        call->M,
        st->indiv_down_K1,
        n_shape_c,
        s3_blocks,
        st->indiv_down_B_block_stride,
        st->A_token_bytes + BINGO_MOE_L15_ROW_PADDING_BYTES,
        st->rescale_mult,
        st->rescale_shift);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S4_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        call->M, n_shape_c, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u, 0u, 0u, result);
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_store(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) return BINGO_RET_SUCC;
    MOE_PROFILE_BEGIN(profile);

    uint32_t row_stride = st->A_token_bytes + BINGO_MOE_L15_ROW_PADDING_BYTES;
    uint64_t expert_out_base = st->output_l3_base +
        (uint64_t)cfg->expert_id * (uint64_t)st->output_expert_stride_bytes;
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_START);
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u ||
        MOE_DYN_CTRL_SKIP_S4(cfg->ctrl) == 0u) {
        uint64_t dst = expert_out_base +
            (uint64_t)cfg->token_start_rank * (uint64_t)row_stride;
        uint64_t src = __moe_dyn_l1_wide(st->l1_down_d_addr);
        uint32_t bytes = cfg->ntokens * row_stride;
        MOE_INDIV_PRINT(
            "[INDIV_STORE_BEGIN] C%u slot=%u eid=%u start=%u ntok=%u "
            "src=0x%08x_%08x dst=0x%08x_%08x bytes=%u dma=%u\r\n",
            snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
            cfg->token_start_rank, cfg->ntokens,
            (uint32_t)(src >> 32u), (uint32_t)src,
            (uint32_t)(dst >> 32u), (uint32_t)dst, bytes,
            1u);
        MOE_PROFILE_RESOURCE_BEGIN(profile);
        __moe_dyn_idma_copy(dst, src, bytes);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
        MOE_PROFILE_RESOURCE_END(profile);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_END);

        __moe_dyn_mark_task_complete(cfg, st);
        MOE_INDIV_PRINT(
            "[INDIV_DONE] C%u slot=%u eid=%u start=%u ntok=%u bytes=%u\r\n",
            snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
            cfg->token_start_rank, cfg->ntokens, bytes);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_STORE,
            MOE_PROFILE_RESOURCE_IDMA, 0u, bytes, 0u,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_END);

    __moe_dyn_mark_task_complete(cfg, st);
    MOE_INDIV_PRINT(
        "[INDIV_DONE] C%u slot=%u eid=%u start=%u ntok=%u bytes=0\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->token_start_rank, cfg->ntokens);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_STORE,
        MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
        MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_STORE,
        BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

// ============================================================
// Fused-L15 streamer CSR helpers. These consume the named 91-word L15 config
// image directly and keep every CSR address compile-time constant. The generic
// streamer helper is also expanded, but these helpers preserve the distinct
// Mode-0 and Mode-1 field sets without constructing temporary arrays.
// ============================================================

__attribute__((always_inline)) static inline void
__l15_cfg_mode0_streamer(
    const __snax_bingo_moe_l15_shape_cfg_t *cfg,
    uint32_t tcdm_base)
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
__l15_cfg_mode1_streamer(
    const __snax_bingo_moe_l15_shape_cfg_t *cfg,
    uint32_t tcdm_base)
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
// Dual-VersaCore L15 MoE kernel: Mode-0 (SwiGLU) + Mode-1 (down-proj GEMM)
// All tensors must be staged in the full-size L15 layout before this call.
//
// Args (uint32_t array, 4 fields):
//   [0] = __snax_bingo_moe_l15_shape_cfg_t * (device TCDM ptr)
//   [1] = tcdm_base   (absolute L1 address of tensor region start)
//   [2] = rescale_mult
//   [3] = rescale_shift
// ============================================================
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_l15_moe_full(void *arg)
{
    const __snax_bingo_moe_l15_shape_cfg_t *cfg =
        (const __snax_bingo_moe_l15_shape_cfg_t *)(uintptr_t)
        ((uint32_t *)arg)[0];
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

    // ---- Start Mode-0, then preload Mode-1 while Mode-0 is running ----
    // Non-START writes update the staging CSR bank. The active Mode-0 bank was
    // latched by START and is unaffected; the next START atomically publishes
    // the staged Mode-1 configuration.
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_START);
    moe_start_dual_vc_and_streamer();

    // ---- Mode-1 preload: down projection (D0 as A, W2L/W2R -> D0/D1) ----
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_START);
    __l15_cfg_mode1_streamer(cfg, tcdm_base);

    moe_set_dual_versacore_mode(1);  // Mode 1 = GEMM
    uint32_t mode1_output_tiles =
        (uint32_t)cfg->mode1_D_tbound[2] * (uint32_t)cfg->mode1_D_tbound[3];
    moe_set_dual_versacore_csr(
        1, cfg->K1, mode1_output_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_END);

    // Release both START CSRs and wait for the active Mode-0 bank to finish.
    // Clearing START only touches the launch registers; it does not discard
    // the staged Mode-1 configuration written above.
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_END);

    // ---- Publish the preloaded Mode-1 bank and run it ----
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_END);

    return BINGO_RET_SUCC;
}
