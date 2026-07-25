// Dual-VersaCore compute, DMA and dynamic individual-expert device kernels.
#pragma once

#include "../macros.h"
#include "moe_runtime_timing_record.h"
#include <snax_versacore_lib.h>
#include <snax_xdma_lib.h>

#define MOE_DUAL_VC_BANK_WIDTH 64u
#define MOE_BANK_TCDM_ROW_BYTES 512u
#define MOE_BANK_WEIGHT_ROW_BYTES 64u
#define MOE_BANK_A_TOKEN_TILE_BYTES 16u
#define MOE_BANK_TOKEN_LANES 8u
#define MOE_BANK_B_PHASE_DELTA_BYTES 128u
#define MOE_BANK_B0_PING_OFFSET 128u
#define MOE_BANK_B1_PING_OFFSET 192u
#define MOE_BANK_MODE0_D_OFFSET 384u
#define MOE_BANK_MODE1_D0_OFFSET 0u
#define MOE_BANK_MODE1_D1_OFFSET 64u
#define MOE_PIPELINE_CTRL_S2_DELTA_BYTES 512u
#define MOE_PIPELINE_CTRL_SLOT_BYTES 1024u
#define MOE_PIPELINE_INIT_COOKIE 0x4d4f4550u
#define MOE_PIPELINE_S3_SYNC_COOKIE 0x53335330u
#define MOE_PIPELINE_COMPUTE_READY_BIT 0x1u

#define MOE_S4_CSR_LAYOUT_SEQUENTIAL 0u
#define MOE_S4_CSR_LAYOUT_BLOCK_SYNC 1u

typedef struct __attribute__((aligned(8))) {
    uint64_t gate_src_base;
    uint64_t up_src_base;
    uint32_t gate_dst_base;
    uint32_t up_dst_base;
    uint32_t block_bytes;
    uint32_t block_count;
    uint32_t binding;
    uint32_t valid;
    volatile uint32_t csr_prepared_stage;
    volatile uint32_t xdma_prepared_stage;
    volatile uint32_t csr_prepared_reserved;
} __moe_s1_dma_ctrl_t;

typedef struct __attribute__((aligned(8))) {
    uint64_t down_src_base;
    uint32_t down_dst_base;
    uint32_t half_bytes;
    uint32_t block_bytes;
    uint32_t block_count;
    uint32_t s1_block_count;
    uint32_t binding;
    uint32_t valid;
    uint32_t sync_enabled;
    volatile uint32_t compute_done;
    volatile uint32_t prefetch_done;
    volatile uint32_t error;
    volatile uint32_t store_prepared;
    uint32_t reserved;
} __moe_s2_prefetch_ctrl_t;

_Static_assert(sizeof(__moe_s1_dma_ctrl_t) <= 64u,
               "S1 DMA control must remain inside banks 56..63");
_Static_assert(sizeof(__moe_s2_prefetch_ctrl_t) <= 64u,
               "S2 prefetch control must remain inside banks 56..63");

#define MOE_CSR_PREPARED_NONE 0u
#define MOE_CSR_PREPARED_S2   2u
#define MOE_CSR_PREPARED_S3   3u
#define MOE_CSR_PREPARED_S4   4u

#define MOE_XDMA_PREPARED_NONE 0u
#define MOE_XDMA_PREPARED_S1   1u
#define MOE_XDMA_PREPARED_S3   3u
#define MOE_XDMA_PREPARED_S4PF 5u

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
    uint32_t resource_units;
} __moe_runtime_timing_local_t;

#define MOE_PROFILE_BEGIN(name) \
    __moe_runtime_timing_local_t name = { \
        .start = snrt_mcycle(), .resource_start = 0u, \
        .resource_end = 0u, .peer_wait = 0u, .resource_units = 0u }

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

#define MOE_PROFILE_CAPTURE_VC_COUNTER(name) \
    do { (name).resource_units = read_dual_vc_perf_counter(); } while (0)

#define MOE_PROFILE_RESOURCE_UNITS(name) ((name).resource_units)

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
#define MOE_PROFILE_CAPTURE_VC_COUNTER(name) do { } while (0)
#define MOE_PROFILE_RESOURCE_UNITS(name) 0u
#define MOE_PROFILE_COMMIT(...) do { } while (0)
#endif

static inline uint32_t __moe_profile_dma_resource(uint32_t binding)
{
    return binding == 1u ? MOE_PROFILE_RESOURCE_IDMA :
        (binding == 2u ? MOE_PROFILE_RESOURCE_XDMA :
                         MOE_PROFILE_RESOURCE_DMA_BOTH);
}

// Zero a TCDM byte range on the current core for semantically observable gaps.
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
    // payload bytes in a larger physical row. The GEMM reader streams only the
    // payload and uses the L15 row stride to skip the trailing address gap.
    uint32_t a_row_stride = K * tileSize * 2u +
        BINGO_MOE_L15_ROW_GAP_BYTES;
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

__attribute__((always_inline)) static inline void
__moe_configure_swiglu(
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
    // Dynamic gather copies only each token payload into row starts separated
    // by the L15 stride. The streamer never accesses the 32-byte row gaps.
    // Ats[0] = tileSize*2 (K stride: advance one K-tile within token row).
    // Ats[2] = meshRow * a_row_stride (M stride: advance by physical rows).
    uint32_t a_row_stride = (uint32_t)K * (uint32_t)tileSize * 2u +
        BINGO_MOE_L15_ROW_GAP_BYTES;
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
#define MOE_DYN_DMA_EID(eids, s) (((eids) >> ((s)*6u)) & 0x3fu)

#define MOE_DYN_DMA_SLOT_S1          0u
#define MOE_DYN_DMA_SLOT_S3          1u
#define MOE_DYN_DMA_SLOT_S2_PREFETCH 2u
#define MOE_DYN_DMA_SLOT_S4_PREFETCH 3u
#define MOE_DYN_DMA_IDMA             1u
#define MOE_DYN_DMA_XDMA             2u
#define MOE_DYN_DMA_BOTH             3u
#define MOE_DYN_RT_C2_ACTIVE_SLOTS   2u
#define MOE_DYN_RT_C3_ACTIVE_SLOTS   3u

static inline __moe_s1_dma_ctrl_t *__moe_s1_dma_ctrl(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk)
{
    return (__moe_s1_dma_ctrl_t *)(uintptr_t)blk->pipeline_ctrl_addr;
}

static inline __moe_s2_prefetch_ctrl_t *__moe_s2_prefetch_ctrl(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk)
{
    return (__moe_s2_prefetch_ctrl_t *)(uintptr_t)(
        blk->pipeline_ctrl_addr + MOE_PIPELINE_CTRL_S2_DELTA_BYTES);
}

static inline void __moe_pipeline_publish(
    volatile uint32_t *counter, uint32_t value)
{
    asm volatile("fence rw, rw" ::: "memory");
    *counter = value;
    asm volatile("fence rw, rw" ::: "memory");
}

static inline uint32_t __moe_pipeline_wait(
    volatile uint32_t *counter, uint32_t value,
    volatile uint32_t *error)
{
    while (*counter < value) {
        if (*error != BINGO_RET_SUCC) return *error;
    }
    asm volatile("fence rw, rw" ::: "memory");
    return (*error == BINGO_RET_SUCC) ? BINGO_RET_SUCC : *error;
}

static inline uint32_t __moe_pipeline_wait_cookie(
    volatile uint32_t *cookie, uint32_t base,
    uint32_t require_compute_ready, volatile uint32_t *error)
{
    for (;;) {
        uint32_t value = *cookie;
        uint32_t base_matches =
            (value & ~MOE_PIPELINE_COMPUTE_READY_BIT) == base;
        uint32_t ready_matches = require_compute_ready == 0u ||
            (value & MOE_PIPELINE_COMPUTE_READY_BIT) != 0u;
        if (base_matches && ready_matches) break;
        if (*error != BINGO_RET_SUCC) return *error;
    }
    asm volatile("fence rw, rw" ::: "memory");
    return (*error == BINGO_RET_SUCC) ? BINGO_RET_SUCC : *error;
}

static inline uint32_t __moe_dyn_binding_uses_xdma(uint32_t binding)
{
    return binding == MOE_DYN_DMA_XDMA || binding == MOE_DYN_DMA_BOTH;
}

static inline uint32_t __moe_dyn_shape_valid(uint32_t shape)
{
    return shape <= 2u;
}

static inline uint32_t __moe_dyn_stage_block_n(
    uint32_t stage_n, uint32_t block_count)
{
    if (block_count == 0u || stage_n == 0u ||
        stage_n % block_count != 0u) {
        return 0u;
    }
    return stage_n / block_count;
}

static inline uint32_t __moe_s4_block_prefetch_layout_valid(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return st->s1_block_count != 0u &&
        st->indiv_B_block_stride != 0u &&
        st->indiv_B_block_stride % MOE_BANK_WEIGHT_ROW_BYTES == 0u;
}

static inline uint32_t __moe_s4_block_compute_layout_valid(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return cfg->s4_call.valid != 0u && cfg->s4_call.M != 0u &&
        __moe_dyn_shape_valid(cfg->s4_call.array_shape) != 0u &&
        st->s3_block_count != 0u &&
        __moe_dyn_stage_block_n(
            cfg->s4_call.N, st->s3_block_count) != 0u;
}

static inline uint32_t __moe_s4_block_initial_phase(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (st->s3_block_count - 1u) & 1u;
}

static inline uint32_t __moe_s4_block_at_step(
    uint32_t step, uint32_t block_count, uint32_t initial_phase,
    uint32_t *block)
{
    uint32_t n = step ^ initial_phase;
    if (n >= block_count) return 0u;
    *block = n;
    return 1u;
}

static inline uint32_t __moe_s4_block_step_count(
    uint32_t block_count, uint32_t initial_phase)
{
    return block_count +
        ((initial_phase != 0u && (block_count & 1u) != 0u) ? 1u : 0u);
}

static inline uint32_t __moe_s4_sync_step_count(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint32_t initial_phase = __moe_s4_block_initial_phase(st);
    uint32_t dma_steps = __moe_s4_block_step_count(
        st->s1_block_count, initial_phase);
    uint32_t compute_steps = __moe_s4_block_step_count(
        st->s3_block_count, initial_phase);
    return dma_steps > compute_steps ? dma_steps : compute_steps;
}

static inline uint32_t __moe_xdma_stage_is_prepared(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    uint32_t stage)
{
    asm volatile("fence rw, rw" ::: "memory");
    return __moe_s1_dma_ctrl(blk)->xdma_prepared_stage == stage;
}

static inline void __moe_prepare_pair_xdma_shape(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    uint32_t binding, uint32_t bytes, uint32_t stage)
{
    if (__moe_dyn_binding_uses_xdma(binding) == 0u) return;
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
    xdma_memcpy_2d_fast_configure(
        MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES,
        bytes / MOE_BANK_WEIGHT_ROW_BYTES);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    __moe_pipeline_publish(
        &__moe_s1_dma_ctrl(blk)->xdma_prepared_stage, stage);
}

static inline void __moe_prepare_s3_xdma_shape(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) != 0u) return;
    __moe_prepare_pair_xdma_shape(
        blk, MOE_DYN_CTRL_DMA_S3(cfg->ctrl),
        st->indiv_down_B_block_stride, MOE_XDMA_PREPARED_S3);
}

static inline void __moe_prepare_s1_xdma_shape(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    if (MOE_DYN_CTRL_SKIP_S1(cfg->ctrl) != 0u) return;
    __moe_prepare_pair_xdma_shape(
        blk, MOE_DYN_CTRL_DMA_S1(cfg->ctrl),
        st->indiv_B_block_stride, MOE_XDMA_PREPARED_S1);
}

static inline void __moe_prepare_s4pf_xdma_shape(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint32_t slot = MOE_DYN_DMA_SLOT_S4_PREFETCH;
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) return;
    uint32_t binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    if (__moe_dyn_binding_uses_xdma(binding) == 0u) return;
    __moe_prepare_pair_xdma_shape(
        blk, MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot),
        st->indiv_B_block_stride, MOE_XDMA_PREPARED_S4PF);
}

static inline uint32_t __moe_dyn_shape_m(uint32_t shape)
{
    if (shape == 0u) return 8u;
    if (shape == 1u) return 4u;
    return 2u;
}

/* Return per-VC meshCol for the selected hardware shape.
 * S0: meshCol=4, S1: meshCol=8, S2: meshCol=16. Device compute nodes use it
 * only to derive shape-dependent streamer and full-N dimensions. */
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

static inline uint64_t __moe_dyn_l1_wide(uint32_t local_addr)
{
    return chiplet_addr_transform((uint64_t)local_addr);
}

static inline uint32_t __moe_bank_token_page_span(uint32_t token_bytes)
{
    return (token_bytes / MOE_BANK_A_TOKEN_TILE_BYTES) *
        MOE_BANK_TCDM_ROW_BYTES;
}

static inline uint32_t __moe_bank_paged_token_addr(
    uint32_t base, uint32_t token, uint32_t page_span,
    uint32_t lane_bytes)
{
    return base + (token / MOE_BANK_TOKEN_LANES) * page_span +
        (token % MOE_BANK_TOKEN_LANES) * lane_bytes;
}

static inline uint32_t __moe_bank_a_addr(
    uint32_t base, uint32_t token, uint32_t token_bytes)
{
    return __moe_bank_paged_token_addr(
        base, token, __moe_bank_token_page_span(token_bytes),
        MOE_BANK_A_TOKEN_TILE_BYTES);
}

static inline uint32_t __moe_bank_mode0_output_addr(
    uint32_t base, uint32_t token_start, uint32_t block,
    uint32_t chunk_cols, uint32_t block_count)
{
    uint32_t block_span =
        (chunk_cols / MOE_BANK_TOKEN_LANES) * MOE_BANK_TCDM_ROW_BYTES;
    return __moe_bank_paged_token_addr(
               base, token_start, block_count * block_span,
               MOE_BANK_A_TOKEN_TILE_BYTES) +
        block * block_span;
}

static inline uint32_t __moe_bank_mode1_output_addr(
    uint32_t base, uint32_t token_start, uint32_t block,
    uint32_t chunk_cols, uint32_t block_count)
{
    uint32_t block_span = (chunk_cols / 4u) * MOE_BANK_TCDM_ROW_BYTES;
    return __moe_bank_paged_token_addr(
               base, token_start, block_count * block_span, 8u) +
        block * block_span;
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode0_block_bases(
    uint32_t B0_addr, uint32_t B1_addr, uint32_t D_addr)
{
    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(BASE_PTR_WRITER_0_LOW, D_addr);
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode0_run_bases(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr, uint32_t D_addr)
{
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    __moe_bank_patch_mode0_block_bases(B0_addr, B1_addr, D_addr);
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode1_block_bases(
    uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr)
{
    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode1_run_bases(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr)
{
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    __moe_bank_patch_mode1_block_bases(B0_addr, B1_addr, D0_addr, D1_addr);
}

static inline void __moe_bank_configure_mode0(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr, uint32_t D_addr,
    uint32_t K, uint32_t N, uint32_t array_shape,
    uint32_t rscl_mult, uint32_t rscl_shift)
{
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t panel_span = (K / 4u) * MOE_BANK_TCDM_ROW_BYTES;
    int32_t Asl[2] = {8, 16};
    int32_t Atb[6] = {(int32_t)K, (int32_t)N, 1, 1, 1, 1};
    int32_t Ats[6] = {MOE_BANK_TCDM_ROW_BYTES, 0, 0, 0, 0, 0};
    int32_t Bsl[2] = {8, (int32_t)panel_span};
    int32_t Btb[4] = {4, (int32_t)(K / 4u), (int32_t)N, 1};
    int32_t Bts[4] = {
        16, MOE_BANK_TCDM_ROW_BYTES,
        (int32_t)((1u << array_shape) * panel_span), 0};
    int32_t Dsl[1] = {8};
    int32_t Dtb[4];
    int32_t Dts[4];
    if (array_shape == 0u) {
        Dtb[0] = 1; Dtb[1] = 8; Dtb[2] = 2; Dtb[3] = (int32_t)(N / 2u);
        Dts[0] = 8; Dts[1] = 16; Dts[2] = 8; Dts[3] = 512;
    } else if (array_shape == 1u) {
        Dtb[0] = 2; Dtb[1] = 4; Dtb[2] = 1; Dtb[3] = (int32_t)N;
        Dts[0] = 8; Dts[1] = 16; Dts[2] = 0; Dts[3] = 512;
    } else {
        Dtb[0] = 2; Dtb[1] = 2; Dtb[2] = 2; Dtb[3] = (int32_t)N;
        Dts[0] = 8; Dts[1] = 512; Dts[2] = 16; Dts[3] = 1024;
    }
    int32_t disabled_bound[4] = {0, 0, 0, 0};
    int32_t disabled_stride[4] = {0, 0, 0, 0};
    int32_t chan_a[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape)};
    int32_t chan_b[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape)};
    int32_t chan_d[1] = {1};
    int32_t chan_off[1] = {0};
    moe_set_dual_versacore_streamer_csr(
        A_addr, Asl, Atb, Ats, 0, chan_a,
        B0_addr, Bsl, Btb, Bts, 0, chan_b,
        B1_addr, Bsl, Btb, Bts, 0, chan_b,
        D_addr, Dsl, Dtb, Dts, 0, chan_d,
        D_addr, Dsl, disabled_bound, disabled_stride, 0, chan_off);
    moe_set_dual_versacore_mode(0u);
    moe_set_dual_versacore_csr(1u, K, N, 0u, array_shape, 0u);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale_mul(0, rscl_mult, 0, rscl_shift);
}

static inline void __moe_bank_configure_mode1(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr,
    uint32_t K, uint32_t N, uint32_t array_shape,
    uint32_t rscl_mult, uint32_t rscl_shift)
{
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t meshCol = __moe_dyn_meshcol(array_shape);
    uint32_t panel_span = (K / 4u) * MOE_BANK_TCDM_ROW_BYTES;
    int32_t Asl[2] = {8, 16};
    int32_t Atb[6] = {(int32_t)K, (int32_t)N, 1, 1, 1, 1};
    int32_t Ats[6] = {MOE_BANK_TCDM_ROW_BYTES, 0, 0, 0, 0, 0};
    int32_t Bsl[2] = {8, (int32_t)panel_span};
    int32_t Btb[4] = {4, (int32_t)(K / 4u), (int32_t)N, 1};
    int32_t Bts[4] = {
        16, MOE_BANK_TCDM_ROW_BYTES,
        (int32_t)((1u << array_shape) * panel_span), 0};
    int32_t Dsl[1] = {8};
    int32_t Dtb[4] = {
        (int32_t)(meshCol / 4u), (int32_t)meshRow, (int32_t)N, 1};
    int32_t Dts[4] = {
        MOE_BANK_TCDM_ROW_BYTES, 8,
        (int32_t)((meshCol / 4u) * MOE_BANK_TCDM_ROW_BYTES), 0};
    int32_t chan_a[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape)};
    int32_t chan_b[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape)};
    int32_t chan_d[1] = {1};
    moe_set_dual_versacore_streamer_csr(
        A_addr, Asl, Atb, Ats, 0, chan_a,
        B0_addr, Bsl, Btb, Bts, 0, chan_b,
        B1_addr, Bsl, Btb, Bts, 0, chan_b,
        D0_addr, Dsl, Dtb, Dts, 0, chan_d,
        D1_addr, Dsl, Dtb, Dts, 0, chan_d);
    moe_set_dual_versacore_mode(1u);
    moe_set_dual_versacore_csr(1u, K, N, 0u, array_shape, 0u);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
}

static inline uint32_t __moe_bank_weight_block_addr(
    uint32_t ping_base, uint32_t block, uint32_t payload_bytes);

static inline uint32_t __moe_bank_down_weight_block_addr(
    uint32_t ping_base, uint32_t block, uint32_t payload_bytes);

static inline uint32_t __moe_dyn_input_base(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) & 1u) != 0u ?
        st->l1_d_addr : st->l1_a_addr;
}

static inline uint32_t __moe_dyn_intermediate_base(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) & 1u) != 0u ?
        st->l1_a_addr : st->l1_d_addr;
}

static inline uint32_t __moe_dyn_output_base(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) & 1u) != 0u ?
        st->l1_d_addr : st->l1_down_d_addr;
}

static inline uint32_t __moe_csr_stage_is_prepared(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    uint32_t stage)
{
    asm volatile("fence rw, rw" ::: "memory");
    return __moe_s1_dma_ctrl(blk)->csr_prepared_stage == stage;
}

static inline void __moe_csr_publish_prepared(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    uint32_t stage)
{
    __moe_pipeline_publish(
        &__moe_s1_dma_ctrl(blk)->csr_prepared_stage, stage);
}

static inline uint32_t __moe_prepare_s2_csr(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    const __snax_bingo_moe_dyn_s2_call_args_t *call = &cfg->s2_call;
    if (call->valid == 0u || call->M == 0u ||
        __moe_dyn_shape_valid(call->array_shape) == 0u) {
        return 0u;
    }

    uint32_t token = call->token_start;
    uint32_t n_tiles =
        __moe_dyn_stage_block_n(call->N, st->s1_block_count);
    if (n_tiles == 0u) return 0u;
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    __moe_bank_configure_mode0(
        __moe_bank_a_addr(
            __moe_dyn_input_base(cfg, st), token, st->A_token_bytes),
        __moe_bank_weight_block_addr(
            st->l1_b_gate_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_weight_block_addr(
            st->l1_b_up_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_mode0_output_addr(
            __moe_dyn_intermediate_base(cfg, st), token, 0u,
            st->indiv_N_per_block,
            st->s1_block_count),
        st->indiv_K1, n_tiles, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S2);
    return 1u;
}

static inline uint32_t __moe_prepare_s3_csr(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[0];
    if (call->valid == 0u) return 0u;

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    __moe_bank_configure_mode1(
        __moe_dyn_intermediate_base(cfg, st),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr, 0u, st->indiv_down_B_block_stride),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr + 64u, 0u,
            st->indiv_down_B_block_stride),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st), 0u, 0u,
            st->indiv_down_N_per_block,
            st->s3_block_count),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st) + 64u, 0u, 0u,
            st->indiv_down_N_per_block, st->s3_block_count),
        st->indiv_down_K1, call->N, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S3);
    return 1u;
}

static inline uint32_t __moe_prepare_s4_csr(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout)
{
    const __snax_bingo_moe_dyn_s4_call_args_t *call = &cfg->s4_call;
    if (call->valid == 0u || call->M == 0u ||
        __moe_dyn_shape_valid(call->array_shape) == 0u) {
        return 0u;
    }

    uint32_t token = call->token_start;
    uint32_t n_tiles =
        __moe_dyn_stage_block_n(call->N, st->s3_block_count);
    if (n_tiles == 0u) return 0u;
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    uint32_t block = s4_csr_layout == MOE_S4_CSR_LAYOUT_BLOCK_SYNC ?
        __moe_s4_block_initial_phase(st) : 0u;
    __moe_bank_configure_mode1(
        __moe_bank_mode0_output_addr(
            __moe_dyn_intermediate_base(cfg, st), token, 0u,
            st->indiv_N_per_block,
            st->s1_block_count),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr, block,
            st->indiv_down_B_block_stride),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr + 64u, block,
            st->indiv_down_B_block_stride),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st), token, block,
            st->indiv_down_N_per_block, st->s3_block_count),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st) + 64u, token, block,
            st->indiv_down_N_per_block, st->s3_block_count),
        st->indiv_down_K1, n_tiles, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S4);
    return 1u;
}

static inline void __moe_prepare_after_s1(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout)
{
    if (__moe_prepare_s2_csr(blk, cfg, st) != 0u) return;
    if (__moe_prepare_s3_csr(blk, cfg, st) != 0u) return;
    (void)__moe_prepare_s4_csr(blk, cfg, st, s4_csr_layout);
}

static inline void __moe_prepare_after_s2(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout)
{
    if (__moe_prepare_s3_csr(blk, cfg, st) != 0u) return;
    (void)__moe_prepare_s4_csr(blk, cfg, st, s4_csr_layout);
}

__attribute__((always_inline)) static inline void
__moe_configure_down(
    uint32_t A_addr,
    uint32_t B0_addr,
    uint32_t B1_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t array_shape,
    uint32_t b_k_section,
    uint32_t b_n_stride,
    uint32_t a_m_stride,
    uint32_t d_m_stride,
    uint32_t d_row_stride,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t meshCol = __moe_dyn_meshcol(array_shape);
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
    csrw_ss(T_STRIDE_BASE_READER_1+1, b_n_stride);
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
    csrw_ss(T_STRIDE_BASE_READER_2+1, b_n_stride);
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
}

__attribute__((always_inline)) static inline void
__moe_configure_down_shape_c(
    uint32_t A_addr,
    uint32_t B0_addr,
    uint32_t B1_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t b_block_count,
    uint32_t b_block_stride,
    uint32_t b_k_section,
    uint32_t b_n_per_block,
    uint32_t b_n_stride,
    uint32_t a_m_stride,
    uint32_t d_m_stride,
    uint32_t d_row_stride,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);

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
        /* Both copies have the same 1D shape. Configure the 26 invariant CSRs
         * once, then use xDMA's committed-task FIFO: START snapshots copy 0,
         * allowing copy 1's four address CSRs to be written while copy 0 runs. */
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
        (void)xdma_memcpy_1d_fast_configure(bytes);
        xdma_memcpy_fast_set_addresses(src0_addr, dst0_addr);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
        int32_t xdma_task0 = xdma_start();

        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
        xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
        int32_t xdma_task1 = xdma_start();

        __moe_dyn_wait_xdma(dst0_addr, src0_addr, xdma_task0);
        __moe_dyn_wait_xdma(dst1_addr, src1_addr, xdma_task1);
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

typedef struct {
    uint32_t binding;
    uint32_t repeats;
    uint32_t wait_idma;
    int32_t xdma_task0;
    int32_t xdma_task1;
    uint64_t xdma_dst0;
    uint64_t xdma_src0;
    uint64_t xdma_dst1;
    uint64_t xdma_src1;
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
    uint32_t probe_start;
#endif
} __moe_dyn_2d_pair_pending_t;

static inline uint32_t __moe_dyn_start_pair_2d(
    uint32_t binding,
    uint64_t dst0_addr,
    uint64_t src0_addr,
    uint64_t dst1_addr,
    uint64_t src1_addr,
    uint32_t bytes,
    uint32_t configure_xdma,
    __moe_dyn_2d_pair_pending_t *pending)
{
    uint32_t repeats = bytes / MOE_BANK_WEIGHT_ROW_BYTES;
    pending->binding = binding;
    pending->repeats = repeats;
    pending->wait_idma = 0u;
    pending->xdma_task0 = -1;
    pending->xdma_task1 = -1;
    pending->xdma_dst0 = dst0_addr;
    pending->xdma_src0 = src0_addr;
    pending->xdma_dst1 = dst1_addr;
    pending->xdma_src1 = src1_addr;

    if (binding == MOE_DYN_DMA_IDMA) {
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
        const enum snrt_perf_cnt_type probe_types[] = {
            SNRT_PERF_CNT_DMA_BUSY,
            SNRT_PERF_CNT_DMA_AR_DONE,
            SNRT_PERF_CNT_DMA_R_DONE,
            SNRT_PERF_CNT_DMA_AW_DONE,
            SNRT_PERF_CNT_DMA_W_DONE,
            SNRT_PERF_CNT_DMA_B_DONE,
            SNRT_PERF_CNT_DMA_AR_STALL,
            SNRT_PERF_CNT_DMA_AW_STALL,
            SNRT_PERF_CNT_TCDM_ACCESSED,
            SNRT_PERF_CNT_TCDM_CONGESTED,
        };
        for (uint32_t i = 0u; i < 10u; i++) {
            snrt_reset_perf_counter((enum snrt_perf_cnt)i);
            snrt_start_perf_counter(
                (enum snrt_perf_cnt)i, probe_types[i], 0u);
        }
        pending->probe_start = snrt_mcycle();
#endif
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        snrt_dma_start_2d_wideptr(
            dst0_addr, src0_addr, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        snrt_dma_start_2d_wideptr(
            dst1_addr, src1_addr, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        pending->wait_idma = 1u;
        return BINGO_RET_SUCC;
    }

    if (binding == MOE_DYN_DMA_XDMA) {
        if (configure_xdma != 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
            xdma_memcpy_2d_fast_configure(
                MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
                MOE_BANK_TCDM_ROW_BYTES, repeats);
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
        }
        xdma_memcpy_fast_set_addresses(src0_addr, dst0_addr);
        pending->xdma_task0 = xdma_start();
        xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
        pending->xdma_task1 = xdma_start();
        return BINGO_RET_SUCC;
    }

    if (binding != MOE_DYN_DMA_BOTH) return BINGO_RET_FAIL;

    if (configure_xdma != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
        xdma_memcpy_2d_fast_configure(
            MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, repeats);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    }
    xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
    pending->xdma_task0 = xdma_start();
    pending->xdma_dst0 = dst1_addr;
    pending->xdma_src0 = src1_addr;
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_2d_wideptr(
        dst0_addr, src0_addr, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    pending->wait_idma = 1u;
    return BINGO_RET_SUCC;
}

static inline uint32_t __moe_dyn_wait_pair_2d(
    __moe_dyn_2d_pair_pending_t *pending)
{
    if (pending->wait_idma != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
        if (pending->binding == MOE_DYN_DMA_IDMA) {
            uint32_t probe_cycles = snrt_mcycle() - pending->probe_start;
            for (uint32_t i = 0u; i < 10u; i++) {
                snrt_stop_perf_counter((enum snrt_perf_cnt)i);
            }
            printf_safe(
                "IDMA2D_PROBE cycles=%u busy=%u ar=%u r=%u aw=%u w=%u "
                "b=%u ar_stall=%u aw_stall=%u tcdm=%u tcdm_cong=%u "
                "rows=%u\r\n",
                probe_cycles,
                snrt_get_perf_counter(SNRT_PERF_CNT0),
                snrt_get_perf_counter(SNRT_PERF_CNT1),
                snrt_get_perf_counter(SNRT_PERF_CNT2),
                snrt_get_perf_counter(SNRT_PERF_CNT3),
                snrt_get_perf_counter(SNRT_PERF_CNT4),
                snrt_get_perf_counter(SNRT_PERF_CNT5),
                snrt_get_perf_counter(SNRT_PERF_CNT6),
                snrt_get_perf_counter(SNRT_PERF_CNT7),
                snrt_get_perf_counter(SNRT_PERF_CNT8),
                snrt_get_perf_counter(SNRT_PERF_CNT9),
                2u * pending->repeats);
        }
#endif
    }
    if (pending->xdma_task0 >= 0) {
        __moe_dyn_wait_xdma(
            pending->xdma_dst0, pending->xdma_src0,
            pending->xdma_task0);
    }
    if (pending->xdma_task1 >= 0) {
        __moe_dyn_wait_xdma(
            pending->xdma_dst1, pending->xdma_src1,
            pending->xdma_task1);
    }
    return BINGO_RET_SUCC;
}

static inline uint32_t __moe_dyn_copy_pair_2d(
    uint32_t binding,
    uint64_t dst0_addr,
    uint64_t src0_addr,
    uint64_t dst1_addr,
    uint64_t src1_addr,
    uint32_t bytes,
    uint32_t configure_xdma)
{
    __moe_dyn_2d_pair_pending_t pending;
    uint32_t rc = __moe_dyn_start_pair_2d(
        binding, dst0_addr, src0_addr, dst1_addr, src1_addr,
        bytes, configure_xdma, &pending);
    if (rc != BINGO_RET_SUCC) return rc;
    return __moe_dyn_wait_pair_2d(&pending);
}

static inline uint32_t __moe_bank_weight_block_addr_phase(
    uint32_t ping_base, uint32_t block, uint32_t payload_bytes,
    uint32_t phase_xor)
{
    uint32_t phase = (block & 1u) ^ phase_xor;
    return ping_base + phase * MOE_BANK_B_PHASE_DELTA_BYTES +
        (block >> 1u) * payload_bytes *
            (MOE_BANK_TCDM_ROW_BYTES / MOE_BANK_WEIGHT_ROW_BYTES);
}

static inline uint32_t __moe_bank_weight_block_addr(
    uint32_t ping_base, uint32_t block, uint32_t payload_bytes)
{
    return __moe_bank_weight_block_addr_phase(
        ping_base, block, payload_bytes, 0u);
}

static inline uint32_t __moe_bank_down_weight_block_addr(
    uint32_t ping_base, uint32_t block, uint32_t payload_bytes)
{
    return __moe_bank_weight_block_addr_phase(
        ping_base, block, payload_bytes, 1u);
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_store_page(
    uint64_t src, uint64_t dst, uint32_t token_count)
{
    uint32_t enabled = (1u << token_count) - 1u;
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, enabled);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, enabled);
    xdma_memcpy_fast_set_addresses(src, dst);
}

static inline void __moe_bank_configure_store(
    uint64_t src, uint64_t dst, uint32_t token_count,
    uint32_t token_bytes)
{
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLE_PTR, 0u);
    snax_write_xdma_cfg_reg(XDMA_SRC_SPATIAL_STRIDE_PTR, 8u);
    snax_write_xdma_cfg_reg(XDMA_DST_SPATIAL_STRIDE_PTR, token_bytes);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR, token_bytes / 16u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR, MOE_BANK_TCDM_ROW_BYTES);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 1u, 2u);
    snax_write_xdma_cfg_reg(
        XDMA_SRC_TEMP_STRIDE_PTR + 1u, MOE_BANK_MODE1_D1_OFFSET);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR, token_bytes / 16u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR, 8u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 1u, 2u);
    snax_write_xdma_cfg_reg(
        XDMA_DST_TEMP_STRIDE_PTR + 1u, token_bytes / 2u);
#pragma GCC unroll 3
    for (uint32_t i = 2u; i < XDMA_SRC_TEMP_DIM; i++) {
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + i, 1u);
        snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + i, 0u);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + i, 1u);
        snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + i, 0u);
    }
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xffffffffu);
    __moe_bank_patch_store_page(src, dst, token_count);
}

static inline uint32_t __moe_dyn_has_output(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    return MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u ||
        MOE_DYN_CTRL_SKIP_S4(cfg->ctrl) == 0u;
}

static inline void __moe_dyn_prepare_store_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint64_t dst = st->output_l3_base +
        (uint64_t)cfg->expert_id * (uint64_t)st->output_expert_stride_bytes +
        (uint64_t)cfg->token_ref_start * (uint64_t)st->A_row_stride;
    uint64_t src = __moe_dyn_l1_wide(__moe_dyn_output_base(cfg, st));
    uint32_t first_page_tokens = cfg->ntokens < MOE_BANK_TOKEN_LANES ?
        cfg->ntokens : MOE_BANK_TOKEN_LANES;
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
    __moe_bank_configure_store(
        src, dst, first_page_tokens, st->A_token_bytes);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_stage_tokens_2d(void *arg)
{
    const __snax_bingo_kernel_moe_stage_tokens_2d_args_t *cfg =
        (const __snax_bingo_kernel_moe_stage_tokens_2d_args_t *)arg;
    uint32_t repeats = cfg->token_bytes / MOE_BANK_A_TOKEN_TILE_BYTES;
    for (uint32_t token = 0u; token < cfg->token_count; token++) {
        snrt_dma_start_2d_wideptr(
            __moe_dyn_l1_wide(
                __moe_bank_a_addr(
                    cfg->dst_addr, token, cfg->token_bytes)),
            cfg->src_addr + (uint64_t)token * cfg->token_bytes,
            MOE_BANK_A_TOKEN_TILE_BYTES, MOE_BANK_TCDM_ROW_BYTES,
            MOE_BANK_A_TOKEN_TILE_BYTES, repeats);
    }
    snrt_dma_wait_all();
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_stage_weight_pair_2d(void *arg)
{
    const __snax_bingo_kernel_moe_stage_weight_pair_2d_args_t *cfg =
        (const __snax_bingo_kernel_moe_stage_weight_pair_2d_args_t *)arg;

    /* Bingo task arguments live in TCDM. Parse them before any concurrent VC
     * run can monopolize the A banks; the transfer loop must not reread cfg. */
    const uint64_t src0_base = cfg->src0_addr;
    const uint64_t src1_base = cfg->src1_addr;
    const uint32_t dst0_base = cfg->dst0_addr;
    const uint32_t dst1_base = cfg->dst1_addr;
    const uint32_t bytes_per_block = cfg->bytes_per_block;
    const uint32_t block_count = cfg->block_count;
    const uint32_t binding = cfg->binding;
    const uint32_t phase_xor = cfg->phase_xor;

    if (block_count == 0u || block_count > BINGO_MOE_MAX_BLOCKS ||
        bytes_per_block == 0u ||
        bytes_per_block % MOE_BANK_WEIGHT_ROW_BYTES != 0u ||
        phase_xor > 1u) {
        return BINGO_RET_FAIL;
    }

    uint32_t repeats = bytes_per_block / MOE_BANK_WEIGHT_ROW_BYTES;
    if (binding == MOE_DYN_DMA_IDMA) {
        /* Submit the complete resident tensor before the single wait. With the
         * workload ABI this is at most 2*BINGO_MOE_MAX_BLOCKS descriptors. */
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
        const enum snrt_perf_cnt_type probe_types[] = {
            SNRT_PERF_CNT_DMA_BUSY,
            SNRT_PERF_CNT_DMA_AR_DONE,
            SNRT_PERF_CNT_DMA_R_DONE,
            SNRT_PERF_CNT_DMA_AW_DONE,
            SNRT_PERF_CNT_DMA_W_DONE,
            SNRT_PERF_CNT_DMA_B_DONE,
            SNRT_PERF_CNT_DMA_AR_STALL,
            SNRT_PERF_CNT_DMA_AW_STALL,
            SNRT_PERF_CNT_TCDM_ACCESSED,
            SNRT_PERF_CNT_TCDM_CONGESTED,
        };
        for (uint32_t i = 0u; i < 10u; i++) {
            snrt_reset_perf_counter((enum snrt_perf_cnt)i);
            snrt_start_perf_counter(
                (enum snrt_perf_cnt)i, probe_types[i], 0u);
        }
        uint32_t probe_start = snrt_mcycle();
#endif
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        for (uint32_t block = 0u; block < block_count; block++) {
            uint32_t src_offset = block * bytes_per_block;
            uint32_t dst0 = __moe_bank_weight_block_addr_phase(
                dst0_base, block, bytes_per_block, phase_xor);
            uint32_t dst1 = __moe_bank_weight_block_addr_phase(
                dst1_base, block, bytes_per_block, phase_xor);
            snrt_dma_start_2d_wideptr(
                __moe_dyn_l1_wide(dst0), src0_base + src_offset,
                MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
                MOE_BANK_WEIGHT_ROW_BYTES, repeats);
            snrt_dma_start_2d_wideptr(
                __moe_dyn_l1_wide(dst1), src1_base + src_offset,
                MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
                MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
        uint32_t probe_cycles = snrt_mcycle() - probe_start;
        for (uint32_t i = 0u; i < 10u; i++) {
            snrt_stop_perf_counter((enum snrt_perf_cnt)i);
        }
        printf_safe(
            "IDMA2D_PROBE cycles=%u busy=%u ar=%u r=%u aw=%u w=%u b=%u "
            "ar_stall=%u aw_stall=%u tcdm=%u tcdm_cong=%u rows=%u\r\n",
            probe_cycles,
            snrt_get_perf_counter(SNRT_PERF_CNT0),
            snrt_get_perf_counter(SNRT_PERF_CNT1),
            snrt_get_perf_counter(SNRT_PERF_CNT2),
            snrt_get_perf_counter(SNRT_PERF_CNT3),
            snrt_get_perf_counter(SNRT_PERF_CNT4),
            snrt_get_perf_counter(SNRT_PERF_CNT5),
            snrt_get_perf_counter(SNRT_PERF_CNT6),
            snrt_get_perf_counter(SNRT_PERF_CNT7),
            snrt_get_perf_counter(SNRT_PERF_CNT8),
            snrt_get_perf_counter(SNRT_PERF_CNT9),
            2u * block_count * repeats);
#endif
        return BINGO_RET_SUCC;
    }

    if (binding != MOE_DYN_DMA_XDMA && binding != MOE_DYN_DMA_BOTH) {
        return BINGO_RET_FAIL;
    }

    /* Shape is invariant across blocks. Configure xDMA once and only commit
     * new source/destination addresses for each following descriptor. */
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
    xdma_memcpy_2d_fast_configure(
        MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, repeats);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    int32_t last_xdma_task = -1;
    uint64_t last_xdma_src = 0u;
    uint64_t last_xdma_dst = 0u;
    for (uint32_t block = 0u; block < block_count; block++) {
        uint32_t src_offset = block * bytes_per_block;
        uint64_t dst0 = __moe_dyn_l1_wide(__moe_bank_weight_block_addr_phase(
            dst0_base, block, bytes_per_block, phase_xor));
        uint64_t dst1 = __moe_dyn_l1_wide(__moe_bank_weight_block_addr_phase(
            dst1_base, block, bytes_per_block, phase_xor));
        uint64_t src0 = src0_base + src_offset;
        uint64_t src1 = src1_base + src_offset;

        if (binding == MOE_DYN_DMA_XDMA) {
            xdma_memcpy_fast_set_addresses(src0, dst0);
            last_xdma_task = (int32_t)xdma_start_remote();
            xdma_memcpy_fast_set_addresses(src1, dst1);
            last_xdma_task = (int32_t)xdma_start_remote();
        } else {
            xdma_memcpy_fast_set_addresses(src1, dst1);
            last_xdma_task = (int32_t)xdma_start_remote();
            snrt_dma_start_2d_wideptr(
                dst0, src0, MOE_BANK_WEIGHT_ROW_BYTES,
                MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        }
        last_xdma_src = src1;
        last_xdma_dst = dst1;
    }
    if (binding == MOE_DYN_DMA_BOTH) snrt_dma_wait_all();
    __moe_dyn_wait_xdma(last_xdma_dst, last_xdma_src, last_xdma_task);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_store_tokens_2d(void *arg)
{
    const __snax_bingo_kernel_moe_store_tokens_2d_args_t *cfg =
        (const __snax_bingo_kernel_moe_store_tokens_2d_args_t *)arg;
    uint32_t page_span = __moe_bank_token_page_span(cfg->token_bytes);
    for (uint32_t token_start = 0u; token_start < cfg->token_count;
         token_start += MOE_BANK_TOKEN_LANES) {
        uint32_t remaining = cfg->token_count - token_start;
        uint32_t count = remaining < MOE_BANK_TOKEN_LANES ?
            remaining : MOE_BANK_TOKEN_LANES;
        uint32_t page = token_start / MOE_BANK_TOKEN_LANES;
        uint64_t src0 = __moe_dyn_l1_wide(
            cfg->src_d0_addr + page * page_span);
        uint64_t src1 = __moe_dyn_l1_wide(
            cfg->src_d1_addr + page * page_span);
        uint64_t dst = cfg->dst_addr +
            (uint64_t)token_start * cfg->token_bytes;
        if (token_start == 0u) {
            __moe_bank_configure_store(src0, dst, count, cfg->token_bytes);
        } else {
            __moe_bank_patch_store_page(src0, dst, count);
        }
        int32_t task0 = (int32_t)xdma_start_remote();
        xdma_memcpy_fast_set_addresses(
            src1, dst + cfg->token_bytes / 2u);
        int32_t task1 = (int32_t)xdma_start_remote();
        __moe_dyn_wait_xdma(dst, src0, task0);
        __moe_dyn_wait_xdma(dst + cfg->token_bytes / 2u, src1, task1);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_bank_moe_full(void *arg)
{
    const __snax_bingo_kernel_dual_vc_bank_moe_full_args_t *cfg =
        (const __snax_bingo_kernel_dual_vc_bank_moe_full_args_t *)arg;
    if (cfg->token_count == 0u || cfg->s1_block_count == 0u ||
        cfg->s1_block_count > BINGO_MOE_MAX_BLOCKS ||
        cfg->s3_block_count == 0u ||
        cfg->s3_block_count > BINGO_MOE_MAX_BLOCKS ||
        cfg->chunk_cols == 0u) {
        return BINGO_RET_FAIL;
    }

    uint32_t s1_block_count = cfg->s1_block_count;
    uint32_t s3_block_count = cfg->s3_block_count;

    uint32_t mode0_block_bytes =
        cfg->hidden_size * cfg->chunk_cols / 2u;
    uint32_t mode1_block_bytes =
        cfg->intermediate_size * cfg->chunk_cols / 2u;
    uint32_t mode1_region =
        ((s1_block_count + 1u) / 2u) * mode0_block_bytes * 8u;
    uint32_t mode0_k = cfg->hidden_size / MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t mode1_k = cfg->intermediate_size / MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t mode1_b0 = cfg->tcdm_base + mode1_region +
        MOE_BANK_B0_PING_OFFSET;
    uint32_t mode1_b1 = cfg->tcdm_base + mode1_region +
        MOE_BANK_B1_PING_OFFSET;

    for (uint32_t token_start = 0u; token_start < cfg->token_count;
         token_start += MOE_BANK_TOKEN_LANES) {
        uint32_t remaining = cfg->token_count - token_start;
        uint32_t valid = remaining < MOE_BANK_TOKEN_LANES ?
            remaining : MOE_BANK_TOKEN_LANES;
        uint32_t shape = valid > 4u ? 0u : (valid > 2u ? 1u : 2u);
        uint32_t n_tiles = cfg->chunk_cols / __moe_dyn_meshcol(shape);

        for (uint32_t block = 0u; block < s1_block_count; block++) {
            uint32_t A_addr = __moe_bank_a_addr(
                cfg->tcdm_base, token_start,
                cfg->hidden_size * sizeof(int16_t));
            uint32_t B0_addr = __moe_bank_weight_block_addr(
                cfg->tcdm_base + MOE_BANK_B0_PING_OFFSET,
                block, mode0_block_bytes);
            uint32_t B1_addr = __moe_bank_weight_block_addr(
                cfg->tcdm_base + MOE_BANK_B1_PING_OFFSET,
                block, mode0_block_bytes);
            uint32_t D_addr = __moe_bank_mode0_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE0_D_OFFSET,
                token_start, block, cfg->chunk_cols, s1_block_count);
            if (block == 0u) {
                __moe_bank_configure_mode0(
                    A_addr, B0_addr, B1_addr, D_addr,
                    mode0_k, n_tiles, shape,
                    cfg->rescale_mult, cfg->rescale_shift);
            } else {
                __moe_bank_patch_mode0_block_bases(
                    B0_addr, B1_addr, D_addr);
            }
            moe_start_dual_vc_and_streamer();
            moe_wait_dual_vc_and_streamer();
        }

        for (uint32_t block = 0u; block < s3_block_count; block++) {
            uint32_t A_addr = __moe_bank_mode0_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE0_D_OFFSET,
                token_start, 0u, cfg->chunk_cols, s1_block_count);
            uint32_t B0_addr = __moe_bank_down_weight_block_addr(
                mode1_b0, block, mode1_block_bytes);
            uint32_t B1_addr = __moe_bank_down_weight_block_addr(
                mode1_b1, block, mode1_block_bytes);
            uint32_t D0_addr = __moe_bank_mode1_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE1_D0_OFFSET,
                token_start, block, cfg->chunk_cols, s3_block_count);
            uint32_t D1_addr = __moe_bank_mode1_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE1_D1_OFFSET,
                token_start, block, cfg->chunk_cols, s3_block_count);
            if (block == 0u) {
                __moe_bank_configure_mode1(
                    A_addr, B0_addr, B1_addr, D0_addr, D1_addr,
                    mode1_k, n_tiles, shape,
                    cfg->rescale_mult, cfg->rescale_shift);
            } else {
                __moe_bank_patch_mode1_block_bases(
                    B0_addr, B1_addr, D0_addr, D1_addr);
            }
            moe_start_dual_vc_and_streamer();
            moe_wait_dual_vc_and_streamer();
        }
    }
    return BINGO_RET_SUCC;
}

// Dynamic individual-expert slot API
//
// Production DFGs use exactly one optimized entry per worker and stage:
//   DM core: opt_gather, opt_load_s1, opt_prefetch_s2, opt_load_s3,
//            opt_prefetch_s4, opt_store_gather
//   VC core: opt_config_s1, opt_compute_s1, opt_compute_s2, opt_config_s3,
//            opt_compute_s3, opt_compute_s4
// The per-block dynamic_expert_* entries below are retained only for the
// discrete comparison DFG. There is no run-time dispatch or fallback between
// those two API families.
//
// main_bingo.py fixes each kernel to either GEMM core 0 or DM core 1. These
// workload-private entry points therefore do not repeat core-id checks at run time.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_init_output_gaps(void *arg)
{
    const __snax_bingo_kernel_moe_init_output_gaps_args_t *cfg =
        (const __snax_bingo_kernel_moe_init_output_gaps_args_t *)arg;
    uint32_t gap_bytes = cfg->row_stride_bytes - cfg->row_payload_bytes;
    for (uint32_t row = 0u; row < cfg->rows; row++) {
        __moe_zero_tcdm(
            cfg->output_base + row * cfg->row_stride_bytes +
                cfg->row_payload_bytes,
            gap_bytes);
    }
    return BINGO_RET_SUCC;
}

static inline uint32_t __moe_dyn_prepare_pipeline_impl(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st);

static inline uint32_t __moe_gather_rows_idma_start(
    const uint16_t *token_refs,
    uint32_t token_ref_start,
    uint64_t input_l3_base,
    uint32_t gather_l1_addr,
    uint32_t ntokens,
    uint32_t token_bytes,
    uint32_t row_stride,
    uint32_t batch_arithmetic_runs)
{
    (void)batch_arithmetic_runs;
    uint32_t repeats = token_bytes / MOE_BANK_A_TOKEN_TILE_BYTES;
    for (uint32_t local_t = 0u; local_t < ntokens; local_t++) {
        uint16_t token_ref = token_refs[token_ref_start + local_t];
        uint32_t token_id = BINGO_MOE_TOKEN_REF_TOKEN(token_ref);
        uint64_t src = input_l3_base +
            (uint64_t)token_id * (uint64_t)row_stride;
        uint64_t dst = __moe_dyn_l1_wide(
            __moe_bank_a_addr(gather_l1_addr, local_t, token_bytes));
        snrt_dma_start_2d_wideptr(
            dst, src, MOE_BANK_A_TOKEN_TILE_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_A_TOKEN_TILE_BYTES, repeats);
    }
    return BINGO_RET_SUCC;
}

static inline uint32_t __moe_dyn_prepare_pipeline_if_needed(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    asm volatile("fence rw, rw" ::: "memory");
    if (__moe_s1_dma_ctrl(blk)->csr_prepared_reserved ==
        MOE_PIPELINE_INIT_COOKIE) {
        return BINGO_RET_SUCC;
    }
    return __moe_dyn_prepare_pipeline_impl(blk, cfg, st);
}

static inline uint32_t __moe_dyn_prepare_next_slot_pipeline(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    volatile uint32_t *state =
        (volatile uint32_t *)(uintptr_t)st->active_state_l1_addr;
    uint32_t active_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C2_ACTIVE_SLOTS : MOE_DYN_RT_C3_ACTIVE_SLOTS;
    uint32_t slot = MOE_DYN_CTRL_SLOT_ID(cfg->ctrl);
    if (slot + 1u >= state[active_idx]) return BINGO_RET_SUCC;

    __snax_bingo_kernel_moe_dynamic_expert_block_args_t next_blk = *blk;
    next_blk.task_arg_addr += BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES;
    next_blk.pipeline_ctrl_addr += MOE_PIPELINE_CTRL_SLOT_BYTES;
    next_blk.block_idx = 0u;
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *next_cfg =
        (const __snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        next_blk.task_arg_addr;
    return __moe_dyn_prepare_pipeline_if_needed(&next_blk, next_cfg, st);
}

static inline uint32_t __moe_gather_rows_idma_wait(void)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
    return BINGO_RET_SUCC;
}

#ifndef BINGO_MOE_GATHER_BATCH_ARITHMETIC_RUNS
#define BINGO_MOE_GATHER_BATCH_ARITHMETIC_RUNS 0
#endif

static inline uint32_t __moe_dyn_gather_slot_tokens_start(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    /* L3 rows are dense.  L1 maps each local token to one two-bank lane;
     * tokens beyond lane 7 continue at a deeper TCDM page.  A task submits
     * exactly cfg->ntokens descriptors in this single gather operation. */
    uint32_t a_row_stride = st->A_row_stride;
    const uint16_t *token_refs =
        (const uint16_t *)(uintptr_t)st->token_refs_addr;
    uint32_t expert_token_offset = cfg->expert_id * st->max_tokens_per_expert;

    uint32_t rc = __moe_gather_rows_idma_start(
        token_refs, expert_token_offset + cfg->token_ref_start,
        st->input_A_l3_base, __moe_dyn_input_base(cfg, st), cfg->ntokens,
        st->A_token_bytes, a_row_stride,
        BINGO_MOE_GATHER_BATCH_ARITHMETIC_RUNS);
    return rc;
}

static inline uint32_t __moe_dyn_gather_slot_tokens_wait(void)
{
    return __moe_gather_rows_idma_wait();
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_gather_s1(void *arg)
{
    __snax_bingo_kernel_moe_dynamic_expert_block_args_t *node =
        (__snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        node->task_arg_addr;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        node->static_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return __moe_dyn_prepare_pipeline_impl(node, cfg, st);
    }
    MOE_PROFILE_BEGIN(profile);

    MOE_INDIV_PRINT(
        "[INDIV_BEGIN] C%u slot=%u eid=%u start=%u ntok=%u "
        "shape_s1=%u shape_s3=%u skip=%u%u%u%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->token_ref_start, cfg->ntokens,
        MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl), MOE_DYN_CTRL_SHAPE_S3(cfg->ctrl),
        MOE_DYN_CTRL_SKIP_S1(cfg->ctrl), MOE_DYN_CTRL_SKIP_S2(cfg->ctrl),
        MOE_DYN_CTRL_SKIP_S3(cfg->ctrl), MOE_DYN_CTRL_SKIP_S4(cfg->ctrl));

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t rc = __moe_dyn_gather_slot_tokens_start(cfg, st);
    if (rc == BINGO_RET_SUCC) {
        rc = __moe_dyn_prepare_pipeline_impl(node, cfg, st);
    }
    if (rc == BINGO_RET_SUCC) {
        __moe_prepare_s1_xdma_shape(node, cfg, st);
    }
    uint32_t wait_rc = __moe_dyn_gather_slot_tokens_wait();
    if (rc == BINGO_RET_SUCC) rc = wait_rc;
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_GATHER_DONE] C%u slot=%u eid=%u ntok=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->ntokens, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_GATHER_S1,
        MOE_PROFILE_RESOURCE_IDMA,
        0u,
        cfg->ntokens * st->A_token_bytes,
        0u, rc);
    return rc;
}

static inline uint32_t __moe_dyn_prepare_pipeline_impl(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);

    s1->valid = 0u;
    s1->csr_prepared_stage = MOE_CSR_PREPARED_NONE;
    s1->xdma_prepared_stage = MOE_XDMA_PREPARED_NONE;
    s1->csr_prepared_reserved = 0u;
    s2->valid = 0u;
    s2->sync_enabled = 0u;
    s2->compute_done = 0u;
    s2->prefetch_done = 0u;
    s2->error = BINGO_RET_SUCC;
    s2->store_prepared = 0u;
    s2->reserved = 0u;

    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        asm volatile("fence rw, rw" ::: "memory");
        return BINGO_RET_SUCC;
    }

    uint32_t ctrl = cfg->ctrl;
    uint32_t expert_id = cfg->expert_id;
    uint32_t s1_binding = MOE_DYN_CTRL_DMA_S1(ctrl);
    if (MOE_DYN_CTRL_SKIP_S1(ctrl) == 0u &&
        s1_binding >= MOE_DYN_DMA_IDMA &&
        s1_binding <= MOE_DYN_DMA_BOTH) {
        s1->gate_src_base = st->indiv_gate_B_l3 +
            (uint64_t)expert_id * st->indiv_B_expert_stride;
        s1->up_src_base = st->indiv_up_B_l3 +
            (uint64_t)expert_id * st->indiv_B_expert_stride;
        s1->gate_dst_base = st->l1_b_gate_addr;
        s1->up_dst_base = st->l1_b_up_addr;
        s1->block_bytes = st->indiv_B_block_stride;
        s1->block_count = st->s1_block_count;
        s1->binding = s1_binding;
        s1->valid = 1u;
    }

    uint32_t pf_slot = MOE_DYN_DMA_SLOT_S2_PREFETCH;
    uint32_t pf_binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, pf_slot);
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, pf_slot) != 0u &&
        pf_binding >= MOE_DYN_DMA_IDMA &&
        pf_binding <= MOE_DYN_DMA_BOTH) {
        uint32_t pf_expert = MOE_DYN_DMA_EID(cfg->dma_slot_eids, pf_slot);
        s2->down_src_base = st->indiv_down_B_l3 +
            (uint64_t)pf_expert * st->indiv_down_B_expert_stride;
        s2->down_dst_base = st->l1_b_down_addr;
        s2->half_bytes = st->down_half_weight_bytes;
        s2->block_bytes = st->indiv_down_B_block_stride;
        s2->block_count = st->s3_block_count;
        s2->s1_block_count = st->s1_block_count;
        s2->binding = pf_binding;
        s2->reserved = pf_expert;
        s2->valid = 1u;
        s2->sync_enabled = cfg->s2_call.valid != 0u;
    }
    __moe_pipeline_publish(
        &s1->csr_prepared_reserved, MOE_PIPELINE_INIT_COOKIE);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prepare_pipeline(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (const __snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t active = __moe_dyn_slot_active_this_round(cfg, st);
    asm volatile("fence rw, rw" ::: "memory");
    if (active != 0u &&
        __moe_s1_dma_ctrl(blk)->csr_prepared_reserved ==
            MOE_PIPELINE_INIT_COOKIE) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    uint32_t rc = __moe_dyn_prepare_pipeline_impl(blk, cfg, st);
    if (rc == BINGO_RET_SUCC) {
        __moe_prepare_s1_xdma_shape(blk, cfg, st);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return rc;
}

__attribute__((always_inline)) static inline uint32_t
__moe_dynamic_expert_load_gate_up_block_impl(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __moe_s1_dma_ctrl_t *s1, uint32_t n)
{
    MOE_PROFILE_BEGIN(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_START);
    uint32_t block_bytes = s1->block_bytes;
    uint32_t weight_offset = n * block_bytes;
    uint32_t gate_dst = __moe_bank_weight_block_addr(
        s1->gate_dst_base, n, block_bytes);
    uint32_t up_dst = __moe_bank_weight_block_addr(
        s1->up_dst_base, n, block_bytes);
    uint32_t dma_binding = s1->binding;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t configure_xdma =
        n == 0u &&
        !__moe_xdma_stage_is_prepared(blk, MOE_XDMA_PREPARED_S1);
    uint32_t rc = __moe_dyn_copy_pair_2d(
        dma_binding,
        __moe_dyn_l1_wide(gate_dst),
        s1->gate_src_base + weight_offset,
        __moe_dyn_l1_wide(up_dst),
        s1->up_src_base + weight_offset,
        block_bytes, configure_xdma);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_LOAD_S1_DONE] C%u slot=%u eid=%u block=%u dma=%u "
        "bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, dma_binding, block_bytes, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_LOAD_S1,
        __moe_profile_dma_resource(dma_binding), n,
        2u * block_bytes, 0u, rc);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    uint32_t n = blk->block_idx;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    const __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    if (s1->valid == 0u || n >= s1->block_count) {
        MOE_PROFILE_BEGIN(profile);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S1,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    return __moe_dynamic_expert_load_gate_up_block_impl(
        blk, cfg, s1, n);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_config_s1_block0(void *arg)
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
    if (blk->block_idx != 0u ||
        !__moe_dyn_slot_active_this_round(cfg, st) ||
        cfg->s1_call[0].valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_s1_call_args_t *call = &cfg->s1_call[0];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    __moe_bank_configure_mode0(
        __moe_dyn_input_base(cfg, st),
        __moe_bank_weight_block_addr(
            st->l1_b_gate_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_weight_block_addr(
            st->l1_b_up_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_mode0_output_addr(
            __moe_dyn_intermediate_base(cfg, st), 0u, 0u,
            st->indiv_N_per_block,
            st->s1_block_count),
        st->indiv_K1, call->N, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    return BINGO_RET_SUCC;
}

__attribute__((always_inline)) static inline uint32_t
__moe_dynamic_expert_compute_gate_up_block_impl(
    void *arg, uint32_t configure_block0, uint32_t s4_csr_layout)
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
    const __snax_bingo_moe_dyn_s1_call_args_t *call = &cfg->s1_call[n];
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S1,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result = BINGO_RET_SUCC;
    if (n == 0u && configure_block0 != 0u) {
        __moe_bank_configure_mode0(
            __moe_dyn_input_base(cfg, st),
            __moe_bank_weight_block_addr(
                st->l1_b_gate_addr, n, st->indiv_B_block_stride),
            __moe_bank_weight_block_addr(
                st->l1_b_up_addr, n, st->indiv_B_block_stride),
            __moe_bank_mode0_output_addr(
                __moe_dyn_intermediate_base(cfg, st), 0u, n,
                st->indiv_N_per_block,
                st->s1_block_count),
            st->indiv_K1, call->N, call->array_shape,
            st->rescale_mult, st->rescale_shift);
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    uint32_t preload_next = (n + 1u < st->s1_block_count) ? 1u : 0u;
    if (preload_next != 0u) {
        uint32_t next_block = n + 1u;
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
        csrw_ss(BASE_PTR_READER_1_LOW,
            __moe_bank_weight_block_addr(
                st->l1_b_gate_addr, next_block, st->indiv_B_block_stride));
        csrw_ss(BASE_PTR_READER_2_LOW,
            __moe_bank_weight_block_addr(
                st->l1_b_up_addr, next_block, st->indiv_B_block_stride));
        csrw_ss(BASE_PTR_WRITER_0_LOW,
            __moe_bank_mode0_output_addr(
                __moe_dyn_intermediate_base(cfg, st), 0u, next_block,
                st->indiv_N_per_block,
                st->s1_block_count));
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    } else {
        __moe_prepare_after_s1(blk, cfg, st, s4_csr_layout);
    }
    moe_wait_dual_vc_and_streamer();
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S1_DONE] C%u slot=%u eid=%u block=%u shape=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, call->array_shape, call->N, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S1,
        MOE_PROFILE_RESOURCE_VERSACORE, n,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, result);
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block(void *arg)
{
    return __moe_dynamic_expert_compute_gate_up_block_impl(arg, 1u, 0u);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block_pc(void *arg)
{
    return __moe_dynamic_expert_compute_gate_up_block_impl(arg, 0u, 0u);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_load_s1_stage(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (!__moe_dyn_slot_active_this_round(cfg, st) || s1->valid == 0u) {
        return BINGO_RET_SUCC;
    }
    if (s1->block_count > BINGO_MOE_MAX_BLOCKS) return BINGO_RET_FAIL;

    uint32_t block_count = s1->block_count;
    for (uint32_t n = 0u; n < block_count; n++) {
        uint32_t rc = BINGO_RET_SUCC;
        if (n == 1u) {
            rc = __moe_pipeline_wait_cookie(
                &s1->csr_prepared_reserved, MOE_PIPELINE_INIT_COOKIE,
                1u, &sync->error);
        } else if (n >= 2u) {
            rc = __moe_pipeline_wait(
                &sync->compute_done, n - 1u, &sync->error);
        }
        if (rc != BINGO_RET_SUCC) return rc;

        rc = __moe_dynamic_expert_load_gate_up_block_impl(
            blk, cfg, s1, n);
        if (rc != BINGO_RET_SUCC) {
            __moe_pipeline_publish(&sync->error, rc);
            return rc;
        }
        __moe_pipeline_publish(&sync->prefetch_done, n + 1u);
    }
    __moe_prepare_s4pf_xdma_shape(blk, cfg, st);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s1_stage(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (!__moe_dyn_slot_active_this_round(cfg, st) || s1->valid == 0u) {
        return BINGO_RET_SUCC;
    }
    if (s1->block_count > BINGO_MOE_MAX_BLOCKS) return BINGO_RET_FAIL;

    /* Read every dynamic call record before allowing load1 to enter the B-bank
     * path. This preserves the guarded handoff that avoided the earlier
     * task-argument/TCDM conflict while keeping load1 overlapped with compute0. */
    uint32_t call_checksum = 0u;
    for (uint32_t n = 0u; n < s1->block_count; n++) {
        call_checksum ^= cfg->s1_call[n].valid;
        call_checksum ^= cfg->s1_call[n].N;
        call_checksum ^= cfg->s1_call[n].array_shape;
    }
    asm volatile("" : : "r"(call_checksum) : "memory");
    __moe_pipeline_publish(
        &s1->csr_prepared_reserved,
        MOE_PIPELINE_INIT_COOKIE | MOE_PIPELINE_COMPUTE_READY_BIT);

    __snax_bingo_kernel_moe_dynamic_expert_block_args_t block_args = *blk;
    uint32_t result = BINGO_RET_SUCC;
    for (uint32_t n = 0u; n < s1->block_count; n++) {
        result = __moe_pipeline_wait(
            &sync->prefetch_done, n + 1u, &sync->error);
        if (result != BINGO_RET_SUCC) break;
        block_args.block_idx = n;
        result = __moe_dynamic_expert_compute_gate_up_block_impl(
            &block_args, 0u, MOE_S4_CSR_LAYOUT_BLOCK_SYNC);
        if (result != BINGO_RET_SUCC) {
            __moe_pipeline_publish(&sync->error, result);
            break;
        }
        if (n + 2u < s1->block_count) {
            __moe_pipeline_publish(&sync->compute_done, n + 1u);
        }
    }

    /* The compute consumer cannot finish before the final producer transfer,
     * so it is the unique safe point to recycle the counters for S2. */
    if (result == BINGO_RET_SUCC) {
        asm volatile("fence rw, rw" ::: "memory");
        sync->compute_done = 0u;
        sync->prefetch_done = 0u;
        sync->error = BINGO_RET_SUCC;
        asm volatile("fence rw, rw" ::: "memory");
    }
    return result;
}

__attribute__((always_inline)) static inline uint32_t
__moe_dynamic_expert_load_down_block_impl(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t n, uint64_t down_src, uint32_t block_bytes,
    uint32_t block_count, uint32_t dma_binding)
{
    MOE_PROFILE_BEGIN(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_START);
    uint32_t left_offset = n * block_bytes;
    uint32_t right_offset = (block_count + n) * block_bytes;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t left_dst = __moe_bank_down_weight_block_addr(
        st->l1_b_down_addr, n, block_bytes);
    uint32_t right_dst = __moe_bank_down_weight_block_addr(
        st->l1_b_down_addr + 64u, n, block_bytes);
    uint32_t configure_xdma =
        n == 0u &&
        !__moe_xdma_stage_is_prepared(blk, MOE_XDMA_PREPARED_S3);
    uint32_t rc = __moe_dyn_copy_pair_2d(
        dma_binding,
        __moe_dyn_l1_wide(left_dst),
        down_src + left_offset,
        __moe_dyn_l1_wide(right_dst),
        down_src + right_offset,
        block_bytes, configure_xdma);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_LOAD_S3_DONE] C%u slot=%u eid=%u block=%u dma=%u "
        "bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, dma_binding, block_bytes, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_LOAD_S3,
        __moe_profile_dma_resource(dma_binding), n,
        2u * block_bytes, 0u, rc);
    return rc;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_down_block(void *arg)
{
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
        return BINGO_RET_SUCC;
    }
    if (MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u) {
        MOE_PROFILE_BEGIN(profile);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S3,
            MOE_PROFILE_RESOURCE_NONE, blk->block_idx, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    uint32_t block_bytes = st->indiv_down_B_block_stride;
    uint64_t down_src = st->indiv_down_B_l3 +
        (uint64_t)cfg->expert_id * st->indiv_down_B_expert_stride;
    return __moe_dynamic_expert_load_down_block_impl(
        blk, cfg, st, blk->block_idx, down_src, block_bytes,
        st->s3_block_count, MOE_DYN_CTRL_DMA_S3(ctrl));
}

__attribute__((always_inline)) static inline uint32_t
__moe_dynamic_expert_configure_down_block0_impl(
    void *arg, uint32_t s4_csr_layout)
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

    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (s4_csr_layout != MOE_S4_CSR_LAYOUT_SEQUENTIAL) {
        sync->sync_enabled = 0u;
        sync->reserved = 0u;
        asm volatile("fence rw, rw" ::: "memory");
    }
    if (blk->block_idx != 0u || cfg->s3_call[0].valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }

    asm volatile("fence rw, rw" ::: "memory");
    sync->compute_done = 0u;
    sync->prefetch_done = 0u;
    sync->error = BINGO_RET_SUCC;
    __moe_pipeline_publish(
        &__moe_s1_dma_ctrl(blk)->csr_prepared_reserved,
        MOE_PIPELINE_S3_SYNC_COOKIE);

    if (__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S3)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }

    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[0];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    __moe_bank_configure_mode1(
        __moe_dyn_intermediate_base(cfg, st),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr, 0u, st->indiv_down_B_block_stride),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr + 64u, 0u, st->indiv_down_B_block_stride),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st), 0u, 0u,
            st->indiv_down_N_per_block,
            st->s3_block_count),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st) + 64u, 0u, 0u,
            st->indiv_down_N_per_block, st->s3_block_count),
        st->indiv_down_K1, call->N, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S3);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_configure_down_block0(void *arg)
{
    return __moe_dynamic_expert_configure_down_block0_impl(arg, 0u);
}

__attribute__((always_inline)) static inline uint32_t
__moe_dynamic_expert_compute_down_block_impl(
    void *arg, uint32_t configure_block0, uint32_t s4_csr_layout)
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
    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[n];
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result = BINGO_RET_SUCC;
    if (n == 0u && configure_block0 != 0u &&
        !__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S3)) {
        __moe_bank_configure_mode1(
            __moe_dyn_intermediate_base(cfg, st),
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr, n, st->indiv_down_B_block_stride),
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr + 64u, n, st->indiv_down_B_block_stride),
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st), 0u, n,
                st->indiv_down_N_per_block,
                st->s3_block_count),
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st) + 64u, 0u, n,
                st->indiv_down_N_per_block, st->s3_block_count),
            st->indiv_down_K1, call->N, call->array_shape,
            st->rescale_mult, st->rescale_shift);
        __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S3);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    uint32_t preload_next = (n + 1u < st->s3_block_count) ? 1u : 0u;
    if (preload_next != 0u) {
        uint32_t next_block = n + 1u;
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
        csrw_ss(BASE_PTR_READER_1_LOW,
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr, next_block,
                st->indiv_down_B_block_stride));
        csrw_ss(BASE_PTR_READER_2_LOW,
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr + 64u, next_block,
                st->indiv_down_B_block_stride));
        csrw_ss(BASE_PTR_WRITER_0_LOW,
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st), 0u, next_block,
                st->indiv_down_N_per_block, st->s3_block_count));
        csrw_ss(BASE_PTR_WRITER_1_LOW,
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st) + 64u, 0u, next_block,
                st->indiv_down_N_per_block, st->s3_block_count));
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    } else {
        (void)__moe_prepare_s4_csr(
            blk, cfg, st, s4_csr_layout);
    }
    moe_wait_dual_vc_and_streamer();
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S3_DONE] C%u slot=%u eid=%u block=%u shape=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, call->array_shape, call->N, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
        MOE_PROFILE_RESOURCE_VERSACORE, n,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, result);
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_block(void *arg)
{
    return __moe_dynamic_expert_compute_down_block_impl(arg, 1u, 0u);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_block_pc(void *arg)
{
    return __moe_dynamic_expert_compute_down_block_impl(arg, 0u, 0u);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_load_s3_stage(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t ctrl = cfg->ctrl;
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (!__moe_dyn_slot_active_this_round(cfg, st) ||
        MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u ||
        cfg->s3_call[0].valid == 0u) {
        return BINGO_RET_SUCC;
    }
    if (st->s3_block_count > BINGO_MOE_MAX_BLOCKS) return BINGO_RET_FAIL;

    uint32_t result = __moe_pipeline_wait_cookie(
        &s1->csr_prepared_reserved, MOE_PIPELINE_S3_SYNC_COOKIE,
        0u, &sync->error);
    if (result != BINGO_RET_SUCC) return result;

    uint32_t block_count = st->s3_block_count;
    uint32_t block_bytes = st->indiv_down_B_block_stride;
    uint32_t dma_binding = MOE_DYN_CTRL_DMA_S3(ctrl);
    uint64_t down_src = st->indiv_down_B_l3 +
        (uint64_t)cfg->expert_id * st->indiv_down_B_expert_stride;
    for (uint32_t n = 0u; n < block_count; n++) {
        if (n == 1u) {
            result = __moe_pipeline_wait_cookie(
                &s1->csr_prepared_reserved, MOE_PIPELINE_S3_SYNC_COOKIE,
                1u, &sync->error);
        } else if (n >= 2u) {
            result = __moe_pipeline_wait(
                &sync->compute_done, n - 1u, &sync->error);
        }
        if (result != BINGO_RET_SUCC) return result;

        result = __moe_dynamic_expert_load_down_block_impl(
            blk, cfg, st, n, down_src, block_bytes,
            block_count, dma_binding);
        if (result != BINGO_RET_SUCC) {
            __moe_pipeline_publish(&sync->error, result);
            return result;
        }
        __moe_pipeline_publish(&sync->prefetch_done, n + 1u);
    }
    return BINGO_RET_SUCC;
}

__attribute__((always_inline)) static inline uint32_t
__moe_dyn_opt_run_s3_block(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t n)
{
    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[n];
    MOE_PROFILE_BEGIN(profile);
    if (call->valid == 0u) {
        MOE_PROFILE_COMMIT(
            (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();

    if (n + 1u < st->s3_block_count) {
        uint32_t next_block = n + 1u;
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
        csrw_ss(BASE_PTR_READER_1_LOW,
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr, next_block,
                st->indiv_down_B_block_stride));
        csrw_ss(BASE_PTR_READER_2_LOW,
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr + 64u, next_block,
                st->indiv_down_B_block_stride));
        csrw_ss(BASE_PTR_WRITER_0_LOW,
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st), 0u, next_block,
                st->indiv_down_N_per_block, st->s3_block_count));
        csrw_ss(BASE_PTR_WRITER_1_LOW,
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st) + 64u, 0u, next_block,
                st->indiv_down_N_per_block, st->s3_block_count));
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    } else {
        (void)__moe_prepare_s4_csr(
            blk, cfg, st, MOE_S4_CSR_LAYOUT_BLOCK_SYNC);
    }

    moe_wait_dual_vc_and_streamer();
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S3_DONE] C%u slot=%u eid=%u block=%u shape=%u N=%u "
        "rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, call->array_shape, call->N, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
        MOE_PROFILE_RESOURCE_VERSACORE, n,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s3_stage(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t ctrl = cfg->ctrl;
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (!__moe_dyn_slot_active_this_round(cfg, st) ||
        MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u ||
        cfg->s3_call[0].valid == 0u) {
        return BINGO_RET_SUCC;
    }
    if (st->s3_block_count > BINGO_MOE_MAX_BLOCKS) return BINGO_RET_FAIL;

    uint32_t call_checksum = 0u;
    for (uint32_t n = 0u; n < st->s3_block_count; n++) {
        call_checksum ^= cfg->s3_call[n].valid;
        call_checksum ^= cfg->s3_call[n].N;
        call_checksum ^= cfg->s3_call[n].array_shape;
    }
    asm volatile("" : : "r"(call_checksum) : "memory");
    __moe_pipeline_publish(
        &s1->csr_prepared_reserved,
        MOE_PIPELINE_S3_SYNC_COOKIE | MOE_PIPELINE_COMPUTE_READY_BIT);

    uint32_t result = BINGO_RET_SUCC;
    for (uint32_t n = 0u; n < st->s3_block_count; n++) {
        result = __moe_pipeline_wait(
            &sync->prefetch_done, n + 1u, &sync->error);
        if (result != BINGO_RET_SUCC) break;
        result = __moe_dyn_opt_run_s3_block(blk, cfg, st, n);
        if (result != BINGO_RET_SUCC) {
            __moe_pipeline_publish(&sync->error, result);
            break;
        }
        __moe_pipeline_publish(&sync->compute_done, n + 1u);
    }
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_prefetch_s2(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    MOE_PROFILE_BEGIN(profile);
    if (s2->valid == 0u) {
        __moe_prepare_s3_xdma_shape(blk, cfg, st);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_START);
    uint32_t half_bytes = s2->half_bytes;
    uint32_t block_bytes = s2->block_bytes;
    uint32_t dma_binding = s2->binding;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t rc = BINGO_RET_SUCC;
    uint32_t prepare_s3_during_final_dma =
        MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u &&
        __moe_dyn_binding_uses_xdma(MOE_DYN_CTRL_DMA_S3(cfg->ctrl)) != 0u;
    uint32_t s3_prepared_early = 0u;
    for (uint32_t n = 0u; n < s2->block_count; n++) {
        if (s2->sync_enabled != 0u && n != 0u) {
            uint32_t target = n < s2->s1_block_count ? n : s2->s1_block_count;
            rc = __moe_pipeline_wait(
                &s2->compute_done, target, &s2->error);
            if (rc != BINGO_RET_SUCC) break;
        }
        uint32_t src_offset = n * block_bytes;
        uint32_t left_dst = __moe_bank_down_weight_block_addr(
            s2->down_dst_base, n, block_bytes);
        uint32_t right_dst = __moe_bank_down_weight_block_addr(
            s2->down_dst_base + 64u, n, block_bytes);
        if (prepare_s3_during_final_dma != 0u &&
            n + 1u == s2->block_count) {
            __moe_dyn_2d_pair_pending_t pending;
            rc = __moe_dyn_start_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(left_dst),
                s2->down_src_base + src_offset,
                __moe_dyn_l1_wide(right_dst),
                s2->down_src_base + half_bytes + src_offset,
                block_bytes, n == 0u, &pending);
            if (rc == BINGO_RET_SUCC) {
                __moe_prepare_s3_xdma_shape(blk, cfg, st);
                rc = __moe_dyn_wait_pair_2d(&pending);
            }
            if (rc == BINGO_RET_SUCC) {
                s3_prepared_early = 1u;
            }
        } else {
            rc = __moe_dyn_copy_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(left_dst),
                s2->down_src_base + src_offset,
                __moe_dyn_l1_wide(right_dst),
                s2->down_src_base + half_bytes + src_offset,
                block_bytes, n == 0u);
        }
        if (rc != BINGO_RET_SUCC) {
            __moe_pipeline_publish(&s2->error, rc);
            break;
        }
        if (s2->sync_enabled != 0u && n + 1u < s2->block_count) {
            __moe_pipeline_publish(&s2->prefetch_done, n + 1u);
        }
    }
    if (rc == BINGO_RET_SUCC && prepare_s3_during_final_dma != 0u &&
        s3_prepared_early == 0u) {
        __moe_prepare_s3_xdma_shape(blk, cfg, st);
    }
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S2_DONE] C%u slot=%u eid=%u target_eid=%u "
        "dma=%u bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        s2->reserved, dma_binding, half_bytes, rc);
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
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    uint32_t slot = MOE_DYN_DMA_SLOT_S4_PREFETCH;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    uint32_t next_prepare_rc =
        __moe_dyn_prepare_next_slot_pipeline(blk, cfg, st);
    if (next_prepare_rc != BINGO_RET_SUCC) return next_prepare_rc;
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        if (__moe_dyn_has_output(cfg) != 0u &&
            s2->store_prepared == 0u) {
            __moe_dyn_prepare_store_xdma(cfg, st);
            __moe_pipeline_publish(&s2->store_prepared, 1u);
        }
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_START);
    uint32_t expert_id = MOE_DYN_DMA_EID(cfg->dma_slot_eids, slot);
    uint32_t weight_bytes = st->indiv_B_expert_stride;
    uint32_t dma_binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint64_t gate_src = st->indiv_gate_B_l3 +
        (uint64_t)expert_id * st->indiv_B_expert_stride;
    uint64_t up_src = st->indiv_up_B_l3 +
        (uint64_t)expert_id * st->indiv_B_expert_stride;
    uint32_t rc = BINGO_RET_SUCC;
    uint32_t store_prepared_early = 0u;
    for (uint32_t n = 0u; n < st->s1_block_count; n++) {
        uint32_t src_offset = n * st->indiv_B_block_stride;
        uint32_t gate_dst = __moe_bank_weight_block_addr(
            st->l1_b_gate_addr, n, st->indiv_B_block_stride);
        uint32_t up_dst = __moe_bank_weight_block_addr(
            st->l1_b_up_addr, n, st->indiv_B_block_stride);
        uint32_t configure_xdma =
            n == 0u &&
            !__moe_xdma_stage_is_prepared(
                blk, MOE_XDMA_PREPARED_S4PF);
        if (n + 1u == st->s1_block_count &&
            __moe_dyn_has_output(cfg) != 0u) {
            __moe_dyn_2d_pair_pending_t pending;
            rc = __moe_dyn_start_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                st->indiv_B_block_stride, configure_xdma, &pending);
            if (rc == BINGO_RET_SUCC) {
                __moe_dyn_prepare_store_xdma(cfg, st);
                rc = __moe_dyn_wait_pair_2d(&pending);
            }
            if (rc == BINGO_RET_SUCC) {
                __moe_pipeline_publish(&s2->store_prepared, 1u);
                store_prepared_early = 1u;
            }
        } else {
            rc = __moe_dyn_copy_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                st->indiv_B_block_stride, configure_xdma);
        }
        if (rc != BINGO_RET_SUCC) break;
    }
    if (rc == BINGO_RET_SUCC && __moe_dyn_has_output(cfg) != 0u &&
        store_prepared_early == 0u) {
        __moe_dyn_prepare_store_xdma(cfg, st);
        __moe_pipeline_publish(&s2->store_prepared, 1u);
    }
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

/* Production S4 prefetch: advance one bank-safe block step at a time with the
 * S4 compute worker. The two workers use reserved and sync_enabled as their
 * monotonically increasing completion counters. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_prefetch_s4(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    uint32_t slot = MOE_DYN_DMA_SLOT_S4_PREFETCH;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }

    MOE_PROFILE_BEGIN(profile);
    uint32_t next_prepare_rc =
        __moe_dyn_prepare_next_slot_pipeline(blk, cfg, st);
    if (next_prepare_rc != BINGO_RET_SUCC) return next_prepare_rc;
    if (__moe_s4_block_prefetch_layout_valid(st) == 0u ||
        st->s3_block_count == 0u) {
        __moe_pipeline_publish(&s2->error, BINGO_RET_FAIL);
        return BINGO_RET_FAIL;
    }

    uint32_t initial_phase = __moe_s4_block_initial_phase(st);
    uint32_t sync_steps = __moe_s4_sync_step_count(st);

    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        if (__moe_dyn_has_output(cfg) != 0u &&
            s2->store_prepared == 0u) {
            __moe_dyn_prepare_store_xdma(cfg, st);
            __moe_pipeline_publish(&s2->store_prepared, 1u);
        }
        __moe_pipeline_publish(&s2->reserved, sync_steps);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    uint32_t dma_binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    if (dma_binding != MOE_DYN_DMA_IDMA &&
        dma_binding != MOE_DYN_DMA_XDMA &&
        dma_binding != MOE_DYN_DMA_BOTH) {
        __moe_pipeline_publish(&s2->error, BINGO_RET_FAIL);
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_START);
    uint32_t expert_id = MOE_DYN_DMA_EID(cfg->dma_slot_eids, slot);
    uint32_t weight_bytes = st->indiv_B_expert_stride;
    uint64_t gate_src = st->indiv_gate_B_l3 +
        (uint64_t)expert_id * weight_bytes;
    uint64_t up_src = st->indiv_up_B_l3 +
        (uint64_t)expert_id * weight_bytes;
    uint32_t rc = BINGO_RET_SUCC;
    uint32_t dma_steps = __moe_s4_block_step_count(
        st->s1_block_count, initial_phase);
    uint32_t compute_active = cfg->s4_call.valid != 0u &&
        cfg->s4_call.M != 0u;
    uint32_t first_dma_block = 1u;

    /* S3 load may finish two blocks before S3 compute. Wait until the
     * penultimate S3 block releases the phase selected for the first S4 DMA;
     * the final S3 block then reads the opposite phase. */
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u &&
        cfg->s3_call[0].valid != 0u && st->s3_block_count > 1u) {
        rc = __moe_pipeline_wait(
            &s2->compute_done, st->s3_block_count - 1u, &s2->error);
        if (rc != BINGO_RET_SUCC) return rc;
    }

    MOE_PROFILE_RESOURCE_BEGIN(profile);
    for (uint32_t step = 0u; step < sync_steps; step++) {
        if (step != 0u && compute_active != 0u) {
            rc = __moe_pipeline_wait(
                &s2->sync_enabled, step, &s2->error);
            if (rc != BINGO_RET_SUCC) break;
        }

        uint32_t n = 0u;
        if (__moe_s4_block_at_step(
                step, st->s1_block_count, initial_phase, &n) != 0u) {
            if ((n & 1u) == 0u) {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_START);
            } else {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_START);
            }

            uint32_t src_offset = n * st->indiv_B_block_stride;
            uint32_t gate_dst = __moe_bank_weight_block_addr(
                st->l1_b_gate_addr, n, st->indiv_B_block_stride);
            uint32_t up_dst = __moe_bank_weight_block_addr(
                st->l1_b_up_addr, n, st->indiv_B_block_stride);
            uint32_t configure_xdma = first_dma_block != 0u &&
                !__moe_xdma_stage_is_prepared(
                    blk, MOE_XDMA_PREPARED_S4PF);
            uint32_t final_dma_block = step + 1u == dma_steps;

            if (final_dma_block != 0u &&
                __moe_dyn_has_output(cfg) != 0u) {
                __moe_dyn_2d_pair_pending_t pending;
                rc = __moe_dyn_start_pair_2d(
                    dma_binding,
                    __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                    __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                    st->indiv_B_block_stride, configure_xdma, &pending);
                if (rc == BINGO_RET_SUCC) {
                    __moe_dyn_prepare_store_xdma(cfg, st);
                    __moe_pipeline_publish(&s2->store_prepared, 1u);
                    rc = __moe_dyn_wait_pair_2d(&pending);
                }
            } else {
                rc = __moe_dyn_copy_pair_2d(
                    dma_binding,
                    __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                    __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                    st->indiv_B_block_stride, configure_xdma);
            }
            first_dma_block = 0u;

            if ((n & 1u) == 0u) {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_END);
            } else {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_END);
            }
            if (rc != BINGO_RET_SUCC) {
                __moe_pipeline_publish(&s2->error, rc);
                break;
            }
        }

        __moe_pipeline_publish(&s2->reserved, step + 1u);
    }

    if (rc == BINGO_RET_SUCC && __moe_dyn_has_output(cfg) != 0u &&
        s2->store_prepared == 0u) {
        __moe_dyn_prepare_store_xdma(cfg, st);
        __moe_pipeline_publish(&s2->store_prepared, 1u);
    }
    MOE_PROFILE_RESOURCE_END(profile);

    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S4_BLOCK_DONE] C%u slot=%u eid=%u "
        "target_eid=%u dma=%u bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        expert_id, dma_binding, weight_bytes, rc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
        __moe_profile_dma_resource(dma_binding), 0u,
        2u * weight_bytes, 0u, rc);
    return rc;
}

/* S2 consumes the final call record lowered by CVA6 in MoEPrepare. */
__attribute__((always_inline)) static inline uint32_t
__moe_dynamic_expert_compute_gate_up_full_impl(
    void *arg, uint32_t s4_csr_layout)
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
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    uint32_t m_tiles = call->M;
    uint32_t block_count = st->s1_block_count;
    uint32_t n_tiles =
        __moe_dyn_stage_block_n(call->N, block_count);
    if (m_tiles == 0u ||
        __moe_dyn_shape_valid(call->array_shape) == 0u ||
        n_tiles == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        __moe_pipeline_publish(&s2->error, BINGO_RET_FAIL);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u, 0u, BINGO_RET_FAIL);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result = BINGO_RET_SUCC;
    uint32_t token_start = call->token_start;
    uint32_t token_step = __moe_dyn_shape_m(call->array_shape);
    uint32_t input_base = __moe_dyn_input_base(cfg, st);
    uint32_t intermediate_base = __moe_dyn_intermediate_base(cfg, st);
    for (uint32_t mt = 0u; mt < m_tiles && result == BINGO_RET_SUCC; mt++) {
        uint32_t token = token_start + mt * token_step;
        for (uint32_t n = 0u; n < block_count; n++) {
            if (mt == 0u && s2->sync_enabled != 0u && n != 0u &&
                n <= s2->block_count) {
                result = __moe_pipeline_wait(
                    &s2->prefetch_done, n, &s2->error);
                if (result != BINGO_RET_SUCC) break;
            }
            if (mt == 0u && n == 0u &&
                !__moe_csr_stage_is_prepared(
                    blk, MOE_CSR_PREPARED_S2)) {
                __moe_bank_configure_mode0(
                    __moe_bank_a_addr(
                        input_base, token, st->A_token_bytes),
                    __moe_bank_weight_block_addr(
                        st->l1_b_gate_addr, n, st->indiv_B_block_stride),
                    __moe_bank_weight_block_addr(
                        st->l1_b_up_addr, n, st->indiv_B_block_stride),
                    __moe_bank_mode0_output_addr(
                        intermediate_base, token, n,
                        st->indiv_N_per_block,
                        block_count),
                    st->indiv_K1, n_tiles, call->array_shape,
                    st->rescale_mult, st->rescale_shift);
                __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S2);
            }
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
            moe_start_dual_vc_and_streamer();

            uint32_t next_n = n + 1u;
            uint32_t next_mt = mt;
            if (next_n == block_count) {
                next_n = 0u;
                next_mt++;
            }
            if (next_mt < m_tiles) {
                uint32_t next_token = token_start +
                    next_mt * token_step;
                __moe_bank_patch_mode0_run_bases(
                    __moe_bank_a_addr(
                        input_base, next_token, st->A_token_bytes),
                    __moe_bank_weight_block_addr(
                        st->l1_b_gate_addr, next_n,
                        st->indiv_B_block_stride),
                    __moe_bank_weight_block_addr(
                        st->l1_b_up_addr, next_n,
                        st->indiv_B_block_stride),
                    __moe_bank_mode0_output_addr(
                        intermediate_base, next_token, next_n,
                        st->indiv_N_per_block, block_count));
            } else {
                __moe_prepare_after_s2(
                    blk, cfg, st, s4_csr_layout);
            }
            moe_wait_dual_vc_and_streamer();
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
            if (mt == 0u && s2->sync_enabled != 0u &&
                n + 1u < s2->block_count) {
                __moe_pipeline_publish(&s2->compute_done, n + 1u);
            }
        }
    }
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S2_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        m_tiles, call->N, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S2,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, result);
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full(void *arg)
{
    return __moe_dynamic_expert_compute_gate_up_full_impl(arg, 0u);
}

/* S4 consumes the final call record lowered by CVA6 in MoEPrepare. */
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
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    uint32_t n_tiles =
        __moe_dyn_stage_block_n(call->N, st->s3_block_count);
    if (call->M == 0u ||
        __moe_dyn_shape_valid(call->array_shape) == 0u ||
        n_tiles == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u, 0u, BINGO_RET_FAIL);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t result = BINGO_RET_SUCC;
    uint32_t token_start = call->token_start;
    uint32_t token_step = __moe_dyn_shape_m(call->array_shape);
    uint32_t intermediate_base = __moe_dyn_intermediate_base(cfg, st);
    uint32_t output_base = __moe_dyn_output_base(cfg, st);
    for (uint32_t mt = 0u; mt < call->M; mt++) {
        uint32_t token = token_start + mt * token_step;
        for (uint32_t n = 0u; n < st->s3_block_count; n++) {
            if (mt == 0u && n == 0u &&
                !__moe_csr_stage_is_prepared(
                    blk, MOE_CSR_PREPARED_S4)) {
                __moe_bank_configure_mode1(
                    __moe_bank_mode0_output_addr(
                        intermediate_base, token, 0u,
                        st->indiv_N_per_block,
                        st->s1_block_count),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr, n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr + 64u, n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_mode1_output_addr(
                        output_base, token, n,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    __moe_bank_mode1_output_addr(
                        output_base + 64u, token, n,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    st->indiv_down_K1, n_tiles, call->array_shape,
                    st->rescale_mult, st->rescale_shift);
                __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S4);
            }
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
            moe_start_dual_vc_and_streamer();

            uint32_t next_n = n + 1u;
            uint32_t next_mt = mt;
            if (next_n == st->s3_block_count) {
                next_n = 0u;
                next_mt++;
            }
            if (next_mt < call->M) {
                uint32_t next_token = token_start +
                    next_mt * token_step;
                __moe_bank_patch_mode1_run_bases(
                    __moe_bank_mode0_output_addr(
                        intermediate_base, next_token, 0u,
                        st->indiv_N_per_block, st->s1_block_count),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr, next_n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr + 64u, next_n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_mode1_output_addr(
                        output_base, next_token, next_n,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    __moe_bank_mode1_output_addr(
                        output_base + 64u, next_token, next_n,
                        st->indiv_down_N_per_block, st->s3_block_count));
            }
            moe_wait_dual_vc_and_streamer();
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
        }
    }
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S4_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        call->M, call->N, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, result);
    return result;
}

/* Production S4 compute: run one logical block per synchronization step. The
 * block order starts on the phase opposite the final S3 reader and remains
 * valid for arbitrary runtime M and either equal or unequal stage counts. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s4(void *arg)
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
    const __snax_bingo_moe_dyn_s4_call_args_t *call = &cfg->s4_call;
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (__moe_s4_block_compute_layout_valid(cfg, st) == 0u ||
        call->array_shape > 2u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        __moe_pipeline_publish(&sync->error, BINGO_RET_FAIL);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u, 0u, BINGO_RET_FAIL);
        return BINGO_RET_FAIL;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);

    uint32_t result = BINGO_RET_SUCC;
    uint32_t token_start = call->token_start;
    uint32_t token_step = __moe_dyn_shape_m(call->array_shape);
    uint32_t m_tiles = call->M;
    uint32_t intermediate_base = __moe_dyn_intermediate_base(cfg, st);
    uint32_t output_base = __moe_dyn_output_base(cfg, st);
    uint32_t initial_phase = __moe_s4_block_initial_phase(st);
    uint32_t sync_steps = __moe_s4_sync_step_count(st);

    if (!__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S4)) {
        (void)__moe_prepare_s4_csr(
            blk, cfg, st, MOE_S4_CSR_LAYOUT_BLOCK_SYNC);
    }

    for (uint32_t step = 0u; step < sync_steps; step++) {
        if (step != 0u) {
            result = __moe_pipeline_wait(
                &sync->reserved, step, &sync->error);
            if (result != BINGO_RET_SUCC) break;
        }

        uint32_t n = 0u;
        if (__moe_s4_block_at_step(
                step, st->s3_block_count, initial_phase, &n) == 0u) {
            __moe_pipeline_publish(&sync->sync_enabled, step + 1u);
            continue;
        }

        if ((n & 1u) == 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE0_START);
        } else {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE1_START);
        }

        for (uint32_t mt = 0u; mt < m_tiles; mt++) {
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
            moe_start_dual_vc_and_streamer();

            uint32_t next_block = n;
            uint32_t next_token = token_start;
            uint32_t have_next_run = 0u;
            if (mt + 1u < m_tiles) {
                next_token = token_start + (mt + 1u) * token_step;
                have_next_run = 1u;
            } else {
                for (uint32_t next_step = step + 1u;
                     next_step < sync_steps; next_step++) {
                    if (__moe_s4_block_at_step(
                            next_step, st->s3_block_count,
                            initial_phase, &next_block) != 0u) {
                        have_next_run = 1u;
                        break;
                    }
                }
            }

            if (have_next_run != 0u) {
                __moe_bank_patch_mode1_run_bases(
                    __moe_bank_mode0_output_addr(
                        intermediate_base, next_token, 0u,
                        st->indiv_N_per_block, st->s1_block_count),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr, next_block,
                        st->indiv_down_B_block_stride),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr + 64u, next_block,
                        st->indiv_down_B_block_stride),
                    __moe_bank_mode1_output_addr(
                        output_base, next_token, next_block,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    __moe_bank_mode1_output_addr(
                        output_base + 64u, next_token, next_block,
                        st->indiv_down_N_per_block, st->s3_block_count));
            }

            moe_wait_dual_vc_and_streamer();
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
        }

        if ((n & 1u) == 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE0_END);
        } else {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE1_END);
        }
        __moe_pipeline_publish(&sync->sync_enabled, step + 1u);
    }

    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S4_BLOCK_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        m_tiles, call->N, result);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, result);
    return result;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prepare_store_xdma_2d(void *arg)
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
    if (__moe_dyn_has_output(cfg) == 0u) {
        return BINGO_RET_SUCC;
    }
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    if (s2->store_prepared != 0u) return BINGO_RET_SUCC;
    __moe_dyn_prepare_store_xdma(cfg, st);
    __moe_pipeline_publish(&s2->store_prepared, 1u);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_store_gather(void *arg)
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

    uint32_t row_stride = st->A_row_stride;
    uint64_t expert_out_base = st->output_l3_base +
        (uint64_t)cfg->expert_id * (uint64_t)st->output_expert_stride_bytes;
    uint32_t slot = MOE_DYN_CTRL_SLOT_ID(cfg->ctrl);
    volatile uint32_t *state =
        (volatile uint32_t *)(uintptr_t)st->active_state_l1_addr;
    uint32_t active_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C2_ACTIVE_SLOTS : MOE_DYN_RT_C3_ACTIVE_SLOTS;
    uint32_t has_next = (slot + 1u < state[active_idx]);
    __snax_bingo_kernel_moe_dynamic_expert_args_t *next_cfg = has_next ?
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)(
            (uintptr_t)cfg + BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES) : 0;
    uint32_t store_bytes = 0u;
    uint32_t idma_bytes = 0u;
    int32_t xdma_task0 = -1;
    uint32_t result = BINGO_RET_SUCC;
    __snax_bingo_kernel_moe_dynamic_expert_block_args_t next_blk;

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    if (has_next) {
        next_blk = *blk;
        next_blk.task_arg_addr += BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES;
        next_blk.pipeline_ctrl_addr += MOE_PIPELINE_CTRL_SLOT_BYTES;
        next_blk.block_idx = 0u;
    }

    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u ||
        MOE_DYN_CTRL_SKIP_S4(cfg->ctrl) == 0u) {
        uint64_t dst = expert_out_base +
            (uint64_t)cfg->token_ref_start * (uint64_t)row_stride;
        uint32_t output_base = __moe_dyn_output_base(cfg, st);
        uint64_t src = __moe_dyn_l1_wide(output_base);
        store_bytes = cfg->ntokens * st->A_token_bytes;
        MOE_INDIV_PRINT(
            "[INDIV_STORE_BEGIN] C%u slot=%u eid=%u start=%u ntok=%u "
            "src=0x%08x_%08x dst=0x%08x_%08x "
            "row_bytes=%u row_stride=%u bytes=%u\r\n",
            snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
            cfg->token_ref_start, cfg->ntokens,
            (uint32_t)(src >> 32u), (uint32_t)src,
            (uint32_t)(dst >> 32u), (uint32_t)dst,
            st->A_token_bytes, row_stride, store_bytes);
        xdma_task0 = (int32_t)xdma_start_remote();
    }

    if (has_next) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_START);
        idma_bytes = next_cfg->ntokens * st->A_token_bytes;
        result = __moe_dyn_gather_slot_tokens_start(next_cfg, st);
        uint32_t prepare_rc =
            __moe_dyn_prepare_pipeline_if_needed(&next_blk, next_cfg, st);
        if (result == BINGO_RET_SUCC) result = prepare_rc;
        if (result == BINGO_RET_SUCC &&
            (xdma_task0 < 0 || cfg->ntokens <= MOE_BANK_TOKEN_LANES)) {
            __moe_prepare_s1_xdma_shape(&next_blk, next_cfg, st);
        }
    }

    /* Slot parity keeps the current output and next input in opposite 16-bank
     * groups, so all output pages may drain while the next gather is active. */
    if (xdma_task0 >= 0 && cfg->ntokens > MOE_BANK_TOKEN_LANES) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)xdma_task0);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);

        uint32_t page_span = __moe_bank_token_page_span(st->A_token_bytes);
        uint32_t output_base = __moe_dyn_output_base(cfg, st);
        for (uint32_t token_start = MOE_BANK_TOKEN_LANES;
             token_start < cfg->ntokens;
             token_start += MOE_BANK_TOKEN_LANES) {
            uint32_t remaining = cfg->ntokens - token_start;
            uint32_t count = remaining < MOE_BANK_TOKEN_LANES ?
                remaining : MOE_BANK_TOKEN_LANES;
            uint32_t page = token_start / MOE_BANK_TOKEN_LANES;
            uint64_t src0 = __moe_dyn_l1_wide(
                output_base + page * page_span);
            uint64_t dst = expert_out_base +
                (uint64_t)(cfg->token_ref_start + token_start) *
                    (uint64_t)row_stride;
            __moe_bank_patch_store_page(src0, dst, count);
            int32_t page_task0 = (int32_t)xdma_start_remote();
            xdma_remote_wait((uint32_t)page_task0);
        }
        xdma_task0 = -1;
        if (has_next && result == BINGO_RET_SUCC) {
            __moe_prepare_s1_xdma_shape(&next_blk, next_cfg, st);
        }
    }

    if (has_next) {
        uint32_t wait_rc = __moe_dyn_gather_slot_tokens_wait();
        if (result == BINGO_RET_SUCC) result = wait_rc;
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_END);
    }

    if (xdma_task0 >= 0) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)xdma_task0);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
    }
    MOE_PROFILE_RESOURCE_END(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_END);

    MOE_INDIV_PRINT(
        "[INDIV_DONE] C%u slot=%u eid=%u start=%u ntok=%u "
        "store_bytes=%u next_gather_bytes=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->token_ref_start, cfg->ntokens, store_bytes, idma_bytes);
    uint32_t profile_resource = MOE_PROFILE_RESOURCE_NONE;
    if (store_bytes != 0u) {
        profile_resource = idma_bytes != 0u ? MOE_PROFILE_RESOURCE_DMA_BOTH :
                                             MOE_PROFILE_RESOURCE_XDMA;
    } else if (idma_bytes != 0u) {
        profile_resource = MOE_PROFILE_RESOURCE_IDMA;
    }
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_STORE,
        profile_resource,
        0u, store_bytes + idma_bytes, 0u,
        result);
    return result;
}

/* Remaining optimized entry points specialize bodies shared with the
 * per-block comparison API. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s2(void *arg)
{
    return __moe_dynamic_expert_compute_gate_up_full_impl(
        arg, MOE_S4_CSR_LAYOUT_BLOCK_SYNC);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_config_s3_block0(void *arg)
{
    return __moe_dynamic_expert_configure_down_block0_impl(
        arg, MOE_S4_CSR_LAYOUT_BLOCK_SYNC);
}

/* Legacy ABI adapters are retained only for the discrete comparison DFG. The
 * production optimized DFG enters each implementation directly. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_gather_s1(void *arg)
{
    return __snax_bingo_kernel_moe_dyn_opt_gather_s1(arg);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_configure_gate_up_block0(void *arg)
{
    return __snax_bingo_kernel_moe_dyn_opt_config_s1_block0(arg);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down(void *arg)
{
    return __snax_bingo_kernel_moe_dyn_opt_prefetch_s2(arg);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_store_and_gather_next(void *arg)
{
    return __snax_bingo_kernel_moe_dyn_opt_store_gather(arg);
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
