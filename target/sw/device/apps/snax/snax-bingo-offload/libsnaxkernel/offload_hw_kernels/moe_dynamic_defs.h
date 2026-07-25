// Dynamic MoE constants, control records, profiling, and common utilities.
#pragma once

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
