// Dynamic-slot protocol decoding, synchronization, and shape helpers.
#pragma once

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

static inline void __moe_pipeline_wait(
    volatile uint32_t *counter, uint32_t value)
{
    while (*counter < value) {}
    asm volatile("fence rw, rw" ::: "memory");
}

static inline void __moe_pipeline_wait_cookie(
    volatile uint32_t *cookie, uint32_t base,
    uint32_t require_compute_ready)
{
    for (;;) {
        uint32_t value = *cookie;
        uint32_t base_matches =
            (value & ~MOE_PIPELINE_COMPUTE_READY_BIT) == base;
        uint32_t ready_matches = require_compute_ready == 0u ||
            (value & MOE_PIPELINE_COMPUTE_READY_BIT) != 0u;
        if (base_matches && ready_matches) break;
    }
    asm volatile("fence rw, rw" ::: "memory");
}

static inline uint32_t __moe_dyn_binding_uses_xdma(uint32_t binding)
{
    return binding == MOE_DYN_DMA_XDMA || binding == MOE_DYN_DMA_BOTH;
}

static inline uint32_t __moe_dyn_stage_block_n(
    uint32_t stage_n, uint32_t block_count)
{
    return stage_n / block_count;
}

static inline uint32_t __moe_s4_block_initial_phase(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (st->s3_block_count - 1u) & 1u;
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
