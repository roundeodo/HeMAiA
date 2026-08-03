// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

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

/* Dynamic-MoE 2D pairs always cross L3 and cluster-local L1, so their xDMA
 * completion is remote and does not need retained source/destination state. */
typedef struct {
    uint32_t binding;
    uint32_t repeats;
    uint32_t wait_idma;
    int32_t xdma_task0;
    int32_t xdma_task1;
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
    uint32_t probe_start;
#endif
} __moe_dyn_2d_pair_pending_t;

__attribute__((always_inline)) static inline void __moe_dyn_start_pair_2d(
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
        return;
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
        pending->xdma_task0 = xdma_start_remote();
        xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
        pending->xdma_task1 = xdma_start_remote();
        return;
    }

    if (configure_xdma != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
        xdma_memcpy_2d_fast_configure(
            MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, repeats);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    }
    xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
    pending->xdma_task0 = xdma_start_remote();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_2d_wideptr(
        dst0_addr, src0_addr, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    pending->wait_idma = 1u;
}

__attribute__((always_inline)) static inline void __moe_dyn_start_single_2d(
    uint32_t binding,
    uint64_t dst_addr,
    uint64_t src_addr,
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

    if (binding == MOE_DYN_DMA_IDMA) {
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        snrt_dma_start_2d_wideptr(
            dst_addr, src_addr, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        pending->wait_idma = 1u;
        return;
    }

    if (configure_xdma != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
        xdma_memcpy_2d_fast_configure(
            MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, repeats);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    }
    xdma_memcpy_fast_set_addresses(src_addr, dst_addr);
    pending->xdma_task0 = xdma_start_remote();
}

__attribute__((always_inline)) static inline int32_t
__moe_dyn_start_single_2d_preloaded_xdma(
    void)
{
    return xdma_start_remote();
}

__attribute__((always_inline)) static inline void
__moe_dyn_start_single_2d_idma(
    uint64_t dst_addr,
    uint64_t src_addr,
    uint32_t bytes)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_2d_wideptr(
        dst_addr, src_addr, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
        bytes / MOE_BANK_WEIGHT_ROW_BYTES);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
}

__attribute__((always_inline)) static inline void
__moe_dyn_wait_single_2d_idma(void)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
}

__attribute__((always_inline)) static inline void
__moe_dyn_wait_single_2d_xdma(int32_t task_id)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
    xdma_remote_wait((uint32_t)task_id);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
}

__attribute__((always_inline)) static inline void __moe_dyn_wait_pair_2d(
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
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)pending->xdma_task0);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
    }
    if (pending->xdma_task1 >= 0) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)pending->xdma_task1);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
    }
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_pair_2d_xdma_addresses(
    uint32_t binding,
    uint64_t dst0_addr,
    uint64_t src0_addr,
    uint64_t dst1_addr,
    uint64_t src1_addr)
{
    if (binding == MOE_DYN_DMA_XDMA) {
        xdma_memcpy_fast_set_addresses(src0_addr, dst0_addr);
    } else if (binding == MOE_DYN_DMA_BOTH) {
        xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
    }
}

__attribute__((always_inline)) static inline void
__moe_dyn_s2pf_both_addresses(
    const __moe_s2_prefetch_ctrl_t *s2,
    uint32_t block,
    uint64_t *dst0_addr,
    uint64_t *src0_addr,
    uint64_t *dst1_addr,
    uint64_t *src1_addr)
{
    uint32_t block_bytes = s2->block_bytes;
    *dst0_addr = __moe_dyn_l1_wide(__moe_bank_down_weight_block_addr(
        s2->down_dst_base, block, block_bytes));
    *src0_addr = s2->down_src_base + (uint64_t)block * block_bytes;
    *dst1_addr = __moe_dyn_l1_wide(__moe_bank_down_weight_block_addr(
        s2->down_dst_base + 64u, block, block_bytes));
    *src1_addr = *src0_addr + s2->half_bytes;
}

__attribute__((always_inline)) static inline void
__moe_dyn_s2pf_single_address(
    const __moe_s2_prefetch_ctrl_t *s2,
    uint32_t block,
    uint32_t side,
    uint64_t *dst_addr,
    uint64_t *src_addr)
{
    uint32_t block_bytes = s2->block_bytes;
    uint32_t src_offset = block * block_bytes + side * s2->half_bytes;
    *dst_addr = __moe_dyn_l1_wide(__moe_bank_down_weight_block_addr(
        s2->down_dst_base + side * 64u, block, block_bytes));
    *src_addr = s2->down_src_base + src_offset;
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s2pf_both_xdma_address(
    const __moe_s2_prefetch_ctrl_t *s2,
    uint32_t block)
{
    uint64_t dst0 = 0u;
    uint64_t src0 = 0u;
    uint64_t dst1 = 0u;
    uint64_t src1 = 0u;
    __moe_dyn_s2pf_both_addresses(
        s2, block, &dst0, &src0, &dst1, &src1);
    __moe_dyn_prepare_pair_2d_xdma_addresses(
        MOE_DYN_DMA_BOTH, dst0, src0, dst1, src1);
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s2pf_single_xdma_address(
    const __moe_s2_prefetch_ctrl_t *s2,
    uint32_t block,
    uint32_t side)
{
    uint64_t dst = 0u;
    uint64_t src = 0u;
    __moe_dyn_s2pf_single_address(
        s2, block, side, &dst, &src);
    xdma_memcpy_fast_set_addresses(src, dst);
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s2pf_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk)
{
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    __moe_prepare_s2pf_xdma_shape(blk);
    if (s2->binding == MOE_DYN_DMA_BOTH) {
        __moe_dyn_prepare_s2pf_both_xdma_address(s2, 0u);
    } else {
        __moe_dyn_prepare_s2pf_single_xdma_address(s2, 0u, 0u);
    }
}

__attribute__((always_inline)) static inline void
__moe_dyn_s1_pair_addresses(
    const __moe_s1_dma_ctrl_t *s1,
    uint32_t block,
    uint64_t *dst0_addr,
    uint64_t *src0_addr,
    uint64_t *dst1_addr,
    uint64_t *src1_addr)
{
    uint32_t block_bytes = s1->block_bytes;
    uint32_t weight_offset = block * block_bytes;
    *dst0_addr = __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
        s1->gate_dst_base, block, block_bytes));
    *src0_addr = s1->gate_src_base + weight_offset;
    *dst1_addr = __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
        s1->up_dst_base, block, block_bytes));
    *src1_addr = s1->up_src_base + weight_offset;
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s1_xdma_address(
    const __moe_s1_dma_ctrl_t *s1,
    uint32_t block)
{
    uint64_t dst0 = 0u;
    uint64_t src0 = 0u;
    uint64_t dst1 = 0u;
    uint64_t src1 = 0u;
    __moe_dyn_s1_pair_addresses(
        s1, block, &dst0, &src0, &dst1, &src1);
    if (s1->binding == MOE_DYN_DMA_XDMA) {
        xdma_memcpy_fast_set_addresses(src0, dst0);
    } else {
        xdma_memcpy_fast_set_addresses(src1, dst1);
    }
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s1_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    if (s1->valid == 0u ||
        __moe_dyn_binding_uses_xdma(s1->binding) == 0u) {
        return;
    }
    __moe_prepare_s1_xdma_shape(blk, cfg, st);
    __moe_dyn_prepare_s1_xdma_address(s1, 0u);
}

__attribute__((always_inline)) static inline void
__moe_dyn_s3_pair_addresses(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t down_src,
    uint32_t block,
    uint64_t *dst0_addr,
    uint64_t *src0_addr,
    uint64_t *dst1_addr,
    uint64_t *src1_addr)
{
    uint32_t block_bytes = st->indiv_down_B_block_stride;
    *dst0_addr = __moe_dyn_l1_wide(__moe_bank_down_weight_block_addr(
        st->l1_b_down_addr, block, block_bytes));
    *src0_addr = down_src + (uint64_t)block * block_bytes;
    *dst1_addr = __moe_dyn_l1_wide(__moe_bank_down_weight_block_addr(
        st->l1_b_down_addr + 64u, block, block_bytes));
    *src1_addr = down_src +
        (uint64_t)(st->s3_block_count + block) * block_bytes;
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s3_xdma_address(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t down_src,
    uint32_t binding,
    uint32_t block)
{
    uint64_t dst0 = 0u;
    uint64_t src0 = 0u;
    uint64_t dst1 = 0u;
    uint64_t src1 = 0u;
    __moe_dyn_s3_pair_addresses(
        st, down_src, block, &dst0, &src0, &dst1, &src1);
    if (binding == MOE_DYN_DMA_XDMA) {
        xdma_memcpy_fast_set_addresses(src0, dst0);
    } else {
        xdma_memcpy_fast_set_addresses(src1, dst1);
    }
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s3_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint32_t binding = MOE_DYN_CTRL_DMA_S3(cfg->ctrl);
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) != 0u ||
        cfg->s3_call[0].valid == 0u ||
        __moe_dyn_binding_uses_xdma(binding) == 0u) {
        return;
    }
    uint32_t weight_eid = MOE_DYN_DMA_EID(
        cfg->dma_slot_eids, MOE_DYN_DMA_SLOT_S3);
    uint64_t down_src = st->indiv_down_B_l3 +
        (uint64_t)weight_eid * st->indiv_down_B_expert_stride;
    __moe_prepare_s3_xdma_shape(blk, cfg, st);
    __moe_dyn_prepare_s3_xdma_address(st, down_src, binding, 0u);
}

__attribute__((always_inline)) static inline void
__moe_dyn_s4pf_phase_addresses(
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t phase,
    uint64_t *gate_dst,
    uint64_t *gate_src_phase,
    uint64_t *up_dst,
    uint64_t *up_src_phase)
{
    uint32_t block_bytes = st->indiv_B_block_stride;
    uint32_t src_offset = phase * block_bytes;
    *gate_dst = __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
        st->l1_b_gate_addr, phase, block_bytes));
    *gate_src_phase = gate_src + src_offset;
    *up_dst = __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
        st->l1_b_up_addr, phase, block_bytes));
    *up_src_phase = up_src + src_offset;
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s4pf_xdma_phase(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t binding,
    uint32_t phase)
{
    uint64_t gate_dst = 0u;
    uint64_t gate_src_phase = 0u;
    uint64_t up_dst = 0u;
    uint64_t up_src_phase = 0u;
    __moe_prepare_s4pf_xdma_phase_shape(
        blk, st->indiv_B_block_stride, st->s1_block_count, phase);
    __moe_dyn_s4pf_phase_addresses(
        st, gate_src, up_src, phase,
        &gate_dst, &gate_src_phase, &up_dst, &up_src_phase);
    if (binding == MOE_DYN_DMA_XDMA) {
        xdma_memcpy_fast_set_addresses(gate_src_phase, gate_dst);
    } else {
        xdma_memcpy_fast_set_addresses(up_src_phase, up_dst);
    }
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_s4pf_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint32_t slot = MOE_DYN_DMA_SLOT_S4_PREFETCH;
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) return;
    uint32_t binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    if (__moe_dyn_binding_uses_xdma(binding) == 0u) return;
    uint32_t weight_eid = MOE_DYN_DMA_EID(cfg->dma_slot_eids, slot);
    uint64_t gate_src = st->indiv_gate_B_l3 +
        (uint64_t)weight_eid * st->indiv_B_expert_stride;
    uint64_t up_src = st->indiv_up_B_l3 +
        (uint64_t)weight_eid * st->indiv_B_expert_stride;
    __moe_dyn_prepare_s4pf_xdma_phase(
        blk, st, gate_src, up_src, binding,
        __moe_s4_block_initial_phase(st));
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_after_s2pf_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    uint32_t s3_active = MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u &&
        cfg->s3_call[0].valid != 0u;
    uint32_t s3_binding = MOE_DYN_CTRL_DMA_S3(cfg->ctrl);
    if (s3_active != 0u &&
        __moe_dyn_binding_uses_xdma(s3_binding) != 0u) {
        __moe_dyn_prepare_s3_xdma(blk, cfg, st);
        return;
    }
    __moe_dyn_prepare_s4pf_xdma(blk, cfg, st);
}

__attribute__((always_inline)) static inline int32_t
__moe_dyn_xdma_start_remote_begin(void)
{
    uint32_t previous =
        snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
    snax_write_xdma_cfg_reg(XDMA_START_PTR, 1u);
    return (int32_t)previous;
}

__attribute__((always_inline)) static inline int32_t
__moe_dyn_xdma_start_remote_commit(int32_t previous)
{
    uint32_t committed;
    do {
        committed = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
    } while (committed == (uint32_t)previous);
    return (int32_t)committed;
}

__attribute__((always_inline)) static inline int32_t
__moe_dyn_start_both_2d_preloaded_xdma(
    uint64_t dst0_addr,
    uint64_t src0_addr,
    uint32_t bytes)
{
    uint32_t repeats = bytes / MOE_BANK_WEIGHT_ROW_BYTES;
    int32_t previous = __moe_dyn_xdma_start_remote_begin();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_2d_wideptr(
        dst0_addr, src0_addr, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    return __moe_dyn_xdma_start_remote_commit(previous);
}

__attribute__((always_inline)) static inline void
__moe_dyn_start_pair_2d_idma(
    uint64_t dst0_addr,
    uint64_t src0_addr,
    uint64_t dst1_addr,
    uint64_t src1_addr,
    uint32_t bytes)
{
    uint32_t repeats = bytes / MOE_BANK_WEIGHT_ROW_BYTES;
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_2d_wideptr(
        dst0_addr, src0_addr, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    snrt_dma_start_2d_wideptr(
        dst1_addr, src1_addr, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
}

__attribute__((always_inline)) static inline int32_t
__moe_dyn_start_pair_2d_preloaded_xdma(
    uint64_t dst1_addr,
    uint64_t src1_addr)
{
    (void)xdma_start_remote();
    xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
    return (int32_t)xdma_start_remote();
}

__attribute__((always_inline)) static inline void
__moe_dyn_wait_pair_2d_idma(void)
{
    __moe_dyn_wait_single_2d_idma();
}

__attribute__((always_inline)) static inline void
__moe_dyn_wait_pair_2d_xdma(int32_t last_task)
{
    __moe_dyn_wait_single_2d_xdma(last_task);
}

__attribute__((always_inline)) static inline void
__moe_dyn_wait_both_2d(int32_t xdma_task)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
    xdma_remote_wait((uint32_t)xdma_task);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
}

__attribute__((always_inline)) static inline void __moe_dyn_copy_pair_2d(
    uint32_t binding,
    uint64_t dst0_addr,
    uint64_t src0_addr,
    uint64_t dst1_addr,
    uint64_t src1_addr,
    uint32_t bytes,
    uint32_t configure_xdma)
{
    __moe_dyn_2d_pair_pending_t pending;
    __moe_dyn_start_pair_2d(
        binding, dst0_addr, src0_addr, dst1_addr, src1_addr,
        bytes, configure_xdma, &pending);
    __moe_dyn_wait_pair_2d(&pending);
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

/* Keep the final-S4 store preparation independent of TCDM stack traffic.
 * This deliberately writes a complete xDMA context: callers may arrive after
 * any runtime-selected DMA binding or shape.  Every CSR address is explicit so
 * Clang emits direct CSR instructions instead of the dynamic csrw_ss switch. */
__attribute__((noinline)) static void __moe_bank_configure_store(
    uint64_t src, uint64_t dst, uint32_t token_count,
    uint32_t token_bytes)
{
    uint32_t enabled = (1u << token_count) - 1u;

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

    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 2u, 1u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 2u, 0u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 3u, 1u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 3u, 0u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_BOUND_PTR + 4u, 1u);
    snax_write_xdma_cfg_reg(XDMA_SRC_TEMP_STRIDE_PTR + 4u, 0u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 2u, 1u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 2u, 0u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 3u, 1u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 3u, 0u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_BOUND_PTR + 4u, 1u);
    snax_write_xdma_cfg_reg(XDMA_DST_TEMP_STRIDE_PTR + 4u, 0u);

    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_BYTE_PTR, 0xffffffffu);
    snax_write_xdma_cfg_reg(XDMA_SRC_ENABLED_CHAN_PTR, enabled);
    snax_write_xdma_cfg_reg(XDMA_DST_ENABLED_CHAN_PTR, enabled);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, (uint32_t)(src >> 32));
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, (uint32_t)(dst >> 32));
}

static inline uint32_t __moe_dyn_has_output(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg)
{
    return MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u ||
        MOE_DYN_CTRL_SKIP_S4(cfg->ctrl) == 0u;
}

__attribute__((always_inline)) static inline void
__moe_dyn_prepare_store_xdma(
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

/* Final S4 block with a following store.  Launch every weight transfer before
 * deriving the following store configuration so L1 argument traffic cannot
 * delay either DMA engine. */
__attribute__((always_inline)) static inline void
__moe_dyn_start_final_pair_2d_and_prepare_store(
    uint32_t binding,
    uint64_t dst0_addr,
    uint64_t src0_addr,
    uint64_t dst1_addr,
    uint64_t src1_addr,
    uint32_t bytes,
    uint32_t configure_xdma,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    __moe_dyn_2d_pair_pending_t *pending)
{
    uint32_t repeats = bytes / MOE_BANK_WEIGHT_ROW_BYTES;

    pending->binding = binding;
    pending->repeats = repeats;
    pending->wait_idma = binding != MOE_DYN_DMA_XDMA;
    pending->xdma_task0 = -1;
    pending->xdma_task1 = -1;
    if (binding == MOE_DYN_DMA_IDMA) {
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        snrt_dma_start_2d_wideptr(
            dst0_addr, src0_addr, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        snrt_dma_start_2d_wideptr(
            dst1_addr, src1_addr, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        __moe_dyn_prepare_store_xdma(cfg, st);
        return;
    }

    if (configure_xdma != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
        xdma_memcpy_2d_fast_configure(
            MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, repeats);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    }

    if (binding == MOE_DYN_DMA_XDMA) {
        xdma_memcpy_fast_set_addresses(src0_addr, dst0_addr);
        int32_t task0 = (int32_t)xdma_start_remote();
        xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
        int32_t task1 = (int32_t)xdma_start_remote();
        pending->xdma_task0 = task0;
        pending->xdma_task1 = task1;
        __moe_dyn_prepare_store_xdma(cfg, st);
        return;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    snrt_dma_start_2d_wideptr(
        dst0_addr, src0_addr, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    xdma_memcpy_fast_set_addresses(src1_addr, dst1_addr);
    int32_t task0 = (int32_t)xdma_start_remote();
    pending->xdma_task0 = task0;
    __moe_dyn_prepare_store_xdma(cfg, st);
}
