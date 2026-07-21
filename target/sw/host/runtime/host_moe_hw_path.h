#pragma once

// Pure-HW scheduler state used while CVA6 drains dense RTL task words and
// materializes complete device call records in per-cluster L3 slots. This
// header has no SW schedule ABI, validation path, or fallback path.

#ifndef MOE_ENABLE_HW_SCHEDULER
#error "host_moe_hw_path.h is only valid in a pure-HW scheduler build"
#endif

typedef struct {
    uint32_t slot_bytes;
    uint32_t slot_count[2];
    uintptr_t stage_base[2];
    uint32_t l1_a_addr[2];
    uint32_t l1_d_addr[2];
    uint32_t l1_down_d_addr[2];
    int32_t final_cam[2];
} __host_moe_lower_state_t;

static inline __attribute__((always_inline)) void
__host_moe_lower_task_word_to_slot(
    uint64_t task_word,
    __host_moe_lower_state_t *state)
{
#if !defined(MOE_HW_S1_BLOCKS) || !defined(MOE_HW_S3_BLOCKS) || \
    !defined(MOE_HW_S1_ROW_BYTES) || !defined(MOE_HW_DOWN_ROW_BYTES) || \
    !defined(MOE_HW_A_ROW_STRIDE) || !defined(MOE_HW_DOWN_ROW_STRIDE) || \
    !defined(MOE_HW_DOWN_HALF_ROW_BYTES) || !defined(MOE_HW_S1_BLOCK_SPAN) || \
    !defined(MOE_HW_S1_N_BASE) || \
    !defined(MOE_HW_S3_N_BASE)
#error "pure-HW lowering requires compile-time MOE_HW_* layout constants"
#endif
#if MOE_HW_S1_BLOCKS > BINGO_MOE_MAX_BLOCKS || \
    MOE_HW_S3_BLOCKS > BINGO_MOE_MAX_BLOCKS
#error "generated MoE block count exceeds the dynamic task ABI"
#endif
    uint32_t ctrl =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_CTRL_LSB) &
                   MOE_SCHED_TASK_WORD_CTRL_MASK);
    uint32_t skip_s3 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SKIP_S3_LSB) & 0x1u);
    uint32_t skip_s1 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SKIP_S1_LSB) & 0x1u);
    uint32_t shape_s3 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SHAPE_S3_LSB) &
                   MOE_SCHED_TASK_WORD_SHAPE_MASK);
    uint32_t shape_s1 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SHAPE_S1_LSB) &
                   MOE_SCHED_TASK_WORD_SHAPE_MASK);
    uint32_t token_ref_start =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_TOKEN_START_LSB) &
                   MOE_SCHED_TASK_WORD_NTOK_MASK);
    uint32_t ntok_u =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_NTOK_LSB) &
                   MOE_SCHED_TASK_WORD_NTOK_MASK);
    uint32_t expert_id =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_EID_LSB) &
                   MOE_SCHED_TASK_WORD_EID_MASK);
    uint32_t ci =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_CLUSTER_LSB) & 0x1u);
    uint32_t has_s2pf =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_HAS_S2PF_LSB) & 0x1u);
    uint32_t local_slot =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_LOCAL_SLOT_LSB) &
                   MOE_SCHED_TASK_WORD_LOCAL_SLOT_MASK);
    uint32_t m_s2_exec =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_M_S2_LSB) &
                   MOE_SCHED_TASK_WORD_M_EXEC_MASK);
    uint32_t m_s4_exec =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_M_S4_LSB) &
                   MOE_SCHED_TASK_WORD_M_EXEC_MASK);
    uint32_t s4pf_desc =
        (uint32_t)((task_word >> MOE_SCHED_TASK_WORD_S4PF_DESC_LSB) & 0xffu);
    uint32_t s4pf_target_eid =
        (s4pf_desc >> MOE_SCHED_S4PF_DESC_TARGET_EID_LSB) &
        (uint32_t)MOE_SCHED_S4PF_DESC_TARGET_EID_MASK;
    uint32_t skip_s2 = (m_s2_exec == 0u) ? 1u : 0u;
    uint32_t skip_s4 = (m_s4_exec == 0u) ? 1u : 0u;
    uint32_t dma_s1 = skip_s1 ? (uint32_t)MOE_DMA_NONE :
        ((shape_s1 >= 2u) ? (uint32_t)MOE_DMA_BOTH : (uint32_t)MOE_DMA_IDMA);
    uint32_t dma_s3 = skip_s3 ? (uint32_t)MOE_DMA_NONE :
        ((shape_s3 >= 2u) ? (uint32_t)MOE_DMA_BOTH : (uint32_t)MOE_DMA_XDMA);
    uint32_t arg_ctrl =
        1u |
        (skip_s1 << 1u) |
        (skip_s3 << 2u) |
        (skip_s2 << 3u) |
        (skip_s4 << 4u) |
        (shape_s1 << 5u) |
        (shape_s3 << 7u) |
        (dma_s1 << 9u) |
        (dma_s3 << 11u) |
        (ci << 13u) |
        ((local_slot & 0x3fu) << 14u);
    __snax_bingo_kernel_moe_dynamic_expert_args_t *dst_arg;

#if MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_HW_LOWER);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_START);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_LOWER_SLOT);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_SLOT_SELECT_START);
    dst_arg = (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(
        state->stage_base[ci] +
        (uintptr_t)local_slot * (uintptr_t)state->slot_bytes);
    state->slot_count[ci] = local_slot + 1u;
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_SLOT_SELECT_END);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_LOWER_SLOT);
#endif

#if MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_LOWER_ARG);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_ARG_PROGRAM_START);
    dst_arg->ctrl_expert_word = MOE_PACK_U32_PAIR(arg_ctrl, expert_id);
    dst_arg->token_range_word = MOE_PACK_U32_PAIR(token_ref_start, ntok_u);
    dst_arg->m_exec_word = MOE_PACK_U32_PAIR(m_s2_exec, m_s4_exec);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_ARG_PROGRAM_END);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_LOWER_ARG);
#endif

#if MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_LOWER_PRE);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PRELOWER_START);
    uint32_t l1_d_addr = (local_slot & 1u) != 0u ?
        state->l1_a_addr[ci] : state->l1_d_addr[ci];
    uint32_t s1_shape_m = 8u >> shape_s1;
    uint32_t s3_shape_m = 8u >> shape_s3;
    uint32_t s1_n = MOE_HW_S1_N_BASE >> (shape_s1 + 2u);
    uint32_t s3_n = MOE_HW_S3_N_BASE >> (shape_s3 + 2u);

#pragma GCC unroll 8
    for (uint32_t n = 0u; n < MOE_HW_S1_BLOCKS; n++) {
        __snax_bingo_moe_dyn_s1_call_args_t *call = &dst_arg->s1_call[n];
        call->valid_output_word = MOE_PACK_U32_PAIR(
            skip_s1 == 0u,
            l1_d_addr + n * MOE_HW_S1_BLOCK_SPAN);
        call->n_shape_word = MOE_PACK_U32_PAIR(s1_n, shape_s1);
    }

    uint32_t s2_token_start = (skip_s1 == 0u) ? s1_shape_m : 0u;
    dst_arg->s2_call.valid_input_word = MOE_PACK_U32_PAIR(
        skip_s2 == 0u && m_s2_exec != 0u,
        s2_token_start);
    dst_arg->s2_call.output_m_word = MOE_PACK_U32_PAIR(
        0u, m_s2_exec);
    dst_arg->s2_call.n_shape_word = MOE_PACK_U32_PAIR(
        (MOE_HW_S1_BLOCKS * MOE_HW_S1_N_BASE) >> 4u, 2u);

#pragma GCC unroll 8
    for (uint32_t n = 0u; n < MOE_HW_S3_BLOCKS; n++) {
        __snax_bingo_moe_dyn_s3_call_args_t *call = &dst_arg->s3_call[n];
        call->valid_n_word = MOE_PACK_U32_PAIR(skip_s3 == 0u, s3_n);
        call->shape_reserved_word = MOE_PACK_U32_PAIR(shape_s3, 0u);
    }

    uint32_t s4_token_start = (skip_s3 == 0u) ? s3_shape_m : 0u;
    dst_arg->s4_call.valid_input_word = MOE_PACK_U32_PAIR(
        skip_s4 == 0u && m_s4_exec != 0u,
        s4_token_start);
    dst_arg->s4_call.output_pair_word = MOE_PACK_U32_PAIR(0u, 0u);
    dst_arg->s4_call.m_n_word = MOE_PACK_U32_PAIR(
        m_s4_exec, (MOE_HW_S3_BLOCKS * MOE_HW_S3_N_BASE) >> 4u);
    dst_arg->s4_call.shape_reserved_word = MOE_PACK_U32_PAIR(2u, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PRELOWER_END);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_LOWER_PRE);
#endif

#if MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_LOWER_DMA);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_DMA_PATCH_START);
    uint32_t dma_slot_vd = 0u;
    uint32_t dma_slot_eids = 0u;
    if (skip_s1 == 0u) {
        dma_slot_vd |=
            ((uint32_t)1u | (dma_s1 << 1u)) <<
            ((uint32_t)MOE_TASK_DMA_SLOT_S1 * 3u);
        dma_slot_eids |=
            (expert_id & 0x3fu) << ((uint32_t)MOE_TASK_DMA_SLOT_S1 * 6u);
    }
    if (skip_s3 == 0u || has_s2pf != 0u) {
        uint32_t slot = (has_s2pf != 0u) ?
            MOE_TASK_DMA_SLOT_S2_PREFETCH : MOE_TASK_DMA_SLOT_S3;
        uint32_t dma_s3_for_slot = (has_s2pf != 0u) ?
            ((shape_s3 >= 2u) ? (uint32_t)MOE_DMA_BOTH : (uint32_t)MOE_DMA_XDMA) :
            dma_s3;
        dma_slot_vd |=
            ((uint32_t)1u | (dma_s3_for_slot << 1u)) << (slot * 3u);
        dma_slot_eids |= (expert_id & 0x3fu) << (slot * 6u);
    }
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_LOWER_S4PF);
#endif
    if ((s4pf_desc & (1u << MOE_SCHED_S4PF_DESC_VALID_LSB)) != 0u &&
        (s4pf_desc & (1u << MOE_SCHED_S4PF_DESC_NO_COPY_LSB)) == 0u) {
        dma_slot_vd |=
            ((uint32_t)1u | ((uint32_t)MOE_DMA_IDMA << 1u)) <<
            ((uint32_t)MOE_TASK_DMA_SLOT_S4_PREFETCH * 3u);
        dma_slot_eids |=
            (s4pf_target_eid & 0x3fu) <<
            ((uint32_t)MOE_TASK_DMA_SLOT_S4_PREFETCH * 6u);
    }
    dst_arg->dma_slot_word = MOE_PACK_U32_PAIR(dma_slot_vd, dma_slot_eids);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_LOWER_S4PF);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_DMA_PATCH_END);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_LOWER_DMA);
#endif

    state->final_cam[ci] = (int32_t)expert_id;
#if defined(MOE_INDIV_BRINGUP) && MOE_INDIV_BRINGUP
    printf_safe(
        "[INDIV_PLAN] C%u slot=%u eid=%u start=%u ntok=%u "
        "shape_s1=%u shape_s3=%u skip=%u%u%u%u s2pf=%u "
        "s4pf=%u/%u/e%u word=0x%08x_%08x\r\n",
        ci + 2u, local_slot, expert_id, token_ref_start, ntok_u,
        shape_s1, shape_s3, skip_s1, skip_s2, skip_s3, skip_s4,
        has_s2pf,
        (s4pf_desc >> MOE_SCHED_S4PF_DESC_VALID_LSB) & 1u,
        (s4pf_desc >> MOE_SCHED_S4PF_DESC_NO_COPY_LSB) & 1u,
        s4pf_target_eid, (uint32_t)(task_word >> 32u),
        (uint32_t)task_word);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_END);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_HW_LOWER);
#endif
}

static inline void __host_moe_run_scheduler_and_lower(
    const uint32_t *expert_token_counts,
    uint32_t n_experts,
    int16_t cache_eid_c2,
    int16_t cache_eid_c3,
    __host_moe_lower_state_t *state)
{
    uint16_t rem_head[MOE_MAX_EXPERTS];
    uint16_t total_conc = 0u;
    uint16_t next_rem_pos = 0u;
    uint16_t rem_count = 0u;
    uintptr_t sched_base = moe_sched_base();

#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_HW_SORT);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_SORT_START);
    for (uint32_t e = 0u; e < n_experts; e++) {
        uint32_t cur_ntokens = expert_token_counts[e];
        if (cur_ntokens == 0u) {
            continue;
        }
        uint16_t pos = rem_count;
        while (pos > 0u &&
               (rem_head[pos - 1u] & 0x1ffu) < cur_ntokens) {
            rem_head[pos] = rem_head[pos - 1u];
            pos--;
        }
        rem_head[pos] = (uint16_t)(
            (cur_ntokens & 0x1ffu) |
            ((e & 0x3fu) << MOE_SCHED_NTOK_W) |
            (1u << (MOE_SCHED_NTOK_W + MOE_SCHED_EID_RAW_W)));
        rem_count++;
        total_conc = (uint16_t)(total_conc +
                                (uint16_t)(((cur_ntokens + 3u) >> 2) * 6u));
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_SORT_END);
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_HW_SORT);
#endif

#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_HW_INIT);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_INIT_WRITE_START);

    uint64_t init_head_word = 0u;
    uint64_t reserve_head_word = 0u;
    for (uint8_t s = 0; s < 4u; s++) {
        if (next_rem_pos < rem_count) {
            init_head_word |= (uint64_t)rem_head[next_rem_pos] << (s * 16u);
            next_rem_pos++;
        }
    }
    for (uint8_t s = 0; s < 4u; s++) {
        if (next_rem_pos < rem_count) {
            reserve_head_word |= (uint64_t)rem_head[next_rem_pos] << (s * 16u);
            next_rem_pos++;
        }
    }
    writed(((uint64_t)((cache_eid_c2 < 0) ? 0x80u : (uint8_t)cache_eid_c2)) |
           ((uint64_t)((cache_eid_c3 < 0) ? 0x80u : (uint8_t)cache_eid_c3) << 8) |
           ((uint64_t)(rem_count & 0x7fu) << 16) |
           ((uint64_t)total_conc << 32),
           sched_base + (uintptr_t)MOE_SCHED_CONFIG);
    writed(init_head_word, sched_base + (uintptr_t)MOE_SCHED_HEAD_QUAD);
    writed(reserve_head_word, sched_base + (uintptr_t)MOE_SCHED_RESERVE_QUAD);
    moe_sched_fence();
    writed(MOE_SCHED_CTRL_INIT | MOE_SCHED_CTRL_START,
           sched_base + (uintptr_t)MOE_SCHED_CTRL);
    moe_sched_fence();
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_HW_INIT);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_INIT_WRITE_END);

    while (1) {
        uint64_t status;
        uint32_t status_fifo_count;
        uint32_t status_refill_req;
        uint32_t status_refill_count;
        uint32_t status_active_empty;
        uint32_t event_seen;

#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
        __moe_host_timing_start(MOE_HOST_TIMING_HW_WAIT);
#endif
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_WAIT_START);
        event_seen = 0u;
        while (event_seen == 0u) {
            status = readd(sched_base + (uintptr_t)MOE_SCHED_STATUS);
            uint32_t wait_fifo_count =
                (uint32_t)((status >> MOE_SCHED_STATUS_TASK_COUNT_LSB) & 0xfu);
            if ((status & (MOE_SCHED_STATUS_REFILL_REQ |
                           MOE_SCHED_STATUS_ACTIVE_EMPTY)) != 0u ||
                ((status & MOE_SCHED_STATUS_TASK_VALID) != 0u &&
                 wait_fifo_count >= MOE_SCHED_TASK_FIFO_DEPTH)) {
                event_seen = 1u;
                moe_sched_fence();
                break;
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_WAIT_END);
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
        __moe_host_timing_end(MOE_HOST_TIMING_HW_WAIT);
#endif

#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
        __moe_host_timing_start(MOE_HOST_TIMING_HW_STATUS);
#endif
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_STATUS_DECODE_START);
        status_fifo_count =
            (uint32_t)((status >> MOE_SCHED_STATUS_TASK_COUNT_LSB) & 0xfu);
        status_refill_req = ((status & MOE_SCHED_STATUS_REFILL_REQ) != 0u) ? 1u : 0u;
        status_refill_count = (uint32_t)((status >> 28) & 0xfu);
        status_active_empty = ((status & MOE_SCHED_STATUS_ACTIVE_EMPTY) != 0u) ? 1u : 0u;
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_STATUS_DECODE_END);
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
        __moe_host_timing_end(MOE_HOST_TIMING_HW_STATUS);
#endif

        if (status_refill_req != 0u && status_refill_count != 0u) {
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
            __moe_host_timing_start(MOE_HOST_TIMING_HW_CONTROL);
#endif
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_CONTROL_WRITE_START);
            uint64_t push_head_word = 0u;
            for (uint32_t ph = 0u; ph < status_refill_count; ph++) {
                push_head_word |=
                    (uint64_t)rem_head[next_rem_pos] << (ph * 16u);
                next_rem_pos++;
            }
            writed(push_head_word,
                   sched_base + (uintptr_t)MOE_SCHED_HEAD_PUSH_QUAD);
            moe_sched_fence();
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_CONTROL_WRITE_END);
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
            __moe_host_timing_end(MOE_HOST_TIMING_HW_CONTROL);
#endif
        }

        /* Refill only feeds the scheduler input stream. Keep a partial dense
         * task FIFO resident so later rounds can fill the remaining entries;
         * drain only when the FIFO is full or the batch has completed. */
        if ((status & MOE_SCHED_STATUS_TASK_VALID) != 0u &&
            (status_fifo_count >= MOE_SCHED_TASK_FIFO_DEPTH ||
             status_active_empty != 0u)) {
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
            __moe_host_timing_start(MOE_HOST_TIMING_HW_DRAIN);
#endif
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_DRAIN_TASKS_START);
            uint32_t drain_tasks = status_fifo_count;

            for (uint32_t task = 0u; task < drain_tasks; task++) {
                uint64_t task_word = readd(
                    sched_base + (uintptr_t)(
                        MOE_SCHED_TASK_DATA_BASE +
                        task * MOE_SCHED_TASK_DATA_STRIDE));
                __host_moe_lower_task_word_to_slot(task_word, state);
            }
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_DRAIN_TASKS_END);
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
            __moe_host_timing_end(MOE_HOST_TIMING_HW_DRAIN);
#endif

#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
            __moe_host_timing_start(MOE_HOST_TIMING_HW_CONTROL);
#endif
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_CONTROL_WRITE_START);
            writed((uint64_t)(drain_tasks & 0xfu),
                   sched_base + (uintptr_t)MOE_SCHED_TASK_POP);
            moe_sched_fence();
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_CONTROL_WRITE_END);
#if defined(MOE_ENABLE_HW_SCHEDULER) && MOE_MCYCLE_DETAIL
            __moe_host_timing_end(MOE_HOST_TIMING_HW_CONTROL);
#endif

            continue;
        }

        if (status_refill_req != 0u && status_refill_count != 0u) {
            continue;
        }

        if (status_active_empty != 0u &&
            (status & MOE_SCHED_STATUS_TASK_VALID) == 0u) {
            break;
        }
    }

}

static inline uint64_t __host_bingo_kernel_moe_prepare_request(void *arg)
{
    __moe_host_timing_start(MOE_HOST_TIMING_PREPARE);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __host_bingo_kernel_moe_prepare_request_args_t *cfg =
        (const __host_bingo_kernel_moe_prepare_request_args_t *)arg;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PREPARE_START);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PREPARE_INIT_START);

    const uint32_t *expert_token_counts =
        (const uint32_t *)(uintptr_t)cfg->expert_token_counts_addr;
    int32_t *cam_state = (int32_t *)(uintptr_t)cfg->cam_state_addr;
    volatile uint32_t *runtime_state =
        (volatile uint32_t *)(uintptr_t)cfg->runtime_state_addr;
    uint32_t n_experts = (uint32_t)cfg->n_experts;
    int32_t c2_resident = cam_state[0];
    int32_t c3_resident = cam_state[1];
    uint32_t slot_bytes = (uint32_t)cfg->dynamic_arg_slot_bytes;
    __host_moe_lower_state_t lower_state = {
        .slot_bytes = slot_bytes,
        .slot_count = {0u, 0u},
        .stage_base = {
            (uintptr_t)cfg->c2_stage_base,
            (uintptr_t)cfg->c3_stage_base,
        },
        .l1_a_addr = {(uint32_t)cfg->c2_l1_a, (uint32_t)cfg->c3_l1_a},
        .l1_d_addr = {(uint32_t)cfg->c2_l1_d, (uint32_t)cfg->c3_l1_d},
        .l1_down_d_addr = {
            (uint32_t)cfg->c2_l1_down_d,
            (uint32_t)cfg->c3_l1_down_d,
        },
        .final_cam = {-1, -1},
    };
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PREPARE_INIT_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_SCHED_START);
    __moe_host_timing_start(MOE_HOST_TIMING_HW_SCHED);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_SCHED_START);
    __host_moe_run_scheduler_and_lower(
        expert_token_counts,
        n_experts,
        (int16_t)c2_resident,
        (int16_t)c3_resident,
        &lower_state);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_SCHED_END);
    __moe_host_timing_end(MOE_HOST_TIMING_HW_SCHED);

#if MOE_MCYCLE_DETAIL
    __moe_host_timing_start(MOE_HOST_TIMING_LOWER_FINAL);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_START);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_FINAL_STATE_START);
    uint32_t c2_slots = lower_state.slot_count[0];
    uint32_t c3_slots = lower_state.slot_count[1];
    cam_state[0] = lower_state.final_cam[0];
    cam_state[1] = lower_state.final_cam[1];
    runtime_state[2] = c2_slots;
    runtime_state[3] = c3_slots;
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_FINAL_STATE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_LOWER_END);
#if MOE_MCYCLE_DETAIL
    __moe_host_timing_end(MOE_HOST_TIMING_LOWER_FINAL);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_SCHED_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PREPARE_END);
    __moe_host_timing_end(MOE_HOST_TIMING_PREPARE);
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_moe_execute(void *arg)
{
    __moe_host_timing_start(MOE_HOST_TIMING_EXECUTE);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __host_bingo_kernel_moe_execute_args_t *cfg =
        (const __host_bingo_kernel_moe_execute_args_t *)arg;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXECUTE_START);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_INIT_START);

    volatile uint32_t *runtime_state =
        (volatile uint32_t *)(uintptr_t)cfg->runtime_state_addr;
    uint32_t c2_slots = runtime_state[2];
    uint32_t c3_slots = runtime_state[3];
    uint32_t slot_bytes = (uint32_t)cfg->dynamic_arg_slot_bytes;
    uint32_t *c2_stage_header = (uint32_t *)(uintptr_t)cfg->c2_stage_base;
    uint32_t *c3_stage_header = (uint32_t *)(uintptr_t)cfg->c3_stage_base;
    uint64_t active_slot_word = MOE_PACK_U32_PAIR(c2_slots, c3_slots);
    ((uint64_t *)c2_stage_header)[1] = active_slot_word;
    ((uint64_t *)c3_stage_header)[1] = active_slot_word;
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_INIT_END);

    uint64_t c2_runtime_bytes = 64u +
        (uint64_t)c2_slots * (uint64_t)slot_bytes;
    uint64_t c3_runtime_bytes = 64u +
        (uint64_t)c3_slots * (uint64_t)slot_bytes;
    uint32_t chip_id = get_current_chip_id();

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_FLUSH1_START);
    asm volatile("fence rw, rw" ::: "memory");
    (void)sys_dma_memcpy(
        chip_id, cfg->c2_active_state_l1_addr,
        (uint64_t)chiplet_addr_transform_full(chip_id, cfg->c2_stage_base),
        c2_runtime_bytes);
    uint32_t last_tf_id = (uint32_t)sys_dma_memcpy(
        chip_id, cfg->c3_active_state_l1_addr,
        (uint64_t)chiplet_addr_transform_full(chip_id, cfg->c3_stage_base),
        c3_runtime_bytes);
    /* Gather runs on a cluster DM core. Keep its scalar token-reference reads
     * in local TCDM instead of stalling descriptor submission on L3 reads. */
    uint64_t token_refs_src = (uint64_t)chiplet_addr_transform_full(
        chip_id, cfg->expert_token_refs_addr);
    if (c2_slots != 0u) {
        last_tf_id = (uint32_t)sys_dma_memcpy(
            chip_id, cfg->c2_token_refs_l1_addr, token_refs_src,
            cfg->token_refs_bytes);
    }
    if (c3_slots != 0u) {
        last_tf_id = (uint32_t)sys_dma_memcpy(
            chip_id, cfg->c3_token_refs_l1_addr, token_refs_src,
            cfg->token_refs_bytes);
    }
    while (*(sys_dma_done_ptr(chip_id)) != last_tf_id) {
        asm volatile("nop");
    }
    asm volatile("fence rw, rw" ::: "memory");
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_FLUSH1_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXECUTE_END);
    __moe_host_timing_end(MOE_HOST_TIMING_EXECUTE);
    return BINGO_RET_SUCC;
}
