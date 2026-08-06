#pragma once

// Pure-software scheduling and CVA6-side complete call-record generation. This
// file is not included in a MOE_ENABLE_HW_SCHEDULER build.

#ifndef MOE_HW_WEIGHT_BACKING_MASK
#error "pure-SW lowering requires MOE_HW_WEIGHT_BACKING_MASK"
#endif

#include "host_moe_s2pf_mode.h"

static inline __attribute__((always_inline)) uint32_t
__host_moe_weight_backing_id(uint32_t logical_eid)
{
    return logical_eid & (uint32_t)MOE_HW_WEIGHT_BACKING_MASK;
}

static inline int __host_moe_dma_slot_index(moe_dma_op_kind_t kind)
{
    switch (kind) {
    case MOE_DMA_OP_S1: return (int)MOE_TASK_DMA_SLOT_S1;
    case MOE_DMA_OP_S3: return (int)MOE_TASK_DMA_SLOT_S3;
    case MOE_DMA_OP_S2_PREFETCH: return (int)MOE_TASK_DMA_SLOT_S2_PREFETCH;
    case MOE_DMA_OP_S4_PREFETCH: return (int)MOE_TASK_DMA_SLOT_S4_PREFETCH;
    default: return -1;
    }
}

static inline uint32_t __host_moe_write_task_slot(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *arg,
    const __host_bingo_kernel_moe_execute_args_t *cfg,
    uint32_t runtime_cluster_idx,
    const moe_task_t *task,
    uint32_t local_slot)
{
    arg->ctrl =
        1u |
        ((uint32_t)task->skip_s1 << 1u) |
        ((uint32_t)task->skip_s3 << 2u) |
        ((uint32_t)task->skip_s2 << 3u) |
        ((uint32_t)task->skip_s4 << 4u) |
        ((uint32_t)task->shape_s1 << 5u) |
        ((uint32_t)task->shape_s3 << 7u) |
        ((uint32_t)task->dma_s1 << 9u) |
        ((uint32_t)task->dma_s3 << 11u) |
        (runtime_cluster_idx << 13u) |
        ((local_slot & 0x3fu) << 14u);
    arg->expert_id = task->expert_id;
    arg->token_ref_start = task->token_start_rank;
    arg->ntokens = task->ntokens;
    arg->m_s2_exec = task->m_s2_exec;
    arg->m_s4_exec = task->m_s4_exec;
    arg->dma_slot_word = 0u;

    uint32_t s1_blocks = (uint32_t)cfg->s1_block_count;
    uint32_t s3_blocks = (uint32_t)cfg->s3_block_count;
    uint32_t shape_s1 = (uint32_t)task->shape_s1;
    uint32_t shape_s3 = (uint32_t)task->shape_s3;
    uint32_t s1_shape_m = 8u >> shape_s1;
    uint32_t s3_shape_m = 8u >> shape_s3;
    uint32_t l1_d_addr;
    if ((local_slot & 1u) != 0u) {
        l1_d_addr = (runtime_cluster_idx == 0u) ?
            (uint32_t)cfg->c2_l1_a : (uint32_t)cfg->c3_l1_a;
    } else {
        l1_d_addr = (runtime_cluster_idx == 0u) ?
            (uint32_t)cfg->c2_l1_d : (uint32_t)cfg->c3_l1_d;
    }
    uint32_t s1_block_span =
        ((uint32_t)cfg->indiv_N_per_block / 8u) * 512u;

    for (uint32_t n = 0u; n < s1_blocks; n++) {
        __snax_bingo_moe_dyn_s1_call_args_t *call = &arg->s1_call[n];
        call->valid = task->skip_s1 == 0u;
        call->output_D0_addr = l1_d_addr + n * s1_block_span;
        call->N = (uint32_t)cfg->indiv_N_per_block >> (shape_s1 + 2u);
        call->array_shape = shape_s1;
    }

    uint32_t s2_token_start = (task->skip_s1 == 0u) ? s1_shape_m : 0u;
    arg->s2_call.valid = task->skip_s2 == 0u && task->m_s2_exec != 0u;
    arg->s2_call.token_start = s2_token_start;
    arg->s2_call.reserved = 0u;
    arg->s2_call.M = task->m_s2_exec;
    arg->s2_call.N = s1_blocks * (uint32_t)cfg->indiv_N_per_block >> 4u;
    arg->s2_call.array_shape = 2u;

    for (uint32_t n = 0u; n < s3_blocks; n++) {
        __snax_bingo_moe_dyn_s3_call_args_t *call = &arg->s3_call[n];
        call->valid = task->skip_s3 == 0u;
        call->N = (uint32_t)cfg->indiv_down_N_per_block >> (shape_s3 + 2u);
        call->array_shape = shape_s3;
        call->reserved = 0u;
    }

    uint32_t s4_token_start = (task->skip_s3 == 0u) ? s3_shape_m : 0u;
    arg->s4_call.valid = task->skip_s4 == 0u && task->m_s4_exec != 0u;
    arg->s4_call.token_start = s4_token_start;
    arg->s4_call.reserved0 = 0u;
    arg->s4_call.reserved1 = 0u;
    arg->s4_call.M = task->m_s4_exec;
    arg->s4_call.N = s3_blocks * (uint32_t)cfg->indiv_down_N_per_block >> 4u;
    arg->s4_call.array_shape = 2u;
    arg->s4_call.reserved = 0u;

    return BINGO_RET_SUCC;
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

    uint32_t *expert_token_counts =
        (uint32_t *)(uintptr_t)cfg->expert_token_counts_addr;
    uint16_t *expert_token_refs =
        (uint16_t *)(uintptr_t)cfg->expert_token_refs_addr;
    const uint16_t *topk_indices =
        (const uint16_t *)(uintptr_t)cfg->topk_indices_l3;
    int32_t *cam_state = (int32_t *)(uintptr_t)cfg->cam_state_addr;
    moe_request_t *request = (moe_request_t *)(uintptr_t)cfg->request_out_addr;
    moe_schedule_t *schedule =
        (moe_schedule_t *)(uintptr_t)cfg->schedule_out_addr;
    uint32_t n_experts = (uint32_t)cfg->n_experts;
    uint32_t total_tokens = (uint32_t)cfg->M_total;
    uint32_t top_k = (uint32_t)cfg->top_k;
    uint32_t token_stride = (uint32_t)cfg->max_tokens_per_expert;
    request->n_experts = 0u;
    schedule->n_tasks = 0u;
    schedule->n_dma_ops = 0u;
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PREPARE_INIT_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_TOKEN_COUNT_START);
    uint32_t cursor[MOE_MAX_EXPERTS];
    for (uint32_t e = 0; e < n_experts; e++) {
        cursor[e] = e * token_stride;
    }
    for (uint32_t t = 0; t < total_tokens; t++) {
        uint32_t base = t * top_k;
        for (uint32_t k = 0; k < top_k; k++) {
            uint32_t expert_id = topk_indices[base + k];
            expert_token_refs[cursor[expert_id]++] =
                BINGO_MOE_TOKEN_REF_PACK(t, k);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_TOKEN_COUNT_END);

    int32_t c2_resident = cam_state[0];
    int32_t c3_resident = cam_state[1];
    if (c2_resident < 0 || c2_resident >= (int32_t)n_experts) {
        c2_resident = 0;
    }
    if (c3_resident < 0 || c3_resident >= (int32_t)n_experts ||
        (c2_resident == 0 && c3_resident == 0 && n_experts > 1u)) {
        c3_resident = (int32_t)(n_experts - 1u);
    }
    cam_state[0] = c2_resident;
    cam_state[1] = c3_resident;

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_REQUEST_BUILD_START);
    request->cache_eid_c2 = (int16_t)c2_resident;
    request->cache_eid_c3 = (int16_t)c3_resident;
    for (uint32_t e = 0; e < n_experts; e++) {
        if (expert_token_counts[e] == 0u) continue;
        uint32_t idx = request->n_experts++;
        request->experts[idx].expert_id = (uint16_t)e;
        request->experts[idx].ntokens = (uint16_t)expert_token_counts[e];
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_REQUEST_BUILD_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_SCHED_START);
    __moe_host_timing_start(MOE_HOST_TIMING_SW_SCHED);
    moe_status_t status = moe_schedule(request, schedule);
    __moe_host_timing_end(MOE_HOST_TIMING_SW_SCHED);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_SCHED_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_SCHED_PRINT_START);
    __moe_dbg_print_schedule(schedule);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_SCHED_PRINT_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_PREPARE_END);
    __moe_host_timing_end(MOE_HOST_TIMING_PREPARE);
    return (status == MOE_OK) ? BINGO_RET_SUCC : BINGO_RET_FAIL;
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

    moe_schedule_t *schedule =
        (moe_schedule_t *)(uintptr_t)cfg->schedule_addr;
    int32_t *cam_state = (int32_t *)(uintptr_t)cfg->cam_state_addr;
    volatile uint32_t *runtime_state =
        (volatile uint32_t *)(uintptr_t)cfg->runtime_state_addr;
    uint32_t slot_bytes = (uint32_t)cfg->dynamic_arg_slot_bytes;
    uint32_t c2_total = 0u;
    uint32_t c3_total = 0u;
    for (uint32_t ti = 0; ti < schedule->n_tasks; ti++) {
        if (schedule->tasks[ti].cluster == MOE_CLUSTER_C2) c2_total++;
        else c3_total++;
    }

    uint32_t *c2_stage_header = (uint32_t *)(uintptr_t)cfg->c2_stage_base;
    uint32_t *c3_stage_header = (uint32_t *)(uintptr_t)cfg->c3_stage_base;
    c2_stage_header[2] = c2_total;
    c2_stage_header[3] = c3_total;
    c3_stage_header[2] = c2_total;
    c3_stage_header[3] = c3_total;

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_INIT_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_PROGRAM_START);
    __snax_bingo_kernel_moe_dynamic_expert_args_t *task_args[MOE_MAX_TASKS];
    uint32_t c2_slots = 0u;
    uint32_t c3_slots = 0u;
    for (uint32_t task_idx = 0; task_idx < schedule->n_tasks; task_idx++) {
        const moe_task_t *task = &schedule->tasks[task_idx];
        uint32_t runtime_cluster_idx;
        uint32_t local_slot;
        __snax_bingo_kernel_moe_dynamic_expert_args_t *dst_arg;
        if (task->cluster == MOE_CLUSTER_C2) {
            runtime_cluster_idx = 0u;
            local_slot = c2_slots++;
            dst_arg = (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
                (cfg->c2_stage_base + 64u + (uint64_t)local_slot * slot_bytes);
        } else {
            runtime_cluster_idx = 1u;
            local_slot = c3_slots++;
            dst_arg = (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
                (cfg->c3_stage_base + 64u + (uint64_t)local_slot * slot_bytes);
        }
        __host_moe_write_task_slot(
            dst_arg,
            cfg,
            runtime_cluster_idx,
            task,
            local_slot);
        task_args[task_idx] = dst_arg;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_PROGRAM_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_DMA_FILL_START);
    for (uint32_t op_idx = 0; op_idx < schedule->n_dma_ops; op_idx++) {
        const moe_dma_op_t *op = &schedule->dma_ops[op_idx];
        __snax_bingo_kernel_moe_dynamic_expert_args_t *dst_arg =
            task_args[op->task_idx];
        uint32_t slot = (uint32_t)__host_moe_dma_slot_index(op->kind);
        if (op->kind == MOE_DMA_OP_S2_PREFETCH) {
            const moe_task_t *task = &schedule->tasks[op->task_idx];
            dst_arg->ctrl |= __host_moe_s2pf_runtime_ctrl(
                1u, (uint32_t)task->skip_s1,
                (uint32_t)task->shape_s1, (uint32_t)task->dma_s1);
        }
        dst_arg->dma_slot_vd |=
            (1u | ((uint32_t)op->dma << 1u)) << (slot * 3u);
        dst_arg->dma_slot_eids |=
            (__host_moe_weight_backing_id((uint32_t)op->expert_id) & 0x3fu) <<
            (slot * 6u);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_DMA_FILL_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_CAM_START);
    int32_t final_cam[2] = {-1, -1};
    for (uint32_t ti = 0; ti < schedule->n_tasks; ti++) {
        const moe_task_t *task = &schedule->tasks[ti];
        final_cam[(task->cluster == MOE_CLUSTER_C2) ? 0 : 1] =
            (int32_t)task->expert_id;
    }
    for (uint32_t oi = 0; oi < schedule->n_dma_ops; oi++) {
        const moe_dma_op_t *op = &schedule->dma_ops[oi];
        if (op->kind != MOE_DMA_OP_S4_PREFETCH) continue;
        const moe_task_t *task = &schedule->tasks[op->task_idx];
        final_cam[(task->cluster == MOE_CLUSTER_C2) ? 0 : 1] =
            (int32_t)op->expert_id;
    }
    cam_state[0] = final_cam[0];
    cam_state[1] = final_cam[1];
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_CAM_END);

    uint32_t chip_id = get_current_chip_id();
    uint64_t c2_runtime_bytes = 64u + (uint64_t)c2_slots * slot_bytes;
    uint64_t c3_runtime_bytes = 64u + (uint64_t)c3_slots * slot_bytes;
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_FLUSH1_START);
    asm volatile("fence rw, rw" ::: "memory");
    sys_dma_blk_memcpy(
        chip_id, cfg->c2_active_state_l1_addr,
        (uint64_t)chiplet_addr_transform_full(chip_id, cfg->c2_stage_base),
        c2_runtime_bytes);
    sys_dma_blk_memcpy(
        chip_id, cfg->c3_active_state_l1_addr,
        (uint64_t)chiplet_addr_transform_full(chip_id, cfg->c3_stage_base),
        c3_runtime_bytes);
    asm volatile("fence rw, rw" ::: "memory");
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXEC_FLUSH1_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_EXECUTE_END);
    __moe_host_timing_end(MOE_HOST_TIMING_EXECUTE);
    return BINGO_RET_SUCC;
}
