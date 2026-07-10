#pragma once

// Pure-software MoE lowering. This file is not included in a
// MOE_SCHED_PURE_HW_PATH build.

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

static inline uint32_t __host_moe_shape_m(uint32_t shape)
{
    if (shape > 2u) shape = 2u;
    return 8u >> shape;
}

static inline uint32_t __host_moe_div_meshcol(uint32_t value, uint32_t shape)
{
    if (shape > 2u) shape = 2u;
    return value >> (shape + 2u);
}

static inline uint32_t __host_moe_prelower_task_arg(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *arg,
    const __host_bingo_kernel_moe_execute_args_t *cfg,
    uint32_t runtime_cluster_idx,
    const moe_task_t *task)
{
    for (uint32_t i = 0; i < 2u; i++) {
        arg->s1_call[i].valid = 0u;
        arg->s3_call[i].valid = 0u;
    }
    arg->s2_call.valid = 0u;
    arg->s4_call.valid = 0u;

    if (task->ntokens == 0u) return BINGO_RET_SUCC;

    uint32_t s1_blocks = (cfg->s1_block_count != 0u) ?
        (uint32_t)cfg->s1_block_count : (uint32_t)cfg->indiv_N2;
    uint32_t s3_blocks = (cfg->s3_block_count != 0u) ?
        (uint32_t)cfg->s3_block_count : (uint32_t)cfg->indiv_down_N2;
    if (s1_blocks > 2u || s3_blocks > 2u) {
        return BINGO_RET_FAIL;
    }
    if (cfg->max_tokens_per_expert == 0u ||
        task->ntokens > cfg->max_tokens_per_expert ||
        task->m_s2_exec > cfg->max_tokens_per_expert / 2u ||
        task->m_s4_exec > cfg->max_tokens_per_expert / 2u) {
        return BINGO_RET_FAIL;
    }

    uint32_t shape_s1 = (uint32_t)task->shape_s1;
    uint32_t shape_s3 = (uint32_t)task->shape_s3;
    uint32_t s1_shape_m = __host_moe_shape_m(shape_s1);
    uint32_t s3_shape_m = __host_moe_shape_m(shape_s3);
    uint32_t l1_a_addr = (runtime_cluster_idx == 0u) ?
        (uint32_t)cfg->c2_l1_a : (uint32_t)cfg->c3_l1_a;
    uint32_t l1_d_addr = (runtime_cluster_idx == 0u) ?
        (uint32_t)cfg->c2_l1_d : (uint32_t)cfg->c3_l1_d;
    uint32_t l1_down_d_addr = (runtime_cluster_idx == 0u) ?
        (uint32_t)cfg->c2_l1_down_d : (uint32_t)cfg->c3_l1_down_d;
    uint32_t s1_row_bytes = (uint32_t)cfg->indiv_D_tile_bytes /
        (uint32_t)cfg->max_tokens_per_expert;
    uint32_t down_row_bytes = (uint32_t)cfg->indiv_down_D_tile_bytes /
        (uint32_t)cfg->max_tokens_per_expert;

    if (task->skip_s1 == 0u) {
        uint32_t n_per_block =
            __host_moe_div_meshcol((uint32_t)cfg->indiv_N_per_block, shape_s1);
        for (uint32_t n = 0; n < s1_blocks; n++) {
            __snax_bingo_moe_dyn_s1_call_args_t *call = &arg->s1_call[n];
            call->valid = 1u;
            call->output_D0_addr = l1_d_addr + n * s1_shape_m * s1_row_bytes;
            call->N = n_per_block;
            call->array_shape = shape_s1;
        }
    }

    if (task->skip_s2 == 0u && task->m_s2_exec != 0u) {
        uint32_t a_offset = 0u;
        uint32_t d_offset = 0u;
        if (task->skip_s1 == 0u) {
            a_offset = s1_shape_m * (uint32_t)cfg->A_token_bytes;
            d_offset = s1_shape_m * s1_blocks * s1_row_bytes;
        }
        arg->s2_call.valid = 1u;
        arg->s2_call.input_A_addr = l1_a_addr + a_offset;
        arg->s2_call.output_D0_addr = l1_d_addr + d_offset;
        arg->s2_call.M = task->m_s2_exec;
    }

    if (task->skip_s3 == 0u) {
        uint32_t n_per_block = __host_moe_div_meshcol(
            (uint32_t)cfg->indiv_down_N_per_block, shape_s3);
        for (uint32_t n = 0; n < s3_blocks; n++) {
            __snax_bingo_moe_dyn_s3_call_args_t *call = &arg->s3_call[n];
            call->valid = 1u;
            call->N = n_per_block;
            call->array_shape = shape_s3;
        }
    }

    if (task->skip_s4 == 0u && task->m_s4_exec != 0u) {
        uint32_t a_offset = 0u;
        uint32_t d_offset = 0u;
        if (task->skip_s3 == 0u) {
            a_offset = s3_shape_m * s3_blocks * s1_row_bytes;
            d_offset = s3_shape_m * s3_blocks * down_row_bytes;
        }
        arg->s4_call.valid = 1u;
        arg->s4_call.input_A_addr = l1_d_addr + a_offset;
        arg->s4_call.output_D0_addr = l1_down_d_addr + d_offset;
        arg->s4_call.output_D1_addr = l1_down_d_addr +
            s3_blocks * (uint32_t)cfg->indiv_down_D_tile_bytes + d_offset;
        arg->s4_call.M = task->m_s4_exec;
    }

    return BINGO_RET_SUCC;
}

static inline void __host_moe_program_task_arg(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *arg,
    const moe_task_t *task,
    uint32_t local_slot,
    uint32_t runtime_cluster_idx,
    uint32_t wait_for_peer_slots)
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
    arg->token_start_rank = task->token_start_rank;
    arg->ntokens = task->ntokens;
    arg->m_s2_exec = task->m_s2_exec;
    arg->m_s4_exec = task->m_s4_exec;
    arg->wait_for_peer_slots = wait_for_peer_slots;
}
