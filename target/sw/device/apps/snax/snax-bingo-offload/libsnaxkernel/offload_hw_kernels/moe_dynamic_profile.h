// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

static inline uint32_t __moe_profile_cluster_marker(
    void *arg, uint32_t stage)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    MOE_SCOPE_PROFILE_BEGIN(profile);
    MOE_SCOPE_PROFILE_COMMIT(arg, cfg, profile, stage);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_cluster_begin(void *arg)
{
    return __moe_profile_cluster_marker(
        arg, MOE_PROFILE_STAGE_CLUSTER_BEGIN);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_cluster_end(void *arg)
{
    return __moe_profile_cluster_marker(
        arg, MOE_PROFILE_STAGE_CLUSTER_END);
}

// Low-overhead marker for full-workload scopes that do not have dynamic-expert
// arguments (Router release and shared-expert branch). dummy_input carries one
// bingo_moe_runtime_timing_stage value; the Bingo trailer supplies scratchpad.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_timing_marker(void *arg)
{
    const __snax_bingo_kernel_dummy_args_t *cfg =
        (const __snax_bingo_kernel_dummy_args_t *)arg;
#if MOE_RUNTIME_TIMING >= 1
    bingo_kernel_scratchpad_t *sp =
        BINGO_GET_SP(arg, __snax_bingo_kernel_dummy_args_t);
    uint32_t start = snrt_mcycle();
    sp->reserved[MOE_SP_PROFILE_MAGIC_IDX] = 0u;
    sp->start_time = start;
    sp->reserved[MOE_SP_PROFILE_META_IDX] = MOE_PROFILE_META(
        cfg->dummy_input, MOE_PROFILE_RESOURCE_NONE,
        snrt_cluster_idx(), snrt_cluster_core_idx(), 0u);
    sp->reserved[MOE_SP_PROFILE_TASK_IDX] = 0u;
    sp->reserved[MOE_SP_PROFILE_RESOURCE_START_IDX] = start;
    sp->reserved[MOE_SP_PROFILE_RESOURCE_END_IDX] = start;
    sp->reserved[MOE_SP_PROFILE_PEER_WAIT_IDX] = 0u;
    sp->reserved[MOE_SP_PROFILE_UNITS_IDX] = 0u;
    sp->reserved[MOE_SP_PROFILE_RESULT_IDX] = BINGO_RET_SUCC;
    sp->reserved[MOE_SP_PROFILE_FLAGS_IDX] = MOE_PROFILE_FLAG_ACTIVE;
    sp->end_time = snrt_mcycle();
    sp->reserved[MOE_SP_PROFILE_MAGIC_IDX] = MOE_RUNTIME_TIMING_MAGIC;
#else
    (void)cfg;
#endif
    return BINGO_RET_SUCC;
}
