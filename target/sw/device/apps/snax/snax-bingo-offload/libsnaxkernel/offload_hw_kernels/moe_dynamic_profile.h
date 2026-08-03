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
