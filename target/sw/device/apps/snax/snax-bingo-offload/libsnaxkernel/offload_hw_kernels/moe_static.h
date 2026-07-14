// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// These wrappers expose the already validated MoE streamer helpers to static
// DFGs. They intentionally contain no scheduler state, slot checks, prefetch
// policy, or runtime lowering.

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_swiglu(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_swiglu_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_swiglu_args_t *cfg =
        (const __snax_bingo_kernel_moe_swiglu_args_t *)arg;
    if (cfg->array_shape > 2u || cfg->b_block_count == 0u ||
        cfg->N % cfg->b_block_count != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    return __moe_dual_vc_swiglu_full_params(
        cfg->input_A_addr,
        cfg->input_gate_B_addr,
        cfg->input_up_B_addr,
        cfg->output_D0_addr,
        cfg->output_D1_addr,
        cfg->M,
        cfg->K,
        cfg->N,
        cfg->b_block_count,
        cfg->b_block_stride,
        cfg->array_shape,
        cfg->rescale_mult,
        cfg->rescale_shift);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_down(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_down_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_down_args_t *cfg =
        (const __snax_bingo_kernel_moe_down_args_t *)arg;
    if (cfg->array_shape > 2u || cfg->b_block_count == 0u ||
        cfg->N % cfg->b_block_count != 0u ||
        (cfg->array_shape != 2u && cfg->b_block_count != 1u)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    if (cfg->array_shape == 2u) {
        return __moe_dyn_run_down_shape_c(
            cfg->input_A_addr,
            cfg->input_B0_addr,
            cfg->input_B1_addr,
            cfg->output_D0_addr,
            cfg->output_D1_addr,
            cfg->M,
            cfg->K,
            cfg->N,
            cfg->b_block_count,
            cfg->b_block_stride,
            cfg->output_row_stride,
            cfg->rescale_mult,
            cfg->rescale_shift);
    }

    return __moe_dyn_run_down(
        cfg->input_A_addr,
        cfg->input_B0_addr,
        cfg->input_B1_addr,
        cfg->output_D0_addr,
        cfg->output_D1_addr,
        cfg->M,
        cfg->K,
        cfg->N,
        cfg->array_shape,
        cfg->output_row_stride,
        cfg->rescale_mult,
        cfg->rescale_shift);
}
