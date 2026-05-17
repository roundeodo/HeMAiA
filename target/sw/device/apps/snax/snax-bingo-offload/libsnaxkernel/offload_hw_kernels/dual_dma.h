// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Core-level bingo Dual-DMA kernel: launches iDMA and xDMA concurrently for
// the same Bingo DFG node, hiding xDMA latency behind iDMA and vice-versa.
// Must be included after offload_hw_kernels/idma.h and offload_hw_kernels/xdma.h
// so that snrt_dma_* and xdma_* APIs are already declared.

#pragma once

#include "../macros.h"

// __snax_bingo_kernel_dual_dma
// Arg layout (__snax_bingo_kernel_dual_dma_args_t):
//   [0]  idma_src_addr_hi
//   [1]  idma_src_addr_lo
//   [2]  idma_dst_addr_hi
//   [3]  idma_dst_addr_lo
//   [4]  idma_size          (Bytes)
//   [5]  xdma_src_addr_hi
//   [6]  xdma_src_addr_lo
//   [7]  xdma_dst_addr_hi
//   [8]  xdma_dst_addr_lo
//   [9]  xdma_size          (Bytes)
//
// Execution order:
//   1. Issue iDMA  → snrt_dma_start_1d_wideptr() (single instruction, ~6 cc)
//   2. Configure xDMA registers (851 cc, overlaps with iDMA data transfer!)
//   1. Configure xDMA registers (non-blocking setup, ~851 cc)
//   2. Start xDMA  → xdma_start() returns task_id immediately
//   3. Issue iDMA  → snrt_dma_start_1d_wideptr() (single instruction, ~6 cc)
//   4. Wait iDMA   → snrt_dma_wait_all()
//   5. Wait xDMA   → xdma_wait_task()
// xDMA is configured first: its 851-cc CSR setup must complete before iDMA starts
// because iDMA bus traffic will stall xDMA CSR writes (15000+ cc penalty observed).

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_dma(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_dual_dma_args_t);
    if (snrt_is_dm_core()) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
        uint32_t *a = (uint32_t *)arg;
        uint64_t idma_src  = make_u64(a[0], a[1]);
        uint64_t idma_dst  = make_u64(a[2], a[3]);
        uint32_t idma_size = a[4];
        uint64_t xdma_src  = make_u64(a[5], a[6]);
        uint64_t xdma_dst  = make_u64(a[7], a[8]);
        uint32_t xdma_size = a[9];
        bingo_kernel_scratchpad_t *sp = BINGO_GET_SP(arg, __snax_bingo_kernel_dual_dma_args_t);
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

        // --- Outer span: marks the entire concurrent DMA transfer ---
        BINGO_TRACE_MARKER(BINGO_TRACE_DUAL_DMA_CFG_START);

        // --- Configure and non-blocking start xDMA first ---
        // xDMA CSR writes must complete before iDMA starts; otherwise iDMA
        // bus traffic stalls xDMA configuration (15000+ cc penalty).
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_START);
        // xdma_memcpy_1d_fast_full_addr: 30 CSR writes (vs 60 for
        // xdma_disable_all_extensions+xdma_memcpy_1d_full_addr).
        // Skips clearing 15 unused multicast dst slots (saves 30 writes).
        xdma_memcpy_1d_fast_full_addr(xdma_src, xdma_dst, xdma_size);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_START);
        int xdma_task_id = xdma_start();   // non-blocking, xDMA engine begins

        // --- Issue iDMA (non-blocking) while xDMA is already running ---
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        snrt_dma_start_1d_wideptr(idma_dst, idma_src, idma_size);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);

        // --- Wait for both engines ---
        snrt_dma_wait_all();               // iDMA done
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
        xdma_wait_task(xdma_src, xdma_dst, xdma_task_id);  // xDMA done
        BINGO_TRACE_MARKER(BINGO_TRACE_XDMA_RUN_END);

        // --- Close outer span ---
        BINGO_TRACE_MARKER(BINGO_TRACE_DUAL_DMA_CFG_END);

        sp->return_value     = 0;
        sp->num_return_values = 0;
        return BINGO_RET_SUCC;
    } else {
        printf_safe("[Cluster %d Core %d]: Error! Dual DMA must be called from a DM core!\r\n",
                    snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
}
