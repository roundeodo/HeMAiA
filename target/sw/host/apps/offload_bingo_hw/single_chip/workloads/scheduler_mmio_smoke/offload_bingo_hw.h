// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal workload used to smoke-test the MoE scheduler MMIO slave from CVA6.

#pragma once

#include "libbingo/bingo_api.h"
#include "host.h"
#include "moe_scheduler_mmio.h"

static inline uint32_t scheduler_mmio_smoke_round(void)
{
    uint32_t errors = 0;
    const uint8_t eid0 = 0u;
    const uint16_t ntok0 = 1u;
    const uint16_t descriptors[4] = {
        moe_sched_pack_descriptor(eid0, ntok0), 0u, 0u, 0u
    };
    const uint32_t small_block_hist[4] = {1u, 0u, 0u, 0u};
    uintptr_t sched_base = moe_sched_base();

    printf_safe("[SCHED_SMOKE] base=0x%lx local=0x%lx\r\n",
                (uint64_t)sched_base, (uint64_t)MOE_SCHED_LOCAL_BASE);

    writed(moe_sched_pack_config(-1, -1, 1u, 0u, 0u),
           sched_base + (uintptr_t)MOE_SCHED_CONFIG);
    writed(moe_sched_pack_aggregate(1u, 1u, 1u, small_block_hist),
           sched_base + (uintptr_t)MOE_SCHED_AGGREGATE);
    writed(moe_sched_pack_descriptor_quad(descriptors, 0u, 1u),
           sched_base + (uintptr_t)MOE_SCHED_WINDOW0);
    writed(0u, sched_base + (uintptr_t)MOE_SCHED_WINDOW1);
    writed(0u, sched_base + (uintptr_t)MOE_SCHED_WINDOW2);
    moe_sched_fence();
    writed(0u, sched_base + (uintptr_t)MOE_SCHED_WINDOW3_START);
    moe_sched_fence();

    uint64_t event = readd(sched_base + (uintptr_t)MOE_SCHED_EVENT_WAIT);
    uint32_t task_count = moe_sched_event_task_count(event);
    printf_safe("[SCHED_SMOKE] event=0x%lx task_count=%u\r\n",
                event, task_count);
    if (moe_sched_event_batch_done(event) == 0u || task_count != 1u) {
        printf_safe("[SCHED_SMOKE] unexpected terminal event\r\n");
        errors++;
    }
    if (task_count == 0u) {
        return errors;
    }

    uint64_t task_word0 = readd(
        sched_base + (uintptr_t)MOE_SCHED_TASK_STREAM);
    uint32_t ctrl =
        (uint32_t)((task_word0 >> MOE_SCHED_TASK_WORD_CTRL_LSB) &
                   MOE_SCHED_TASK_WORD_CTRL_MASK);
    uint32_t task0_has_s2pf =
        (uint32_t)((task_word0 >> MOE_SCHED_TASK_WORD_HAS_S2PF_LSB) & 0x1u);
    uint32_t task0_skip_s3 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SKIP_S3_LSB) & 0x1u);
    uint32_t task0_skip_s1 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SKIP_S1_LSB) & 0x1u);
    uint32_t task0_s3 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SHAPE_S3_LSB) &
                   MOE_SCHED_TASK_WORD_SHAPE_MASK);
    uint32_t task0_s1 =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_SHAPE_S1_LSB) &
                   MOE_SCHED_TASK_WORD_SHAPE_MASK);
    uint32_t task0_tok_start =
        (uint32_t)((task_word0 >> MOE_SCHED_TASK_WORD_TOKEN_START_LSB) &
                   MOE_SCHED_TASK_WORD_NTOK_MASK);
    uint32_t task0_ntok =
        (uint32_t)((task_word0 >> MOE_SCHED_TASK_WORD_NTOK_LSB) &
                   MOE_SCHED_TASK_WORD_NTOK_MASK);
    uint32_t task0_eid =
        (uint32_t)((task_word0 >> MOE_SCHED_TASK_WORD_EID_LSB) &
                   MOE_SCHED_TASK_WORD_EID_MASK);
    uint32_t task0_cluster =
        (uint32_t)((ctrl >> MOE_SCHED_TASK_CTRL_CLUSTER_LSB) & 0x1u);
    uint32_t task0_s4pf_desc =
        (uint32_t)((task_word0 >> MOE_SCHED_TASK_WORD_S4PF_DESC_LSB) & 0xffu);

    printf_safe("[SCHED_SMOKE] task0=0x%lx cluster=%u eid=%u ntok=%u tok_start=%u s1=%u s3=%u skip_s1=%u skip_s3=%u has_s2pf=%u s4pf_desc=0x%x\r\n",
                task_word0, task0_cluster, task0_eid, task0_ntok, task0_tok_start,
                task0_s1, task0_s3, task0_skip_s1, task0_skip_s3,
                task0_has_s2pf, task0_s4pf_desc);

    if (task0_eid != eid0 || task0_ntok != ntok0) {
        printf_safe("[SCHED_SMOKE] unexpected task0 fields\r\n");
        errors++;
    }

    return errors;
}

int kernel_execution(void)
{
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing scheduler_mmio_smoke Workload\r\n",
                get_current_chip_loc_x(), get_current_chip_loc_y());

    uint32_t errors = scheduler_mmio_smoke_round();

    if (errors == 0u) {
        printf_safe("=== SCHEDULER MMIO SMOKE PASSED ===\r\n");
        return 0;
    }

    printf_safe("=== SCHEDULER MMIO SMOKE FAILED: %u errors ===\r\n", errors);
    return (int)errors;
}
