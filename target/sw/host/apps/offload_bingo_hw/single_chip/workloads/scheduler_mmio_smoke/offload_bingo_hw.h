// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal workload used to smoke-test the MoE scheduler MMIO slave from CVA6.

#pragma once

#include <string.h>
#include "libbingo/bingo_api.h"
#include "host.h"
#include "moe_scheduler_mmio.h"

static inline uint32_t scheduler_mmio_expect_u64(const char *name,
                                                 uint64_t got,
                                                 uint64_t exp)
{
    if (got != exp) {
        printf_safe("[SCHED_SMOKE] %s mismatch: got=0x%lx exp=0x%lx\r\n",
                    name, got, exp);
        return 1u;
    }
    return 0u;
}

static inline uint32_t scheduler_mmio_smoke_round(void)
{
    uint32_t errors = 0;
    uint64_t status = 0;

    const uint8_t cache_none = 0x80u;
    const uint8_t eid0 = 0u;
    const uint16_t ntok0 = 1u;
    const uint16_t best_conc0 = moe_sched_best_conc(ntok0);

    moe_sched_head_t head0 = {
        .valid = 1u,
        .rem_index = 0u,
        .eid = eid0,
        .ntok = ntok0,
        .input_order = 0u,
        .best_conc = best_conc0,
    };

    uint64_t cfg_word = moe_sched_pack_config(cache_none, cache_none, 1u, best_conc0);
    uint64_t head0_word = moe_sched_pack_head(head0);

    printf_safe("[SCHED_SMOKE] base=0x%lx local=0x%lx\r\n",
                (uint64_t)moe_sched_base(), (uint64_t)MOE_SCHED_LOCAL_BASE);

    moe_sched_write64(MOE_SCHED_CONFIG, cfg_word);
    moe_sched_clear_heads();
    moe_sched_write64(MOE_SCHED_HEAD0, head0_word);

    errors += scheduler_mmio_expect_u64("CONFIG",
                                        moe_sched_read64(MOE_SCHED_CONFIG),
                                        cfg_word);
    errors += scheduler_mmio_expect_u64("HEAD0",
                                        moe_sched_read64(MOE_SCHED_HEAD0),
                                        head0_word);
    errors += scheduler_mmio_expect_u64("HEAD1",
                                        moe_sched_read64(MOE_SCHED_HEAD1),
                                        0u);

    if (errors != 0u) {
        return errors;
    }

    moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_INIT);
    moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_START);

    if (moe_sched_wait_done(&status) != 0) {
        printf_safe("[SCHED_SMOKE] timeout waiting done, status=0x%lx\r\n",
                    status);
        return errors + 1u;
    }

    printf_safe("[SCHED_SMOKE] status=0x%lx\r\n", status);
    if ((status & MOE_SCHED_STATUS_REMOVE_VALID) == 0u) {
        printf_safe("[SCHED_SMOKE] remove_valid not set\r\n");
        errors++;
    }
    if ((status & MOE_SCHED_STATUS_PLAN_VALID) == 0u) {
        printf_safe("[SCHED_SMOKE] plan_valid not set\r\n");
        errors++;
    }

    uint64_t remove = moe_sched_read64(MOE_SCHED_REMOVE);
    uint64_t meta = moe_sched_read64(MOE_SCHED_PLAN_META);
    uint64_t plan0 = moe_sched_read64(MOE_SCHED_PLAN0);
    uint64_t makespan = moe_sched_read64(MOE_SCHED_MAKESPAN);
    moe_sched_task_t task0 = moe_sched_unpack_task(plan0);

    uint32_t remove_count = (uint32_t)(remove & 0x3u);
    uint32_t remove_slot_mask = (uint32_t)((remove >> 4) & 0xfu);
    uint32_t plan_count = (uint32_t)(meta & 0x3u);
    uint32_t slot_valid = (uint32_t)((meta >> 8) & 0x3u);

    printf_safe("[SCHED_SMOKE] remove=0x%lx count=%u slot_mask=0x%x\r\n",
                remove, remove_count, remove_slot_mask);
    printf_safe("[SCHED_SMOKE] meta=0x%lx plan_count=%u slot_valid=0x%x\r\n",
                meta, plan_count, slot_valid);
    printf_safe("[SCHED_SMOKE] plan0=0x%lx cluster=%u eid=%u ntok=%u tok_start=%u s1=%u s3=%u skip_s1=%u skip_s3=%u has_s2pf=%u allow_s4pf=%u\r\n",
                plan0, task0.cluster, task0.eid, task0.ntok, task0.tok_start,
                task0.s1, task0.s3, task0.skip_s1, task0.skip_s3,
                task0.has_s2pf, task0.allow_s4pf);
    printf_safe("[SCHED_SMOKE] makespan=%lu\r\n", makespan);

    if (remove_count != 1u || remove_slot_mask != 0x1u) {
        printf_safe("[SCHED_SMOKE] unexpected remove result\r\n");
        errors++;
    }
    if (plan_count == 0u || plan_count > 2u || slot_valid == 0u) {
        printf_safe("[SCHED_SMOKE] unexpected plan metadata\r\n");
        errors++;
    }
    if (task0.eid != eid0 || task0.ntok != ntok0) {
        printf_safe("[SCHED_SMOKE] unexpected plan0 task fields\r\n");
        errors++;
    }

    moe_sched_write64(MOE_SCHED_CTRL,
                      MOE_SCHED_CTRL_REMOVE_READY | MOE_SCHED_CTRL_PLAN_POP);

    return errors;
}

static inline void scheduler_mmio_make_case(moe_request_t *req, uint32_t case_id)
{
    uint16_t n = 1u;
    uint32_t pattern = case_id;

    switch (case_id) {
    case 0: n = 1u; break;
    case 1: n = 2u; break;
    case 2: n = 3u; break;
    case 3: n = 4u; break;
    case 4: n = 8u; break;
    case 5: n = 16u; break;
    case 6: n = 32u; break;
    default: n = 64u; break;
    }

    memset(req, 0, sizeof(*req));
    req->n_experts = n;

    for (uint16_t i = 0; i < n; i++) {
        req->experts[i].expert_id = i;
        switch (pattern & 3u) {
        case 0:
            req->experts[i].ntokens = (uint16_t)(1u + (i % 32u));
            break;
        case 1:
            req->experts[i].ntokens = (uint16_t)(1u + ((n - i) % 64u));
            break;
        case 2:
            req->experts[i].ntokens = (uint16_t)(1u + ((i + n) % 4u));
            break;
        default:
            req->experts[i].ntokens =
                (uint16_t)(1u + (((uint32_t)i * 17u + n * 5u + case_id * 11u) % 96u));
            break;
        }
    }

    switch (case_id % 5u) {
    case 0:
        req->cache_eid_c2 = -1;
        req->cache_eid_c3 = -1;
        break;
    case 1:
        req->cache_eid_c2 = (int16_t)(n / 2u);
        req->cache_eid_c3 = -1;
        break;
    case 2:
        req->cache_eid_c2 = -1;
        req->cache_eid_c3 = (int16_t)(n - 1u);
        break;
    default:
        req->cache_eid_c2 = 0;
        req->cache_eid_c3 = (int16_t)(n - 1u);
        break;
    }
}

static inline uint32_t scheduler_mmio_run_batch_case(uint32_t case_id)
{
    static moe_request_t req;
    static moe_hw_plan_entry_t hw_plan[MOE_MAX_TASKS];
    static moe_hw_plan_entry_t sw_plan[MOE_MAX_TASKS];

    uint16_t hw_n = 0;
    uint16_t sw_n = 0;
    uint32_t errors = 0;

    memset(hw_plan, 0, sizeof(hw_plan));
    memset(sw_plan, 0, sizeof(sw_plan));
    scheduler_mmio_make_case(&req, case_id);

    printf_safe("[SCHED_BATCH] case=%u n=%u cache_c2=%d cache_c3=%d\r\n",
                case_id, req.n_experts, req.cache_eid_c2, req.cache_eid_c3);

    moe_status_t sw_st = moe_make_hw_plan(&req, sw_plan, &sw_n);
    if (sw_st != MOE_OK) {
        printf_safe("[SCHED_BATCH] software golden failed case=%u st=%d\r\n",
                    case_id, sw_st);
        return 1u;
    }

    int hw_st = moe_sched_hw_make_plan_mmio(&req, hw_plan, &hw_n, 0u);
    if (hw_st != 0) {
        printf_safe("[SCHED_BATCH] hardware driver failed case=%u st=%d\r\n",
                    case_id, hw_st);
        return 1u;
    }

    errors += moe_sched_compare_plan(hw_plan, hw_n, sw_plan, sw_n, case_id);
    if (errors == 0u) {
        printf_safe("[SCHED_BATCH] case=%u PASSED plan_entries=%u\r\n",
                    case_id, hw_n);
    } else {
        printf_safe("[SCHED_BATCH] case=%u FAILED errors=%u hw_n=%u sw_n=%u\r\n",
                    case_id, errors, hw_n, sw_n);
    }
    return errors;
}

static inline uint32_t scheduler_mmio_run_batch_suite(void)
{
    uint32_t total_errors = 0;
    const uint32_t num_cases = 8u;

    for (uint32_t case_id = 0; case_id < num_cases; case_id++) {
        total_errors += scheduler_mmio_run_batch_case(case_id);
    }

    if (total_errors == 0u) {
        printf_safe("=== SCHEDULER MMIO BATCH TEST PASSED ===\r\n");
    } else {
        printf_safe("=== SCHEDULER MMIO BATCH TEST FAILED: %u errors ===\r\n",
                    total_errors);
    }

    return total_errors;
}

int kernel_execution(void)
{
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing scheduler_mmio_smoke Workload\r\n",
                get_current_chip_loc_x(), get_current_chip_loc_y());

    uint32_t errors = scheduler_mmio_smoke_round();
    if (errors == 0u) {
        errors += scheduler_mmio_run_batch_suite();
    }

    if (errors == 0u) {
        printf_safe("=== SCHEDULER MMIO SMOKE PASSED ===\r\n");
        return 0;
    }

    printf_safe("=== SCHEDULER MMIO SMOKE FAILED: %u errors ===\r\n", errors);
    return (int)errors;
}
