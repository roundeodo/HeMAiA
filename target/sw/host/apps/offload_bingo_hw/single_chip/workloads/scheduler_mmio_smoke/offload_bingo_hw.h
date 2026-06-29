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
    printf_safe("[SCHED_SMOKE] base=0x%lx local=0x%lx\r\n",
                (uint64_t)moe_sched_base(), (uint64_t)MOE_SCHED_LOCAL_BASE);

    moe_sched_write64(MOE_SCHED_CONFIG, cfg_word);
    moe_sched_write64(MOE_SCHED_HEAD_PAIR1, 0u);
    moe_sched_write64(MOE_SCHED_HEAD_PAIR0,
                      moe_sched_pack_head_pair(head0, (moe_sched_head_t){0}));

    errors += scheduler_mmio_expect_u64("CONFIG",
                                        moe_sched_read64(MOE_SCHED_CONFIG),
                                        cfg_word);

    if (errors != 0u) {
        return errors;
    }

    moe_sched_write64(MOE_SCHED_CTRL,
                      MOE_SCHED_CTRL_INIT | MOE_SCHED_CTRL_START);

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

    uint64_t fifo_status = moe_sched_read64(MOE_SCHED_PLAN_FIFO_STATUS);
    uint64_t plan0 = moe_sched_read64(MOE_SCHED_PLAN_FIFO_DATA0);
    uint32_t task0_has_s2pf =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_HAS_S2PF_LSB) & 0x1u);
    uint32_t task0_skip_s3 =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_SKIP_S3_LSB) & 0x1u);
    uint32_t task0_skip_s1 =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_SKIP_S1_LSB) & 0x1u);
    uint32_t task0_s3 =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_SHAPE_S3_LSB) &
                   MOE_SCHED_PLAN_SHAPE_MASK);
    uint32_t task0_s1 =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_SHAPE_S1_LSB) &
                   MOE_SCHED_PLAN_SHAPE_MASK);
    uint32_t task0_tok_start =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_TOKEN_START_LSB) &
                   MOE_SCHED_PLAN_NTOK_MASK);
    uint32_t task0_ntok =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_NTOK_LSB) &
                   MOE_SCHED_PLAN_NTOK_MASK);
    uint32_t task0_eid =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_EID_LSB) &
                   MOE_SCHED_PLAN_EID_MASK);
    uint32_t task0_cluster =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_CLUSTER_LSB) & 0x1u);
    uint32_t task0_allow_s4pf =
        (uint32_t)((plan0 >> MOE_SCHED_PLAN_ALLOW_S4PF_LSB) & 0x1u);

    uint32_t remove_count = (uint32_t)((fifo_status >> 12) & 0x3u);
    uint32_t plan_count = (uint32_t)((fifo_status >> 8) & 0x3u);
    uint32_t slot_valid = (uint32_t)((fifo_status >> 16) & 0x3u);

    printf_safe("[SCHED_SMOKE] fifo_status=0x%lx remove_count=%u plan_count=%u slot_valid=0x%x\r\n",
                fifo_status, remove_count, plan_count, slot_valid);
    printf_safe("[SCHED_SMOKE] plan0=0x%lx cluster=%u eid=%u ntok=%u tok_start=%u s1=%u s3=%u skip_s1=%u skip_s3=%u has_s2pf=%u allow_s4pf=%u\r\n",
                plan0, task0_cluster, task0_eid, task0_ntok, task0_tok_start,
                task0_s1, task0_s3, task0_skip_s1, task0_skip_s3,
                task0_has_s2pf, task0_allow_s4pf);

    if (remove_count != 1u) {
        printf_safe("[SCHED_SMOKE] unexpected remove result\r\n");
        errors++;
    }
    if (plan_count == 0u || plan_count > 2u || slot_valid == 0u) {
        printf_safe("[SCHED_SMOKE] unexpected plan metadata\r\n");
        errors++;
    }
    if (task0_eid != eid0 || task0_ntok != ntok0) {
        printf_safe("[SCHED_SMOKE] unexpected plan0 task fields\r\n");
        errors++;
    }

    moe_sched_write64(MOE_SCHED_ROUND_COMMIT,
                      moe_sched_pack_round_commit(1u, 1u, 0u, 0u));

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
