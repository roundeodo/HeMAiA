// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: Apache-2.0
//
// CVA6-side MMIO driver for the MoE scheduler register slave.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "chip_id.h"
#include "io.h"
#include "uart.h"
#include "moe_scheduler.h"

#ifndef MOE_SCHED_LOCAL_BASE
#define MOE_SCHED_LOCAL_BASE 0x05010000ull
#endif

#define MOE_SCHED_CTRL      0x00u
#define MOE_SCHED_STATUS    0x08u
#define MOE_SCHED_CONFIG    0x10u
#define MOE_SCHED_HEAD0     0x18u
#define MOE_SCHED_HEAD1     0x20u
#define MOE_SCHED_HEAD2     0x28u
#define MOE_SCHED_HEAD3     0x30u
#define MOE_SCHED_REMOVE    0x38u
#define MOE_SCHED_PLAN0     0x40u
#define MOE_SCHED_PLAN1     0x48u
#define MOE_SCHED_PLAN_META 0x50u
#define MOE_SCHED_MAKESPAN  0x58u

#define MOE_SCHED_CTRL_INIT         (1ull << 0)
#define MOE_SCHED_CTRL_START        (1ull << 1)
#define MOE_SCHED_CTRL_REMOVE_READY (1ull << 2)
#define MOE_SCHED_CTRL_PLAN_POP     (1ull << 3)

#define MOE_SCHED_STATUS_BUSY         (1ull << 0)
#define MOE_SCHED_STATUS_DONE         (1ull << 1)
#define MOE_SCHED_STATUS_REMOVE_VALID (1ull << 2)
#define MOE_SCHED_STATUS_PLAN_VALID   (1ull << 3)
#define MOE_SCHED_STATUS_PLAN_FULL    (1ull << 4)

#define MOE_SCHED_E_MAX      64u
#define MOE_SCHED_EID_RAW_W  6u
#define MOE_SCHED_NTOK_W     9u
#define MOE_SCHED_NR_W       7u

#ifndef MOE_SCHED_TIMEOUT_POLLS
#define MOE_SCHED_TIMEOUT_POLLS 1000000u
#endif

typedef struct {
    uint8_t valid;
    uint8_t rem_index;
    uint8_t eid;
    uint16_t ntok;
    uint8_t input_order;
    uint16_t best_conc;
} moe_sched_head_t;

typedef struct {
    uint8_t cluster;
    uint8_t eid;
    uint16_t ntok;
    uint16_t tok_start;
    uint8_t s1;
    uint8_t s3;
    uint8_t skip_s1;
    uint8_t skip_s3;
    uint8_t has_s2pf;
    uint8_t allow_s4pf;
} moe_sched_task_t;

typedef struct {
    uint16_t eid;
    uint16_t ntokens;
    uint16_t best_conc;
    uint8_t active;
} moe_sched_rem_item_t;

static inline uintptr_t moe_sched_base(void)
{
    return (uintptr_t)chiplet_addr_transform((uint64_t)MOE_SCHED_LOCAL_BASE);
}

static inline void moe_sched_write64(uint32_t off, uint64_t value)
{
    writed(value, moe_sched_base() + (uintptr_t)off);
    asm volatile("fence iorw, iorw" ::: "memory");
}

static inline uint64_t moe_sched_read64(uint32_t off)
{
    asm volatile("fence iorw, iorw" ::: "memory");
    return readd(moe_sched_base() + (uintptr_t)off);
}

static inline uint64_t moe_sched_pack_config(uint8_t cache_eid_c2,
                                             uint8_t cache_eid_c3,
                                             uint8_t active_count,
                                             uint16_t total_conc)
{
    return ((uint64_t)cache_eid_c2) |
           ((uint64_t)cache_eid_c3 << 8) |
           ((uint64_t)(active_count & 0x7fu) << 16) |
           ((uint64_t)total_conc << 32);
}

static inline uint64_t moe_sched_pack_head(moe_sched_head_t h)
{
    uint64_t word = 0;

    word |= ((uint64_t)(h.best_conc & 0xffffu) << 0);
    word |= ((uint64_t)(h.input_order & 0x7fu) << 16);
    word |= ((uint64_t)(h.ntok & 0x1ffu) << 23);
    word |= ((uint64_t)(h.eid & 0x3fu) << 32);
    word |= ((uint64_t)(h.rem_index & 0x7fu) << 38);
    word |= ((uint64_t)(h.valid & 0x1u) << 45);
    return word;
}

static inline moe_sched_task_t moe_sched_unpack_task(uint64_t word)
{
    moe_sched_task_t task;
    memset(&task, 0, sizeof(task));
    task.has_s2pf   = (uint8_t)((word >> 0) & 0x1u);
    task.skip_s3    = (uint8_t)((word >> 1) & 0x1u);
    task.skip_s1    = (uint8_t)((word >> 2) & 0x1u);
    task.s3         = (uint8_t)((word >> 3) & 0x3u);
    task.s1         = (uint8_t)((word >> 5) & 0x3u);
    task.tok_start  = (uint16_t)((word >> 7) & 0x1ffu);
    task.ntok       = (uint16_t)((word >> 16) & 0x1ffu);
    task.eid        = (uint8_t)((word >> 25) & 0x3fu);
    task.cluster    = (uint8_t)((word >> 31) & 0x1u);
    task.allow_s4pf = (uint8_t)((word >> 32) & 0x1u);
    return task;
}

static inline uint16_t moe_sched_best_conc(uint16_t ntok)
{
    return (uint16_t)(((ntok + 3u) >> 2) * 6u);
}

static inline int moe_sched_wait_done(uint64_t *status_out)
{
    for (uint32_t i = 0; i < MOE_SCHED_TIMEOUT_POLLS; i++) {
        uint64_t status = moe_sched_read64(MOE_SCHED_STATUS);
        if ((status & MOE_SCHED_STATUS_DONE) != 0u) {
            if (status_out != NULL) {
                *status_out = status;
            }
            return 0;
        }
    }
    if (status_out != NULL) {
        *status_out = moe_sched_read64(MOE_SCHED_STATUS);
    }
    return -1;
}

static inline void moe_sched_clear_heads(void)
{
    moe_sched_write64(MOE_SCHED_HEAD0, 0);
    moe_sched_write64(MOE_SCHED_HEAD1, 0);
    moe_sched_write64(MOE_SCHED_HEAD2, 0);
    moe_sched_write64(MOE_SCHED_HEAD3, 0);
}

static inline uint8_t moe_sched_cache_to_rtl(int16_t cache_eid)
{
    return (cache_eid < 0) ? 0x80u : (uint8_t)cache_eid;
}

static inline int moe_sched_make_sorted_rem(const moe_request_t *req,
                                            moe_sched_rem_item_t *rem,
                                            uint16_t *total_conc_out)
{
    uint16_t total_conc = 0;

    if (req == NULL || rem == NULL || total_conc_out == NULL) {
        return -1;
    }

    for (uint16_t i = 0; i < req->n_experts; i++) {
        if (req->experts[i].ntokens == 0u ||
            req->experts[i].expert_id >= MOE_SCHED_E_MAX) {
            return -2;
        }
        rem[i].eid = req->experts[i].expert_id;
        rem[i].ntokens = req->experts[i].ntokens;
        rem[i].best_conc = moe_sched_best_conc(req->experts[i].ntokens);
        rem[i].active = 1u;
    }

    for (uint16_t i = 1; i < req->n_experts; i++) {
        moe_sched_rem_item_t key = rem[i];
        int j = (int)i - 1;
        while (j >= 0 && rem[j].ntokens < key.ntokens) {
            rem[j + 1] = rem[j];
            j--;
        }
        rem[j + 1] = key;
    }

    for (uint16_t i = 0; i < req->n_experts; i++) {
        total_conc = (uint16_t)(total_conc + rem[i].best_conc);
    }
    *total_conc_out = total_conc;
    return 0;
}

static inline void moe_sched_write_head_slot(uint8_t slot, moe_sched_head_t head)
{
    uint64_t word = moe_sched_pack_head(head);

    switch (slot) {
    case 0: moe_sched_write64(MOE_SCHED_HEAD0, word); break;
    case 1: moe_sched_write64(MOE_SCHED_HEAD1, word); break;
    case 2: moe_sched_write64(MOE_SCHED_HEAD2, word); break;
    default: moe_sched_write64(MOE_SCHED_HEAD3, word); break;
    }
}

static inline void moe_sched_write_round_context(const moe_request_t *req,
                                                 const moe_sched_rem_item_t *rem,
                                                 uint16_t active_count,
                                                 uint16_t total_conc)
{
    uint8_t h = 0;

    moe_sched_write64(
        MOE_SCHED_CONFIG,
        moe_sched_pack_config(moe_sched_cache_to_rtl(req->cache_eid_c2),
                              moe_sched_cache_to_rtl(req->cache_eid_c3),
                              (uint8_t)active_count,
                              total_conc));

    moe_sched_clear_heads();

    for (uint16_t i = 0; i < req->n_experts && h < 4u; i++) {
        if (rem[i].active == 0u) {
            continue;
        }

        moe_sched_head_t head = {
            .valid = 1u,
            .rem_index = (uint8_t)i,
            .eid = (uint8_t)rem[i].eid,
            .ntok = rem[i].ntokens,
            .input_order = (uint8_t)i,
            .best_conc = rem[i].best_conc,
        };
        moe_sched_write_head_slot(h, head);
        h++;
    }
}

static inline moe_hw_plan_entry_t moe_sched_entry_from_task(moe_sched_task_t task)
{
    moe_hw_plan_entry_t entry;
    memset(&entry, 0, sizeof(entry));
    entry.valid = 1u;
    entry.desc.cluster = task.cluster ? MOE_CLUSTER_C3 : MOE_CLUSTER_C2;
    entry.desc.expert_id = task.eid;
    entry.desc.token_start_rank = task.tok_start;
    entry.desc.ntokens = task.ntok;
    entry.desc.shape_s1 = (moe_shape_t)task.s1;
    entry.desc.shape_s3 = (moe_shape_t)task.s3;
    entry.desc.skip_s1 = task.skip_s1;
    entry.desc.skip_s3 = task.skip_s3;
    entry.desc.has_s2pf = task.has_s2pf;
    entry.allow_s4pf = task.allow_s4pf;
    return entry;
}

static inline int moe_sched_remove_one(moe_sched_rem_item_t *rem,
                                       uint16_t n_experts,
                                       uint16_t idx,
                                       uint16_t *active_count,
                                       uint16_t *total_conc)
{
    if (rem == NULL || active_count == NULL || total_conc == NULL ||
        idx >= n_experts || rem[idx].active == 0u || *active_count == 0u) {
        return -1;
    }
    rem[idx].active = 0u;
    *active_count = (uint16_t)(*active_count - 1u);
    *total_conc = (uint16_t)(*total_conc - rem[idx].best_conc);
    return 0;
}

static inline int moe_sched_drain_plan_head(moe_hw_plan_entry_t *plan,
                                            uint16_t *out_n,
                                            uint32_t round,
                                            uint32_t verbose)
{
    uint64_t meta = moe_sched_read64(MOE_SCHED_PLAN_META);
    uint32_t plan_count = (uint32_t)(meta & 0x3u);
    uint32_t slot_valid = (uint32_t)((meta >> 8) & 0x3u);

    if (plan_count == 0u || plan_count > 2u || slot_valid == 0u) {
        if (verbose != 0u) {
            printf_safe("[SCHED_DRV] bad plan metadata round=%u meta=0x%lx\r\n",
                        round, meta);
        }
        return -1;
    }
    if (*out_n + (uint16_t)plan_count > MOE_MAX_TASKS) {
        return -2;
    }

    uint64_t plan_word0 = moe_sched_read64(MOE_SCHED_PLAN0);
    plan[(*out_n)++] = moe_sched_entry_from_task(moe_sched_unpack_task(plan_word0));

    if (plan_count == 2u) {
        uint64_t plan_word1 = moe_sched_read64(MOE_SCHED_PLAN1);
        plan[(*out_n)++] = moe_sched_entry_from_task(moe_sched_unpack_task(plan_word1));
    }

    moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_PLAN_POP);
    return 0;
}

static inline int moe_sched_hw_make_plan_mmio(const moe_request_t *req,
                                              moe_hw_plan_entry_t *plan,
                                              uint16_t *n_plan,
                                              uint32_t verbose)
{
    moe_sched_rem_item_t rem[MOE_MAX_EXPERTS];
    uint16_t out_n = 0;
    uint32_t round = 0;
    uint16_t active_count = 0;
    uint16_t total_conc = 0;

    if (req == NULL || plan == NULL || n_plan == NULL ||
        req->n_experts == 0u || req->n_experts > MOE_MAX_EXPERTS ||
        req->n_experts > MOE_SCHED_E_MAX) {
        return -1;
    }

    memset(rem, 0, sizeof(rem));
    if (moe_sched_make_sorted_rem(req, rem, &total_conc) != 0) {
        return -2;
    }
    active_count = req->n_experts;

    moe_sched_write64(
        MOE_SCHED_CONFIG,
        moe_sched_pack_config(moe_sched_cache_to_rtl(req->cache_eid_c2),
                              moe_sched_cache_to_rtl(req->cache_eid_c3),
                              0u, 0u));
    moe_sched_clear_heads();
    moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_INIT);

    moe_sched_write_round_context(req, rem, active_count, total_conc);
    moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_START);

    while (1) {
        uint64_t status;
        uint64_t remove;
        uint32_t remove_count;
        uint32_t remove_idx0;
        uint32_t remove_idx1;

        if (moe_sched_wait_done(&status) != 0) {
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] timeout round=%u status=0x%lx\r\n",
                            round, status);
            }
            return -3;
        }

        if ((status & MOE_SCHED_STATUS_REMOVE_VALID) == 0u ||
            (status & MOE_SCHED_STATUS_PLAN_VALID) == 0u) {
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] invalid status round=%u status=0x%lx\r\n",
                            round, status);
            }
            return -4;
        }

        remove = moe_sched_read64(MOE_SCHED_REMOVE);
        remove_count = (uint32_t)(remove & 0x3u);
        remove_idx0 = (uint32_t)((remove >> 2) & 0x7fu);
        remove_idx1 = (uint32_t)((remove >> (2 + MOE_SCHED_NR_W)) & 0x7fu);

        if (remove_count == 0u || remove_count > 2u ||
            remove_count > active_count) {
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] bad remove metadata round=%u remove=0x%lx active=%u\r\n",
                            round, remove, active_count);
            }
            return -5;
        }

        if (moe_sched_remove_one(rem, req->n_experts, (uint16_t)remove_idx0,
                                 &active_count, &total_conc) != 0) {
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] bad remove_idx0 round=%u idx=%u\r\n",
                            round, remove_idx0);
            }
            return -7;
        }
        if (remove_count == 2u &&
            moe_sched_remove_one(rem, req->n_experts, (uint16_t)remove_idx1,
                                 &active_count, &total_conc) != 0) {
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] bad remove_idx1 round=%u idx=%u\r\n",
                            round, remove_idx1);
            }
            return -8;
        }

        moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_REMOVE_READY);

        if (active_count != 0u) {
            moe_sched_write_round_context(req, rem, active_count, total_conc);
            moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_START);
        }

        if (moe_sched_drain_plan_head(plan, &out_n, round, verbose) != 0) {
            return -9;
        }

        if (verbose != 0u) {
            printf_safe("[SCHED_DRV] round=%u status=0x%lx remove=0x%lx out_n=%u active=%u\r\n",
                        round, status, remove, out_n, active_count);
        }

        round++;
        if (active_count == 0u) {
            break;
        }
    }

    *n_plan = out_n;
    return 0;
}

static inline uint64_t moe_sched_plan_signature(const moe_hw_plan_entry_t *entry)
{
    return ((uint64_t)entry->valid << 0) |
           ((uint64_t)entry->desc.cluster << 1) |
           ((uint64_t)entry->desc.expert_id << 2) |
           ((uint64_t)entry->desc.token_start_rank << 18) |
           ((uint64_t)entry->desc.ntokens << 34) |
           ((uint64_t)entry->desc.shape_s1 << 50) |
           ((uint64_t)entry->desc.shape_s3 << 52) |
           ((uint64_t)entry->desc.skip_s1 << 54) |
           ((uint64_t)entry->desc.skip_s3 << 55) |
           ((uint64_t)entry->desc.has_s2pf << 56) |
           ((uint64_t)entry->allow_s4pf << 57);
}

static inline uint32_t moe_sched_hw_plan_matches(const moe_hw_plan_entry_t *got,
                                                 uint16_t got_n,
                                                 const moe_hw_plan_entry_t *exp,
                                                 uint16_t exp_n)
{
    if (got == NULL || exp == NULL || got_n != exp_n) {
        return 0u;
    }

    for (uint16_t i = 0; i < got_n; i++) {
        if (moe_sched_plan_signature(&got[i]) != moe_sched_plan_signature(&exp[i])) {
            return 0u;
        }
    }
    return 1u;
}

static inline uint32_t moe_sched_compare_plan(const moe_hw_plan_entry_t *got,
                                              uint16_t got_n,
                                              const moe_hw_plan_entry_t *exp,
                                              uint16_t exp_n,
                                              uint32_t case_id)
{
    uint32_t errors = 0;
    if (got_n != exp_n) {
        printf_safe("[SCHED_CMP] case=%u n_plan mismatch got=%u exp=%u\r\n",
                    case_id, got_n, exp_n);
        errors++;
    }

    uint16_t n = (got_n < exp_n) ? got_n : exp_n;
    for (uint16_t i = 0; i < n; i++) {
        uint64_t got_sig = moe_sched_plan_signature(&got[i]);
        uint64_t exp_sig = moe_sched_plan_signature(&exp[i]);

        if (got_sig != exp_sig) {
            printf_safe("[SCHED_CMP] case=%u entry=%u mismatch\r\n", case_id, i);
            printf_safe("  sig: got=0x%lx exp=0x%lx\r\n", got_sig, exp_sig);
            printf_safe("  got: v=%u c=%u eid=%u ts=%u n=%u s1=%u s3=%u sk1=%u sk3=%u s2pf=%u s4pf=%u\r\n",
                        got[i].valid, got[i].desc.cluster, got[i].desc.expert_id,
                        got[i].desc.token_start_rank, got[i].desc.ntokens,
                        got[i].desc.shape_s1, got[i].desc.shape_s3,
                        got[i].desc.skip_s1, got[i].desc.skip_s3,
                        got[i].desc.has_s2pf, got[i].allow_s4pf);
            printf_safe("  exp: v=%u c=%u eid=%u ts=%u n=%u s1=%u s3=%u sk1=%u sk3=%u s2pf=%u s4pf=%u\r\n",
                        exp[i].valid, exp[i].desc.cluster, exp[i].desc.expert_id,
                        exp[i].desc.token_start_rank, exp[i].desc.ntokens,
                        exp[i].desc.shape_s1, exp[i].desc.shape_s3,
                        exp[i].desc.skip_s1, exp[i].desc.skip_s3,
                        exp[i].desc.has_s2pf, exp[i].allow_s4pf);
            errors++;
        }
    }
    return errors;
}
