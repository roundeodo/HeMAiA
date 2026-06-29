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
#include "perf_tracing.h"
#include "uart.h"
#include "moe_scheduler.h"

#ifndef MOE_SCHED_LOCAL_BASE
#define MOE_SCHED_LOCAL_BASE 0x05010000ull
#endif

#define MOE_SCHED_CTRL      0x00u
#define MOE_SCHED_STATUS    0x08u
#define MOE_SCHED_CONFIG    0x10u
#define MOE_SCHED_ROUND_COMMIT 0x68u
#define MOE_SCHED_PLAN_FIFO_STATUS 0x80u
#define MOE_SCHED_PLAN_FIFO_DATA0  0x90u
#define MOE_SCHED_PLAN_FIFO_DATA1  0x98u
#define MOE_SCHED_HEAD_PAIR0       0xa8u
#define MOE_SCHED_HEAD_PAIR1       0xb0u
#define MOE_SCHED_HEAD_PUSH_PAIR   0xb8u

#define MOE_SCHED_CTRL_INIT         (1ull << 0)
#define MOE_SCHED_CTRL_START        (1ull << 1)

#define MOE_SCHED_COMMIT_PLAN_POP     (1ull << 0)
#define MOE_SCHED_COMMIT_REMOVE_READY (1ull << 1)
#define MOE_SCHED_COMMIT_START_NEXT   (1ull << 2)

#define MOE_SCHED_STATUS_BUSY         (1ull << 0)
#define MOE_SCHED_STATUS_DONE         (1ull << 1)
#define MOE_SCHED_STATUS_REMOVE_VALID (1ull << 2)
#define MOE_SCHED_STATUS_PLAN_VALID   (1ull << 3)
#define MOE_SCHED_STATUS_PLAN_FULL    (1ull << 4)

#define MOE_SCHED_E_MAX      64u
#define MOE_SCHED_EID_RAW_W  6u
#define MOE_SCHED_NTOK_W     9u
#define MOE_SCHED_NR_W       7u
#define MOE_SCHED_T_W        16u
#define MOE_SCHED_EID_POS_INVALID 0xffu

#define MOE_SCHED_PLAN_HAS_S2PF_LSB       0u
#define MOE_SCHED_PLAN_SKIP_S3_LSB        1u
#define MOE_SCHED_PLAN_SKIP_S1_LSB        2u
#define MOE_SCHED_PLAN_SHAPE_S3_LSB       3u
#define MOE_SCHED_PLAN_SHAPE_S1_LSB       5u
#define MOE_SCHED_PLAN_TOKEN_START_LSB    7u
#define MOE_SCHED_PLAN_NTOK_LSB           16u
#define MOE_SCHED_PLAN_EID_LSB            25u
#define MOE_SCHED_PLAN_CLUSTER_LSB        31u
#define MOE_SCHED_PLAN_ALLOW_S4PF_LSB     32u

#define MOE_SCHED_PLAN_NTOK_MASK          0x1ffull
#define MOE_SCHED_PLAN_EID_MASK           0x3full
#define MOE_SCHED_PLAN_SHAPE_MASK         0x3ull

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
    uint16_t eid;
    uint16_t ntokens;
    uint16_t best_conc;
    uint8_t active;
} moe_sched_rem_item_t;

static inline uintptr_t moe_sched_base(void)
{
    return (uintptr_t)chiplet_addr_transform((uint64_t)MOE_SCHED_LOCAL_BASE);
}

static inline void moe_sched_fence(void)
{
    asm volatile("fence iorw, iorw" ::: "memory");
}

static inline void moe_sched_write64_relaxed(uint32_t off, uint64_t value)
{
    writed(value, moe_sched_base() + (uintptr_t)off);
}

static inline uint64_t moe_sched_read64_relaxed(uint32_t off)
{
    return readd(moe_sched_base() + (uintptr_t)off);
}

static inline void moe_sched_write64(uint32_t off, uint64_t value)
{
    moe_sched_write64_relaxed(off, value);
    moe_sched_fence();
}

static inline uint64_t moe_sched_read64(uint32_t off)
{
    moe_sched_fence();
    return moe_sched_read64_relaxed(off);
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

static inline uint32_t moe_sched_pack_head32(moe_sched_head_t h)
{
    uint32_t word = 0;

    word |= ((uint32_t)(h.best_conc & 0xffffu) << 0);
    word |= ((uint32_t)(h.ntok & 0x1ffu) << MOE_SCHED_T_W);
    word |= ((uint32_t)(h.eid & 0x3fu) << (MOE_SCHED_T_W + MOE_SCHED_NTOK_W));
    word |= ((uint32_t)(h.valid & 0x1u) <<
             (MOE_SCHED_T_W + MOE_SCHED_NTOK_W + MOE_SCHED_EID_RAW_W));
    return word;
}

static inline uint64_t moe_sched_pack_head_pair(moe_sched_head_t low,
                                                moe_sched_head_t high)
{
    return ((uint64_t)moe_sched_pack_head32(low)) |
           ((uint64_t)moe_sched_pack_head32(high) << 32);
}

static inline uint64_t moe_sched_pack_round_commit(uint32_t plan_pop,
                                                   uint32_t remove_ready,
                                                   uint32_t start_next,
                                                   uint32_t push_count)
{
    uint64_t word = 0;

    if (plan_pop != 0u) {
        word |= MOE_SCHED_COMMIT_PLAN_POP;
    }
    if (remove_ready != 0u) {
        word |= MOE_SCHED_COMMIT_REMOVE_READY;
    }
    if (start_next != 0u) {
        word |= MOE_SCHED_COMMIT_START_NEXT;
    }
    word |= ((uint64_t)(push_count & 0x3u) << 4);
    return word;
}

static inline uint16_t moe_sched_best_conc(uint16_t ntok)
{
    return (uint16_t)(((ntok + 3u) >> 2) * 6u);
}

static inline int moe_sched_note_unique_eid(uint8_t eid,
                                            uint8_t removed_eids[2],
                                            uint32_t *removed_count)
{
    if (removed_eids == NULL || removed_count == NULL) {
        return 0;
    }
    for (uint32_t i = 0; i < *removed_count; i++) {
        if (removed_eids[i] == eid) {
            return 0;
        }
    }
    if (*removed_count >= 2u) {
        return -1;
    }
    removed_eids[*removed_count] = eid;
    (*removed_count)++;
    return 0;
}

static inline int moe_sched_wait_done(uint64_t *status_out)
{
    for (uint32_t i = 0; i < MOE_SCHED_TIMEOUT_POLLS; i++) {
        uint64_t status = moe_sched_read64_relaxed(MOE_SCHED_STATUS);
        if ((status & MOE_SCHED_STATUS_DONE) != 0u) {
            moe_sched_fence();
            if (status_out != NULL) {
                *status_out = status;
            }
            return 0;
        }
    }
    if (status_out != NULL) {
        moe_sched_fence();
        *status_out = moe_sched_read64_relaxed(MOE_SCHED_STATUS);
    }
    return -1;
}

static inline uint8_t moe_sched_cache_to_rtl(int16_t cache_eid)
{
    return (cache_eid < 0) ? 0x80u : (uint8_t)cache_eid;
}

static inline int moe_sched_make_sorted_rem(const moe_request_t *req,
                                            moe_sched_rem_item_t *rem,
                                            uint8_t eid_to_pos[MOE_SCHED_E_MAX],
                                            uint16_t *total_conc_out)
{
    uint16_t eid[MOE_MAX_EXPERTS];
    uint16_t ntokens[MOE_MAX_EXPERTS];
    uint16_t best_conc[MOE_MAX_EXPERTS];
    uint8_t order[MOE_MAX_EXPERTS];
    uint16_t total_conc = 0;

    if (req == NULL || rem == NULL || eid_to_pos == NULL ||
        total_conc_out == NULL) {
        return -1;
    }

    for (uint16_t i = 0; i < MOE_SCHED_E_MAX; i++) {
        eid_to_pos[i] = MOE_SCHED_EID_POS_INVALID;
    }

    for (uint16_t i = 0; i < req->n_experts; i++) {
        uint16_t cur_eid = req->experts[i].expert_id;
        uint16_t cur_ntokens = req->experts[i].ntokens;
        if (cur_ntokens == 0u ||
            cur_eid >= MOE_SCHED_E_MAX ||
            cur_ntokens >= (1u << MOE_SCHED_NTOK_W) ||
            eid_to_pos[cur_eid] != MOE_SCHED_EID_POS_INVALID) {
            return -2;
        }
        eid[i] = cur_eid;
        ntokens[i] = cur_ntokens;
        best_conc[i] = moe_sched_best_conc(cur_ntokens);
        order[i] = (uint8_t)i;
        total_conc = (uint16_t)(total_conc + best_conc[i]);
        eid_to_pos[cur_eid] = 0u;
    }

    for (uint16_t i = 1; i < req->n_experts; i++) {
        uint8_t key = order[i];
        uint16_t key_ntokens = ntokens[key];
        int j = (int)i - 1;
        while (j >= 0 && ntokens[order[j]] < key_ntokens) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    for (uint16_t i = 0; i < req->n_experts; i++) {
        uint8_t src = order[i];
        rem[i].eid = eid[src];
        rem[i].ntokens = ntokens[src];
        rem[i].best_conc = best_conc[src];
        rem[i].active = 1u;
        eid_to_pos[rem[i].eid] = (uint8_t)i;
    }
    *total_conc_out = total_conc;
    return 0;
}

static inline int moe_sched_remove_active_eid(moe_sched_rem_item_t *rem,
                                              const uint8_t eid_to_pos[MOE_SCHED_E_MAX],
                                              uint8_t eid,
                                              uint16_t *active_count,
                                              uint16_t *total_conc)
{
    if (rem == NULL || eid_to_pos == NULL ||
        active_count == NULL || total_conc == NULL ||
        *active_count == 0u) {
        return -1;
    }
    if (eid >= MOE_SCHED_E_MAX ||
        eid_to_pos[eid] == MOE_SCHED_EID_POS_INVALID) {
        return -3;
    }

    uint8_t pos = eid_to_pos[eid];
    if (rem[pos].eid != eid || rem[pos].active == 0u) {
        return -2;
    }

    rem[pos].active = 0u;
    *active_count = (uint16_t)(*active_count - 1u);
    *total_conc = (uint16_t)(*total_conc - rem[pos].best_conc);
    return 0;
}

static inline void moe_sched_write_head_pair_relaxed(uint8_t pair,
                                                     moe_sched_head_t low,
                                                     moe_sched_head_t high)
{
    uint32_t off = (pair == 0u) ? MOE_SCHED_HEAD_PAIR0 : MOE_SCHED_HEAD_PAIR1;
    moe_sched_write64_relaxed(off, moe_sched_pack_head_pair(low, high));
}

static inline void moe_sched_write_head_push_pair_relaxed(moe_sched_head_t low,
                                                          moe_sched_head_t high)
{
    moe_sched_write64_relaxed(MOE_SCHED_HEAD_PUSH_PAIR,
                              moe_sched_pack_head_pair(low, high));
}

static inline moe_sched_head_t moe_sched_head_from_rem(const moe_sched_rem_item_t *rem,
                                                       uint16_t idx)
{
    moe_sched_head_t head;

    head.valid = 1u;
    head.rem_index = (uint8_t)idx;
    head.eid = (uint8_t)rem[idx].eid;
    head.ntok = rem[idx].ntokens;
    head.input_order = (uint8_t)idx;
    head.best_conc = rem[idx].best_conc;
    return head;
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
