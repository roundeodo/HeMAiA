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
#define MOE_SCHED_HEAD0     0x18u
#define MOE_SCHED_HEAD1     0x20u
#define MOE_SCHED_HEAD2     0x28u
#define MOE_SCHED_HEAD3     0x30u
#define MOE_SCHED_REMOVE    0x38u
#define MOE_SCHED_PLAN0     0x40u
#define MOE_SCHED_PLAN1     0x48u
#define MOE_SCHED_PLAN_META 0x50u
#define MOE_SCHED_MAKESPAN  0x58u
#define MOE_SCHED_HEAD_PUSH 0x60u
#define MOE_SCHED_ROUND_COMMIT 0x68u
#define MOE_SCHED_HEAD_PUSH0   0x70u
#define MOE_SCHED_HEAD_PUSH1   0x78u
#define MOE_SCHED_PLAN_FIFO_STATUS 0x80u
#define MOE_SCHED_PLAN_FIFO_META   0x88u
#define MOE_SCHED_PLAN_FIFO_DATA0  0x90u
#define MOE_SCHED_PLAN_FIFO_DATA1  0x98u
#define MOE_SCHED_PLAN_FIFO_POP    0xa0u

#define MOE_SCHED_CTRL_INIT         (1ull << 0)
#define MOE_SCHED_CTRL_START        (1ull << 1)
#define MOE_SCHED_CTRL_REMOVE_READY (1ull << 2)
#define MOE_SCHED_CTRL_PLAN_POP     (1ull << 3)

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

typedef int (*moe_sched_plan_entry_cb_t)(const moe_hw_plan_entry_t *entry,
                                         void *ctx);

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

static inline uint64_t moe_sched_pack_round_commit(uint32_t plan_pop,
                                                   uint32_t remove_ready,
                                                   uint32_t start_next,
                                                   uint32_t push_count,
                                                   uint32_t consumed_count)
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
    word |= ((uint64_t)(consumed_count & 0xffu) << 8);
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

static inline int moe_sched_remove_active_eid(moe_sched_rem_item_t *rem,
                                              uint16_t n_experts,
                                              uint8_t eid,
                                              uint16_t *active_count,
                                              uint16_t *total_conc)
{
    if (rem == NULL || active_count == NULL || total_conc == NULL ||
        *active_count == 0u) {
        return -1;
    }

    for (uint16_t i = 0; i < n_experts; i++) {
        if (rem[i].eid == eid) {
            if (rem[i].active == 0u) {
                return -2;
            }
            rem[i].active = 0u;
            *active_count = (uint16_t)(*active_count - 1u);
            *total_conc = (uint16_t)(*total_conc - rem[i].best_conc);
            return 0;
        }
    }
    return -3;
}

static inline void moe_sched_write_head_slot_relaxed(uint8_t slot, moe_sched_head_t head)
{
    uint64_t word = moe_sched_pack_head(head);

    switch (slot) {
    case 0: moe_sched_write64_relaxed(MOE_SCHED_HEAD0, word); break;
    case 1: moe_sched_write64_relaxed(MOE_SCHED_HEAD1, word); break;
    case 2: moe_sched_write64_relaxed(MOE_SCHED_HEAD2, word); break;
    default: moe_sched_write64_relaxed(MOE_SCHED_HEAD3, word); break;
    }
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

static inline int moe_sched_emit_plan_entry(moe_hw_plan_entry_t *plan,
                                            uint16_t *out_n,
                                            moe_sched_plan_entry_cb_t cb,
                                            void *cb_ctx,
                                            uint8_t removed_eids[2],
                                            uint32_t *removed_count,
                                            uint64_t plan_word)
{
    moe_hw_plan_entry_t entry;

    entry.valid = 1u;
    entry.desc.cluster =
        ((plan_word >> 31) & 0x1u) ? MOE_CLUSTER_C3 : MOE_CLUSTER_C2;
    entry.desc.expert_id = (uint16_t)((plan_word >> 25) & 0x3fu);
    entry.desc.token_start_rank = (uint16_t)((plan_word >> 7) & 0x1ffu);
    entry.desc.ntokens = (uint16_t)((plan_word >> 16) & 0x1ffu);
    entry.desc.shape_s1 = (moe_shape_t)((plan_word >> 5) & 0x3u);
    entry.desc.shape_s3 = (moe_shape_t)((plan_word >> 3) & 0x3u);
    entry.desc.skip_s1 = (uint8_t)((plan_word >> 2) & 0x1u);
    entry.desc.skip_s3 = (uint8_t)((plan_word >> 1) & 0x1u);
    entry.desc.has_s2pf = (uint8_t)((plan_word >> 0) & 0x1u);
    entry.allow_s4pf = (uint8_t)((plan_word >> 32) & 0x1u);

    if (out_n == NULL || *out_n >= MOE_MAX_TASKS) {
        return -1;
    }
    if (plan != NULL) {
        plan[*out_n] = entry;
    }
    (*out_n)++;
    if (cb != NULL && cb(&entry, cb_ctx) != 0) {
        return -2;
    }
    if (moe_sched_note_unique_eid((uint8_t)entry.desc.expert_id,
                                  removed_eids, removed_count) != 0) {
        return -3;
    }
    return 0;
}

static inline int moe_sched_drain_plan_words_cb(moe_hw_plan_entry_t *plan,
                                                uint16_t *out_n,
                                                moe_sched_plan_entry_cb_t cb,
                                                void *cb_ctx,
                                                uint8_t removed_eids[2],
                                                uint32_t *removed_count,
                                                uint32_t plan_count,
                                                uint32_t slot_valid,
                                                uint32_t round,
                                                uint32_t verbose);

static inline int moe_sched_drain_plan_head_cb(moe_hw_plan_entry_t *plan,
                                               uint16_t *out_n,
                                               moe_sched_plan_entry_cb_t cb,
                                               void *cb_ctx,
                                               uint8_t removed_eids[2],
                                               uint32_t *removed_count,
                                               uint32_t round,
                                               uint32_t verbose)
{
    uint64_t meta = moe_sched_read64_relaxed(MOE_SCHED_PLAN_FIFO_META);
    uint32_t plan_count = (uint32_t)(meta & 0x3u);
    uint32_t slot_valid = (uint32_t)((meta >> 8) & 0x3u);

    return moe_sched_drain_plan_words_cb(plan, out_n, cb, cb_ctx,
                                         removed_eids, removed_count,
                                         plan_count, slot_valid,
                                         round, verbose);
}

static inline int moe_sched_drain_plan_words_cb(moe_hw_plan_entry_t *plan,
                                                uint16_t *out_n,
                                                moe_sched_plan_entry_cb_t cb,
                                                void *cb_ctx,
                                                uint8_t removed_eids[2],
                                                uint32_t *removed_count,
                                                uint32_t plan_count,
                                                uint32_t slot_valid,
                                                uint32_t round,
                                                uint32_t verbose)
{
    if (plan_count == 0u || plan_count > 2u || slot_valid == 0u) {
        if (verbose != 0u) {
            printf_safe("[SCHED_DRV] bad plan metadata round=%u count=%u slot_valid=0x%x\r\n",
                        round, plan_count, slot_valid);
        }
        return -1;
    }
    if (*out_n + (uint16_t)plan_count > MOE_MAX_TASKS) {
        return -2;
    }

    uint64_t plan_word0 = moe_sched_read64_relaxed(MOE_SCHED_PLAN_FIFO_DATA0);
    if (moe_sched_emit_plan_entry(plan, out_n, cb, cb_ctx, removed_eids,
                                  removed_count, plan_word0) != 0) {
        return -3;
    }

    if (plan_count == 2u) {
        uint64_t plan_word1 = moe_sched_read64_relaxed(MOE_SCHED_PLAN_FIFO_DATA1);
        if (moe_sched_emit_plan_entry(plan, out_n, cb, cb_ctx, removed_eids,
                                      removed_count, plan_word1) != 0) {
            return -4;
        }
    }

    return 0;
}

static inline int moe_sched_drain_plan_head(moe_hw_plan_entry_t *plan,
                                            uint16_t *out_n,
                                            uint32_t round,
                                            uint32_t verbose)
{
    int rc = moe_sched_drain_plan_head_cb(plan, out_n, NULL, NULL, NULL, NULL,
                                          round, verbose);
    if (rc == 0) {
        moe_sched_fence();
        moe_sched_write64(MOE_SCHED_PLAN_FIFO_POP, 1u);
    }
    return rc;
}

static inline int moe_sched_hw_make_plan_mmio_cb(const moe_request_t *req,
                                                 moe_hw_plan_entry_t *plan,
                                                 uint16_t *n_plan,
                                                 moe_sched_plan_entry_cb_t cb,
                                                 void *cb_ctx,
                                                 uint32_t verbose)
{
    moe_sched_rem_item_t rem[MOE_MAX_EXPERTS];
    uint16_t out_n = 0;
    uint32_t round = 0;
    uint16_t active_count = 0;
    uint16_t total_conc = 0;
    uint16_t next_rem_pos = 0;

    if (req == NULL || n_plan == NULL ||
        req->n_experts == 0u || req->n_experts > MOE_MAX_EXPERTS ||
        req->n_experts > MOE_SCHED_E_MAX) {
        return -1;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_SORT_START);
    int sort_rc = moe_sched_make_sorted_rem(req, rem, &total_conc);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_SORT_END);
    if (sort_rc != 0) {
        return -2;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_INIT_WRITE_START);
    active_count = req->n_experts;
    next_rem_pos = (req->n_experts < 4u) ? req->n_experts : 4u;

    moe_sched_write64_relaxed(
        MOE_SCHED_CONFIG,
        moe_sched_pack_config(moe_sched_cache_to_rtl(req->cache_eid_c2),
                              moe_sched_cache_to_rtl(req->cache_eid_c3),
                              (uint8_t)active_count,
                              total_conc));
    for (uint8_t s = 0; s < 4u; s++) {
        moe_sched_head_t head = {0};
        if (s < req->n_experts) {
            head = moe_sched_head_from_rem(rem, s);
        }
        moe_sched_write_head_slot_relaxed(s, head);
    }
    moe_sched_fence();
    moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_INIT);
    moe_sched_write64(MOE_SCHED_CTRL, MOE_SCHED_CTRL_START);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_INIT_WRITE_END);

    while (1) {
        uint64_t status;
        uint32_t status_remove_count;
        uint32_t status_plan_count;
        uint32_t status_slot_valid;
        uint8_t removed_eids[2] = {0};
        uint32_t removed_count = 0;
        uint32_t push_count = 0;

        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_WAIT_START);
        int wait_rc = moe_sched_wait_done(&status);
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_WAIT_END);
        if (wait_rc != 0) {
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] timeout round=%u status=0x%lx\r\n",
                            round, status);
            }
            return -3;
        }

        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_READ_REMOVE_START);
        if ((status & MOE_SCHED_STATUS_REMOVE_VALID) == 0u ||
            (status & MOE_SCHED_STATUS_PLAN_VALID) == 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_READ_REMOVE_END);
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] invalid status round=%u status=0x%lx\r\n",
                            round, status);
            }
            return -4;
        }

        status_remove_count = (uint32_t)((status >> 8) & 0x3u);
        status_plan_count = (uint32_t)((status >> 16) & 0x3u);
        status_slot_valid = (uint32_t)((status >> 24) & 0x3u);
        if (status_remove_count == 0u || status_remove_count > 2u ||
            status_remove_count > active_count) {
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_READ_REMOVE_END);
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] bad remove count round=%u status=0x%lx active=%u\r\n",
                            round, status, active_count);
            }
            return -5;
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_READ_REMOVE_END);

        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_DRAIN_PLAN_START);
        int drain_rc = moe_sched_drain_plan_words_cb(plan, &out_n, cb, cb_ctx,
                                                     removed_eids,
                                                     &removed_count,
                                                     status_plan_count,
                                                     status_slot_valid,
                                                     round, verbose);
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_DRAIN_PLAN_END);
        if (drain_rc != 0) {
            return -9;
        }

        if (removed_count != status_remove_count) {
            if (verbose != 0u) {
                printf_safe("[SCHED_DRV] remove/plan mismatch round=%u status_count=%u plan_unique=%u\r\n",
                            round, status_remove_count, removed_count);
            }
            return -10;
        }

        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_HEAD_UPDATE_START);
        for (uint32_t r = 0; r < removed_count; r++) {
            int remove_rc = moe_sched_remove_active_eid(rem, req->n_experts,
                                                        removed_eids[r],
                                                        &active_count,
                                                        &total_conc);
            if (remove_rc != 0) {
                BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_HEAD_UPDATE_END);
                if (verbose != 0u) {
                    printf_safe("[SCHED_DRV] remove eid failed round=%u eid=%u rc=%d\r\n",
                                round, removed_eids[r], remove_rc);
                }
                return -7;
            }
        }

        for (uint32_t p = 0; p < removed_count && next_rem_pos < req->n_experts; p++) {
            while (next_rem_pos < req->n_experts &&
                   rem[next_rem_pos].active == 0u) {
                next_rem_pos++;
            }
            if (next_rem_pos >= req->n_experts) {
                break;
            }
            moe_sched_head_t new_head = moe_sched_head_from_rem(rem, next_rem_pos);
            uint32_t head_reg = (push_count == 0u) ?
                MOE_SCHED_HEAD_PUSH0 : MOE_SCHED_HEAD_PUSH1;
            moe_sched_write64_relaxed(head_reg, moe_sched_pack_head(new_head));
            next_rem_pos++;
            push_count++;
        }

        if (active_count != 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_RESTART_START);
        }
        moe_sched_fence();
        moe_sched_write64(
            MOE_SCHED_ROUND_COMMIT,
            moe_sched_pack_round_commit(1u, 1u, active_count != 0u,
                                        push_count, removed_count));
        if (active_count != 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_RESTART_END);
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_HOST_MOE_HW_HEAD_UPDATE_END);

        if (verbose != 0u) {
            printf_safe("[SCHED_DRV] round=%u status=0x%lx removed=%u out_n=%u active=%u\r\n",
                        round, status, removed_count, out_n, active_count);
        }

        round++;
        if (active_count == 0u) {
            break;
        }
    }

    *n_plan = out_n;
    return 0;
}

static inline int moe_sched_hw_make_plan_mmio(const moe_request_t *req,
                                              moe_hw_plan_entry_t *plan,
                                              uint16_t *n_plan,
                                              uint32_t verbose)
{
    return moe_sched_hw_make_plan_mmio_cb(req, plan, n_plan, NULL, NULL,
                                          verbose);
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
