// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: Apache-2.0
//
// CVA6-side MMIO driver for the MoE scheduler register slave.

#pragma once

#include <stdint.h>

#include "chip_id.h"
#include "io.h"

#ifndef MOE_SCHED_LOCAL_BASE
#define MOE_SCHED_LOCAL_BASE 0x05010000ull
#endif

// Top9 + bottom5 window. EVENT_WAIT/TASK_STREAM are blocking reads.
#define MOE_SCHED_CONFIG        0x00u
#define MOE_SCHED_WINDOW0       0x08u
#define MOE_SCHED_WINDOW1       0x10u
#define MOE_SCHED_WINDOW2       0x18u
#define MOE_SCHED_REFILL_QUAD   0x20u
#define MOE_SCHED_EVENT_WAIT    0x28u
#define MOE_SCHED_TASK_STREAM   0x30u
#define MOE_SCHED_AGGREGATE     0x38u
#define MOE_SCHED_WINDOW3_START 0x40u

#define MOE_SCHED_INITIAL_WINDOW_ENTRIES 14u
#define MOE_SCHED_QUAD_ENTRIES            4u
#define MOE_SCHED_MAX_REFILL_ENTRIES       6u
#define MOE_SCHED_MAX_REFILL_BEATS         2u
#define MOE_SCHED_TASK_FIFO_DEPTH          8u

#define MOE_SCHED_EVENT_BATCH_DONE_LSB   0u
#define MOE_SCHED_EVENT_REFILL_REQ_LSB   1u
#define MOE_SCHED_EVENT_REFILL_TOP_COUNT_LSB 2u
#define MOE_SCHED_EVENT_REFILL_BOTTOM_COUNT_LSB 5u
#define MOE_SCHED_EVENT_REFILL_COUNT_MASK 0x7ull
#define MOE_SCHED_EVENT_TASK_COUNT_LSB   8u
#define MOE_SCHED_EVENT_TASK_COUNT_MASK  0xfull

#define MOE_SCHED_CONFIG_ACTIVE_COUNT_LSB 16u
#define MOE_SCHED_CONFIG_ACTIVE_COUNT_MASK 0x7full
#define MOE_SCHED_CONFIG_PARALLEL_WORK_LSB 32u
#define MOE_SCHED_CONFIG_SERIAL_WORK_LSB   48u
#define MOE_SCHED_CONFIG_WORK_MASK         0xffffull

#define MOE_SCHED_CACHE_NONE 0x80u

#define MOE_SCHED_EID_RAW_W  6u
#define MOE_SCHED_NTOK_W     9u

#define MOE_SCHED_DESC_NTOK_MASK  0x1ffu
#define MOE_SCHED_DESC_EID_MASK   0x3fu
#define MOE_SCHED_DESC_EID_LSB    MOE_SCHED_NTOK_W
#define MOE_SCHED_DESC_VALID_LSB  (MOE_SCHED_NTOK_W + MOE_SCHED_EID_RAW_W)

#define MOE_SCHED_TASK_WORD_EID_LSB            0u
#define MOE_SCHED_TASK_WORD_TOKEN_START_LSB    6u
#define MOE_SCHED_TASK_WORD_NTOK_LSB           15u
#define MOE_SCHED_TASK_WORD_HAS_S2PF_LSB       24u
#define MOE_SCHED_TASK_WORD_CTRL_LSB           25u
#define MOE_SCHED_TASK_WORD_M_S2_LSB           38u
#define MOE_SCHED_TASK_WORD_M_S4_LSB           47u
#define MOE_SCHED_TASK_WORD_S4PF_DESC_LSB      56u
#define MOE_SCHED_TASK_WORD_S1_BOTH_LSB        46u
#define MOE_SCHED_TASK_WORD_LATE_BOTH_LSB      55u

#define MOE_SCHED_TASK_WORD_NTOK_MASK          0x1ffull
#define MOE_SCHED_TASK_WORD_EID_MASK           0x3full
#define MOE_SCHED_TASK_WORD_SHAPE_MASK         0x3ull
#define MOE_SCHED_TASK_WORD_LOCAL_SLOT_MASK    0x3full
#define MOE_SCHED_TASK_WORD_M_EXEC_MASK        0xffull
#define MOE_SCHED_TASK_WORD_CTRL_MASK          0x1fffull

#define MOE_SCHED_TASK_CTRL_SKIP_S1_LSB    0u
#define MOE_SCHED_TASK_CTRL_SKIP_S3_LSB    1u
#define MOE_SCHED_TASK_CTRL_SHAPE_S1_LSB   2u
#define MOE_SCHED_TASK_CTRL_SHAPE_S3_LSB   4u
#define MOE_SCHED_TASK_CTRL_CLUSTER_LSB    6u
#define MOE_SCHED_TASK_CTRL_LOCAL_SLOT_LSB 7u

#define MOE_SCHED_S4PF_DESC_OP_LSB          0u
#define MOE_SCHED_S4PF_DESC_OP_MASK         0x3u
#define MOE_SCHED_S4PF_DESC_OP_NONE         0u
#define MOE_SCHED_S4PF_DESC_OP_SINGLE       1u
#define MOE_SCHED_S4PF_DESC_OP_BOTH         2u
#define MOE_SCHED_S4PF_DESC_OP_NO_COPY      3u
#define MOE_SCHED_S4PF_DESC_TARGET_EID_LSB  2u
#define MOE_SCHED_S4PF_DESC_TARGET_EID_MASK 0x3full

static inline uintptr_t moe_sched_base(void)
{
    return (uintptr_t)chiplet_addr_transform((uint64_t)MOE_SCHED_LOCAL_BASE);
}

static inline __attribute__((always_inline)) uint16_t moe_sched_pack_descriptor(
    uint32_t expert_id,
    uint32_t ntokens)
{
    return (uint16_t)(
        (ntokens & MOE_SCHED_DESC_NTOK_MASK) |
        ((expert_id & MOE_SCHED_DESC_EID_MASK) << MOE_SCHED_DESC_EID_LSB) |
        (1u << MOE_SCHED_DESC_VALID_LSB));
}

static inline __attribute__((always_inline)) uint64_t moe_sched_pack_descriptor_quad(
    const uint16_t *descriptors,
    uint32_t start,
    uint32_t count)
{
    uint64_t word = 0u;
    switch (count) {
        case 4u:
            word |= (uint64_t)descriptors[start + 3u] << 48u;
            /* fall through */
        case 3u:
            word |= (uint64_t)descriptors[start + 2u] << 32u;
            /* fall through */
        case 2u:
            word |= (uint64_t)descriptors[start + 1u] << 16u;
            /* fall through */
        case 1u:
            word |= (uint64_t)descriptors[start];
            break;
        default:
            break;
    }
    return word;
}

static inline __attribute__((always_inline)) uint64_t moe_sched_pack_config(
    int16_t cache_eid_c2,
    int16_t cache_eid_c3,
    uint32_t active_count,
    uint32_t total_parallel_work,
    uint32_t total_serial_work)
{
    uint32_t c2 = (cache_eid_c2 < 0) ?
        MOE_SCHED_CACHE_NONE : ((uint32_t)cache_eid_c2 & MOE_SCHED_DESC_EID_MASK);
    uint32_t c3 = (cache_eid_c3 < 0) ?
        MOE_SCHED_CACHE_NONE : ((uint32_t)cache_eid_c3 & MOE_SCHED_DESC_EID_MASK);
    return (uint64_t)c2 |
        ((uint64_t)c3 << 8u) |
        ((uint64_t)(active_count & MOE_SCHED_CONFIG_ACTIVE_COUNT_MASK) <<
         MOE_SCHED_CONFIG_ACTIVE_COUNT_LSB) |
        ((uint64_t)(total_parallel_work & MOE_SCHED_CONFIG_WORK_MASK) <<
         MOE_SCHED_CONFIG_PARALLEL_WORK_LSB) |
        ((uint64_t)(total_serial_work & MOE_SCHED_CONFIG_WORK_MASK) <<
         MOE_SCHED_CONFIG_SERIAL_WORK_LSB);
}

static inline __attribute__((always_inline)) uint64_t moe_sched_pack_aggregate(
    uint32_t token_sum,
    uint32_t odd_count,
    uint32_t block_sum,
    const uint32_t small_block_hist[4])
{
    uint64_t word = (uint64_t)(token_sum & 0x1ffu) |
        ((uint64_t)(odd_count & 0x7fu) << 9u) |
        ((uint64_t)(block_sum & 0x1ffu) << 16u);
    for (uint32_t bucket = 0u; bucket < 4u; bucket++) {
        word |= (uint64_t)(small_block_hist[bucket] & 0x7fu) <<
                (25u + 7u * bucket);
    }
    return word;
}

static inline __attribute__((always_inline)) uint32_t
moe_sched_event_batch_done(uint64_t event)
{
    return (uint32_t)((event >> MOE_SCHED_EVENT_BATCH_DONE_LSB) & 1u);
}

static inline __attribute__((always_inline)) uint32_t
moe_sched_event_refill_requested(uint64_t event)
{
    return (uint32_t)((event >> MOE_SCHED_EVENT_REFILL_REQ_LSB) & 1u);
}

static inline __attribute__((always_inline)) uint32_t
moe_sched_event_refill_top_count(uint64_t event)
{
    return (uint32_t)((event >> MOE_SCHED_EVENT_REFILL_TOP_COUNT_LSB) &
                      MOE_SCHED_EVENT_REFILL_COUNT_MASK);
}

static inline __attribute__((always_inline)) uint32_t
moe_sched_event_refill_bottom_count(uint64_t event)
{
    return (uint32_t)((event >> MOE_SCHED_EVENT_REFILL_BOTTOM_COUNT_LSB) &
                      MOE_SCHED_EVENT_REFILL_COUNT_MASK);
}

static inline __attribute__((always_inline)) uint32_t
moe_sched_event_task_count(uint64_t event)
{
    return (uint32_t)((event >> MOE_SCHED_EVENT_TASK_COUNT_LSB) &
                      MOE_SCHED_EVENT_TASK_COUNT_MASK);
}

static inline __attribute__((always_inline)) void moe_sched_fence(void)
{
    asm volatile("fence iorw, iorw" ::: "memory");
}
