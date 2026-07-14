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

#define MOE_SCHED_CTRL      0x00u
#define MOE_SCHED_STATUS    0x08u
#define MOE_SCHED_CONFIG    0x10u
#define MOE_SCHED_TASK_POP 0x68u
#define MOE_SCHED_HEAD_QUAD        0xa8u
#define MOE_SCHED_RESERVE_QUAD     0xb0u
#define MOE_SCHED_HEAD_PUSH_QUAD   0xb8u
#define MOE_SCHED_TASK_DATA_BASE   0x100u
#define MOE_SCHED_TASK_DATA_STRIDE 0x08u

#define MOE_SCHED_CTRL_INIT         (1ull << 0)
#define MOE_SCHED_CTRL_START        (1ull << 1)

#define MOE_SCHED_STATUS_TASK_VALID   (1ull << 3)
#define MOE_SCHED_STATUS_ACTIVE_EMPTY (1ull << 5)
#define MOE_SCHED_STATUS_REFILL_REQ   (1ull << 6)
#define MOE_SCHED_STATUS_TASK_COUNT_LSB 36u
#define MOE_SCHED_TASK_FIFO_DEPTH 8u

#define MOE_SCHED_EID_RAW_W  6u
#define MOE_SCHED_NTOK_W     9u

#define MOE_SCHED_TASK_WORD_EID_LSB            0u
#define MOE_SCHED_TASK_WORD_TOKEN_START_LSB    6u
#define MOE_SCHED_TASK_WORD_NTOK_LSB           15u
#define MOE_SCHED_TASK_WORD_HAS_S2PF_LSB       24u
#define MOE_SCHED_TASK_WORD_CTRL_LSB           25u
#define MOE_SCHED_TASK_WORD_M_S2_LSB           38u
#define MOE_SCHED_TASK_WORD_M_S4_LSB           47u
#define MOE_SCHED_TASK_WORD_S4PF_DESC_LSB      56u

#define MOE_SCHED_TASK_WORD_NTOK_MASK          0x1ffull
#define MOE_SCHED_TASK_WORD_EID_MASK           0x3full
#define MOE_SCHED_TASK_WORD_SHAPE_MASK         0x3ull
#define MOE_SCHED_TASK_WORD_LOCAL_SLOT_MASK    0x3full
#define MOE_SCHED_TASK_WORD_M_EXEC_MASK        0x1ffull
#define MOE_SCHED_TASK_WORD_CTRL_MASK          0x1fffull

#define MOE_SCHED_TASK_CTRL_SKIP_S1_LSB    0u
#define MOE_SCHED_TASK_CTRL_SKIP_S3_LSB    1u
#define MOE_SCHED_TASK_CTRL_SHAPE_S1_LSB   2u
#define MOE_SCHED_TASK_CTRL_SHAPE_S3_LSB   4u
#define MOE_SCHED_TASK_CTRL_CLUSTER_LSB    6u
#define MOE_SCHED_TASK_CTRL_LOCAL_SLOT_LSB 7u

#define MOE_SCHED_S4PF_DESC_VALID_LSB       0u
#define MOE_SCHED_S4PF_DESC_NO_COPY_LSB     1u
#define MOE_SCHED_S4PF_DESC_TARGET_EID_LSB  2u
#define MOE_SCHED_S4PF_DESC_TARGET_EID_MASK 0x3full

static inline uintptr_t moe_sched_base(void)
{
    return (uintptr_t)chiplet_addr_transform((uint64_t)MOE_SCHED_LOCAL_BASE);
}

static inline void moe_sched_fence(void)
{
    asm volatile("fence iorw, iorw" ::: "memory");
}
