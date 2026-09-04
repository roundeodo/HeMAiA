// Emit compact timing records after the DFG exits. Analysis stays off-target.
#pragma once

#include "heterogeneous_runtime.h"
#include "moe_runtime_timing_record.h"

#if MOE_RUNTIME_TIMING

#ifndef MOE_RUNTIME_TIMING_SCOPE_RECORDS
#define MOE_RUNTIME_TIMING_SCOPE_RECORDS 4u
#endif

static inline void __host_bingo_moe_print_runtime_timing(
    uint64_t *scratchpad_addrs, uint32_t scratchpad_count)
{
    uint32_t record_count = 0u;

    printf_safe("[MOE_TIMING_BEGIN] version=2\r\n");
    for (uint32_t i = 0u; i < scratchpad_count; ++i) {
        volatile bingo_kernel_scratchpad_t *sp =
            (volatile bingo_kernel_scratchpad_t *)(uintptr_t)scratchpad_addrs[i];
        if (sp->reserved[MOE_SP_PROFILE_MAGIC_IDX] != MOE_RUNTIME_TIMING_MAGIC)
            continue;
        uint32_t start = sp->start_time;
        uint32_t total = sp->end_time - start;
        uint32_t resource_offset =
            sp->reserved[MOE_SP_PROFILE_RESOURCE_START_IDX] - start;
        uint32_t resource_cycles =
            sp->reserved[MOE_SP_PROFILE_RESOURCE_END_IDX] -
            sp->reserved[MOE_SP_PROFILE_RESOURCE_START_IDX];

        // Schema v2: meta task start total resource_offset resource_cycles
        //            peer_wait units flags result. "units" is DMA bytes for a
        //            DMA resource and hardware-busy cycles for VersaCore.
        //            Relative fields keep every record below the FPGA console's
        //            80-column capture limit.
        printf_safe(
            "[MOE_TIMING_RECORD] %x %x %u %u %u %u %u %u %x %u\r\n",
            sp->reserved[MOE_SP_PROFILE_META_IDX],
            sp->reserved[MOE_SP_PROFILE_TASK_IDX],
            start,
            total,
            resource_offset,
            resource_cycles,
            sp->reserved[MOE_SP_PROFILE_PEER_WAIT_IDX],
            sp->reserved[MOE_SP_PROFILE_UNITS_IDX],
            sp->reserved[MOE_SP_PROFILE_FLAGS_IDX],
            sp->reserved[MOE_SP_PROFILE_RESULT_IDX]);
        record_count++;
    }
    printf_safe("[MOE_TIMING_END] records=%u\r\n", record_count);
}

static inline void __host_bingo_moe_print_runtime_timing_v3(
    uint64_t *scratchpad_addrs,
    uint32_t *node_ids,
    uint32_t scratchpad_count)
{
    uint32_t record_count = 0u;

    printf_safe(
        "[MOE_TIMING_BEGIN] version=3 level=%u\r\n",
        (uint32_t)MOE_RUNTIME_TIMING);
    for (uint32_t i = 0u; i < scratchpad_count; ++i) {
        volatile bingo_kernel_scratchpad_t *sp =
            (volatile bingo_kernel_scratchpad_t *)(uintptr_t)scratchpad_addrs[i];
        if (sp->reserved[MOE_SP_PROFILE_MAGIC_IDX] != MOE_RUNTIME_TIMING_MAGIC)
            continue;
#if MOE_RUNTIME_TIMING == 1
        uint32_t stage = MOE_PROFILE_META_STAGE(
            sp->reserved[MOE_SP_PROFILE_META_IDX]);
        if (stage != MOE_PROFILE_STAGE_CLUSTER_BEGIN &&
            stage != MOE_PROFILE_STAGE_CLUSTER_END &&
            stage != MOE_PROFILE_STAGE_WORKLOAD_BEGIN &&
            stage != MOE_PROFILE_STAGE_SHARED_BEGIN &&
            stage != MOE_PROFILE_STAGE_SHARED_END &&
            stage != MOE_PROFILE_STAGE_WORKLOAD_END &&
            (stage < MOE_PROFILE_STAGE_M8_FIXED_A_BEGIN ||
             stage > MOE_PROFILE_STAGE_M32_DISTILLED_END))
            continue;
#endif

        uint32_t start = sp->start_time;
        uint32_t total = sp->end_time - start;
        uint32_t resource_offset =
            sp->reserved[MOE_SP_PROFILE_RESOURCE_START_IDX] - start;
        uint32_t resource_cycles =
            sp->reserved[MOE_SP_PROFILE_RESOURCE_END_IDX] -
            sp->reserved[MOE_SP_PROFILE_RESOURCE_START_IDX];

        // Schema v3 adds the DFG node ID before the unchanged v2 payload.
        printf_safe(
            "[MTR3] %x %x %x %u %u %u %u %u %u %x %u\r\n",
            node_ids[i],
            sp->reserved[MOE_SP_PROFILE_META_IDX],
            sp->reserved[MOE_SP_PROFILE_TASK_IDX],
            start,
            total,
            resource_offset,
            resource_cycles,
            sp->reserved[MOE_SP_PROFILE_PEER_WAIT_IDX],
            sp->reserved[MOE_SP_PROFILE_UNITS_IDX],
            sp->reserved[MOE_SP_PROFILE_FLAGS_IDX],
            sp->reserved[MOE_SP_PROFILE_RESULT_IDX]);
        record_count++;
    }
    uint32_t expected_records = scratchpad_count;
#if MOE_RUNTIME_TIMING == 1
    expected_records = MOE_RUNTIME_TIMING_SCOPE_RECORDS;
#endif
    printf_safe(
        "[MOE_TIMING_END] records=%u expected=%u\r\n",
        record_count,
        expected_records);
}

#endif
