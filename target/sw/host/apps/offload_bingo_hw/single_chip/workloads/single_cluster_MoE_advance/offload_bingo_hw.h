// Auto-generated offload_hw_bingo.h
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "single_cluster_MoE_advance_data.h"

// Kernel Name List
// Note: This list is currently for debugging purposes only and is not used in the runtime.
// It will be enabled in the future.
/*
char kernel_name_list[96][64] = {
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 0
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 1
    "__snax_bingo_kernel_gemm_full", // Node ID 2
    "__host_bingo_kernel_idma", // Node ID 3
    "__host_bingo_kernel_moe_router_schedule", // Node ID 4
    "__host_bingo_kernel_compute_delayed_softmax", // Node ID 5
    "__host_bingo_kernel_build_scatter_metadata", // Node ID 6
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 7
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 8
    "__snax_bingo_kernel_gemm_full", // Node ID 9
    "__host_bingo_kernel_idma", // Node ID 10
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 11
    "__snax_bingo_kernel_gemm_minimal", // Node ID 12
    "__host_bingo_kernel_idma", // Node ID 13
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 14
    "__snax_bingo_kernel_gemm_full", // Node ID 15
    "__host_bingo_kernel_idma", // Node ID 16
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 17
    "__snax_bingo_kernel_gemm_minimal", // Node ID 18
    "__host_bingo_kernel_idma", // Node ID 19
    "__host_bingo_kernel_compute_hw_silu_glu", // Node ID 20
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 21
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 22
    "__snax_bingo_kernel_gemm_full", // Node ID 23
    "__host_bingo_kernel_idma", // Node ID 24
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 25
    "__snax_bingo_kernel_gemm_minimal", // Node ID 26
    "__host_bingo_kernel_idma", // Node ID 27
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 28
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 29
    "__snax_bingo_kernel_gemm_full", // Node ID 30
    "__host_bingo_kernel_idma", // Node ID 31
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 32
    "__snax_bingo_kernel_gemm_minimal", // Node ID 33
    "__host_bingo_kernel_idma", // Node ID 34
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 35
    "__snax_bingo_kernel_gemm_full", // Node ID 36
    "__host_bingo_kernel_idma", // Node ID 37
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 38
    "__snax_bingo_kernel_gemm_minimal", // Node ID 39
    "__host_bingo_kernel_idma", // Node ID 40
    "__host_bingo_kernel_compute_hw_silu_glu", // Node ID 41
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 42
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 43
    "__snax_bingo_kernel_gemm_full", // Node ID 44
    "__host_bingo_kernel_idma", // Node ID 45
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 46
    "__snax_bingo_kernel_gemm_minimal", // Node ID 47
    "__host_bingo_kernel_idma", // Node ID 48
    "__host_bingo_kernel_scatter_and_pad_input", // Node ID 49
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 50
    "__snax_bingo_kernel_gemm_full", // Node ID 51
    "__host_bingo_kernel_idma", // Node ID 52
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 53
    "__snax_bingo_kernel_gemm_minimal", // Node ID 54
    "__host_bingo_kernel_idma", // Node ID 55
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 56
    "__snax_bingo_kernel_gemm_full", // Node ID 57
    "__host_bingo_kernel_idma", // Node ID 58
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 59
    "__snax_bingo_kernel_gemm_minimal", // Node ID 60
    "__host_bingo_kernel_idma", // Node ID 61
    "__host_bingo_kernel_compute_hw_silu_glu", // Node ID 62
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 63
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 64
    "__snax_bingo_kernel_gemm_full", // Node ID 65
    "__host_bingo_kernel_idma", // Node ID 66
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 67
    "__snax_bingo_kernel_gemm_minimal", // Node ID 68
    "__host_bingo_kernel_idma", // Node ID 69
    "__host_bingo_kernel_scatter_and_pad_input", // Node ID 70
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 71
    "__snax_bingo_kernel_gemm_full", // Node ID 72
    "__host_bingo_kernel_idma", // Node ID 73
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 74
    "__snax_bingo_kernel_gemm_minimal", // Node ID 75
    "__host_bingo_kernel_idma", // Node ID 76
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 77
    "__snax_bingo_kernel_gemm_full", // Node ID 78
    "__host_bingo_kernel_idma", // Node ID 79
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 80
    "__snax_bingo_kernel_gemm_minimal", // Node ID 81
    "__host_bingo_kernel_idma", // Node ID 82
    "__host_bingo_kernel_compute_hw_silu_glu", // Node ID 83
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 84
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 85
    "__snax_bingo_kernel_gemm_full", // Node ID 86
    "__host_bingo_kernel_idma", // Node ID 87
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 88
    "__snax_bingo_kernel_gemm_minimal", // Node ID 89
    "__host_bingo_kernel_idma", // Node ID 90
    "__host_bingo_kernel_experts_result_accumulate", // Node ID 91
    "__host_bingo_kernel_entry", // Node ID 92
    "__snax_bingo_kernel_exit", // Node ID 93
    "__snax_bingo_kernel_exit", // Node ID 94
    "__host_bingo_kernel_exit", // Node ID 95
};
*/

int kernel_execution(){
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing single_cluster_MoE_advance Workload\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    uint32_t current_chip_id = get_current_chip_id();
    if (current_chip_id == 0x00) {
        uint32_t num_total_tasks = 147;
        // Task Description List
        uint32_t bingo_hw_scheduler_num_task_desc_chip_00 = 147;
        uint64_t* bingo_hw_scheduler_task_desc_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, bingo_hw_scheduler_num_task_desc_chip_00 * sizeof(uint64_t));
        bingo_hw_scheduler_task_desc_list_chip_00[0] = 0x00000100108000B8; // Node ID 92
            // Fields: Type=0, TaskID=92
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[1] = 0x000000801940008E; // Node ID 71
            // Fields: Type=0, TaskID=71
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[2] = 0x0000010010800107; // Node ID 131
            // Fields: Type=1, TaskID=131
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[3] = 0x0000010010800109; // Node ID 132
            // Fields: Type=1, TaskID=132
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[4] = 0x0000008019400000; // Node ID 0
            // Fields: Type=0, TaskID=0
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[5] = 0x0000008019400064; // Node ID 50
            // Fields: Type=0, TaskID=50
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[6] = 0x00000100104000C1; // Node ID 96
            // Fields: Type=1, TaskID=96
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[7] = 0x000000000500010B; // Node ID 133
            // Fields: Type=1, TaskID=133
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[8] = 0x0000008015400002; // Node ID 1
            // Fields: Type=0, TaskID=1
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[9] = 0x0000020015000004; // Node ID 2
            // Fields: Type=0, TaskID=2
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[10] = 0x0000020013800006; // Node ID 3
            // Fields: Type=0, TaskID=3
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[11] = 0x0000020019800008; // Node ID 4
            // Fields: Type=0, TaskID=4
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[12] = 0x000002001980000C; // Node ID 6
            // Fields: Type=0, TaskID=6
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[13] = 0x000001001980000A; // Node ID 5
            // Fields: Type=0, TaskID=5
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[14] = 0x000000801940000E; // Node ID 7
            // Fields: Type=0, TaskID=7
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[15] = 0x00000100104000C3; // Node ID 97
            // Fields: Type=1, TaskID=97
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[16] = 0x000000000500010D; // Node ID 134
            // Fields: Type=1, TaskID=134
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[17] = 0x0000008015400010; // Node ID 8
            // Fields: Type=0, TaskID=8
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[18] = 0x0000020015000012; // Node ID 9
            // Fields: Type=0, TaskID=9
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[19] = 0x00000100104000C5; // Node ID 98
            // Fields: Type=1, TaskID=98
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[20] = 0x0000008013800014; // Node ID 10
            // Fields: Type=0, TaskID=10
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[21] = 0x00000080100000C7; // Node ID 99
            // Fields: Type=1, TaskID=99
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[22] = 0x0000008015400016; // Node ID 11
            // Fields: Type=0, TaskID=11
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[23] = 0x0000000009000119; // Node ID 140
            // Fields: Type=1, TaskID=140
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[24] = 0x0000010017000018; // Node ID 12
            // Fields: Type=0, TaskID=12
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[25] = 0x00000200100000C9; // Node ID 100
            // Fields: Type=1, TaskID=100
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[26] = 0x000001001380001A; // Node ID 13
            // Fields: Type=0, TaskID=13
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[27] = 0x000000801B40001C; // Node ID 14
            // Fields: Type=0, TaskID=14
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[28] = 0x000002001500001E; // Node ID 15
            // Fields: Type=0, TaskID=15
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[29] = 0x00000100104000CB; // Node ID 101
            // Fields: Type=1, TaskID=101
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[30] = 0x0000008013800020; // Node ID 16
            // Fields: Type=0, TaskID=16
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[31] = 0x00000080100000CD; // Node ID 102
            // Fields: Type=1, TaskID=102
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[32] = 0x0000008015400022; // Node ID 17
            // Fields: Type=0, TaskID=17
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[33] = 0x000000000900011B; // Node ID 141
            // Fields: Type=1, TaskID=141
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[34] = 0x0000010017000024; // Node ID 18
            // Fields: Type=0, TaskID=18
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[35] = 0x00000200100000CF; // Node ID 103
            // Fields: Type=1, TaskID=103
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[36] = 0x0000020013800026; // Node ID 19
            // Fields: Type=0, TaskID=19
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[37] = 0x0000010019800028; // Node ID 20
            // Fields: Type=0, TaskID=20
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[38] = 0x000000801B40002A; // Node ID 21
            // Fields: Type=0, TaskID=21
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[39] = 0x00000100104000D1; // Node ID 104
            // Fields: Type=1, TaskID=104
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[40] = 0x000000000500010F; // Node ID 135
            // Fields: Type=1, TaskID=135
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[41] = 0x000000801540002C; // Node ID 22
            // Fields: Type=0, TaskID=22
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[42] = 0x000002001500002E; // Node ID 23
            // Fields: Type=0, TaskID=23
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[43] = 0x00000100104000D3; // Node ID 105
            // Fields: Type=1, TaskID=105
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[44] = 0x0000020013800030; // Node ID 24
            // Fields: Type=0, TaskID=24
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[45] = 0x00000080100000D5; // Node ID 106
            // Fields: Type=1, TaskID=106
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[46] = 0x0000008015400032; // Node ID 25
            // Fields: Type=0, TaskID=25
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[47] = 0x0000020017000034; // Node ID 26
            // Fields: Type=0, TaskID=26
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[48] = 0x000001001B800036; // Node ID 27
            // Fields: Type=0, TaskID=27
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[49] = 0x0000008019400038; // Node ID 28
            // Fields: Type=0, TaskID=28
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[50] = 0x00000100104000D7; // Node ID 107
            // Fields: Type=1, TaskID=107
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[51] = 0x0000000005000111; // Node ID 136
            // Fields: Type=1, TaskID=136
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[52] = 0x000000801540003A; // Node ID 29
            // Fields: Type=0, TaskID=29
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[53] = 0x000002001500003C; // Node ID 30
            // Fields: Type=0, TaskID=30
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[54] = 0x00000100104000D9; // Node ID 108
            // Fields: Type=1, TaskID=108
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[55] = 0x000000801380003E; // Node ID 31
            // Fields: Type=0, TaskID=31
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[56] = 0x00000080100000DB; // Node ID 109
            // Fields: Type=1, TaskID=109
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[57] = 0x0000008015400040; // Node ID 32
            // Fields: Type=0, TaskID=32
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[58] = 0x000000000900011D; // Node ID 142
            // Fields: Type=1, TaskID=142
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[59] = 0x0000010017000042; // Node ID 33
            // Fields: Type=0, TaskID=33
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[60] = 0x00000200100000DD; // Node ID 110
            // Fields: Type=1, TaskID=110
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[61] = 0x0000010013800044; // Node ID 34
            // Fields: Type=0, TaskID=34
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[62] = 0x000000801B400046; // Node ID 35
            // Fields: Type=0, TaskID=35
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[63] = 0x0000020015000048; // Node ID 36
            // Fields: Type=0, TaskID=36
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[64] = 0x00000100104000DF; // Node ID 111
            // Fields: Type=1, TaskID=111
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[65] = 0x000000801380004A; // Node ID 37
            // Fields: Type=0, TaskID=37
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[66] = 0x00000080100000E1; // Node ID 112
            // Fields: Type=1, TaskID=112
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[67] = 0x000000801540004C; // Node ID 38
            // Fields: Type=0, TaskID=38
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[68] = 0x000000000900011F; // Node ID 143
            // Fields: Type=1, TaskID=143
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[69] = 0x000001001700004E; // Node ID 39
            // Fields: Type=0, TaskID=39
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[70] = 0x00000200100000E3; // Node ID 113
            // Fields: Type=1, TaskID=113
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[71] = 0x0000020013800050; // Node ID 40
            // Fields: Type=0, TaskID=40
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[72] = 0x0000010019800052; // Node ID 41
            // Fields: Type=0, TaskID=41
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[73] = 0x000000801B400054; // Node ID 42
            // Fields: Type=0, TaskID=42
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[74] = 0x00000100104000E5; // Node ID 114
            // Fields: Type=1, TaskID=114
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[75] = 0x0000000005000113; // Node ID 137
            // Fields: Type=1, TaskID=137
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[76] = 0x0000008015400056; // Node ID 43
            // Fields: Type=0, TaskID=43
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[77] = 0x0000020015000058; // Node ID 44
            // Fields: Type=0, TaskID=44
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[78] = 0x00000100104000E7; // Node ID 115
            // Fields: Type=1, TaskID=115
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[79] = 0x000002001380005A; // Node ID 45
            // Fields: Type=0, TaskID=45
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[80] = 0x00000080100000E9; // Node ID 116
            // Fields: Type=1, TaskID=116
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[81] = 0x000000801540005C; // Node ID 46
            // Fields: Type=0, TaskID=46
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[82] = 0x000002001700005E; // Node ID 47
            // Fields: Type=0, TaskID=47
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[83] = 0x000002001B800060; // Node ID 48
            // Fields: Type=0, TaskID=48
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[84] = 0x0000008019800062; // Node ID 49
            // Fields: Type=0, TaskID=49
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[85] = 0x000002001D000066; // Node ID 51
            // Fields: Type=0, TaskID=51
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b110
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[86] = 0x0000010013800068; // Node ID 52
            // Fields: Type=0, TaskID=52
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[87] = 0x0000008019400070; // Node ID 56
            // Fields: Type=0, TaskID=56
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[88] = 0x00000100108000EB; // Node ID 117
            // Fields: Type=1, TaskID=117
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[89] = 0x000000801940006A; // Node ID 53
            // Fields: Type=0, TaskID=53
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[90] = 0x000002001500006C; // Node ID 54
            // Fields: Type=0, TaskID=54
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[91] = 0x000000801380006E; // Node ID 55
            // Fields: Type=0, TaskID=55
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[92] = 0x00000080100000ED; // Node ID 118
            // Fields: Type=1, TaskID=118
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[93] = 0x0000000009000121; // Node ID 144
            // Fields: Type=1, TaskID=144
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[94] = 0x0000020017000072; // Node ID 57
            // Fields: Type=0, TaskID=57
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[95] = 0x0000010013800074; // Node ID 58
            // Fields: Type=0, TaskID=58
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[96] = 0x0000008019400076; // Node ID 59
            // Fields: Type=0, TaskID=59
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[97] = 0x0000010015000078; // Node ID 60
            // Fields: Type=0, TaskID=60
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[98] = 0x00000200100000EF; // Node ID 119
            // Fields: Type=1, TaskID=119
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[99] = 0x000002001380007A; // Node ID 61
            // Fields: Type=0, TaskID=61
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[100] = 0x000001001980007C; // Node ID 62
            // Fields: Type=0, TaskID=62
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[101] = 0x000000801B40007E; // Node ID 63
            // Fields: Type=0, TaskID=63
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[102] = 0x00000200108000F1; // Node ID 120
            // Fields: Type=1, TaskID=120
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[103] = 0x00000100104000F3; // Node ID 121
            // Fields: Type=1, TaskID=121
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[104] = 0x0000000005000115; // Node ID 138
            // Fields: Type=1, TaskID=138
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[105] = 0x0000008015400080; // Node ID 64
            // Fields: Type=0, TaskID=64
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[106] = 0x0000020015000082; // Node ID 65
            // Fields: Type=0, TaskID=65
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[107] = 0x00000100104000F5; // Node ID 122
            // Fields: Type=1, TaskID=122
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[108] = 0x0000020013800084; // Node ID 66
            // Fields: Type=0, TaskID=66
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[109] = 0x00000080100000F7; // Node ID 123
            // Fields: Type=1, TaskID=123
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[110] = 0x0000008015400086; // Node ID 67
            // Fields: Type=0, TaskID=67
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[111] = 0x0000020017000088; // Node ID 68
            // Fields: Type=0, TaskID=68
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[112] = 0x000000801B80008A; // Node ID 69
            // Fields: Type=0, TaskID=69
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[113] = 0x00000200100000F9; // Node ID 124
            // Fields: Type=1, TaskID=124
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[114] = 0x0000000009000123; // Node ID 145
            // Fields: Type=1, TaskID=145
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[115] = 0x000000801B80008C; // Node ID 70
            // Fields: Type=0, TaskID=70
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[116] = 0x000002001D000090; // Node ID 72
            // Fields: Type=0, TaskID=72
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b110
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[117] = 0x0000010013800092; // Node ID 73
            // Fields: Type=0, TaskID=73
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[118] = 0x000000801940009A; // Node ID 77
            // Fields: Type=0, TaskID=77
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[119] = 0x00000100108000FB; // Node ID 125
            // Fields: Type=1, TaskID=125
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[120] = 0x0000008019400094; // Node ID 74
            // Fields: Type=0, TaskID=74
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[121] = 0x0000020015000096; // Node ID 75
            // Fields: Type=0, TaskID=75
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[122] = 0x0000008013800098; // Node ID 76
            // Fields: Type=0, TaskID=76
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[123] = 0x00000080100000FD; // Node ID 126
            // Fields: Type=1, TaskID=126
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[124] = 0x0000000009000125; // Node ID 146
            // Fields: Type=1, TaskID=146
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[125] = 0x000002001700009C; // Node ID 78
            // Fields: Type=0, TaskID=78
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[126] = 0x000001001380009E; // Node ID 79
            // Fields: Type=0, TaskID=79
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[127] = 0x00000080194000A0; // Node ID 80
            // Fields: Type=0, TaskID=80
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[128] = 0x00000100150000A2; // Node ID 81
            // Fields: Type=0, TaskID=81
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[129] = 0x00000200100000FF; // Node ID 127
            // Fields: Type=1, TaskID=127
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[130] = 0x00000200138000A4; // Node ID 82
            // Fields: Type=0, TaskID=82
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[131] = 0x00000100198000A6; // Node ID 83
            // Fields: Type=0, TaskID=83
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[132] = 0x000000801B4000A8; // Node ID 84
            // Fields: Type=0, TaskID=84
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[133] = 0x0000010010400101; // Node ID 128
            // Fields: Type=1, TaskID=128
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[134] = 0x0000000005000117; // Node ID 139
            // Fields: Type=1, TaskID=139
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[135] = 0x00000080154000AA; // Node ID 85
            // Fields: Type=0, TaskID=85
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[136] = 0x00000200150000AC; // Node ID 86
            // Fields: Type=0, TaskID=86
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[137] = 0x0000010010400103; // Node ID 129
            // Fields: Type=1, TaskID=129
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[138] = 0x00000200138000AE; // Node ID 87
            // Fields: Type=0, TaskID=87
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[139] = 0x0000008010000105; // Node ID 130
            // Fields: Type=1, TaskID=130
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[140] = 0x00000080154000B0; // Node ID 88
            // Fields: Type=0, TaskID=88
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[141] = 0x00000200170000B2; // Node ID 89
            // Fields: Type=0, TaskID=89
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b011
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[142] = 0x000002001B8000B4; // Node ID 90
            // Fields: Type=0, TaskID=90
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b101
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[143] = 0x00000080198000B6; // Node ID 91
            // Fields: Type=0, TaskID=91
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[144] = 0x00000100190000BA; // Node ID 93
            // Fields: Type=0, TaskID=93
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[145] = 0x00000200134000BC; // Node ID 94
            // Fields: Type=0, TaskID=94
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[146] = 0x00000000058000BE; // Node ID 95
            // Fields: Type=0, TaskID=95
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        // Task ID Mapping Lists
        int32_t* global_task_id_to_dev_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 147 * sizeof(int32_t));
        global_task_id_to_dev_task_id_chip_00[0] = 0; // Node ID 0 -> Dev Task 0 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[1] = 1; // Node ID 1 -> Dev Task 1 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[2] = 2; // Node ID 2 -> Dev Task 2 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[3] = -1; // Node ID 3 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[4] = -1; // Node ID 4 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule)
        global_task_id_to_dev_task_id_chip_00[5] = -1; // Node ID 5 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_delayed_softmax)
        global_task_id_to_dev_task_id_chip_00[6] = -1; // Node ID 6 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_build_scatter_metadata)
        global_task_id_to_dev_task_id_chip_00[7] = 3; // Node ID 7 -> Dev Task 3 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[8] = 4; // Node ID 8 -> Dev Task 4 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[9] = 5; // Node ID 9 -> Dev Task 5 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[10] = -1; // Node ID 10 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[11] = 6; // Node ID 11 -> Dev Task 6 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[12] = 7; // Node ID 12 -> Dev Task 7 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[13] = -1; // Node ID 13 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[14] = 8; // Node ID 14 -> Dev Task 8 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[15] = 9; // Node ID 15 -> Dev Task 9 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[16] = -1; // Node ID 16 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[17] = 10; // Node ID 17 -> Dev Task 10 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[18] = 11; // Node ID 18 -> Dev Task 11 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[19] = -1; // Node ID 19 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[20] = -1; // Node ID 20 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_dev_task_id_chip_00[21] = 12; // Node ID 21 -> Dev Task 12 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[22] = 13; // Node ID 22 -> Dev Task 13 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[23] = 14; // Node ID 23 -> Dev Task 14 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[24] = -1; // Node ID 24 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[25] = 15; // Node ID 25 -> Dev Task 15 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[26] = 16; // Node ID 26 -> Dev Task 16 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[27] = -1; // Node ID 27 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[28] = 17; // Node ID 28 -> Dev Task 17 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[29] = 18; // Node ID 29 -> Dev Task 18 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[30] = 19; // Node ID 30 -> Dev Task 19 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[31] = -1; // Node ID 31 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[32] = 20; // Node ID 32 -> Dev Task 20 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[33] = 21; // Node ID 33 -> Dev Task 21 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[34] = -1; // Node ID 34 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[35] = 22; // Node ID 35 -> Dev Task 22 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[36] = 23; // Node ID 36 -> Dev Task 23 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[37] = -1; // Node ID 37 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[38] = 24; // Node ID 38 -> Dev Task 24 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[39] = 25; // Node ID 39 -> Dev Task 25 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[40] = -1; // Node ID 40 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[41] = -1; // Node ID 41 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_dev_task_id_chip_00[42] = 26; // Node ID 42 -> Dev Task 26 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[43] = 27; // Node ID 43 -> Dev Task 27 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[44] = 28; // Node ID 44 -> Dev Task 28 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[45] = -1; // Node ID 45 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[46] = 29; // Node ID 46 -> Dev Task 29 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[47] = 30; // Node ID 47 -> Dev Task 30 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[48] = -1; // Node ID 48 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[49] = -1; // Node ID 49 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_scatter_and_pad_input)
        global_task_id_to_dev_task_id_chip_00[50] = 31; // Node ID 50 -> Dev Task 31 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[51] = 32; // Node ID 51 -> Dev Task 32 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[52] = -1; // Node ID 52 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[53] = 33; // Node ID 53 -> Dev Task 33 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[54] = 34; // Node ID 54 -> Dev Task 34 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[55] = -1; // Node ID 55 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[56] = 35; // Node ID 56 -> Dev Task 35 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[57] = 36; // Node ID 57 -> Dev Task 36 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[58] = -1; // Node ID 58 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[59] = 37; // Node ID 59 -> Dev Task 37 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[60] = 38; // Node ID 60 -> Dev Task 38 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[61] = -1; // Node ID 61 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[62] = -1; // Node ID 62 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_dev_task_id_chip_00[63] = 39; // Node ID 63 -> Dev Task 39 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[64] = 40; // Node ID 64 -> Dev Task 40 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[65] = 41; // Node ID 65 -> Dev Task 41 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[66] = -1; // Node ID 66 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[67] = 42; // Node ID 67 -> Dev Task 42 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[68] = 43; // Node ID 68 -> Dev Task 43 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[69] = -1; // Node ID 69 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[70] = -1; // Node ID 70 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_scatter_and_pad_input)
        global_task_id_to_dev_task_id_chip_00[71] = 44; // Node ID 71 -> Dev Task 44 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[72] = 45; // Node ID 72 -> Dev Task 45 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[73] = -1; // Node ID 73 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[74] = 46; // Node ID 74 -> Dev Task 46 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[75] = 47; // Node ID 75 -> Dev Task 47 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[76] = -1; // Node ID 76 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[77] = 48; // Node ID 77 -> Dev Task 48 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[78] = 49; // Node ID 78 -> Dev Task 49 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[79] = -1; // Node ID 79 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[80] = 50; // Node ID 80 -> Dev Task 50 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[81] = 51; // Node ID 81 -> Dev Task 51 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[82] = -1; // Node ID 82 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[83] = -1; // Node ID 83 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_dev_task_id_chip_00[84] = 52; // Node ID 84 -> Dev Task 52 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[85] = 53; // Node ID 85 -> Dev Task 53 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[86] = 54; // Node ID 86 -> Dev Task 54 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_dev_task_id_chip_00[87] = -1; // Node ID 87 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[88] = 55; // Node ID 88 -> Dev Task 55 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[89] = 56; // Node ID 89 -> Dev Task 56 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_dev_task_id_chip_00[90] = -1; // Node ID 90 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[91] = -1; // Node ID 91 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_experts_result_accumulate)
        global_task_id_to_dev_task_id_chip_00[92] = -1; // Node ID 92 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_dev_task_id_chip_00[93] = 57; // Node ID 93 -> Dev Task 57 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[94] = 58; // Node ID 94 -> Dev Task 58 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[95] = -1; // Node ID 95 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[96] = -1; // Node ID 96 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[97] = -1; // Node ID 97 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[98] = -1; // Node ID 98 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[99] = -1; // Node ID 99 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[100] = -1; // Node ID 100 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[101] = -1; // Node ID 101 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[102] = -1; // Node ID 102 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[103] = -1; // Node ID 103 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[104] = -1; // Node ID 104 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[105] = -1; // Node ID 105 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[106] = -1; // Node ID 106 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[107] = -1; // Node ID 107 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[108] = -1; // Node ID 108 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[109] = -1; // Node ID 109 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[110] = -1; // Node ID 110 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[111] = -1; // Node ID 111 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[112] = -1; // Node ID 112 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[113] = -1; // Node ID 113 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[114] = -1; // Node ID 114 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[115] = -1; // Node ID 115 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[116] = -1; // Node ID 116 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[117] = -1; // Node ID 117 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[118] = -1; // Node ID 118 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[119] = -1; // Node ID 119 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[120] = -1; // Node ID 120 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[121] = -1; // Node ID 121 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[122] = -1; // Node ID 122 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[123] = -1; // Node ID 123 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[124] = -1; // Node ID 124 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[125] = -1; // Node ID 125 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[126] = -1; // Node ID 126 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[127] = -1; // Node ID 127 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[128] = -1; // Node ID 128 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[129] = -1; // Node ID 129 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[130] = -1; // Node ID 130 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[131] = -1; // Node ID 131 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[132] = -1; // Node ID 132 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[133] = -1; // Node ID 133 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[134] = -1; // Node ID 134 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[135] = -1; // Node ID 135 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[136] = -1; // Node ID 136 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[137] = -1; // Node ID 137 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[138] = -1; // Node ID 138 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[139] = -1; // Node ID 139 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[140] = -1; // Node ID 140 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[141] = -1; // Node ID 141 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[142] = -1; // Node ID 142 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[143] = -1; // Node ID 143 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[144] = -1; // Node ID 144 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[145] = -1; // Node ID 145 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[146] = -1; // Node ID 146 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        uint32_t num_dev_tasks_chip_00 = 59;
        int32_t* global_task_id_to_host_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 147 * sizeof(int32_t));
        global_task_id_to_host_task_id_chip_00[0] = -1; // Node ID 0 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[1] = -1; // Node ID 1 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[2] = -1; // Node ID 2 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[3] = 0; // Node ID 3 -> Host Task 0 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[4] = 1; // Node ID 4 -> Host Task 1 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule)
        global_task_id_to_host_task_id_chip_00[5] = 2; // Node ID 5 -> Host Task 2 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_delayed_softmax)
        global_task_id_to_host_task_id_chip_00[6] = 3; // Node ID 6 -> Host Task 3 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_build_scatter_metadata)
        global_task_id_to_host_task_id_chip_00[7] = -1; // Node ID 7 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[8] = -1; // Node ID 8 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[9] = -1; // Node ID 9 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[10] = 4; // Node ID 10 -> Host Task 4 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[11] = -1; // Node ID 11 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[12] = -1; // Node ID 12 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[13] = 5; // Node ID 13 -> Host Task 5 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[14] = -1; // Node ID 14 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[15] = -1; // Node ID 15 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[16] = 6; // Node ID 16 -> Host Task 6 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[17] = -1; // Node ID 17 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[18] = -1; // Node ID 18 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[19] = 7; // Node ID 19 -> Host Task 7 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[20] = 8; // Node ID 20 -> Host Task 8 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_host_task_id_chip_00[21] = -1; // Node ID 21 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[22] = -1; // Node ID 22 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[23] = -1; // Node ID 23 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[24] = 9; // Node ID 24 -> Host Task 9 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[25] = -1; // Node ID 25 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[26] = -1; // Node ID 26 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[27] = 10; // Node ID 27 -> Host Task 10 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[28] = -1; // Node ID 28 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[29] = -1; // Node ID 29 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[30] = -1; // Node ID 30 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[31] = 11; // Node ID 31 -> Host Task 11 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[32] = -1; // Node ID 32 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[33] = -1; // Node ID 33 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[34] = 12; // Node ID 34 -> Host Task 12 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[35] = -1; // Node ID 35 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[36] = -1; // Node ID 36 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[37] = 13; // Node ID 37 -> Host Task 13 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[38] = -1; // Node ID 38 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[39] = -1; // Node ID 39 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[40] = 14; // Node ID 40 -> Host Task 14 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[41] = 15; // Node ID 41 -> Host Task 15 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_host_task_id_chip_00[42] = -1; // Node ID 42 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[43] = -1; // Node ID 43 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[44] = -1; // Node ID 44 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[45] = 16; // Node ID 45 -> Host Task 16 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[46] = -1; // Node ID 46 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[47] = -1; // Node ID 47 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[48] = 17; // Node ID 48 -> Host Task 17 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[49] = 18; // Node ID 49 -> Host Task 18 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_scatter_and_pad_input)
        global_task_id_to_host_task_id_chip_00[50] = -1; // Node ID 50 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[51] = -1; // Node ID 51 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[52] = 19; // Node ID 52 -> Host Task 19 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[53] = -1; // Node ID 53 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[54] = -1; // Node ID 54 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[55] = 20; // Node ID 55 -> Host Task 20 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[56] = -1; // Node ID 56 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[57] = -1; // Node ID 57 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[58] = 21; // Node ID 58 -> Host Task 21 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[59] = -1; // Node ID 59 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[60] = -1; // Node ID 60 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[61] = 22; // Node ID 61 -> Host Task 22 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[62] = 23; // Node ID 62 -> Host Task 23 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_host_task_id_chip_00[63] = -1; // Node ID 63 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[64] = -1; // Node ID 64 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[65] = -1; // Node ID 65 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[66] = 24; // Node ID 66 -> Host Task 24 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[67] = -1; // Node ID 67 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[68] = -1; // Node ID 68 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[69] = 25; // Node ID 69 -> Host Task 25 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[70] = 26; // Node ID 70 -> Host Task 26 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_scatter_and_pad_input)
        global_task_id_to_host_task_id_chip_00[71] = -1; // Node ID 71 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[72] = -1; // Node ID 72 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[73] = 27; // Node ID 73 -> Host Task 27 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[74] = -1; // Node ID 74 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[75] = -1; // Node ID 75 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[76] = 28; // Node ID 76 -> Host Task 28 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[77] = -1; // Node ID 77 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[78] = -1; // Node ID 78 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[79] = 29; // Node ID 79 -> Host Task 29 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[80] = -1; // Node ID 80 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[81] = -1; // Node ID 81 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[82] = 30; // Node ID 82 -> Host Task 30 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[83] = 31; // Node ID 83 -> Host Task 31 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu)
        global_task_id_to_host_task_id_chip_00[84] = -1; // Node ID 84 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[85] = -1; // Node ID 85 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[86] = -1; // Node ID 86 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full)
        global_task_id_to_host_task_id_chip_00[87] = 32; // Node ID 87 -> Host Task 32 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[88] = -1; // Node ID 88 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[89] = -1; // Node ID 89 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal)
        global_task_id_to_host_task_id_chip_00[90] = 33; // Node ID 90 -> Host Task 33 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[91] = 34; // Node ID 91 -> Host Task 34 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_experts_result_accumulate)
        global_task_id_to_host_task_id_chip_00[92] = 35; // Node ID 92 -> Host Task 35 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_host_task_id_chip_00[93] = -1; // Node ID 93 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[94] = -1; // Node ID 94 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[95] = 36; // Node ID 95 -> Host Task 36 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[96] = -1; // Node ID 96 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[97] = -1; // Node ID 97 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[98] = -1; // Node ID 98 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[99] = -1; // Node ID 99 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[100] = -1; // Node ID 100 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[101] = -1; // Node ID 101 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[102] = -1; // Node ID 102 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[103] = -1; // Node ID 103 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[104] = -1; // Node ID 104 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[105] = -1; // Node ID 105 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[106] = -1; // Node ID 106 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[107] = -1; // Node ID 107 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[108] = -1; // Node ID 108 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[109] = -1; // Node ID 109 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[110] = -1; // Node ID 110 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[111] = -1; // Node ID 111 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[112] = -1; // Node ID 112 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[113] = -1; // Node ID 113 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[114] = -1; // Node ID 114 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[115] = -1; // Node ID 115 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[116] = -1; // Node ID 116 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[117] = -1; // Node ID 117 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[118] = -1; // Node ID 118 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[119] = -1; // Node ID 119 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[120] = -1; // Node ID 120 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[121] = -1; // Node ID 121 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[122] = -1; // Node ID 122 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[123] = -1; // Node ID 123 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[124] = -1; // Node ID 124 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[125] = -1; // Node ID 125 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[126] = -1; // Node ID 126 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[127] = -1; // Node ID 127 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[128] = -1; // Node ID 128 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[129] = -1; // Node ID 129 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[130] = -1; // Node ID 130 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[131] = -1; // Node ID 131 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[132] = -1; // Node ID 132 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[133] = -1; // Node ID 133 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[134] = -1; // Node ID 134 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[135] = -1; // Node ID 135 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[136] = -1; // Node ID 136 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[137] = -1; // Node ID 137 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[138] = -1; // Node ID 138 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[139] = -1; // Node ID 139 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[140] = -1; // Node ID 140 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[141] = -1; // Node ID 141 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[142] = -1; // Node ID 142 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[143] = -1; // Node ID 143 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[144] = -1; // Node ID 144 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[145] = -1; // Node ID 145 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[146] = -1; // Node ID 146 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        uint32_t num_host_tasks_chip_00 = 37;
        // 1. Memory Allocations
        uint64_t ptr_l1_buf_A = bingo_l1_alloc(0x00, 0, 1024);
        uint64_t ptr_l1_buf_B_ping = bingo_l1_alloc(0x00, 0, 65536);
        uint64_t ptr_l1_buf_B_pong = bingo_l1_alloc(0x00, 0, 65536);
        uint64_t ptr_l1_buf_D_ping = bingo_l1_alloc(0x00, 0, 256);
        uint64_t ptr_l1_buf_D_pong = bingo_l1_alloc(0x00, 0, 256);
        uint64_t ptr_l3_exp_hist = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_exp_offset = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_final_moe_out = bingo_l3_alloc(0x00, 512);
        uint64_t ptr_l3_indiv_act_a = bingo_l3_alloc(0x00, 256);
        uint64_t ptr_l3_indiv_down_out = bingo_l3_alloc(0x00, 1024);
        uint64_t ptr_l3_indiv_gate_out = bingo_l3_alloc(0x00, 1024);
        uint64_t ptr_l3_indiv_up_out = bingo_l3_alloc(0x00, 1024);
        uint64_t ptr_l3_rev_idx = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_router_hw_out = bingo_l3_alloc(0x00, 256);
        uint64_t ptr_l3_router_prob = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_router_topk_idx = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_router_topk_scores = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_shared_act_a = bingo_l3_alloc(0x00, 256);
        uint64_t ptr_l3_shared_down_out = bingo_l3_alloc(0x00, 1024);
        uint64_t ptr_l3_shared_gate_out = bingo_l3_alloc(0x00, 1024);
        uint64_t ptr_l3_shared_up_out = bingo_l3_alloc(0x00, 1024);

        // 2. Prepare device/host arg/kernel lists
        uint32_t* device_arg_list_chip_00 = (uint32_t*)bingo_l3_alloc(0x00, num_dev_tasks_chip_00 * sizeof(uint32_t));
        uint32_t* device_kernel_list_chip_00 = (uint32_t*)bingo_l3_alloc(0x00, num_dev_tasks_chip_00 * sizeof(uint32_t));
        uint64_t* host_arg_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, num_host_tasks_chip_00 * sizeof(uint64_t));
        uint64_t* host_kernel_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, num_host_tasks_chip_00 * sizeof(uint64_t));

        // 3. Task Arguments Init
        // Node ID: 0 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_0 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_0->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)router_B)));
        args_dev_chip00_0->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)router_B))) >> 32);
        args_dev_chip00_0->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_0->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_0->size = 65536;
        device_arg_list_chip_00[0] = (uint32_t)(uintptr_t)args_dev_chip00_0;
        device_kernel_list_chip_00[0] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 1 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_1 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_1->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_1->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)input_A))) >> 32);
        args_dev_chip00_1->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_1->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_1->size = 1024;
        device_arg_list_chip_00[1] = (uint32_t)(uintptr_t)args_dev_chip00_1;
        device_kernel_list_chip_00[1] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 2 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_2 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_2->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_2->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_2->input_C_addr = (uint32_t)0;
        args_dev_chip00_2->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_2->M = 1;
        args_dev_chip00_2->K = 128;
        args_dev_chip00_2->N = 1;
        args_dev_chip00_2->array_shape_idx = 1;
        args_dev_chip00_2->transpose_A = 0;
        args_dev_chip00_2->transpose_B = 0;
        args_dev_chip00_2->accumPrevC = 0;
        args_dev_chip00_2->silu_enable = 0;
        device_arg_list_chip_00[2] = (uint32_t)(uintptr_t)args_dev_chip00_2;
        device_kernel_list_chip_00[2] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 3 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_3 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_3->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_3->dst_addr = (uint64_t)ptr_l3_router_hw_out;
        args_host_chip00_3->size = 256;
        host_arg_list_chip_00[0] = (uint64_t)(uintptr_t)args_host_chip00_3;
        host_kernel_list_chip_00[0] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 4 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule (__host_bingo_kernel_moe_router_schedule)
        __host_bingo_kernel_moe_router_schedule_args_t* args_host_chip00_4 = (__host_bingo_kernel_moe_router_schedule_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_moe_router_schedule_args_t));
        args_host_chip00_4->total_tokens = 1;
        args_host_chip00_4->hardware_output_buffer_addr = (uint64_t)ptr_l3_router_hw_out;
        args_host_chip00_4->global_indices_out_addr = (uint64_t)ptr_l3_router_topk_idx;
        args_host_chip00_4->global_scores_out_addr = (uint64_t)ptr_l3_router_topk_scores;
        args_host_chip00_4->expert_number_each_layer = 2;
        args_host_chip00_4->individual_expert_number_k = 2;
        args_host_chip00_4->mesh_row = 1;
        args_host_chip00_4->mesh_col = 64;
        args_host_chip00_4->router_m1 = 1;
        args_host_chip00_4->router_n1 = 1;
        host_arg_list_chip_00[1] = (uint64_t)(uintptr_t)args_host_chip00_4;
        host_kernel_list_chip_00[1] = (uint64_t)(uintptr_t)&__host_bingo_kernel_moe_router_schedule;
        // Node ID: 5 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_delayed_softmax (__host_bingo_kernel_compute_delayed_softmax)
        __host_bingo_kernel_compute_delayed_softmax_args_t* args_host_chip00_5 = (__host_bingo_kernel_compute_delayed_softmax_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_compute_delayed_softmax_args_t));
        args_host_chip00_5->global_top_k_scores_ptr_addr = (uint64_t)ptr_l3_router_topk_scores;
        args_host_chip00_5->global_calculated_probability_ptr_addr = (uint64_t)ptr_l3_router_prob;
        args_host_chip00_5->actual_total_tokens = 1;
        args_host_chip00_5->individual_expert_number_k = 2;
        args_host_chip00_5->softmax_scale_raw = 1199570944;
        host_arg_list_chip_00[2] = (uint64_t)(uintptr_t)args_host_chip00_5;
        host_kernel_list_chip_00[2] = (uint64_t)(uintptr_t)&__host_bingo_kernel_compute_delayed_softmax;
        // Node ID: 6 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_build_scatter_metadata (__host_bingo_kernel_build_scatter_metadata)
        __host_bingo_kernel_build_scatter_metadata_args_t* args_host_chip00_6 = (__host_bingo_kernel_build_scatter_metadata_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_build_scatter_metadata_args_t));
        args_host_chip00_6->global_top_k_indices_addr = (uint64_t)ptr_l3_router_topk_idx;
        args_host_chip00_6->actual_total_tokens = 1;
        args_host_chip00_6->expert_token_counts_addr = (uint64_t)ptr_l3_exp_hist;
        args_host_chip00_6->expert_memory_offsets_addr = (uint64_t)ptr_l3_exp_offset;
        args_host_chip00_6->reverse_original_token_flat_idx_addr = (uint64_t)ptr_l3_rev_idx;
        args_host_chip00_6->expert_number_each_layer = 2;
        args_host_chip00_6->individual_expert_number_k = 2;
        host_arg_list_chip_00[3] = (uint64_t)(uintptr_t)args_host_chip00_6;
        host_kernel_list_chip_00[3] = (uint64_t)(uintptr_t)&__host_bingo_kernel_build_scatter_metadata;
        // Node ID: 7 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_7 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_7->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_7->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)input_A))) >> 32);
        args_dev_chip00_7->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_7->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_7->size = 1024;
        device_arg_list_chip_00[3] = (uint32_t)(uintptr_t)args_dev_chip00_7;
        device_kernel_list_chip_00[3] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 8 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_8 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_8->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B)));
        args_dev_chip00_8->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B))) >> 32);
        args_dev_chip00_8->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_8->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_8->size = 65536;
        device_arg_list_chip_00[4] = (uint32_t)(uintptr_t)args_dev_chip00_8;
        device_kernel_list_chip_00[4] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 9 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_9 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_9->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_9->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_9->input_C_addr = (uint32_t)0;
        args_dev_chip00_9->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_9->M = 1;
        args_dev_chip00_9->K = 128;
        args_dev_chip00_9->N = 1;
        args_dev_chip00_9->array_shape_idx = 1;
        args_dev_chip00_9->transpose_A = 0;
        args_dev_chip00_9->transpose_B = 0;
        args_dev_chip00_9->accumPrevC = 0;
        args_dev_chip00_9->silu_enable = 1;
        device_arg_list_chip_00[5] = (uint32_t)(uintptr_t)args_dev_chip00_9;
        device_kernel_list_chip_00[5] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 10 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_10 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_10->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_10->dst_addr = (uint64_t)ptr_l3_shared_gate_out;
        args_host_chip00_10->size = 256;
        host_arg_list_chip_00[4] = (uint64_t)(uintptr_t)args_host_chip00_10;
        host_kernel_list_chip_00[4] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 11 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_11 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_11->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B + 65536)));
        args_dev_chip00_11->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B + 65536))) >> 32);
        args_dev_chip00_11->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_11->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_11->size = 65536;
        device_arg_list_chip_00[6] = (uint32_t)(uintptr_t)args_dev_chip00_11;
        device_kernel_list_chip_00[6] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 12 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_12 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_12->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_12->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_12->input_C_addr = (uint32_t)0;
        args_dev_chip00_12->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_12->silu_enable = 1;
        device_arg_list_chip_00[7] = (uint32_t)(uintptr_t)args_dev_chip00_12;
        device_kernel_list_chip_00[7] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 13 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_13 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_13->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_13->dst_addr = (uint64_t)ptr_l3_shared_gate_out + 256;
        args_host_chip00_13->size = 256;
        host_arg_list_chip_00[5] = (uint64_t)(uintptr_t)args_host_chip00_13;
        host_kernel_list_chip_00[5] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 14 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_14 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_14->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B)));
        args_dev_chip00_14->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B))) >> 32);
        args_dev_chip00_14->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_14->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_14->size = 65536;
        device_arg_list_chip_00[8] = (uint32_t)(uintptr_t)args_dev_chip00_14;
        device_kernel_list_chip_00[8] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 15 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_15 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_15->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_15->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_15->input_C_addr = (uint32_t)0;
        args_dev_chip00_15->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_15->M = 1;
        args_dev_chip00_15->K = 128;
        args_dev_chip00_15->N = 1;
        args_dev_chip00_15->array_shape_idx = 1;
        args_dev_chip00_15->transpose_A = 0;
        args_dev_chip00_15->transpose_B = 0;
        args_dev_chip00_15->accumPrevC = 0;
        args_dev_chip00_15->silu_enable = 0;
        device_arg_list_chip_00[9] = (uint32_t)(uintptr_t)args_dev_chip00_15;
        device_kernel_list_chip_00[9] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 16 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_16 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_16->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_16->dst_addr = (uint64_t)ptr_l3_shared_up_out;
        args_host_chip00_16->size = 256;
        host_arg_list_chip_00[6] = (uint64_t)(uintptr_t)args_host_chip00_16;
        host_kernel_list_chip_00[6] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 17 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_17 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_17->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B + 65536)));
        args_dev_chip00_17->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B + 65536))) >> 32);
        args_dev_chip00_17->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_17->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_17->size = 65536;
        device_arg_list_chip_00[10] = (uint32_t)(uintptr_t)args_dev_chip00_17;
        device_kernel_list_chip_00[10] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 18 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_18 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_18->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_18->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_18->input_C_addr = (uint32_t)0;
        args_dev_chip00_18->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_18->silu_enable = 0;
        device_arg_list_chip_00[11] = (uint32_t)(uintptr_t)args_dev_chip00_18;
        device_kernel_list_chip_00[11] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 19 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_19 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_19->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_19->dst_addr = (uint64_t)ptr_l3_shared_up_out + 256;
        args_host_chip00_19->size = 256;
        host_arg_list_chip_00[7] = (uint64_t)(uintptr_t)args_host_chip00_19;
        host_kernel_list_chip_00[7] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 20 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu (__host_bingo_kernel_compute_hw_silu_glu)
        __host_bingo_kernel_compute_hw_silu_glu_args_t* args_host_chip00_20 = (__host_bingo_kernel_compute_hw_silu_glu_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_compute_hw_silu_glu_args_t));
        args_host_chip00_20->gate_silu_hw_data_addr = (uint64_t)ptr_l3_shared_gate_out;
        args_host_chip00_20->up_projection_hw_data_addr = (uint64_t)ptr_l3_shared_up_out;
        args_host_chip00_20->activated_out_data_addr = (uint64_t)ptr_l3_shared_act_a;
        args_host_chip00_20->valid_elements = 128;
        args_host_chip00_20->swish_glu_scale_in_raw = 931135488;
        args_host_chip00_20->swish_glu_scale_out_raw = 1107296256;
        host_arg_list_chip_00[8] = (uint64_t)(uintptr_t)args_host_chip00_20;
        host_kernel_list_chip_00[8] = (uint64_t)(uintptr_t)&__host_bingo_kernel_compute_hw_silu_glu;
        // Node ID: 21 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_21 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_21->src_addr_lo = (uint32_t)ptr_l3_shared_act_a;
        args_dev_chip00_21->src_addr_hi = (uint32_t)(ptr_l3_shared_act_a >> 32);
        args_dev_chip00_21->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_21->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_21->size = 128;
        device_arg_list_chip_00[12] = (uint32_t)(uintptr_t)args_dev_chip00_21;
        device_kernel_list_chip_00[12] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 22 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_22 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_22->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B)));
        args_dev_chip00_22->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B))) >> 32);
        args_dev_chip00_22->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_22->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_22->size = 8192;
        device_arg_list_chip_00[13] = (uint32_t)(uintptr_t)args_dev_chip00_22;
        device_kernel_list_chip_00[13] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 23 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_23 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_23->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_23->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_23->input_C_addr = (uint32_t)0;
        args_dev_chip00_23->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_23->M = 1;
        args_dev_chip00_23->K = 16;
        args_dev_chip00_23->N = 1;
        args_dev_chip00_23->array_shape_idx = 1;
        args_dev_chip00_23->transpose_A = 0;
        args_dev_chip00_23->transpose_B = 0;
        args_dev_chip00_23->accumPrevC = 0;
        args_dev_chip00_23->silu_enable = 0;
        device_arg_list_chip_00[14] = (uint32_t)(uintptr_t)args_dev_chip00_23;
        device_kernel_list_chip_00[14] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 24 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_24 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_24->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_24->dst_addr = (uint64_t)ptr_l3_shared_down_out;
        args_host_chip00_24->size = 256;
        host_arg_list_chip_00[9] = (uint64_t)(uintptr_t)args_host_chip00_24;
        host_kernel_list_chip_00[9] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 25 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_25 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_25->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B + 8192)));
        args_dev_chip00_25->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B + 8192))) >> 32);
        args_dev_chip00_25->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_25->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_25->size = 8192;
        device_arg_list_chip_00[15] = (uint32_t)(uintptr_t)args_dev_chip00_25;
        device_kernel_list_chip_00[15] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 26 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_26 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_26->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_26->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_26->input_C_addr = (uint32_t)0;
        args_dev_chip00_26->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_26->silu_enable = 0;
        device_arg_list_chip_00[16] = (uint32_t)(uintptr_t)args_dev_chip00_26;
        device_kernel_list_chip_00[16] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 27 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_27 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_27->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_27->dst_addr = (uint64_t)ptr_l3_shared_down_out + 256;
        args_host_chip00_27->size = 256;
        host_arg_list_chip_00[10] = (uint64_t)(uintptr_t)args_host_chip00_27;
        host_kernel_list_chip_00[10] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 28 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_28 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_28->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_28->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)input_A))) >> 32);
        args_dev_chip00_28->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_28->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_28->size = 1024;
        device_arg_list_chip_00[17] = (uint32_t)(uintptr_t)args_dev_chip00_28;
        device_kernel_list_chip_00[17] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 29 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_29 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_29->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B + 131072)));
        args_dev_chip00_29->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B + 131072))) >> 32);
        args_dev_chip00_29->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_29->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_29->size = 65536;
        device_arg_list_chip_00[18] = (uint32_t)(uintptr_t)args_dev_chip00_29;
        device_kernel_list_chip_00[18] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 30 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_30 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_30->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_30->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_30->input_C_addr = (uint32_t)0;
        args_dev_chip00_30->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_30->M = 1;
        args_dev_chip00_30->K = 128;
        args_dev_chip00_30->N = 1;
        args_dev_chip00_30->array_shape_idx = 1;
        args_dev_chip00_30->transpose_A = 0;
        args_dev_chip00_30->transpose_B = 0;
        args_dev_chip00_30->accumPrevC = 0;
        args_dev_chip00_30->silu_enable = 1;
        device_arg_list_chip_00[19] = (uint32_t)(uintptr_t)args_dev_chip00_30;
        device_kernel_list_chip_00[19] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 31 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_31 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_31->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_31->dst_addr = (uint64_t)ptr_l3_shared_gate_out + 512;
        args_host_chip00_31->size = 256;
        host_arg_list_chip_00[11] = (uint64_t)(uintptr_t)args_host_chip00_31;
        host_kernel_list_chip_00[11] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 32 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_32 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_32->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B + 196608)));
        args_dev_chip00_32->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B + 196608))) >> 32);
        args_dev_chip00_32->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_32->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_32->size = 65536;
        device_arg_list_chip_00[20] = (uint32_t)(uintptr_t)args_dev_chip00_32;
        device_kernel_list_chip_00[20] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 33 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_33 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_33->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_33->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_33->input_C_addr = (uint32_t)0;
        args_dev_chip00_33->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_33->silu_enable = 1;
        device_arg_list_chip_00[21] = (uint32_t)(uintptr_t)args_dev_chip00_33;
        device_kernel_list_chip_00[21] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 34 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_34 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_34->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_34->dst_addr = (uint64_t)ptr_l3_shared_gate_out + 768;
        args_host_chip00_34->size = 256;
        host_arg_list_chip_00[12] = (uint64_t)(uintptr_t)args_host_chip00_34;
        host_kernel_list_chip_00[12] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 35 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_35 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_35->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B + 131072)));
        args_dev_chip00_35->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B + 131072))) >> 32);
        args_dev_chip00_35->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_35->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_35->size = 65536;
        device_arg_list_chip_00[22] = (uint32_t)(uintptr_t)args_dev_chip00_35;
        device_kernel_list_chip_00[22] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 36 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_36 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_36->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_36->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_36->input_C_addr = (uint32_t)0;
        args_dev_chip00_36->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_36->M = 1;
        args_dev_chip00_36->K = 128;
        args_dev_chip00_36->N = 1;
        args_dev_chip00_36->array_shape_idx = 1;
        args_dev_chip00_36->transpose_A = 0;
        args_dev_chip00_36->transpose_B = 0;
        args_dev_chip00_36->accumPrevC = 0;
        args_dev_chip00_36->silu_enable = 0;
        device_arg_list_chip_00[23] = (uint32_t)(uintptr_t)args_dev_chip00_36;
        device_kernel_list_chip_00[23] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 37 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_37 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_37->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_37->dst_addr = (uint64_t)ptr_l3_shared_up_out + 512;
        args_host_chip00_37->size = 256;
        host_arg_list_chip_00[13] = (uint64_t)(uintptr_t)args_host_chip00_37;
        host_kernel_list_chip_00[13] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 38 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_38 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_38->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B + 196608)));
        args_dev_chip00_38->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B + 196608))) >> 32);
        args_dev_chip00_38->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_38->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_38->size = 65536;
        device_arg_list_chip_00[24] = (uint32_t)(uintptr_t)args_dev_chip00_38;
        device_kernel_list_chip_00[24] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 39 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_39 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_39->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_39->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_39->input_C_addr = (uint32_t)0;
        args_dev_chip00_39->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_39->silu_enable = 0;
        device_arg_list_chip_00[25] = (uint32_t)(uintptr_t)args_dev_chip00_39;
        device_kernel_list_chip_00[25] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 40 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_40 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_40->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_40->dst_addr = (uint64_t)ptr_l3_shared_up_out + 768;
        args_host_chip00_40->size = 256;
        host_arg_list_chip_00[14] = (uint64_t)(uintptr_t)args_host_chip00_40;
        host_kernel_list_chip_00[14] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 41 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu (__host_bingo_kernel_compute_hw_silu_glu)
        __host_bingo_kernel_compute_hw_silu_glu_args_t* args_host_chip00_41 = (__host_bingo_kernel_compute_hw_silu_glu_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_compute_hw_silu_glu_args_t));
        args_host_chip00_41->gate_silu_hw_data_addr = (uint64_t)ptr_l3_shared_gate_out + 512;
        args_host_chip00_41->up_projection_hw_data_addr = (uint64_t)ptr_l3_shared_up_out + 512;
        args_host_chip00_41->activated_out_data_addr = (uint64_t)ptr_l3_shared_act_a + 128;
        args_host_chip00_41->valid_elements = 128;
        args_host_chip00_41->swish_glu_scale_in_raw = 931135488;
        args_host_chip00_41->swish_glu_scale_out_raw = 1107296256;
        host_arg_list_chip_00[15] = (uint64_t)(uintptr_t)args_host_chip00_41;
        host_kernel_list_chip_00[15] = (uint64_t)(uintptr_t)&__host_bingo_kernel_compute_hw_silu_glu;
        // Node ID: 42 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_42 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_42->src_addr_lo = (uint32_t)ptr_l3_shared_act_a + 128;
        args_dev_chip00_42->src_addr_hi = (uint32_t)(ptr_l3_shared_act_a + 128 >> 32);
        args_dev_chip00_42->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_42->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_42->size = 128;
        device_arg_list_chip_00[26] = (uint32_t)(uintptr_t)args_dev_chip00_42;
        device_kernel_list_chip_00[26] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 43 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_43 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_43->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B + 16384)));
        args_dev_chip00_43->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B + 16384))) >> 32);
        args_dev_chip00_43->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_43->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_43->size = 8192;
        device_arg_list_chip_00[27] = (uint32_t)(uintptr_t)args_dev_chip00_43;
        device_kernel_list_chip_00[27] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 44 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_44 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_44->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_44->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_44->input_C_addr = (uint32_t)0;
        args_dev_chip00_44->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_44->M = 1;
        args_dev_chip00_44->K = 16;
        args_dev_chip00_44->N = 1;
        args_dev_chip00_44->array_shape_idx = 1;
        args_dev_chip00_44->transpose_A = 0;
        args_dev_chip00_44->transpose_B = 0;
        args_dev_chip00_44->accumPrevC = 0;
        args_dev_chip00_44->silu_enable = 0;
        device_arg_list_chip_00[28] = (uint32_t)(uintptr_t)args_dev_chip00_44;
        device_kernel_list_chip_00[28] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 45 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_45 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_45->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_45->dst_addr = (uint64_t)ptr_l3_shared_down_out + 512;
        args_host_chip00_45->size = 256;
        host_arg_list_chip_00[16] = (uint64_t)(uintptr_t)args_host_chip00_45;
        host_kernel_list_chip_00[16] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 46 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_46 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_46->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B + 24576)));
        args_dev_chip00_46->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_down_projection_B + 24576))) >> 32);
        args_dev_chip00_46->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_46->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_46->size = 8192;
        device_arg_list_chip_00[29] = (uint32_t)(uintptr_t)args_dev_chip00_46;
        device_kernel_list_chip_00[29] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 47 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_47 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_47->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_47->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_47->input_C_addr = (uint32_t)0;
        args_dev_chip00_47->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_47->silu_enable = 0;
        device_arg_list_chip_00[30] = (uint32_t)(uintptr_t)args_dev_chip00_47;
        device_kernel_list_chip_00[30] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 48 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_48 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_48->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_48->dst_addr = (uint64_t)ptr_l3_shared_down_out + 768;
        args_host_chip00_48->size = 256;
        host_arg_list_chip_00[17] = (uint64_t)(uintptr_t)args_host_chip00_48;
        host_kernel_list_chip_00[17] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 49 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_scatter_and_pad_input (__host_bingo_kernel_scatter_and_pad_input)
        __host_bingo_kernel_scatter_and_pad_input_args_t* args_host_chip00_49 = (__host_bingo_kernel_scatter_and_pad_input_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_scatter_and_pad_input_args_t));
        args_host_chip00_49->expert_id = 0;
        args_host_chip00_49->global_input_A_addr = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_host_chip00_49->padded_scatter_pool_addr = (uint64_t)ptr_l1_buf_A;
        args_host_chip00_49->expert_token_counts_addr = (uint64_t)ptr_l3_exp_hist;
        args_host_chip00_49->expert_memory_offsets_addr = (uint64_t)ptr_l3_exp_offset;
        args_host_chip00_49->reverse_original_token_flat_idx_addr = (uint64_t)ptr_l3_rev_idx;
        args_host_chip00_49->input_dimension = 1024;
        args_host_chip00_49->max_padded_tokens = 1;
        args_host_chip00_49->individual_expert_number_k = 2;
        host_arg_list_chip_00[18] = (uint64_t)(uintptr_t)args_host_chip00_49;
        host_kernel_list_chip_00[18] = (uint64_t)(uintptr_t)&__host_bingo_kernel_scatter_and_pad_input;
        // Node ID: 50 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_50 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_50->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B)));
        args_dev_chip00_50->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B))) >> 32);
        args_dev_chip00_50->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_50->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_50->size = 65536;
        device_arg_list_chip_00[31] = (uint32_t)(uintptr_t)args_dev_chip00_50;
        device_kernel_list_chip_00[31] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 51 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_51 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_51->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_51->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_51->input_C_addr = (uint32_t)0;
        args_dev_chip00_51->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_51->M = 1;
        args_dev_chip00_51->K = 128;
        args_dev_chip00_51->N = 1;
        args_dev_chip00_51->array_shape_idx = 1;
        args_dev_chip00_51->transpose_A = 0;
        args_dev_chip00_51->transpose_B = 0;
        args_dev_chip00_51->accumPrevC = 0;
        args_dev_chip00_51->silu_enable = 1;
        device_arg_list_chip_00[32] = (uint32_t)(uintptr_t)args_dev_chip00_51;
        device_kernel_list_chip_00[32] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 52 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_52 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_52->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_52->dst_addr = (uint64_t)ptr_l3_indiv_gate_out;
        args_host_chip00_52->size = 256;
        host_arg_list_chip_00[19] = (uint64_t)(uintptr_t)args_host_chip00_52;
        host_kernel_list_chip_00[19] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 53 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_53 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_53->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B + 65536)));
        args_dev_chip00_53->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B + 65536))) >> 32);
        args_dev_chip00_53->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_53->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_53->size = 65536;
        device_arg_list_chip_00[33] = (uint32_t)(uintptr_t)args_dev_chip00_53;
        device_kernel_list_chip_00[33] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 54 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_54 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_54->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_54->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_54->input_C_addr = (uint32_t)0;
        args_dev_chip00_54->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_54->silu_enable = 1;
        device_arg_list_chip_00[34] = (uint32_t)(uintptr_t)args_dev_chip00_54;
        device_kernel_list_chip_00[34] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 55 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_55 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_55->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_55->dst_addr = (uint64_t)ptr_l3_indiv_gate_out + 256;
        args_host_chip00_55->size = 256;
        host_arg_list_chip_00[20] = (uint64_t)(uintptr_t)args_host_chip00_55;
        host_kernel_list_chip_00[20] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 56 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_56 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_56->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B)));
        args_dev_chip00_56->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B))) >> 32);
        args_dev_chip00_56->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_56->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_56->size = 65536;
        device_arg_list_chip_00[35] = (uint32_t)(uintptr_t)args_dev_chip00_56;
        device_kernel_list_chip_00[35] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 57 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_57 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_57->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_57->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_57->input_C_addr = (uint32_t)0;
        args_dev_chip00_57->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_57->M = 1;
        args_dev_chip00_57->K = 128;
        args_dev_chip00_57->N = 1;
        args_dev_chip00_57->array_shape_idx = 1;
        args_dev_chip00_57->transpose_A = 0;
        args_dev_chip00_57->transpose_B = 0;
        args_dev_chip00_57->accumPrevC = 0;
        args_dev_chip00_57->silu_enable = 0;
        device_arg_list_chip_00[36] = (uint32_t)(uintptr_t)args_dev_chip00_57;
        device_kernel_list_chip_00[36] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 58 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_58 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_58->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_58->dst_addr = (uint64_t)ptr_l3_indiv_up_out;
        args_host_chip00_58->size = 256;
        host_arg_list_chip_00[21] = (uint64_t)(uintptr_t)args_host_chip00_58;
        host_kernel_list_chip_00[21] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 59 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_59 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_59->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B + 65536)));
        args_dev_chip00_59->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B + 65536))) >> 32);
        args_dev_chip00_59->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_59->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_59->size = 65536;
        device_arg_list_chip_00[37] = (uint32_t)(uintptr_t)args_dev_chip00_59;
        device_kernel_list_chip_00[37] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 60 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_60 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_60->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_60->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_60->input_C_addr = (uint32_t)0;
        args_dev_chip00_60->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_60->silu_enable = 0;
        device_arg_list_chip_00[38] = (uint32_t)(uintptr_t)args_dev_chip00_60;
        device_kernel_list_chip_00[38] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 61 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_61 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_61->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_61->dst_addr = (uint64_t)ptr_l3_indiv_up_out + 256;
        args_host_chip00_61->size = 256;
        host_arg_list_chip_00[22] = (uint64_t)(uintptr_t)args_host_chip00_61;
        host_kernel_list_chip_00[22] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 62 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu (__host_bingo_kernel_compute_hw_silu_glu)
        __host_bingo_kernel_compute_hw_silu_glu_args_t* args_host_chip00_62 = (__host_bingo_kernel_compute_hw_silu_glu_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_compute_hw_silu_glu_args_t));
        args_host_chip00_62->gate_silu_hw_data_addr = (uint64_t)ptr_l3_indiv_gate_out;
        args_host_chip00_62->up_projection_hw_data_addr = (uint64_t)ptr_l3_indiv_up_out;
        args_host_chip00_62->activated_out_data_addr = (uint64_t)ptr_l3_indiv_act_a;
        args_host_chip00_62->valid_elements = 128;
        args_host_chip00_62->swish_glu_scale_in_raw = 931135488;
        args_host_chip00_62->swish_glu_scale_out_raw = 1107296256;
        host_arg_list_chip_00[23] = (uint64_t)(uintptr_t)args_host_chip00_62;
        host_kernel_list_chip_00[23] = (uint64_t)(uintptr_t)&__host_bingo_kernel_compute_hw_silu_glu;
        // Node ID: 63 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_63 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_63->src_addr_lo = (uint32_t)ptr_l3_indiv_act_a;
        args_dev_chip00_63->src_addr_hi = (uint32_t)(ptr_l3_indiv_act_a >> 32);
        args_dev_chip00_63->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_63->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_63->size = 128;
        device_arg_list_chip_00[39] = (uint32_t)(uintptr_t)args_dev_chip00_63;
        device_kernel_list_chip_00[39] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 64 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_64 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_64->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B)));
        args_dev_chip00_64->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B))) >> 32);
        args_dev_chip00_64->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_64->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_64->size = 8192;
        device_arg_list_chip_00[40] = (uint32_t)(uintptr_t)args_dev_chip00_64;
        device_kernel_list_chip_00[40] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 65 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_65 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_65->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_65->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_65->input_C_addr = (uint32_t)0;
        args_dev_chip00_65->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_65->M = 1;
        args_dev_chip00_65->K = 16;
        args_dev_chip00_65->N = 1;
        args_dev_chip00_65->array_shape_idx = 1;
        args_dev_chip00_65->transpose_A = 0;
        args_dev_chip00_65->transpose_B = 0;
        args_dev_chip00_65->accumPrevC = 0;
        args_dev_chip00_65->silu_enable = 0;
        device_arg_list_chip_00[41] = (uint32_t)(uintptr_t)args_dev_chip00_65;
        device_kernel_list_chip_00[41] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 66 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_66 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_66->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_66->dst_addr = (uint64_t)ptr_l3_indiv_down_out;
        args_host_chip00_66->size = 256;
        host_arg_list_chip_00[24] = (uint64_t)(uintptr_t)args_host_chip00_66;
        host_kernel_list_chip_00[24] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 67 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_67 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_67->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B + 8192)));
        args_dev_chip00_67->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B + 8192))) >> 32);
        args_dev_chip00_67->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_67->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_67->size = 8192;
        device_arg_list_chip_00[42] = (uint32_t)(uintptr_t)args_dev_chip00_67;
        device_kernel_list_chip_00[42] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 68 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_68 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_68->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_68->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_68->input_C_addr = (uint32_t)0;
        args_dev_chip00_68->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_68->silu_enable = 0;
        device_arg_list_chip_00[43] = (uint32_t)(uintptr_t)args_dev_chip00_68;
        device_kernel_list_chip_00[43] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 69 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_69 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_69->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_69->dst_addr = (uint64_t)ptr_l3_indiv_down_out + 256;
        args_host_chip00_69->size = 256;
        host_arg_list_chip_00[25] = (uint64_t)(uintptr_t)args_host_chip00_69;
        host_kernel_list_chip_00[25] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 70 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_scatter_and_pad_input (__host_bingo_kernel_scatter_and_pad_input)
        __host_bingo_kernel_scatter_and_pad_input_args_t* args_host_chip00_70 = (__host_bingo_kernel_scatter_and_pad_input_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_scatter_and_pad_input_args_t));
        args_host_chip00_70->expert_id = 1;
        args_host_chip00_70->global_input_A_addr = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_host_chip00_70->padded_scatter_pool_addr = (uint64_t)ptr_l1_buf_A;
        args_host_chip00_70->expert_token_counts_addr = (uint64_t)ptr_l3_exp_hist;
        args_host_chip00_70->expert_memory_offsets_addr = (uint64_t)ptr_l3_exp_offset;
        args_host_chip00_70->reverse_original_token_flat_idx_addr = (uint64_t)ptr_l3_rev_idx;
        args_host_chip00_70->input_dimension = 1024;
        args_host_chip00_70->max_padded_tokens = 1;
        args_host_chip00_70->individual_expert_number_k = 2;
        host_arg_list_chip_00[26] = (uint64_t)(uintptr_t)args_host_chip00_70;
        host_kernel_list_chip_00[26] = (uint64_t)(uintptr_t)&__host_bingo_kernel_scatter_and_pad_input;
        // Node ID: 71 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_71 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_71->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B + 131072)));
        args_dev_chip00_71->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B + 131072))) >> 32);
        args_dev_chip00_71->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_71->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_71->size = 65536;
        device_arg_list_chip_00[44] = (uint32_t)(uintptr_t)args_dev_chip00_71;
        device_kernel_list_chip_00[44] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 72 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_72 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_72->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_72->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_72->input_C_addr = (uint32_t)0;
        args_dev_chip00_72->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_72->M = 1;
        args_dev_chip00_72->K = 128;
        args_dev_chip00_72->N = 1;
        args_dev_chip00_72->array_shape_idx = 1;
        args_dev_chip00_72->transpose_A = 0;
        args_dev_chip00_72->transpose_B = 0;
        args_dev_chip00_72->accumPrevC = 0;
        args_dev_chip00_72->silu_enable = 1;
        device_arg_list_chip_00[45] = (uint32_t)(uintptr_t)args_dev_chip00_72;
        device_kernel_list_chip_00[45] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 73 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_73 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_73->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_73->dst_addr = (uint64_t)ptr_l3_indiv_gate_out + 512;
        args_host_chip00_73->size = 256;
        host_arg_list_chip_00[27] = (uint64_t)(uintptr_t)args_host_chip00_73;
        host_kernel_list_chip_00[27] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 74 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_74 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_74->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B + 196608)));
        args_dev_chip00_74->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_gate_B + 196608))) >> 32);
        args_dev_chip00_74->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_74->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_74->size = 65536;
        device_arg_list_chip_00[46] = (uint32_t)(uintptr_t)args_dev_chip00_74;
        device_kernel_list_chip_00[46] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 75 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_75 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_75->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_75->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_75->input_C_addr = (uint32_t)0;
        args_dev_chip00_75->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_75->silu_enable = 1;
        device_arg_list_chip_00[47] = (uint32_t)(uintptr_t)args_dev_chip00_75;
        device_kernel_list_chip_00[47] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 76 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_76 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_76->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_76->dst_addr = (uint64_t)ptr_l3_indiv_gate_out + 768;
        args_host_chip00_76->size = 256;
        host_arg_list_chip_00[28] = (uint64_t)(uintptr_t)args_host_chip00_76;
        host_kernel_list_chip_00[28] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 77 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_77 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_77->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B + 131072)));
        args_dev_chip00_77->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B + 131072))) >> 32);
        args_dev_chip00_77->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_77->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_77->size = 65536;
        device_arg_list_chip_00[48] = (uint32_t)(uintptr_t)args_dev_chip00_77;
        device_kernel_list_chip_00[48] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 78 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_78 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_78->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_78->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_78->input_C_addr = (uint32_t)0;
        args_dev_chip00_78->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_78->M = 1;
        args_dev_chip00_78->K = 128;
        args_dev_chip00_78->N = 1;
        args_dev_chip00_78->array_shape_idx = 1;
        args_dev_chip00_78->transpose_A = 0;
        args_dev_chip00_78->transpose_B = 0;
        args_dev_chip00_78->accumPrevC = 0;
        args_dev_chip00_78->silu_enable = 0;
        device_arg_list_chip_00[49] = (uint32_t)(uintptr_t)args_dev_chip00_78;
        device_kernel_list_chip_00[49] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 79 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_79 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_79->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_79->dst_addr = (uint64_t)ptr_l3_indiv_up_out + 512;
        args_host_chip00_79->size = 256;
        host_arg_list_chip_00[29] = (uint64_t)(uintptr_t)args_host_chip00_79;
        host_kernel_list_chip_00[29] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 80 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_80 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_80->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B + 196608)));
        args_dev_chip00_80->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_up_projection_B + 196608))) >> 32);
        args_dev_chip00_80->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_80->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_80->size = 65536;
        device_arg_list_chip_00[50] = (uint32_t)(uintptr_t)args_dev_chip00_80;
        device_kernel_list_chip_00[50] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 81 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_81 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_81->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_81->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_81->input_C_addr = (uint32_t)0;
        args_dev_chip00_81->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_81->silu_enable = 0;
        device_arg_list_chip_00[51] = (uint32_t)(uintptr_t)args_dev_chip00_81;
        device_kernel_list_chip_00[51] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 82 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_82 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_82->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_82->dst_addr = (uint64_t)ptr_l3_indiv_up_out + 768;
        args_host_chip00_82->size = 256;
        host_arg_list_chip_00[30] = (uint64_t)(uintptr_t)args_host_chip00_82;
        host_kernel_list_chip_00[30] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 83 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_compute_hw_silu_glu (__host_bingo_kernel_compute_hw_silu_glu)
        __host_bingo_kernel_compute_hw_silu_glu_args_t* args_host_chip00_83 = (__host_bingo_kernel_compute_hw_silu_glu_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_compute_hw_silu_glu_args_t));
        args_host_chip00_83->gate_silu_hw_data_addr = (uint64_t)ptr_l3_indiv_gate_out + 512;
        args_host_chip00_83->up_projection_hw_data_addr = (uint64_t)ptr_l3_indiv_up_out + 512;
        args_host_chip00_83->activated_out_data_addr = (uint64_t)ptr_l3_indiv_act_a + 128;
        args_host_chip00_83->valid_elements = 128;
        args_host_chip00_83->swish_glu_scale_in_raw = 931135488;
        args_host_chip00_83->swish_glu_scale_out_raw = 1107296256;
        host_arg_list_chip_00[31] = (uint64_t)(uintptr_t)args_host_chip00_83;
        host_kernel_list_chip_00[31] = (uint64_t)(uintptr_t)&__host_bingo_kernel_compute_hw_silu_glu;
        // Node ID: 84 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_84 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_84->src_addr_lo = (uint32_t)ptr_l3_indiv_act_a + 128;
        args_dev_chip00_84->src_addr_hi = (uint32_t)(ptr_l3_indiv_act_a + 128 >> 32);
        args_dev_chip00_84->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_84->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_84->size = 128;
        device_arg_list_chip_00[52] = (uint32_t)(uintptr_t)args_dev_chip00_84;
        device_kernel_list_chip_00[52] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 85 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_85 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_85->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B + 16384)));
        args_dev_chip00_85->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B + 16384))) >> 32);
        args_dev_chip00_85->dst_addr_lo = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_85->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_ping >> 32);
        args_dev_chip00_85->size = 8192;
        device_arg_list_chip_00[53] = (uint32_t)(uintptr_t)args_dev_chip00_85;
        device_kernel_list_chip_00[53] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 86 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_full (__snax_bingo_kernel_gemm_full)
        __snax_bingo_kernel_gemm_full_args_t* args_dev_chip00_86 = (__snax_bingo_kernel_gemm_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_full_args_t));
        args_dev_chip00_86->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_86->input_B_addr = (uint32_t)ptr_l1_buf_B_ping;
        args_dev_chip00_86->input_C_addr = (uint32_t)0;
        args_dev_chip00_86->output_D_addr = (uint32_t)ptr_l1_buf_D_ping;
        args_dev_chip00_86->M = 1;
        args_dev_chip00_86->K = 16;
        args_dev_chip00_86->N = 1;
        args_dev_chip00_86->array_shape_idx = 1;
        args_dev_chip00_86->transpose_A = 0;
        args_dev_chip00_86->transpose_B = 0;
        args_dev_chip00_86->accumPrevC = 0;
        args_dev_chip00_86->silu_enable = 0;
        device_arg_list_chip_00[54] = (uint32_t)(uintptr_t)args_dev_chip00_86;
        device_kernel_list_chip_00[54] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_full");
        // Node ID: 87 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_87 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_87->src_addr = (uint64_t)ptr_l1_buf_D_ping;
        args_host_chip00_87->dst_addr = (uint64_t)ptr_l3_indiv_down_out + 512;
        args_host_chip00_87->size = 256;
        host_arg_list_chip_00[32] = (uint64_t)(uintptr_t)args_host_chip00_87;
        host_kernel_list_chip_00[32] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 88 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_88 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_88->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B + 24576)));
        args_dev_chip00_88->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)individual_experts_down_projection_B + 24576))) >> 32);
        args_dev_chip00_88->dst_addr_lo = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_88->dst_addr_hi = (uint32_t)(ptr_l1_buf_B_pong >> 32);
        args_dev_chip00_88->size = 8192;
        device_arg_list_chip_00[55] = (uint32_t)(uintptr_t)args_dev_chip00_88;
        device_kernel_list_chip_00[55] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 89 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_gemm_minimal (__snax_bingo_kernel_gemm_minimal)
        __snax_bingo_kernel_gemm_minimal_args_t* args_dev_chip00_89 = (__snax_bingo_kernel_gemm_minimal_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_gemm_minimal_args_t));
        args_dev_chip00_89->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_89->input_B_addr = (uint32_t)ptr_l1_buf_B_pong;
        args_dev_chip00_89->input_C_addr = (uint32_t)0;
        args_dev_chip00_89->output_D_addr = (uint32_t)ptr_l1_buf_D_pong;
        args_dev_chip00_89->silu_enable = 0;
        device_arg_list_chip_00[56] = (uint32_t)(uintptr_t)args_dev_chip00_89;
        device_kernel_list_chip_00[56] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_gemm_minimal");
        // Node ID: 90 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_90 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_90->src_addr = (uint64_t)ptr_l1_buf_D_pong;
        args_host_chip00_90->dst_addr = (uint64_t)ptr_l3_indiv_down_out + 768;
        args_host_chip00_90->size = 256;
        host_arg_list_chip_00[33] = (uint64_t)(uintptr_t)args_host_chip00_90;
        host_kernel_list_chip_00[33] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 91 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_experts_result_accumulate (__host_bingo_kernel_experts_result_accumulate)
        __host_bingo_kernel_experts_result_accumulate_args_t* args_host_chip00_91 = (__host_bingo_kernel_experts_result_accumulate_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_experts_result_accumulate_args_t));
        args_host_chip00_91->shared_expert_hw_output_addr = (uint64_t)ptr_l3_shared_down_out;
        args_host_chip00_91->individual_experts_hw_output_addr = (uint64_t)ptr_l3_indiv_down_out;
        args_host_chip00_91->reverse_original_token_flat_idx_addr = (uint64_t)ptr_l3_rev_idx;
        args_host_chip00_91->global_calculated_probability_addr = (uint64_t)ptr_l3_router_prob;
        args_host_chip00_91->final_layer_output_addr = (uint64_t)ptr_l3_final_moe_out;
        args_host_chip00_91->actual_total_tokens = 1;
        args_host_chip00_91->input_dimension = 1024;
        args_host_chip00_91->individual_expert_number_k = 2;
        args_host_chip00_91->softmax_scale_raw = 1199570944;
        host_arg_list_chip_00[34] = (uint64_t)(uintptr_t)args_host_chip00_91;
        host_kernel_list_chip_00[34] = (uint64_t)(uintptr_t)&__host_bingo_kernel_experts_result_accumulate;
        // Node ID: 92 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry (__host_bingo_kernel_entry)
        host_arg_list_chip_00[35] = 0;
        host_kernel_list_chip_00[35] = (uint64_t)(uintptr_t)&__host_bingo_kernel_entry;
        // Node ID: 93 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_93 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_93->exit_code = 0;
        device_arg_list_chip_00[57] = (uint32_t)(uintptr_t)args_dev_chip00_93;
        device_kernel_list_chip_00[57] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 94 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_94 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_94->exit_code = 0;
        device_arg_list_chip_00[58] = (uint32_t)(uintptr_t)args_dev_chip00_94;
        device_kernel_list_chip_00[58] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 95 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit (__host_bingo_kernel_exit)
        __host_bingo_kernel_exit_args_t* args_host_chip00_95 = (__host_bingo_kernel_exit_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_exit_args_t));
        args_host_chip00_95->exit_code = 0;
        host_arg_list_chip_00[36] = (uint64_t)(uintptr_t)args_host_chip00_95;
        host_kernel_list_chip_00[36] = (uint64_t)(uintptr_t)&__host_bingo_kernel_exit;

        printf_safe("Chip(%x, %x): [Host] Init HW Bingo Scheduler\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());

        bingo_hw_scheduler_init((uint64_t)(uintptr_t)device_arg_list_chip_00,
                                (uint64_t)(uintptr_t)device_kernel_list_chip_00,
                                num_dev_tasks_chip_00,
                                (uint64_t)(uintptr_t)global_task_id_to_dev_task_id_chip_00,
                                num_total_tasks,
                                (uint64_t)(uintptr_t)bingo_hw_scheduler_task_desc_list_chip_00,
                                bingo_hw_scheduler_num_task_desc_chip_00);

        uint32_t err = bingo_hw_scheduler(host_arg_list_chip_00,
                                          host_kernel_list_chip_00,
                                          global_task_id_to_host_task_id_chip_00);
    }
    return 0;
}
