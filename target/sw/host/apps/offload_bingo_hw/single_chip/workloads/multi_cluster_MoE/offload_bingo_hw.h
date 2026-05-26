// Auto-generated offload_hw_bingo.h
#pragma once
#include "libbingo/bingo_api.h"
#include "MoE_operator.h"
#define MOE_OPERATOR_CUSTOM
#define MOE_ENABLE_DYNAMIC_BASELINE
#include "host.h"
#include "multi_cluster_MoE_data.h"

// Kernel Name List
// Note: This list is currently for debugging purposes only and is not used in the runtime.
// It will be enabled in the future.
/*
char kernel_name_list[151][64] = {
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 0
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 1
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 2
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 3
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 4
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 5
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 6
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 7
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 8
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 9
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 10
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 11
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 12
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 13
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 14
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 15
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 16
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 17
    "__snax_bingo_kernel_dual_vc_l15_moe_swiglu", // Node ID 18
    "__snax_bingo_kernel_dual_vc_l15_moe_down", // Node ID 19
    "__host_bingo_kernel_idma", // Node ID 20
    "__snax_bingo_kernel_dual_vc_l15_moe_swiglu", // Node ID 21
    "__snax_bingo_kernel_dual_vc_l15_moe_down", // Node ID 22
    "__host_bingo_kernel_idma", // Node ID 23
    "__snax_bingo_kernel_dual_vc_gemm_full", // Node ID 24
    "__host_bingo_kernel_idma", // Node ID 25
    "__host_bingo_kernel_moe_router_schedule", // Node ID 26
    "__host_bingo_kernel_moe_prepare_request", // Node ID 27
    "__host_bingo_kernel_moe_execute", // Node ID 28
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 29
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 30
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 31
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 32
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 33
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 34
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 35
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 36
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 37
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 38
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 39
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 40
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 41
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 42
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 43
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 44
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 45
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 46
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 47
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 48
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 49
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 50
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 51
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 52
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 53
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 54
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 55
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 56
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 57
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 58
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 59
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 60
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 61
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 62
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 63
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 64
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 65
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 66
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 67
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 68
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 69
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 70
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 71
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 72
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 73
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 74
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 75
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 76
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 77
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 78
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 79
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 80
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 81
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 82
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 83
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 84
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 85
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 86
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 87
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 88
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 89
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 90
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 91
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 92
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 93
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 94
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 95
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 96
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 97
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 98
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 99
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 100
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 101
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 102
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 103
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 104
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 105
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 106
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 107
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 108
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 109
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 110
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 111
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 112
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 113
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 114
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 115
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 116
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 117
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 118
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 119
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 120
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 121
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 122
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 123
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 124
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 125
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 126
    "__snax_bingo_kernel_moe_dynamic_expert_gather_s1", // Node ID 127
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 128
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 129
    "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block", // Node ID 130
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block", // Node ID 131
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down", // Node ID 132
    "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full", // Node ID 133
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 134
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 135
    "__snax_bingo_kernel_moe_dynamic_expert_load_down_block", // Node ID 136
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block", // Node ID 137
    "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1", // Node ID 138
    "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full", // Node ID 139
    "__snax_bingo_kernel_moe_dynamic_expert_store", // Node ID 140
    "__host_bingo_kernel_entry", // Node ID 141
    "__snax_bingo_kernel_exit", // Node ID 142
    "__snax_bingo_kernel_exit", // Node ID 143
    "__snax_bingo_kernel_exit", // Node ID 144
    "__snax_bingo_kernel_exit", // Node ID 145
    "__snax_bingo_kernel_exit", // Node ID 146
    "__snax_bingo_kernel_exit", // Node ID 147
    "__snax_bingo_kernel_exit", // Node ID 148
    "__snax_bingo_kernel_exit", // Node ID 149
    "__host_bingo_kernel_exit", // Node ID 150
};
*/

int kernel_execution(){
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing multi_cluster_MoE Workload\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    uint32_t current_chip_id = get_current_chip_id();
    if (current_chip_id == 0x00) {
        uint32_t num_total_tasks = 295;
        // Task Description List
        uint32_t bingo_hw_scheduler_num_task_desc_chip_00 = 295;
        uint64_t* bingo_hw_scheduler_task_desc_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, bingo_hw_scheduler_num_task_desc_chip_00 * sizeof(uint64_t));
        bingo_hw_scheduler_task_desc_list_chip_00[0] = 0x0004002100011A00; // Node ID 141
            // Fields: Type=0, TaskID=141
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[1] = 0x000400210001BC80; // Node ID 222
            // Fields: Type=1, TaskID=222
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[2] = 0x0004003280000000; // Node ID 0
            // Fields: Type=0, TaskID=0
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[3] = 0x0004802080012E80; // Node ID 151
            // Fields: Type=1, TaskID=151
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[4] = 0x0000000A8001BE80; // Node ID 223
            // Fields: Type=1, TaskID=223
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[5] = 0x0004802AA0000200; // Node ID 1
            // Fields: Type=0, TaskID=1
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[6] = 0x0004003280000A00; // Node ID 5
            // Fields: Type=0, TaskID=5
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[7] = 0x00050020A0013080; // Node ID 152
            // Fields: Type=1, TaskID=152
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[8] = 0x0000000AA001C080; // Node ID 224
            // Fields: Type=1, TaskID=224
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[9] = 0x0004002A80001C00; // Node ID 14
            // Fields: Type=0, TaskID=14
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[10] = 0x0004802080013880; // Node ID 156
            // Fields: Type=1, TaskID=156
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[11] = 0x0005002AC0000400; // Node ID 2
            // Fields: Type=0, TaskID=2
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[12] = 0x0002002A80001E00; // Node ID 15
            // Fields: Type=0, TaskID=15
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[13] = 0x0004802AA0000C00; // Node ID 6
            // Fields: Type=0, TaskID=6
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[14] = 0x00058020C0013280; // Node ID 153
            // Fields: Type=1, TaskID=153
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[15] = 0x0000000AC001C280; // Node ID 225
            // Fields: Type=1, TaskID=225
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[16] = 0x0004002080014080; // Node ID 160
            // Fields: Type=1, TaskID=160
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[17] = 0x0004802AA0002000; // Node ID 16
            // Fields: Type=0, TaskID=16
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[18] = 0x00050020A0013A80; // Node ID 157
            // Fields: Type=1, TaskID=157
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[19] = 0x0005802AE0000600; // Node ID 3
            // Fields: Type=0, TaskID=3
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[20] = 0x0002802AA0002200; // Node ID 17
            // Fields: Type=0, TaskID=17
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[21] = 0x0005002AC0000E00; // Node ID 7
            // Fields: Type=0, TaskID=7
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[22] = 0x0005802AE0000800; // Node ID 4
            // Fields: Type=0, TaskID=4
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[23] = 0x00048020A0014280; // Node ID 161
            // Fields: Type=1, TaskID=161
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[24] = 0x0005802AC0001000; // Node ID 8
            // Fields: Type=0, TaskID=8
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[25] = 0x00040020E0013480; // Node ID 154
            // Fields: Type=1, TaskID=154
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[26] = 0x00048020E0013680; // Node ID 155
            // Fields: Type=1, TaskID=155
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[27] = 0x0000000AE001C480; // Node ID 226
            // Fields: Type=1, TaskID=226
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[28] = 0x0000000A8001C680; // Node ID 227
            // Fields: Type=1, TaskID=227
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[29] = 0x0000000AA001CC80; // Node ID 230
            // Fields: Type=1, TaskID=230
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[30] = 0x0005802AE0001200; // Node ID 9
            // Fields: Type=0, TaskID=9
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[31] = 0x0004802AE0001400; // Node ID 10
            // Fields: Type=0, TaskID=10
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[32] = 0x00040020E0013C80; // Node ID 158
            // Fields: Type=1, TaskID=158
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[33] = 0x0000000AA001CA80; // Node ID 229
            // Fields: Type=1, TaskID=229
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[34] = 0x0000000A8001C880; // Node ID 228
            // Fields: Type=1, TaskID=228
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[35] = 0x0002802AA0001800; // Node ID 12
            // Fields: Type=0, TaskID=12
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[36] = 0x0002002A80001600; // Node ID 11
            // Fields: Type=0, TaskID=11
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[37] = 0x0002802A20002A00; // Node ID 21
            // Fields: Type=0, TaskID=21
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[38] = 0x0002002A00002400; // Node ID 18
            // Fields: Type=0, TaskID=18
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[39] = 0x0005802080013E80; // Node ID 159
            // Fields: Type=1, TaskID=159
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[40] = 0x000000062001D280; // Node ID 233
            // Fields: Type=1, TaskID=233
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[41] = 0x000000060001CE80; // Node ID 231
            // Fields: Type=1, TaskID=231
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[42] = 0x0003802AE0001A00; // Node ID 13
            // Fields: Type=0, TaskID=13
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[43] = 0x0008002A20002C00; // Node ID 22
            // Fields: Type=0, TaskID=22
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[44] = 0x0008002A00002600; // Node ID 19
            // Fields: Type=0, TaskID=19
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[45] = 0x0008002A60003000; // Node ID 24
            // Fields: Type=0, TaskID=24
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[46] = 0x000000070001D480; // Node ID 234
            // Fields: Type=1, TaskID=234
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[47] = 0x000000070001D080; // Node ID 232
            // Fields: Type=1, TaskID=232
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[48] = 0x0008002700003200; // Node ID 25
            // Fields: Type=0, TaskID=25
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[49] = 0x0002003300002800; // Node ID 20
            // Fields: Type=0, TaskID=20
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[50] = 0x0008002100014880; // Node ID 164
            // Fields: Type=1, TaskID=164
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[51] = 0x0008002100014480; // Node ID 162
            // Fields: Type=1, TaskID=162
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[52] = 0x0000001200024880; // Node ID 292
            // Fields: Type=1, TaskID=292
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[53] = 0x0002003300002E00; // Node ID 23
            // Fields: Type=0, TaskID=23
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[54] = 0x0008002100014680; // Node ID 163
            // Fields: Type=1, TaskID=163
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[55] = 0x000000130001D680; // Node ID 235
            // Fields: Type=1, TaskID=235
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[56] = 0x0008003300003400; // Node ID 26
            // Fields: Type=0, TaskID=26
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[57] = 0x0008003300003600; // Node ID 27
            // Fields: Type=0, TaskID=27
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[58] = 0x0005803300003800; // Node ID 28
            // Fields: Type=0, TaskID=28
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[59] = 0x00058032E0005600; // Node ID 43
            // Fields: Type=0, TaskID=43
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[60] = 0x0005002100014A80; // Node ID 165
            // Fields: Type=1, TaskID=165
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[61] = 0x00058020E0015A80; // Node ID 173
            // Fields: Type=1, TaskID=173
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[62] = 0x0000000AE001E680; // Node ID 243
            // Fields: Type=1, TaskID=243
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[63] = 0x00050032C0003A00; // Node ID 29
            // Fields: Type=0, TaskID=29
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[64] = 0x0003802AE0005800; // Node ID 44
            // Fields: Type=0, TaskID=44
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[65] = 0x00050020C0014C80; // Node ID 166
            // Fields: Type=1, TaskID=166
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[66] = 0x0000000AC001D880; // Node ID 236
            // Fields: Type=1, TaskID=236
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[67] = 0x0003802A60005A00; // Node ID 45
            // Fields: Type=0, TaskID=45
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[68] = 0x00058020E0015C80; // Node ID 174
            // Fields: Type=1, TaskID=174
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[69] = 0x0003002AC0003C00; // Node ID 30
            // Fields: Type=0, TaskID=30
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[70] = 0x000000066001E880; // Node ID 244
            // Fields: Type=1, TaskID=244
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[71] = 0x0003802AE0005C00; // Node ID 46
            // Fields: Type=0, TaskID=46
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[72] = 0x0003002A40003E00; // Node ID 31
            // Fields: Type=0, TaskID=31
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[73] = 0x00050020C0014E80; // Node ID 167
            // Fields: Type=1, TaskID=167
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[74] = 0x0003802A60005E00; // Node ID 47
            // Fields: Type=0, TaskID=47
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[75] = 0x00058020E0015E80; // Node ID 175
            // Fields: Type=1, TaskID=175
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[76] = 0x000000064001DA80; // Node ID 237
            // Fields: Type=1, TaskID=237
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[77] = 0x0003002AC0004000; // Node ID 32
            // Fields: Type=0, TaskID=32
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[78] = 0x0005802660006200; // Node ID 49
            // Fields: Type=0, TaskID=49
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[79] = 0x0005802AE0006000; // Node ID 48
            // Fields: Type=0, TaskID=48
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[80] = 0x0003002A40004200; // Node ID 33
            // Fields: Type=0, TaskID=33
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[81] = 0x00050020C0015080; // Node ID 168
            // Fields: Type=1, TaskID=168
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[82] = 0x0005802060016280; // Node ID 177
            // Fields: Type=1, TaskID=177
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[83] = 0x00000006E001EE80; // Node ID 247
            // Fields: Type=1, TaskID=247
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[84] = 0x00058020E0016080; // Node ID 176
            // Fields: Type=1, TaskID=176
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[85] = 0x0000000AE001EC80; // Node ID 246
            // Fields: Type=1, TaskID=246
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[86] = 0x0005002640004600; // Node ID 35
            // Fields: Type=0, TaskID=35
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[87] = 0x0005002AC0004400; // Node ID 34
            // Fields: Type=0, TaskID=34
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[88] = 0x00000006E001EA80; // Node ID 245
            // Fields: Type=1, TaskID=245
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[89] = 0x0005002040015480; // Node ID 170
            // Fields: Type=1, TaskID=170
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[90] = 0x00000006C001E080; // Node ID 240
            // Fields: Type=1, TaskID=240
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[91] = 0x00050020C0015280; // Node ID 169
            // Fields: Type=1, TaskID=169
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[92] = 0x0000000AC001DE80; // Node ID 239
            // Fields: Type=1, TaskID=239
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[93] = 0x0003802AE0006400; // Node ID 50
            // Fields: Type=0, TaskID=50
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[94] = 0x00000006C001DC80; // Node ID 238
            // Fields: Type=1, TaskID=238
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[95] = 0x0003802A60006600; // Node ID 51
            // Fields: Type=0, TaskID=51
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[96] = 0x00058020E0016480; // Node ID 178
            // Fields: Type=1, TaskID=178
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[97] = 0x0003002AC0004800; // Node ID 36
            // Fields: Type=0, TaskID=36
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[98] = 0x000000066001F080; // Node ID 248
            // Fields: Type=1, TaskID=248
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[99] = 0x0003802AE0006800; // Node ID 52
            // Fields: Type=0, TaskID=52
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[100] = 0x0003002A40004A00; // Node ID 37
            // Fields: Type=0, TaskID=37
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[101] = 0x00050020C0015680; // Node ID 171
            // Fields: Type=1, TaskID=171
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[102] = 0x0003802A60006A00; // Node ID 53
            // Fields: Type=0, TaskID=53
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[103] = 0x00058020E0016680; // Node ID 179
            // Fields: Type=1, TaskID=179
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[104] = 0x000000064001E280; // Node ID 241
            // Fields: Type=1, TaskID=241
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[105] = 0x0003002AC0004C00; // Node ID 38
            // Fields: Type=0, TaskID=38
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[106] = 0x0005802660006E00; // Node ID 55
            // Fields: Type=0, TaskID=55
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[107] = 0x0005802AE0006C00; // Node ID 54
            // Fields: Type=0, TaskID=54
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[108] = 0x0003002A40004E00; // Node ID 39
            // Fields: Type=0, TaskID=39
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[109] = 0x00050020C0015880; // Node ID 172
            // Fields: Type=1, TaskID=172
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[110] = 0x00000006E001F280; // Node ID 249
            // Fields: Type=1, TaskID=249
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[111] = 0x0005002640005200; // Node ID 41
            // Fields: Type=0, TaskID=41
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[112] = 0x0005002AC0005000; // Node ID 40
            // Fields: Type=0, TaskID=40
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[113] = 0x0005802AE0007000; // Node ID 56
            // Fields: Type=0, TaskID=56
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[114] = 0x00000006C001E480; // Node ID 242
            // Fields: Type=1, TaskID=242
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[115] = 0x0005802AE0008E00; // Node ID 71
            // Fields: Type=0, TaskID=71
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[116] = 0x0005002AC0005400; // Node ID 42
            // Fields: Type=0, TaskID=42
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[117] = 0x00058020E0017680; // Node ID 187
            // Fields: Type=1, TaskID=187
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[118] = 0x0000000AE0020280; // Node ID 257
            // Fields: Type=1, TaskID=257
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[119] = 0x0005002AC0007200; // Node ID 57
            // Fields: Type=0, TaskID=57
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[120] = 0x0003802AE0009000; // Node ID 72
            // Fields: Type=0, TaskID=72
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[121] = 0x00050020C0016880; // Node ID 180
            // Fields: Type=1, TaskID=180
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[122] = 0x0000000AC001F480; // Node ID 250
            // Fields: Type=1, TaskID=250
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[123] = 0x0003802A60009200; // Node ID 73
            // Fields: Type=0, TaskID=73
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[124] = 0x00058020E0017880; // Node ID 188
            // Fields: Type=1, TaskID=188
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[125] = 0x0003002AC0007400; // Node ID 58
            // Fields: Type=0, TaskID=58
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[126] = 0x0000000660020480; // Node ID 258
            // Fields: Type=1, TaskID=258
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[127] = 0x0003802AE0009400; // Node ID 74
            // Fields: Type=0, TaskID=74
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[128] = 0x0003002A40007600; // Node ID 59
            // Fields: Type=0, TaskID=59
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[129] = 0x00050020C0016A80; // Node ID 181
            // Fields: Type=1, TaskID=181
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[130] = 0x0003802A60009600; // Node ID 75
            // Fields: Type=0, TaskID=75
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[131] = 0x00058020E0017A80; // Node ID 189
            // Fields: Type=1, TaskID=189
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[132] = 0x000000064001F680; // Node ID 251
            // Fields: Type=1, TaskID=251
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[133] = 0x0003002AC0007800; // Node ID 60
            // Fields: Type=0, TaskID=60
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[134] = 0x0005802660009A00; // Node ID 77
            // Fields: Type=0, TaskID=77
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[135] = 0x0005802AE0009800; // Node ID 76
            // Fields: Type=0, TaskID=76
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[136] = 0x0003002A40007A00; // Node ID 61
            // Fields: Type=0, TaskID=61
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[137] = 0x00050020C0016C80; // Node ID 182
            // Fields: Type=1, TaskID=182
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[138] = 0x0005802060017E80; // Node ID 191
            // Fields: Type=1, TaskID=191
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[139] = 0x00000006E0020A80; // Node ID 261
            // Fields: Type=1, TaskID=261
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[140] = 0x00058020E0017C80; // Node ID 190
            // Fields: Type=1, TaskID=190
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[141] = 0x0000000AE0020880; // Node ID 260
            // Fields: Type=1, TaskID=260
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[142] = 0x0005002640007E00; // Node ID 63
            // Fields: Type=0, TaskID=63
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[143] = 0x0005002AC0007C00; // Node ID 62
            // Fields: Type=0, TaskID=62
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[144] = 0x00000006E0020680; // Node ID 259
            // Fields: Type=1, TaskID=259
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[145] = 0x0005002040017080; // Node ID 184
            // Fields: Type=1, TaskID=184
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[146] = 0x00000006C001FC80; // Node ID 254
            // Fields: Type=1, TaskID=254
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[147] = 0x00050020C0016E80; // Node ID 183
            // Fields: Type=1, TaskID=183
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[148] = 0x0000000AC001FA80; // Node ID 253
            // Fields: Type=1, TaskID=253
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[149] = 0x0003802AE0009C00; // Node ID 78
            // Fields: Type=0, TaskID=78
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[150] = 0x00000006C001F880; // Node ID 252
            // Fields: Type=1, TaskID=252
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[151] = 0x0003802A60009E00; // Node ID 79
            // Fields: Type=0, TaskID=79
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[152] = 0x00058020E0018080; // Node ID 192
            // Fields: Type=1, TaskID=192
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[153] = 0x0003002AC0008000; // Node ID 64
            // Fields: Type=0, TaskID=64
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[154] = 0x0000000660020C80; // Node ID 262
            // Fields: Type=1, TaskID=262
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[155] = 0x0003802AE000A000; // Node ID 80
            // Fields: Type=0, TaskID=80
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[156] = 0x0003002A40008200; // Node ID 65
            // Fields: Type=0, TaskID=65
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[157] = 0x00050020C0017280; // Node ID 185
            // Fields: Type=1, TaskID=185
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[158] = 0x0003802A6000A200; // Node ID 81
            // Fields: Type=0, TaskID=81
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[159] = 0x00058020E0018280; // Node ID 193
            // Fields: Type=1, TaskID=193
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[160] = 0x000000064001FE80; // Node ID 255
            // Fields: Type=1, TaskID=255
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[161] = 0x0003002AC0008400; // Node ID 66
            // Fields: Type=0, TaskID=66
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[162] = 0x000580266000A600; // Node ID 83
            // Fields: Type=0, TaskID=83
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[163] = 0x0005802AE000A400; // Node ID 82
            // Fields: Type=0, TaskID=82
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[164] = 0x0003002A40008600; // Node ID 67
            // Fields: Type=0, TaskID=67
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[165] = 0x00050020C0017480; // Node ID 186
            // Fields: Type=1, TaskID=186
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[166] = 0x00000006E0020E80; // Node ID 263
            // Fields: Type=1, TaskID=263
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[167] = 0x0005002640008A00; // Node ID 69
            // Fields: Type=0, TaskID=69
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[168] = 0x0005002AC0008800; // Node ID 68
            // Fields: Type=0, TaskID=68
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[169] = 0x0005802AE000A800; // Node ID 84
            // Fields: Type=0, TaskID=84
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[170] = 0x00000006C0020080; // Node ID 256
            // Fields: Type=1, TaskID=256
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[171] = 0x0005802AE000C600; // Node ID 99
            // Fields: Type=0, TaskID=99
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[172] = 0x0005002AC0008C00; // Node ID 70
            // Fields: Type=0, TaskID=70
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[173] = 0x00058020E0019280; // Node ID 201
            // Fields: Type=1, TaskID=201
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[174] = 0x0000000AE0021E80; // Node ID 271
            // Fields: Type=1, TaskID=271
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[175] = 0x0005002AC000AA00; // Node ID 85
            // Fields: Type=0, TaskID=85
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[176] = 0x0003802AE000C800; // Node ID 100
            // Fields: Type=0, TaskID=100
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[177] = 0x00050020C0018480; // Node ID 194
            // Fields: Type=1, TaskID=194
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[178] = 0x0000000AC0021080; // Node ID 264
            // Fields: Type=1, TaskID=264
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[179] = 0x0003802A6000CA00; // Node ID 101
            // Fields: Type=0, TaskID=101
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[180] = 0x00058020E0019480; // Node ID 202
            // Fields: Type=1, TaskID=202
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[181] = 0x0003002AC000AC00; // Node ID 86
            // Fields: Type=0, TaskID=86
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[182] = 0x0000000660022080; // Node ID 272
            // Fields: Type=1, TaskID=272
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[183] = 0x0003802AE000CC00; // Node ID 102
            // Fields: Type=0, TaskID=102
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[184] = 0x0003002A4000AE00; // Node ID 87
            // Fields: Type=0, TaskID=87
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[185] = 0x00050020C0018680; // Node ID 195
            // Fields: Type=1, TaskID=195
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[186] = 0x0003802A6000CE00; // Node ID 103
            // Fields: Type=0, TaskID=103
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[187] = 0x00058020E0019680; // Node ID 203
            // Fields: Type=1, TaskID=203
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[188] = 0x0000000640021280; // Node ID 265
            // Fields: Type=1, TaskID=265
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[189] = 0x0003002AC000B000; // Node ID 88
            // Fields: Type=0, TaskID=88
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[190] = 0x000580266000D200; // Node ID 105
            // Fields: Type=0, TaskID=105
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[191] = 0x0005802AE000D000; // Node ID 104
            // Fields: Type=0, TaskID=104
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[192] = 0x0003002A4000B200; // Node ID 89
            // Fields: Type=0, TaskID=89
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[193] = 0x00050020C0018880; // Node ID 196
            // Fields: Type=1, TaskID=196
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[194] = 0x0005802060019A80; // Node ID 205
            // Fields: Type=1, TaskID=205
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[195] = 0x00000006E0022680; // Node ID 275
            // Fields: Type=1, TaskID=275
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[196] = 0x00058020E0019880; // Node ID 204
            // Fields: Type=1, TaskID=204
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[197] = 0x0000000AE0022480; // Node ID 274
            // Fields: Type=1, TaskID=274
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[198] = 0x000500264000B600; // Node ID 91
            // Fields: Type=0, TaskID=91
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[199] = 0x0005002AC000B400; // Node ID 90
            // Fields: Type=0, TaskID=90
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[200] = 0x00000006E0022280; // Node ID 273
            // Fields: Type=1, TaskID=273
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[201] = 0x0005002040018C80; // Node ID 198
            // Fields: Type=1, TaskID=198
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[202] = 0x00000006C0021880; // Node ID 268
            // Fields: Type=1, TaskID=268
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[203] = 0x00050020C0018A80; // Node ID 197
            // Fields: Type=1, TaskID=197
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[204] = 0x0000000AC0021680; // Node ID 267
            // Fields: Type=1, TaskID=267
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[205] = 0x0003802AE000D400; // Node ID 106
            // Fields: Type=0, TaskID=106
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[206] = 0x00000006C0021480; // Node ID 266
            // Fields: Type=1, TaskID=266
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[207] = 0x0003802A6000D600; // Node ID 107
            // Fields: Type=0, TaskID=107
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[208] = 0x00058020E0019C80; // Node ID 206
            // Fields: Type=1, TaskID=206
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[209] = 0x0003002AC000B800; // Node ID 92
            // Fields: Type=0, TaskID=92
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[210] = 0x0000000660022880; // Node ID 276
            // Fields: Type=1, TaskID=276
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[211] = 0x0003802AE000D800; // Node ID 108
            // Fields: Type=0, TaskID=108
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[212] = 0x0003002A4000BA00; // Node ID 93
            // Fields: Type=0, TaskID=93
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[213] = 0x00050020C0018E80; // Node ID 199
            // Fields: Type=1, TaskID=199
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[214] = 0x0003802A6000DA00; // Node ID 109
            // Fields: Type=0, TaskID=109
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[215] = 0x00058020E0019E80; // Node ID 207
            // Fields: Type=1, TaskID=207
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[216] = 0x0000000640021A80; // Node ID 269
            // Fields: Type=1, TaskID=269
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[217] = 0x0003002AC000BC00; // Node ID 94
            // Fields: Type=0, TaskID=94
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[218] = 0x000580266000DE00; // Node ID 111
            // Fields: Type=0, TaskID=111
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[219] = 0x0005802AE000DC00; // Node ID 110
            // Fields: Type=0, TaskID=110
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[220] = 0x0003002A4000BE00; // Node ID 95
            // Fields: Type=0, TaskID=95
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[221] = 0x00050020C0019080; // Node ID 200
            // Fields: Type=1, TaskID=200
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[222] = 0x00000006E0022A80; // Node ID 277
            // Fields: Type=1, TaskID=277
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[223] = 0x000500264000C200; // Node ID 97
            // Fields: Type=0, TaskID=97
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[224] = 0x0005002AC000C000; // Node ID 96
            // Fields: Type=0, TaskID=96
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[225] = 0x0005802AE000E000; // Node ID 112
            // Fields: Type=0, TaskID=112
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[226] = 0x00000006C0021C80; // Node ID 270
            // Fields: Type=1, TaskID=270
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[227] = 0x0005802AE000FE00; // Node ID 127
            // Fields: Type=0, TaskID=127
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[228] = 0x0005002AC000C400; // Node ID 98
            // Fields: Type=0, TaskID=98
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[229] = 0x00058020E001AE80; // Node ID 215
            // Fields: Type=1, TaskID=215
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[230] = 0x0000000AE0023A80; // Node ID 285
            // Fields: Type=1, TaskID=285
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[231] = 0x0005002AC000E200; // Node ID 113
            // Fields: Type=0, TaskID=113
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[232] = 0x0003802AE0010000; // Node ID 128
            // Fields: Type=0, TaskID=128
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[233] = 0x00050020C001A080; // Node ID 208
            // Fields: Type=1, TaskID=208
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[234] = 0x0000000AC0022C80; // Node ID 278
            // Fields: Type=1, TaskID=278
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[235] = 0x0003802A60010200; // Node ID 129
            // Fields: Type=0, TaskID=129
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[236] = 0x00058020E001B080; // Node ID 216
            // Fields: Type=1, TaskID=216
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[237] = 0x0003002AC000E400; // Node ID 114
            // Fields: Type=0, TaskID=114
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[238] = 0x0000000660023C80; // Node ID 286
            // Fields: Type=1, TaskID=286
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[239] = 0x0003802AE0010400; // Node ID 130
            // Fields: Type=0, TaskID=130
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[240] = 0x0003002A4000E600; // Node ID 115
            // Fields: Type=0, TaskID=115
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[241] = 0x00050020C001A280; // Node ID 209
            // Fields: Type=1, TaskID=209
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[242] = 0x0003802A60010600; // Node ID 131
            // Fields: Type=0, TaskID=131
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[243] = 0x00058020E001B280; // Node ID 217
            // Fields: Type=1, TaskID=217
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[244] = 0x0000000640022E80; // Node ID 279
            // Fields: Type=1, TaskID=279
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[245] = 0x0003002AC000E800; // Node ID 116
            // Fields: Type=0, TaskID=116
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[246] = 0x0005802660010A00; // Node ID 133
            // Fields: Type=0, TaskID=133
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[247] = 0x0005802AE0010800; // Node ID 132
            // Fields: Type=0, TaskID=132
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[248] = 0x0003002A4000EA00; // Node ID 117
            // Fields: Type=0, TaskID=117
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[249] = 0x00050020C001A480; // Node ID 210
            // Fields: Type=1, TaskID=210
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[250] = 0x000580206001B680; // Node ID 219
            // Fields: Type=1, TaskID=219
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[251] = 0x00000006E0024280; // Node ID 289
            // Fields: Type=1, TaskID=289
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[252] = 0x00058020E001B480; // Node ID 218
            // Fields: Type=1, TaskID=218
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[253] = 0x0000000AE0024080; // Node ID 288
            // Fields: Type=1, TaskID=288
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[254] = 0x000500264000EE00; // Node ID 119
            // Fields: Type=0, TaskID=119
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[255] = 0x0005002AC000EC00; // Node ID 118
            // Fields: Type=0, TaskID=118
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[256] = 0x00000006E0023E80; // Node ID 287
            // Fields: Type=1, TaskID=287
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[257] = 0x000500204001A880; // Node ID 212
            // Fields: Type=1, TaskID=212
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[258] = 0x00000006C0023480; // Node ID 282
            // Fields: Type=1, TaskID=282
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[259] = 0x00050020C001A680; // Node ID 211
            // Fields: Type=1, TaskID=211
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[260] = 0x0000000AC0023280; // Node ID 281
            // Fields: Type=1, TaskID=281
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[261] = 0x0003802AE0010C00; // Node ID 134
            // Fields: Type=0, TaskID=134
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[262] = 0x00000006C0023080; // Node ID 280
            // Fields: Type=1, TaskID=280
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[263] = 0x0003802A60010E00; // Node ID 135
            // Fields: Type=0, TaskID=135
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[264] = 0x00058020E001B880; // Node ID 220
            // Fields: Type=1, TaskID=220
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[265] = 0x0003002AC000F000; // Node ID 120
            // Fields: Type=0, TaskID=120
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[266] = 0x0000000660024480; // Node ID 290
            // Fields: Type=1, TaskID=290
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[267] = 0x0003802AE0011000; // Node ID 136
            // Fields: Type=0, TaskID=136
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[268] = 0x0003002A4000F200; // Node ID 121
            // Fields: Type=0, TaskID=121
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[269] = 0x00050020C001AA80; // Node ID 213
            // Fields: Type=1, TaskID=213
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[270] = 0x0003802A60011200; // Node ID 137
            // Fields: Type=0, TaskID=137
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[271] = 0x00058020E001BA80; // Node ID 221
            // Fields: Type=1, TaskID=221
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[272] = 0x0000000640023680; // Node ID 283
            // Fields: Type=1, TaskID=283
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[273] = 0x0003002AC000F400; // Node ID 122
            // Fields: Type=0, TaskID=122
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[274] = 0x0005802660011600; // Node ID 139
            // Fields: Type=0, TaskID=139
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[275] = 0x0005802AE0011400; // Node ID 138
            // Fields: Type=0, TaskID=138
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[276] = 0x0003002A4000F600; // Node ID 123
            // Fields: Type=0, TaskID=123
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[277] = 0x00050020C001AC80; // Node ID 214
            // Fields: Type=1, TaskID=214
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[278] = 0x00000006E0024680; // Node ID 291
            // Fields: Type=1, TaskID=291
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[279] = 0x000500264000FA00; // Node ID 125
            // Fields: Type=0, TaskID=125
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[280] = 0x0005002AC000F800; // Node ID 124
            // Fields: Type=0, TaskID=124
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[281] = 0x0002002AE0011800; // Node ID 140
            // Fields: Type=0, TaskID=140
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[282] = 0x00000006C0023880; // Node ID 284
            // Fields: Type=1, TaskID=284
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[283] = 0x0000000A00024C80; // Node ID 294
            // Fields: Type=1, TaskID=294
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[284] = 0x0002002AC000FC00; // Node ID 126
            // Fields: Type=0, TaskID=126
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[285] = 0x0000000A00024A80; // Node ID 293
            // Fields: Type=1, TaskID=293
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[286] = 0x0004003200011C00; // Node ID 142
            // Fields: Type=0, TaskID=142
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[287] = 0x0002802680011E00; // Node ID 143
            // Fields: Type=0, TaskID=143
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[288] = 0x0004802A20012000; // Node ID 144
            // Fields: Type=0, TaskID=144
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[289] = 0x00030026A0012200; // Node ID 145
            // Fields: Type=0, TaskID=145
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[290] = 0x0005002A40012400; // Node ID 146
            // Fields: Type=0, TaskID=146
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[291] = 0x00038026C0012600; // Node ID 147
            // Fields: Type=0, TaskID=147
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[292] = 0x0005802A60012800; // Node ID 148
            // Fields: Type=0, TaskID=148
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[293] = 0x00080026E0012A00; // Node ID 149
            // Fields: Type=0, TaskID=149
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[294] = 0x0000000B00012C00; // Node ID 150
            // Fields: Type=0, TaskID=150
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        // Task ID Mapping Lists
        int32_t* global_task_id_to_dev_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 295 * sizeof(int32_t));
        global_task_id_to_dev_task_id_chip_00[0] = 0; // Node ID 0 -> Dev Task 0 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[1] = 1; // Node ID 1 -> Dev Task 1 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[2] = 2; // Node ID 2 -> Dev Task 2 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[3] = 3; // Node ID 3 -> Dev Task 3 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[4] = 4; // Node ID 4 -> Dev Task 4 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[5] = 5; // Node ID 5 -> Dev Task 5 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[6] = 6; // Node ID 6 -> Dev Task 6 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[7] = 7; // Node ID 7 -> Dev Task 7 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[8] = 8; // Node ID 8 -> Dev Task 8 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[9] = 9; // Node ID 9 -> Dev Task 9 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[10] = 10; // Node ID 10 -> Dev Task 10 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[11] = 11; // Node ID 11 -> Dev Task 11 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[12] = 12; // Node ID 12 -> Dev Task 12 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[13] = 13; // Node ID 13 -> Dev Task 13 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[14] = 14; // Node ID 14 -> Dev Task 14 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[15] = 15; // Node ID 15 -> Dev Task 15 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[16] = 16; // Node ID 16 -> Dev Task 16 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[17] = 17; // Node ID 17 -> Dev Task 17 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[18] = 18; // Node ID 18 -> Dev Task 18 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_swiglu)
        global_task_id_to_dev_task_id_chip_00[19] = 19; // Node ID 19 -> Dev Task 19 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_down)
        global_task_id_to_dev_task_id_chip_00[20] = -1; // Node ID 20 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[21] = 20; // Node ID 21 -> Dev Task 20 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_swiglu)
        global_task_id_to_dev_task_id_chip_00[22] = 21; // Node ID 22 -> Dev Task 21 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_down)
        global_task_id_to_dev_task_id_chip_00[23] = -1; // Node ID 23 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[24] = 22; // Node ID 24 -> Dev Task 22 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_dual_vc_gemm_full)
        global_task_id_to_dev_task_id_chip_00[25] = -1; // Node ID 25 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[26] = -1; // Node ID 26 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule)
        global_task_id_to_dev_task_id_chip_00[27] = -1; // Node ID 27 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_prepare_request)
        global_task_id_to_dev_task_id_chip_00[28] = -1; // Node ID 28 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_execute)
        global_task_id_to_dev_task_id_chip_00[29] = 23; // Node ID 29 -> Dev Task 23 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[30] = 24; // Node ID 30 -> Dev Task 24 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[31] = 25; // Node ID 31 -> Dev Task 25 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[32] = 26; // Node ID 32 -> Dev Task 26 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[33] = 27; // Node ID 33 -> Dev Task 27 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[34] = 28; // Node ID 34 -> Dev Task 28 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[35] = 29; // Node ID 35 -> Dev Task 29 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[36] = 30; // Node ID 36 -> Dev Task 30 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[37] = 31; // Node ID 37 -> Dev Task 31 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[38] = 32; // Node ID 38 -> Dev Task 32 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[39] = 33; // Node ID 39 -> Dev Task 33 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[40] = 34; // Node ID 40 -> Dev Task 34 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[41] = 35; // Node ID 41 -> Dev Task 35 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[42] = 36; // Node ID 42 -> Dev Task 36 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[43] = 37; // Node ID 43 -> Dev Task 37 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[44] = 38; // Node ID 44 -> Dev Task 38 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[45] = 39; // Node ID 45 -> Dev Task 39 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[46] = 40; // Node ID 46 -> Dev Task 40 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[47] = 41; // Node ID 47 -> Dev Task 41 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[48] = 42; // Node ID 48 -> Dev Task 42 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[49] = 43; // Node ID 49 -> Dev Task 43 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[50] = 44; // Node ID 50 -> Dev Task 44 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[51] = 45; // Node ID 51 -> Dev Task 45 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[52] = 46; // Node ID 52 -> Dev Task 46 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[53] = 47; // Node ID 53 -> Dev Task 47 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[54] = 48; // Node ID 54 -> Dev Task 48 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[55] = 49; // Node ID 55 -> Dev Task 49 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[56] = 50; // Node ID 56 -> Dev Task 50 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[57] = 51; // Node ID 57 -> Dev Task 51 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[58] = 52; // Node ID 58 -> Dev Task 52 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[59] = 53; // Node ID 59 -> Dev Task 53 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[60] = 54; // Node ID 60 -> Dev Task 54 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[61] = 55; // Node ID 61 -> Dev Task 55 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[62] = 56; // Node ID 62 -> Dev Task 56 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[63] = 57; // Node ID 63 -> Dev Task 57 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[64] = 58; // Node ID 64 -> Dev Task 58 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[65] = 59; // Node ID 65 -> Dev Task 59 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[66] = 60; // Node ID 66 -> Dev Task 60 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[67] = 61; // Node ID 67 -> Dev Task 61 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[68] = 62; // Node ID 68 -> Dev Task 62 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[69] = 63; // Node ID 69 -> Dev Task 63 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[70] = 64; // Node ID 70 -> Dev Task 64 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[71] = 65; // Node ID 71 -> Dev Task 65 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[72] = 66; // Node ID 72 -> Dev Task 66 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[73] = 67; // Node ID 73 -> Dev Task 67 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[74] = 68; // Node ID 74 -> Dev Task 68 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[75] = 69; // Node ID 75 -> Dev Task 69 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[76] = 70; // Node ID 76 -> Dev Task 70 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[77] = 71; // Node ID 77 -> Dev Task 71 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[78] = 72; // Node ID 78 -> Dev Task 72 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[79] = 73; // Node ID 79 -> Dev Task 73 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[80] = 74; // Node ID 80 -> Dev Task 74 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[81] = 75; // Node ID 81 -> Dev Task 75 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[82] = 76; // Node ID 82 -> Dev Task 76 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[83] = 77; // Node ID 83 -> Dev Task 77 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[84] = 78; // Node ID 84 -> Dev Task 78 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[85] = 79; // Node ID 85 -> Dev Task 79 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[86] = 80; // Node ID 86 -> Dev Task 80 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[87] = 81; // Node ID 87 -> Dev Task 81 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[88] = 82; // Node ID 88 -> Dev Task 82 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[89] = 83; // Node ID 89 -> Dev Task 83 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[90] = 84; // Node ID 90 -> Dev Task 84 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[91] = 85; // Node ID 91 -> Dev Task 85 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[92] = 86; // Node ID 92 -> Dev Task 86 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[93] = 87; // Node ID 93 -> Dev Task 87 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[94] = 88; // Node ID 94 -> Dev Task 88 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[95] = 89; // Node ID 95 -> Dev Task 89 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[96] = 90; // Node ID 96 -> Dev Task 90 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[97] = 91; // Node ID 97 -> Dev Task 91 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[98] = 92; // Node ID 98 -> Dev Task 92 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[99] = 93; // Node ID 99 -> Dev Task 93 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[100] = 94; // Node ID 100 -> Dev Task 94 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[101] = 95; // Node ID 101 -> Dev Task 95 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[102] = 96; // Node ID 102 -> Dev Task 96 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[103] = 97; // Node ID 103 -> Dev Task 97 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[104] = 98; // Node ID 104 -> Dev Task 98 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[105] = 99; // Node ID 105 -> Dev Task 99 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[106] = 100; // Node ID 106 -> Dev Task 100 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[107] = 101; // Node ID 107 -> Dev Task 101 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[108] = 102; // Node ID 108 -> Dev Task 102 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[109] = 103; // Node ID 109 -> Dev Task 103 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[110] = 104; // Node ID 110 -> Dev Task 104 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[111] = 105; // Node ID 111 -> Dev Task 105 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[112] = 106; // Node ID 112 -> Dev Task 106 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[113] = 107; // Node ID 113 -> Dev Task 107 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[114] = 108; // Node ID 114 -> Dev Task 108 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[115] = 109; // Node ID 115 -> Dev Task 109 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[116] = 110; // Node ID 116 -> Dev Task 110 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[117] = 111; // Node ID 117 -> Dev Task 111 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[118] = 112; // Node ID 118 -> Dev Task 112 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[119] = 113; // Node ID 119 -> Dev Task 113 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[120] = 114; // Node ID 120 -> Dev Task 114 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[121] = 115; // Node ID 121 -> Dev Task 115 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[122] = 116; // Node ID 122 -> Dev Task 116 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[123] = 117; // Node ID 123 -> Dev Task 117 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[124] = 118; // Node ID 124 -> Dev Task 118 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[125] = 119; // Node ID 125 -> Dev Task 119 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[126] = 120; // Node ID 126 -> Dev Task 120 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[127] = 121; // Node ID 127 -> Dev Task 121 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_dev_task_id_chip_00[128] = 122; // Node ID 128 -> Dev Task 122 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[129] = 123; // Node ID 129 -> Dev Task 123 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[130] = 124; // Node ID 130 -> Dev Task 124 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[131] = 125; // Node ID 131 -> Dev Task 125 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_dev_task_id_chip_00[132] = 126; // Node ID 132 -> Dev Task 126 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_dev_task_id_chip_00[133] = 127; // Node ID 133 -> Dev Task 127 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_dev_task_id_chip_00[134] = 128; // Node ID 134 -> Dev Task 128 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[135] = 129; // Node ID 135 -> Dev Task 129 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[136] = 130; // Node ID 136 -> Dev Task 130 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_dev_task_id_chip_00[137] = 131; // Node ID 137 -> Dev Task 131 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_dev_task_id_chip_00[138] = 132; // Node ID 138 -> Dev Task 132 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_dev_task_id_chip_00[139] = 133; // Node ID 139 -> Dev Task 133 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_dev_task_id_chip_00[140] = 134; // Node ID 140 -> Dev Task 134 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_dev_task_id_chip_00[141] = -1; // Node ID 141 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_dev_task_id_chip_00[142] = 135; // Node ID 142 -> Dev Task 135 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[143] = 136; // Node ID 143 -> Dev Task 136 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[144] = 137; // Node ID 144 -> Dev Task 137 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[145] = 138; // Node ID 145 -> Dev Task 138 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[146] = 139; // Node ID 146 -> Dev Task 139 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[147] = 140; // Node ID 147 -> Dev Task 140 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[148] = 141; // Node ID 148 -> Dev Task 141 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[149] = 142; // Node ID 149 -> Dev Task 142 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[150] = -1; // Node ID 150 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[151] = -1; // Node ID 151 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[152] = -1; // Node ID 152 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[153] = -1; // Node ID 153 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[154] = -1; // Node ID 154 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[155] = -1; // Node ID 155 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[156] = -1; // Node ID 156 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[157] = -1; // Node ID 157 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[158] = -1; // Node ID 158 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[159] = -1; // Node ID 159 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[160] = -1; // Node ID 160 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[161] = -1; // Node ID 161 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[162] = -1; // Node ID 162 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[163] = -1; // Node ID 163 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[164] = -1; // Node ID 164 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[165] = -1; // Node ID 165 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[166] = -1; // Node ID 166 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[167] = -1; // Node ID 167 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[168] = -1; // Node ID 168 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[169] = -1; // Node ID 169 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[170] = -1; // Node ID 170 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[171] = -1; // Node ID 171 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[172] = -1; // Node ID 172 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[173] = -1; // Node ID 173 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[174] = -1; // Node ID 174 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[175] = -1; // Node ID 175 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[176] = -1; // Node ID 176 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[177] = -1; // Node ID 177 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[178] = -1; // Node ID 178 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[179] = -1; // Node ID 179 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[180] = -1; // Node ID 180 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[181] = -1; // Node ID 181 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[182] = -1; // Node ID 182 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[183] = -1; // Node ID 183 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[184] = -1; // Node ID 184 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[185] = -1; // Node ID 185 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[186] = -1; // Node ID 186 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[187] = -1; // Node ID 187 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[188] = -1; // Node ID 188 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[189] = -1; // Node ID 189 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[190] = -1; // Node ID 190 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[191] = -1; // Node ID 191 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[192] = -1; // Node ID 192 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[193] = -1; // Node ID 193 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[194] = -1; // Node ID 194 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[195] = -1; // Node ID 195 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[196] = -1; // Node ID 196 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[197] = -1; // Node ID 197 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[198] = -1; // Node ID 198 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[199] = -1; // Node ID 199 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[200] = -1; // Node ID 200 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[201] = -1; // Node ID 201 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[202] = -1; // Node ID 202 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[203] = -1; // Node ID 203 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[204] = -1; // Node ID 204 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[205] = -1; // Node ID 205 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[206] = -1; // Node ID 206 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[207] = -1; // Node ID 207 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[208] = -1; // Node ID 208 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[209] = -1; // Node ID 209 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[210] = -1; // Node ID 210 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[211] = -1; // Node ID 211 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[212] = -1; // Node ID 212 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[213] = -1; // Node ID 213 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[214] = -1; // Node ID 214 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[215] = -1; // Node ID 215 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[216] = -1; // Node ID 216 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[217] = -1; // Node ID 217 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[218] = -1; // Node ID 218 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[219] = -1; // Node ID 219 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[220] = -1; // Node ID 220 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[221] = -1; // Node ID 221 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[222] = -1; // Node ID 222 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[223] = -1; // Node ID 223 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[224] = -1; // Node ID 224 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[225] = -1; // Node ID 225 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[226] = -1; // Node ID 226 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[227] = -1; // Node ID 227 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[228] = -1; // Node ID 228 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[229] = -1; // Node ID 229 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[230] = -1; // Node ID 230 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[231] = -1; // Node ID 231 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[232] = -1; // Node ID 232 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[233] = -1; // Node ID 233 (Node_ID0_Chiplet0_Cluster1_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[234] = -1; // Node ID 234 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[235] = -1; // Node ID 235 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[236] = -1; // Node ID 236 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[237] = -1; // Node ID 237 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[238] = -1; // Node ID 238 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[239] = -1; // Node ID 239 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[240] = -1; // Node ID 240 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[241] = -1; // Node ID 241 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[242] = -1; // Node ID 242 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[243] = -1; // Node ID 243 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[244] = -1; // Node ID 244 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[245] = -1; // Node ID 245 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[246] = -1; // Node ID 246 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[247] = -1; // Node ID 247 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[248] = -1; // Node ID 248 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[249] = -1; // Node ID 249 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[250] = -1; // Node ID 250 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[251] = -1; // Node ID 251 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[252] = -1; // Node ID 252 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[253] = -1; // Node ID 253 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[254] = -1; // Node ID 254 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[255] = -1; // Node ID 255 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[256] = -1; // Node ID 256 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[257] = -1; // Node ID 257 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[258] = -1; // Node ID 258 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[259] = -1; // Node ID 259 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[260] = -1; // Node ID 260 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[261] = -1; // Node ID 261 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[262] = -1; // Node ID 262 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[263] = -1; // Node ID 263 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[264] = -1; // Node ID 264 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[265] = -1; // Node ID 265 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[266] = -1; // Node ID 266 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[267] = -1; // Node ID 267 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[268] = -1; // Node ID 268 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[269] = -1; // Node ID 269 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[270] = -1; // Node ID 270 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[271] = -1; // Node ID 271 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[272] = -1; // Node ID 272 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[273] = -1; // Node ID 273 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[274] = -1; // Node ID 274 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[275] = -1; // Node ID 275 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[276] = -1; // Node ID 276 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[277] = -1; // Node ID 277 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[278] = -1; // Node ID 278 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[279] = -1; // Node ID 279 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[280] = -1; // Node ID 280 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[281] = -1; // Node ID 281 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[282] = -1; // Node ID 282 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[283] = -1; // Node ID 283 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[284] = -1; // Node ID 284 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[285] = -1; // Node ID 285 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[286] = -1; // Node ID 286 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[287] = -1; // Node ID 287 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[288] = -1; // Node ID 288 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[289] = -1; // Node ID 289 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[290] = -1; // Node ID 290 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[291] = -1; // Node ID 291 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[292] = -1; // Node ID 292 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[293] = -1; // Node ID 293 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[294] = -1; // Node ID 294 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        uint32_t num_dev_tasks_chip_00 = 143;
        int32_t* global_task_id_to_host_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 295 * sizeof(int32_t));
        global_task_id_to_host_task_id_chip_00[0] = -1; // Node ID 0 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[1] = -1; // Node ID 1 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[2] = -1; // Node ID 2 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[3] = -1; // Node ID 3 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[4] = -1; // Node ID 4 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[5] = -1; // Node ID 5 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[6] = -1; // Node ID 6 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[7] = -1; // Node ID 7 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[8] = -1; // Node ID 8 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[9] = -1; // Node ID 9 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[10] = -1; // Node ID 10 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[11] = -1; // Node ID 11 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[12] = -1; // Node ID 12 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[13] = -1; // Node ID 13 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[14] = -1; // Node ID 14 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[15] = -1; // Node ID 15 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[16] = -1; // Node ID 16 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[17] = -1; // Node ID 17 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[18] = -1; // Node ID 18 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_swiglu)
        global_task_id_to_host_task_id_chip_00[19] = -1; // Node ID 19 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_down)
        global_task_id_to_host_task_id_chip_00[20] = 0; // Node ID 20 -> Host Task 0 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[21] = -1; // Node ID 21 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_swiglu)
        global_task_id_to_host_task_id_chip_00[22] = -1; // Node ID 22 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_down)
        global_task_id_to_host_task_id_chip_00[23] = 1; // Node ID 23 -> Host Task 1 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[24] = -1; // Node ID 24 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_dual_vc_gemm_full)
        global_task_id_to_host_task_id_chip_00[25] = 2; // Node ID 25 -> Host Task 2 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[26] = 3; // Node ID 26 -> Host Task 3 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule)
        global_task_id_to_host_task_id_chip_00[27] = 4; // Node ID 27 -> Host Task 4 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_prepare_request)
        global_task_id_to_host_task_id_chip_00[28] = 5; // Node ID 28 -> Host Task 5 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_execute)
        global_task_id_to_host_task_id_chip_00[29] = -1; // Node ID 29 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[30] = -1; // Node ID 30 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[31] = -1; // Node ID 31 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[32] = -1; // Node ID 32 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[33] = -1; // Node ID 33 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[34] = -1; // Node ID 34 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[35] = -1; // Node ID 35 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[36] = -1; // Node ID 36 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[37] = -1; // Node ID 37 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[38] = -1; // Node ID 38 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[39] = -1; // Node ID 39 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[40] = -1; // Node ID 40 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[41] = -1; // Node ID 41 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[42] = -1; // Node ID 42 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[43] = -1; // Node ID 43 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[44] = -1; // Node ID 44 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[45] = -1; // Node ID 45 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[46] = -1; // Node ID 46 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[47] = -1; // Node ID 47 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[48] = -1; // Node ID 48 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[49] = -1; // Node ID 49 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[50] = -1; // Node ID 50 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[51] = -1; // Node ID 51 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[52] = -1; // Node ID 52 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[53] = -1; // Node ID 53 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[54] = -1; // Node ID 54 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[55] = -1; // Node ID 55 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[56] = -1; // Node ID 56 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[57] = -1; // Node ID 57 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[58] = -1; // Node ID 58 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[59] = -1; // Node ID 59 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[60] = -1; // Node ID 60 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[61] = -1; // Node ID 61 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[62] = -1; // Node ID 62 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[63] = -1; // Node ID 63 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[64] = -1; // Node ID 64 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[65] = -1; // Node ID 65 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[66] = -1; // Node ID 66 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[67] = -1; // Node ID 67 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[68] = -1; // Node ID 68 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[69] = -1; // Node ID 69 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[70] = -1; // Node ID 70 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[71] = -1; // Node ID 71 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[72] = -1; // Node ID 72 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[73] = -1; // Node ID 73 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[74] = -1; // Node ID 74 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[75] = -1; // Node ID 75 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[76] = -1; // Node ID 76 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[77] = -1; // Node ID 77 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[78] = -1; // Node ID 78 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[79] = -1; // Node ID 79 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[80] = -1; // Node ID 80 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[81] = -1; // Node ID 81 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[82] = -1; // Node ID 82 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[83] = -1; // Node ID 83 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[84] = -1; // Node ID 84 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[85] = -1; // Node ID 85 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[86] = -1; // Node ID 86 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[87] = -1; // Node ID 87 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[88] = -1; // Node ID 88 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[89] = -1; // Node ID 89 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[90] = -1; // Node ID 90 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[91] = -1; // Node ID 91 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[92] = -1; // Node ID 92 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[93] = -1; // Node ID 93 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[94] = -1; // Node ID 94 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[95] = -1; // Node ID 95 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[96] = -1; // Node ID 96 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[97] = -1; // Node ID 97 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[98] = -1; // Node ID 98 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[99] = -1; // Node ID 99 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[100] = -1; // Node ID 100 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[101] = -1; // Node ID 101 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[102] = -1; // Node ID 102 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[103] = -1; // Node ID 103 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[104] = -1; // Node ID 104 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[105] = -1; // Node ID 105 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[106] = -1; // Node ID 106 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[107] = -1; // Node ID 107 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[108] = -1; // Node ID 108 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[109] = -1; // Node ID 109 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[110] = -1; // Node ID 110 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[111] = -1; // Node ID 111 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[112] = -1; // Node ID 112 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[113] = -1; // Node ID 113 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[114] = -1; // Node ID 114 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[115] = -1; // Node ID 115 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[116] = -1; // Node ID 116 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[117] = -1; // Node ID 117 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[118] = -1; // Node ID 118 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[119] = -1; // Node ID 119 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[120] = -1; // Node ID 120 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[121] = -1; // Node ID 121 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[122] = -1; // Node ID 122 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[123] = -1; // Node ID 123 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[124] = -1; // Node ID 124 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[125] = -1; // Node ID 125 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[126] = -1; // Node ID 126 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[127] = -1; // Node ID 127 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        global_task_id_to_host_task_id_chip_00[128] = -1; // Node ID 128 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[129] = -1; // Node ID 129 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[130] = -1; // Node ID 130 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        global_task_id_to_host_task_id_chip_00[131] = -1; // Node ID 131 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        global_task_id_to_host_task_id_chip_00[132] = -1; // Node ID 132 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        global_task_id_to_host_task_id_chip_00[133] = -1; // Node ID 133 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        global_task_id_to_host_task_id_chip_00[134] = -1; // Node ID 134 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[135] = -1; // Node ID 135 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[136] = -1; // Node ID 136 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        global_task_id_to_host_task_id_chip_00[137] = -1; // Node ID 137 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        global_task_id_to_host_task_id_chip_00[138] = -1; // Node ID 138 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        global_task_id_to_host_task_id_chip_00[139] = -1; // Node ID 139 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        global_task_id_to_host_task_id_chip_00[140] = -1; // Node ID 140 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store)
        global_task_id_to_host_task_id_chip_00[141] = 6; // Node ID 141 -> Host Task 6 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_host_task_id_chip_00[142] = -1; // Node ID 142 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[143] = -1; // Node ID 143 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[144] = -1; // Node ID 144 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[145] = -1; // Node ID 145 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[146] = -1; // Node ID 146 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[147] = -1; // Node ID 147 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[148] = -1; // Node ID 148 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[149] = -1; // Node ID 149 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[150] = 7; // Node ID 150 -> Host Task 7 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[151] = -1; // Node ID 151 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[152] = -1; // Node ID 152 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[153] = -1; // Node ID 153 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[154] = -1; // Node ID 154 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[155] = -1; // Node ID 155 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[156] = -1; // Node ID 156 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[157] = -1; // Node ID 157 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[158] = -1; // Node ID 158 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[159] = -1; // Node ID 159 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[160] = -1; // Node ID 160 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[161] = -1; // Node ID 161 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[162] = -1; // Node ID 162 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[163] = -1; // Node ID 163 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[164] = -1; // Node ID 164 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[165] = -1; // Node ID 165 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[166] = -1; // Node ID 166 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[167] = -1; // Node ID 167 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[168] = -1; // Node ID 168 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[169] = -1; // Node ID 169 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[170] = -1; // Node ID 170 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[171] = -1; // Node ID 171 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[172] = -1; // Node ID 172 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[173] = -1; // Node ID 173 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[174] = -1; // Node ID 174 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[175] = -1; // Node ID 175 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[176] = -1; // Node ID 176 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[177] = -1; // Node ID 177 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[178] = -1; // Node ID 178 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[179] = -1; // Node ID 179 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[180] = -1; // Node ID 180 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[181] = -1; // Node ID 181 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[182] = -1; // Node ID 182 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[183] = -1; // Node ID 183 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[184] = -1; // Node ID 184 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[185] = -1; // Node ID 185 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[186] = -1; // Node ID 186 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[187] = -1; // Node ID 187 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[188] = -1; // Node ID 188 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[189] = -1; // Node ID 189 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[190] = -1; // Node ID 190 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[191] = -1; // Node ID 191 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[192] = -1; // Node ID 192 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[193] = -1; // Node ID 193 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[194] = -1; // Node ID 194 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[195] = -1; // Node ID 195 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[196] = -1; // Node ID 196 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[197] = -1; // Node ID 197 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[198] = -1; // Node ID 198 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[199] = -1; // Node ID 199 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[200] = -1; // Node ID 200 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[201] = -1; // Node ID 201 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[202] = -1; // Node ID 202 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[203] = -1; // Node ID 203 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[204] = -1; // Node ID 204 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[205] = -1; // Node ID 205 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[206] = -1; // Node ID 206 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[207] = -1; // Node ID 207 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[208] = -1; // Node ID 208 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[209] = -1; // Node ID 209 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[210] = -1; // Node ID 210 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[211] = -1; // Node ID 211 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[212] = -1; // Node ID 212 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[213] = -1; // Node ID 213 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[214] = -1; // Node ID 214 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[215] = -1; // Node ID 215 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[216] = -1; // Node ID 216 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[217] = -1; // Node ID 217 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[218] = -1; // Node ID 218 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[219] = -1; // Node ID 219 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[220] = -1; // Node ID 220 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[221] = -1; // Node ID 221 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[222] = -1; // Node ID 222 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[223] = -1; // Node ID 223 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[224] = -1; // Node ID 224 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[225] = -1; // Node ID 225 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[226] = -1; // Node ID 226 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[227] = -1; // Node ID 227 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[228] = -1; // Node ID 228 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[229] = -1; // Node ID 229 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[230] = -1; // Node ID 230 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[231] = -1; // Node ID 231 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[232] = -1; // Node ID 232 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[233] = -1; // Node ID 233 (Node_ID0_Chiplet0_Cluster1_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[234] = -1; // Node ID 234 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[235] = -1; // Node ID 235 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[236] = -1; // Node ID 236 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[237] = -1; // Node ID 237 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[238] = -1; // Node ID 238 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[239] = -1; // Node ID 239 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[240] = -1; // Node ID 240 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[241] = -1; // Node ID 241 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[242] = -1; // Node ID 242 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[243] = -1; // Node ID 243 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[244] = -1; // Node ID 244 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[245] = -1; // Node ID 245 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[246] = -1; // Node ID 246 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[247] = -1; // Node ID 247 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[248] = -1; // Node ID 248 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[249] = -1; // Node ID 249 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[250] = -1; // Node ID 250 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[251] = -1; // Node ID 251 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[252] = -1; // Node ID 252 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[253] = -1; // Node ID 253 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[254] = -1; // Node ID 254 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[255] = -1; // Node ID 255 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[256] = -1; // Node ID 256 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[257] = -1; // Node ID 257 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[258] = -1; // Node ID 258 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[259] = -1; // Node ID 259 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[260] = -1; // Node ID 260 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[261] = -1; // Node ID 261 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[262] = -1; // Node ID 262 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[263] = -1; // Node ID 263 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[264] = -1; // Node ID 264 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[265] = -1; // Node ID 265 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[266] = -1; // Node ID 266 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[267] = -1; // Node ID 267 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[268] = -1; // Node ID 268 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[269] = -1; // Node ID 269 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[270] = -1; // Node ID 270 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[271] = -1; // Node ID 271 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[272] = -1; // Node ID 272 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[273] = -1; // Node ID 273 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[274] = -1; // Node ID 274 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[275] = -1; // Node ID 275 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[276] = -1; // Node ID 276 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[277] = -1; // Node ID 277 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[278] = -1; // Node ID 278 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[279] = -1; // Node ID 279 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[280] = -1; // Node ID 280 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[281] = -1; // Node ID 281 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[282] = -1; // Node ID 282 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[283] = -1; // Node ID 283 (Node_ID0_Chiplet0_Cluster2_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[284] = -1; // Node ID 284 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[285] = -1; // Node ID 285 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[286] = -1; // Node ID 286 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[287] = -1; // Node ID 287 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[288] = -1; // Node ID 288 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[289] = -1; // Node ID 289 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[290] = -1; // Node ID 290 (Node_ID0_Chiplet0_Cluster3_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[291] = -1; // Node ID 291 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[292] = -1; // Node ID 292 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[293] = -1; // Node ID 293 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[294] = -1; // Node ID 294 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        uint32_t num_host_tasks_chip_00 = 8;
        // 1. Memory Allocations
        uint64_t ptr_c0_l1_layout = bingo_l1_alloc(0x00, 0, 1281280);
        uint64_t ptr_c1_l1_layout = bingo_l1_alloc(0x00, 1, 1281280);
        uint64_t ptr_c2_indiv_dyn_args = bingo_l1_alloc(0x00, 2, 1024);
        uint64_t ptr_c2_indiv_l1_a = bingo_l1_alloc(0x00, 2, 65536);
        uint64_t ptr_c2_indiv_l1_b_down = bingo_l1_alloc(0x00, 2, 262144);
        uint64_t ptr_c2_indiv_l1_b_gate = bingo_l1_alloc(0x00, 2, 262144);
        uint64_t ptr_c2_indiv_l1_b_up = bingo_l1_alloc(0x00, 2, 262144);
        uint64_t ptr_c2_indiv_l1_d = bingo_l1_alloc(0x00, 2, 32768);
        uint64_t ptr_c2_indiv_l1_d1_scratch = bingo_l1_alloc(0x00, 2, 16384);
        uint64_t ptr_c2_indiv_l1_down_d = bingo_l1_alloc(0x00, 2, 65536);
        uint64_t ptr_c3_indiv_dyn_args = bingo_l1_alloc(0x00, 3, 1024);
        uint64_t ptr_c3_indiv_l1_a = bingo_l1_alloc(0x00, 3, 65536);
        uint64_t ptr_c3_indiv_l1_b_down = bingo_l1_alloc(0x00, 3, 262144);
        uint64_t ptr_c3_indiv_l1_b_gate = bingo_l1_alloc(0x00, 3, 262144);
        uint64_t ptr_c3_indiv_l1_b_up = bingo_l1_alloc(0x00, 3, 262144);
        uint64_t ptr_c3_indiv_l1_d = bingo_l1_alloc(0x00, 3, 32768);
        uint64_t ptr_c3_indiv_l1_d1_scratch = bingo_l1_alloc(0x00, 3, 16384);
        uint64_t ptr_c3_indiv_l1_down_d = bingo_l1_alloc(0x00, 3, 65536);
        uint64_t ptr_c3_router_l1_a = bingo_l1_alloc(0x00, 3, 65536);
        uint64_t ptr_c3_router_l1_b = bingo_l1_alloc(0x00, 3, 4096);
        uint64_t ptr_c3_router_l1_d = bingo_l1_alloc(0x00, 3, 512);
        uint64_t ptr_l3_c2_stage = bingo_l3_alloc(0x00, 8192);
        uint64_t ptr_l3_c3_stage = bingo_l3_alloc(0x00, 8192);
        uint64_t ptr_l3_cam_state = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_expert_counts = bingo_l3_alloc(0x00, 32);
        uint64_t ptr_l3_expert_token_ids = bingo_l3_alloc(0x00, 128);
        uint64_t ptr_l3_expert_token_kpos = bingo_l3_alloc(0x00, 128);
        uint64_t ptr_l3_expert_token_offsets = bingo_l3_alloc(0x00, 36);
        uint64_t ptr_l3_indiv_down_out = bingo_l3_alloc(0x00, 524288);
        uint64_t ptr_l3_moe_request = bingo_l3_alloc(0x00, 256);
        uint64_t ptr_l3_moe_runtime_state = bingo_l3_alloc(0x00, 16);
        uint64_t ptr_l3_moe_schedule = bingo_l3_alloc(0x00, 32768);
        uint64_t ptr_l3_router_out = bingo_l3_alloc(0x00, 512);
        uint64_t ptr_l3_shared_down_out = bingo_l3_alloc(0x00, 262144);
        uint64_t ptr_l3_topk_idx = bingo_l3_alloc(0x00, 128);
        uint64_t ptr_l3_topk_scores = bingo_l3_alloc(0x00, 256);

        // 2. Prepare device/host arg/kernel lists
        uint32_t* device_arg_list_chip_00 = (uint32_t*)bingo_l3_alloc(0x00, num_dev_tasks_chip_00 * sizeof(uint32_t));
        uint32_t* device_kernel_list_chip_00 = (uint32_t*)bingo_l3_alloc(0x00, num_dev_tasks_chip_00 * sizeof(uint32_t));
        uint64_t* host_arg_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, num_host_tasks_chip_00 * sizeof(uint64_t));
        uint64_t* host_kernel_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, num_host_tasks_chip_00 * sizeof(uint64_t));

        // 3. Task Arguments Init
        // 3a. Pre-allocate scratchpads for all tasks
        bingo_kernel_scratchpad_t* sp_dev_0 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_1 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_2 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_3 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_4 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_5 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_6 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_7 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_8 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_9 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_10 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_11 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_12 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_13 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_14 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_15 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_16 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_17 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_18 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_19 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_20 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_21 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_22 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_23 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_24 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_25 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_26 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_27 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_28 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_29 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_30 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_31 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_32 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_33 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_34 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_35 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_36 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_37 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_38 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_39 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_40 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_41 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_42 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_43 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_44 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_45 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_46 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_47 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_48 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_49 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_50 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_51 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_52 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_53 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_54 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_55 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_56 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_57 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_58 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_59 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_60 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_61 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_62 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_63 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_64 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_65 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_66 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_67 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_68 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_69 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_70 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_71 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_72 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_73 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_74 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_75 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_76 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_77 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_78 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_79 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_80 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_81 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_82 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_83 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_84 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_85 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_86 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_87 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_88 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_89 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_90 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_91 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_92 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_93 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_94 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_95 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_96 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_97 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_98 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_99 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_100 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_101 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_102 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_103 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_104 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_105 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_106 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_107 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_108 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_109 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_110 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_111 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_112 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_113 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_114 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_115 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_116 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_117 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_118 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_119 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_120 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_121 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_122 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_123 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_124 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_125 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_126 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_127 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_128 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_129 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_130 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_131 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_132 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_133 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_134 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_135 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_136 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_137 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_138 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_139 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_140 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_141 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_142 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_143 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_144 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_145 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_146 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_147 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_148 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_149 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_150 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);

        // Node ID: 0 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_0 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_0->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W)));
        args_dev_chip00_0->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W))) >> 32);
        args_dev_chip00_0->dst_addr_lo = (uint32_t)ptr_c0_l1_layout;
        args_dev_chip00_0->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout >> 32);
        args_dev_chip00_0->size = 262144;
        args_dev_chip00_0->gating_sp_addr = 0;
        args_dev_chip00_0->cond_node_index = 0;
        args_dev_chip00_0->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_0;
        device_arg_list_chip_00[0] = (uint32_t)(uintptr_t)args_dev_chip00_0;
        device_kernel_list_chip_00[0] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 1 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_1 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_1->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W)));
        args_dev_chip00_1->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W))) >> 32);
        args_dev_chip00_1->dst_addr_lo = (uint32_t)ptr_c1_l1_layout;
        args_dev_chip00_1->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout >> 32);
        args_dev_chip00_1->size = 262144;
        args_dev_chip00_1->gating_sp_addr = 0;
        args_dev_chip00_1->cond_node_index = 0;
        args_dev_chip00_1->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_1;
        device_arg_list_chip_00[1] = (uint32_t)(uintptr_t)args_dev_chip00_1;
        device_kernel_list_chip_00[1] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 2 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_2 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_2->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_2->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B))) >> 32);
        args_dev_chip00_2->dst_addr_lo = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_2->dst_addr_hi = (uint32_t)(ptr_c2_indiv_l1_b_gate >> 32);
        args_dev_chip00_2->size = 262144;
        args_dev_chip00_2->gating_sp_addr = 0;
        args_dev_chip00_2->cond_node_index = 0;
        args_dev_chip00_2->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_2;
        device_arg_list_chip_00[2] = (uint32_t)(uintptr_t)args_dev_chip00_2;
        device_kernel_list_chip_00[2] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 3 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_3 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_3->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)router_B)));
        args_dev_chip00_3->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)router_B))) >> 32);
        args_dev_chip00_3->dst_addr_lo = (uint32_t)ptr_c3_router_l1_b;
        args_dev_chip00_3->dst_addr_hi = (uint32_t)(ptr_c3_router_l1_b >> 32);
        args_dev_chip00_3->size = 4096;
        args_dev_chip00_3->gating_sp_addr = 0;
        args_dev_chip00_3->cond_node_index = 0;
        args_dev_chip00_3->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_3;
        device_arg_list_chip_00[3] = (uint32_t)(uintptr_t)args_dev_chip00_3;
        device_kernel_list_chip_00[3] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 4 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_4 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_4->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B + 1835008)));
        args_dev_chip00_4->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B + 1835008))) >> 32);
        args_dev_chip00_4->dst_addr_lo = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_4->dst_addr_hi = (uint32_t)(ptr_c3_indiv_l1_b_gate >> 32);
        args_dev_chip00_4->size = 262144;
        args_dev_chip00_4->gating_sp_addr = 0;
        args_dev_chip00_4->cond_node_index = 0;
        args_dev_chip00_4->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_4;
        device_arg_list_chip_00[4] = (uint32_t)(uintptr_t)args_dev_chip00_4;
        device_kernel_list_chip_00[4] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 5 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_5 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_5->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_V)));
        args_dev_chip00_5->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_V))) >> 32);
        args_dev_chip00_5->dst_addr_lo = (uint32_t)ptr_c0_l1_layout + 262416;
        args_dev_chip00_5->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout + 262416 >> 32);
        args_dev_chip00_5->size = 262144;
        args_dev_chip00_5->gating_sp_addr = 0;
        args_dev_chip00_5->cond_node_index = 0;
        args_dev_chip00_5->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_5;
        device_arg_list_chip_00[5] = (uint32_t)(uintptr_t)args_dev_chip00_5;
        device_kernel_list_chip_00[5] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 6 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_6 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_6->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_V)));
        args_dev_chip00_6->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_V))) >> 32);
        args_dev_chip00_6->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 262416;
        args_dev_chip00_6->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 262416 >> 32);
        args_dev_chip00_6->size = 262144;
        args_dev_chip00_6->gating_sp_addr = 0;
        args_dev_chip00_6->cond_node_index = 0;
        args_dev_chip00_6->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_6;
        device_arg_list_chip_00[6] = (uint32_t)(uintptr_t)args_dev_chip00_6;
        device_kernel_list_chip_00[6] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 7 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_7 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_7->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_7->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B))) >> 32);
        args_dev_chip00_7->dst_addr_lo = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_7->dst_addr_hi = (uint32_t)(ptr_c2_indiv_l1_b_up >> 32);
        args_dev_chip00_7->size = 262144;
        args_dev_chip00_7->gating_sp_addr = 0;
        args_dev_chip00_7->cond_node_index = 0;
        args_dev_chip00_7->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_7;
        device_arg_list_chip_00[7] = (uint32_t)(uintptr_t)args_dev_chip00_7;
        device_kernel_list_chip_00[7] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 8 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_8 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_8->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_8->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B))) >> 32);
        args_dev_chip00_8->dst_addr_lo = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_8->dst_addr_hi = (uint32_t)(ptr_c2_indiv_l1_b_down >> 32);
        args_dev_chip00_8->size = 262144;
        args_dev_chip00_8->gating_sp_addr = 0;
        args_dev_chip00_8->cond_node_index = 0;
        args_dev_chip00_8->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_8;
        device_arg_list_chip_00[8] = (uint32_t)(uintptr_t)args_dev_chip00_8;
        device_kernel_list_chip_00[8] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 9 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_9 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_9->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B + 1835008)));
        args_dev_chip00_9->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B + 1835008))) >> 32);
        args_dev_chip00_9->dst_addr_lo = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_9->dst_addr_hi = (uint32_t)(ptr_c3_indiv_l1_b_up >> 32);
        args_dev_chip00_9->size = 262144;
        args_dev_chip00_9->gating_sp_addr = 0;
        args_dev_chip00_9->cond_node_index = 0;
        args_dev_chip00_9->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_9;
        device_arg_list_chip_00[9] = (uint32_t)(uintptr_t)args_dev_chip00_9;
        device_kernel_list_chip_00[9] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 10 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_10 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_10->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B + 1835008)));
        args_dev_chip00_10->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B + 1835008))) >> 32);
        args_dev_chip00_10->dst_addr_lo = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_10->dst_addr_hi = (uint32_t)(ptr_c3_indiv_l1_b_down >> 32);
        args_dev_chip00_10->size = 262144;
        args_dev_chip00_10->gating_sp_addr = 0;
        args_dev_chip00_10->cond_node_index = 0;
        args_dev_chip00_10->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_10;
        device_arg_list_chip_00[10] = (uint32_t)(uintptr_t)args_dev_chip00_10;
        device_kernel_list_chip_00[10] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 11 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_11 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_11->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_A_row_stride_2080)));
        args_dev_chip00_11->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_A_row_stride_2080))) >> 32);
        args_dev_chip00_11->dst_addr_lo = (uint32_t)ptr_c0_l1_layout + 1050624;
        args_dev_chip00_11->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout + 1050624 >> 32);
        args_dev_chip00_11->size = 66560;
        args_dev_chip00_11->gating_sp_addr = 0;
        args_dev_chip00_11->cond_node_index = 0;
        args_dev_chip00_11->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_11;
        device_arg_list_chip_00[11] = (uint32_t)(uintptr_t)args_dev_chip00_11;
        device_kernel_list_chip_00[11] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 12 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_12 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_12->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_A_row_stride_2080)));
        args_dev_chip00_12->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_A_row_stride_2080))) >> 32);
        args_dev_chip00_12->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 1050624;
        args_dev_chip00_12->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 1050624 >> 32);
        args_dev_chip00_12->size = 66560;
        args_dev_chip00_12->gating_sp_addr = 0;
        args_dev_chip00_12->cond_node_index = 0;
        args_dev_chip00_12->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_12;
        device_arg_list_chip_00[12] = (uint32_t)(uintptr_t)args_dev_chip00_12;
        device_kernel_list_chip_00[12] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 13 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_13 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_13->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A_tiled)));
        args_dev_chip00_13->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)input_A_tiled))) >> 32);
        args_dev_chip00_13->dst_addr_lo = (uint32_t)ptr_c3_router_l1_a;
        args_dev_chip00_13->dst_addr_hi = (uint32_t)(ptr_c3_router_l1_a >> 32);
        args_dev_chip00_13->size = 65536;
        args_dev_chip00_13->gating_sp_addr = 0;
        args_dev_chip00_13->cond_node_index = 0;
        args_dev_chip00_13->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_13;
        device_arg_list_chip_00[13] = (uint32_t)(uintptr_t)args_dev_chip00_13;
        device_kernel_list_chip_00[13] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 14 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_14 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_14->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_left)));
        args_dev_chip00_14->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_left))) >> 32);
        args_dev_chip00_14->dst_addr_lo = (uint32_t)ptr_c0_l1_layout + 525440;
        args_dev_chip00_14->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout + 525440 >> 32);
        args_dev_chip00_14->size = 262144;
        args_dev_chip00_14->gating_sp_addr = 0;
        args_dev_chip00_14->cond_node_index = 0;
        args_dev_chip00_14->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_14;
        device_arg_list_chip_00[14] = (uint32_t)(uintptr_t)args_dev_chip00_14;
        device_kernel_list_chip_00[14] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 15 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_15 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_15->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right)));
        args_dev_chip00_15->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right))) >> 32);
        args_dev_chip00_15->dst_addr_lo = (uint32_t)ptr_c0_l1_layout + 788480;
        args_dev_chip00_15->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout + 788480 >> 32);
        args_dev_chip00_15->size = 262144;
        args_dev_chip00_15->gating_sp_addr = 0;
        args_dev_chip00_15->cond_node_index = 0;
        args_dev_chip00_15->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_15;
        device_arg_list_chip_00[15] = (uint32_t)(uintptr_t)args_dev_chip00_15;
        device_kernel_list_chip_00[15] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 16 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_16 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_16->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_left)));
        args_dev_chip00_16->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_left))) >> 32);
        args_dev_chip00_16->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 525440;
        args_dev_chip00_16->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 525440 >> 32);
        args_dev_chip00_16->size = 262144;
        args_dev_chip00_16->gating_sp_addr = 0;
        args_dev_chip00_16->cond_node_index = 0;
        args_dev_chip00_16->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_16;
        device_arg_list_chip_00[16] = (uint32_t)(uintptr_t)args_dev_chip00_16;
        device_kernel_list_chip_00[16] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 17 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_17 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_17->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right)));
        args_dev_chip00_17->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right))) >> 32);
        args_dev_chip00_17->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 788480;
        args_dev_chip00_17->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 788480 >> 32);
        args_dev_chip00_17->size = 262144;
        args_dev_chip00_17->gating_sp_addr = 0;
        args_dev_chip00_17->cond_node_index = 0;
        args_dev_chip00_17->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_17;
        device_arg_list_chip_00[17] = (uint32_t)(uintptr_t)args_dev_chip00_17;
        device_kernel_list_chip_00[17] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 18 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_swiglu (__snax_bingo_kernel_dual_vc_l15_moe_swiglu)
        __snax_bingo_kernel_dual_vc_l15_moe_swiglu_args_t* args_dev_chip00_18 = (__snax_bingo_kernel_dual_vc_l15_moe_swiglu_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_dual_vc_l15_moe_swiglu_args_t));
        args_dev_chip00_18->shape_cfg_addr = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg)));
        args_dev_chip00_18->tcdm_base = (uint32_t)ptr_c0_l1_layout;
        args_dev_chip00_18->rescale_mult = 1;
        args_dev_chip00_18->rescale_shift = 0;
        args_dev_chip00_18->gating_sp_addr = 0;
        args_dev_chip00_18->cond_node_index = 0;
        args_dev_chip00_18->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_18;
        device_arg_list_chip_00[18] = (uint32_t)(uintptr_t)args_dev_chip00_18;
        device_kernel_list_chip_00[18] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_l15_moe_swiglu");
        // Node ID: 19 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_down (__snax_bingo_kernel_dual_vc_l15_moe_down)
        __snax_bingo_kernel_dual_vc_l15_moe_down_args_t* args_dev_chip00_19 = (__snax_bingo_kernel_dual_vc_l15_moe_down_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_dual_vc_l15_moe_down_args_t));
        args_dev_chip00_19->shape_cfg_addr = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg)));
        args_dev_chip00_19->tcdm_base = (uint32_t)ptr_c0_l1_layout;
        args_dev_chip00_19->rescale_mult = 1;
        args_dev_chip00_19->rescale_shift = 0;
        args_dev_chip00_19->gating_sp_addr = 0;
        args_dev_chip00_19->cond_node_index = 0;
        args_dev_chip00_19->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_19;
        device_arg_list_chip_00[19] = (uint32_t)(uintptr_t)args_dev_chip00_19;
        device_kernel_list_chip_00[19] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_l15_moe_down");
        // Node ID: 20 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_20 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_20->src_addr = (uint64_t)ptr_c0_l1_layout + 1150208;
        args_host_chip00_20->dst_addr = (uint64_t)ptr_l3_shared_down_out;
        args_host_chip00_20->size = 131072;
        args_host_chip00_20->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_20;
        host_arg_list_chip_00[0] = (uint64_t)(uintptr_t)args_host_chip00_20;
        host_kernel_list_chip_00[0] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 21 Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_swiglu (__snax_bingo_kernel_dual_vc_l15_moe_swiglu)
        __snax_bingo_kernel_dual_vc_l15_moe_swiglu_args_t* args_dev_chip00_21 = (__snax_bingo_kernel_dual_vc_l15_moe_swiglu_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_dual_vc_l15_moe_swiglu_args_t));
        args_dev_chip00_21->shape_cfg_addr = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg)));
        args_dev_chip00_21->tcdm_base = (uint32_t)ptr_c1_l1_layout;
        args_dev_chip00_21->rescale_mult = 1;
        args_dev_chip00_21->rescale_shift = 0;
        args_dev_chip00_21->gating_sp_addr = 0;
        args_dev_chip00_21->cond_node_index = 0;
        args_dev_chip00_21->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_21;
        device_arg_list_chip_00[20] = (uint32_t)(uintptr_t)args_dev_chip00_21;
        device_kernel_list_chip_00[20] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_l15_moe_swiglu");
        // Node ID: 22 Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_down (__snax_bingo_kernel_dual_vc_l15_moe_down)
        __snax_bingo_kernel_dual_vc_l15_moe_down_args_t* args_dev_chip00_22 = (__snax_bingo_kernel_dual_vc_l15_moe_down_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_dual_vc_l15_moe_down_args_t));
        args_dev_chip00_22->shape_cfg_addr = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg)));
        args_dev_chip00_22->tcdm_base = (uint32_t)ptr_c1_l1_layout;
        args_dev_chip00_22->rescale_mult = 1;
        args_dev_chip00_22->rescale_shift = 0;
        args_dev_chip00_22->gating_sp_addr = 0;
        args_dev_chip00_22->cond_node_index = 0;
        args_dev_chip00_22->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_22;
        device_arg_list_chip_00[21] = (uint32_t)(uintptr_t)args_dev_chip00_22;
        device_kernel_list_chip_00[21] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_l15_moe_down");
        // Node ID: 23 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_23 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_23->src_addr = (uint64_t)ptr_c1_l1_layout + 1150208;
        args_host_chip00_23->dst_addr = (uint64_t)ptr_l3_shared_down_out + 131072;
        args_host_chip00_23->size = 131072;
        args_host_chip00_23->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_23;
        host_arg_list_chip_00[1] = (uint64_t)(uintptr_t)args_host_chip00_23;
        host_kernel_list_chip_00[1] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 24 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_dual_vc_gemm_full (__snax_bingo_kernel_dual_vc_gemm_full)
        __snax_bingo_kernel_dual_vc_gemm_full_args_t* args_dev_chip00_24 = (__snax_bingo_kernel_dual_vc_gemm_full_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_dual_vc_gemm_full_args_t));
        args_dev_chip00_24->input_A_addr = (uint32_t)ptr_c3_router_l1_a;
        args_dev_chip00_24->input_B0_addr = (uint32_t)ptr_c3_router_l1_b;
        args_dev_chip00_24->input_B1_addr = (uint32_t)ptr_c3_router_l1_b + 2048;
        args_dev_chip00_24->output_D0_addr = (uint32_t)ptr_c3_router_l1_d;
        args_dev_chip00_24->output_D1_addr = (uint32_t)ptr_c3_router_l1_d + 256;
        args_dev_chip00_24->M = 4;
        args_dev_chip00_24->K = 128;
        args_dev_chip00_24->N = 1;
        args_dev_chip00_24->array_shape = 0;
        args_dev_chip00_24->rescale_mult = 1;
        args_dev_chip00_24->rescale_shift = 0;
        args_dev_chip00_24->gating_sp_addr = 0;
        args_dev_chip00_24->cond_node_index = 0;
        args_dev_chip00_24->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_24;
        device_arg_list_chip_00[22] = (uint32_t)(uintptr_t)args_dev_chip00_24;
        device_kernel_list_chip_00[22] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_gemm_full");
        // Node ID: 25 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_25 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_25->src_addr = (uint64_t)ptr_c3_router_l1_d;
        args_host_chip00_25->dst_addr = (uint64_t)ptr_l3_router_out;
        args_host_chip00_25->size = 512;
        args_host_chip00_25->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_25;
        host_arg_list_chip_00[2] = (uint64_t)(uintptr_t)args_host_chip00_25;
        host_kernel_list_chip_00[2] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 26 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule (__host_bingo_kernel_moe_router_schedule)
        __host_bingo_kernel_moe_router_schedule_args_t* args_host_chip00_26 = (__host_bingo_kernel_moe_router_schedule_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_moe_router_schedule_args_t));
        args_host_chip00_26->total_tokens = 32;
        args_host_chip00_26->hardware_output_buffer_addr = (uint64_t)ptr_l3_router_out;
        args_host_chip00_26->global_indices_out_addr = (uint64_t)ptr_l3_topk_idx;
        args_host_chip00_26->global_scores_out_addr = (uint64_t)ptr_l3_topk_scores;
        args_host_chip00_26->expert_number_each_layer = 8;
        args_host_chip00_26->individual_expert_number_k = 2;
        args_host_chip00_26->mesh_row = 8;
        args_host_chip00_26->mesh_col = 4;
        args_host_chip00_26->router_m1 = 4;
        args_host_chip00_26->router_n1 = 1;
        args_host_chip00_26->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_26;
        host_arg_list_chip_00[3] = (uint64_t)(uintptr_t)args_host_chip00_26;
        host_kernel_list_chip_00[3] = (uint64_t)(uintptr_t)&__host_bingo_kernel_moe_router_schedule;
        // Node ID: 27 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_prepare_request (__host_bingo_kernel_moe_prepare_request)
        __host_bingo_kernel_moe_prepare_request_args_t* args_host_chip00_27 = (__host_bingo_kernel_moe_prepare_request_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_moe_prepare_request_args_t));
        args_host_chip00_27->expert_token_counts_addr = (uint64_t)ptr_l3_expert_counts;
        args_host_chip00_27->cam_state_addr = (uint64_t)ptr_l3_cam_state;
        args_host_chip00_27->request_out_addr = (uint64_t)ptr_l3_moe_request;
        args_host_chip00_27->schedule_out_addr = (uint64_t)ptr_l3_moe_schedule;
        args_host_chip00_27->expert_token_offsets_addr = (uint64_t)ptr_l3_expert_token_offsets;
        args_host_chip00_27->expert_token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_host_chip00_27->expert_token_kpos_addr = (uint64_t)ptr_l3_expert_token_kpos;
        args_host_chip00_27->n_experts = 8;
        args_host_chip00_27->topk_indices_l3 = (uint64_t)ptr_l3_topk_idx;
        args_host_chip00_27->M_total = 32;
        args_host_chip00_27->top_k = 2;
        args_host_chip00_27->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_27;
        host_arg_list_chip_00[4] = (uint64_t)(uintptr_t)args_host_chip00_27;
        host_kernel_list_chip_00[4] = (uint64_t)(uintptr_t)&__host_bingo_kernel_moe_prepare_request;
        // Node ID: 28 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_execute (__host_bingo_kernel_moe_execute)
        __host_bingo_kernel_moe_execute_args_t* args_host_chip00_28 = (__host_bingo_kernel_moe_execute_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_moe_execute_args_t));
        args_host_chip00_28->request_addr = (uint64_t)ptr_l3_moe_request;
        args_host_chip00_28->schedule_addr = (uint64_t)ptr_l3_moe_schedule;
        args_host_chip00_28->expert_token_offsets_addr = (uint64_t)ptr_l3_expert_token_offsets;
        args_host_chip00_28->expert_token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_host_chip00_28->expert_token_kpos_addr = (uint64_t)ptr_l3_expert_token_kpos;
        args_host_chip00_28->cam_state_addr = (uint64_t)ptr_l3_cam_state;
        args_host_chip00_28->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_host_chip00_28->topk_indices_l3 = (uint64_t)ptr_l3_topk_idx;
        args_host_chip00_28->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_host_chip00_28->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_host_chip00_28->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_host_chip00_28->c2_l1_b_gate = (uint64_t)ptr_c2_indiv_l1_b_gate;
        args_host_chip00_28->c2_l1_b_up = (uint64_t)ptr_c2_indiv_l1_b_up;
        args_host_chip00_28->c2_l1_b_down = (uint64_t)ptr_c2_indiv_l1_b_down;
        args_host_chip00_28->c2_l1_a = (uint64_t)ptr_c2_indiv_l1_a;
        args_host_chip00_28->c2_l1_d = (uint64_t)ptr_c2_indiv_l1_d;
        args_host_chip00_28->c2_l1_down_d = (uint64_t)ptr_c2_indiv_l1_down_d;
        args_host_chip00_28->c3_l1_b_gate = (uint64_t)ptr_c3_indiv_l1_b_gate;
        args_host_chip00_28->c3_l1_b_up = (uint64_t)ptr_c3_indiv_l1_b_up;
        args_host_chip00_28->c3_l1_b_down = (uint64_t)ptr_c3_indiv_l1_b_down;
        args_host_chip00_28->c3_l1_a = (uint64_t)ptr_c3_indiv_l1_a;
        args_host_chip00_28->c3_l1_d = (uint64_t)ptr_c3_indiv_l1_d;
        args_host_chip00_28->c3_l1_down_d = (uint64_t)ptr_c3_indiv_l1_down_d;
        args_host_chip00_28->output_l3_addr = (uint64_t)ptr_l3_indiv_down_out;
        args_host_chip00_28->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_host_chip00_28->A_token_bytes = 2048;
        args_host_chip00_28->indiv_B_expert_stride = 262144;
        args_host_chip00_28->indiv_down_B_expert_stride = 262144;
        args_host_chip00_28->down_D_bytes_per_expert = 65536;
        args_host_chip00_28->M_total = 32;
        args_host_chip00_28->top_k = 2;
        args_host_chip00_28->c2_dynamic_args_base = (uint64_t)ptr_c2_indiv_dyn_args;
        args_host_chip00_28->c3_dynamic_args_base = (uint64_t)ptr_c3_indiv_dyn_args;
        args_host_chip00_28->dynamic_arg_slot_bytes = 256;
        args_host_chip00_28->dynamic_num_slots = 4;
        args_host_chip00_28->c2_stage_base = (uint64_t)ptr_l3_c2_stage;
        args_host_chip00_28->c3_stage_base = (uint64_t)ptr_l3_c3_stage;
        args_host_chip00_28->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_28;
        host_arg_list_chip_00[5] = (uint64_t)(uintptr_t)args_host_chip00_28;
        host_kernel_list_chip_00[5] = (uint64_t)(uintptr_t)&__host_bingo_kernel_moe_execute;
        // Node ID: 29 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_29 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage);
        args_dev_chip00_29->ctrl = 0;
        args_dev_chip00_29->expert_id = 0;
        args_dev_chip00_29->token_start_rank = 0;
        args_dev_chip00_29->ntokens = 0;
        args_dev_chip00_29->m_s2_exec = 0;
        args_dev_chip00_29->m_s4_exec = 0;
        args_dev_chip00_29->wait_for_peer_slots = 0;
        args_dev_chip00_29->dma_slot_vd = 0;
        args_dev_chip00_29->s1_block_count = 2;
        args_dev_chip00_29->s3_block_count = 2;
        args_dev_chip00_29->dma_slot_expert_id[0] = -1;
        args_dev_chip00_29->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_29->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_29->dma_slot_expert_id[1] = -1;
        args_dev_chip00_29->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_29->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_29->dma_slot_expert_id[2] = -1;
        args_dev_chip00_29->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_29->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_29->dma_slot_expert_id[3] = -1;
        args_dev_chip00_29->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_29->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_29->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_29->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_29->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_29->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_29->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_29->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_29->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_29->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_29->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_29->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_29->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_29->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_29->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_29->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_29->A_token_bytes = 2048;
        args_dev_chip00_29->indiv_B_expert_stride = 262144;
        args_dev_chip00_29->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_29->indiv_B_tile_bytes = 131072;
        args_dev_chip00_29->indiv_D_tile_bytes = 16384;
        args_dev_chip00_29->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_29->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_29->indiv_N2 = 2;
        args_dev_chip00_29->indiv_down_N2 = 2;
        args_dev_chip00_29->indiv_K1 = 128;
        args_dev_chip00_29->indiv_N_per_block = 256;
        args_dev_chip00_29->indiv_down_K1 = 64;
        args_dev_chip00_29->indiv_down_N_per_block = 256;
        args_dev_chip00_29->rescale_mult = 1;
        args_dev_chip00_29->rescale_shift = 0;
        args_dev_chip00_29->output_expert_stride_bytes = 65536;
        args_dev_chip00_29->max_tokens_per_expert = 32;
        args_dev_chip00_29->gating_sp_addr = 0;
        args_dev_chip00_29->cond_node_index = 0;
        args_dev_chip00_29->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_29;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[23] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args;
        device_kernel_list_chip_00[23] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 30 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_30 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_30->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_30->block_idx = 0;
        args_dev_chip00_30->gating_sp_addr = 0;
        args_dev_chip00_30->cond_node_index = 0;
        args_dev_chip00_30->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_30;
        device_arg_list_chip_00[24] = (uint32_t)(uintptr_t)args_dev_chip00_30;
        device_kernel_list_chip_00[24] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 31 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_31 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_31->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_31->block_idx = 0;
        args_dev_chip00_31->gating_sp_addr = 0;
        args_dev_chip00_31->cond_node_index = 0;
        args_dev_chip00_31->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_31;
        device_arg_list_chip_00[25] = (uint32_t)(uintptr_t)args_dev_chip00_31;
        device_kernel_list_chip_00[25] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 32 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_32 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_32->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_32->block_idx = 1;
        args_dev_chip00_32->gating_sp_addr = 0;
        args_dev_chip00_32->cond_node_index = 0;
        args_dev_chip00_32->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_32;
        device_arg_list_chip_00[26] = (uint32_t)(uintptr_t)args_dev_chip00_32;
        device_kernel_list_chip_00[26] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 33 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_33 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_33->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_33->block_idx = 1;
        args_dev_chip00_33->gating_sp_addr = 0;
        args_dev_chip00_33->cond_node_index = 0;
        args_dev_chip00_33->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_33;
        device_arg_list_chip_00[27] = (uint32_t)(uintptr_t)args_dev_chip00_33;
        device_kernel_list_chip_00[27] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 34 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_34 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage);
        args_dev_chip00_34->ctrl = 0;
        args_dev_chip00_34->expert_id = 0;
        args_dev_chip00_34->token_start_rank = 0;
        args_dev_chip00_34->ntokens = 0;
        args_dev_chip00_34->m_s2_exec = 0;
        args_dev_chip00_34->m_s4_exec = 0;
        args_dev_chip00_34->wait_for_peer_slots = 0;
        args_dev_chip00_34->dma_slot_vd = 0;
        args_dev_chip00_34->s1_block_count = 2;
        args_dev_chip00_34->s3_block_count = 2;
        args_dev_chip00_34->dma_slot_expert_id[0] = -1;
        args_dev_chip00_34->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_34->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_34->dma_slot_expert_id[1] = -1;
        args_dev_chip00_34->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_34->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_34->dma_slot_expert_id[2] = -1;
        args_dev_chip00_34->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_34->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_34->dma_slot_expert_id[3] = -1;
        args_dev_chip00_34->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_34->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_34->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_34->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_34->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_34->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_34->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_34->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_34->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_34->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_34->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_34->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_34->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_34->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_34->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_34->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_34->A_token_bytes = 2048;
        args_dev_chip00_34->indiv_B_expert_stride = 262144;
        args_dev_chip00_34->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_34->indiv_B_tile_bytes = 131072;
        args_dev_chip00_34->indiv_D_tile_bytes = 16384;
        args_dev_chip00_34->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_34->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_34->indiv_N2 = 2;
        args_dev_chip00_34->indiv_down_N2 = 2;
        args_dev_chip00_34->indiv_K1 = 128;
        args_dev_chip00_34->indiv_N_per_block = 256;
        args_dev_chip00_34->indiv_down_K1 = 64;
        args_dev_chip00_34->indiv_down_N_per_block = 256;
        args_dev_chip00_34->rescale_mult = 1;
        args_dev_chip00_34->rescale_shift = 0;
        args_dev_chip00_34->output_expert_stride_bytes = 65536;
        args_dev_chip00_34->max_tokens_per_expert = 32;
        args_dev_chip00_34->gating_sp_addr = 0;
        args_dev_chip00_34->cond_node_index = 0;
        args_dev_chip00_34->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_34;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[28] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args;
        device_kernel_list_chip_00[28] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 35 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_35 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage);
        args_dev_chip00_35->ctrl = 0;
        args_dev_chip00_35->expert_id = 0;
        args_dev_chip00_35->token_start_rank = 0;
        args_dev_chip00_35->ntokens = 0;
        args_dev_chip00_35->m_s2_exec = 0;
        args_dev_chip00_35->m_s4_exec = 0;
        args_dev_chip00_35->wait_for_peer_slots = 0;
        args_dev_chip00_35->dma_slot_vd = 0;
        args_dev_chip00_35->s1_block_count = 2;
        args_dev_chip00_35->s3_block_count = 2;
        args_dev_chip00_35->dma_slot_expert_id[0] = -1;
        args_dev_chip00_35->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_35->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_35->dma_slot_expert_id[1] = -1;
        args_dev_chip00_35->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_35->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_35->dma_slot_expert_id[2] = -1;
        args_dev_chip00_35->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_35->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_35->dma_slot_expert_id[3] = -1;
        args_dev_chip00_35->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_35->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_35->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_35->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_35->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_35->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_35->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_35->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_35->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_35->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_35->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_35->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_35->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_35->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_35->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_35->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_35->A_token_bytes = 2048;
        args_dev_chip00_35->indiv_B_expert_stride = 262144;
        args_dev_chip00_35->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_35->indiv_B_tile_bytes = 131072;
        args_dev_chip00_35->indiv_D_tile_bytes = 16384;
        args_dev_chip00_35->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_35->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_35->indiv_N2 = 2;
        args_dev_chip00_35->indiv_down_N2 = 2;
        args_dev_chip00_35->indiv_K1 = 128;
        args_dev_chip00_35->indiv_N_per_block = 256;
        args_dev_chip00_35->indiv_down_K1 = 64;
        args_dev_chip00_35->indiv_down_N_per_block = 256;
        args_dev_chip00_35->rescale_mult = 1;
        args_dev_chip00_35->rescale_shift = 0;
        args_dev_chip00_35->output_expert_stride_bytes = 65536;
        args_dev_chip00_35->max_tokens_per_expert = 32;
        args_dev_chip00_35->gating_sp_addr = 0;
        args_dev_chip00_35->cond_node_index = 0;
        args_dev_chip00_35->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_35;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[29] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args;
        device_kernel_list_chip_00[29] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 36 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_36 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_36->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_36->block_idx = 0;
        args_dev_chip00_36->gating_sp_addr = 0;
        args_dev_chip00_36->cond_node_index = 0;
        args_dev_chip00_36->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_36;
        device_arg_list_chip_00[30] = (uint32_t)(uintptr_t)args_dev_chip00_36;
        device_kernel_list_chip_00[30] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 37 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_37 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_37->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_37->block_idx = 0;
        args_dev_chip00_37->gating_sp_addr = 0;
        args_dev_chip00_37->cond_node_index = 0;
        args_dev_chip00_37->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_37;
        device_arg_list_chip_00[31] = (uint32_t)(uintptr_t)args_dev_chip00_37;
        device_kernel_list_chip_00[31] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 38 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_38 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_38->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_38->block_idx = 1;
        args_dev_chip00_38->gating_sp_addr = 0;
        args_dev_chip00_38->cond_node_index = 0;
        args_dev_chip00_38->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_38;
        device_arg_list_chip_00[32] = (uint32_t)(uintptr_t)args_dev_chip00_38;
        device_kernel_list_chip_00[32] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 39 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_39 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_39->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args;
        args_dev_chip00_39->block_idx = 1;
        args_dev_chip00_39->gating_sp_addr = 0;
        args_dev_chip00_39->cond_node_index = 0;
        args_dev_chip00_39->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_39;
        device_arg_list_chip_00[33] = (uint32_t)(uintptr_t)args_dev_chip00_39;
        device_kernel_list_chip_00[33] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 40 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_40 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage);
        args_dev_chip00_40->ctrl = 0;
        args_dev_chip00_40->expert_id = 0;
        args_dev_chip00_40->token_start_rank = 0;
        args_dev_chip00_40->ntokens = 0;
        args_dev_chip00_40->m_s2_exec = 0;
        args_dev_chip00_40->m_s4_exec = 0;
        args_dev_chip00_40->wait_for_peer_slots = 0;
        args_dev_chip00_40->dma_slot_vd = 0;
        args_dev_chip00_40->s1_block_count = 2;
        args_dev_chip00_40->s3_block_count = 2;
        args_dev_chip00_40->dma_slot_expert_id[0] = -1;
        args_dev_chip00_40->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_40->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_40->dma_slot_expert_id[1] = -1;
        args_dev_chip00_40->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_40->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_40->dma_slot_expert_id[2] = -1;
        args_dev_chip00_40->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_40->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_40->dma_slot_expert_id[3] = -1;
        args_dev_chip00_40->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_40->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_40->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_40->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_40->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_40->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_40->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_40->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_40->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_40->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_40->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_40->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_40->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_40->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_40->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_40->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_40->A_token_bytes = 2048;
        args_dev_chip00_40->indiv_B_expert_stride = 262144;
        args_dev_chip00_40->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_40->indiv_B_tile_bytes = 131072;
        args_dev_chip00_40->indiv_D_tile_bytes = 16384;
        args_dev_chip00_40->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_40->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_40->indiv_N2 = 2;
        args_dev_chip00_40->indiv_down_N2 = 2;
        args_dev_chip00_40->indiv_K1 = 128;
        args_dev_chip00_40->indiv_N_per_block = 256;
        args_dev_chip00_40->indiv_down_K1 = 64;
        args_dev_chip00_40->indiv_down_N_per_block = 256;
        args_dev_chip00_40->rescale_mult = 1;
        args_dev_chip00_40->rescale_shift = 0;
        args_dev_chip00_40->output_expert_stride_bytes = 65536;
        args_dev_chip00_40->max_tokens_per_expert = 32;
        args_dev_chip00_40->gating_sp_addr = 0;
        args_dev_chip00_40->cond_node_index = 0;
        args_dev_chip00_40->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_40;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[34] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args;
        device_kernel_list_chip_00[34] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 41 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_41 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage);
        args_dev_chip00_41->ctrl = 0;
        args_dev_chip00_41->expert_id = 0;
        args_dev_chip00_41->token_start_rank = 0;
        args_dev_chip00_41->ntokens = 0;
        args_dev_chip00_41->m_s2_exec = 0;
        args_dev_chip00_41->m_s4_exec = 0;
        args_dev_chip00_41->wait_for_peer_slots = 0;
        args_dev_chip00_41->dma_slot_vd = 0;
        args_dev_chip00_41->s1_block_count = 2;
        args_dev_chip00_41->s3_block_count = 2;
        args_dev_chip00_41->dma_slot_expert_id[0] = -1;
        args_dev_chip00_41->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_41->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_41->dma_slot_expert_id[1] = -1;
        args_dev_chip00_41->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_41->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_41->dma_slot_expert_id[2] = -1;
        args_dev_chip00_41->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_41->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_41->dma_slot_expert_id[3] = -1;
        args_dev_chip00_41->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_41->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_41->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_41->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_41->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_41->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_41->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_41->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_41->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_41->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_41->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_41->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_41->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_41->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_41->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_41->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_41->A_token_bytes = 2048;
        args_dev_chip00_41->indiv_B_expert_stride = 262144;
        args_dev_chip00_41->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_41->indiv_B_tile_bytes = 131072;
        args_dev_chip00_41->indiv_D_tile_bytes = 16384;
        args_dev_chip00_41->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_41->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_41->indiv_N2 = 2;
        args_dev_chip00_41->indiv_down_N2 = 2;
        args_dev_chip00_41->indiv_K1 = 128;
        args_dev_chip00_41->indiv_N_per_block = 256;
        args_dev_chip00_41->indiv_down_K1 = 64;
        args_dev_chip00_41->indiv_down_N_per_block = 256;
        args_dev_chip00_41->rescale_mult = 1;
        args_dev_chip00_41->rescale_shift = 0;
        args_dev_chip00_41->output_expert_stride_bytes = 65536;
        args_dev_chip00_41->max_tokens_per_expert = 32;
        args_dev_chip00_41->gating_sp_addr = 0;
        args_dev_chip00_41->cond_node_index = 0;
        args_dev_chip00_41->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_41;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[35] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args;
        device_kernel_list_chip_00[35] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 42 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_42 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage);
        args_dev_chip00_42->ctrl = 0;
        args_dev_chip00_42->expert_id = 0;
        args_dev_chip00_42->token_start_rank = 0;
        args_dev_chip00_42->ntokens = 0;
        args_dev_chip00_42->m_s2_exec = 0;
        args_dev_chip00_42->m_s4_exec = 0;
        args_dev_chip00_42->wait_for_peer_slots = 0;
        args_dev_chip00_42->dma_slot_vd = 0;
        args_dev_chip00_42->s1_block_count = 2;
        args_dev_chip00_42->s3_block_count = 2;
        args_dev_chip00_42->dma_slot_expert_id[0] = -1;
        args_dev_chip00_42->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_42->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_42->dma_slot_expert_id[1] = -1;
        args_dev_chip00_42->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_42->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_42->dma_slot_expert_id[2] = -1;
        args_dev_chip00_42->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_42->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_42->dma_slot_expert_id[3] = -1;
        args_dev_chip00_42->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_42->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_42->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_42->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_42->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_42->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_42->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_42->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_42->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_42->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_42->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_42->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_42->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_42->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_42->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_42->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_42->A_token_bytes = 2048;
        args_dev_chip00_42->indiv_B_expert_stride = 262144;
        args_dev_chip00_42->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_42->indiv_B_tile_bytes = 131072;
        args_dev_chip00_42->indiv_D_tile_bytes = 16384;
        args_dev_chip00_42->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_42->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_42->indiv_N2 = 2;
        args_dev_chip00_42->indiv_down_N2 = 2;
        args_dev_chip00_42->indiv_K1 = 128;
        args_dev_chip00_42->indiv_N_per_block = 256;
        args_dev_chip00_42->indiv_down_K1 = 64;
        args_dev_chip00_42->indiv_down_N_per_block = 256;
        args_dev_chip00_42->rescale_mult = 1;
        args_dev_chip00_42->rescale_shift = 0;
        args_dev_chip00_42->output_expert_stride_bytes = 65536;
        args_dev_chip00_42->max_tokens_per_expert = 32;
        args_dev_chip00_42->gating_sp_addr = 0;
        args_dev_chip00_42->cond_node_index = 0;
        args_dev_chip00_42->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_42;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[36] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args;
        device_kernel_list_chip_00[36] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 43 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_43 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage);
        args_dev_chip00_43->ctrl = 0;
        args_dev_chip00_43->expert_id = 0;
        args_dev_chip00_43->token_start_rank = 0;
        args_dev_chip00_43->ntokens = 0;
        args_dev_chip00_43->m_s2_exec = 0;
        args_dev_chip00_43->m_s4_exec = 0;
        args_dev_chip00_43->wait_for_peer_slots = 0;
        args_dev_chip00_43->dma_slot_vd = 0;
        args_dev_chip00_43->s1_block_count = 2;
        args_dev_chip00_43->s3_block_count = 2;
        args_dev_chip00_43->dma_slot_expert_id[0] = -1;
        args_dev_chip00_43->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_43->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_43->dma_slot_expert_id[1] = -1;
        args_dev_chip00_43->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_43->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_43->dma_slot_expert_id[2] = -1;
        args_dev_chip00_43->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_43->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_43->dma_slot_expert_id[3] = -1;
        args_dev_chip00_43->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_43->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_43->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_43->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_43->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_43->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_43->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_43->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_43->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_43->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_43->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_43->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_43->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_43->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_43->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_43->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_43->A_token_bytes = 2048;
        args_dev_chip00_43->indiv_B_expert_stride = 262144;
        args_dev_chip00_43->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_43->indiv_B_tile_bytes = 131072;
        args_dev_chip00_43->indiv_D_tile_bytes = 16384;
        args_dev_chip00_43->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_43->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_43->indiv_N2 = 2;
        args_dev_chip00_43->indiv_down_N2 = 2;
        args_dev_chip00_43->indiv_K1 = 128;
        args_dev_chip00_43->indiv_N_per_block = 256;
        args_dev_chip00_43->indiv_down_K1 = 64;
        args_dev_chip00_43->indiv_down_N_per_block = 256;
        args_dev_chip00_43->rescale_mult = 1;
        args_dev_chip00_43->rescale_shift = 0;
        args_dev_chip00_43->output_expert_stride_bytes = 65536;
        args_dev_chip00_43->max_tokens_per_expert = 32;
        args_dev_chip00_43->gating_sp_addr = 0;
        args_dev_chip00_43->cond_node_index = 0;
        args_dev_chip00_43->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_43;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[37] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args;
        device_kernel_list_chip_00[37] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 44 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_44 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_44->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_44->block_idx = 0;
        args_dev_chip00_44->gating_sp_addr = 0;
        args_dev_chip00_44->cond_node_index = 0;
        args_dev_chip00_44->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_44;
        device_arg_list_chip_00[38] = (uint32_t)(uintptr_t)args_dev_chip00_44;
        device_kernel_list_chip_00[38] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 45 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_45 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_45->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_45->block_idx = 0;
        args_dev_chip00_45->gating_sp_addr = 0;
        args_dev_chip00_45->cond_node_index = 0;
        args_dev_chip00_45->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_45;
        device_arg_list_chip_00[39] = (uint32_t)(uintptr_t)args_dev_chip00_45;
        device_kernel_list_chip_00[39] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 46 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_46 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_46->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_46->block_idx = 1;
        args_dev_chip00_46->gating_sp_addr = 0;
        args_dev_chip00_46->cond_node_index = 0;
        args_dev_chip00_46->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_46;
        device_arg_list_chip_00[40] = (uint32_t)(uintptr_t)args_dev_chip00_46;
        device_kernel_list_chip_00[40] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 47 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_47 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_47->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_47->block_idx = 1;
        args_dev_chip00_47->gating_sp_addr = 0;
        args_dev_chip00_47->cond_node_index = 0;
        args_dev_chip00_47->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_47;
        device_arg_list_chip_00[41] = (uint32_t)(uintptr_t)args_dev_chip00_47;
        device_kernel_list_chip_00[41] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 48 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_48 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage);
        args_dev_chip00_48->ctrl = 0;
        args_dev_chip00_48->expert_id = 0;
        args_dev_chip00_48->token_start_rank = 0;
        args_dev_chip00_48->ntokens = 0;
        args_dev_chip00_48->m_s2_exec = 0;
        args_dev_chip00_48->m_s4_exec = 0;
        args_dev_chip00_48->wait_for_peer_slots = 0;
        args_dev_chip00_48->dma_slot_vd = 0;
        args_dev_chip00_48->s1_block_count = 2;
        args_dev_chip00_48->s3_block_count = 2;
        args_dev_chip00_48->dma_slot_expert_id[0] = -1;
        args_dev_chip00_48->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_48->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_48->dma_slot_expert_id[1] = -1;
        args_dev_chip00_48->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_48->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_48->dma_slot_expert_id[2] = -1;
        args_dev_chip00_48->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_48->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_48->dma_slot_expert_id[3] = -1;
        args_dev_chip00_48->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_48->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_48->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_48->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_48->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_48->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_48->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_48->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_48->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_48->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_48->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_48->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_48->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_48->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_48->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_48->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_48->A_token_bytes = 2048;
        args_dev_chip00_48->indiv_B_expert_stride = 262144;
        args_dev_chip00_48->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_48->indiv_B_tile_bytes = 131072;
        args_dev_chip00_48->indiv_D_tile_bytes = 16384;
        args_dev_chip00_48->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_48->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_48->indiv_N2 = 2;
        args_dev_chip00_48->indiv_down_N2 = 2;
        args_dev_chip00_48->indiv_K1 = 128;
        args_dev_chip00_48->indiv_N_per_block = 256;
        args_dev_chip00_48->indiv_down_K1 = 64;
        args_dev_chip00_48->indiv_down_N_per_block = 256;
        args_dev_chip00_48->rescale_mult = 1;
        args_dev_chip00_48->rescale_shift = 0;
        args_dev_chip00_48->output_expert_stride_bytes = 65536;
        args_dev_chip00_48->max_tokens_per_expert = 32;
        args_dev_chip00_48->gating_sp_addr = 0;
        args_dev_chip00_48->cond_node_index = 0;
        args_dev_chip00_48->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_48;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[42] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args;
        device_kernel_list_chip_00[42] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 49 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_49 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage);
        args_dev_chip00_49->ctrl = 0;
        args_dev_chip00_49->expert_id = 0;
        args_dev_chip00_49->token_start_rank = 0;
        args_dev_chip00_49->ntokens = 0;
        args_dev_chip00_49->m_s2_exec = 0;
        args_dev_chip00_49->m_s4_exec = 0;
        args_dev_chip00_49->wait_for_peer_slots = 0;
        args_dev_chip00_49->dma_slot_vd = 0;
        args_dev_chip00_49->s1_block_count = 2;
        args_dev_chip00_49->s3_block_count = 2;
        args_dev_chip00_49->dma_slot_expert_id[0] = -1;
        args_dev_chip00_49->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_49->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_49->dma_slot_expert_id[1] = -1;
        args_dev_chip00_49->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_49->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_49->dma_slot_expert_id[2] = -1;
        args_dev_chip00_49->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_49->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_49->dma_slot_expert_id[3] = -1;
        args_dev_chip00_49->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_49->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_49->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_49->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_49->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_49->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_49->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_49->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_49->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_49->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_49->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_49->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_49->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_49->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_49->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_49->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_49->A_token_bytes = 2048;
        args_dev_chip00_49->indiv_B_expert_stride = 262144;
        args_dev_chip00_49->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_49->indiv_B_tile_bytes = 131072;
        args_dev_chip00_49->indiv_D_tile_bytes = 16384;
        args_dev_chip00_49->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_49->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_49->indiv_N2 = 2;
        args_dev_chip00_49->indiv_down_N2 = 2;
        args_dev_chip00_49->indiv_K1 = 128;
        args_dev_chip00_49->indiv_N_per_block = 256;
        args_dev_chip00_49->indiv_down_K1 = 64;
        args_dev_chip00_49->indiv_down_N_per_block = 256;
        args_dev_chip00_49->rescale_mult = 1;
        args_dev_chip00_49->rescale_shift = 0;
        args_dev_chip00_49->output_expert_stride_bytes = 65536;
        args_dev_chip00_49->max_tokens_per_expert = 32;
        args_dev_chip00_49->gating_sp_addr = 0;
        args_dev_chip00_49->cond_node_index = 0;
        args_dev_chip00_49->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_49;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[43] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args;
        device_kernel_list_chip_00[43] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 50 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_50 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_50->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_50->block_idx = 0;
        args_dev_chip00_50->gating_sp_addr = 0;
        args_dev_chip00_50->cond_node_index = 0;
        args_dev_chip00_50->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_50;
        device_arg_list_chip_00[44] = (uint32_t)(uintptr_t)args_dev_chip00_50;
        device_kernel_list_chip_00[44] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 51 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_51 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_51->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_51->block_idx = 0;
        args_dev_chip00_51->gating_sp_addr = 0;
        args_dev_chip00_51->cond_node_index = 0;
        args_dev_chip00_51->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_51;
        device_arg_list_chip_00[45] = (uint32_t)(uintptr_t)args_dev_chip00_51;
        device_kernel_list_chip_00[45] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 52 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_52 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_52->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_52->block_idx = 1;
        args_dev_chip00_52->gating_sp_addr = 0;
        args_dev_chip00_52->cond_node_index = 0;
        args_dev_chip00_52->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_52;
        device_arg_list_chip_00[46] = (uint32_t)(uintptr_t)args_dev_chip00_52;
        device_kernel_list_chip_00[46] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 53 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_53 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_53->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args;
        args_dev_chip00_53->block_idx = 1;
        args_dev_chip00_53->gating_sp_addr = 0;
        args_dev_chip00_53->cond_node_index = 0;
        args_dev_chip00_53->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_53;
        device_arg_list_chip_00[47] = (uint32_t)(uintptr_t)args_dev_chip00_53;
        device_kernel_list_chip_00[47] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 54 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_54 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage);
        args_dev_chip00_54->ctrl = 0;
        args_dev_chip00_54->expert_id = 0;
        args_dev_chip00_54->token_start_rank = 0;
        args_dev_chip00_54->ntokens = 0;
        args_dev_chip00_54->m_s2_exec = 0;
        args_dev_chip00_54->m_s4_exec = 0;
        args_dev_chip00_54->wait_for_peer_slots = 0;
        args_dev_chip00_54->dma_slot_vd = 0;
        args_dev_chip00_54->s1_block_count = 2;
        args_dev_chip00_54->s3_block_count = 2;
        args_dev_chip00_54->dma_slot_expert_id[0] = -1;
        args_dev_chip00_54->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_54->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_54->dma_slot_expert_id[1] = -1;
        args_dev_chip00_54->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_54->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_54->dma_slot_expert_id[2] = -1;
        args_dev_chip00_54->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_54->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_54->dma_slot_expert_id[3] = -1;
        args_dev_chip00_54->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_54->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_54->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_54->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_54->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_54->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_54->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_54->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_54->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_54->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_54->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_54->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_54->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_54->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_54->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_54->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_54->A_token_bytes = 2048;
        args_dev_chip00_54->indiv_B_expert_stride = 262144;
        args_dev_chip00_54->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_54->indiv_B_tile_bytes = 131072;
        args_dev_chip00_54->indiv_D_tile_bytes = 16384;
        args_dev_chip00_54->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_54->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_54->indiv_N2 = 2;
        args_dev_chip00_54->indiv_down_N2 = 2;
        args_dev_chip00_54->indiv_K1 = 128;
        args_dev_chip00_54->indiv_N_per_block = 256;
        args_dev_chip00_54->indiv_down_K1 = 64;
        args_dev_chip00_54->indiv_down_N_per_block = 256;
        args_dev_chip00_54->rescale_mult = 1;
        args_dev_chip00_54->rescale_shift = 0;
        args_dev_chip00_54->output_expert_stride_bytes = 65536;
        args_dev_chip00_54->max_tokens_per_expert = 32;
        args_dev_chip00_54->gating_sp_addr = 0;
        args_dev_chip00_54->cond_node_index = 0;
        args_dev_chip00_54->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_54;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[48] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args;
        device_kernel_list_chip_00[48] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 55 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_55 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage);
        args_dev_chip00_55->ctrl = 0;
        args_dev_chip00_55->expert_id = 0;
        args_dev_chip00_55->token_start_rank = 0;
        args_dev_chip00_55->ntokens = 0;
        args_dev_chip00_55->m_s2_exec = 0;
        args_dev_chip00_55->m_s4_exec = 0;
        args_dev_chip00_55->wait_for_peer_slots = 0;
        args_dev_chip00_55->dma_slot_vd = 0;
        args_dev_chip00_55->s1_block_count = 2;
        args_dev_chip00_55->s3_block_count = 2;
        args_dev_chip00_55->dma_slot_expert_id[0] = -1;
        args_dev_chip00_55->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_55->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_55->dma_slot_expert_id[1] = -1;
        args_dev_chip00_55->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_55->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_55->dma_slot_expert_id[2] = -1;
        args_dev_chip00_55->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_55->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_55->dma_slot_expert_id[3] = -1;
        args_dev_chip00_55->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_55->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_55->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_55->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_55->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_55->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_55->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_55->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_55->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_55->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_55->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_55->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_55->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_55->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_55->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_55->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_55->A_token_bytes = 2048;
        args_dev_chip00_55->indiv_B_expert_stride = 262144;
        args_dev_chip00_55->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_55->indiv_B_tile_bytes = 131072;
        args_dev_chip00_55->indiv_D_tile_bytes = 16384;
        args_dev_chip00_55->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_55->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_55->indiv_N2 = 2;
        args_dev_chip00_55->indiv_down_N2 = 2;
        args_dev_chip00_55->indiv_K1 = 128;
        args_dev_chip00_55->indiv_N_per_block = 256;
        args_dev_chip00_55->indiv_down_K1 = 64;
        args_dev_chip00_55->indiv_down_N_per_block = 256;
        args_dev_chip00_55->rescale_mult = 1;
        args_dev_chip00_55->rescale_shift = 0;
        args_dev_chip00_55->output_expert_stride_bytes = 65536;
        args_dev_chip00_55->max_tokens_per_expert = 32;
        args_dev_chip00_55->gating_sp_addr = 0;
        args_dev_chip00_55->cond_node_index = 0;
        args_dev_chip00_55->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_55;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[49] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args;
        device_kernel_list_chip_00[49] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 56 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_56 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage);
        args_dev_chip00_56->ctrl = 0;
        args_dev_chip00_56->expert_id = 0;
        args_dev_chip00_56->token_start_rank = 0;
        args_dev_chip00_56->ntokens = 0;
        args_dev_chip00_56->m_s2_exec = 0;
        args_dev_chip00_56->m_s4_exec = 0;
        args_dev_chip00_56->wait_for_peer_slots = 0;
        args_dev_chip00_56->dma_slot_vd = 0;
        args_dev_chip00_56->s1_block_count = 2;
        args_dev_chip00_56->s3_block_count = 2;
        args_dev_chip00_56->dma_slot_expert_id[0] = -1;
        args_dev_chip00_56->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_56->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_56->dma_slot_expert_id[1] = -1;
        args_dev_chip00_56->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_56->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_56->dma_slot_expert_id[2] = -1;
        args_dev_chip00_56->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_56->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_56->dma_slot_expert_id[3] = -1;
        args_dev_chip00_56->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_56->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_56->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_56->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_56->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_56->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_56->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_56->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_56->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_56->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_56->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_56->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_56->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_56->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_56->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_56->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_56->A_token_bytes = 2048;
        args_dev_chip00_56->indiv_B_expert_stride = 262144;
        args_dev_chip00_56->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_56->indiv_B_tile_bytes = 131072;
        args_dev_chip00_56->indiv_D_tile_bytes = 16384;
        args_dev_chip00_56->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_56->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_56->indiv_N2 = 2;
        args_dev_chip00_56->indiv_down_N2 = 2;
        args_dev_chip00_56->indiv_K1 = 128;
        args_dev_chip00_56->indiv_N_per_block = 256;
        args_dev_chip00_56->indiv_down_K1 = 64;
        args_dev_chip00_56->indiv_down_N_per_block = 256;
        args_dev_chip00_56->rescale_mult = 1;
        args_dev_chip00_56->rescale_shift = 0;
        args_dev_chip00_56->output_expert_stride_bytes = 65536;
        args_dev_chip00_56->max_tokens_per_expert = 32;
        args_dev_chip00_56->gating_sp_addr = 0;
        args_dev_chip00_56->cond_node_index = 0;
        args_dev_chip00_56->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_56;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[50] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args;
        device_kernel_list_chip_00[50] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 57 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_57 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 256);
        args_dev_chip00_57->ctrl = 0;
        args_dev_chip00_57->expert_id = 0;
        args_dev_chip00_57->token_start_rank = 0;
        args_dev_chip00_57->ntokens = 0;
        args_dev_chip00_57->m_s2_exec = 0;
        args_dev_chip00_57->m_s4_exec = 0;
        args_dev_chip00_57->wait_for_peer_slots = 0;
        args_dev_chip00_57->dma_slot_vd = 0;
        args_dev_chip00_57->s1_block_count = 2;
        args_dev_chip00_57->s3_block_count = 2;
        args_dev_chip00_57->dma_slot_expert_id[0] = -1;
        args_dev_chip00_57->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_57->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_57->dma_slot_expert_id[1] = -1;
        args_dev_chip00_57->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_57->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_57->dma_slot_expert_id[2] = -1;
        args_dev_chip00_57->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_57->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_57->dma_slot_expert_id[3] = -1;
        args_dev_chip00_57->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_57->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_57->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_57->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_57->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_57->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_57->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_57->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_57->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_57->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_57->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_57->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_57->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_57->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_57->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_57->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_57->A_token_bytes = 2048;
        args_dev_chip00_57->indiv_B_expert_stride = 262144;
        args_dev_chip00_57->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_57->indiv_B_tile_bytes = 131072;
        args_dev_chip00_57->indiv_D_tile_bytes = 16384;
        args_dev_chip00_57->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_57->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_57->indiv_N2 = 2;
        args_dev_chip00_57->indiv_down_N2 = 2;
        args_dev_chip00_57->indiv_K1 = 128;
        args_dev_chip00_57->indiv_N_per_block = 256;
        args_dev_chip00_57->indiv_down_K1 = 64;
        args_dev_chip00_57->indiv_down_N_per_block = 256;
        args_dev_chip00_57->rescale_mult = 1;
        args_dev_chip00_57->rescale_shift = 0;
        args_dev_chip00_57->output_expert_stride_bytes = 65536;
        args_dev_chip00_57->max_tokens_per_expert = 32;
        args_dev_chip00_57->gating_sp_addr = 0;
        args_dev_chip00_57->cond_node_index = 0;
        args_dev_chip00_57->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_57;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[51] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 256;
        device_kernel_list_chip_00[51] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 58 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_58 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_58->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_58->block_idx = 0;
        args_dev_chip00_58->gating_sp_addr = 0;
        args_dev_chip00_58->cond_node_index = 0;
        args_dev_chip00_58->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_58;
        device_arg_list_chip_00[52] = (uint32_t)(uintptr_t)args_dev_chip00_58;
        device_kernel_list_chip_00[52] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 59 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_59 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_59->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_59->block_idx = 0;
        args_dev_chip00_59->gating_sp_addr = 0;
        args_dev_chip00_59->cond_node_index = 0;
        args_dev_chip00_59->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_59;
        device_arg_list_chip_00[53] = (uint32_t)(uintptr_t)args_dev_chip00_59;
        device_kernel_list_chip_00[53] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 60 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_60 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_60->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_60->block_idx = 1;
        args_dev_chip00_60->gating_sp_addr = 0;
        args_dev_chip00_60->cond_node_index = 0;
        args_dev_chip00_60->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_60;
        device_arg_list_chip_00[54] = (uint32_t)(uintptr_t)args_dev_chip00_60;
        device_kernel_list_chip_00[54] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 61 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_61 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_61->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_61->block_idx = 1;
        args_dev_chip00_61->gating_sp_addr = 0;
        args_dev_chip00_61->cond_node_index = 0;
        args_dev_chip00_61->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_61;
        device_arg_list_chip_00[55] = (uint32_t)(uintptr_t)args_dev_chip00_61;
        device_kernel_list_chip_00[55] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 62 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_62 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 256);
        args_dev_chip00_62->ctrl = 0;
        args_dev_chip00_62->expert_id = 0;
        args_dev_chip00_62->token_start_rank = 0;
        args_dev_chip00_62->ntokens = 0;
        args_dev_chip00_62->m_s2_exec = 0;
        args_dev_chip00_62->m_s4_exec = 0;
        args_dev_chip00_62->wait_for_peer_slots = 0;
        args_dev_chip00_62->dma_slot_vd = 0;
        args_dev_chip00_62->s1_block_count = 2;
        args_dev_chip00_62->s3_block_count = 2;
        args_dev_chip00_62->dma_slot_expert_id[0] = -1;
        args_dev_chip00_62->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_62->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_62->dma_slot_expert_id[1] = -1;
        args_dev_chip00_62->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_62->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_62->dma_slot_expert_id[2] = -1;
        args_dev_chip00_62->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_62->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_62->dma_slot_expert_id[3] = -1;
        args_dev_chip00_62->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_62->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_62->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_62->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_62->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_62->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_62->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_62->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_62->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_62->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_62->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_62->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_62->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_62->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_62->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_62->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_62->A_token_bytes = 2048;
        args_dev_chip00_62->indiv_B_expert_stride = 262144;
        args_dev_chip00_62->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_62->indiv_B_tile_bytes = 131072;
        args_dev_chip00_62->indiv_D_tile_bytes = 16384;
        args_dev_chip00_62->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_62->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_62->indiv_N2 = 2;
        args_dev_chip00_62->indiv_down_N2 = 2;
        args_dev_chip00_62->indiv_K1 = 128;
        args_dev_chip00_62->indiv_N_per_block = 256;
        args_dev_chip00_62->indiv_down_K1 = 64;
        args_dev_chip00_62->indiv_down_N_per_block = 256;
        args_dev_chip00_62->rescale_mult = 1;
        args_dev_chip00_62->rescale_shift = 0;
        args_dev_chip00_62->output_expert_stride_bytes = 65536;
        args_dev_chip00_62->max_tokens_per_expert = 32;
        args_dev_chip00_62->gating_sp_addr = 0;
        args_dev_chip00_62->cond_node_index = 0;
        args_dev_chip00_62->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_62;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[56] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 256;
        device_kernel_list_chip_00[56] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 63 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_63 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 256);
        args_dev_chip00_63->ctrl = 0;
        args_dev_chip00_63->expert_id = 0;
        args_dev_chip00_63->token_start_rank = 0;
        args_dev_chip00_63->ntokens = 0;
        args_dev_chip00_63->m_s2_exec = 0;
        args_dev_chip00_63->m_s4_exec = 0;
        args_dev_chip00_63->wait_for_peer_slots = 0;
        args_dev_chip00_63->dma_slot_vd = 0;
        args_dev_chip00_63->s1_block_count = 2;
        args_dev_chip00_63->s3_block_count = 2;
        args_dev_chip00_63->dma_slot_expert_id[0] = -1;
        args_dev_chip00_63->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_63->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_63->dma_slot_expert_id[1] = -1;
        args_dev_chip00_63->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_63->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_63->dma_slot_expert_id[2] = -1;
        args_dev_chip00_63->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_63->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_63->dma_slot_expert_id[3] = -1;
        args_dev_chip00_63->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_63->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_63->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_63->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_63->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_63->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_63->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_63->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_63->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_63->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_63->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_63->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_63->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_63->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_63->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_63->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_63->A_token_bytes = 2048;
        args_dev_chip00_63->indiv_B_expert_stride = 262144;
        args_dev_chip00_63->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_63->indiv_B_tile_bytes = 131072;
        args_dev_chip00_63->indiv_D_tile_bytes = 16384;
        args_dev_chip00_63->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_63->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_63->indiv_N2 = 2;
        args_dev_chip00_63->indiv_down_N2 = 2;
        args_dev_chip00_63->indiv_K1 = 128;
        args_dev_chip00_63->indiv_N_per_block = 256;
        args_dev_chip00_63->indiv_down_K1 = 64;
        args_dev_chip00_63->indiv_down_N_per_block = 256;
        args_dev_chip00_63->rescale_mult = 1;
        args_dev_chip00_63->rescale_shift = 0;
        args_dev_chip00_63->output_expert_stride_bytes = 65536;
        args_dev_chip00_63->max_tokens_per_expert = 32;
        args_dev_chip00_63->gating_sp_addr = 0;
        args_dev_chip00_63->cond_node_index = 0;
        args_dev_chip00_63->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_63;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[57] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 256;
        device_kernel_list_chip_00[57] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 64 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_64 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_64->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_64->block_idx = 0;
        args_dev_chip00_64->gating_sp_addr = 0;
        args_dev_chip00_64->cond_node_index = 0;
        args_dev_chip00_64->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_64;
        device_arg_list_chip_00[58] = (uint32_t)(uintptr_t)args_dev_chip00_64;
        device_kernel_list_chip_00[58] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 65 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_65 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_65->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_65->block_idx = 0;
        args_dev_chip00_65->gating_sp_addr = 0;
        args_dev_chip00_65->cond_node_index = 0;
        args_dev_chip00_65->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_65;
        device_arg_list_chip_00[59] = (uint32_t)(uintptr_t)args_dev_chip00_65;
        device_kernel_list_chip_00[59] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 66 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_66 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_66->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_66->block_idx = 1;
        args_dev_chip00_66->gating_sp_addr = 0;
        args_dev_chip00_66->cond_node_index = 0;
        args_dev_chip00_66->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_66;
        device_arg_list_chip_00[60] = (uint32_t)(uintptr_t)args_dev_chip00_66;
        device_kernel_list_chip_00[60] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 67 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_67 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_67->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 256;
        args_dev_chip00_67->block_idx = 1;
        args_dev_chip00_67->gating_sp_addr = 0;
        args_dev_chip00_67->cond_node_index = 0;
        args_dev_chip00_67->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_67;
        device_arg_list_chip_00[61] = (uint32_t)(uintptr_t)args_dev_chip00_67;
        device_kernel_list_chip_00[61] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 68 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_68 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 256);
        args_dev_chip00_68->ctrl = 0;
        args_dev_chip00_68->expert_id = 0;
        args_dev_chip00_68->token_start_rank = 0;
        args_dev_chip00_68->ntokens = 0;
        args_dev_chip00_68->m_s2_exec = 0;
        args_dev_chip00_68->m_s4_exec = 0;
        args_dev_chip00_68->wait_for_peer_slots = 0;
        args_dev_chip00_68->dma_slot_vd = 0;
        args_dev_chip00_68->s1_block_count = 2;
        args_dev_chip00_68->s3_block_count = 2;
        args_dev_chip00_68->dma_slot_expert_id[0] = -1;
        args_dev_chip00_68->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_68->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_68->dma_slot_expert_id[1] = -1;
        args_dev_chip00_68->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_68->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_68->dma_slot_expert_id[2] = -1;
        args_dev_chip00_68->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_68->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_68->dma_slot_expert_id[3] = -1;
        args_dev_chip00_68->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_68->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_68->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_68->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_68->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_68->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_68->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_68->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_68->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_68->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_68->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_68->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_68->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_68->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_68->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_68->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_68->A_token_bytes = 2048;
        args_dev_chip00_68->indiv_B_expert_stride = 262144;
        args_dev_chip00_68->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_68->indiv_B_tile_bytes = 131072;
        args_dev_chip00_68->indiv_D_tile_bytes = 16384;
        args_dev_chip00_68->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_68->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_68->indiv_N2 = 2;
        args_dev_chip00_68->indiv_down_N2 = 2;
        args_dev_chip00_68->indiv_K1 = 128;
        args_dev_chip00_68->indiv_N_per_block = 256;
        args_dev_chip00_68->indiv_down_K1 = 64;
        args_dev_chip00_68->indiv_down_N_per_block = 256;
        args_dev_chip00_68->rescale_mult = 1;
        args_dev_chip00_68->rescale_shift = 0;
        args_dev_chip00_68->output_expert_stride_bytes = 65536;
        args_dev_chip00_68->max_tokens_per_expert = 32;
        args_dev_chip00_68->gating_sp_addr = 0;
        args_dev_chip00_68->cond_node_index = 0;
        args_dev_chip00_68->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_68;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[62] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 256;
        device_kernel_list_chip_00[62] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 69 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_69 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 256);
        args_dev_chip00_69->ctrl = 0;
        args_dev_chip00_69->expert_id = 0;
        args_dev_chip00_69->token_start_rank = 0;
        args_dev_chip00_69->ntokens = 0;
        args_dev_chip00_69->m_s2_exec = 0;
        args_dev_chip00_69->m_s4_exec = 0;
        args_dev_chip00_69->wait_for_peer_slots = 0;
        args_dev_chip00_69->dma_slot_vd = 0;
        args_dev_chip00_69->s1_block_count = 2;
        args_dev_chip00_69->s3_block_count = 2;
        args_dev_chip00_69->dma_slot_expert_id[0] = -1;
        args_dev_chip00_69->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_69->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_69->dma_slot_expert_id[1] = -1;
        args_dev_chip00_69->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_69->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_69->dma_slot_expert_id[2] = -1;
        args_dev_chip00_69->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_69->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_69->dma_slot_expert_id[3] = -1;
        args_dev_chip00_69->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_69->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_69->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_69->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_69->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_69->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_69->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_69->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_69->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_69->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_69->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_69->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_69->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_69->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_69->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_69->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_69->A_token_bytes = 2048;
        args_dev_chip00_69->indiv_B_expert_stride = 262144;
        args_dev_chip00_69->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_69->indiv_B_tile_bytes = 131072;
        args_dev_chip00_69->indiv_D_tile_bytes = 16384;
        args_dev_chip00_69->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_69->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_69->indiv_N2 = 2;
        args_dev_chip00_69->indiv_down_N2 = 2;
        args_dev_chip00_69->indiv_K1 = 128;
        args_dev_chip00_69->indiv_N_per_block = 256;
        args_dev_chip00_69->indiv_down_K1 = 64;
        args_dev_chip00_69->indiv_down_N_per_block = 256;
        args_dev_chip00_69->rescale_mult = 1;
        args_dev_chip00_69->rescale_shift = 0;
        args_dev_chip00_69->output_expert_stride_bytes = 65536;
        args_dev_chip00_69->max_tokens_per_expert = 32;
        args_dev_chip00_69->gating_sp_addr = 0;
        args_dev_chip00_69->cond_node_index = 0;
        args_dev_chip00_69->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_69;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[63] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 256;
        device_kernel_list_chip_00[63] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 70 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_70 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 256);
        args_dev_chip00_70->ctrl = 0;
        args_dev_chip00_70->expert_id = 0;
        args_dev_chip00_70->token_start_rank = 0;
        args_dev_chip00_70->ntokens = 0;
        args_dev_chip00_70->m_s2_exec = 0;
        args_dev_chip00_70->m_s4_exec = 0;
        args_dev_chip00_70->wait_for_peer_slots = 0;
        args_dev_chip00_70->dma_slot_vd = 0;
        args_dev_chip00_70->s1_block_count = 2;
        args_dev_chip00_70->s3_block_count = 2;
        args_dev_chip00_70->dma_slot_expert_id[0] = -1;
        args_dev_chip00_70->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_70->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_70->dma_slot_expert_id[1] = -1;
        args_dev_chip00_70->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_70->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_70->dma_slot_expert_id[2] = -1;
        args_dev_chip00_70->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_70->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_70->dma_slot_expert_id[3] = -1;
        args_dev_chip00_70->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_70->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_70->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_70->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_70->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_70->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_70->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_70->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_70->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_70->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_70->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_70->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_70->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_70->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_70->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_70->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_70->A_token_bytes = 2048;
        args_dev_chip00_70->indiv_B_expert_stride = 262144;
        args_dev_chip00_70->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_70->indiv_B_tile_bytes = 131072;
        args_dev_chip00_70->indiv_D_tile_bytes = 16384;
        args_dev_chip00_70->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_70->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_70->indiv_N2 = 2;
        args_dev_chip00_70->indiv_down_N2 = 2;
        args_dev_chip00_70->indiv_K1 = 128;
        args_dev_chip00_70->indiv_N_per_block = 256;
        args_dev_chip00_70->indiv_down_K1 = 64;
        args_dev_chip00_70->indiv_down_N_per_block = 256;
        args_dev_chip00_70->rescale_mult = 1;
        args_dev_chip00_70->rescale_shift = 0;
        args_dev_chip00_70->output_expert_stride_bytes = 65536;
        args_dev_chip00_70->max_tokens_per_expert = 32;
        args_dev_chip00_70->gating_sp_addr = 0;
        args_dev_chip00_70->cond_node_index = 0;
        args_dev_chip00_70->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_70;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[64] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 256;
        device_kernel_list_chip_00[64] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 71 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_71 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 256);
        args_dev_chip00_71->ctrl = 0;
        args_dev_chip00_71->expert_id = 0;
        args_dev_chip00_71->token_start_rank = 0;
        args_dev_chip00_71->ntokens = 0;
        args_dev_chip00_71->m_s2_exec = 0;
        args_dev_chip00_71->m_s4_exec = 0;
        args_dev_chip00_71->wait_for_peer_slots = 0;
        args_dev_chip00_71->dma_slot_vd = 0;
        args_dev_chip00_71->s1_block_count = 2;
        args_dev_chip00_71->s3_block_count = 2;
        args_dev_chip00_71->dma_slot_expert_id[0] = -1;
        args_dev_chip00_71->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_71->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_71->dma_slot_expert_id[1] = -1;
        args_dev_chip00_71->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_71->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_71->dma_slot_expert_id[2] = -1;
        args_dev_chip00_71->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_71->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_71->dma_slot_expert_id[3] = -1;
        args_dev_chip00_71->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_71->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_71->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_71->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_71->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_71->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_71->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_71->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_71->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_71->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_71->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_71->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_71->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_71->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_71->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_71->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_71->A_token_bytes = 2048;
        args_dev_chip00_71->indiv_B_expert_stride = 262144;
        args_dev_chip00_71->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_71->indiv_B_tile_bytes = 131072;
        args_dev_chip00_71->indiv_D_tile_bytes = 16384;
        args_dev_chip00_71->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_71->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_71->indiv_N2 = 2;
        args_dev_chip00_71->indiv_down_N2 = 2;
        args_dev_chip00_71->indiv_K1 = 128;
        args_dev_chip00_71->indiv_N_per_block = 256;
        args_dev_chip00_71->indiv_down_K1 = 64;
        args_dev_chip00_71->indiv_down_N_per_block = 256;
        args_dev_chip00_71->rescale_mult = 1;
        args_dev_chip00_71->rescale_shift = 0;
        args_dev_chip00_71->output_expert_stride_bytes = 65536;
        args_dev_chip00_71->max_tokens_per_expert = 32;
        args_dev_chip00_71->gating_sp_addr = 0;
        args_dev_chip00_71->cond_node_index = 0;
        args_dev_chip00_71->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_71;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[65] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 256;
        device_kernel_list_chip_00[65] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 72 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_72 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_72->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_72->block_idx = 0;
        args_dev_chip00_72->gating_sp_addr = 0;
        args_dev_chip00_72->cond_node_index = 0;
        args_dev_chip00_72->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_72;
        device_arg_list_chip_00[66] = (uint32_t)(uintptr_t)args_dev_chip00_72;
        device_kernel_list_chip_00[66] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 73 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_73 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_73->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_73->block_idx = 0;
        args_dev_chip00_73->gating_sp_addr = 0;
        args_dev_chip00_73->cond_node_index = 0;
        args_dev_chip00_73->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_73;
        device_arg_list_chip_00[67] = (uint32_t)(uintptr_t)args_dev_chip00_73;
        device_kernel_list_chip_00[67] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 74 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_74 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_74->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_74->block_idx = 1;
        args_dev_chip00_74->gating_sp_addr = 0;
        args_dev_chip00_74->cond_node_index = 0;
        args_dev_chip00_74->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_74;
        device_arg_list_chip_00[68] = (uint32_t)(uintptr_t)args_dev_chip00_74;
        device_kernel_list_chip_00[68] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 75 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_75 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_75->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_75->block_idx = 1;
        args_dev_chip00_75->gating_sp_addr = 0;
        args_dev_chip00_75->cond_node_index = 0;
        args_dev_chip00_75->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_75;
        device_arg_list_chip_00[69] = (uint32_t)(uintptr_t)args_dev_chip00_75;
        device_kernel_list_chip_00[69] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 76 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_76 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 256);
        args_dev_chip00_76->ctrl = 0;
        args_dev_chip00_76->expert_id = 0;
        args_dev_chip00_76->token_start_rank = 0;
        args_dev_chip00_76->ntokens = 0;
        args_dev_chip00_76->m_s2_exec = 0;
        args_dev_chip00_76->m_s4_exec = 0;
        args_dev_chip00_76->wait_for_peer_slots = 0;
        args_dev_chip00_76->dma_slot_vd = 0;
        args_dev_chip00_76->s1_block_count = 2;
        args_dev_chip00_76->s3_block_count = 2;
        args_dev_chip00_76->dma_slot_expert_id[0] = -1;
        args_dev_chip00_76->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_76->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_76->dma_slot_expert_id[1] = -1;
        args_dev_chip00_76->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_76->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_76->dma_slot_expert_id[2] = -1;
        args_dev_chip00_76->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_76->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_76->dma_slot_expert_id[3] = -1;
        args_dev_chip00_76->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_76->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_76->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_76->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_76->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_76->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_76->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_76->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_76->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_76->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_76->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_76->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_76->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_76->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_76->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_76->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_76->A_token_bytes = 2048;
        args_dev_chip00_76->indiv_B_expert_stride = 262144;
        args_dev_chip00_76->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_76->indiv_B_tile_bytes = 131072;
        args_dev_chip00_76->indiv_D_tile_bytes = 16384;
        args_dev_chip00_76->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_76->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_76->indiv_N2 = 2;
        args_dev_chip00_76->indiv_down_N2 = 2;
        args_dev_chip00_76->indiv_K1 = 128;
        args_dev_chip00_76->indiv_N_per_block = 256;
        args_dev_chip00_76->indiv_down_K1 = 64;
        args_dev_chip00_76->indiv_down_N_per_block = 256;
        args_dev_chip00_76->rescale_mult = 1;
        args_dev_chip00_76->rescale_shift = 0;
        args_dev_chip00_76->output_expert_stride_bytes = 65536;
        args_dev_chip00_76->max_tokens_per_expert = 32;
        args_dev_chip00_76->gating_sp_addr = 0;
        args_dev_chip00_76->cond_node_index = 0;
        args_dev_chip00_76->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_76;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[70] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 256;
        device_kernel_list_chip_00[70] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 77 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_77 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 256);
        args_dev_chip00_77->ctrl = 0;
        args_dev_chip00_77->expert_id = 0;
        args_dev_chip00_77->token_start_rank = 0;
        args_dev_chip00_77->ntokens = 0;
        args_dev_chip00_77->m_s2_exec = 0;
        args_dev_chip00_77->m_s4_exec = 0;
        args_dev_chip00_77->wait_for_peer_slots = 0;
        args_dev_chip00_77->dma_slot_vd = 0;
        args_dev_chip00_77->s1_block_count = 2;
        args_dev_chip00_77->s3_block_count = 2;
        args_dev_chip00_77->dma_slot_expert_id[0] = -1;
        args_dev_chip00_77->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_77->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_77->dma_slot_expert_id[1] = -1;
        args_dev_chip00_77->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_77->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_77->dma_slot_expert_id[2] = -1;
        args_dev_chip00_77->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_77->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_77->dma_slot_expert_id[3] = -1;
        args_dev_chip00_77->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_77->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_77->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_77->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_77->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_77->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_77->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_77->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_77->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_77->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_77->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_77->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_77->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_77->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_77->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_77->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_77->A_token_bytes = 2048;
        args_dev_chip00_77->indiv_B_expert_stride = 262144;
        args_dev_chip00_77->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_77->indiv_B_tile_bytes = 131072;
        args_dev_chip00_77->indiv_D_tile_bytes = 16384;
        args_dev_chip00_77->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_77->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_77->indiv_N2 = 2;
        args_dev_chip00_77->indiv_down_N2 = 2;
        args_dev_chip00_77->indiv_K1 = 128;
        args_dev_chip00_77->indiv_N_per_block = 256;
        args_dev_chip00_77->indiv_down_K1 = 64;
        args_dev_chip00_77->indiv_down_N_per_block = 256;
        args_dev_chip00_77->rescale_mult = 1;
        args_dev_chip00_77->rescale_shift = 0;
        args_dev_chip00_77->output_expert_stride_bytes = 65536;
        args_dev_chip00_77->max_tokens_per_expert = 32;
        args_dev_chip00_77->gating_sp_addr = 0;
        args_dev_chip00_77->cond_node_index = 0;
        args_dev_chip00_77->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_77;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[71] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 256;
        device_kernel_list_chip_00[71] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 78 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_78 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_78->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_78->block_idx = 0;
        args_dev_chip00_78->gating_sp_addr = 0;
        args_dev_chip00_78->cond_node_index = 0;
        args_dev_chip00_78->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_78;
        device_arg_list_chip_00[72] = (uint32_t)(uintptr_t)args_dev_chip00_78;
        device_kernel_list_chip_00[72] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 79 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_79 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_79->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_79->block_idx = 0;
        args_dev_chip00_79->gating_sp_addr = 0;
        args_dev_chip00_79->cond_node_index = 0;
        args_dev_chip00_79->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_79;
        device_arg_list_chip_00[73] = (uint32_t)(uintptr_t)args_dev_chip00_79;
        device_kernel_list_chip_00[73] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 80 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_80 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_80->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_80->block_idx = 1;
        args_dev_chip00_80->gating_sp_addr = 0;
        args_dev_chip00_80->cond_node_index = 0;
        args_dev_chip00_80->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_80;
        device_arg_list_chip_00[74] = (uint32_t)(uintptr_t)args_dev_chip00_80;
        device_kernel_list_chip_00[74] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 81 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_81 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_81->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 256;
        args_dev_chip00_81->block_idx = 1;
        args_dev_chip00_81->gating_sp_addr = 0;
        args_dev_chip00_81->cond_node_index = 0;
        args_dev_chip00_81->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_81;
        device_arg_list_chip_00[75] = (uint32_t)(uintptr_t)args_dev_chip00_81;
        device_kernel_list_chip_00[75] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 82 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_82 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 256);
        args_dev_chip00_82->ctrl = 0;
        args_dev_chip00_82->expert_id = 0;
        args_dev_chip00_82->token_start_rank = 0;
        args_dev_chip00_82->ntokens = 0;
        args_dev_chip00_82->m_s2_exec = 0;
        args_dev_chip00_82->m_s4_exec = 0;
        args_dev_chip00_82->wait_for_peer_slots = 0;
        args_dev_chip00_82->dma_slot_vd = 0;
        args_dev_chip00_82->s1_block_count = 2;
        args_dev_chip00_82->s3_block_count = 2;
        args_dev_chip00_82->dma_slot_expert_id[0] = -1;
        args_dev_chip00_82->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_82->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_82->dma_slot_expert_id[1] = -1;
        args_dev_chip00_82->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_82->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_82->dma_slot_expert_id[2] = -1;
        args_dev_chip00_82->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_82->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_82->dma_slot_expert_id[3] = -1;
        args_dev_chip00_82->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_82->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_82->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_82->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_82->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_82->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_82->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_82->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_82->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_82->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_82->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_82->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_82->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_82->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_82->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_82->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_82->A_token_bytes = 2048;
        args_dev_chip00_82->indiv_B_expert_stride = 262144;
        args_dev_chip00_82->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_82->indiv_B_tile_bytes = 131072;
        args_dev_chip00_82->indiv_D_tile_bytes = 16384;
        args_dev_chip00_82->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_82->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_82->indiv_N2 = 2;
        args_dev_chip00_82->indiv_down_N2 = 2;
        args_dev_chip00_82->indiv_K1 = 128;
        args_dev_chip00_82->indiv_N_per_block = 256;
        args_dev_chip00_82->indiv_down_K1 = 64;
        args_dev_chip00_82->indiv_down_N_per_block = 256;
        args_dev_chip00_82->rescale_mult = 1;
        args_dev_chip00_82->rescale_shift = 0;
        args_dev_chip00_82->output_expert_stride_bytes = 65536;
        args_dev_chip00_82->max_tokens_per_expert = 32;
        args_dev_chip00_82->gating_sp_addr = 0;
        args_dev_chip00_82->cond_node_index = 0;
        args_dev_chip00_82->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_82;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[76] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 256;
        device_kernel_list_chip_00[76] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 83 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_83 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 256);
        args_dev_chip00_83->ctrl = 0;
        args_dev_chip00_83->expert_id = 0;
        args_dev_chip00_83->token_start_rank = 0;
        args_dev_chip00_83->ntokens = 0;
        args_dev_chip00_83->m_s2_exec = 0;
        args_dev_chip00_83->m_s4_exec = 0;
        args_dev_chip00_83->wait_for_peer_slots = 0;
        args_dev_chip00_83->dma_slot_vd = 0;
        args_dev_chip00_83->s1_block_count = 2;
        args_dev_chip00_83->s3_block_count = 2;
        args_dev_chip00_83->dma_slot_expert_id[0] = -1;
        args_dev_chip00_83->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_83->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_83->dma_slot_expert_id[1] = -1;
        args_dev_chip00_83->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_83->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_83->dma_slot_expert_id[2] = -1;
        args_dev_chip00_83->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_83->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_83->dma_slot_expert_id[3] = -1;
        args_dev_chip00_83->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_83->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_83->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_83->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_83->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_83->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_83->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_83->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_83->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_83->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_83->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_83->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_83->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_83->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_83->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_83->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_83->A_token_bytes = 2048;
        args_dev_chip00_83->indiv_B_expert_stride = 262144;
        args_dev_chip00_83->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_83->indiv_B_tile_bytes = 131072;
        args_dev_chip00_83->indiv_D_tile_bytes = 16384;
        args_dev_chip00_83->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_83->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_83->indiv_N2 = 2;
        args_dev_chip00_83->indiv_down_N2 = 2;
        args_dev_chip00_83->indiv_K1 = 128;
        args_dev_chip00_83->indiv_N_per_block = 256;
        args_dev_chip00_83->indiv_down_K1 = 64;
        args_dev_chip00_83->indiv_down_N_per_block = 256;
        args_dev_chip00_83->rescale_mult = 1;
        args_dev_chip00_83->rescale_shift = 0;
        args_dev_chip00_83->output_expert_stride_bytes = 65536;
        args_dev_chip00_83->max_tokens_per_expert = 32;
        args_dev_chip00_83->gating_sp_addr = 0;
        args_dev_chip00_83->cond_node_index = 0;
        args_dev_chip00_83->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_83;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[77] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 256;
        device_kernel_list_chip_00[77] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 84 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_84 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 256);
        args_dev_chip00_84->ctrl = 0;
        args_dev_chip00_84->expert_id = 0;
        args_dev_chip00_84->token_start_rank = 0;
        args_dev_chip00_84->ntokens = 0;
        args_dev_chip00_84->m_s2_exec = 0;
        args_dev_chip00_84->m_s4_exec = 0;
        args_dev_chip00_84->wait_for_peer_slots = 0;
        args_dev_chip00_84->dma_slot_vd = 0;
        args_dev_chip00_84->s1_block_count = 2;
        args_dev_chip00_84->s3_block_count = 2;
        args_dev_chip00_84->dma_slot_expert_id[0] = -1;
        args_dev_chip00_84->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_84->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_84->dma_slot_expert_id[1] = -1;
        args_dev_chip00_84->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_84->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_84->dma_slot_expert_id[2] = -1;
        args_dev_chip00_84->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_84->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_84->dma_slot_expert_id[3] = -1;
        args_dev_chip00_84->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_84->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_84->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_84->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_84->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_84->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_84->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_84->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_84->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_84->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_84->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_84->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_84->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_84->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_84->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_84->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_84->A_token_bytes = 2048;
        args_dev_chip00_84->indiv_B_expert_stride = 262144;
        args_dev_chip00_84->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_84->indiv_B_tile_bytes = 131072;
        args_dev_chip00_84->indiv_D_tile_bytes = 16384;
        args_dev_chip00_84->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_84->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_84->indiv_N2 = 2;
        args_dev_chip00_84->indiv_down_N2 = 2;
        args_dev_chip00_84->indiv_K1 = 128;
        args_dev_chip00_84->indiv_N_per_block = 256;
        args_dev_chip00_84->indiv_down_K1 = 64;
        args_dev_chip00_84->indiv_down_N_per_block = 256;
        args_dev_chip00_84->rescale_mult = 1;
        args_dev_chip00_84->rescale_shift = 0;
        args_dev_chip00_84->output_expert_stride_bytes = 65536;
        args_dev_chip00_84->max_tokens_per_expert = 32;
        args_dev_chip00_84->gating_sp_addr = 0;
        args_dev_chip00_84->cond_node_index = 0;
        args_dev_chip00_84->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_84;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 256), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 256)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[78] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 256;
        device_kernel_list_chip_00[78] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 85 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_85 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 512);
        args_dev_chip00_85->ctrl = 0;
        args_dev_chip00_85->expert_id = 0;
        args_dev_chip00_85->token_start_rank = 0;
        args_dev_chip00_85->ntokens = 0;
        args_dev_chip00_85->m_s2_exec = 0;
        args_dev_chip00_85->m_s4_exec = 0;
        args_dev_chip00_85->wait_for_peer_slots = 0;
        args_dev_chip00_85->dma_slot_vd = 0;
        args_dev_chip00_85->s1_block_count = 2;
        args_dev_chip00_85->s3_block_count = 2;
        args_dev_chip00_85->dma_slot_expert_id[0] = -1;
        args_dev_chip00_85->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_85->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_85->dma_slot_expert_id[1] = -1;
        args_dev_chip00_85->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_85->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_85->dma_slot_expert_id[2] = -1;
        args_dev_chip00_85->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_85->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_85->dma_slot_expert_id[3] = -1;
        args_dev_chip00_85->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_85->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_85->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_85->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_85->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_85->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_85->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_85->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_85->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_85->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_85->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_85->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_85->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_85->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_85->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_85->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_85->A_token_bytes = 2048;
        args_dev_chip00_85->indiv_B_expert_stride = 262144;
        args_dev_chip00_85->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_85->indiv_B_tile_bytes = 131072;
        args_dev_chip00_85->indiv_D_tile_bytes = 16384;
        args_dev_chip00_85->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_85->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_85->indiv_N2 = 2;
        args_dev_chip00_85->indiv_down_N2 = 2;
        args_dev_chip00_85->indiv_K1 = 128;
        args_dev_chip00_85->indiv_N_per_block = 256;
        args_dev_chip00_85->indiv_down_K1 = 64;
        args_dev_chip00_85->indiv_down_N_per_block = 256;
        args_dev_chip00_85->rescale_mult = 1;
        args_dev_chip00_85->rescale_shift = 0;
        args_dev_chip00_85->output_expert_stride_bytes = 65536;
        args_dev_chip00_85->max_tokens_per_expert = 32;
        args_dev_chip00_85->gating_sp_addr = 0;
        args_dev_chip00_85->cond_node_index = 0;
        args_dev_chip00_85->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_85;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[79] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 512;
        device_kernel_list_chip_00[79] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 86 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_86 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_86->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_86->block_idx = 0;
        args_dev_chip00_86->gating_sp_addr = 0;
        args_dev_chip00_86->cond_node_index = 0;
        args_dev_chip00_86->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_86;
        device_arg_list_chip_00[80] = (uint32_t)(uintptr_t)args_dev_chip00_86;
        device_kernel_list_chip_00[80] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 87 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_87 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_87->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_87->block_idx = 0;
        args_dev_chip00_87->gating_sp_addr = 0;
        args_dev_chip00_87->cond_node_index = 0;
        args_dev_chip00_87->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_87;
        device_arg_list_chip_00[81] = (uint32_t)(uintptr_t)args_dev_chip00_87;
        device_kernel_list_chip_00[81] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 88 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_88 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_88->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_88->block_idx = 1;
        args_dev_chip00_88->gating_sp_addr = 0;
        args_dev_chip00_88->cond_node_index = 0;
        args_dev_chip00_88->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_88;
        device_arg_list_chip_00[82] = (uint32_t)(uintptr_t)args_dev_chip00_88;
        device_kernel_list_chip_00[82] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 89 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_89 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_89->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_89->block_idx = 1;
        args_dev_chip00_89->gating_sp_addr = 0;
        args_dev_chip00_89->cond_node_index = 0;
        args_dev_chip00_89->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_89;
        device_arg_list_chip_00[83] = (uint32_t)(uintptr_t)args_dev_chip00_89;
        device_kernel_list_chip_00[83] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 90 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_90 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 512);
        args_dev_chip00_90->ctrl = 0;
        args_dev_chip00_90->expert_id = 0;
        args_dev_chip00_90->token_start_rank = 0;
        args_dev_chip00_90->ntokens = 0;
        args_dev_chip00_90->m_s2_exec = 0;
        args_dev_chip00_90->m_s4_exec = 0;
        args_dev_chip00_90->wait_for_peer_slots = 0;
        args_dev_chip00_90->dma_slot_vd = 0;
        args_dev_chip00_90->s1_block_count = 2;
        args_dev_chip00_90->s3_block_count = 2;
        args_dev_chip00_90->dma_slot_expert_id[0] = -1;
        args_dev_chip00_90->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_90->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_90->dma_slot_expert_id[1] = -1;
        args_dev_chip00_90->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_90->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_90->dma_slot_expert_id[2] = -1;
        args_dev_chip00_90->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_90->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_90->dma_slot_expert_id[3] = -1;
        args_dev_chip00_90->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_90->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_90->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_90->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_90->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_90->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_90->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_90->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_90->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_90->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_90->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_90->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_90->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_90->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_90->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_90->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_90->A_token_bytes = 2048;
        args_dev_chip00_90->indiv_B_expert_stride = 262144;
        args_dev_chip00_90->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_90->indiv_B_tile_bytes = 131072;
        args_dev_chip00_90->indiv_D_tile_bytes = 16384;
        args_dev_chip00_90->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_90->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_90->indiv_N2 = 2;
        args_dev_chip00_90->indiv_down_N2 = 2;
        args_dev_chip00_90->indiv_K1 = 128;
        args_dev_chip00_90->indiv_N_per_block = 256;
        args_dev_chip00_90->indiv_down_K1 = 64;
        args_dev_chip00_90->indiv_down_N_per_block = 256;
        args_dev_chip00_90->rescale_mult = 1;
        args_dev_chip00_90->rescale_shift = 0;
        args_dev_chip00_90->output_expert_stride_bytes = 65536;
        args_dev_chip00_90->max_tokens_per_expert = 32;
        args_dev_chip00_90->gating_sp_addr = 0;
        args_dev_chip00_90->cond_node_index = 0;
        args_dev_chip00_90->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_90;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[84] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 512;
        device_kernel_list_chip_00[84] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 91 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_91 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 512);
        args_dev_chip00_91->ctrl = 0;
        args_dev_chip00_91->expert_id = 0;
        args_dev_chip00_91->token_start_rank = 0;
        args_dev_chip00_91->ntokens = 0;
        args_dev_chip00_91->m_s2_exec = 0;
        args_dev_chip00_91->m_s4_exec = 0;
        args_dev_chip00_91->wait_for_peer_slots = 0;
        args_dev_chip00_91->dma_slot_vd = 0;
        args_dev_chip00_91->s1_block_count = 2;
        args_dev_chip00_91->s3_block_count = 2;
        args_dev_chip00_91->dma_slot_expert_id[0] = -1;
        args_dev_chip00_91->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_91->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_91->dma_slot_expert_id[1] = -1;
        args_dev_chip00_91->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_91->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_91->dma_slot_expert_id[2] = -1;
        args_dev_chip00_91->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_91->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_91->dma_slot_expert_id[3] = -1;
        args_dev_chip00_91->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_91->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_91->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_91->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_91->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_91->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_91->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_91->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_91->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_91->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_91->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_91->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_91->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_91->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_91->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_91->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_91->A_token_bytes = 2048;
        args_dev_chip00_91->indiv_B_expert_stride = 262144;
        args_dev_chip00_91->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_91->indiv_B_tile_bytes = 131072;
        args_dev_chip00_91->indiv_D_tile_bytes = 16384;
        args_dev_chip00_91->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_91->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_91->indiv_N2 = 2;
        args_dev_chip00_91->indiv_down_N2 = 2;
        args_dev_chip00_91->indiv_K1 = 128;
        args_dev_chip00_91->indiv_N_per_block = 256;
        args_dev_chip00_91->indiv_down_K1 = 64;
        args_dev_chip00_91->indiv_down_N_per_block = 256;
        args_dev_chip00_91->rescale_mult = 1;
        args_dev_chip00_91->rescale_shift = 0;
        args_dev_chip00_91->output_expert_stride_bytes = 65536;
        args_dev_chip00_91->max_tokens_per_expert = 32;
        args_dev_chip00_91->gating_sp_addr = 0;
        args_dev_chip00_91->cond_node_index = 0;
        args_dev_chip00_91->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_91;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[85] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 512;
        device_kernel_list_chip_00[85] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 92 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_92 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_92->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_92->block_idx = 0;
        args_dev_chip00_92->gating_sp_addr = 0;
        args_dev_chip00_92->cond_node_index = 0;
        args_dev_chip00_92->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_92;
        device_arg_list_chip_00[86] = (uint32_t)(uintptr_t)args_dev_chip00_92;
        device_kernel_list_chip_00[86] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 93 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_93 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_93->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_93->block_idx = 0;
        args_dev_chip00_93->gating_sp_addr = 0;
        args_dev_chip00_93->cond_node_index = 0;
        args_dev_chip00_93->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_93;
        device_arg_list_chip_00[87] = (uint32_t)(uintptr_t)args_dev_chip00_93;
        device_kernel_list_chip_00[87] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 94 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_94 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_94->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_94->block_idx = 1;
        args_dev_chip00_94->gating_sp_addr = 0;
        args_dev_chip00_94->cond_node_index = 0;
        args_dev_chip00_94->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_94;
        device_arg_list_chip_00[88] = (uint32_t)(uintptr_t)args_dev_chip00_94;
        device_kernel_list_chip_00[88] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 95 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_95 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_95->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 512;
        args_dev_chip00_95->block_idx = 1;
        args_dev_chip00_95->gating_sp_addr = 0;
        args_dev_chip00_95->cond_node_index = 0;
        args_dev_chip00_95->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_95;
        device_arg_list_chip_00[89] = (uint32_t)(uintptr_t)args_dev_chip00_95;
        device_kernel_list_chip_00[89] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 96 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_96 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 512);
        args_dev_chip00_96->ctrl = 0;
        args_dev_chip00_96->expert_id = 0;
        args_dev_chip00_96->token_start_rank = 0;
        args_dev_chip00_96->ntokens = 0;
        args_dev_chip00_96->m_s2_exec = 0;
        args_dev_chip00_96->m_s4_exec = 0;
        args_dev_chip00_96->wait_for_peer_slots = 0;
        args_dev_chip00_96->dma_slot_vd = 0;
        args_dev_chip00_96->s1_block_count = 2;
        args_dev_chip00_96->s3_block_count = 2;
        args_dev_chip00_96->dma_slot_expert_id[0] = -1;
        args_dev_chip00_96->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_96->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_96->dma_slot_expert_id[1] = -1;
        args_dev_chip00_96->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_96->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_96->dma_slot_expert_id[2] = -1;
        args_dev_chip00_96->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_96->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_96->dma_slot_expert_id[3] = -1;
        args_dev_chip00_96->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_96->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_96->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_96->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_96->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_96->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_96->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_96->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_96->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_96->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_96->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_96->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_96->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_96->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_96->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_96->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_96->A_token_bytes = 2048;
        args_dev_chip00_96->indiv_B_expert_stride = 262144;
        args_dev_chip00_96->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_96->indiv_B_tile_bytes = 131072;
        args_dev_chip00_96->indiv_D_tile_bytes = 16384;
        args_dev_chip00_96->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_96->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_96->indiv_N2 = 2;
        args_dev_chip00_96->indiv_down_N2 = 2;
        args_dev_chip00_96->indiv_K1 = 128;
        args_dev_chip00_96->indiv_N_per_block = 256;
        args_dev_chip00_96->indiv_down_K1 = 64;
        args_dev_chip00_96->indiv_down_N_per_block = 256;
        args_dev_chip00_96->rescale_mult = 1;
        args_dev_chip00_96->rescale_shift = 0;
        args_dev_chip00_96->output_expert_stride_bytes = 65536;
        args_dev_chip00_96->max_tokens_per_expert = 32;
        args_dev_chip00_96->gating_sp_addr = 0;
        args_dev_chip00_96->cond_node_index = 0;
        args_dev_chip00_96->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_96;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[90] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 512;
        device_kernel_list_chip_00[90] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 97 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_97 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 512);
        args_dev_chip00_97->ctrl = 0;
        args_dev_chip00_97->expert_id = 0;
        args_dev_chip00_97->token_start_rank = 0;
        args_dev_chip00_97->ntokens = 0;
        args_dev_chip00_97->m_s2_exec = 0;
        args_dev_chip00_97->m_s4_exec = 0;
        args_dev_chip00_97->wait_for_peer_slots = 0;
        args_dev_chip00_97->dma_slot_vd = 0;
        args_dev_chip00_97->s1_block_count = 2;
        args_dev_chip00_97->s3_block_count = 2;
        args_dev_chip00_97->dma_slot_expert_id[0] = -1;
        args_dev_chip00_97->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_97->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_97->dma_slot_expert_id[1] = -1;
        args_dev_chip00_97->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_97->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_97->dma_slot_expert_id[2] = -1;
        args_dev_chip00_97->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_97->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_97->dma_slot_expert_id[3] = -1;
        args_dev_chip00_97->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_97->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_97->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_97->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_97->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_97->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_97->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_97->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_97->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_97->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_97->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_97->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_97->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_97->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_97->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_97->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_97->A_token_bytes = 2048;
        args_dev_chip00_97->indiv_B_expert_stride = 262144;
        args_dev_chip00_97->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_97->indiv_B_tile_bytes = 131072;
        args_dev_chip00_97->indiv_D_tile_bytes = 16384;
        args_dev_chip00_97->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_97->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_97->indiv_N2 = 2;
        args_dev_chip00_97->indiv_down_N2 = 2;
        args_dev_chip00_97->indiv_K1 = 128;
        args_dev_chip00_97->indiv_N_per_block = 256;
        args_dev_chip00_97->indiv_down_K1 = 64;
        args_dev_chip00_97->indiv_down_N_per_block = 256;
        args_dev_chip00_97->rescale_mult = 1;
        args_dev_chip00_97->rescale_shift = 0;
        args_dev_chip00_97->output_expert_stride_bytes = 65536;
        args_dev_chip00_97->max_tokens_per_expert = 32;
        args_dev_chip00_97->gating_sp_addr = 0;
        args_dev_chip00_97->cond_node_index = 0;
        args_dev_chip00_97->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_97;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[91] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 512;
        device_kernel_list_chip_00[91] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 98 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_98 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 512);
        args_dev_chip00_98->ctrl = 0;
        args_dev_chip00_98->expert_id = 0;
        args_dev_chip00_98->token_start_rank = 0;
        args_dev_chip00_98->ntokens = 0;
        args_dev_chip00_98->m_s2_exec = 0;
        args_dev_chip00_98->m_s4_exec = 0;
        args_dev_chip00_98->wait_for_peer_slots = 0;
        args_dev_chip00_98->dma_slot_vd = 0;
        args_dev_chip00_98->s1_block_count = 2;
        args_dev_chip00_98->s3_block_count = 2;
        args_dev_chip00_98->dma_slot_expert_id[0] = -1;
        args_dev_chip00_98->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_98->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_98->dma_slot_expert_id[1] = -1;
        args_dev_chip00_98->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_98->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_98->dma_slot_expert_id[2] = -1;
        args_dev_chip00_98->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_98->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_98->dma_slot_expert_id[3] = -1;
        args_dev_chip00_98->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_98->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_98->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_98->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_98->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_98->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_98->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_98->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_98->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_98->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_98->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_98->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_98->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_98->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_98->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_98->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_98->A_token_bytes = 2048;
        args_dev_chip00_98->indiv_B_expert_stride = 262144;
        args_dev_chip00_98->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_98->indiv_B_tile_bytes = 131072;
        args_dev_chip00_98->indiv_D_tile_bytes = 16384;
        args_dev_chip00_98->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_98->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_98->indiv_N2 = 2;
        args_dev_chip00_98->indiv_down_N2 = 2;
        args_dev_chip00_98->indiv_K1 = 128;
        args_dev_chip00_98->indiv_N_per_block = 256;
        args_dev_chip00_98->indiv_down_K1 = 64;
        args_dev_chip00_98->indiv_down_N_per_block = 256;
        args_dev_chip00_98->rescale_mult = 1;
        args_dev_chip00_98->rescale_shift = 0;
        args_dev_chip00_98->output_expert_stride_bytes = 65536;
        args_dev_chip00_98->max_tokens_per_expert = 32;
        args_dev_chip00_98->gating_sp_addr = 0;
        args_dev_chip00_98->cond_node_index = 0;
        args_dev_chip00_98->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_98;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[92] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 512;
        device_kernel_list_chip_00[92] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 99 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_99 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 512);
        args_dev_chip00_99->ctrl = 0;
        args_dev_chip00_99->expert_id = 0;
        args_dev_chip00_99->token_start_rank = 0;
        args_dev_chip00_99->ntokens = 0;
        args_dev_chip00_99->m_s2_exec = 0;
        args_dev_chip00_99->m_s4_exec = 0;
        args_dev_chip00_99->wait_for_peer_slots = 0;
        args_dev_chip00_99->dma_slot_vd = 0;
        args_dev_chip00_99->s1_block_count = 2;
        args_dev_chip00_99->s3_block_count = 2;
        args_dev_chip00_99->dma_slot_expert_id[0] = -1;
        args_dev_chip00_99->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_99->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_99->dma_slot_expert_id[1] = -1;
        args_dev_chip00_99->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_99->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_99->dma_slot_expert_id[2] = -1;
        args_dev_chip00_99->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_99->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_99->dma_slot_expert_id[3] = -1;
        args_dev_chip00_99->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_99->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_99->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_99->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_99->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_99->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_99->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_99->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_99->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_99->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_99->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_99->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_99->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_99->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_99->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_99->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_99->A_token_bytes = 2048;
        args_dev_chip00_99->indiv_B_expert_stride = 262144;
        args_dev_chip00_99->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_99->indiv_B_tile_bytes = 131072;
        args_dev_chip00_99->indiv_D_tile_bytes = 16384;
        args_dev_chip00_99->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_99->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_99->indiv_N2 = 2;
        args_dev_chip00_99->indiv_down_N2 = 2;
        args_dev_chip00_99->indiv_K1 = 128;
        args_dev_chip00_99->indiv_N_per_block = 256;
        args_dev_chip00_99->indiv_down_K1 = 64;
        args_dev_chip00_99->indiv_down_N_per_block = 256;
        args_dev_chip00_99->rescale_mult = 1;
        args_dev_chip00_99->rescale_shift = 0;
        args_dev_chip00_99->output_expert_stride_bytes = 65536;
        args_dev_chip00_99->max_tokens_per_expert = 32;
        args_dev_chip00_99->gating_sp_addr = 0;
        args_dev_chip00_99->cond_node_index = 0;
        args_dev_chip00_99->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_99;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[93] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 512;
        device_kernel_list_chip_00[93] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 100 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_100 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_100->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_100->block_idx = 0;
        args_dev_chip00_100->gating_sp_addr = 0;
        args_dev_chip00_100->cond_node_index = 0;
        args_dev_chip00_100->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_100;
        device_arg_list_chip_00[94] = (uint32_t)(uintptr_t)args_dev_chip00_100;
        device_kernel_list_chip_00[94] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 101 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_101 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_101->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_101->block_idx = 0;
        args_dev_chip00_101->gating_sp_addr = 0;
        args_dev_chip00_101->cond_node_index = 0;
        args_dev_chip00_101->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_101;
        device_arg_list_chip_00[95] = (uint32_t)(uintptr_t)args_dev_chip00_101;
        device_kernel_list_chip_00[95] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 102 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_102 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_102->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_102->block_idx = 1;
        args_dev_chip00_102->gating_sp_addr = 0;
        args_dev_chip00_102->cond_node_index = 0;
        args_dev_chip00_102->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_102;
        device_arg_list_chip_00[96] = (uint32_t)(uintptr_t)args_dev_chip00_102;
        device_kernel_list_chip_00[96] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 103 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_103 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_103->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_103->block_idx = 1;
        args_dev_chip00_103->gating_sp_addr = 0;
        args_dev_chip00_103->cond_node_index = 0;
        args_dev_chip00_103->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_103;
        device_arg_list_chip_00[97] = (uint32_t)(uintptr_t)args_dev_chip00_103;
        device_kernel_list_chip_00[97] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 104 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_104 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 512);
        args_dev_chip00_104->ctrl = 0;
        args_dev_chip00_104->expert_id = 0;
        args_dev_chip00_104->token_start_rank = 0;
        args_dev_chip00_104->ntokens = 0;
        args_dev_chip00_104->m_s2_exec = 0;
        args_dev_chip00_104->m_s4_exec = 0;
        args_dev_chip00_104->wait_for_peer_slots = 0;
        args_dev_chip00_104->dma_slot_vd = 0;
        args_dev_chip00_104->s1_block_count = 2;
        args_dev_chip00_104->s3_block_count = 2;
        args_dev_chip00_104->dma_slot_expert_id[0] = -1;
        args_dev_chip00_104->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_104->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_104->dma_slot_expert_id[1] = -1;
        args_dev_chip00_104->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_104->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_104->dma_slot_expert_id[2] = -1;
        args_dev_chip00_104->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_104->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_104->dma_slot_expert_id[3] = -1;
        args_dev_chip00_104->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_104->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_104->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_104->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_104->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_104->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_104->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_104->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_104->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_104->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_104->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_104->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_104->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_104->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_104->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_104->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_104->A_token_bytes = 2048;
        args_dev_chip00_104->indiv_B_expert_stride = 262144;
        args_dev_chip00_104->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_104->indiv_B_tile_bytes = 131072;
        args_dev_chip00_104->indiv_D_tile_bytes = 16384;
        args_dev_chip00_104->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_104->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_104->indiv_N2 = 2;
        args_dev_chip00_104->indiv_down_N2 = 2;
        args_dev_chip00_104->indiv_K1 = 128;
        args_dev_chip00_104->indiv_N_per_block = 256;
        args_dev_chip00_104->indiv_down_K1 = 64;
        args_dev_chip00_104->indiv_down_N_per_block = 256;
        args_dev_chip00_104->rescale_mult = 1;
        args_dev_chip00_104->rescale_shift = 0;
        args_dev_chip00_104->output_expert_stride_bytes = 65536;
        args_dev_chip00_104->max_tokens_per_expert = 32;
        args_dev_chip00_104->gating_sp_addr = 0;
        args_dev_chip00_104->cond_node_index = 0;
        args_dev_chip00_104->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_104;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[98] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 512;
        device_kernel_list_chip_00[98] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 105 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_105 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 512);
        args_dev_chip00_105->ctrl = 0;
        args_dev_chip00_105->expert_id = 0;
        args_dev_chip00_105->token_start_rank = 0;
        args_dev_chip00_105->ntokens = 0;
        args_dev_chip00_105->m_s2_exec = 0;
        args_dev_chip00_105->m_s4_exec = 0;
        args_dev_chip00_105->wait_for_peer_slots = 0;
        args_dev_chip00_105->dma_slot_vd = 0;
        args_dev_chip00_105->s1_block_count = 2;
        args_dev_chip00_105->s3_block_count = 2;
        args_dev_chip00_105->dma_slot_expert_id[0] = -1;
        args_dev_chip00_105->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_105->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_105->dma_slot_expert_id[1] = -1;
        args_dev_chip00_105->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_105->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_105->dma_slot_expert_id[2] = -1;
        args_dev_chip00_105->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_105->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_105->dma_slot_expert_id[3] = -1;
        args_dev_chip00_105->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_105->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_105->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_105->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_105->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_105->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_105->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_105->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_105->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_105->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_105->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_105->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_105->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_105->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_105->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_105->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_105->A_token_bytes = 2048;
        args_dev_chip00_105->indiv_B_expert_stride = 262144;
        args_dev_chip00_105->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_105->indiv_B_tile_bytes = 131072;
        args_dev_chip00_105->indiv_D_tile_bytes = 16384;
        args_dev_chip00_105->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_105->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_105->indiv_N2 = 2;
        args_dev_chip00_105->indiv_down_N2 = 2;
        args_dev_chip00_105->indiv_K1 = 128;
        args_dev_chip00_105->indiv_N_per_block = 256;
        args_dev_chip00_105->indiv_down_K1 = 64;
        args_dev_chip00_105->indiv_down_N_per_block = 256;
        args_dev_chip00_105->rescale_mult = 1;
        args_dev_chip00_105->rescale_shift = 0;
        args_dev_chip00_105->output_expert_stride_bytes = 65536;
        args_dev_chip00_105->max_tokens_per_expert = 32;
        args_dev_chip00_105->gating_sp_addr = 0;
        args_dev_chip00_105->cond_node_index = 0;
        args_dev_chip00_105->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_105;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[99] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 512;
        device_kernel_list_chip_00[99] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 106 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_106 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_106->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_106->block_idx = 0;
        args_dev_chip00_106->gating_sp_addr = 0;
        args_dev_chip00_106->cond_node_index = 0;
        args_dev_chip00_106->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_106;
        device_arg_list_chip_00[100] = (uint32_t)(uintptr_t)args_dev_chip00_106;
        device_kernel_list_chip_00[100] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 107 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_107 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_107->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_107->block_idx = 0;
        args_dev_chip00_107->gating_sp_addr = 0;
        args_dev_chip00_107->cond_node_index = 0;
        args_dev_chip00_107->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_107;
        device_arg_list_chip_00[101] = (uint32_t)(uintptr_t)args_dev_chip00_107;
        device_kernel_list_chip_00[101] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 108 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_108 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_108->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_108->block_idx = 1;
        args_dev_chip00_108->gating_sp_addr = 0;
        args_dev_chip00_108->cond_node_index = 0;
        args_dev_chip00_108->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_108;
        device_arg_list_chip_00[102] = (uint32_t)(uintptr_t)args_dev_chip00_108;
        device_kernel_list_chip_00[102] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 109 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_109 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_109->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 512;
        args_dev_chip00_109->block_idx = 1;
        args_dev_chip00_109->gating_sp_addr = 0;
        args_dev_chip00_109->cond_node_index = 0;
        args_dev_chip00_109->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_109;
        device_arg_list_chip_00[103] = (uint32_t)(uintptr_t)args_dev_chip00_109;
        device_kernel_list_chip_00[103] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 110 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_110 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 512);
        args_dev_chip00_110->ctrl = 0;
        args_dev_chip00_110->expert_id = 0;
        args_dev_chip00_110->token_start_rank = 0;
        args_dev_chip00_110->ntokens = 0;
        args_dev_chip00_110->m_s2_exec = 0;
        args_dev_chip00_110->m_s4_exec = 0;
        args_dev_chip00_110->wait_for_peer_slots = 0;
        args_dev_chip00_110->dma_slot_vd = 0;
        args_dev_chip00_110->s1_block_count = 2;
        args_dev_chip00_110->s3_block_count = 2;
        args_dev_chip00_110->dma_slot_expert_id[0] = -1;
        args_dev_chip00_110->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_110->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_110->dma_slot_expert_id[1] = -1;
        args_dev_chip00_110->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_110->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_110->dma_slot_expert_id[2] = -1;
        args_dev_chip00_110->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_110->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_110->dma_slot_expert_id[3] = -1;
        args_dev_chip00_110->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_110->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_110->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_110->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_110->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_110->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_110->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_110->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_110->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_110->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_110->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_110->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_110->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_110->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_110->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_110->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_110->A_token_bytes = 2048;
        args_dev_chip00_110->indiv_B_expert_stride = 262144;
        args_dev_chip00_110->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_110->indiv_B_tile_bytes = 131072;
        args_dev_chip00_110->indiv_D_tile_bytes = 16384;
        args_dev_chip00_110->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_110->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_110->indiv_N2 = 2;
        args_dev_chip00_110->indiv_down_N2 = 2;
        args_dev_chip00_110->indiv_K1 = 128;
        args_dev_chip00_110->indiv_N_per_block = 256;
        args_dev_chip00_110->indiv_down_K1 = 64;
        args_dev_chip00_110->indiv_down_N_per_block = 256;
        args_dev_chip00_110->rescale_mult = 1;
        args_dev_chip00_110->rescale_shift = 0;
        args_dev_chip00_110->output_expert_stride_bytes = 65536;
        args_dev_chip00_110->max_tokens_per_expert = 32;
        args_dev_chip00_110->gating_sp_addr = 0;
        args_dev_chip00_110->cond_node_index = 0;
        args_dev_chip00_110->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_110;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[104] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 512;
        device_kernel_list_chip_00[104] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 111 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_111 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 512);
        args_dev_chip00_111->ctrl = 0;
        args_dev_chip00_111->expert_id = 0;
        args_dev_chip00_111->token_start_rank = 0;
        args_dev_chip00_111->ntokens = 0;
        args_dev_chip00_111->m_s2_exec = 0;
        args_dev_chip00_111->m_s4_exec = 0;
        args_dev_chip00_111->wait_for_peer_slots = 0;
        args_dev_chip00_111->dma_slot_vd = 0;
        args_dev_chip00_111->s1_block_count = 2;
        args_dev_chip00_111->s3_block_count = 2;
        args_dev_chip00_111->dma_slot_expert_id[0] = -1;
        args_dev_chip00_111->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_111->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_111->dma_slot_expert_id[1] = -1;
        args_dev_chip00_111->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_111->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_111->dma_slot_expert_id[2] = -1;
        args_dev_chip00_111->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_111->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_111->dma_slot_expert_id[3] = -1;
        args_dev_chip00_111->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_111->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_111->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_111->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_111->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_111->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_111->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_111->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_111->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_111->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_111->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_111->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_111->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_111->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_111->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_111->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_111->A_token_bytes = 2048;
        args_dev_chip00_111->indiv_B_expert_stride = 262144;
        args_dev_chip00_111->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_111->indiv_B_tile_bytes = 131072;
        args_dev_chip00_111->indiv_D_tile_bytes = 16384;
        args_dev_chip00_111->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_111->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_111->indiv_N2 = 2;
        args_dev_chip00_111->indiv_down_N2 = 2;
        args_dev_chip00_111->indiv_K1 = 128;
        args_dev_chip00_111->indiv_N_per_block = 256;
        args_dev_chip00_111->indiv_down_K1 = 64;
        args_dev_chip00_111->indiv_down_N_per_block = 256;
        args_dev_chip00_111->rescale_mult = 1;
        args_dev_chip00_111->rescale_shift = 0;
        args_dev_chip00_111->output_expert_stride_bytes = 65536;
        args_dev_chip00_111->max_tokens_per_expert = 32;
        args_dev_chip00_111->gating_sp_addr = 0;
        args_dev_chip00_111->cond_node_index = 0;
        args_dev_chip00_111->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_111;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[105] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 512;
        device_kernel_list_chip_00[105] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 112 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_112 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 512);
        args_dev_chip00_112->ctrl = 0;
        args_dev_chip00_112->expert_id = 0;
        args_dev_chip00_112->token_start_rank = 0;
        args_dev_chip00_112->ntokens = 0;
        args_dev_chip00_112->m_s2_exec = 0;
        args_dev_chip00_112->m_s4_exec = 0;
        args_dev_chip00_112->wait_for_peer_slots = 0;
        args_dev_chip00_112->dma_slot_vd = 0;
        args_dev_chip00_112->s1_block_count = 2;
        args_dev_chip00_112->s3_block_count = 2;
        args_dev_chip00_112->dma_slot_expert_id[0] = -1;
        args_dev_chip00_112->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_112->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_112->dma_slot_expert_id[1] = -1;
        args_dev_chip00_112->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_112->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_112->dma_slot_expert_id[2] = -1;
        args_dev_chip00_112->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_112->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_112->dma_slot_expert_id[3] = -1;
        args_dev_chip00_112->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_112->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_112->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_112->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_112->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_112->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_112->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_112->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_112->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_112->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_112->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_112->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_112->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_112->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_112->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_112->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_112->A_token_bytes = 2048;
        args_dev_chip00_112->indiv_B_expert_stride = 262144;
        args_dev_chip00_112->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_112->indiv_B_tile_bytes = 131072;
        args_dev_chip00_112->indiv_D_tile_bytes = 16384;
        args_dev_chip00_112->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_112->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_112->indiv_N2 = 2;
        args_dev_chip00_112->indiv_down_N2 = 2;
        args_dev_chip00_112->indiv_K1 = 128;
        args_dev_chip00_112->indiv_N_per_block = 256;
        args_dev_chip00_112->indiv_down_K1 = 64;
        args_dev_chip00_112->indiv_down_N_per_block = 256;
        args_dev_chip00_112->rescale_mult = 1;
        args_dev_chip00_112->rescale_shift = 0;
        args_dev_chip00_112->output_expert_stride_bytes = 65536;
        args_dev_chip00_112->max_tokens_per_expert = 32;
        args_dev_chip00_112->gating_sp_addr = 0;
        args_dev_chip00_112->cond_node_index = 0;
        args_dev_chip00_112->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_112;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 512), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 512)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[106] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 512;
        device_kernel_list_chip_00[106] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 113 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_113 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 768);
        args_dev_chip00_113->ctrl = 0;
        args_dev_chip00_113->expert_id = 0;
        args_dev_chip00_113->token_start_rank = 0;
        args_dev_chip00_113->ntokens = 0;
        args_dev_chip00_113->m_s2_exec = 0;
        args_dev_chip00_113->m_s4_exec = 0;
        args_dev_chip00_113->wait_for_peer_slots = 0;
        args_dev_chip00_113->dma_slot_vd = 0;
        args_dev_chip00_113->s1_block_count = 2;
        args_dev_chip00_113->s3_block_count = 2;
        args_dev_chip00_113->dma_slot_expert_id[0] = -1;
        args_dev_chip00_113->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_113->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_113->dma_slot_expert_id[1] = -1;
        args_dev_chip00_113->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_113->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_113->dma_slot_expert_id[2] = -1;
        args_dev_chip00_113->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_113->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_113->dma_slot_expert_id[3] = -1;
        args_dev_chip00_113->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_113->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_113->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_113->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_113->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_113->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_113->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_113->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_113->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_113->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_113->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_113->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_113->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_113->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_113->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_113->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_113->A_token_bytes = 2048;
        args_dev_chip00_113->indiv_B_expert_stride = 262144;
        args_dev_chip00_113->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_113->indiv_B_tile_bytes = 131072;
        args_dev_chip00_113->indiv_D_tile_bytes = 16384;
        args_dev_chip00_113->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_113->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_113->indiv_N2 = 2;
        args_dev_chip00_113->indiv_down_N2 = 2;
        args_dev_chip00_113->indiv_K1 = 128;
        args_dev_chip00_113->indiv_N_per_block = 256;
        args_dev_chip00_113->indiv_down_K1 = 64;
        args_dev_chip00_113->indiv_down_N_per_block = 256;
        args_dev_chip00_113->rescale_mult = 1;
        args_dev_chip00_113->rescale_shift = 0;
        args_dev_chip00_113->output_expert_stride_bytes = 65536;
        args_dev_chip00_113->max_tokens_per_expert = 32;
        args_dev_chip00_113->gating_sp_addr = 0;
        args_dev_chip00_113->cond_node_index = 0;
        args_dev_chip00_113->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_113;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[107] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 768;
        device_kernel_list_chip_00[107] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 114 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_114 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_114->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_114->block_idx = 0;
        args_dev_chip00_114->gating_sp_addr = 0;
        args_dev_chip00_114->cond_node_index = 0;
        args_dev_chip00_114->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_114;
        device_arg_list_chip_00[108] = (uint32_t)(uintptr_t)args_dev_chip00_114;
        device_kernel_list_chip_00[108] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 115 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_115 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_115->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_115->block_idx = 0;
        args_dev_chip00_115->gating_sp_addr = 0;
        args_dev_chip00_115->cond_node_index = 0;
        args_dev_chip00_115->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_115;
        device_arg_list_chip_00[109] = (uint32_t)(uintptr_t)args_dev_chip00_115;
        device_kernel_list_chip_00[109] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 116 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_116 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_116->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_116->block_idx = 1;
        args_dev_chip00_116->gating_sp_addr = 0;
        args_dev_chip00_116->cond_node_index = 0;
        args_dev_chip00_116->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_116;
        device_arg_list_chip_00[110] = (uint32_t)(uintptr_t)args_dev_chip00_116;
        device_kernel_list_chip_00[110] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 117 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_117 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_117->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_117->block_idx = 1;
        args_dev_chip00_117->gating_sp_addr = 0;
        args_dev_chip00_117->cond_node_index = 0;
        args_dev_chip00_117->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_117;
        device_arg_list_chip_00[111] = (uint32_t)(uintptr_t)args_dev_chip00_117;
        device_kernel_list_chip_00[111] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 118 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_118 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 768);
        args_dev_chip00_118->ctrl = 0;
        args_dev_chip00_118->expert_id = 0;
        args_dev_chip00_118->token_start_rank = 0;
        args_dev_chip00_118->ntokens = 0;
        args_dev_chip00_118->m_s2_exec = 0;
        args_dev_chip00_118->m_s4_exec = 0;
        args_dev_chip00_118->wait_for_peer_slots = 0;
        args_dev_chip00_118->dma_slot_vd = 0;
        args_dev_chip00_118->s1_block_count = 2;
        args_dev_chip00_118->s3_block_count = 2;
        args_dev_chip00_118->dma_slot_expert_id[0] = -1;
        args_dev_chip00_118->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_118->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_118->dma_slot_expert_id[1] = -1;
        args_dev_chip00_118->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_118->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_118->dma_slot_expert_id[2] = -1;
        args_dev_chip00_118->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_118->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_118->dma_slot_expert_id[3] = -1;
        args_dev_chip00_118->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_118->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_118->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_118->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_118->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_118->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_118->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_118->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_118->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_118->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_118->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_118->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_118->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_118->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_118->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_118->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_118->A_token_bytes = 2048;
        args_dev_chip00_118->indiv_B_expert_stride = 262144;
        args_dev_chip00_118->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_118->indiv_B_tile_bytes = 131072;
        args_dev_chip00_118->indiv_D_tile_bytes = 16384;
        args_dev_chip00_118->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_118->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_118->indiv_N2 = 2;
        args_dev_chip00_118->indiv_down_N2 = 2;
        args_dev_chip00_118->indiv_K1 = 128;
        args_dev_chip00_118->indiv_N_per_block = 256;
        args_dev_chip00_118->indiv_down_K1 = 64;
        args_dev_chip00_118->indiv_down_N_per_block = 256;
        args_dev_chip00_118->rescale_mult = 1;
        args_dev_chip00_118->rescale_shift = 0;
        args_dev_chip00_118->output_expert_stride_bytes = 65536;
        args_dev_chip00_118->max_tokens_per_expert = 32;
        args_dev_chip00_118->gating_sp_addr = 0;
        args_dev_chip00_118->cond_node_index = 0;
        args_dev_chip00_118->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_118;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[112] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 768;
        device_kernel_list_chip_00[112] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 119 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_119 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 768);
        args_dev_chip00_119->ctrl = 0;
        args_dev_chip00_119->expert_id = 0;
        args_dev_chip00_119->token_start_rank = 0;
        args_dev_chip00_119->ntokens = 0;
        args_dev_chip00_119->m_s2_exec = 0;
        args_dev_chip00_119->m_s4_exec = 0;
        args_dev_chip00_119->wait_for_peer_slots = 0;
        args_dev_chip00_119->dma_slot_vd = 0;
        args_dev_chip00_119->s1_block_count = 2;
        args_dev_chip00_119->s3_block_count = 2;
        args_dev_chip00_119->dma_slot_expert_id[0] = -1;
        args_dev_chip00_119->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_119->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_119->dma_slot_expert_id[1] = -1;
        args_dev_chip00_119->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_119->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_119->dma_slot_expert_id[2] = -1;
        args_dev_chip00_119->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_119->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_119->dma_slot_expert_id[3] = -1;
        args_dev_chip00_119->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_119->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_119->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_119->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_119->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_119->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_119->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_119->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_119->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_119->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_119->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_119->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_119->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_119->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_119->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_119->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_119->A_token_bytes = 2048;
        args_dev_chip00_119->indiv_B_expert_stride = 262144;
        args_dev_chip00_119->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_119->indiv_B_tile_bytes = 131072;
        args_dev_chip00_119->indiv_D_tile_bytes = 16384;
        args_dev_chip00_119->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_119->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_119->indiv_N2 = 2;
        args_dev_chip00_119->indiv_down_N2 = 2;
        args_dev_chip00_119->indiv_K1 = 128;
        args_dev_chip00_119->indiv_N_per_block = 256;
        args_dev_chip00_119->indiv_down_K1 = 64;
        args_dev_chip00_119->indiv_down_N_per_block = 256;
        args_dev_chip00_119->rescale_mult = 1;
        args_dev_chip00_119->rescale_shift = 0;
        args_dev_chip00_119->output_expert_stride_bytes = 65536;
        args_dev_chip00_119->max_tokens_per_expert = 32;
        args_dev_chip00_119->gating_sp_addr = 0;
        args_dev_chip00_119->cond_node_index = 0;
        args_dev_chip00_119->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_119;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[113] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 768;
        device_kernel_list_chip_00[113] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 120 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_120 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_120->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_120->block_idx = 0;
        args_dev_chip00_120->gating_sp_addr = 0;
        args_dev_chip00_120->cond_node_index = 0;
        args_dev_chip00_120->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_120;
        device_arg_list_chip_00[114] = (uint32_t)(uintptr_t)args_dev_chip00_120;
        device_kernel_list_chip_00[114] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 121 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_121 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_121->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_121->block_idx = 0;
        args_dev_chip00_121->gating_sp_addr = 0;
        args_dev_chip00_121->cond_node_index = 0;
        args_dev_chip00_121->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_121;
        device_arg_list_chip_00[115] = (uint32_t)(uintptr_t)args_dev_chip00_121;
        device_kernel_list_chip_00[115] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 122 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_122 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_122->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_122->block_idx = 1;
        args_dev_chip00_122->gating_sp_addr = 0;
        args_dev_chip00_122->cond_node_index = 0;
        args_dev_chip00_122->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_122;
        device_arg_list_chip_00[116] = (uint32_t)(uintptr_t)args_dev_chip00_122;
        device_kernel_list_chip_00[116] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 123 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_123 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_123->task_arg_addr = (uint64_t)ptr_c2_indiv_dyn_args + 768;
        args_dev_chip00_123->block_idx = 1;
        args_dev_chip00_123->gating_sp_addr = 0;
        args_dev_chip00_123->cond_node_index = 0;
        args_dev_chip00_123->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_123;
        device_arg_list_chip_00[117] = (uint32_t)(uintptr_t)args_dev_chip00_123;
        device_kernel_list_chip_00[117] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 124 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_124 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 768);
        args_dev_chip00_124->ctrl = 0;
        args_dev_chip00_124->expert_id = 0;
        args_dev_chip00_124->token_start_rank = 0;
        args_dev_chip00_124->ntokens = 0;
        args_dev_chip00_124->m_s2_exec = 0;
        args_dev_chip00_124->m_s4_exec = 0;
        args_dev_chip00_124->wait_for_peer_slots = 0;
        args_dev_chip00_124->dma_slot_vd = 0;
        args_dev_chip00_124->s1_block_count = 2;
        args_dev_chip00_124->s3_block_count = 2;
        args_dev_chip00_124->dma_slot_expert_id[0] = -1;
        args_dev_chip00_124->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_124->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_124->dma_slot_expert_id[1] = -1;
        args_dev_chip00_124->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_124->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_124->dma_slot_expert_id[2] = -1;
        args_dev_chip00_124->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_124->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_124->dma_slot_expert_id[3] = -1;
        args_dev_chip00_124->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_124->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_124->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_124->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_124->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_124->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_124->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_124->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_124->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_124->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_124->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_124->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_124->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_124->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_124->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_124->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_124->A_token_bytes = 2048;
        args_dev_chip00_124->indiv_B_expert_stride = 262144;
        args_dev_chip00_124->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_124->indiv_B_tile_bytes = 131072;
        args_dev_chip00_124->indiv_D_tile_bytes = 16384;
        args_dev_chip00_124->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_124->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_124->indiv_N2 = 2;
        args_dev_chip00_124->indiv_down_N2 = 2;
        args_dev_chip00_124->indiv_K1 = 128;
        args_dev_chip00_124->indiv_N_per_block = 256;
        args_dev_chip00_124->indiv_down_K1 = 64;
        args_dev_chip00_124->indiv_down_N_per_block = 256;
        args_dev_chip00_124->rescale_mult = 1;
        args_dev_chip00_124->rescale_shift = 0;
        args_dev_chip00_124->output_expert_stride_bytes = 65536;
        args_dev_chip00_124->max_tokens_per_expert = 32;
        args_dev_chip00_124->gating_sp_addr = 0;
        args_dev_chip00_124->cond_node_index = 0;
        args_dev_chip00_124->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_124;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[118] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 768;
        device_kernel_list_chip_00[118] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 125 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_125 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 768);
        args_dev_chip00_125->ctrl = 0;
        args_dev_chip00_125->expert_id = 0;
        args_dev_chip00_125->token_start_rank = 0;
        args_dev_chip00_125->ntokens = 0;
        args_dev_chip00_125->m_s2_exec = 0;
        args_dev_chip00_125->m_s4_exec = 0;
        args_dev_chip00_125->wait_for_peer_slots = 0;
        args_dev_chip00_125->dma_slot_vd = 0;
        args_dev_chip00_125->s1_block_count = 2;
        args_dev_chip00_125->s3_block_count = 2;
        args_dev_chip00_125->dma_slot_expert_id[0] = -1;
        args_dev_chip00_125->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_125->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_125->dma_slot_expert_id[1] = -1;
        args_dev_chip00_125->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_125->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_125->dma_slot_expert_id[2] = -1;
        args_dev_chip00_125->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_125->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_125->dma_slot_expert_id[3] = -1;
        args_dev_chip00_125->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_125->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_125->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_125->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_125->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_125->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_125->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_125->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_125->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_125->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_125->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_125->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_125->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_125->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_125->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_125->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_125->A_token_bytes = 2048;
        args_dev_chip00_125->indiv_B_expert_stride = 262144;
        args_dev_chip00_125->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_125->indiv_B_tile_bytes = 131072;
        args_dev_chip00_125->indiv_D_tile_bytes = 16384;
        args_dev_chip00_125->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_125->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_125->indiv_N2 = 2;
        args_dev_chip00_125->indiv_down_N2 = 2;
        args_dev_chip00_125->indiv_K1 = 128;
        args_dev_chip00_125->indiv_N_per_block = 256;
        args_dev_chip00_125->indiv_down_K1 = 64;
        args_dev_chip00_125->indiv_down_N_per_block = 256;
        args_dev_chip00_125->rescale_mult = 1;
        args_dev_chip00_125->rescale_shift = 0;
        args_dev_chip00_125->output_expert_stride_bytes = 65536;
        args_dev_chip00_125->max_tokens_per_expert = 32;
        args_dev_chip00_125->gating_sp_addr = 0;
        args_dev_chip00_125->cond_node_index = 0;
        args_dev_chip00_125->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_125;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[119] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 768;
        device_kernel_list_chip_00[119] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 126 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_126 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c2_stage + 768);
        args_dev_chip00_126->ctrl = 0;
        args_dev_chip00_126->expert_id = 0;
        args_dev_chip00_126->token_start_rank = 0;
        args_dev_chip00_126->ntokens = 0;
        args_dev_chip00_126->m_s2_exec = 0;
        args_dev_chip00_126->m_s4_exec = 0;
        args_dev_chip00_126->wait_for_peer_slots = 0;
        args_dev_chip00_126->dma_slot_vd = 0;
        args_dev_chip00_126->s1_block_count = 2;
        args_dev_chip00_126->s3_block_count = 2;
        args_dev_chip00_126->dma_slot_expert_id[0] = -1;
        args_dev_chip00_126->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_126->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_126->dma_slot_expert_id[1] = -1;
        args_dev_chip00_126->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_126->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_126->dma_slot_expert_id[2] = -1;
        args_dev_chip00_126->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_126->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_126->dma_slot_expert_id[3] = -1;
        args_dev_chip00_126->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_126->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_126->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_126->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_126->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_126->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_126->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_126->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_126->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_126->l1_a_addr = (uint32_t)ptr_c2_indiv_l1_a;
        args_dev_chip00_126->l1_b_gate_addr = (uint32_t)ptr_c2_indiv_l1_b_gate;
        args_dev_chip00_126->l1_b_up_addr = (uint32_t)ptr_c2_indiv_l1_b_up;
        args_dev_chip00_126->l1_b_down_addr = (uint32_t)ptr_c2_indiv_l1_b_down;
        args_dev_chip00_126->l1_d_addr = (uint32_t)ptr_c2_indiv_l1_d;
        args_dev_chip00_126->l1_down_d_addr = (uint32_t)ptr_c2_indiv_l1_down_d;
        args_dev_chip00_126->l1_d1_scratch_addr = (uint32_t)ptr_c2_indiv_l1_d1_scratch;
        args_dev_chip00_126->A_token_bytes = 2048;
        args_dev_chip00_126->indiv_B_expert_stride = 262144;
        args_dev_chip00_126->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_126->indiv_B_tile_bytes = 131072;
        args_dev_chip00_126->indiv_D_tile_bytes = 16384;
        args_dev_chip00_126->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_126->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_126->indiv_N2 = 2;
        args_dev_chip00_126->indiv_down_N2 = 2;
        args_dev_chip00_126->indiv_K1 = 128;
        args_dev_chip00_126->indiv_N_per_block = 256;
        args_dev_chip00_126->indiv_down_K1 = 64;
        args_dev_chip00_126->indiv_down_N_per_block = 256;
        args_dev_chip00_126->rescale_mult = 1;
        args_dev_chip00_126->rescale_shift = 0;
        args_dev_chip00_126->output_expert_stride_bytes = 65536;
        args_dev_chip00_126->max_tokens_per_expert = 32;
        args_dev_chip00_126->gating_sp_addr = 0;
        args_dev_chip00_126->cond_node_index = 0;
        args_dev_chip00_126->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_126;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c2_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c2_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[120] = (uint32_t)(uintptr_t)ptr_c2_indiv_dyn_args + 768;
        device_kernel_list_chip_00[120] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 127 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_gather_s1 (__snax_bingo_kernel_moe_dynamic_expert_gather_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_127 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 768);
        args_dev_chip00_127->ctrl = 0;
        args_dev_chip00_127->expert_id = 0;
        args_dev_chip00_127->token_start_rank = 0;
        args_dev_chip00_127->ntokens = 0;
        args_dev_chip00_127->m_s2_exec = 0;
        args_dev_chip00_127->m_s4_exec = 0;
        args_dev_chip00_127->wait_for_peer_slots = 0;
        args_dev_chip00_127->dma_slot_vd = 0;
        args_dev_chip00_127->s1_block_count = 2;
        args_dev_chip00_127->s3_block_count = 2;
        args_dev_chip00_127->dma_slot_expert_id[0] = -1;
        args_dev_chip00_127->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_127->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_127->dma_slot_expert_id[1] = -1;
        args_dev_chip00_127->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_127->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_127->dma_slot_expert_id[2] = -1;
        args_dev_chip00_127->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_127->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_127->dma_slot_expert_id[3] = -1;
        args_dev_chip00_127->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_127->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_127->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_127->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_127->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_127->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_127->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_127->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_127->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_127->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_127->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_127->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_127->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_127->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_127->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_127->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_127->A_token_bytes = 2048;
        args_dev_chip00_127->indiv_B_expert_stride = 262144;
        args_dev_chip00_127->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_127->indiv_B_tile_bytes = 131072;
        args_dev_chip00_127->indiv_D_tile_bytes = 16384;
        args_dev_chip00_127->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_127->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_127->indiv_N2 = 2;
        args_dev_chip00_127->indiv_down_N2 = 2;
        args_dev_chip00_127->indiv_K1 = 128;
        args_dev_chip00_127->indiv_N_per_block = 256;
        args_dev_chip00_127->indiv_down_K1 = 64;
        args_dev_chip00_127->indiv_down_N_per_block = 256;
        args_dev_chip00_127->rescale_mult = 1;
        args_dev_chip00_127->rescale_shift = 0;
        args_dev_chip00_127->output_expert_stride_bytes = 65536;
        args_dev_chip00_127->max_tokens_per_expert = 32;
        args_dev_chip00_127->gating_sp_addr = 0;
        args_dev_chip00_127->cond_node_index = 0;
        args_dev_chip00_127->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_127;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[121] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 768;
        device_kernel_list_chip_00[121] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_gather_s1");
        // Node ID: 128 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_128 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_128->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_128->block_idx = 0;
        args_dev_chip00_128->gating_sp_addr = 0;
        args_dev_chip00_128->cond_node_index = 0;
        args_dev_chip00_128->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_128;
        device_arg_list_chip_00[122] = (uint32_t)(uintptr_t)args_dev_chip00_128;
        device_kernel_list_chip_00[122] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 129 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_129 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_129->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_129->block_idx = 0;
        args_dev_chip00_129->gating_sp_addr = 0;
        args_dev_chip00_129->cond_node_index = 0;
        args_dev_chip00_129->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_129;
        device_arg_list_chip_00[123] = (uint32_t)(uintptr_t)args_dev_chip00_129;
        device_kernel_list_chip_00[123] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 130 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_130 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_130->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_130->block_idx = 1;
        args_dev_chip00_130->gating_sp_addr = 0;
        args_dev_chip00_130->cond_node_index = 0;
        args_dev_chip00_130->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_130;
        device_arg_list_chip_00[124] = (uint32_t)(uintptr_t)args_dev_chip00_130;
        device_kernel_list_chip_00[124] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block");
        // Node ID: 131 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_131 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_131->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_131->block_idx = 1;
        args_dev_chip00_131->gating_sp_addr = 0;
        args_dev_chip00_131->cond_node_index = 0;
        args_dev_chip00_131->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_131;
        device_arg_list_chip_00[125] = (uint32_t)(uintptr_t)args_dev_chip00_131;
        device_kernel_list_chip_00[125] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block");
        // Node ID: 132 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_132 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 768);
        args_dev_chip00_132->ctrl = 0;
        args_dev_chip00_132->expert_id = 0;
        args_dev_chip00_132->token_start_rank = 0;
        args_dev_chip00_132->ntokens = 0;
        args_dev_chip00_132->m_s2_exec = 0;
        args_dev_chip00_132->m_s4_exec = 0;
        args_dev_chip00_132->wait_for_peer_slots = 0;
        args_dev_chip00_132->dma_slot_vd = 0;
        args_dev_chip00_132->s1_block_count = 2;
        args_dev_chip00_132->s3_block_count = 2;
        args_dev_chip00_132->dma_slot_expert_id[0] = -1;
        args_dev_chip00_132->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_132->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_132->dma_slot_expert_id[1] = -1;
        args_dev_chip00_132->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_132->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_132->dma_slot_expert_id[2] = -1;
        args_dev_chip00_132->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_132->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_132->dma_slot_expert_id[3] = -1;
        args_dev_chip00_132->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_132->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_132->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_132->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_132->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_132->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_132->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_132->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_132->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_132->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_132->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_132->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_132->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_132->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_132->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_132->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_132->A_token_bytes = 2048;
        args_dev_chip00_132->indiv_B_expert_stride = 262144;
        args_dev_chip00_132->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_132->indiv_B_tile_bytes = 131072;
        args_dev_chip00_132->indiv_D_tile_bytes = 16384;
        args_dev_chip00_132->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_132->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_132->indiv_N2 = 2;
        args_dev_chip00_132->indiv_down_N2 = 2;
        args_dev_chip00_132->indiv_K1 = 128;
        args_dev_chip00_132->indiv_N_per_block = 256;
        args_dev_chip00_132->indiv_down_K1 = 64;
        args_dev_chip00_132->indiv_down_N_per_block = 256;
        args_dev_chip00_132->rescale_mult = 1;
        args_dev_chip00_132->rescale_shift = 0;
        args_dev_chip00_132->output_expert_stride_bytes = 65536;
        args_dev_chip00_132->max_tokens_per_expert = 32;
        args_dev_chip00_132->gating_sp_addr = 0;
        args_dev_chip00_132->cond_node_index = 0;
        args_dev_chip00_132->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_132;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[126] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 768;
        device_kernel_list_chip_00[126] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down");
        // Node ID: 133 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full (__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_133 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 768);
        args_dev_chip00_133->ctrl = 0;
        args_dev_chip00_133->expert_id = 0;
        args_dev_chip00_133->token_start_rank = 0;
        args_dev_chip00_133->ntokens = 0;
        args_dev_chip00_133->m_s2_exec = 0;
        args_dev_chip00_133->m_s4_exec = 0;
        args_dev_chip00_133->wait_for_peer_slots = 0;
        args_dev_chip00_133->dma_slot_vd = 0;
        args_dev_chip00_133->s1_block_count = 2;
        args_dev_chip00_133->s3_block_count = 2;
        args_dev_chip00_133->dma_slot_expert_id[0] = -1;
        args_dev_chip00_133->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_133->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_133->dma_slot_expert_id[1] = -1;
        args_dev_chip00_133->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_133->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_133->dma_slot_expert_id[2] = -1;
        args_dev_chip00_133->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_133->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_133->dma_slot_expert_id[3] = -1;
        args_dev_chip00_133->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_133->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_133->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_133->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_133->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_133->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_133->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_133->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_133->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_133->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_133->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_133->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_133->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_133->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_133->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_133->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_133->A_token_bytes = 2048;
        args_dev_chip00_133->indiv_B_expert_stride = 262144;
        args_dev_chip00_133->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_133->indiv_B_tile_bytes = 131072;
        args_dev_chip00_133->indiv_D_tile_bytes = 16384;
        args_dev_chip00_133->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_133->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_133->indiv_N2 = 2;
        args_dev_chip00_133->indiv_down_N2 = 2;
        args_dev_chip00_133->indiv_K1 = 128;
        args_dev_chip00_133->indiv_N_per_block = 256;
        args_dev_chip00_133->indiv_down_K1 = 64;
        args_dev_chip00_133->indiv_down_N_per_block = 256;
        args_dev_chip00_133->rescale_mult = 1;
        args_dev_chip00_133->rescale_shift = 0;
        args_dev_chip00_133->output_expert_stride_bytes = 65536;
        args_dev_chip00_133->max_tokens_per_expert = 32;
        args_dev_chip00_133->gating_sp_addr = 0;
        args_dev_chip00_133->cond_node_index = 0;
        args_dev_chip00_133->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_133;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[127] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 768;
        device_kernel_list_chip_00[127] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full");
        // Node ID: 134 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_134 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_134->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_134->block_idx = 0;
        args_dev_chip00_134->gating_sp_addr = 0;
        args_dev_chip00_134->cond_node_index = 0;
        args_dev_chip00_134->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_134;
        device_arg_list_chip_00[128] = (uint32_t)(uintptr_t)args_dev_chip00_134;
        device_kernel_list_chip_00[128] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 135 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_135 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_135->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_135->block_idx = 0;
        args_dev_chip00_135->gating_sp_addr = 0;
        args_dev_chip00_135->cond_node_index = 0;
        args_dev_chip00_135->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_135;
        device_arg_list_chip_00[129] = (uint32_t)(uintptr_t)args_dev_chip00_135;
        device_kernel_list_chip_00[129] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 136 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_load_down_block (__snax_bingo_kernel_moe_dynamic_expert_load_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_136 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_136->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_136->block_idx = 1;
        args_dev_chip00_136->gating_sp_addr = 0;
        args_dev_chip00_136->cond_node_index = 0;
        args_dev_chip00_136->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_136;
        device_arg_list_chip_00[130] = (uint32_t)(uintptr_t)args_dev_chip00_136;
        device_kernel_list_chip_00[130] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_load_down_block");
        // Node ID: 137 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_block (__snax_bingo_kernel_moe_dynamic_expert_compute_down_block)
        __snax_bingo_kernel_moe_dynamic_expert_block_args_t* args_dev_chip00_137 = (__snax_bingo_kernel_moe_dynamic_expert_block_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_moe_dynamic_expert_block_args_t));
        args_dev_chip00_137->task_arg_addr = (uint64_t)ptr_c3_indiv_dyn_args + 768;
        args_dev_chip00_137->block_idx = 1;
        args_dev_chip00_137->gating_sp_addr = 0;
        args_dev_chip00_137->cond_node_index = 0;
        args_dev_chip00_137->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_137;
        device_arg_list_chip_00[131] = (uint32_t)(uintptr_t)args_dev_chip00_137;
        device_kernel_list_chip_00[131] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_block");
        // Node ID: 138 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1 (__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_138 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 768);
        args_dev_chip00_138->ctrl = 0;
        args_dev_chip00_138->expert_id = 0;
        args_dev_chip00_138->token_start_rank = 0;
        args_dev_chip00_138->ntokens = 0;
        args_dev_chip00_138->m_s2_exec = 0;
        args_dev_chip00_138->m_s4_exec = 0;
        args_dev_chip00_138->wait_for_peer_slots = 0;
        args_dev_chip00_138->dma_slot_vd = 0;
        args_dev_chip00_138->s1_block_count = 2;
        args_dev_chip00_138->s3_block_count = 2;
        args_dev_chip00_138->dma_slot_expert_id[0] = -1;
        args_dev_chip00_138->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_138->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_138->dma_slot_expert_id[1] = -1;
        args_dev_chip00_138->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_138->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_138->dma_slot_expert_id[2] = -1;
        args_dev_chip00_138->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_138->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_138->dma_slot_expert_id[3] = -1;
        args_dev_chip00_138->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_138->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_138->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_138->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_138->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_138->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_138->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_138->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_138->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_138->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_138->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_138->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_138->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_138->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_138->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_138->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_138->A_token_bytes = 2048;
        args_dev_chip00_138->indiv_B_expert_stride = 262144;
        args_dev_chip00_138->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_138->indiv_B_tile_bytes = 131072;
        args_dev_chip00_138->indiv_D_tile_bytes = 16384;
        args_dev_chip00_138->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_138->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_138->indiv_N2 = 2;
        args_dev_chip00_138->indiv_down_N2 = 2;
        args_dev_chip00_138->indiv_K1 = 128;
        args_dev_chip00_138->indiv_N_per_block = 256;
        args_dev_chip00_138->indiv_down_K1 = 64;
        args_dev_chip00_138->indiv_down_N_per_block = 256;
        args_dev_chip00_138->rescale_mult = 1;
        args_dev_chip00_138->rescale_shift = 0;
        args_dev_chip00_138->output_expert_stride_bytes = 65536;
        args_dev_chip00_138->max_tokens_per_expert = 32;
        args_dev_chip00_138->gating_sp_addr = 0;
        args_dev_chip00_138->cond_node_index = 0;
        args_dev_chip00_138->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_138;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[132] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 768;
        device_kernel_list_chip_00[132] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1");
        // Node ID: 139 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_moe_dynamic_expert_compute_down_full (__snax_bingo_kernel_moe_dynamic_expert_compute_down_full)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_139 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 768);
        args_dev_chip00_139->ctrl = 0;
        args_dev_chip00_139->expert_id = 0;
        args_dev_chip00_139->token_start_rank = 0;
        args_dev_chip00_139->ntokens = 0;
        args_dev_chip00_139->m_s2_exec = 0;
        args_dev_chip00_139->m_s4_exec = 0;
        args_dev_chip00_139->wait_for_peer_slots = 0;
        args_dev_chip00_139->dma_slot_vd = 0;
        args_dev_chip00_139->s1_block_count = 2;
        args_dev_chip00_139->s3_block_count = 2;
        args_dev_chip00_139->dma_slot_expert_id[0] = -1;
        args_dev_chip00_139->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_139->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_139->dma_slot_expert_id[1] = -1;
        args_dev_chip00_139->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_139->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_139->dma_slot_expert_id[2] = -1;
        args_dev_chip00_139->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_139->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_139->dma_slot_expert_id[3] = -1;
        args_dev_chip00_139->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_139->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_139->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_139->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_139->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_139->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_139->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_139->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_139->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_139->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_139->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_139->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_139->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_139->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_139->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_139->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_139->A_token_bytes = 2048;
        args_dev_chip00_139->indiv_B_expert_stride = 262144;
        args_dev_chip00_139->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_139->indiv_B_tile_bytes = 131072;
        args_dev_chip00_139->indiv_D_tile_bytes = 16384;
        args_dev_chip00_139->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_139->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_139->indiv_N2 = 2;
        args_dev_chip00_139->indiv_down_N2 = 2;
        args_dev_chip00_139->indiv_K1 = 128;
        args_dev_chip00_139->indiv_N_per_block = 256;
        args_dev_chip00_139->indiv_down_K1 = 64;
        args_dev_chip00_139->indiv_down_N_per_block = 256;
        args_dev_chip00_139->rescale_mult = 1;
        args_dev_chip00_139->rescale_shift = 0;
        args_dev_chip00_139->output_expert_stride_bytes = 65536;
        args_dev_chip00_139->max_tokens_per_expert = 32;
        args_dev_chip00_139->gating_sp_addr = 0;
        args_dev_chip00_139->cond_node_index = 0;
        args_dev_chip00_139->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_139;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[133] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 768;
        device_kernel_list_chip_00[133] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_compute_down_full");
        // Node ID: 140 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_moe_dynamic_expert_store (__snax_bingo_kernel_moe_dynamic_expert_store)
        __snax_bingo_kernel_moe_dynamic_expert_args_t* args_dev_chip00_140 = (__snax_bingo_kernel_moe_dynamic_expert_args_t*)(ptr_l3_c3_stage + 768);
        args_dev_chip00_140->ctrl = 0;
        args_dev_chip00_140->expert_id = 0;
        args_dev_chip00_140->token_start_rank = 0;
        args_dev_chip00_140->ntokens = 0;
        args_dev_chip00_140->m_s2_exec = 0;
        args_dev_chip00_140->m_s4_exec = 0;
        args_dev_chip00_140->wait_for_peer_slots = 0;
        args_dev_chip00_140->dma_slot_vd = 0;
        args_dev_chip00_140->s1_block_count = 2;
        args_dev_chip00_140->s3_block_count = 2;
        args_dev_chip00_140->dma_slot_expert_id[0] = -1;
        args_dev_chip00_140->dma_slot_idma_seq[0] = 0;
        args_dev_chip00_140->dma_slot_xdma_seq[0] = 0;
        args_dev_chip00_140->dma_slot_expert_id[1] = -1;
        args_dev_chip00_140->dma_slot_idma_seq[1] = 0;
        args_dev_chip00_140->dma_slot_xdma_seq[1] = 0;
        args_dev_chip00_140->dma_slot_expert_id[2] = -1;
        args_dev_chip00_140->dma_slot_idma_seq[2] = 0;
        args_dev_chip00_140->dma_slot_xdma_seq[2] = 0;
        args_dev_chip00_140->dma_slot_expert_id[3] = -1;
        args_dev_chip00_140->dma_slot_idma_seq[3] = 0;
        args_dev_chip00_140->dma_slot_xdma_seq[3] = 0;
        args_dev_chip00_140->token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_dev_chip00_140->input_A_l3_base = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_140->indiv_gate_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_gate_B)));
        args_dev_chip00_140->indiv_up_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_up_B)));
        args_dev_chip00_140->indiv_down_B_l3 = (uint64_t)(chiplet_addr_transform((uint64_t)((uintptr_t)indiv_down_B)));
        args_dev_chip00_140->output_l3_base = (uint64_t)ptr_l3_indiv_down_out;
        args_dev_chip00_140->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_dev_chip00_140->l1_a_addr = (uint32_t)ptr_c3_indiv_l1_a;
        args_dev_chip00_140->l1_b_gate_addr = (uint32_t)ptr_c3_indiv_l1_b_gate;
        args_dev_chip00_140->l1_b_up_addr = (uint32_t)ptr_c3_indiv_l1_b_up;
        args_dev_chip00_140->l1_b_down_addr = (uint32_t)ptr_c3_indiv_l1_b_down;
        args_dev_chip00_140->l1_d_addr = (uint32_t)ptr_c3_indiv_l1_d;
        args_dev_chip00_140->l1_down_d_addr = (uint32_t)ptr_c3_indiv_l1_down_d;
        args_dev_chip00_140->l1_d1_scratch_addr = (uint32_t)ptr_c3_indiv_l1_d1_scratch;
        args_dev_chip00_140->A_token_bytes = 2048;
        args_dev_chip00_140->indiv_B_expert_stride = 262144;
        args_dev_chip00_140->indiv_down_B_expert_stride = 262144;
        args_dev_chip00_140->indiv_B_tile_bytes = 131072;
        args_dev_chip00_140->indiv_D_tile_bytes = 16384;
        args_dev_chip00_140->indiv_down_B_tile_bytes = 65536;
        args_dev_chip00_140->indiv_down_D_tile_bytes = 16384;
        args_dev_chip00_140->indiv_N2 = 2;
        args_dev_chip00_140->indiv_down_N2 = 2;
        args_dev_chip00_140->indiv_K1 = 128;
        args_dev_chip00_140->indiv_N_per_block = 256;
        args_dev_chip00_140->indiv_down_K1 = 64;
        args_dev_chip00_140->indiv_down_N_per_block = 256;
        args_dev_chip00_140->rescale_mult = 1;
        args_dev_chip00_140->rescale_shift = 0;
        args_dev_chip00_140->output_expert_stride_bytes = 65536;
        args_dev_chip00_140->max_tokens_per_expert = 32;
        args_dev_chip00_140->gating_sp_addr = 0;
        args_dev_chip00_140->cond_node_index = 0;
        args_dev_chip00_140->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_140;
        sys_dma_blk_memcpy(get_current_chip_id(), (uint64_t)(ptr_c3_indiv_dyn_args + 768), (uint64_t)chiplet_addr_transform_full(get_current_chip_id(), (uint64_t)(ptr_l3_c3_stage + 768)), sizeof(__snax_bingo_kernel_moe_dynamic_expert_args_t));
        device_arg_list_chip_00[134] = (uint32_t)(uintptr_t)ptr_c3_indiv_dyn_args + 768;
        device_kernel_list_chip_00[134] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_moe_dynamic_expert_store");
        // Node ID: 141 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry (__host_bingo_kernel_entry)
        host_arg_list_chip_00[6] = 0;
        host_kernel_list_chip_00[6] = (uint64_t)(uintptr_t)&__host_bingo_kernel_entry;
        // Node ID: 142 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_142 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_142->exit_code = 0;
        args_dev_chip00_142->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_142;
        device_arg_list_chip_00[135] = (uint32_t)(uintptr_t)args_dev_chip00_142;
        device_kernel_list_chip_00[135] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 143 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_143 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_143->exit_code = 0;
        args_dev_chip00_143->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_143;
        device_arg_list_chip_00[136] = (uint32_t)(uintptr_t)args_dev_chip00_143;
        device_kernel_list_chip_00[136] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 144 Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_144 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_144->exit_code = 0;
        args_dev_chip00_144->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_144;
        device_arg_list_chip_00[137] = (uint32_t)(uintptr_t)args_dev_chip00_144;
        device_kernel_list_chip_00[137] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 145 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_145 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_145->exit_code = 0;
        args_dev_chip00_145->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_145;
        device_arg_list_chip_00[138] = (uint32_t)(uintptr_t)args_dev_chip00_145;
        device_kernel_list_chip_00[138] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 146 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_146 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_146->exit_code = 0;
        args_dev_chip00_146->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_146;
        device_arg_list_chip_00[139] = (uint32_t)(uintptr_t)args_dev_chip00_146;
        device_kernel_list_chip_00[139] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 147 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_147 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_147->exit_code = 0;
        args_dev_chip00_147->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_147;
        device_arg_list_chip_00[140] = (uint32_t)(uintptr_t)args_dev_chip00_147;
        device_kernel_list_chip_00[140] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 148 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_148 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_148->exit_code = 0;
        args_dev_chip00_148->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_148;
        device_arg_list_chip_00[141] = (uint32_t)(uintptr_t)args_dev_chip00_148;
        device_kernel_list_chip_00[141] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 149 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_149 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_149->exit_code = 0;
        args_dev_chip00_149->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_149;
        device_arg_list_chip_00[142] = (uint32_t)(uintptr_t)args_dev_chip00_149;
        device_kernel_list_chip_00[142] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 150 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit (__host_bingo_kernel_exit)
        __host_bingo_kernel_exit_args_t* args_host_chip00_150 = (__host_bingo_kernel_exit_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_exit_args_t));
        args_host_chip00_150->exit_code = 0;
        args_host_chip00_150->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_150;
        host_arg_list_chip_00[7] = (uint64_t)(uintptr_t)args_host_chip00_150;
        host_kernel_list_chip_00[7] = (uint64_t)(uintptr_t)&__host_bingo_kernel_exit;

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
        if (err) return err;
    }
    return 0;
}
