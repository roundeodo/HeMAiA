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
char kernel_name_list[39][64] = {
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
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 16
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 17
    "__snax_bingo_kernel_xdma_1d_copy", // Node ID 18
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 19
    "__snax_bingo_kernel_dual_vc_l15_moe_full", // Node ID 20
    "__host_bingo_kernel_idma", // Node ID 21
    "__snax_bingo_kernel_dual_vc_l15_moe_full", // Node ID 22
    "__host_bingo_kernel_idma", // Node ID 23
    "__snax_bingo_kernel_dual_vc_gemm_full", // Node ID 24
    "__host_bingo_kernel_idma", // Node ID 25
    "__host_bingo_kernel_moe_router_schedule", // Node ID 26
    "__host_bingo_kernel_moe_prepare_request", // Node ID 27
    "__host_bingo_kernel_moe_execute", // Node ID 28
    "__host_bingo_kernel_entry", // Node ID 29
    "__snax_bingo_kernel_exit", // Node ID 30
    "__snax_bingo_kernel_exit", // Node ID 31
    "__snax_bingo_kernel_exit", // Node ID 32
    "__snax_bingo_kernel_exit", // Node ID 33
    "__snax_bingo_kernel_exit", // Node ID 34
    "__snax_bingo_kernel_exit", // Node ID 35
    "__snax_bingo_kernel_exit", // Node ID 36
    "__snax_bingo_kernel_exit", // Node ID 37
    "__host_bingo_kernel_exit", // Node ID 38
};
*/

int kernel_execution(){
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing multi_cluster_MoE Workload\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    uint32_t current_chip_id = get_current_chip_id();
    if (current_chip_id == 0x00) {
        uint32_t num_total_tasks = 67;
        // Task Description List
        uint32_t bingo_hw_scheduler_num_task_desc_chip_00 = 67;
        uint64_t* bingo_hw_scheduler_task_desc_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, bingo_hw_scheduler_num_task_desc_chip_00 * sizeof(uint64_t));
        bingo_hw_scheduler_task_desc_list_chip_00[0] = 0x0004002100003A00; // Node ID 29
            // Fields: Type=0, TaskID=29
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[1] = 0x0004002100006880; // Node ID 52
            // Fields: Type=1, TaskID=52
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[2] = 0x0004003280000000; // Node ID 0
            // Fields: Type=0, TaskID=0
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[3] = 0x0004802080004E80; // Node ID 39
            // Fields: Type=1, TaskID=39
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[4] = 0x0000000A80006A80; // Node ID 53
            // Fields: Type=1, TaskID=53
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
        bingo_hw_scheduler_task_desc_list_chip_00[7] = 0x00050020A0005080; // Node ID 40
            // Fields: Type=1, TaskID=40
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[8] = 0x0000000AA0006C80; // Node ID 54
            // Fields: Type=1, TaskID=54
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[9] = 0x0004002A80001C00; // Node ID 14
            // Fields: Type=0, TaskID=14
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[10] = 0x0004802080005880; // Node ID 44
            // Fields: Type=1, TaskID=44
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[11] = 0x0005002AC0000400; // Node ID 2
            // Fields: Type=0, TaskID=2
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[12] = 0x0004002A80001E00; // Node ID 15
            // Fields: Type=0, TaskID=15
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[13] = 0x0004802AA0000C00; // Node ID 6
            // Fields: Type=0, TaskID=6
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[14] = 0x00058020C0005280; // Node ID 41
            // Fields: Type=1, TaskID=41
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[15] = 0x0000000AC0006E80; // Node ID 55
            // Fields: Type=1, TaskID=55
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[16] = 0x0002002A80002000; // Node ID 16
            // Fields: Type=0, TaskID=16
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[17] = 0x0004802AA0002200; // Node ID 17
            // Fields: Type=0, TaskID=17
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[18] = 0x00050020A0005A80; // Node ID 45
            // Fields: Type=1, TaskID=45
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[19] = 0x0005802AE0000600; // Node ID 3
            // Fields: Type=0, TaskID=3
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[20] = 0x0004002080006080; // Node ID 48
            // Fields: Type=1, TaskID=48
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[21] = 0x0004802AA0002400; // Node ID 18
            // Fields: Type=0, TaskID=18
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[22] = 0x0005002AC0000E00; // Node ID 7
            // Fields: Type=0, TaskID=7
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[23] = 0x0005802AE0000800; // Node ID 4
            // Fields: Type=0, TaskID=4
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[24] = 0x0002802AA0002600; // Node ID 19
            // Fields: Type=0, TaskID=19
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[25] = 0x0005802AC0001000; // Node ID 8
            // Fields: Type=0, TaskID=8
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[26] = 0x00040020E0005480; // Node ID 42
            // Fields: Type=1, TaskID=42
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[27] = 0x00048020E0005680; // Node ID 43
            // Fields: Type=1, TaskID=43
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[28] = 0x00048020A0006280; // Node ID 49
            // Fields: Type=1, TaskID=49
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[29] = 0x0000000AE0007080; // Node ID 56
            // Fields: Type=1, TaskID=56
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[30] = 0x0000000A80007280; // Node ID 57
            // Fields: Type=1, TaskID=57
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[31] = 0x0000000AA0007880; // Node ID 60
            // Fields: Type=1, TaskID=60
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[32] = 0x0005802AE0001200; // Node ID 9
            // Fields: Type=0, TaskID=9
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[33] = 0x0004802AE0001400; // Node ID 10
            // Fields: Type=0, TaskID=10
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[34] = 0x00040020E0005C80; // Node ID 46
            // Fields: Type=1, TaskID=46
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[35] = 0x0000000AA0007680; // Node ID 59
            // Fields: Type=1, TaskID=59
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[36] = 0x0000000A80007480; // Node ID 58
            // Fields: Type=1, TaskID=58
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[37] = 0x0002802AA0001800; // Node ID 12
            // Fields: Type=0, TaskID=12
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[38] = 0x0002002A80001600; // Node ID 11
            // Fields: Type=0, TaskID=11
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[39] = 0x0000000A20007C80; // Node ID 62
            // Fields: Type=1, TaskID=62
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[40] = 0x0005802080005E80; // Node ID 47
            // Fields: Type=1, TaskID=47
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[41] = 0x0000000A00007A80; // Node ID 61
            // Fields: Type=1, TaskID=61
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[42] = 0x0008002A20002C00; // Node ID 22
            // Fields: Type=0, TaskID=22
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[43] = 0x0003802AE0001A00; // Node ID 13
            // Fields: Type=0, TaskID=13
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[44] = 0x0008002A00002800; // Node ID 20
            // Fields: Type=0, TaskID=20
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[45] = 0x0000000700007E80; // Node ID 63
            // Fields: Type=1, TaskID=63
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[46] = 0x0008002A60003000; // Node ID 24
            // Fields: Type=0, TaskID=24
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[47] = 0x0002002700002A00; // Node ID 21
            // Fields: Type=0, TaskID=21
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[48] = 0x0000000700008080; // Node ID 64
            // Fields: Type=1, TaskID=64
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[49] = 0x0008002100006480; // Node ID 50
            // Fields: Type=1, TaskID=50
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[50] = 0x0000001200008280; // Node ID 65
            // Fields: Type=1, TaskID=65
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[51] = 0x0002003300002E00; // Node ID 23
            // Fields: Type=0, TaskID=23
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[52] = 0x0008002100006680; // Node ID 51
            // Fields: Type=1, TaskID=51
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[53] = 0x0000001200008480; // Node ID 66
            // Fields: Type=1, TaskID=66
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        bingo_hw_scheduler_task_desc_list_chip_00[54] = 0x0008003300003200; // Node ID 25
            // Fields: Type=0, TaskID=25
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[55] = 0x0008003300003400; // Node ID 26
            // Fields: Type=0, TaskID=26
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[56] = 0x0008003300003600; // Node ID 27
            // Fields: Type=0, TaskID=27
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[57] = 0x0002003300003800; // Node ID 28
            // Fields: Type=0, TaskID=28
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[58] = 0x0004003200003C00; // Node ID 30
            // Fields: Type=0, TaskID=30
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[59] = 0x0002802680003E00; // Node ID 31
            // Fields: Type=0, TaskID=31
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[60] = 0x0004802A20004000; // Node ID 32
            // Fields: Type=0, TaskID=32
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[61] = 0x00030026A0004200; // Node ID 33
            // Fields: Type=0, TaskID=33
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[62] = 0x0005002A40004400; // Node ID 34
            // Fields: Type=0, TaskID=34
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[63] = 0x00038026C0004600; // Node ID 35
            // Fields: Type=0, TaskID=35
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[64] = 0x0005802A60004800; // Node ID 36
            // Fields: Type=0, TaskID=36
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[65] = 0x00080026E0004A00; // Node ID 37
            // Fields: Type=0, TaskID=37
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[66] = 0x0000000B00004C00; // Node ID 38
            // Fields: Type=0, TaskID=38
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        // Task ID Mapping Lists
        int32_t* global_task_id_to_dev_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 67 * sizeof(int32_t));
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
        global_task_id_to_dev_task_id_chip_00[16] = 16; // Node ID 16 -> Dev Task 16 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[17] = 17; // Node ID 17 -> Dev Task 17 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[18] = 18; // Node ID 18 -> Dev Task 18 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[19] = 19; // Node ID 19 -> Dev Task 19 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[20] = 20; // Node ID 20 -> Dev Task 20 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_full)
        global_task_id_to_dev_task_id_chip_00[21] = -1; // Node ID 21 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[22] = 21; // Node ID 22 -> Dev Task 21 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_full)
        global_task_id_to_dev_task_id_chip_00[23] = -1; // Node ID 23 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[24] = 22; // Node ID 24 -> Dev Task 22 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_dual_vc_gemm_full)
        global_task_id_to_dev_task_id_chip_00[25] = -1; // Node ID 25 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_dev_task_id_chip_00[26] = -1; // Node ID 26 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule)
        global_task_id_to_dev_task_id_chip_00[27] = -1; // Node ID 27 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_prepare_request)
        global_task_id_to_dev_task_id_chip_00[28] = -1; // Node ID 28 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_execute)
        global_task_id_to_dev_task_id_chip_00[29] = -1; // Node ID 29 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_dev_task_id_chip_00[30] = 23; // Node ID 30 -> Dev Task 23 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[31] = 24; // Node ID 31 -> Dev Task 24 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[32] = 25; // Node ID 32 -> Dev Task 25 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[33] = 26; // Node ID 33 -> Dev Task 26 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[34] = 27; // Node ID 34 -> Dev Task 27 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[35] = 28; // Node ID 35 -> Dev Task 28 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[36] = 29; // Node ID 36 -> Dev Task 29 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[37] = 30; // Node ID 37 -> Dev Task 30 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[38] = -1; // Node ID 38 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[39] = -1; // Node ID 39 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[40] = -1; // Node ID 40 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[41] = -1; // Node ID 41 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[42] = -1; // Node ID 42 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[43] = -1; // Node ID 43 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[44] = -1; // Node ID 44 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[45] = -1; // Node ID 45 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[46] = -1; // Node ID 46 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[47] = -1; // Node ID 47 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[48] = -1; // Node ID 48 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[49] = -1; // Node ID 49 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[50] = -1; // Node ID 50 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[51] = -1; // Node ID 51 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[52] = -1; // Node ID 52 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[53] = -1; // Node ID 53 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[54] = -1; // Node ID 54 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[55] = -1; // Node ID 55 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[56] = -1; // Node ID 56 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[57] = -1; // Node ID 57 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[58] = -1; // Node ID 58 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[59] = -1; // Node ID 59 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[60] = -1; // Node ID 60 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_dev_task_id_chip_00[61] = -1; // Node ID 61 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[62] = -1; // Node ID 62 (Node_ID0_Chiplet0_Cluster1_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[63] = -1; // Node ID 63 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[64] = -1; // Node ID 64 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_dev_task_id_chip_00[65] = -1; // Node ID 65 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_dev_task_id_chip_00[66] = -1; // Node ID 66 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        uint32_t num_dev_tasks_chip_00 = 31;
        int32_t* global_task_id_to_host_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 67 * sizeof(int32_t));
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
        global_task_id_to_host_task_id_chip_00[16] = -1; // Node ID 16 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[17] = -1; // Node ID 17 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[18] = -1; // Node ID 18 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy)
        global_task_id_to_host_task_id_chip_00[19] = -1; // Node ID 19 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[20] = -1; // Node ID 20 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_full)
        global_task_id_to_host_task_id_chip_00[21] = 0; // Node ID 21 -> Host Task 0 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[22] = -1; // Node ID 22 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_full)
        global_task_id_to_host_task_id_chip_00[23] = 1; // Node ID 23 -> Host Task 1 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[24] = -1; // Node ID 24 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_dual_vc_gemm_full)
        global_task_id_to_host_task_id_chip_00[25] = 2; // Node ID 25 -> Host Task 2 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma)
        global_task_id_to_host_task_id_chip_00[26] = 3; // Node ID 26 -> Host Task 3 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_router_schedule)
        global_task_id_to_host_task_id_chip_00[27] = 4; // Node ID 27 -> Host Task 4 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_prepare_request)
        global_task_id_to_host_task_id_chip_00[28] = 5; // Node ID 28 -> Host Task 5 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_execute)
        global_task_id_to_host_task_id_chip_00[29] = 6; // Node ID 29 -> Host Task 6 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_host_task_id_chip_00[30] = -1; // Node ID 30 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[31] = -1; // Node ID 31 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[32] = -1; // Node ID 32 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[33] = -1; // Node ID 33 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[34] = -1; // Node ID 34 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[35] = -1; // Node ID 35 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[36] = -1; // Node ID 36 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[37] = -1; // Node ID 37 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[38] = 7; // Node ID 38 -> Host Task 7 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[39] = -1; // Node ID 39 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[40] = -1; // Node ID 40 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[41] = -1; // Node ID 41 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[42] = -1; // Node ID 42 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[43] = -1; // Node ID 43 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[44] = -1; // Node ID 44 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[45] = -1; // Node ID 45 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[46] = -1; // Node ID 46 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[47] = -1; // Node ID 47 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[48] = -1; // Node ID 48 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[49] = -1; // Node ID 49 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[50] = -1; // Node ID 50 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[51] = -1; // Node ID 51 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[52] = -1; // Node ID 52 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[53] = -1; // Node ID 53 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[54] = -1; // Node ID 54 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[55] = -1; // Node ID 55 (Node_ID0_Chiplet0_Cluster2_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[56] = -1; // Node ID 56 (Node_ID0_Chiplet0_Cluster3_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[57] = -1; // Node ID 57 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[58] = -1; // Node ID 58 (Node_ID0_Chiplet0_Cluster0_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[59] = -1; // Node ID 59 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[60] = -1; // Node ID 60 (Node_ID0_Chiplet0_Cluster1_Core1_KernelNone)
        global_task_id_to_host_task_id_chip_00[61] = -1; // Node ID 61 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[62] = -1; // Node ID 62 (Node_ID0_Chiplet0_Cluster1_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[63] = -1; // Node ID 63 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[64] = -1; // Node ID 64 (Node_ID0_Chiplet0_Cluster0_Core2_KernelNone)
        global_task_id_to_host_task_id_chip_00[65] = -1; // Node ID 65 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        global_task_id_to_host_task_id_chip_00[66] = -1; // Node ID 66 (Node_ID0_Chiplet0_Cluster0_Core0_KernelNone)
        uint32_t num_host_tasks_chip_00 = 8;
        // 1. Memory Allocations
        uint64_t ptr_c0_l1_layout = bingo_l1_alloc(0x00, 0, 954988);
        uint64_t ptr_c1_l1_layout = bingo_l1_alloc(0x00, 1, 954988);
        uint64_t ptr_c2_indiv_active_state = bingo_l1_alloc(0x00, 2, 64);
        uint64_t ptr_c2_indiv_dyn_args = bingo_l1_alloc(0x00, 2, 6144);
        uint64_t ptr_c2_indiv_l1_a = bingo_l1_alloc(0x00, 2, 65536);
        uint64_t ptr_c2_indiv_l1_b_down = bingo_l1_alloc(0x00, 2, 262144);
        uint64_t ptr_c2_indiv_l1_b_gate = bingo_l1_alloc(0x00, 2, 262144);
        uint64_t ptr_c2_indiv_l1_b_up = bingo_l1_alloc(0x00, 2, 262144);
        uint64_t ptr_c2_indiv_l1_d = bingo_l1_alloc(0x00, 2, 32768);
        uint64_t ptr_c2_indiv_l1_d1_scratch = bingo_l1_alloc(0x00, 2, 16384);
        uint64_t ptr_c2_indiv_l1_down_d = bingo_l1_alloc(0x00, 2, 65536);
        uint64_t ptr_c2_indiv_static_args = bingo_l1_alloc(0x00, 2, 192);
        uint64_t ptr_c3_indiv_active_state = bingo_l1_alloc(0x00, 3, 64);
        uint64_t ptr_c3_indiv_dyn_args = bingo_l1_alloc(0x00, 3, 6144);
        uint64_t ptr_c3_indiv_l1_a = bingo_l1_alloc(0x00, 3, 65536);
        uint64_t ptr_c3_indiv_l1_b_down = bingo_l1_alloc(0x00, 3, 262144);
        uint64_t ptr_c3_indiv_l1_b_gate = bingo_l1_alloc(0x00, 3, 262144);
        uint64_t ptr_c3_indiv_l1_b_up = bingo_l1_alloc(0x00, 3, 262144);
        uint64_t ptr_c3_indiv_l1_d = bingo_l1_alloc(0x00, 3, 32768);
        uint64_t ptr_c3_indiv_l1_d1_scratch = bingo_l1_alloc(0x00, 3, 16384);
        uint64_t ptr_c3_indiv_l1_down_d = bingo_l1_alloc(0x00, 3, 65536);
        uint64_t ptr_c3_indiv_static_args = bingo_l1_alloc(0x00, 3, 192);
        uint64_t ptr_c3_router_l1_a = bingo_l1_alloc(0x00, 3, 66560);
        uint64_t ptr_c3_router_l1_b = bingo_l1_alloc(0x00, 3, 4096);
        uint64_t ptr_c3_router_l1_d = bingo_l1_alloc(0x00, 3, 512);
        uint64_t ptr_l3_c2_stage = bingo_l3_alloc(0x00, 6144);
        uint64_t ptr_l3_c3_stage = bingo_l3_alloc(0x00, 6144);
        uint64_t ptr_l3_cam_state = bingo_l3_alloc(0x00, 8);
        uint64_t ptr_l3_expert_counts = bingo_l3_alloc(0x00, 32);
        uint64_t ptr_l3_expert_token_ids = bingo_l3_alloc(0x00, 128);
        uint64_t ptr_l3_expert_token_kpos = bingo_l3_alloc(0x00, 128);
        uint64_t ptr_l3_expert_token_offsets = bingo_l3_alloc(0x00, 36);
        uint64_t ptr_l3_indiv_down_out = bingo_l3_alloc(0x00, 524288);
        #if !defined(MOE_ENABLE_HW_SCHEDULER) || defined(MOE_ENABLE_HW_SCHEDULER_CHECK)
        uint64_t ptr_l3_moe_request = bingo_l3_alloc(0x00, 256);
        #endif
        uint64_t ptr_l3_moe_runtime_state = bingo_l3_alloc(0x00, 64);
        #if !defined(MOE_ENABLE_HW_SCHEDULER) || defined(MOE_ENABLE_HW_SCHEDULER_CHECK)
        uint64_t ptr_l3_moe_schedule = bingo_l3_alloc(0x00, 32768);
        #endif
        uint64_t ptr_l3_router_out = bingo_l3_alloc(0x00, 512);
        uint64_t ptr_l3_shared_down_out = bingo_l3_alloc(0x00, 133120);
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
        bingo_kernel_scratchpad_t* sp_dev_16 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_17 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_18 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_19 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_20 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_21 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_22 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_23 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_24 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_25 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_26 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_27 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_28 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_29 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_30 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_31 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_32 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_33 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_34 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_35 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_36 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_37 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_38 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);

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
        args_dev_chip00_11->dst_addr_lo = (uint32_t)ptr_c0_l1_layout + 788480;
        args_dev_chip00_11->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout + 788480 >> 32);
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
        args_dev_chip00_12->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 788480;
        args_dev_chip00_12->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 788480 >> 32);
        args_dev_chip00_12->size = 66560;
        args_dev_chip00_12->gating_sp_addr = 0;
        args_dev_chip00_12->cond_node_index = 0;
        args_dev_chip00_12->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_12;
        device_arg_list_chip_00[12] = (uint32_t)(uintptr_t)args_dev_chip00_12;
        device_kernel_list_chip_00[12] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 13 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_13 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_13->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_13->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)input_A))) >> 32);
        args_dev_chip00_13->dst_addr_lo = (uint32_t)ptr_c3_router_l1_a;
        args_dev_chip00_13->dst_addr_hi = (uint32_t)(ptr_c3_router_l1_a >> 32);
        args_dev_chip00_13->size = 66560;
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
        args_dev_chip00_14->size = 131072;
        args_dev_chip00_14->gating_sp_addr = 0;
        args_dev_chip00_14->cond_node_index = 0;
        args_dev_chip00_14->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_14;
        device_arg_list_chip_00[14] = (uint32_t)(uintptr_t)args_dev_chip00_14;
        device_kernel_list_chip_00[14] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 15 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_15 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_15->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right)));
        args_dev_chip00_15->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right))) >> 32);
        args_dev_chip00_15->dst_addr_lo = (uint32_t)ptr_c0_l1_layout + 657408;
        args_dev_chip00_15->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout + 657408 >> 32);
        args_dev_chip00_15->size = 131072;
        args_dev_chip00_15->gating_sp_addr = 0;
        args_dev_chip00_15->cond_node_index = 0;
        args_dev_chip00_15->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_15;
        device_arg_list_chip_00[15] = (uint32_t)(uintptr_t)args_dev_chip00_15;
        device_kernel_list_chip_00[15] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 16 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_16 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_16->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg)));
        args_dev_chip00_16->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg))) >> 32);
        args_dev_chip00_16->dst_addr_lo = (uint32_t)ptr_c0_l1_layout + 954624;
        args_dev_chip00_16->dst_addr_hi = (uint32_t)(ptr_c0_l1_layout + 954624 >> 32);
        args_dev_chip00_16->size = 364;
        args_dev_chip00_16->gating_sp_addr = 0;
        args_dev_chip00_16->cond_node_index = 0;
        args_dev_chip00_16->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_16;
        device_arg_list_chip_00[16] = (uint32_t)(uintptr_t)args_dev_chip00_16;
        device_kernel_list_chip_00[16] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 17 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_17 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_17->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_left)));
        args_dev_chip00_17->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_left))) >> 32);
        args_dev_chip00_17->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 525440;
        args_dev_chip00_17->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 525440 >> 32);
        args_dev_chip00_17->size = 131072;
        args_dev_chip00_17->gating_sp_addr = 0;
        args_dev_chip00_17->cond_node_index = 0;
        args_dev_chip00_17->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_17;
        device_arg_list_chip_00[17] = (uint32_t)(uintptr_t)args_dev_chip00_17;
        device_kernel_list_chip_00[17] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 18 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_xdma_1d_copy (__snax_bingo_kernel_xdma_1d_copy)
        __snax_bingo_kernel_xdma_1d_copy_args_t* args_dev_chip00_18 = (__snax_bingo_kernel_xdma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_xdma_1d_copy_args_t));
        args_dev_chip00_18->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right)));
        args_dev_chip00_18->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)layout_W2_right))) >> 32);
        args_dev_chip00_18->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 657408;
        args_dev_chip00_18->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 657408 >> 32);
        args_dev_chip00_18->size = 131072;
        args_dev_chip00_18->gating_sp_addr = 0;
        args_dev_chip00_18->cond_node_index = 0;
        args_dev_chip00_18->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_18;
        device_arg_list_chip_00[18] = (uint32_t)(uintptr_t)args_dev_chip00_18;
        device_kernel_list_chip_00[18] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_xdma_1d_copy");
        // Node ID: 19 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_19 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_19->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg)));
        args_dev_chip00_19->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)l15_dev_shared_s0_cfg))) >> 32);
        args_dev_chip00_19->dst_addr_lo = (uint32_t)ptr_c1_l1_layout + 954624;
        args_dev_chip00_19->dst_addr_hi = (uint32_t)(ptr_c1_l1_layout + 954624 >> 32);
        args_dev_chip00_19->size = 364;
        args_dev_chip00_19->gating_sp_addr = 0;
        args_dev_chip00_19->cond_node_index = 0;
        args_dev_chip00_19->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_19;
        device_arg_list_chip_00[19] = (uint32_t)(uintptr_t)args_dev_chip00_19;
        device_kernel_list_chip_00[19] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 20 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_full (__snax_bingo_kernel_dual_vc_l15_moe_full)
        __snax_bingo_kernel_dual_vc_l15_moe_full_args_t* args_dev_chip00_20 = (__snax_bingo_kernel_dual_vc_l15_moe_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_dual_vc_l15_moe_full_args_t));
        args_dev_chip00_20->shape_cfg_addr = (uint32_t)ptr_c0_l1_layout + 954624;
        args_dev_chip00_20->tcdm_base = (uint32_t)ptr_c0_l1_layout;
        args_dev_chip00_20->rescale_mult = 1;
        args_dev_chip00_20->rescale_shift = 0;
        args_dev_chip00_20->gating_sp_addr = 0;
        args_dev_chip00_20->cond_node_index = 0;
        args_dev_chip00_20->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_20;
        device_arg_list_chip_00[20] = (uint32_t)(uintptr_t)args_dev_chip00_20;
        device_kernel_list_chip_00[20] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_l15_moe_full");
        // Node ID: 21 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_21 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_21->src_addr = (uint64_t)ptr_c0_l1_layout + 888064;
        args_host_chip00_21->dst_addr = (uint64_t)ptr_l3_shared_down_out;
        args_host_chip00_21->size = 66560;
        args_host_chip00_21->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_21;
        host_arg_list_chip_00[0] = (uint64_t)(uintptr_t)args_host_chip00_21;
        host_kernel_list_chip_00[0] = (uint64_t)(uintptr_t)&__host_bingo_kernel_idma;
        // Node ID: 22 Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_dual_vc_l15_moe_full (__snax_bingo_kernel_dual_vc_l15_moe_full)
        __snax_bingo_kernel_dual_vc_l15_moe_full_args_t* args_dev_chip00_22 = (__snax_bingo_kernel_dual_vc_l15_moe_full_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_dual_vc_l15_moe_full_args_t));
        args_dev_chip00_22->shape_cfg_addr = (uint32_t)ptr_c1_l1_layout + 954624;
        args_dev_chip00_22->tcdm_base = (uint32_t)ptr_c1_l1_layout;
        args_dev_chip00_22->rescale_mult = 1;
        args_dev_chip00_22->rescale_shift = 0;
        args_dev_chip00_22->gating_sp_addr = 0;
        args_dev_chip00_22->cond_node_index = 0;
        args_dev_chip00_22->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_22;
        device_arg_list_chip_00[21] = (uint32_t)(uintptr_t)args_dev_chip00_22;
        device_kernel_list_chip_00[21] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_l15_moe_full");
        // Node ID: 23 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_idma (__host_bingo_kernel_idma)
        __host_bingo_kernel_idma_args_t* args_host_chip00_23 = (__host_bingo_kernel_idma_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_idma_args_t));
        args_host_chip00_23->src_addr = (uint64_t)ptr_c1_l1_layout + 888064;
        args_host_chip00_23->dst_addr = (uint64_t)ptr_l3_shared_down_out + 66560;
        args_host_chip00_23->size = 66560;
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
        args_host_chip00_26->expert_token_counts_out_addr = (uint64_t)ptr_l3_expert_counts;
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
        #if !defined(MOE_ENABLE_HW_SCHEDULER) || defined(MOE_ENABLE_HW_SCHEDULER_CHECK)
        args_host_chip00_27->request_out_addr = (uint64_t)ptr_l3_moe_request;
        args_host_chip00_27->schedule_out_addr = (uint64_t)ptr_l3_moe_schedule;
        #endif
        args_host_chip00_27->expert_token_offsets_addr = (uint64_t)ptr_l3_expert_token_offsets;
        args_host_chip00_27->expert_token_ids_addr = (uint64_t)ptr_l3_expert_token_ids;
        args_host_chip00_27->expert_token_kpos_addr = (uint64_t)ptr_l3_expert_token_kpos;
        args_host_chip00_27->n_experts = 8;
        args_host_chip00_27->topk_indices_l3 = (uint64_t)ptr_l3_topk_idx;
        args_host_chip00_27->M_total = 32;
        args_host_chip00_27->top_k = 2;
        args_host_chip00_27->expert_token_counts_valid = 1;
        args_host_chip00_27->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_host_chip00_27->c2_stage_base = (uint64_t)ptr_l3_c2_stage;
        args_host_chip00_27->c3_stage_base = (uint64_t)ptr_l3_c3_stage;
        args_host_chip00_27->dynamic_arg_slot_bytes = 192;
        args_host_chip00_27->dynamic_num_slots = 32;
        args_host_chip00_27->c2_l1_a = (uint64_t)ptr_c2_indiv_l1_a;
        args_host_chip00_27->c2_l1_d = (uint64_t)ptr_c2_indiv_l1_d;
        args_host_chip00_27->c2_l1_down_d = (uint64_t)ptr_c2_indiv_l1_down_d;
        args_host_chip00_27->c3_l1_a = (uint64_t)ptr_c3_indiv_l1_a;
        args_host_chip00_27->c3_l1_d = (uint64_t)ptr_c3_indiv_l1_d;
        args_host_chip00_27->c3_l1_down_d = (uint64_t)ptr_c3_indiv_l1_down_d;
        args_host_chip00_27->A_token_bytes = 2048;
        args_host_chip00_27->indiv_D_tile_bytes = 16384;
        args_host_chip00_27->indiv_down_D_tile_bytes = 16384;
        args_host_chip00_27->indiv_N_per_block = 256;
        args_host_chip00_27->indiv_down_N_per_block = 256;
        args_host_chip00_27->s1_block_count = 2;
        args_host_chip00_27->s3_block_count = 2;
        args_host_chip00_27->max_tokens_per_expert = 32;
        args_host_chip00_27->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_27;
        host_arg_list_chip_00[4] = (uint64_t)(uintptr_t)args_host_chip00_27;
        host_kernel_list_chip_00[4] = (uint64_t)(uintptr_t)&__host_bingo_kernel_moe_prepare_request;
        // Node ID: 28 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_moe_execute (__host_bingo_kernel_moe_execute)
        __host_bingo_kernel_moe_execute_args_t* args_host_chip00_28 = (__host_bingo_kernel_moe_execute_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_moe_execute_args_t));
        #if !defined(MOE_ENABLE_HW_SCHEDULER) || defined(MOE_ENABLE_HW_SCHEDULER_CHECK)
        args_host_chip00_28->request_addr = (uint64_t)ptr_l3_moe_request;
        args_host_chip00_28->schedule_addr = (uint64_t)ptr_l3_moe_schedule;
        #endif
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
        args_host_chip00_28->c2_l1_d1_scratch = (uint64_t)ptr_c2_indiv_l1_d1_scratch;
        args_host_chip00_28->c3_l1_b_gate = (uint64_t)ptr_c3_indiv_l1_b_gate;
        args_host_chip00_28->c3_l1_b_up = (uint64_t)ptr_c3_indiv_l1_b_up;
        args_host_chip00_28->c3_l1_b_down = (uint64_t)ptr_c3_indiv_l1_b_down;
        args_host_chip00_28->c3_l1_a = (uint64_t)ptr_c3_indiv_l1_a;
        args_host_chip00_28->c3_l1_d = (uint64_t)ptr_c3_indiv_l1_d;
        args_host_chip00_28->c3_l1_down_d = (uint64_t)ptr_c3_indiv_l1_down_d;
        args_host_chip00_28->c3_l1_d1_scratch = (uint64_t)ptr_c3_indiv_l1_d1_scratch;
        args_host_chip00_28->output_l3_addr = (uint64_t)ptr_l3_indiv_down_out;
        args_host_chip00_28->runtime_state_addr = (uint64_t)ptr_l3_moe_runtime_state;
        args_host_chip00_28->c2_active_state_l1_addr = (uint64_t)ptr_c2_indiv_active_state;
        args_host_chip00_28->c3_active_state_l1_addr = (uint64_t)ptr_c3_indiv_active_state;
        args_host_chip00_28->A_token_bytes = 2048;
        args_host_chip00_28->indiv_B_expert_stride = 262144;
        args_host_chip00_28->indiv_down_B_expert_stride = 262144;
        args_host_chip00_28->down_D_bytes_per_expert = 65536;
        args_host_chip00_28->M_total = 32;
        args_host_chip00_28->top_k = 2;
        args_host_chip00_28->indiv_B_tile_bytes = 131072;
        args_host_chip00_28->indiv_D_tile_bytes = 16384;
        args_host_chip00_28->indiv_down_B_tile_bytes = 65536;
        args_host_chip00_28->indiv_down_D_tile_bytes = 16384;
        args_host_chip00_28->indiv_N2 = 2;
        args_host_chip00_28->indiv_down_N2 = 2;
        args_host_chip00_28->s1_block_count = 2;
        args_host_chip00_28->s3_block_count = 2;
        args_host_chip00_28->indiv_K1 = 128;
        args_host_chip00_28->indiv_N_per_block = 256;
        args_host_chip00_28->indiv_down_K1 = 64;
        args_host_chip00_28->indiv_down_N_per_block = 256;
        args_host_chip00_28->rescale_mult = 1;
        args_host_chip00_28->rescale_shift = 0;
        args_host_chip00_28->output_expert_stride_bytes = 65536;
        args_host_chip00_28->max_tokens_per_expert = 32;
        args_host_chip00_28->c2_static_args_base = (uint64_t)ptr_c2_indiv_static_args;
        args_host_chip00_28->c3_static_args_base = (uint64_t)ptr_c3_indiv_static_args;
        args_host_chip00_28->c2_dynamic_args_base = (uint64_t)ptr_c2_indiv_dyn_args;
        args_host_chip00_28->c3_dynamic_args_base = (uint64_t)ptr_c3_indiv_dyn_args;
        args_host_chip00_28->dynamic_arg_slot_bytes = 192;
        args_host_chip00_28->dynamic_num_slots = 32;
        args_host_chip00_28->c2_stage_base = (uint64_t)ptr_l3_c2_stage;
        args_host_chip00_28->c3_stage_base = (uint64_t)ptr_l3_c3_stage;
        args_host_chip00_28->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_28;
        host_arg_list_chip_00[5] = (uint64_t)(uintptr_t)args_host_chip00_28;
        host_kernel_list_chip_00[5] = (uint64_t)(uintptr_t)&__host_bingo_kernel_moe_execute;
        // One-time initialization of C2/C3 dynamic slot templates.
        if (__host_moe_init_stage_templates(args_host_chip00_28) != BINGO_RET_SUCC) return BINGO_RET_FAIL;
        // Node ID: 29 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry (__host_bingo_kernel_entry)
        host_arg_list_chip_00[6] = 0;
        host_kernel_list_chip_00[6] = (uint64_t)(uintptr_t)&__host_bingo_kernel_entry;
        // Node ID: 30 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_30 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_30->exit_code = 0;
        args_dev_chip00_30->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_30;
        device_arg_list_chip_00[23] = (uint32_t)(uintptr_t)args_dev_chip00_30;
        device_kernel_list_chip_00[23] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 31 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_31 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_31->exit_code = 0;
        args_dev_chip00_31->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_31;
        device_arg_list_chip_00[24] = (uint32_t)(uintptr_t)args_dev_chip00_31;
        device_kernel_list_chip_00[24] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 32 Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_32 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_32->exit_code = 0;
        args_dev_chip00_32->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_32;
        device_arg_list_chip_00[25] = (uint32_t)(uintptr_t)args_dev_chip00_32;
        device_kernel_list_chip_00[25] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 33 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_33 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_33->exit_code = 0;
        args_dev_chip00_33->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_33;
        device_arg_list_chip_00[26] = (uint32_t)(uintptr_t)args_dev_chip00_33;
        device_kernel_list_chip_00[26] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 34 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_34 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_34->exit_code = 0;
        args_dev_chip00_34->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_34;
        device_arg_list_chip_00[27] = (uint32_t)(uintptr_t)args_dev_chip00_34;
        device_kernel_list_chip_00[27] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 35 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_35 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_35->exit_code = 0;
        args_dev_chip00_35->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_35;
        device_arg_list_chip_00[28] = (uint32_t)(uintptr_t)args_dev_chip00_35;
        device_kernel_list_chip_00[28] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 36 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_36 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_36->exit_code = 0;
        args_dev_chip00_36->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_36;
        device_arg_list_chip_00[29] = (uint32_t)(uintptr_t)args_dev_chip00_36;
        device_kernel_list_chip_00[29] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 37 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_37 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_37->exit_code = 0;
        args_dev_chip00_37->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_37;
        device_arg_list_chip_00[30] = (uint32_t)(uintptr_t)args_dev_chip00_37;
        device_kernel_list_chip_00[30] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 38 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit (__host_bingo_kernel_exit)
        __host_bingo_kernel_exit_args_t* args_host_chip00_38 = (__host_bingo_kernel_exit_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_exit_args_t));
        args_host_chip00_38->exit_code = 0;
        args_host_chip00_38->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_38;
        host_arg_list_chip_00[7] = (uint64_t)(uintptr_t)args_host_chip00_38;
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
