// Auto-generated offload_hw_bingo.h
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "single_cluster_MoE_data.h"

// Kernel Name List
// Note: This list is currently for debugging purposes only and is not used in the runtime.
// It will be enabled in the future.
/*
char kernel_name_list[13][64] = {
    "__snax_bingo_kernel_idma_1d_copy", // Node ID 0
    "__snax_bingo_kernel_dual_dma", // Node ID 1
    "__snax_bingo_kernel_dual_vc_swiglu_full", // Node ID 2
    "__host_bingo_kernel_entry", // Node ID 3
    "__snax_bingo_kernel_exit", // Node ID 4
    "__snax_bingo_kernel_exit", // Node ID 5
    "__snax_bingo_kernel_exit", // Node ID 6
    "__snax_bingo_kernel_exit", // Node ID 7
    "__snax_bingo_kernel_exit", // Node ID 8
    "__snax_bingo_kernel_exit", // Node ID 9
    "__snax_bingo_kernel_exit", // Node ID 10
    "__snax_bingo_kernel_exit", // Node ID 11
    "__host_bingo_kernel_exit", // Node ID 12
};
*/

int kernel_execution(){
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing single_cluster_MoE Workload\r\n", get_current_chip_loc_x(), get_current_chip_loc_y());
    uint32_t current_chip_id = get_current_chip_id();
    if (current_chip_id == 0x00) {
        uint32_t num_total_tasks = 13;
        // Task Description List
        uint32_t bingo_hw_scheduler_num_task_desc_chip_00 = 13;
        uint64_t* bingo_hw_scheduler_task_desc_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, bingo_hw_scheduler_num_task_desc_chip_00 * sizeof(uint64_t));
        bingo_hw_scheduler_task_desc_list_chip_00[0] = 0x0004002100000600; // Node ID 3
            // Fields: Type=0, TaskID=3
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=0, Code=0b000
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[1] = 0x0004003280000000; // Node ID 0
            // Fields: Type=0, TaskID=0
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b100
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[2] = 0x0002002A80000200; // Node ID 1
            // Fields: Type=0, TaskID=1
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[3] = 0x0002002A00000400; // Node ID 2
            // Fields: Type=0, TaskID=2
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[4] = 0x0004002600000800; // Node ID 4
            // Fields: Type=0, TaskID=4
            //         Assigned: Chiplet=00, Cluster=0, Core=0
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[5] = 0x0002802680000A00; // Node ID 5
            // Fields: Type=0, TaskID=5
            //         Assigned: Chiplet=00, Cluster=0, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[6] = 0x0004802A20000C00; // Node ID 6
            // Fields: Type=0, TaskID=6
            //         Assigned: Chiplet=00, Cluster=1, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=1, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[7] = 0x00030026A0000E00; // Node ID 7
            // Fields: Type=0, TaskID=7
            //         Assigned: Chiplet=00, Cluster=1, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[8] = 0x0005002A40001000; // Node ID 8
            // Fields: Type=0, TaskID=8
            //         Assigned: Chiplet=00, Cluster=2, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=2, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[9] = 0x00038026C0001200; // Node ID 9
            // Fields: Type=0, TaskID=9
            //         Assigned: Chiplet=00, Cluster=2, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b001
        bingo_hw_scheduler_task_desc_list_chip_00[10] = 0x0005802A60001400; // Node ID 10
            // Fields: Type=0, TaskID=10
            //         Assigned: Chiplet=00, Cluster=3, Core=0
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=3, Code=0b010
        bingo_hw_scheduler_task_desc_list_chip_00[11] = 0x00080026E0001600; // Node ID 11
            // Fields: Type=0, TaskID=11
            //         Assigned: Chiplet=00, Cluster=3, Core=1
            //         DepCheck: En=1, Code=0b001
            //         DepSet:   En=1, All=0, Chiplet=00, Cluster=0, Code=0b100
        bingo_hw_scheduler_task_desc_list_chip_00[12] = 0x0000000B00001800; // Node ID 12
            // Fields: Type=0, TaskID=12
            //         Assigned: Chiplet=00, Cluster=0, Core=2
            //         DepCheck: En=1, Code=0b010
            //         DepSet:   En=0, All=0, Chiplet=00, Cluster=0, Code=0b000
        // Task ID Mapping Lists
        int32_t* global_task_id_to_dev_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 13 * sizeof(int32_t));
        global_task_id_to_dev_task_id_chip_00[0] = 0; // Node ID 0 -> Dev Task 0 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_dev_task_id_chip_00[1] = 1; // Node ID 1 -> Dev Task 1 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_dual_dma)
        global_task_id_to_dev_task_id_chip_00[2] = 2; // Node ID 2 -> Dev Task 2 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_swiglu_full)
        global_task_id_to_dev_task_id_chip_00[3] = -1; // Node ID 3 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_dev_task_id_chip_00[4] = 3; // Node ID 4 -> Dev Task 3 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[5] = 4; // Node ID 5 -> Dev Task 4 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[6] = 5; // Node ID 6 -> Dev Task 5 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[7] = 6; // Node ID 7 -> Dev Task 6 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[8] = 7; // Node ID 8 -> Dev Task 7 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[9] = 8; // Node ID 9 -> Dev Task 8 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[10] = 9; // Node ID 10 -> Dev Task 9 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[11] = 10; // Node ID 11 -> Dev Task 10 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_dev_task_id_chip_00[12] = -1; // Node ID 12 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        uint32_t num_dev_tasks_chip_00 = 11;
        int32_t* global_task_id_to_host_task_id_chip_00 = (int32_t*)bingo_l3_alloc(0x00, 13 * sizeof(int32_t));
        global_task_id_to_host_task_id_chip_00[0] = -1; // Node ID 0 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy)
        global_task_id_to_host_task_id_chip_00[1] = -1; // Node ID 1 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_dual_dma)
        global_task_id_to_host_task_id_chip_00[2] = -1; // Node ID 2 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_swiglu_full)
        global_task_id_to_host_task_id_chip_00[3] = 0; // Node ID 3 -> Host Task 0 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry)
        global_task_id_to_host_task_id_chip_00[4] = -1; // Node ID 4 (Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[5] = -1; // Node ID 5 (Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[6] = -1; // Node ID 6 (Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[7] = -1; // Node ID 7 (Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[8] = -1; // Node ID 8 (Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[9] = -1; // Node ID 9 (Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[10] = -1; // Node ID 10 (Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[11] = -1; // Node ID 11 (Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit)
        global_task_id_to_host_task_id_chip_00[12] = 1; // Node ID 12 -> Host Task 1 (Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit)
        uint32_t num_host_tasks_chip_00 = 2;
        // 1. Memory Allocations
        uint64_t ptr_l1_buf_A = bingo_l1_alloc(0x00, 0, 16384);
        uint64_t ptr_l1_d1_scratch = bingo_l1_alloc(0x00, 0, 16384);
        uint64_t ptr_l1_gate_b_n2 = bingo_l1_alloc(0x00, 0, 524288);
        uint64_t ptr_l1_swiglu_d_n2 = bingo_l1_alloc(0x00, 0, 16384);
        uint64_t ptr_l1_up_b_n2 = bingo_l1_alloc(0x00, 0, 524288);

        // 2. Prepare device/host arg/kernel lists
        uint32_t* device_arg_list_chip_00 = (uint32_t*)bingo_l3_alloc(0x00, num_dev_tasks_chip_00 * sizeof(uint32_t));
        uint32_t* device_kernel_list_chip_00 = (uint32_t*)bingo_l3_alloc(0x00, num_dev_tasks_chip_00 * sizeof(uint32_t));
        uint64_t* host_arg_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, num_host_tasks_chip_00 * sizeof(uint64_t));
        uint64_t* host_kernel_list_chip_00 = (uint64_t*)bingo_l3_alloc(0x00, num_host_tasks_chip_00 * sizeof(uint64_t));

        // 3. Task Arguments Init
        // 3a. Pre-allocate scratchpads for all tasks
        bingo_kernel_scratchpad_t* sp_dev_0 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_1 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_2 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_3 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_4 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_5 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 0, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_6 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_7 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 1, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_8 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_9 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 2, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_10 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_dev_11 = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x00, 3, BINGO_KERNEL_SCRATCHPAD_SIZE);
        bingo_kernel_scratchpad_t* sp_host_12 = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x00, BINGO_KERNEL_SCRATCHPAD_SIZE);

        // Node ID: 0 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_idma_1d_copy (__snax_bingo_kernel_idma_1d_copy)
        __snax_bingo_kernel_idma_1d_copy_args_t* args_dev_chip00_0 = (__snax_bingo_kernel_idma_1d_copy_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_idma_1d_copy_args_t));
        args_dev_chip00_0->src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)input_A)));
        args_dev_chip00_0->src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)input_A))) >> 32);
        args_dev_chip00_0->dst_addr_lo = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_0->dst_addr_hi = (uint32_t)(ptr_l1_buf_A >> 32);
        args_dev_chip00_0->size = 16384;
        args_dev_chip00_0->gating_sp_addr = 0;
        args_dev_chip00_0->cond_node_index = 0;
        args_dev_chip00_0->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_0;
        device_arg_list_chip_00[0] = (uint32_t)(uintptr_t)args_dev_chip00_0;
        device_kernel_list_chip_00[0] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_idma_1d_copy");
        // Node ID: 1 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_dual_dma (__snax_bingo_kernel_dual_dma)
        __snax_bingo_kernel_dual_dma_args_t* args_dev_chip00_1 = (__snax_bingo_kernel_dual_dma_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_dual_dma_args_t));
        args_dev_chip00_1->idma_src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B)));
        args_dev_chip00_1->idma_src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_gate_B))) >> 32);
        args_dev_chip00_1->idma_dst_addr_lo = (uint32_t)ptr_l1_gate_b_n2;
        args_dev_chip00_1->idma_dst_addr_hi = (uint32_t)(ptr_l1_gate_b_n2 >> 32);
        args_dev_chip00_1->idma_size = 524288;
        args_dev_chip00_1->xdma_src_addr_lo = (uint32_t)(chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B)));
        args_dev_chip00_1->xdma_src_addr_hi = (uint32_t)((chiplet_addr_transform((uint64_t)((uintptr_t)shared_experts_up_projection_B))) >> 32);
        args_dev_chip00_1->xdma_dst_addr_lo = (uint32_t)ptr_l1_up_b_n2;
        args_dev_chip00_1->xdma_dst_addr_hi = (uint32_t)(ptr_l1_up_b_n2 >> 32);
        args_dev_chip00_1->xdma_size = 524288;
        args_dev_chip00_1->gating_sp_addr = 0;
        args_dev_chip00_1->cond_node_index = 0;
        args_dev_chip00_1->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_1;
        device_arg_list_chip_00[1] = (uint32_t)(uintptr_t)args_dev_chip00_1;
        device_kernel_list_chip_00[1] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_dma");
        // Node ID: 2 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_dual_vc_swiglu_full (__snax_bingo_kernel_dual_vc_swiglu_full)
        __snax_bingo_kernel_dual_vc_swiglu_full_args_t* args_dev_chip00_2 = (__snax_bingo_kernel_dual_vc_swiglu_full_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_dual_vc_swiglu_full_args_t));
        args_dev_chip00_2->input_A_addr = (uint32_t)ptr_l1_buf_A;
        args_dev_chip00_2->input_B_gate_addr = (uint32_t)ptr_l1_gate_b_n2;
        args_dev_chip00_2->input_B_up_addr = (uint32_t)ptr_l1_up_b_n2;
        args_dev_chip00_2->output_D0_addr = (uint32_t)ptr_l1_swiglu_d_n2;
        args_dev_chip00_2->output_D1_addr = (uint32_t)ptr_l1_d1_scratch;
        args_dev_chip00_2->M = 1;
        args_dev_chip00_2->K = 128;
        args_dev_chip00_2->N = 256;
        args_dev_chip00_2->array_shape = 0;
        args_dev_chip00_2->rescale_mult = 1;
        args_dev_chip00_2->rescale_shift = 0;
        args_dev_chip00_2->gating_sp_addr = 0;
        args_dev_chip00_2->cond_node_index = 0;
        args_dev_chip00_2->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_2;
        device_arg_list_chip_00[2] = (uint32_t)(uintptr_t)args_dev_chip00_2;
        device_kernel_list_chip_00[2] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_dual_vc_swiglu_full");
        // Node ID: 3 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_entry (__host_bingo_kernel_entry)
        host_arg_list_chip_00[0] = 0;
        host_kernel_list_chip_00[0] = (uint64_t)(uintptr_t)&__host_bingo_kernel_entry;
        // Node ID: 4 Node_ID0_Chiplet0_Cluster0_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_4 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_4->exit_code = 0;
        args_dev_chip00_4->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_4;
        device_arg_list_chip_00[3] = (uint32_t)(uintptr_t)args_dev_chip00_4;
        device_kernel_list_chip_00[3] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 5 Node_ID0_Chiplet0_Cluster0_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_5 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 0, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_5->exit_code = 0;
        args_dev_chip00_5->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_5;
        device_arg_list_chip_00[4] = (uint32_t)(uintptr_t)args_dev_chip00_5;
        device_kernel_list_chip_00[4] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 6 Node_ID0_Chiplet0_Cluster1_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_6 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_6->exit_code = 0;
        args_dev_chip00_6->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_6;
        device_arg_list_chip_00[5] = (uint32_t)(uintptr_t)args_dev_chip00_6;
        device_kernel_list_chip_00[5] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 7 Node_ID0_Chiplet0_Cluster1_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_7 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 1, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_7->exit_code = 0;
        args_dev_chip00_7->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_7;
        device_arg_list_chip_00[6] = (uint32_t)(uintptr_t)args_dev_chip00_7;
        device_kernel_list_chip_00[6] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 8 Node_ID0_Chiplet0_Cluster2_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_8 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_8->exit_code = 0;
        args_dev_chip00_8->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_8;
        device_arg_list_chip_00[7] = (uint32_t)(uintptr_t)args_dev_chip00_8;
        device_kernel_list_chip_00[7] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 9 Node_ID0_Chiplet0_Cluster2_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_9 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 2, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_9->exit_code = 0;
        args_dev_chip00_9->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_9;
        device_arg_list_chip_00[8] = (uint32_t)(uintptr_t)args_dev_chip00_9;
        device_kernel_list_chip_00[8] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 10 Node_ID0_Chiplet0_Cluster3_Core0_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_10 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_10->exit_code = 0;
        args_dev_chip00_10->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_10;
        device_arg_list_chip_00[9] = (uint32_t)(uintptr_t)args_dev_chip00_10;
        device_kernel_list_chip_00[9] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 11 Node_ID0_Chiplet0_Cluster3_Core1_Kernel__snax_bingo_kernel_exit (__snax_bingo_kernel_exit)
        __snax_bingo_kernel_exit_args_t* args_dev_chip00_11 = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x00, 3, sizeof(__snax_bingo_kernel_exit_args_t));
        args_dev_chip00_11->exit_code = 0;
        args_dev_chip00_11->scratchpad_ptr = (uint32_t)(uintptr_t)sp_dev_11;
        device_arg_list_chip_00[10] = (uint32_t)(uintptr_t)args_dev_chip00_11;
        device_kernel_list_chip_00[10] = (uint32_t)(uintptr_t)get_device_function("__snax_bingo_kernel_exit");
        // Node ID: 12 Node_ID0_Chiplet0_Cluster0_Core2_Kernel__host_bingo_kernel_exit (__host_bingo_kernel_exit)
        __host_bingo_kernel_exit_args_t* args_host_chip00_12 = (__host_bingo_kernel_exit_args_t*)bingo_l3_alloc(0x00, sizeof(__host_bingo_kernel_exit_args_t));
        args_host_chip00_12->exit_code = 0;
        args_host_chip00_12->scratchpad_ptr = (uint64_t)(uintptr_t)sp_host_12;
        host_arg_list_chip_00[1] = (uint64_t)(uintptr_t)args_host_chip00_12;
        host_kernel_list_chip_00[1] = (uint64_t)(uintptr_t)&__host_bingo_kernel_exit;

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
