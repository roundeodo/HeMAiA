#include "libbingo/bingo_api.h"
#include "host.h"
#include "../data/data.h"

// Global Variables for communication buffer
volatile comm_buffer_t *comm_buffer_ptr = (comm_buffer_t *)0;
O1HeapInstance *l2_heap_manager = NULL;
O1HeapInstance *l3_heap_manager = NULL;
volatile O1HeapInstance *dram_heap_manager = NULL;

// uint64_t DRAM_DATA = 0xAAAABBBBCCCCDDDD;
// uint64_t *source_DATA_ptr = (uint64_t *)&DRAM_DATA;
int main()
{
    uint32_t current_chip_id = get_current_chip_id();
    // uart initialization
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    init_uart(address_prefix, 32, 1);
    enable_sw_interrupts();
    // Init bingo runtime
    if (bingo_hemaia_system_mmap_init() < 0)
    {
        printf("Chip(%x, %x): [Host] Error when initializing Allocator\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y());
        return -1;
    }
    else
    {
        printf("[Host] Allocator Init Success\r\n");
    }

    comm_buffer_ptr = (comm_buffer_t *)bingo_get_l2_comm_buffer(current_chip_id);
    l2_heap_manager = (O1HeapInstance *)bingo_get_l2_heap_manager(current_chip_id);
    l3_heap_manager = (O1HeapInstance *)bingo_get_l3_heap_manager(current_chip_id);

    printf("SPM_NARROW_addr %lx\r\n", SPM_NARROW_BASE_ADDR);
    printf("SPM_WIDE_addr %lx\r\n", SPM_WIDE_BASE_ADDR);
    printf("wide_zero_addr %lx\r\n", WIDE_ZERO_MEM_BASE_ADDR);

    printf("l2_manager_addr is %lx \r\n", (uint64_t)l2_heap_manager);
    printf("l3_manager_addr is %lx \r\n", (uint64_t)l3_heap_manager);
    printf("comm_buffer_ptr is %lx \r\n", (uint64_t)comm_buffer_ptr);
    printf("the address of data is %lx \r\n", (uint64_t)data);
    printf("DRAM addr in chiplet style: %lx\r\n", (const uint64_t)chiplet_addr_transform_loc(0, 0, SPM_WIDE_BASE_ADDR));
    // init external DRAM heap
    // dram_heap_manager = (volatile O1HeapInstance *)o1heapInit((const uint64_t)chiplet_addr_transform_loc(0, 0, SPM_WIDE_BASE_ADDR), 64 * 1024 * 1024); // 64 MB DRAM heap

    //     if (dram_heap_manager == NULL)
    // {
    //     printf("DRAM heap init failed\r\n");
    //     return -1;
    // }
    // else
    // {
    //     printf("DRAM init succeed at addr:%lx\r\n", (uintptr_t)dram_heap_manager);
    // }

    uint64_t *data_ptr = (uint64_t *)SPM_WIDE_BASE_ADDR;
    printf("data_ptr is pointed at %lx\r\n", data_ptr);
    printf("data in SPM_WIDE_ADDR IS %lx\r\n", *data_ptr);
    uint8_t *data_dest1;
    data_dest1 = (uint8_t *)o1heapAllocate((const uint64_t)l3_heap_manager, data_size);
    if (!data_dest1)
    {
        printf("allocation for data_dest1 failed\r\n");
    }
    else
    {
        // *(volatile uint64_t **)(chiplet_addr_transform_loc(
        //     0xF, 0xF, (uintptr_t)&data_dest1)) = data_dest1;
        printf("allocated addr for data_dest1 in DRAM is %lx\r\n", data_dest1);
        // hemaia_xdma_memcpy_1d((uint8_t *)data, data_dest1, data_size);
        // uint32_t task_id = hemaia_xdma_start();
        // hemaia_xdma_remote_wait(task_id);
        sys_dma_blk_memcpy(current_chip_id, (uintptr_t)data_dest1, (uintptr_t)data, data_size);
        printf("IDMA copy finished to destination 1\r\n");
        printf("data in DRAM is %d\r\n", *data_dest1);
    }

    uint8_t *data_dest2;
    data_dest2 = (uint8_t *)o1heapAllocate((const uint64_t)l3_heap_manager, data_size);
    if (data_dest2 == NULL)
    {
        printf("allocation for data_dest2 failed\r\n");
        return -1;
    }
    else
    {
        printf("allocation for data_dest2 at local SRAM succeed at %lx\r\n", (uintptr_t)data_dest2);
        sys_dma_blk_memcpy(current_chip_id, (uintptr_t)data_dest2, (uintptr_t)data_dest1, data_size);
        printf("IDMA unicast copy finished to destination 2! \r\n");
        printf("data in SRAM is %d\r\n", *data_dest2);
    }

    o1heapFree((const uint64_t)dram_heap_manager, (const uint64_t)data_dest1);
    printf("DRAM HEAP freed\r\n");
    o1heapFree((const uint64_t)l3_heap_manager, (const uint64_t)data_dest2);
    printf("SRAM HEAP freed\r\n");

    return 0;
}