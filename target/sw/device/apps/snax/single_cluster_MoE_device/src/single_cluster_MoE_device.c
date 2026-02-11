#include "snrt.h"
#include "../../../../host/apps/single_cluster_MoE_host/data/data.h"
// 这是一个简单的 Device 程序，不依赖复杂的 Bingo 任务调度
// 它会被加载到 Cluster 中直接运行
#define TCDM_OFFSET 0x1000
int main()
{
    uint32_t dma_load_input_start;
    uint32_t dma_load_input_end;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addr() + TCDM_OFFSET;
    // 1. 获取当前 Core 的身份信息
    uint32_t core_id = snrt_cluster_core_idx();
    uint32_t cluster_id = snrt_cluster_idx();

    if (snrt_cluster_idx() == 0 && snrt_is_dm_core())
    {
        printf("[DEVICE] the address of data is %lx\r\n", data);
        printf("[DEVICE] dma_load_input_start_address is %lx\r\n", dma_load_input_start);
        printf("[DEVICE] dma_load_input_end address is %lx\r\n", dma_load_input_end);
        xdma_memcpy_1d(data, (void *)tcdm_baseaddress,
                       data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("[DEVICE] XDMA copy from L3 to TCDM C0 is done in %d cycles.\r\n",
               xdma_last_task_cycle());
        printf("[DEVICE] the data in tcdm_bassaddress is %d\r\n", *(uint32_t *)(tcdm_baseaddress));
    }
    snrt_global_barrier();
    // 只让每个 Cluster 的第 0 号核心打印，避免输出混乱
    if (core_id == 0)
    {
        printf("=== [Device] Cluster %d Core %d Started ===\n", cluster_id, core_id);

        // 2. 测试 L1 (TCDM)
        // snrt_l1_next() 返回当前 TCDM 中可用空间的起始地址
        // 这就像是 L1 上的 malloc，但非常轻量级
        volatile uint32_t *l1_ptr = (uint32_t *)snrt_l1_next();

        printf("[Device] L1 TCDM Base Address: 0x%08x\n", (uint32_t)l1_ptr);

        // 写入测试
        *l1_ptr = 0x12345678;

        // 内存屏障 (Fence)，确保写入完成
        asm volatile("fence" ::: "memory");

        // 读取验证
        uint32_t val = *l1_ptr;
        if (val == 0x12345678)
        {
            printf("[Device] L1 Read/Write Verified: 0x%x\n", val);
        }
        else
        {
            printf("[Device] Error: L1 Verification Failed! Read: 0x%x\n", val);
        }
    }

    // 3. 全局同步
    // 等待 Cluster 内所有核心到达这里
    snrt_cluster_hw_barrier();

    return 0;
}