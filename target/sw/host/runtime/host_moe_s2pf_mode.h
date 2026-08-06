#pragma once

static inline __attribute__((always_inline)) uint32_t
__host_moe_s2pf_runtime_ctrl(
    uint32_t has_s2pf,
    uint32_t skip_s1,
    uint32_t shape_s1,
    uint32_t dma_s1)
{
    uint32_t s1_compute_ticks = 8u >> shape_s1;
    uint32_t s1_dma_ticks =
        dma_s1 == (uint32_t)MOE_DMA_BOTH ? 2u : 4u;
    uint32_t valid = has_s2pf != 0u;
    uint32_t early = valid & (skip_s1 == 0u) &
        (s1_compute_ticks > s1_dma_ticks);
    return (valid << BINGO_MOE_DYN_CTRL_S2PF_RUNTIME_RELEASE_BIT) |
        (early << BINGO_MOE_DYN_CTRL_S2PF_EARLY_BIT);
}
