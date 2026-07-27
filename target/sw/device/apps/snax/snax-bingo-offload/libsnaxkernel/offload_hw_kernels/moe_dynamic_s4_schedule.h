// Runtime-generic S4 bank-phase helpers.
#pragma once

__attribute__((always_inline)) static inline uint32_t __moe_s4_blocks_in_phase(
    uint32_t block_count, uint32_t phase)
{
    return (block_count + 1u - phase) / 2u;
}

/* slices_per_block=1 represents one gate+up pair transferred by BOTH.  A
 * single DMA uses two phase-local slices so each transfer covers one side. */
__attribute__((always_inline)) static inline uint32_t __moe_s4_dma_run(
    uint32_t block_count, uint32_t slices_per_block,
    uint32_t phase, uint32_t ordinal,
    uint32_t *block, uint32_t *side)
{
    uint32_t phase_blocks = __moe_s4_blocks_in_phase(block_count, phase);
    if (ordinal >= slices_per_block * phase_blocks) return 0u;
    *side = ordinal / phase_blocks;
    *block = phase + 2u * (ordinal % phase_blocks);
    return 1u;
}

__attribute__((always_inline)) static inline uint32_t __moe_s4_phase_at_step(
    uint32_t initial_phase, uint32_t step)
{
    return (initial_phase ^ step) & 1u;
}
