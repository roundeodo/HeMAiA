// Runtime-generic S4 bank-phase schedule.
#pragma once

__attribute__((always_inline)) static inline uint32_t __moe_s4_blocks_in_phase(
    uint32_t block_count, uint32_t phase)
{
    return (block_count + 1u - phase) / 2u;
}

__attribute__((always_inline)) static inline uint32_t __moe_s4_phase_steps(
    uint32_t s1_block_count, uint32_t s3_block_count,
    uint32_t m_tiles, uint32_t phase)
{
    uint32_t dma_runs = __moe_s4_blocks_in_phase(
        s1_block_count, phase);
    uint32_t compute_runs = m_tiles * __moe_s4_blocks_in_phase(
        s3_block_count, phase);
    return dma_runs > compute_runs ? dma_runs : compute_runs;
}

__attribute__((always_inline)) static inline uint32_t
__moe_s4_phase_schedule_length(
    uint32_t s1_block_count, uint32_t s3_block_count, uint32_t m_tiles)
{
    return __moe_s4_phase_steps(
        s1_block_count, s3_block_count, m_tiles, 0u) +
        __moe_s4_phase_steps(
            s1_block_count, s3_block_count, m_tiles, 1u);
}

/* Return phase in bit 0 and the phase-local ordinal in bits [31:1].  This is
 * equivalent to the former six-counter state machine, but derives every step
 * directly and therefore keeps no address-taken loop state on the DM stack. */
__attribute__((always_inline)) static inline uint32_t __moe_s4_schedule_step(
    uint32_t phase_steps0, uint32_t phase_steps1,
    uint32_t initial_phase, uint32_t step)
{
    if (phase_steps0 == 0u) return (step << 1u) | 1u;
    if (phase_steps1 == 0u) return step << 1u;

    uint32_t paired = 2u * (phase_steps0 < phase_steps1 ?
        phase_steps0 : phase_steps1);
    if (step < paired) {
        return (step & ~1u) |
            ((initial_phase ^ (step & 1u)) & 1u);
    }

    uint32_t tail_phase = phase_steps0 > phase_steps1 ? 0u : 1u;
    uint32_t tail_ordinal = paired / 2u + step - paired;
    return (tail_ordinal << 1u) | tail_phase;
}

__attribute__((always_inline)) static inline uint32_t __moe_s4_dma_block(
    uint32_t block_count, uint32_t phase, uint32_t ordinal,
    uint32_t *block)
{
    if (ordinal >= __moe_s4_blocks_in_phase(block_count, phase)) {
        return 0u;
    }
    *block = phase + 2u * ordinal;
    return 1u;
}

__attribute__((always_inline)) static inline uint32_t __moe_s4_compute_run(
    uint32_t block_count, uint32_t m_tiles,
    uint32_t phase, uint32_t ordinal,
    uint32_t *mt, uint32_t *block)
{
    uint32_t phase_blocks = __moe_s4_blocks_in_phase(
        block_count, phase);
    if (ordinal >= m_tiles * phase_blocks) return 0u;
    *mt = ordinal / phase_blocks;
    *block = phase + 2u * (ordinal % phase_blocks);
    return 1u;
}

__attribute__((always_inline)) static inline uint32_t
__moe_s4_peek_compute_run(
    uint32_t block_count, uint32_t m_tiles, uint32_t initial_phase,
    uint32_t phase_steps0, uint32_t phase_steps1, uint32_t next_step,
    uint32_t *mt, uint32_t *block)
{
    uint32_t sync_steps = phase_steps0 + phase_steps1;
    while (next_step < sync_steps) {
        uint32_t scheduled = __moe_s4_schedule_step(
            phase_steps0, phase_steps1, initial_phase, next_step++);
        if (__moe_s4_compute_run(
                block_count, m_tiles, scheduled & 1u, scheduled >> 1u,
                mt, block) != 0u) {
            return 1u;
        }
    }
    return 0u;
}
