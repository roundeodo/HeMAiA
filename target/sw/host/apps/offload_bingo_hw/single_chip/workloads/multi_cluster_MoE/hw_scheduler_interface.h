#pragma once
#include <stdint.h>

// Software ABI for hemaia_bingo_dynamic_task_csr.sv.
// The dynamic task CSR window is carved out of the existing host ready/done
// AXI-lite page so non-dynamic Bingo ready/done accesses still pass through to
// bingo_hw_manager_top.

#define MOE_DYNAMIC_TASK_CSR_BASE_OFFSET       0x800u

#define MOE_DYNAMIC_TASK_STATUS_OFFSET         0x00u
#define MOE_DYNAMIC_TASK_CONTROL_OFFSET        0x04u
#define MOE_DYNAMIC_TASK_TASK_CREDIT_OFFSET    0x08u
#define MOE_DYNAMIC_TASK_DONE_CREDIT_OFFSET    0x0cu
#define MOE_DYNAMIC_TASK_DONE_WORD_BASE        0x10u
#define MOE_DYNAMIC_TASK_DESC_WORD_BASE        0x40u

#define MOE_DYNAMIC_TASK_CONTROL_COMMIT        0x1u
#define MOE_DYNAMIC_TASK_CONTROL_FLUSH         0x2u
#define MOE_DYNAMIC_TASK_CONTROL_POP_DONE      0x4u

#define MOE_DYNAMIC_TASK_STATUS_TASK_READY     0x00000001u
#define MOE_DYNAMIC_TASK_STATUS_TASK_VALID     0x00000002u
#define MOE_DYNAMIC_TASK_STATUS_DONE_VALID     0x00000004u
#define MOE_DYNAMIC_TASK_STATUS_DONE_READY     0x00000008u
#define MOE_DYNAMIC_TASK_STATUS_TASK_WORDS(s)  (((s) >> 8) & 0xffu)
#define MOE_DYNAMIC_TASK_STATUS_DONE_WORDS(s)  (((s) >> 16) & 0xffu)

#define MOE_DYNAMIC_TASK_DESC_WORDS            2u
#define MOE_DYNAMIC_TASK_DESC_BITS             (MOE_DYNAMIC_TASK_DESC_WORDS * 32u)
#define MOE_DYNAMIC_TASK_DONE_WORDS            1u

#define MOE_DYNAMIC_TASK_FLAG_VALID            0x00000001u
#define MOE_DYNAMIC_TASK_FLAG_TOKEN_SLICE_BASE 0x00000002u

typedef struct {
    uint32_t words[MOE_DYNAMIC_TASK_DESC_WORDS];
} moe_dynamic_task_desc_words_t;

static inline uintptr_t moe_dynamic_task_csr_base(void)
{
    return (uintptr_t)chiplet_addr_transform(
        (uint64_t)quad_ctrl_host_ready_done_queue_addr() +
        (uint64_t)MOE_DYNAMIC_TASK_CSR_BASE_OFFSET);
}

static inline uint32_t moe_dynamic_task_csr_read(uint32_t byte_offset)
{
    return readw(moe_dynamic_task_csr_base() + (uintptr_t)byte_offset);
}

static inline void moe_dynamic_task_csr_write(uint32_t byte_offset, uint32_t data)
{
    writew(data, moe_dynamic_task_csr_base() + (uintptr_t)byte_offset);
}

static inline uint32_t moe_dynamic_task_csr_status(void)
{
    return moe_dynamic_task_csr_read(MOE_DYNAMIC_TASK_STATUS_OFFSET);
}

static inline uint32_t moe_dynamic_task_csr_task_credit(void)
{
    return moe_dynamic_task_csr_read(MOE_DYNAMIC_TASK_TASK_CREDIT_OFFSET);
}

static inline void moe_dynamic_task_csr_flush(void)
{
    moe_dynamic_task_csr_write(MOE_DYNAMIC_TASK_CONTROL_OFFSET,
                               MOE_DYNAMIC_TASK_CONTROL_FLUSH);
}

static inline void moe_dynamic_task_desc_clear(moe_dynamic_task_desc_words_t *desc)
{
    for (uint32_t i = 0; i < MOE_DYNAMIC_TASK_DESC_WORDS; i++) {
        desc->words[i] = 0u;
    }
}

static inline void moe_dynamic_task_desc_set_bits(
    moe_dynamic_task_desc_words_t *desc, uint32_t lsb, uint32_t width,
    uint64_t value)
{
    for (uint32_t bit_idx = 0; bit_idx < width; bit_idx++) {
        uint32_t dst_bit = lsb + bit_idx;
        if (dst_bit >= MOE_DYNAMIC_TASK_DESC_BITS) {
            continue;
        }
        uint32_t word_idx = dst_bit >> 5;
        uint32_t word_bit = dst_bit & 31u;
        uint32_t mask = 1u << word_bit;
        if (((value >> bit_idx) & 1ull) != 0ull) {
            desc->words[word_idx] |= mask;
        } else {
            desc->words[word_idx] &= ~mask;
        }
    }
}

static inline void moe_dynamic_task_desc_pack(
    moe_dynamic_task_desc_words_t *desc,
    uint32_t seq_id,
    uint32_t cluster_id,
    uint32_t expert_id,
    uint32_t token_start_rank,
    uint32_t ntokens,
    uint32_t shape_s1,
    uint32_t shape_s3,
    uint32_t bw_s1,
    uint32_t bw_s3,
    uint32_t dma_s1,
    uint32_t dma_s3,
    uint32_t skip_s1,
    uint32_t skip_s3,
    int32_t prefetch_eid,
    uint32_t est_start_cc,
    uint32_t est_end_cc,
    uint64_t token_ids_base_addr,
    uint64_t token_kpos_base_addr,
    uint64_t input_a_base_addr,
    uint64_t output_base_addr,
    uint32_t flags)
{
    (void)expert_id;
    (void)token_start_rank;
    (void)ntokens;
    (void)shape_s1;
    (void)shape_s3;
    (void)bw_s1;
    (void)bw_s3;
    (void)dma_s1;
    (void)dma_s3;
    (void)skip_s1;
    (void)skip_s3;
    (void)prefetch_eid;
    (void)est_start_cc;
    (void)est_end_cc;
    (void)token_ids_base_addr;
    (void)token_kpos_base_addr;
    (void)input_a_base_addr;
    (void)output_base_addr;

    moe_dynamic_task_desc_clear(desc);

    // The compact CSR FIFO now carries a raw 64-bit Bingo task descriptor.
    // Per-expert MoE metadata lives in the software-programmed slot argument
    // pool, so the CSR path only keeps the fields the hardware manager consumes.
    moe_dynamic_task_desc_set_bits(desc, 0u,  1u, 0u);                  // cond_exec_invert
    moe_dynamic_task_desc_set_bits(desc, 1u,  5u, 0u);                  // cond_exec_group_id
    moe_dynamic_task_desc_set_bits(desc, 6u,  1u, 0u);                  // cond_exec_en
    moe_dynamic_task_desc_set_bits(desc, 7u,  2u, 0u);                  // normal task
    moe_dynamic_task_desc_set_bits(desc, 9u, 12u, seq_id & 0xfffu);     // task_id
    moe_dynamic_task_desc_set_bits(desc, 21u, 8u, 0u);                  // chiplet 0
    moe_dynamic_task_desc_set_bits(desc, 29u, 2u, cluster_id & 0x3u);   // cluster
    moe_dynamic_task_desc_set_bits(desc, 31u, 2u, 0u);                  // core 0
    moe_dynamic_task_desc_set_bits(desc, 63u, 1u,
                                   (flags & MOE_DYNAMIC_TASK_FLAG_VALID) != 0u);
}

static inline void moe_dynamic_task_desc_write_shadow(
    const moe_dynamic_task_desc_words_t *desc)
{
    for (uint32_t word_idx = 0; word_idx < MOE_DYNAMIC_TASK_DESC_WORDS; word_idx++) {
        moe_dynamic_task_csr_write(
            MOE_DYNAMIC_TASK_DESC_WORD_BASE + (word_idx << 2),
            desc->words[word_idx]);
    }
}

static inline uint32_t moe_dynamic_task_try_emit(
    const moe_dynamic_task_desc_words_t *desc)
{
    if (moe_dynamic_task_csr_task_credit() == 0u) {
        return 0u;
    }
    moe_dynamic_task_desc_write_shadow(desc);
    moe_dynamic_task_csr_write(MOE_DYNAMIC_TASK_CONTROL_OFFSET,
                               MOE_DYNAMIC_TASK_CONTROL_COMMIT);
    return 1u;
}
