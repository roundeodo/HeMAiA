#!/usr/bin/env python3

import argparse
import pathlib
import sys

import hjson
import numpy as np

CURRENT_DIR = pathlib.Path(__file__).resolve().parent


def find_repo_root(start: pathlib.Path) -> pathlib.Path:
    for parent in (start, *start.parents):
        if (parent / "Bender.yml").is_file() and (parent / "deps").is_dir():
            return parent
    raise RuntimeError(f"Cannot find HeMAiA repo root from {start}")


REPO_ROOT = find_repo_root(CURRENT_DIR)
sys.path.insert(0, str(REPO_ROOT / "deps" / "snitch_cluster" / "util" / "sim"))
sys.path.insert(0, str(REPO_ROOT / "deps" / "snitch_cluster" / "util" / "silu_pkg"))
from data_utils import format_vector_definition  # noqa: E402
from moe_test_layout import derive_params  # noqa: E402
from moe_test_schedule import (  # noqa: E402
    BASELINE_PROFILE,
    C_TAIL_SMOKE_PROFILE,
    DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS,
    DYNAMIC_DESC_PROFILE,
    ENDS_INWARD_EXPECTED_MAKESPAN_TICKS,
    ENDS_INWARD_PROFILE,
    EXPERT_COUNT,
    HIGH_TO_LOW_COUNTS,
    HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS,
    HIGH_TO_LOW_PROFILE,
    LOW_TO_HIGH_EXPECTED_MAKESPAN_TICKS,
    LOW_TO_HIGH_PROFILE,
    S1_STAGE_SMOKE_PROFILE,
    SCHEDULE_PROFILES,
    STATIC_DESC_EXPECTED_MAKESPAN_TICKS,
    STATIC_DESC_PROFILE,
    STATIC_DESC_TOKEN_IDS_BY_EXPERT,
    build_schedule_profile,
)
from silu_out16_balanced_golden import silu_out16_balanced_eval_q  # noqa: E402


def pack_int4(values) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    lo = flat[0::2].astype(np.uint8) & 0x0F
    hi = flat[1::2].astype(np.uint8) & 0x0F
    return lo | (hi << 4)


def rescale_down_32to16(values) -> np.ndarray:
    return np.clip(values.astype(np.int64), -32768, 32767).astype(np.int16)


def apply_silu(values) -> np.ndarray:
    flat = values.reshape(-1)
    result = np.asarray(
        [silu_out16_balanced_eval_q(int(value)) for value in flat],
        dtype=np.int16,
    )
    return result.reshape(values.shape)


def generate_swiglu_weights(rng: np.random.Generator, p: dict, prefix: str):
    hidden = p["hidden_size"]
    intermediate = p["intermediate_size"]
    blocks = p["s1_block_count"]
    gate_n1_s0 = (intermediate // blocks) // 4

    gate_shape = (blocks, gate_n1_s0, hidden // 8, 8, 4)
    gate = rng.integers(-7, 8, size=gate_shape, dtype=np.int8)
    up = rng.integers(-7, 8, size=gate_shape, dtype=np.int8)

    # Canonical production B order: block, N1, K, meshCol, tileSize.
    gate_stream = pack_int4(gate.transpose(0, 1, 2, 4, 3))
    up_stream = pack_int4(up.transpose(0, 1, 2, 4, 3))
    if gate_stream.size != p["gate_weight_bytes"]:
        raise ValueError("gate weight byte count does not match layout")

    gate_matrix = gate.transpose(2, 3, 0, 1, 4).reshape(hidden, intermediate)
    up_matrix = up.transpose(2, 3, 0, 1, 4).reshape(hidden, intermediate)
    lines = [
        format_vector_definition("uint8_t", f"{prefix}_gate_B", gate_stream, alignment=128),
        format_vector_definition("uint8_t", f"{prefix}_up_B", up_stream, alignment=128),
    ]
    return lines, gate_matrix, up_matrix


def generate_expert_weights(rng: np.random.Generator, p: dict, prefix: str):
    hidden = p["hidden_size"]
    intermediate = p["intermediate_size"]
    blocks = p["s3_block_count"]
    down_n1_s0 = ((hidden // 2) // blocks) // 4
    down_shape = (2, blocks, down_n1_s0, intermediate // 8, 8, 4)
    down = rng.integers(-7, 8, size=down_shape, dtype=np.int8)

    lines, gate_matrix, up_matrix = generate_swiglu_weights(rng, p, prefix)
    # Down keeps left/right VC halves outside the two output blocks.
    down_stream = pack_int4(down.transpose(0, 1, 2, 3, 5, 4))
    if down_stream.size != p["down_weight_bytes"]:
        raise ValueError("down weight byte count does not match layout")
    lines.append(
        format_vector_definition("uint8_t", f"{prefix}_down_B", down_stream, alignment=128)
    )
    down_halves = [
        down[half].transpose(2, 3, 0, 1, 4).reshape(intermediate, hidden // 2)
        for half in range(2)
    ]
    down_matrix = np.concatenate(down_halves, axis=1)
    return lines, gate_matrix, up_matrix, down_matrix


def generate_slot_golden(
    tokens: np.ndarray,
    gate_matrix: np.ndarray,
    up_matrix: np.ndarray,
    down_matrix: np.ndarray,
) -> np.ndarray:
    gate_acc = tokens.astype(np.int32) @ gate_matrix.astype(np.int32)
    up_acc = tokens.astype(np.int32) @ up_matrix.astype(np.int32)
    gate_silu = apply_silu(rescale_down_32to16(gate_acc))
    mode0 = rescale_down_32to16(
        gate_silu.astype(np.int64) * rescale_down_32to16(up_acc).astype(np.int64)
    )
    down_acc = mode0.astype(np.int32) @ down_matrix.astype(np.int32)
    return rescale_down_32to16(down_acc)


def _emit_baseline_header(p: dict, rng: np.random.Generator) -> str:
    token_values = rng.integers(
        -256,
        256,
        size=(p["gather_source_tokens"], p["hidden_size"]),
        dtype=np.int16,
    )
    token_bytes = token_values.view(np.uint8).reshape(
        p["gather_source_tokens"], -1
    )
    input_rows_2d = token_bytes
    input_rows = input_rows_2d.reshape(-1)
    slot0_refs = np.arange(p["prod_slot0_tokens"], dtype=np.uint16)
    slot1_refs = np.asarray([9, 1, 14, 5, 23, 31], dtype=np.uint16)
    if slot1_refs.size != p["prod_slot1_tokens"]:
        raise ValueError(
            "slot1 reference pattern must match the benchmark token count"
        )
    prod_slot_token_refs = np.concatenate((slot0_refs, slot1_refs))
    lines = [
        "// Auto-generated by multi_cluster_MoE_test_datagen.py.",
        "// 32 dense INT16 token rows; L1 bank placement is performed by 2D gather.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        format_vector_definition("uint8_t", "moe_test_input_A", input_rows, alignment=128),
        format_vector_definition(
            "uint16_t",
            "moe_test_prod_slot_token_refs",
            prod_slot_token_refs,
            alignment=128,
        ),
    ]
    for prefix in ("c0", "c1"):
        weight_lines, gate_matrix, up_matrix, down_matrix = generate_expert_weights(
            rng, p, f"moe_test_{prefix}"
        )
        lines += weight_lines
        slot0_golden = generate_slot_golden(
            token_values[slot0_refs], gate_matrix, up_matrix, down_matrix
        )
        slot1_golden = generate_slot_golden(
            token_values[slot1_refs], gate_matrix, up_matrix, down_matrix
        )
        lines.append(
            format_vector_definition(
                "int16_t",
                f"moe_test_{prefix}_slot0_golden",
                slot0_golden.reshape(-1),
                alignment=128,
            )
        )
        lines.append(
            format_vector_definition(
                "int16_t",
                f"moe_test_{prefix}_slot1_golden",
                slot1_golden.reshape(-1),
                alignment=128,
            )
        )
    return "\n\n".join(lines)


def _high_to_low_token_refs(max_tokens_per_expert: int):
    total_routes = sum(HIGH_TO_LOW_COUNTS)
    source_tokens = total_routes // 2
    refs = np.zeros(EXPERT_COUNT * max_tokens_per_expert, dtype=np.uint16)
    token_ids_by_expert = {}
    route = 0
    for expert_id, ntokens in enumerate(HIGH_TO_LOW_COUNTS):
        token_ids = []
        for local_token in range(ntokens):
            token_id = route % source_tokens
            rank = route // source_tokens
            refs[expert_id * max_tokens_per_expert + local_token] = np.uint16(
                token_id | (rank << 15)
            )
            token_ids.append(token_id)
            route += 1
        token_ids_by_expert[expert_id] = np.asarray(token_ids, dtype=np.int64)
    if route != total_routes:
        raise AssertionError("high-to-low token reference count mismatch")
    # Token 0 rank 0 is encoded as zero, so validate using the explicit lists.
    explicit_histogram = np.bincount(
        np.concatenate(tuple(token_ids_by_expert.values())),
        minlength=source_tokens,
    )
    if not np.all(explicit_histogram == 2):
        raise AssertionError("every synthetic source token must route to two experts")
    return refs, token_ids_by_expert


def _static_desc_token_refs(max_tokens_per_expert: int):
    refs = np.zeros(EXPERT_COUNT * max_tokens_per_expert, dtype=np.uint16)
    token_ids_by_expert = {
        expert_id: np.asarray(token_ids, dtype=np.int64)
        for expert_id, token_ids in enumerate(STATIC_DESC_TOKEN_IDS_BY_EXPERT)
    }
    owners = [[] for _ in range(sum(HIGH_TO_LOW_COUNTS) // 2)]
    for expert_id, token_ids in token_ids_by_expert.items():
        for token_id in token_ids:
            owners[int(token_id)].append(expert_id)
    if any(len(token_owners) != 2 for token_owners in owners):
        raise AssertionError("STATIC_DESC routing must have exactly two owners")

    for expert_id, token_ids in token_ids_by_expert.items():
        start = expert_id * max_tokens_per_expert
        for local_rank, token_id in enumerate(token_ids):
            route_rank = owners[int(token_id)].index(expert_id)
            refs[start + local_rank] = np.uint16(
                int(token_id) | (route_rank << 15)
            )
    return refs, token_ids_by_expert


def _smoke_token_refs(queues, max_tokens_per_expert: int):
    slots = sorted(
        (slot for cluster_slots in queues.values() for slot in cluster_slots),
        key=lambda slot: slot.expert_id,
    )
    max_expert_id = max(slot.expert_id for slot in slots)
    refs = np.zeros(
        (max_expert_id + 1) * max_tokens_per_expert,
        dtype=np.uint16,
    )
    token_ids_by_expert = {}
    next_token_id = 0
    for slot in slots:
        token_ids = np.arange(
            next_token_id,
            next_token_id + slot.ntokens,
            dtype=np.int64,
        )
        start = slot.expert_id * max_tokens_per_expert
        refs[start : start + slot.ntokens] = token_ids.astype(np.uint16)
        token_ids_by_expert[slot.expert_id] = token_ids
        next_token_id += slot.ntokens
    return refs, token_ids_by_expert


def _emit_high_to_low_header(p: dict, rng: np.random.Generator) -> str:
    queues = build_schedule_profile(p["schedule_profile"])
    token_values = rng.integers(
        -256,
        256,
        size=(p["gather_source_tokens"], p["hidden_size"]),
        dtype=np.int16,
    )
    input_rows = token_values.view(np.uint8).reshape(-1)
    if p["schedule_profile"] in (
        STATIC_DESC_PROFILE,
        DYNAMIC_DESC_PROFILE,
    ):
        token_refs, token_ids_by_expert = _static_desc_token_refs(
            p["prod_max_tokens_per_expert"]
        )
        counts = np.asarray(HIGH_TO_LOW_COUNTS, dtype=np.uint8)
        if p["schedule_profile"] == STATIC_DESC_PROFILE:
            profile_comment = (
                "// Case 0 STATIC_DESC; exact exported routing and fixed B/B lanes."
            )
        else:
            profile_comment = (
                "// Case 0 DYNAMIC_DESC; exact exported routing and dynamic lanes."
            )
    elif p["schedule_profile"] in (
        HIGH_TO_LOW_PROFILE,
        LOW_TO_HIGH_PROFILE,
        ENDS_INWARD_PROFILE,
    ):
        token_refs, token_ids_by_expert = _high_to_low_token_refs(
            p["prod_max_tokens_per_expert"]
        )
        counts = np.asarray(HIGH_TO_LOW_COUNTS, dtype=np.uint8)
        direction = p["schedule_profile"].replace("_", "-")
        profile_comment = (
            f"// Optional {direction} 64-expert profile; 43 experts are active."
        )
    else:
        token_refs, token_ids_by_expert = _smoke_token_refs(
            queues, p["prod_max_tokens_per_expert"]
        )
        counts = np.zeros(EXPERT_COUNT, dtype=np.uint8)
        for slots in queues.values():
            for slot in slots:
                counts[slot.expert_id] = slot.ntokens
        profile_comment = "// Focused E23/E24 two-token smoke profile."
    lines = [
        "// Auto-generated by multi_cluster_MoE_test_datagen.py.",
        profile_comment,
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define MOE_TEST_HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS {HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS}u",
        f"#define MOE_TEST_STATIC_DESC_EXPECTED_MAKESPAN_TICKS {STATIC_DESC_EXPECTED_MAKESPAN_TICKS}u",
        f"#define MOE_TEST_DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS {DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS}u",
        f"#define MOE_TEST_LOW_TO_HIGH_EXPECTED_MAKESPAN_TICKS {LOW_TO_HIGH_EXPECTED_MAKESPAN_TICKS}u",
        f"#define MOE_TEST_ENDS_INWARD_EXPECTED_MAKESPAN_TICKS {ENDS_INWARD_EXPECTED_MAKESPAN_TICKS}u",
        format_vector_definition(
            "uint8_t", "moe_test_input_A", input_rows, alignment=128
        ),
        format_vector_definition(
            "uint16_t",
            "moe_test_prod_slot_token_refs",
            token_refs,
            alignment=128,
        ),
        format_vector_definition(
            "uint8_t",
            "moe_test_high_to_low_counts",
            counts,
            alignment=64,
        ),
    ]
    for cluster_name in ("c0", "c1"):
        weight_lines, gate_matrix, up_matrix, down_matrix = generate_expert_weights(
            rng, p, f"moe_test_{cluster_name}"
        )
        lines += weight_lines
        for slot in queues[cluster_name]:
            token_ids = token_ids_by_expert[slot.expert_id][
                slot.token_start_rank : slot.token_start_rank + slot.ntokens
            ]
            if len(token_ids) != slot.ntokens:
                raise AssertionError("slot token slice exceeds routed-token list")
            golden = generate_slot_golden(
                token_values[token_ids],
                gate_matrix,
                up_matrix,
                down_matrix,
            )
            lines.append(
                format_vector_definition(
                    "int16_t",
                    f"moe_test_{cluster_name}_e{slot.expert_id:02d}_golden",
                    golden.reshape(-1),
                    alignment=128,
                )
            )
    return "\n\n".join(lines)


def emit_header(config: dict, schedule_profile: str = BASELINE_PROFILE) -> str:
    p = derive_params(config, schedule_profile)
    rng = np.random.default_rng(320)
    if schedule_profile in (
        HIGH_TO_LOW_PROFILE,
        LOW_TO_HIGH_PROFILE,
        ENDS_INWARD_PROFILE,
        STATIC_DESC_PROFILE,
        DYNAMIC_DESC_PROFILE,
        C_TAIL_SMOKE_PROFILE,
        S1_STAGE_SMOKE_PROFILE,
    ):
        return _emit_high_to_low_header(p, rng)
    return _emit_baseline_header(p, rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument(
        "--schedule-profile",
        choices=SCHEDULE_PROFILES,
        default=BASELINE_PROFILE,
    )
    args = parser.parse_args()
    with args.cfg.open() as f:
        cfg = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hwcfg = hjson.loads(f.read())
    print(emit_header({**cfg, **hwcfg}, args.schedule_profile))


if __name__ == "__main__":
    main()
