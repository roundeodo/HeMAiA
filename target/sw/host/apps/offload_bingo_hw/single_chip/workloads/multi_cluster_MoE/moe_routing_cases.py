#!/usr/bin/env python3
"""Deterministic Top-2 routing cases for the full multi-cluster MoE workload."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq

EXPERT_ORDER = (0, 63, *range(1, 63))


@dataclass(frozen=True)
class RoutingCase:
    name: str
    total_tokens: int
    n_experts: int
    token_major_top2: tuple[tuple[int, int], ...]
    expected_counts: tuple[int, ...]

    @property
    def pair_histogram(self) -> tuple[tuple[int, int, int], ...]:
        histogram = Counter(self.token_major_top2)
        return tuple(
            (first, second, count)
            for (first, second), count in sorted(histogram.items())
        )


def _counts_in_expert_order(groups: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    counts = []
    for count, repeat in groups:
        counts.extend([count] * repeat)
    if len(counts) > len(EXPERT_ORDER):
        raise ValueError("routing distribution defines more than 64 experts")
    counts.extend([0] * (len(EXPERT_ORDER) - len(counts)))
    by_eid = [0] * len(EXPERT_ORDER)
    for eid, count in zip(EXPERT_ORDER, counts):
        by_eid[eid] = count
    return tuple(by_eid)


def _pair_degrees(expected_counts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Realize expert marginal counts as deterministic Top-2 token pairs."""
    total_routes = sum(expected_counts)
    if total_routes % 2:
        raise ValueError("Top-2 marginal count sum must be even")
    total_tokens = total_routes // 2
    if max(expected_counts, default=0) > total_tokens:
        raise ValueError("an expert cannot receive more than one route per token")

    rank = {eid: index for index, eid in enumerate(EXPERT_ORDER)}
    heap = [
        (-count, rank[eid], eid) for eid, count in enumerate(expected_counts) if count
    ]
    heapq.heapify(heap)
    pairs = []
    while heap:
        if len(heap) < 2:
            raise ValueError("marginal counts cannot be realized without self-routing")
        neg0, order0, eid0 = heapq.heappop(heap)
        neg1, order1, eid1 = heapq.heappop(heap)
        pairs.append((eid0, eid1))
        neg0 += 1
        neg1 += 1
        if neg0:
            heapq.heappush(heap, (neg0, order0, eid0))
        if neg1:
            heapq.heappush(heap, (neg1, order1, eid1))

    # Keep identical pairs contiguous so the generated workload is readable.
    histogram = Counter(pairs)
    grouped = []
    for pair in sorted(histogram, key=lambda p: (rank[p[0]], rank[p[1]])):
        grouped.extend([pair] * histogram[pair])
    return tuple(grouped)


def _make_case(
    name: str,
    total_tokens: int,
    count_groups: tuple[tuple[int, int], ...],
    explicit_pairs: tuple[tuple[int, int, int], ...] | None = None,
) -> RoutingCase:
    expected_counts = _counts_in_expert_order(count_groups)
    if sum(expected_counts) != 2 * total_tokens:
        raise ValueError(f"{name}: counts must sum to 2*M")
    if explicit_pairs is None:
        top2 = _pair_degrees(expected_counts)
    else:
        top2 = tuple(
            pair
            for first, second, repeat in explicit_pairs
            for pair in [(first, second)] * repeat
        )
    if len(top2) != total_tokens:
        raise ValueError(f"{name}: pair list must contain M tokens")

    actual_counts = [0] * 64
    for first, second in top2:
        if first == second or not (0 <= first < 64 and 0 <= second < 64):
            raise ValueError(f"{name}: invalid Top-2 pair {(first, second)}")
        actual_counts[first] += 1
        actual_counts[second] += 1
    if tuple(actual_counts) != expected_counts:
        raise ValueError(f"{name}: pair list does not realize expected counts")
    return RoutingCase(name, total_tokens, 64, top2, expected_counts)


ROUTING_CASES = {
    8: _make_case(
        "m8_4_2_2",
        8,
        ((6, 2), (4, 1)),
        explicit_pairs=((0, 63, 4), (0, 1, 2), (63, 1, 2)),
    ),
    32: _make_case(
        "m32_moe_style",
        32,
        ((8, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 14), (1, 10), (0, 35)),
    ),
    64: _make_case(
        "m64_scaled_skew",
        64,
        ((14, 2), (5, 1), (4, 7), (3, 12), (2, 10), (1, 11), (0, 21)),
    ),
    128: _make_case(
        "m128_scaled_skew",
        128,
        (
            (29, 1),
            (27, 1),
            (11, 1),
            (9, 5),
            (7, 5),
            (6, 2),
            (5, 7),
            (4, 10),
            (2, 11),
            (0, 21),
        ),
    ),
}


def get_routing_case(total_tokens: int, n_experts: int) -> RoutingCase:
    if n_experts != 64:
        raise ValueError("deterministic routing workload requires 64 experts")
    try:
        return ROUTING_CASES[total_tokens]
    except KeyError as exc:
        supported = ", ".join(str(m) for m in ROUTING_CASES)
        raise ValueError(
            f"no deterministic routing case for M={total_tokens}; supported: {supported}"
        ) from exc
