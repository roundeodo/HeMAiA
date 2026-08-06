#!/usr/bin/env python3

"""Frozen scheduler-only FPGA benchmark distributions."""

from dataclasses import dataclass
import heapq


MOE_SCHEDULER_BENCH_MAX_EXPERTS = 64


@dataclass(frozen=True)
class SchedulerBenchCase:
    name: str
    input_tokens: int
    counts: tuple[int, ...]
    n_experts: int = MOE_SCHEDULER_BENCH_MAX_EXPERTS
    initial_cache_eid_c2: int = -1
    initial_cache_eid_c3: int = -1

    @property
    def active_experts(self) -> int:
        return sum(count != 0 for count in self.counts)

    @property
    def padded_counts(self) -> tuple[int, ...]:
        return self.counts + (0,) * (
            self.n_experts - len(self.counts)
        )

    @property
    def token_major_topk_indices(self) -> tuple[int, ...]:
        """Build a deterministic Top-2 stream with the requested histogram."""
        remaining = [(-count, expert_id) for expert_id, count in enumerate(self.counts)]
        heapq.heapify(remaining)
        indices = []
        for _ in range(self.input_tokens):
            if len(remaining) < 2:
                raise ValueError(f"{self.name}: counts cannot form distinct Top-2 pairs")
            first_count, first_eid = heapq.heappop(remaining)
            second_count, second_eid = heapq.heappop(remaining)
            indices.extend((first_eid, second_eid))
            if first_count + 1 < 0:
                heapq.heappush(remaining, (first_count + 1, first_eid))
            if second_count + 1 < 0:
                heapq.heappush(remaining, (second_count + 1, second_eid))
        if remaining:
            raise ValueError(f"{self.name}: Top-2 construction left unused assignments")
        return tuple(indices)


SCHEDULER_BENCH_CASES = {
    case.name: case
    for case in (
        SchedulerBenchCase(
            name="long_tail_m70",
            input_tokens=70,
            counts=(22, 18, 14) + (3,) * 19 + (2,) * 8 + (1,) * 13,
        ),
        SchedulerBenchCase(
            name="three_hot_m70",
            input_tokens=70,
            counts=(28, 28, 28) + (6,) * 4 + (2,) * 16,
        ),
        SchedulerBenchCase(
            name="parameter_order_m92",
            input_tokens=92,
            counts=(76, 40) + (2,) * 32 + (1,) * 4,
        ),
        SchedulerBenchCase(
            name="high_skew_m60",
            input_tokens=60,
            counts=(36, 22, 13, 6) + (2,) * 17 + (1,) * 9,
        ),
        SchedulerBenchCase(
            name="native_router_e8_m32",
            input_tokens=32,
            # Exact logical-eid counts produced by the seeded native Router.
            counts=(7, 10, 7, 8, 14, 10, 4, 4),
            n_experts=8,
            initial_cache_eid_c2=0,
            initial_cache_eid_c3=7,
        ),
    )
}


def get_scheduler_bench_case(name: str) -> SchedulerBenchCase:
    try:
        case = SCHEDULER_BENCH_CASES[name]
    except KeyError as exc:
        choices = ", ".join(SCHEDULER_BENCH_CASES)
        raise ValueError(f"unknown scheduler benchmark case {name!r}; use {choices}") from exc

    if case.n_experts <= 0 or case.n_experts > MOE_SCHEDULER_BENCH_MAX_EXPERTS:
        raise ValueError(f"{name}: expert count exceeds the scheduler ABI")
    if len(case.counts) > case.n_experts:
        raise ValueError(f"{name}: too many active experts")
    for cache_eid in (case.initial_cache_eid_c2, case.initial_cache_eid_c3):
        if cache_eid < -1 or cache_eid >= case.n_experts:
            raise ValueError(f"{name}: initial cache eid is outside the expert range")
    if any(count <= 0 or count > 0x1FF for count in case.counts):
        raise ValueError(f"{name}: expert counts must fit the 9-bit scheduler ABI")
    if sum(case.counts) != case.input_tokens * 2:
        raise ValueError(f"{name}: counts do not sum to the Top-2 assignment total")
    return case


for _case_name in SCHEDULER_BENCH_CASES:
    get_scheduler_bench_case(_case_name)
