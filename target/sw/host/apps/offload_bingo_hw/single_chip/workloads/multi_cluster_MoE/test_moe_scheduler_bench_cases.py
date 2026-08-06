import unittest
from collections import Counter

from moe_scheduler_bench_cases import (
    MOE_SCHEDULER_BENCH_MAX_EXPERTS,
    SCHEDULER_BENCH_CASES,
    get_scheduler_bench_case,
)


class SchedulerBenchCasesTest(unittest.TestCase):
    def test_case_contracts(self):
        expected = {
            "long_tail_m70": (
                70,
                (22, 18, 14) + (3,) * 19 + (2,) * 8 + (1,) * 13,
                MOE_SCHEDULER_BENCH_MAX_EXPERTS,
                -1,
                -1,
            ),
            "three_hot_m70": (
                70,
                (28, 28, 28) + (6,) * 4 + (2,) * 16,
                MOE_SCHEDULER_BENCH_MAX_EXPERTS,
                -1,
                -1,
            ),
            "parameter_order_m92": (
                92,
                (76, 40) + (2,) * 32 + (1,) * 4,
                MOE_SCHEDULER_BENCH_MAX_EXPERTS,
                -1,
                -1,
            ),
            "high_skew_m60": (
                60,
                (36, 22, 13, 6) + (2,) * 17 + (1,) * 9,
                MOE_SCHEDULER_BENCH_MAX_EXPERTS,
                -1,
                -1,
            ),
            "native_router_e8_m32": (
                32,
                (7, 10, 7, 8, 14, 10, 4, 4),
                8,
                0,
                7,
            ),
        }
        self.assertEqual(set(SCHEDULER_BENCH_CASES), set(expected))

        for name, (tokens, counts, n_experts, cache_c2, cache_c3) in expected.items():
            case = get_scheduler_bench_case(name)
            self.assertEqual(case.input_tokens, tokens)
            self.assertEqual(case.counts, counts)
            self.assertEqual(case.active_experts, len(counts))
            self.assertEqual(case.n_experts, n_experts)
            self.assertEqual(case.initial_cache_eid_c2, cache_c2)
            self.assertEqual(case.initial_cache_eid_c3, cache_c3)
            self.assertEqual(len(case.padded_counts), n_experts)
            self.assertEqual(sum(case.padded_counts), tokens * 2)

            topk_indices = case.token_major_topk_indices
            self.assertEqual(len(topk_indices), tokens * 2)
            self.assertTrue(all(
                topk_indices[i] != topk_indices[i + 1]
                for i in range(0, len(topk_indices), 2)
            ))
            self.assertEqual(
                Counter(topk_indices),
                Counter({eid: count for eid, count in enumerate(counts)}),
            )


if __name__ == "__main__":
    unittest.main()
