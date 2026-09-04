#!/usr/bin/env python3
"""Tests for deterministic 64-expert Top-2 routing distributions."""

from collections import Counter
import unittest

from moe_routing_cases import ROUTING_CASES


class RoutingCasesTest(unittest.TestCase):
    def test_all_requested_token_counts_exist(self) -> None:
        self.assertEqual({8, 32, 64, 128}, set(ROUTING_CASES))

    def test_every_case_realizes_its_marginal_counts(self) -> None:
        for total_tokens, case in ROUTING_CASES.items():
            actual = [0] * case.n_experts
            self.assertEqual(total_tokens, len(case.token_major_top2))
            for first, second in case.token_major_top2:
                self.assertNotEqual(first, second)
                actual[first] += 1
                actual[second] += 1
            self.assertEqual(2 * total_tokens, sum(actual))
            self.assertEqual(case.expected_counts, tuple(actual))

    def test_m8_uses_requested_4_2_2_pair_groups(self) -> None:
        pairs = Counter(ROUTING_CASES[8].token_major_top2)
        self.assertEqual(4, pairs[(0, 63)])
        self.assertEqual(2, pairs[(0, 1)])
        self.assertEqual(2, pairs[(63, 1)])
        self.assertEqual(3, len(pairs))


if __name__ == "__main__":
    unittest.main()
