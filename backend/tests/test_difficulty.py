import unittest

from app.difficulty import combo_branches, feasible_combos


class DifficultyRulesTests(unittest.TestCase):
    def test_feasible_combos_match_expected_pairs(self):
        expected = {
            1: {(1, 1)},
            2: {(1, 2)},
            3: {(1, 3)},
            4: {(2, 2), (2, 3), (1, 4)},
            5: {(2, 4), (1, 5)},
            6: {(3, 4), (2, 5), (1, 6)},
            7: {(3, 5), (2, 6), (1, 7)},
            8: {(3, 6), (2, 7), (1, 8)},
            9: {(3, 7), (2, 8), (1, 9)},
            10: {(3, 8), (2, 9), (3, 9), (1, 10), (2, 10), (3, 10)},
        }

        for level, expected_pairs in expected.items():
            self.assertEqual(set(feasible_combos(level)), expected_pairs)

    def test_combo_branches_match_expected_structure(self):
        self.assertEqual(combo_branches(5), [[(2, 4)], [(1, 5)]])
        self.assertEqual(
            combo_branches(10),
            [[(3, 8)], [(2, 9)], [(3, 9)], [(1, 10)], [(2, 10)], [(3, 10)]],
        )


if __name__ == "__main__":
    unittest.main()
