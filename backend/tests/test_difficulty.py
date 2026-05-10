import unittest

from app.difficulty import combo_branches, feasible_combos


class DifficultyRulesTests(unittest.TestCase):
    def test_feasible_combos_match_expected_pairs(self):
        expected = {
            1: {(1, 1)},
            2: {(1, 2)},
            3: {(1, 3)},
            4: {(1, 4), (2, 2), (3, 2)},
            5: {(1, 5), (2, 3), (3, 3)},
            6: {(1, 6), (4, 4), (5, 4)},
            7: {(2, 7), (3, 7), (5, 5), (6, 5)},
            8: {(2, 8), (3, 8), (7, 6), (8, 6)},
            9: {(4, 9), (5, 9), (8, 7), (9, 7)},
            10: {(6, 10), (7, 10), (8, 10), (9, 10), (10, 10)},
        }

        for level, expected_pairs in expected.items():
            self.assertEqual(set(feasible_combos(level)), expected_pairs)

    def test_combo_branches_match_expected_structure(self):
        self.assertEqual(combo_branches(5), [[(1, 5)], [(2, 3), (3, 3)]])
        self.assertEqual(combo_branches(10), [[(6, 10), (7, 10), (8, 10), (9, 10), (10, 10)]])


if __name__ == "__main__":
    unittest.main()
