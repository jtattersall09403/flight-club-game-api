import unittest

from app.difficulty import feasible_combos


class DifficultyRulesTests(unittest.TestCase):
    def test_feasible_combos_match_expected_obscurity_bands(self):
        expected = {
            1: {1},
            2: {1},
            3: {1},
            4: {2},
            5: {3},
            6: {4, 5},
            7: {6, 7},
            8: {8},
            9: {9},
            10: {10},
        }

        for level, allowed_obscurity in expected.items():
            combos = feasible_combos(level)
            self.assertTrue(combos)
            self.assertEqual({c for _o, c in combos}, {level})
            self.assertEqual({o for o, _c in combos}, allowed_obscurity)


if __name__ == "__main__":
    unittest.main()
