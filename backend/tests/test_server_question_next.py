import unittest
from unittest.mock import patch

from app import server


class QuestionNextEndpointTests(unittest.TestCase):
    def test_question_next_uses_alt_presets_and_legacy_request_shape(self):
        captured = {}

        def fake_sample_alt(*, conn_tier, n_stops, obscurity, mode, rng, level=None):
            captured["conn_tier"] = conn_tier
            captured["n_stops"] = n_stops
            captured["obscurity"] = obscurity
            captured["mode"] = mode
            captured["level"] = level
            return server.gen.sample(level=1, mode=mode, rng=rng)

        req = server.QuestionRequest(level=4, mode="hard", seed=123)
        with patch.object(server.gen, "sample_alt", side_effect=fake_sample_alt):
            out = server.question_next(req)

        self.assertEqual(captured["mode"], "hard")
        self.assertEqual(captured["level"], 4)
        self.assertIn((captured["conn_tier"], captured["n_stops"], captured["obscurity"]), server.LEVEL_ALT_PRESETS[4])
        self.assertIn("group_id", out)
        self.assertIn("a_lat", out)
        self.assertIn("b_lon", out)

    def test_question_next_response_level_matches_requested_level(self):
        req = server.QuestionRequest(level=1, mode="normal", seed=1)

        out = server.question_next(req)

        self.assertEqual(out["level"], 1)


if __name__ == "__main__":
    unittest.main()
