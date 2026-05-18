import random
import unittest
from unittest.mock import patch

from app.data import Dataset
from app.questions import Leg, QuestionGenerator


class QuestionSamplingTests(unittest.TestCase):
    def _dataset_for_sampling(self) -> Dataset:
        nodes = [
            {"iata": "AAA", "tier": 3, "name": "A", "city": "A", "country": "X"},
            {"iata": "BBB", "tier": 3, "name": "B", "city": "B", "country": "X"},
            {"iata": "CCC", "tier": 3, "name": "C", "city": "C", "country": "X"},
            {"iata": "DDD", "tier": 3, "name": "D", "city": "D", "country": "X"},
        ]
        edges = [
            {"a": "AAA", "b": "CCC", "airlines": ["AL1"]},
            {"a": "CCC", "b": "DDD", "airlines": ["AL1"]},
            {"a": "DDD", "b": "BBB", "airlines": ["AL1"]},
            {"a": "AAA", "b": "DDD", "airlines": ["AL2"]},
        ]
        airlines = [{"iata": "AL1", "name": "Airline 1"}, {"iata": "AL2", "name": "Airline 2"}]
        groups = [
            {"id": "g_fail", "name": "Fail", "obscurity": 1, "airlines": ["AL2"], "anchor": None},
            {"id": "g_pass", "name": "Pass", "obscurity": 1, "airlines": ["AL1"], "anchor": None},
        ]
        return Dataset(nodes=nodes, edges=edges, airlines=airlines, groups=groups)

    def test_sample_endpoints_match_target_tier_and_multi_hop_allowed(self):
        gen = QuestionGenerator(self._dataset_for_sampling())
        gen._sample_airport_pair = lambda pool, rng: ("AAA", "BBB")  # type: ignore[assignment]
        q = gen.sample(level=3, rng=random.Random(7))
        self.assertEqual(q.obscurity, 1)
        self.assertEqual(q.conn_tier, 3)
        self.assertEqual(gen._airport_meta[q.a]["tier"], 3)
        self.assertEqual(gen._airport_meta[q.b]["tier"], 3)
        self.assertGreaterEqual(q.min_stops, 2)

    def test_groups_exhausted_for_same_pair_before_resample(self):
        gen = QuestionGenerator(self._dataset_for_sampling())

        gen._sample_airport_pair = lambda pool, rng: ("AAA", "BBB")  # type: ignore[assignment]
        sampled_pairs = []
        original_pair = gen._sample_airport_pair

        def capture_pair(pool, rng):
            pair = original_pair(pool, rng)
            sampled_pairs.append(pair)
            return pair

        subgraph_order = []
        original_subgraph = gen._subgraph

        def capture_subgraph(gid):
            subgraph_order.append(gid)
            return original_subgraph(gid)

        gen._sample_airport_pair = capture_pair  # type: ignore[assignment]
        gen._subgraph = capture_subgraph  # type: ignore[assignment]

        with patch("app.questions._shuffled", side_effect=lambda xs, _rng: list(xs)):
            q = gen.sample(level=3, rng=random.Random(2))

        self.assertTrue(sampled_pairs)
        self.assertEqual(len(sampled_pairs), 1)
        self.assertGreaterEqual(len(subgraph_order), 2)
        self.assertEqual(subgraph_order[0:2], ["g_fail", "g_pass"])
        self.assertEqual(q.group_id, "g_pass")

    def test_seeded_sampling_is_deterministic(self):
        gen = QuestionGenerator(self._dataset_for_sampling())
        q1 = gen.sample(level=3, rng=random.Random(99))
        q2 = gen.sample(level=3, rng=random.Random(99))
        self.assertEqual((q1.group_id, q1.a, q1.b), (q2.group_id, q2.a, q2.b))

    def test_validate_example_and_routes_work_for_multi_hop_sample(self):
        gen = QuestionGenerator(self._dataset_for_sampling())
        gen._sample_airport_pair = lambda pool, rng: ("AAA", "BBB")  # type: ignore[assignment]
        q = gen.sample(level=3, rng=random.Random(5))
        self.assertGreaterEqual(q.min_stops, 2)

        example = gen.example_answer(q, rng=random.Random(11))
        self.assertGreaterEqual(len(example), 2)
        result = gen.validate_answer(q, example)
        self.assertTrue(result.valid)

        one_leg = [example[0]]
        one_leg_result = gen.validate_answer(q, one_leg)
        self.assertFalse(one_leg_result.valid)

    def test_level_one_always_uses_tier_one_endpoints(self):
        nodes = [
            {"iata": "AAA", "tier": 1, "name": "A", "city": "A", "country": "X"},
            {"iata": "BBB", "tier": 1, "name": "B", "city": "B", "country": "X"},
            {"iata": "CCC", "tier": 2, "name": "C", "city": "C", "country": "X"},
        ]
        edges = [{"a": "AAA", "b": "BBB", "airlines": ["AL1"]}]
        airlines = [{"iata": "AL1", "name": "Airline 1"}]
        groups = [{"id": "g1", "name": "G1", "obscurity": 1, "airlines": ["AL1"], "anchor": None}]
        gen = QuestionGenerator(Dataset(nodes=nodes, edges=edges, airlines=airlines, groups=groups))

        q = gen.sample(level=1, rng=random.Random(3))

        self.assertEqual(q.level, 1)
        self.assertEqual(gen._airport_meta[q.a]["tier"], 1)
        self.assertEqual(gen._airport_meta[q.b]["tier"], 1)
        self.assertEqual(q.conn_tier, 1)

    def test_partner_anchor_flights_not_used_for_question_generation(self):
        nodes = [
            {"iata": "AAA", "tier": 2, "name": "A", "city": "A", "country": "X"},
            {"iata": "BBB", "tier": 2, "name": "B", "city": "B", "country": "X"},
        ]
        edges = [{"a": "AAA", "b": "BBB", "airlines": ["AN"]}]
        airlines = [{"iata": "AN", "name": "Anchor"}]
        groups = [
            {
                "id": "partner",
                "name": "Partner",
                "type": "partner_program",
                "obscurity": 2,
                "airlines": ["AL1"],
                "anchor": "AN",
            }
        ]
        gen = QuestionGenerator(Dataset(nodes=nodes, edges=edges, airlines=airlines, groups=groups))
        with self.assertRaises(ValueError):
            gen.question_for("partner", "AAA", "BBB")

    def test_partner_mixed_anchor_legs_keep_non_anchor_airlines(self):
        nodes = [
            {"iata": "AAA", "tier": 2, "name": "A", "city": "A", "country": "X"},
            {"iata": "BBB", "tier": 2, "name": "B", "city": "B", "country": "X"},
        ]
        edges = [{"a": "AAA", "b": "BBB", "airlines": ["AL1", "AN"]}]
        airlines = [{"iata": "AL1", "name": "Partner"}, {"iata": "AN", "name": "Anchor"}]
        groups = [
            {
                "id": "partner",
                "name": "Partner",
                "type": "partner_program",
                "obscurity": 2,
                "airlines": ["AL1"],
                "anchor": "AN",
            }
        ]
        gen = QuestionGenerator(Dataset(nodes=nodes, edges=edges, airlines=airlines, groups=groups))
        q = gen.question_for("partner", "AAA", "BBB")
        self.assertEqual(q.min_stops, 1)


class RouteBuilderValidationTests(unittest.TestCase):
    def _dataset(self) -> Dataset:
        nodes = [
            {"iata": "JFK", "icao": "KJFK", "tier": 2, "name": "John F. Kennedy", "city": "New York", "country": "US", "lat": 40.6, "lon": -73.7},
            {"iata": "LHR", "icao": "EGLL", "tier": 2, "name": "Heathrow", "city": "London", "country": "UK", "lat": 51.47, "lon": -0.45},
            {"iata": "MAD", "icao": "LEMD", "tier": 2, "name": "Madrid", "city": "Madrid", "country": "ES", "lat": 40.47, "lon": -3.56},
            {"iata": "SFO", "icao": "KSFO", "tier": 2, "name": "San Francisco", "city": "San Francisco", "country": "US", "lat": 37.61, "lon": -122.38},
        ]
        edges = [
            {"a": "JFK", "b": "LHR", "airlines": ["BA", "AA"]},
            {"a": "LHR", "b": "MAD", "airlines": ["IB", "BA"]},
            {"a": "MAD", "b": "SFO", "airlines": ["IB"]},
            {"a": "JFK", "b": "SFO", "airlines": ["UA"]},
        ]
        airlines = [
            {"iata": "AA", "name": "American"},
            {"iata": "BA", "name": "British Airways"},
            {"iata": "IB", "name": "Iberia"},
            {"iata": "UA", "name": "United"},
        ]
        groups = [
            {"id": "oneworld", "name": "Oneworld", "obscurity": 1, "airlines": ["AA", "BA", "IB"], "anchor": None}
        ]
        return Dataset(nodes=nodes, edges=edges, airlines=airlines, groups=groups)

    def test_map_data_filters_group_and_includes_start_end(self):
        gen = QuestionGenerator(self._dataset())
        q = gen.question_for("oneworld", "JFK", "MAD")
        payload = gen.map_data_for_question(q)
        self.assertEqual(payload["startAirport"], "JFK")
        self.assertEqual(payload["endAirport"], "MAD")
        self.assertEqual([a["iata"] for a in payload["airlines"]], ["AA", "BA", "IB"])
        self.assertEqual({a["iata"] for a in payload["airports"]}, {"JFK", "LHR", "MAD", "SFO"})

    def test_invalid_stopover_rejected(self):
        gen = QuestionGenerator(self._dataset())
        q = gen.question_for("oneworld", "JFK", "MAD")
        result = gen.validate_answer(q, [
            Leg("JFK", "LHR", "BA"),
            # force an invalid intermediate airport not in group subgraph
            Leg("LHR", "XXX", "BA"),
            Leg("XXX", "MAD", "IB"),
        ])
        self.assertFalse(result.valid)

    def test_invalid_airline_rejected(self):
        gen = QuestionGenerator(self._dataset())
        q = gen.question_for("oneworld", "JFK", "MAD")
        result = gen.validate_answer(q, [
            Leg("JFK", "LHR", "UA"),
            Leg("LHR", "MAD", "IB"),
        ])
        self.assertFalse(result.valid)

    def test_invalid_leg_airline_rejected_and_leg_filtering(self):
        gen = QuestionGenerator(self._dataset())
        q = gen.question_for("oneworld", "JFK", "MAD")
        result = gen.validate_answer(q, [
            Leg("JFK", "LHR", "IB"),
            Leg("LHR", "MAD", "IB"),
        ])
        self.assertFalse(result.valid)
        self.assertEqual(
            [a["iata"] for a in gen.valid_airlines_for_leg("oneworld", "JFK", "LHR")],
            ["AA", "BA"],
        )


if __name__ == "__main__":
    unittest.main()
