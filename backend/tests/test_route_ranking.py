import unittest

from app.data import Dataset
from app.graph import k_shortest_paths
from app.questions import QuestionGenerator


class RouteRankingTests(unittest.TestCase):
    def _dataset(self) -> Dataset:
        nodes = [
            {"iata": "AAA", "tier": 2, "name": "A", "city": "A", "country": "X", "lat": 0.0, "lon": 0.0},
            {"iata": "BBB", "tier": 2, "name": "B", "city": "B", "country": "X", "lat": 1.0, "lon": 1.0},
            {"iata": "CCC", "tier": 2, "name": "C", "city": "C", "country": "X", "lat": 2.0, "lon": 2.0},
            {"iata": "DDD", "tier": 2, "name": "D", "city": "D", "country": "X", "lat": 3.0, "lon": 3.0},
            {"iata": "EEE", "tier": 2, "name": "E", "city": "E", "country": "X", "lat": 4.0, "lon": 4.0},
            {"iata": "FFF", "tier": 2, "name": "F", "city": "F", "country": "X", "lat": 5.0, "lon": 5.0},
        ]
        edges = [
            {"a": "AAA", "b": "CCC", "airlines": ["AL1"]},
            {"a": "CCC", "b": "BBB", "airlines": ["AL1"]},
            {"a": "AAA", "b": "DDD", "airlines": ["AL1"]},
            {"a": "DDD", "b": "EEE", "airlines": ["AL1"]},
            {"a": "EEE", "b": "BBB", "airlines": ["AL1"]},
            {"a": "AAA", "b": "FFF", "airlines": ["AL1"]},
            {"a": "FFF", "b": "EEE", "airlines": ["AL1"]},
        ]
        airlines = [{"iata": "AL1", "name": "Airline 1"}]
        groups = [{"id": "g1", "name": "Group", "obscurity": 2, "airlines": ["AL1"], "anchor": None}]
        return Dataset(nodes=nodes, edges=edges, airlines=airlines, groups=groups)

    def _weights(self):
        return {
            ("AAA", "CCC"): 100.0,
            ("BBB", "CCC"): 100.0,
            ("AAA", "DDD"): 10.0,
            ("DDD", "EEE"): 10.0,
            ("BBB", "EEE"): 10.0,
            ("AAA", "FFF"): 5.0,
            ("EEE", "FFF"): 5.0,
        }

    def test_k_shortest_paths_stops_distance_prefers_fewer_legs(self):
        adj = {
            "AAA": {"CCC", "DDD", "FFF"},
            "BBB": {"CCC", "EEE"},
            "CCC": {"AAA", "BBB"},
            "DDD": {"AAA", "EEE"},
            "EEE": {"DDD", "BBB", "FFF"},
            "FFF": {"AAA", "EEE"},
        }
        out = k_shortest_paths(
            adj,
            self._weights(),
            "AAA",
            "BBB",
            k=2,
            min_legs=2,
            rank_by="stops_distance",
        )
        self.assertEqual(out[0][1], ["AAA", "CCC", "BBB"])

        by_distance = k_shortest_paths(adj, self._weights(), "AAA", "BBB", k=1, min_legs=2)
        self.assertEqual(by_distance[0][1], ["AAA", "FFF", "EEE", "BBB"])

    def test_question_generator_routes_are_sorted_by_stops_then_distance(self):
        gen = QuestionGenerator(self._dataset())
        question = gen.question_for("g1", "AAA", "BBB")
        routes = gen.k_shortest_routes(question, k=3)
        ordering = [(r["stops"], r["total_km"]) for r in routes]
        self.assertEqual(ordering, sorted(ordering))
        self.assertEqual(routes[0]["path"], ["AAA", "CCC", "BBB"])


if __name__ == "__main__":
    unittest.main()
