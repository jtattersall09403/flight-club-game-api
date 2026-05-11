import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.data import build_dataset_from_airline_routes, load_dataset


class DataLoadingTests(unittest.TestCase):
    def _write_static_files(self, root: Path) -> Path:
        processed = root / "data" / "processed"
        processed.mkdir(parents=True)
        (processed / "nodes.json").write_text('[{"iata":"OLD","tier":10}]')
        (processed / "edges.json").write_text('[{"a":"OLD","b":"ZZZ","airlines":["OL"]}]')
        (processed / "airlines.json").write_text('[{"iata":"BA","name":"British Airways"},{"iata":"VS","name":"Virgin Atlantic"},{"iata":"GX","name":"Group X"}]')
        (processed / "groups.json").write_text('[{"id":"g1","name":"Group 1","obscurity":1,"anchor":null,"airlines":["BA","GX"]}]')
        return processed

    def test_undirected_edge_merge_and_tiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = self._write_static_files(Path(tmp))
            raw = {
                "AAA": {
                    "iata": "AAA",
                    "name": "A",
                    "city": "ACity",
                    "country": "Aland",
                    "lat": 1,
                    "lon": 2,
                    "routes": [{"iata": "BBB", "carriers": [{"iata": "BA", "name": "British Airways"}]}],
                },
                "BBB": {
                    "iata": "BBB",
                    "routes": [{"iata": "AAA", "carriers": [{"iata": "BA"}, {"iata": "KL", "name": "KLM"}]}],
                },
            }
            ds = build_dataset_from_airline_routes(raw, processed)
            self.assertEqual(len(ds.edges), 1)
            self.assertEqual(ds.edges[0], {"a": "AAA", "b": "BBB", "airlines": ["BA", "KL"]})
            tiers = {n["iata"]: n["tier"] for n in ds.nodes}
            self.assertEqual(tiers["AAA"], 10)
            self.assertEqual(tiers["BBB"], 10)

    def test_malformed_entries_skipped_and_group_airline_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = self._write_static_files(Path(tmp))
            raw = {
                "AAA": {
                    "routes": [
                        {"iata": None, "carriers": [{"iata": "BA"}]},
                        {"iata": "AAA", "carriers": [{"iata": "BA"}]},
                        {"iata": "BBB", "carriers": [None, {"name": "No code"}, {"iata": "BA", "name": "BA Name"}]},
                    ]
                },
                "BBB": {"routes": "bad"},
            }
            ds = build_dataset_from_airline_routes(raw, processed)
            self.assertEqual(ds.edges, [{"a": "AAA", "b": "BBB", "airlines": ["BA"]}])
            airlines = {a["iata"]: a["name"] for a in ds.airlines}
            self.assertEqual(airlines["BA"], "BA Name")
            self.assertEqual(airlines["GX"], "Group X")
            self.assertEqual(len(ds.groups), 1)
            self.assertEqual(ds.groups[0]["id"], "g1")

    def test_routes_data_mode_static_uses_static_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = self._write_static_files(Path(tmp))
            with patch.dict("os.environ", {"ROUTES_DATA_MODE": "static"}, clear=False):
                ds = load_dataset(processed)
            self.assertEqual(ds.nodes, [{"iata": "OLD", "tier": 10}])

    def test_remote_failure_falls_back_cache_then_static(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = self._write_static_files(root)
            cache = root / "cache.json"
            cache.write_text('{"AAA": {"routes": [{"iata": "BBB", "carriers": [{"iata": "BA"}]}]}, "BBB": {"routes": []}}')

            env = {
                "ROUTES_DATA_MODE": "remote",
                "ROUTES_CACHE_PATH": str(cache),
                "ROUTES_DATA_URL": "https://example.invalid/routes.json",
            }
            with patch.dict("os.environ", env, clear=False):
                with patch("app.data.fetch_remote_routes", side_effect=RuntimeError("boom")):
                    ds = load_dataset(processed)
            self.assertEqual(ds.edges, [{"a": "AAA", "b": "BBB", "airlines": ["BA"]}])

            cache.unlink()
            with patch.dict("os.environ", env, clear=False):
                with patch("app.data.fetch_remote_routes", side_effect=RuntimeError("boom")):
                    ds2 = load_dataset(processed)
            self.assertEqual(ds2.nodes, [{"iata": "OLD", "tier": 10}])

    def test_partner_program_anchor_removed_from_group_airlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = self._write_static_files(Path(tmp))
            (processed / "groups.json").write_text(
                '[{"id":"p1","name":"Partner","type":"partner_program","obscurity":2,"anchor":"ba","airlines":["BA","VS","vs"]}]'
            )
            ds = load_dataset(processed)
            self.assertEqual(ds.groups[0]["anchor"], "BA")
            self.assertEqual(ds.groups[0]["airlines"], ["VS"])


if __name__ == "__main__":
    unittest.main()
