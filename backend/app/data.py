"""Load route dataset, preferring startup-time remote snapshot with fallbacks."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

# backend/app/data.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "cache" / "airline_routes.json"
DEFAULT_ROUTES_DATA_URL = (
    "https://raw.githubusercontent.com/Jonty/airline-route-data/refs/heads/main/airline_routes.json"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    airlines: list[dict[str, Any]]
    groups: list[dict[str, Any]]


def compute_tier(degree: int) -> int:
    if degree >= 100:
        return 1
    if degree >= 50:
        return 2
    if degree >= 20:
        return 3
    if degree >= 5:
        return 4
    return 5


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_static_dataset(data_dir: Path | None = None) -> Dataset:
    d = data_dir or DEFAULT_DATA_DIR
    return Dataset(
        nodes=_load_json(d / "nodes.json"),
        edges=_load_json(d / "edges.json"),
        airlines=_load_json(d / "airlines.json"),
        groups=_load_json(d / "groups.json"),
    )


def fetch_remote_routes(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_s) as response:  # nosec B310 - trusted configurable URL
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Remote route data is not a JSON object")
    return data


def load_cached_routes(cache_path: Path) -> dict[str, Any]:
    data = _load_json(cache_path)
    if not isinstance(data, dict):
        raise ValueError("Cached route data is not a JSON object")
    return data


def write_cached_routes(cache_path: Path, data: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False))


def _parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _pick(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _normalize_iata(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    return code or None


def build_dataset_from_airline_routes(raw: dict[str, Any], data_dir: Path | None = None) -> Dataset:
    d = data_dir or DEFAULT_DATA_DIR
    groups = _load_json(d / "groups.json")
    static_airlines = _load_json(d / "airlines.json")
    static_airline_names = {a.get("iata"): a.get("name") for a in static_airlines if isinstance(a, dict)}

    airport_meta: dict[str, dict[str, Any]] = {}
    edge_airlines: dict[tuple[str, str], set[str]] = {}
    airline_names: dict[str, str] = {}

    def absorb_airport(code: str, rec: dict[str, Any]) -> None:
        meta = airport_meta.setdefault(code, {"iata": code})
        meta["name"] = _pick(rec, "name", "airport", "airport_name") or meta.get("name")
        meta["city"] = _pick(rec, "city", "city_name") or meta.get("city")
        meta["country"] = _pick(rec, "country", "country_name") or meta.get("country")
        lat = _parse_float(_pick(rec, "lat", "latitude"))
        lon = _parse_float(_pick(rec, "lon", "lng", "longitude"))
        if lat is not None:
            meta["lat"] = lat
        if lon is not None:
            meta["lon"] = lon

    for origin_key, airport_record in raw.items():
        if not isinstance(airport_record, dict):
            continue
        origin = _normalize_iata(airport_record.get("iata") or origin_key)
        if not origin:
            continue
        absorb_airport(origin, airport_record)

        routes = airport_record.get("routes")
        if not isinstance(routes, list):
            continue

        for route in routes:
            if not isinstance(route, dict):
                continue
            dest = _normalize_iata(route.get("iata") or route.get("to") or route.get("destination"))
            if not dest or dest == origin:
                continue
            dest_record = raw.get(dest)
            if isinstance(dest_record, dict):
                absorb_airport(dest, dest_record)
            else:
                airport_meta.setdefault(dest, {"iata": dest})

            pair = tuple(sorted((origin, dest)))
            airlines = edge_airlines.setdefault(pair, set())
            carriers = route.get("carriers")
            if not isinstance(carriers, list):
                continue
            for carrier in carriers:
                if not isinstance(carrier, dict):
                    continue
                code = _normalize_iata(carrier.get("iata") or carrier.get("code"))
                if not code:
                    continue
                airlines.add(code)
                cname = carrier.get("name")
                if isinstance(cname, str) and cname.strip():
                    airline_names[code] = cname.strip()

    nodes: list[dict[str, Any]] = []
    neighbors: dict[str, set[str]] = {}
    edges: list[dict[str, Any]] = []

    for (a, b), airlines in edge_airlines.items():
        if not airlines:
            continue
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
        edges.append({"a": a, "b": b, "airlines": sorted(airlines)})

    for code, meta in airport_meta.items():
        degree = len(neighbors.get(code, set()))
        nodes.append(
            {
                "iata": code,
                "name": meta.get("name") or code,
                "city": meta.get("city") or "",
                "country": meta.get("country") or "",
                "tier": compute_tier(degree),
                "lat": meta.get("lat"),
                "lon": meta.get("lon"),
            }
        )

    group_airlines: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        for code in group.get("airlines", []):
            norm = _normalize_iata(code)
            if norm:
                group_airlines.add(norm)

    for code in group_airlines:
        airline_names.setdefault(code, static_airline_names.get(code) or code)

    for code, name in static_airline_names.items():
        if isinstance(code, str) and code in airline_names and not airline_names[code]:
            airline_names[code] = name or code

    airlines = [{"iata": code, "name": name or code} for code, name in sorted(airline_names.items())]

    nodes.sort(key=lambda x: x["iata"])
    edges.sort(key=lambda x: (x["a"], x["b"]))

    return Dataset(nodes=nodes, edges=edges, airlines=airlines, groups=groups)


def load_dataset(data_dir: Path | None = None) -> Dataset:
    d = data_dir or DEFAULT_DATA_DIR
    mode = os.getenv("ROUTES_DATA_MODE", "remote").strip().lower()
    cache_path = Path(os.getenv("ROUTES_CACHE_PATH", str(DEFAULT_CACHE_PATH)))
    url = os.getenv("ROUTES_DATA_URL", DEFAULT_ROUTES_DATA_URL)

    if mode == "static":
        ds = load_static_dataset(d)
        logger.info(
            "Loaded route data source=static airports=%d edges=%d airlines=%d groups=%d",
            len(ds.nodes),
            len(ds.edges),
            len(ds.airlines),
            len(ds.groups),
        )
        return ds

    if mode == "remote":
        try:
            raw = fetch_remote_routes(url)
            write_cached_routes(cache_path, raw)
            logger.info("Cached remote route data at %s", cache_path)
            ds = build_dataset_from_airline_routes(raw, d)
            logger.info(
                "Loaded route data source=remote url=%s airports=%d edges=%d airlines=%d groups=%d",
                url,
                len(ds.nodes),
                len(ds.edges),
                len(ds.airlines),
                len(ds.groups),
            )
            return ds
        except Exception as exc:
            logger.warning("Remote route data fetch failed: %s", exc)

    try:
        raw = load_cached_routes(cache_path)
        ds = build_dataset_from_airline_routes(raw, d)
        logger.info(
            "Loaded route data source=cache path=%s airports=%d edges=%d airlines=%d groups=%d",
            cache_path,
            len(ds.nodes),
            len(ds.edges),
            len(ds.airlines),
            len(ds.groups),
        )
        return ds
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Route cache unavailable at %s: %s", cache_path, exc)

    logger.warning("Falling back to static processed dataset")
    ds = load_static_dataset(d)
    logger.info(
        "Loaded route data source=static-fallback airports=%d edges=%d airlines=%d groups=%d",
        len(ds.nodes),
        len(ds.edges),
        len(ds.airlines),
        len(ds.groups),
    )
    return ds
