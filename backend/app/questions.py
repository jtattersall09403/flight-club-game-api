"""Lazy question generator.

A request for a level-N question runs roughly:

  1. Look at `feasible_combos(N)` -> list of (obscurity, conn_tier) pairs.
  2. Shuffle that list. For each (O, C) combo:
       a. Sample two distinct airports A and B from tier C.
       b. Shuffle groups whose `obscurity == O`.
       c. For each candidate group, build (or fetch) its subgraph and check
          whether A and B are connected by any number of hops.
       d. Keep the same pair while trying all groups; only resample the pair
          after group exhaustion.
  3. If no valid pair+group is found within MAX_ATTEMPTS_PER_REQUEST pair
     attempts, raise.

Subgraph builds happen on demand and are LRU-cached, so repeated group checks
are cheap after first materialization.
"""
from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from . import graph
from .data import Dataset
from .difficulty import LEVEL_MAX, LEVEL_MIN, combo_branches, difficulty_level, feasible_combos

Mode = Literal["normal", "hard"]
VALID_MODES: tuple[Mode, ...] = ("normal", "hard")

# How hard we try before giving up on a level.
MAX_ATTEMPTS_PER_REQUEST = 200


@dataclass(frozen=True)
class Leg:
    src: str
    dst: str
    airline: str

    @classmethod
    def parse(cls, spec: str) -> "Leg":
        for sep in (":", "-", ","):
            parts = spec.split(sep)
            if len(parts) == 3:
                a, b, c = (p.strip().upper() for p in parts)
                return cls(a, b, c)
        raise ValueError(f"could not parse leg spec: {spec!r}")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None
    stops: int | None = None


@dataclass(frozen=True)
class Question:
    group_id: str
    group_name: str
    obscurity: int
    a: str
    b: str
    a_name: str | None
    b_name: str | None
    a_city: str | None
    b_city: str | None
    a_country: str | None
    b_country: str | None
    level: int
    conn_tier: int
    min_stops: int
    direct_available: bool
    mode: Mode

    def display_a(self) -> str:
        return self._display(self.a, self.a_name, self.a_city, self.a_country)

    def display_b(self) -> str:
        return self._display(self.b, self.b_name, self.b_city, self.b_country)

    def _display(
        self,
        code: str,
        name: str | None,
        city: str | None,
        country: str | None,
    ) -> str:
        if self.mode == "hard":
            return code
        parts = [p for p in (name, city, country) if p]
        return f"{code} ({', '.join(parts)})" if parts else code

    def to_dict(self) -> dict[str, Any]:
        normal = self.mode == "normal"
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "obscurity": self.obscurity,
            "a": self.a,
            "b": self.b,
            "a_name": self.a_name if normal else None,
            "b_name": self.b_name if normal else None,
            "a_city": self.a_city if normal else None,
            "b_city": self.b_city if normal else None,
            "a_country": self.a_country if normal else None,
            "b_country": self.b_country if normal else None,
            "level": self.level,
            "connectivity_tier": self.conn_tier,
            "min_stops": self.min_stops,
            "direct_available": self.direct_available,
            "mode": self.mode,
        }


class QuestionGenerator:
    """Lazy on-demand question generator.

    Build cost: O(nodes + groups) (no subgraph materialization).
    Per-sample cost: repeated random airport-pair attempts, where each attempt
    may scan groups at a target obscurity and run BFS reachability checks.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._airport_meta: dict[str, dict[str, Any]] = {n["iata"]: n for n in dataset.nodes}
        self._groups_by_id: dict[str, dict[str, Any]] = {g["id"]: g for g in dataset.groups}
        self._airline_name: dict[str, str] = {a["iata"]: a["name"] for a in dataset.airlines}

        # Index nodes by tier (for random endpoint-pair sampling).
        self._nodes_by_tier: dict[int, list[str]] = {}
        for n in dataset.nodes:
            self._nodes_by_tier.setdefault(int(n["tier"]), []).append(n["iata"])

        # Index groups by obscurity (for random pick of group at target O).
        self._groups_by_obscurity: dict[int, list[str]] = {}
        for g in dataset.groups:
            self._groups_by_obscurity.setdefault(int(g["obscurity"]), []).append(g["id"])

        # Anti-trivial-hub: anchor's hubs (computed lazily, cached).
        # Threshold: anchor airline degree >= 50% of its max degree.
        self._HUB_THRESHOLD = 0.5
        self._anchor_data_cache: dict[str, tuple[set[tuple[str, str]], set[str]]] = {}

    # ------------------------------------------------------------------ cache

    @functools.lru_cache(maxsize=64)
    def _subgraph(self, group_id: str) -> tuple[
        dict[str, set[str]],
        dict[tuple[str, str], list[str]],
    ]:
        group = self._groups_by_id[group_id]
        return graph.build_subgraph(self.dataset.edges, group)

    def _anchor_data(self, group_id: str) -> tuple[set[tuple[str, str]], set[str]]:
        if group_id in self._anchor_data_cache:
            return self._anchor_data_cache[group_id]
        group = self._groups_by_id[group_id]
        anchor = group.get("anchor")
        if not anchor:
            empty: tuple[set[tuple[str, str]], set[str]] = (set(), set())
            self._anchor_data_cache[group_id] = empty
            return empty
        routes: set[tuple[str, str]] = set()
        for e in self.dataset.edges:
            if anchor in e["airlines"]:
                a, b = e["a"], e["b"]
                routes.add((a, b) if a < b else (b, a))
        if not routes:
            self._anchor_data_cache[group_id] = (set(), set())
            return self._anchor_data_cache[group_id]
        deg: dict[str, int] = {}
        for a, b in routes:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        max_deg = max(deg.values())
        cutoff = self._HUB_THRESHOLD * max_deg
        hubs = {iata for iata, d in deg.items() if d >= cutoff}
        self._anchor_data_cache[group_id] = (routes, hubs)
        return routes, hubs

    def _is_trivial_via_anchor_hub(self, group_id: str, a: str, b: str) -> bool:
        routes, hubs = self._anchor_data(group_id)
        if not hubs:
            return False
        for h in hubs:
            if h == a or h == b:
                continue
            ah = (a, h) if a < h else (h, a)
            hb = (h, b) if h < b else (b, h)
            if ah in routes and hb in routes:
                return True
        return False

    # ----------------------------------------------------------------- sample

    def sample(
        self,
        level: int,
        mode: Mode = "normal",
        rng: random.Random | None = None,
    ) -> Question:
        if level < LEVEL_MIN or level > LEVEL_MAX:
            raise ValueError(f"level must be in [{LEVEL_MIN}, {LEVEL_MAX}]")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")
        rng = rng or random

        combos = feasible_combos(level)
        if not combos:
            raise ValueError(f"no feasible (obscurity, tier) combos for level {level}")

        branches = combo_branches(level)
        combo_order: list[tuple[int, int]] = []
        for branch in _shuffled(branches, rng):
            combo_order.extend(_shuffled(branch, rng))

        # Try combos in branch-randomized order.
        attempts = 0
        for combo in combo_order:
            obscurity, target_tier = combo
            group_ids = self._groups_by_obscurity.get(obscurity, [])
            if not group_ids:
                continue
            airport_pool = self._nodes_by_tier.get(target_tier, [])
            if len(airport_pool) < 2:
                continue
            while attempts < MAX_ATTEMPTS_PER_REQUEST:
                attempts += 1
                pair = self._sample_airport_pair(airport_pool, rng)
                if pair is None:
                    break
                a_iata, b_iata = pair
                # Defensive guard: sampled endpoints must both be from the
                # selected connection tier for this combo.
                if not self._pair_matches_tier(a_iata, b_iata, target_tier):
                    continue
                for gid in _shuffled(group_ids, rng):
                    adj, _edge_airlines = self._subgraph(gid)
                    if a_iata not in adj or b_iata not in adj:
                        continue
                    if not self._are_connected(adj, a_iata, b_iata):
                        continue
                    # Optional anti-trivial filter: only applied *after*
                    # full reachability is confirmed for the sampled pair.
                    if self._is_trivial_via_anchor_hub(gid, a_iata, b_iata):
                        continue
                    return self._materialize(gid, a_iata, b_iata, mode)
            if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                break
        raise RuntimeError(
            f"could not build a level-{level} question within {MAX_ATTEMPTS_PER_REQUEST} attempts"
        )

    def sample_many(
        self,
        level: int,
        n: int,
        mode: Mode = "normal",
        rng: random.Random | None = None,
    ) -> list[Question]:
        rng = rng or random
        return [self.sample(level, mode=mode, rng=rng) for _ in range(n)]

    def _sample_airport_pair(
        self,
        pool: list[str],
        rng: random.Random,
    ) -> tuple[str, str] | None:
        if len(pool) < 2:
            return None
        return tuple(rng.sample(pool, 2))  # type: ignore[return-value]

    def _pair_matches_tier(self, a: str, b: str, tier: int) -> bool:
        a_tier = int(self._airport_meta.get(a, {}).get("tier", -1))
        b_tier = int(self._airport_meta.get(b, {}).get("tier", -1))
        return a_tier == tier and b_tier == tier

    def _are_connected(self, adj: dict[str, set[str]], a: str, b: str) -> bool:
        return graph.bfs_distance(adj, a, b) is not None

    def sample_alt(
        self,
        conn_tier: int | tuple[int, int],
        n_stops: int | tuple[int, int],
        obscurity: int | tuple[int, int],
        mode: Mode = "normal",
        rng: random.Random | None = None,
    ) -> Question:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")
        rng = rng or random

        conn_tier_lo, conn_tier_hi = self._to_range(conn_tier)
        n_stops_lo, n_stops_hi = self._to_range(n_stops)
        obscurity_lo, obscurity_hi = self._to_range(obscurity)
        if conn_tier_lo < 1 or conn_tier_hi > 10:
            raise ValueError("conn_tier must be in [1, 10]")
        if obscurity_lo < 1 or obscurity_hi > 3:
            raise ValueError("obscurity must be in [1, 3]")
        if n_stops_lo < 0:
            raise ValueError("n_stops must be >= 0")

        combos = [o for o in range(obscurity_lo, obscurity_hi + 1)]
        tier_values = list(range(conn_tier_lo, conn_tier_hi + 1))
        min_hops = n_stops_lo + 1
        max_hops = n_stops_hi + 1

        candidate_questions: list[tuple[str, str, str, int, int]] = []
        for obscurity_val in _shuffled(combos, rng):
            eligible_groups = [
                g["id"]
                for g in self.dataset.groups
                if obscurity_lo <= int(g.get("obscurity", 99)) <= obscurity_val
            ]
            if not eligible_groups:
                continue

            for gid in _shuffled(eligible_groups, rng):
                adj, _edge_airlines = self._subgraph(gid)
                tier_adj = self._tier_filtered_adj(adj, tier_values)
                tier_nodes = [
                    code
                    for tier in tier_values
                    for code in self._nodes_by_tier.get(tier, [])
                    if code in tier_adj
                ]
                if len(tier_nodes) < 2:
                    continue
                pair, hops = self._pair_with_hop_range(
                    tier_adj,
                    tier_nodes,
                    min_hops=min_hops,
                    max_hops=max_hops,
                    rng=rng,
                )
                if pair is None:
                    continue
                a_iata, b_iata = pair
                conn_tier_for_question = max(
                    int(self._airport_meta.get(a_iata, {}).get("tier", conn_tier_hi)),
                    int(self._airport_meta.get(b_iata, {}).get("tier", conn_tier_hi)),
                )
                candidate_questions.append(
                    (gid, a_iata, b_iata, conn_tier_for_question, hops)
                )
                if len(candidate_questions) >= 64:
                    break
            if len(candidate_questions) >= 64:
                break

        if candidate_questions:
            gid, a_iata, b_iata, conn_tier_val, target_hops = rng.choice(candidate_questions)
            return self._materialize_alt(gid, a_iata, b_iata, conn_tier_val, target_hops, mode)

        raise RuntimeError(
            "could not find a matching airport pair for any eligible airline group"
        )

    def _to_range(self, value: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, int):
            return value, value
        lo, hi = value
        if lo > hi:
            raise ValueError("range start must be <= range end")
        return lo, hi

    def _pair_with_hop_range(
        self,
        adj: dict[str, set[str]],
        candidates: list[str],
        min_hops: int,
        max_hops: int,
        rng: random.Random,
    ) -> tuple[tuple[str, str] | None, int]:
        candidate_set = set(candidates)
        pairs_by_hops: dict[int, set[tuple[str, str]]] = {}
        for src in candidates:
            dist = graph.single_source_distances(adj, src, max_depth=max_hops)
            matches = [
                (dst, d)
                for dst, d in dist.items()
                if min_hops <= d <= max_hops and dst in candidate_set and dst != src
            ]
            for dst, d in matches:
                pair = (src, dst) if src < dst else (dst, src)
                pairs_by_hops.setdefault(d, set()).add(pair)
        if not pairs_by_hops:
            return None, 0
        hop_bucket = rng.choice(sorted(pairs_by_hops.keys()))
        return rng.choice(list(pairs_by_hops[hop_bucket])), hop_bucket

    def _tier_filtered_adj(
        self,
        adj: dict[str, set[str]],
        conn_tiers: list[int],
    ) -> dict[str, set[str]]:
        tier_nodes = {
            code
            for tier in conn_tiers
            for code in self._nodes_by_tier.get(tier, [])
        }
        out: dict[str, set[str]] = {}
        for node in tier_nodes:
            if node not in adj:
                continue
            nbrs = {nbr for nbr in adj[node] if nbr in tier_nodes}
            if nbrs:
                out[node] = nbrs
        return out

    # -------------------------------------------------------- answer handling

    def validate_answer(
        self,
        question: Question,
        legs: list[Leg],
    ) -> ValidationResult:
        if not legs:
            return ValidationResult(False, "no legs provided")
        if len(legs) < 2:
            return ValidationResult(
                False,
                "indirect routings only - you must have at least two legs",
            )

        gid = question.group_id
        member_codes = graph.group_airline_set(self._groups_by_id[gid])
        _adj, edge_airlines = self._subgraph(gid)

        first_src = legs[0].src.upper()
        last_dst = legs[-1].dst.upper()
        if first_src != question.a or last_dst != question.b:
            return ValidationResult(
                False,
                f"routing must start at {question.a} and end at {question.b}; "
                f"got {first_src} -> ... -> {last_dst}",
            )

        for stopover in [l.dst.upper() for l in legs[:-1]]:
            if stopover not in _adj:
                return ValidationResult(
                    False,
                    f"invalid stopover airport for {question.group_name}: {stopover}",
                )

        for i, leg in enumerate(legs):
            src = leg.src.upper()
            dst = leg.dst.upper()
            airline = leg.airline.upper()
            if src == dst:
                return ValidationResult(False, f"leg {i + 1}: src and dst are identical ({src})")
            if i > 0 and src != legs[i - 1].dst.upper():
                return ValidationResult(
                    False,
                    f"leg {i + 1}: starts at {src} but previous leg ended at "
                    f"{legs[i - 1].dst.upper()}",
                )
            key = (src, dst) if src < dst else (dst, src)
            airlines_on_edge = edge_airlines.get(key) or []
            airline_label = self._airline_name.get(airline, airline)
            if airline not in member_codes:
                return ValidationResult(
                    False,
                    f"leg {i + 1}: {airline_label} ({airline}) is not in "
                    f"{question.group_name}",
                )
            if airline not in airlines_on_edge:
                # Either this airline doesn't operate this route, or no member
                # of the group does. Tailor the message accordingly.
                if not airlines_on_edge:
                    return ValidationResult(
                        False,
                        f"leg {i + 1}: no routes available on {airline_label} "
                        f"({airline}) for {src} to {dst}",
                    )
                operators = ", ".join(
                    f"{self._airline_name.get(c, c)} ({c})" for c in airlines_on_edge
                )
                return ValidationResult(
                    False,
                    f"leg {i + 1}: no routes available on {airline_label} "
                    f"({airline}) for {src} to {dst} — try {operators}",
                )

        return ValidationResult(True, None, stops=len(legs) - 1)

    def map_data_for_question(self, question: Question) -> dict[str, Any]:
        gid = question.group_id
        group = self._groups_by_id[gid]
        adj, edge_airlines = self._subgraph(gid)
        airports = set(adj.keys())
        airports.add(question.a)
        airports.add(question.b)
        group_airlines = sorted(graph.group_airline_set(group))

        served_by_airport: dict[str, set[str]] = {code: set() for code in airports}
        for (a, b), airlines in edge_airlines.items():
            if a in served_by_airport:
                served_by_airport[a].update(airlines)
            if b in served_by_airport:
                served_by_airport[b].update(airlines)

        out_airports: list[dict[str, Any]] = []
        for code in sorted(airports):
            meta = self._airport_meta.get(code, {})
            out_airports.append(
                {
                    "iata": code,
                    "icao": meta.get("icao"),
                    "name": meta.get("name"),
                    "city": meta.get("city"),
                    "country": meta.get("country"),
                    "lat": meta.get("lat"),
                    "lng": meta.get("lon"),
                    "servedByAirlines": sorted(served_by_airport.get(code, set())),
                }
            )
        return {
            "group_id": gid,
            "group_name": group["name"],
            "startAirport": question.a,
            "endAirport": question.b,
            "airlines": [
                {"iata": c, "name": self._airline_name.get(c, c)} for c in group_airlines
            ],
            "airports": out_airports,
        }

    def valid_airlines_for_leg(
        self,
        group_id: str,
        src: str,
        dst: str,
    ) -> list[dict[str, str]]:
        src, dst = src.upper(), dst.upper()
        if group_id not in self._groups_by_id:
            raise ValueError(f"unknown group: {group_id}")
        _adj, edge_airlines = self._subgraph(group_id)
        key = (src, dst) if src < dst else (dst, src)
        codes = edge_airlines.get(key, [])
        return [{"iata": c, "name": self._airline_name.get(c, c)} for c in codes]

    def example_answer(
        self,
        question: Question,
        rng: random.Random | None = None,
    ) -> list[Leg]:
        gid = question.group_id
        adj, edge_airlines = self._subgraph(gid)
        _routes, anchor_hubs = self._anchor_data(gid)
        rng = rng or random

        path = graph.shortest_path(adj, question.a, question.b)
        if path is None:
            raise RuntimeError(
                f"unreachable in group {gid}: {question.a} <-> {question.b}"
            )

        if len(path) == 2:
            shared = sorted(adj[question.a] & adj[question.b] - {question.a, question.b})
            non_hub = [x for x in shared if x not in anchor_hubs]
            pool = non_hub or shared
            if pool:
                hop = rng.choice(pool)
                path = [question.a, hop, question.b]
            else:
                candidates = [x for x in adj[question.a] if x != question.b]
                if not candidates:
                    raise RuntimeError("cannot construct indirect routing")
                hop = rng.choice(sorted(candidates))
                tail = graph.shortest_path(adj, hop, question.b)
                if tail is None:
                    raise RuntimeError("cannot construct indirect routing")
                path = [question.a, *tail]

        legs: list[Leg] = []
        for src, dst in zip(path, path[1:]):
            key = (src, dst) if src < dst else (dst, src)
            airlines = edge_airlines[key]
            airline = rng.choice(airlines)
            legs.append(Leg(src=src, dst=dst, airline=airline))
        return legs

    # ------------------------------------------------------- k-shortest routes

    def _edge_weights_km(
        self,
        adj: dict[str, set[str]],
    ) -> dict[tuple[str, str], float]:
        """Compute great-circle distances for every edge in this subgraph."""
        weights: dict[tuple[str, str], float] = {}
        for u, neighbours in adj.items():
            for v in neighbours:
                if u >= v:
                    continue
                a_meta = self._airport_meta.get(u)
                b_meta = self._airport_meta.get(v)
                if not a_meta or not b_meta:
                    continue
                lat1, lon1 = a_meta.get("lat"), a_meta.get("lon")
                lat2, lon2 = b_meta.get("lat"), b_meta.get("lon")
                if None in (lat1, lon1, lat2, lon2):
                    weights[(u, v)] = float("inf")
                    continue
                weights[(u, v)] = graph.haversine_km(lat1, lon1, lat2, lon2)
        return weights

    def k_shortest_routes(
        self,
        question: Question,
        k: int = 11,
    ) -> list[dict[str, Any]]:
        """Top-K routings between question.a and question.b in this group's
        subgraph, ranked by fewest stops, then total great-circle distance.
        Routes shorter than 2 legs are excluded (game rule: indirect only)."""
        gid = question.group_id
        adj, edge_airlines = self._subgraph(gid)
        weights = self._edge_weights_km(adj)
        paths = graph.k_shortest_paths(
            adj,
            weights,
            question.a,
            question.b,
            k=k,
            min_legs=2,
            rank_by="stops_distance",
        )
        out: list[dict[str, Any]] = []
        for cost_km, path in paths:
            legs: list[dict[str, Any]] = []
            for u, v in zip(path, path[1:]):
                key = (u, v) if u < v else (v, u)
                airlines = edge_airlines.get(key) or []
                # Pick the first airline alphabetically for stable display.
                airline = airlines[0] if airlines else ""
                u_meta = self._airport_meta.get(u, {})
                v_meta = self._airport_meta.get(v, {})
                legs.append(
                    {
                        "src": u,
                        "dst": v,
                        "airline": airline,
                        "airline_name": self._airline_name.get(airline, airline),
                        "src_lat": u_meta.get("lat"),
                        "src_lon": u_meta.get("lon"),
                        "dst_lat": v_meta.get("lat"),
                        "dst_lon": v_meta.get("lon"),
                        "alt_airlines": airlines,
                    }
                )
            out.append(
                {
                    "stops": len(path) - 2,
                    "total_km": round(cost_km),
                    "path": path,
                    "legs": legs,
                }
            )
        return out

    def _materialize(self, gid: str, a: str, b: str, mode: Mode) -> Question:
        # Canonicalize (a < b).
        if a > b:
            a, b = b, a
        group = self._groups_by_id[gid]
        adj, _ = self._subgraph(gid)
        dist = graph.bfs_distance(adj, a, b)
        edge_count = dist or 1
        min_stops = max(1, edge_count - 1)
        a_meta = self._airport_meta.get(a, {})
        b_meta = self._airport_meta.get(b, {})
        obscurity = int(group["obscurity"])
        conn_tier = max(int(a_meta.get("tier", 5)), int(b_meta.get("tier", 5)))
        return Question(
            group_id=gid,
            group_name=group["name"],
            obscurity=obscurity,
            a=a,
            b=b,
            a_name=a_meta.get("name"),
            b_name=b_meta.get("name"),
            a_city=a_meta.get("city"),
            b_city=b_meta.get("city"),
            a_country=a_meta.get("country"),
            b_country=b_meta.get("country"),
            level=difficulty_level(obscurity, conn_tier),
            conn_tier=conn_tier,
            min_stops=min_stops,
            direct_available=graph.has_direct(adj, a, b),
            mode=mode,
        )

    def _materialize_alt(
        self,
        gid: str,
        a: str,
        b: str,
        conn_tier: int,
        hops: int,
        mode: Mode,
    ) -> Question:
        if a > b:
            a, b = b, a
        group = self._groups_by_id[gid]
        a_meta = self._airport_meta.get(a, {})
        b_meta = self._airport_meta.get(b, {})
        obscurity = int(group["obscurity"])
        return Question(
            group_id=gid,
            group_name=group["name"],
            obscurity=obscurity,
            a=a,
            b=b,
            a_name=a_meta.get("name"),
            b_name=b_meta.get("name"),
            a_city=a_meta.get("city"),
            b_city=b_meta.get("city"),
            a_country=a_meta.get("country"),
            b_country=b_meta.get("country"),
            level=difficulty_level(obscurity, conn_tier),
            conn_tier=conn_tier,
            min_stops=max(1, hops - 1),
            direct_available=(hops == 1),
            mode=mode,
        )

    def airport_name(self, iata: str) -> str:
        meta = self._airport_meta.get(iata)
        return meta["name"] if meta else iata

    def airport_city(self, iata: str) -> str | None:
        meta = self._airport_meta.get(iata)
        return meta.get("city") if meta else None

    def airline_name(self, code: str) -> str:
        return self._airline_name.get(code, code)

    def question_for(
        self,
        group_id: str,
        a: str,
        b: str,
        mode: Mode = "normal",
    ) -> Question:
        a, b = (a.upper(), b.upper())
        if a > b:
            a, b = b, a
        if group_id not in self._groups_by_id:
            raise ValueError(f"unknown group: {group_id}")
        adj, _ = self._subgraph(group_id)
        if a not in adj or b not in adj:
            raise ValueError(f"{a} and/or {b} are not in subgraph for {group_id}")
        return self._materialize(group_id, a, b, mode)


T = TypeVar("T")


def _shuffled(seq: list[T], rng: random.Random) -> list[T]:
    out = list(seq)
    rng.shuffle(out)
    return out
