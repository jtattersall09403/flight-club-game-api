"""Lazy question generator.

A request for a level-N question runs roughly:

  1. Look at `feasible_combos(N)` -> list of (obscurity, conn_tier) pairs
     whose formula produces level N.
  2. Shuffle that list. For each (O, C) combo:
       a. Pick a random "anchor airport" A whose tier == C (this guarantees
          max(tier_a, tier_b) >= C; we keep B's tier <= C so the max is
          exactly C).
       b. Shuffle groups whose `obscurity == O`.
       c. For each candidate group, build (or fetch from cache) its subgraph
          and check whether A is in it. If so, BFS one step to find a
          neighbour B (or two-step neighbour) with tier <= C and a valid
          indirect routing back to A. Return the first match.
  3. If no question found within MAX_ATTEMPTS, raise.

Subgraph builds happen on demand and are LRU-cached, so the second sample for
the same group is cheap. Building one subgraph is ~5-30ms for our dataset.
"""
from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from typing import Any, Literal

from . import graph
from .data import Dataset
from .difficulty import LEVEL_MAX, LEVEL_MIN, difficulty_level, feasible_combos

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
    Per-sample cost: typically one subgraph build (~5-30ms) for the chosen
    group, then O(degree) random pick + O(BFS) verification.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._airport_meta: dict[str, dict[str, Any]] = {n["iata"]: n for n in dataset.nodes}
        self._groups_by_id: dict[str, dict[str, Any]] = {g["id"]: g for g in dataset.groups}
        self._airline_name: dict[str, str] = {a["iata"]: a["name"] for a in dataset.airlines}

        # Index nodes by tier (for fast random pick of "anchor airport").
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

        # Try combos in random order.
        attempts = 0
        for combo in _shuffled(combos, rng):
            obscurity, target_tier = combo
            group_ids = self._groups_by_obscurity.get(obscurity, [])
            if not group_ids:
                continue
            anchor_pool = self._nodes_by_tier.get(target_tier, [])
            if not anchor_pool:
                continue
            for _ in range(40):
                attempts += 1
                if attempts > MAX_ATTEMPTS_PER_REQUEST:
                    break
                a_iata = rng.choice(anchor_pool)
                # Shuffle groups for this combo.
                for gid in _shuffled(group_ids, rng):
                    adj, _edge_airlines = self._subgraph(gid)
                    if a_iata not in adj:
                        continue
                    # Pick neighbour or 2-hop neighbour with tier <= target_tier.
                    b_iata = self._find_partner(adj, a_iata, target_tier, gid, rng)
                    if b_iata is None:
                        continue
                    return self._materialize(gid, a_iata, b_iata, mode)
            if attempts > MAX_ATTEMPTS_PER_REQUEST:
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

    def _find_partner(
        self,
        adj: dict[str, set[str]],
        a: str,
        target_tier: int,
        group_id: str,
        rng: random.Random,
    ) -> str | None:
        """Pick B in this subgraph with tier <= target_tier, A != B, that
        is reachable from A and is not a trivial-via-anchor-hub pair."""
        # Reachable nodes: BFS from A within this group's subgraph.
        # We don't need full reachability — sample randomly from neighbours and
        # 2-hop neighbours to keep BFS cheap.
        candidates = list(adj[a])
        # Add 2-hop neighbours so we get pairs that need an intermediate.
        two_hop: set[str] = set()
        for n in candidates:
            two_hop.update(adj[n])
        two_hop -= {a}
        two_hop -= set(candidates)
        candidates.extend(two_hop)
        rng.shuffle(candidates)
        for b in candidates:
            if b == a:
                continue
            meta = self._airport_meta.get(b)
            if not meta or int(meta["tier"]) > target_tier:
                continue
            if self._is_trivial_via_anchor_hub(group_id, a, b):
                continue
            return b
        return None

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
        endpoints = {first_src, last_dst}
        if endpoints != {question.a, question.b}:
            return ValidationResult(
                False,
                f"routing must start at {question.a} or {question.b} and end at "
                f"the other; got {first_src} -> ... -> {last_dst}",
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
        subgraph, ranked by total great-circle distance flown. Routes shorter
        than 2 legs are excluded (game rule: indirect only)."""
        gid = question.group_id
        adj, edge_airlines = self._subgraph(gid)
        weights = self._edge_weights_km(adj)
        paths = graph.k_shortest_paths(
            adj, weights, question.a, question.b, k=k, min_legs=2
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


def _shuffled(seq: list, rng: random.Random) -> list:
    out = list(seq)
    rng.shuffle(out)
    return out
