"""Per-group subgraphs and BFS helpers."""
from __future__ import annotations

from collections import deque
from typing import Any

# Edge inclusion rule:
#   An edge belongs to a group's subgraph if at least one of its airlines is in
#   the group's airline list.


def group_airline_set(group: dict[str, Any]) -> set[str]:
    """Airlines whose flights count as in-group."""
    return set(group["airlines"])


def _canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def build_subgraph(
    edges: list[dict], group: dict
) -> tuple[dict[str, set[str]], dict[tuple[str, str], list[str]]]:
    """Return (adjacency, edge_airlines).

    `adjacency`     : iata -> set of neighbour iatas in this group.
    `edge_airlines` : (a, b) sorted tuple -> list of in-group airline codes that
                      operate that pair (in either direction).
    """
    member_codes = group_airline_set(group)
    anchor = group.get("anchor")
    exclude_anchor_legs = group.get("type") == "partner_program" and isinstance(anchor, str) and bool(anchor)
    adj: dict[str, set[str]] = {}
    edge_airlines: dict[tuple[str, str], list[str]] = {}
    for e in edges:
        if exclude_anchor_legs and anchor in e["airlines"]:
            continue
        in_group = [c for c in e["airlines"] if c in member_codes]
        if not in_group:
            continue
        a, b = e["a"], e["b"]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        edge_airlines[_canon(a, b)] = sorted(in_group)
    return adj, edge_airlines


def build_all_subgraphs(
    edges: list[dict], groups: list[dict]
) -> tuple[
    dict[str, dict[str, set[str]]],
    dict[str, dict[tuple[str, str], list[str]]],
]:
    """Build adjacency and edge-airline maps for every group."""
    adjs: dict[str, dict[str, set[str]]] = {}
    edge_airlines: dict[str, dict[tuple[str, str], list[str]]] = {}
    for g in groups:
        a, ea = build_subgraph(edges, g)
        adjs[g["id"]] = a
        edge_airlines[g["id"]] = ea
    return adjs, edge_airlines


def connected_components(adj: dict[str, set[str]]) -> dict[str, int]:
    """Returns {iata: component_id}. Component ids are arbitrary ints."""
    comp: dict[str, int] = {}
    cid = 0
    for start in adj:
        if start in comp:
            continue
        # BFS from start
        comp[start] = cid
        q = deque([start])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in comp:
                    comp[y] = cid
                    q.append(y)
        cid += 1
    return comp


def bfs_distance(adj: dict[str, set[str]], src: str, dst: str) -> int | None:
    """Number of edges on the shortest path src->dst, or None if unreachable."""
    if src == dst:
        return 0
    if src not in adj or dst not in adj:
        return None
    seen = {src: 0}
    q = deque([src])
    while q:
        x = q.popleft()
        d = seen[x]
        for y in adj[x]:
            if y in seen:
                continue
            if y == dst:
                return d + 1
            seen[y] = d + 1
            q.append(y)
    return None


def shortest_path(adj: dict[str, set[str]], src: str, dst: str) -> list[str] | None:
    """Return the list of nodes on a shortest src->dst path, or None."""
    if src == dst:
        return [src]
    if src not in adj or dst not in adj:
        return None
    parent: dict[str, str] = {src: src}
    q = deque([src])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y in parent:
                continue
            parent[y] = x
            if y == dst:
                # reconstruct
                path = [y]
                while path[-1] != src:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            q.append(y)
    return None


def has_direct(adj: dict[str, set[str]], a: str, b: str) -> bool:
    return b in adj.get(a, ())


# ---------------------------------------------------------------- great-circle

import heapq
import math


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _path_cost(
    path: list[str],
    weights: dict[tuple[str, str], float],
) -> float:
    total = 0.0
    for u, v in zip(path, path[1:]):
        total += weights[_canon(u, v)]
    return total


def _dijkstra(
    adj: dict[str, set[str]],
    weights: dict[tuple[str, str], float],
    src: str,
    dst: str,
    removed_edges: set[tuple[str, str]] | None = None,
    removed_nodes: set[str] | None = None,
) -> tuple[float, list[str]] | None:
    if src == dst:
        return 0.0, [src]
    if src not in adj or dst not in adj:
        return None
    removed_edges = removed_edges or set()
    removed_nodes = removed_nodes or set()
    dist: dict[str, float] = {src: 0.0}
    prev: dict[str, str] = {}
    pq: list[tuple[float, str]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            path = [u]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            path.reverse()
            return d, path
        if d > dist.get(u, math.inf):
            continue
        for v in adj.get(u, ()):
            if v in removed_nodes:
                continue
            edge = _canon(u, v)
            if edge in removed_edges:
                continue
            w = weights[edge]
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None


def k_shortest_paths(
    adj: dict[str, set[str]],
    weights: dict[tuple[str, str], float],
    src: str,
    dst: str,
    k: int,
    min_legs: int = 1,
    rank_by: str = "distance",
) -> list[tuple[float, list[str]]]:
    """K-shortest simple paths with configurable ranking.

    rank_by="distance": Yen's algorithm for total edge weight.
    rank_by="stops_distance": ordered by (legs, total_distance).
    """
    if rank_by not in {"distance", "stops_distance"}:
        raise ValueError(f"unsupported rank_by: {rank_by!r}")

    if rank_by == "stops_distance":
        if src not in adj or dst not in adj:
            return []
        results: list[tuple[float, list[str]]] = []
        pq: list[tuple[int, float, int, list[str]]] = [(0, 0.0, 0, [src])]
        counter = 0
        pops = 0
        max_pops = max(1000, k * 500)
        max_legs = min(len(adj) - 1, max(min_legs + 6, min_legs))
        while pq and len(results) < k and pops < max_pops:
            pops += 1
            legs, dist, _, path = heapq.heappop(pq)
            node = path[-1]
            if node == dst and legs >= min_legs:
                results.append((dist, path))
                continue
            if legs >= max_legs:
                continue
            for nxt in sorted(adj.get(node, ())):
                if nxt in path:
                    continue
                edge = _canon(node, nxt)
                w = weights.get(edge)
                if w is None:
                    continue
                counter += 1
                heapq.heappush(pq, (legs + 1, dist + w, counter, path + [nxt]))
        return results

    first = _dijkstra(adj, weights, src, dst)
    if not first:
        return []
    A: list[tuple[float, list[str]]] = [first]
    candidates: list[tuple[float, int, list[str]]] = []
    counter = 0

    def qualifying() -> int:
        return sum(1 for _, p in A if len(p) - 1 >= min_legs)

    # Cap iterations so a tiny dense graph doesn't loop forever.
    max_iter = max(50, k * 20)
    while qualifying() < k and max_iter > 0:
        max_iter -= 1
        last_path = A[-1][1]
        for i in range(len(last_path) - 1):
            spur_node = last_path[i]
            root_path = last_path[: i + 1]
            removed_edges: set[tuple[str, str]] = set()
            for _, p in A:
                if len(p) > i and p[: i + 1] == root_path:
                    removed_edges.add(_canon(p[i], p[i + 1]))
            removed_nodes = set(root_path[:-1])
            spur = _dijkstra(adj, weights, spur_node, dst, removed_edges, removed_nodes)
            if spur is None:
                continue
            _spur_cost, spur_path = spur
            total_path = root_path + spur_path[1:]
            total_cost = _path_cost(total_path, weights)
            if any(tp == total_path for _, _, tp in candidates):
                continue
            counter += 1
            heapq.heappush(candidates, (total_cost, counter, total_path))
        if not candidates:
            break
        cost, _, path = heapq.heappop(candidates)
        A.append((cost, path))

    return [(c, p) for c, p in A if len(p) - 1 >= min_legs][:k]
