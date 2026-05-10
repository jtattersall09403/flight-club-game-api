"""Difficulty rules.

Tiers come pre-baked on each node (data/processed/nodes.json `tier` field),
computed from global degree thresholds in the data pipeline:
  T1: degree >= 150 (mega hubs)
  T2: 120..149
  T3: 100..119
  T4: 80..99
  T5: 60..79
  T6: 45..59
  T7: 30..44
  T8: 20..29
  T9: 10..19
  T10: <10

Allowed combinations per level:
  L1:  (O1, C1)
  L2:  (O1, C2)
  L3:  (O1, C3)
  L4:  (O1, C4) or (O2-3, C2)
  L5:  (O1, C5) or (O2-3, C3)
  L6:  (O1, C6) or (O4-5, C4)
  L7:  (O2-3, C7) or (O5-6, C5)
  L8:  (O2-3, C8) or (O7-8, C6)
  L9:  (O4-5, C9) or (O8-9, C7)
  L10: (O6-10, C10)
"""
from __future__ import annotations

NUM_TIERS = 10
LEVEL_MIN = 1
LEVEL_MAX = 10


def difficulty_level(obscurity: int, conn_tier: int) -> int:
    _ = obscurity
    return max(LEVEL_MIN, min(LEVEL_MAX, conn_tier))


def feasible_combos(level: int) -> list[tuple[int, int]]:
    """Allowed (obscurity, conn_tier) combinations for a requested level."""
    if level < LEVEL_MIN or level > LEVEL_MAX:
        return []

    combos_by_level: dict[int, list[tuple[int, int]]] = {
        1: [(1, 1)],
        2: [(1, 2)],
        3: [(1, 3)],
        4: [(1, 4), (2, 2), (3, 2)],
        5: [(1, 5), (2, 3), (3, 3)],
        6: [(1, 6), (4, 4), (5, 4)],
        7: [(2, 7), (3, 7), (5, 5), (6, 5)],
        8: [(2, 8), (3, 8), (7, 6), (8, 6)],
        9: [(4, 9), (5, 9), (8, 7), (9, 7)],
        10: [(6, 10), (7, 10), (8, 10), (9, 10), (10, 10)],
    }
    return combos_by_level[level]


def combo_branches(level: int) -> list[list[tuple[int, int]]]:
    """Branch options per level; one branch should be sampled first each request."""
    if level < LEVEL_MIN or level > LEVEL_MAX:
        return []
    branches = {
        1: [[(1, 1)]],
        2: [[(1, 2)]],
        3: [[(1, 3)]],
        4: [[(1, 4)], [(2, 2), (3, 2)]],
        5: [[(1, 5)], [(2, 3), (3, 3)]],
        6: [[(1, 6)], [(4, 4), (5, 4)]],
        7: [[(2, 7), (3, 7)], [(5, 5), (6, 5)]],
        8: [[(2, 8), (3, 8)], [(7, 6), (8, 6)]],
        9: [[(4, 9), (5, 9)], [(8, 7), (9, 7)]],
        10: [[(6, 10), (7, 10), (8, 10), (9, 10), (10, 10)]],
    }
    return branches[level]
