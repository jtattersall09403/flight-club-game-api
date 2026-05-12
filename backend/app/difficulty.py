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

Allowed (conn_tier, obscurity) combinations per level:
  L1:  (1,1)
  L2:  (1,2), (2,1)
  L3:  (1,3), (2,2), (3,1)
  L4:  (2,3), (3,2), (4,1)
  L5:  (3,3), (4,2), (5,1)
  L6:  (4,3), (5,2), (6,1)
  L7:  (5,3), (6,2), (7,1)
  L8:  (6,3), (7,2), (8,1)
  L9:  (7,3), (8,2), (9,1)
  L10: (8,3), (9,2), (9,3), (10,1), (10,2), (10,3)
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
        2: [(2, 1), (1, 2)],
        3: [(3, 1), (2, 2), (1, 3)],
        4: [(3, 2), (2, 3), (1, 4)],
        5: [(3, 3), (2, 4), (1, 5)],
        6: [(3, 4), (2, 5), (1, 6)],
        7: [(3, 5), (2, 6), (1, 7)],
        8: [(3, 6), (2, 7), (1, 8)],
        9: [(3, 7), (2, 8), (1, 9)],
        10: [(3, 8), (2, 9), (3, 9), (1, 10), (2, 10), (3, 10)],
    }
    return combos_by_level[level]


def combo_branches(level: int) -> list[list[tuple[int, int]]]:
    """Branch options per level; one branch should be sampled first each request."""
    if level < LEVEL_MIN or level > LEVEL_MAX:
        return []
    return [[combo] for combo in feasible_combos(level)]
