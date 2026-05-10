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

Levels map linearly to connectivity tier, so level N targets tier N.

"""
from __future__ import annotations

NUM_TIERS = 10
LEVEL_MIN = 1
LEVEL_MAX = 10


def difficulty_level(obscurity: int, conn_tier: int) -> int:
    _ = obscurity
    return max(LEVEL_MIN, min(LEVEL_MAX, conn_tier))


def feasible_combos(level: int) -> list[tuple[int, int]]:
    """All (obscurity, conn_tier) combinations whose clamped level == `level`."""
    out: list[tuple[int, int]] = []
    for o in range(1, 11):
        for c in range(1, NUM_TIERS + 1):
            if difficulty_level(o, c) == level:
                out.append((o, c))
    return out
