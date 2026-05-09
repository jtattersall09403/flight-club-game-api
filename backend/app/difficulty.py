"""Difficulty rules.

Tiers come pre-baked on each node (data/processed/nodes.json `tier` field),
computed from global degree thresholds in the data pipeline:
  T1: degree >= 100  (mega hubs)
  T2: 50..99
  T3: 20..49
  T4:  5..19
  T5: <5

"""
from __future__ import annotations

NUM_TIERS = 5
LEVEL_MIN = 1
LEVEL_MAX = 10


def difficulty_level(obscurity: int, conn_tier: int) -> int:
    raw = obscurity + conn_tier - 1
    return max(LEVEL_MIN, min(LEVEL_MAX, raw))


def feasible_combos(level: int) -> list[tuple[int, int]]:
    """All (obscurity, conn_tier) combinations whose clamped level == `level`."""
    out: list[tuple[int, int]] = []
    for o in range(1, 11):
        for c in range(1, NUM_TIERS + 1):
            if difficulty_level(o, c) == level:
                out.append((o, c))
    return out
