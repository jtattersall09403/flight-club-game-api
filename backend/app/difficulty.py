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
Obscurity bands are constrained by level:
  L1-3: O1
  L4: O2
  L5: O3
  L6: O4-O5
  L7: O6-O7
  L8: O8
  L9: O9
  L10: O10

"""
from __future__ import annotations

NUM_TIERS = 10
LEVEL_MIN = 1
LEVEL_MAX = 10


def difficulty_level(obscurity: int, conn_tier: int) -> int:
    _ = obscurity
    return max(LEVEL_MIN, min(LEVEL_MAX, conn_tier))




def _allowed_obscurities_for_level(level: int) -> tuple[int, ...]:
    if level <= 3:
        return (1,)
    if level == 4:
        return (2,)
    if level == 5:
        return (3,)
    if level == 6:
        return (4, 5)
    if level == 7:
        return (6, 7)
    if level == 8:
        return (8,)
    if level == 9:
        return (9,)
    return (10,)

def feasible_combos(level: int) -> list[tuple[int, int]]:
    """Allowed (obscurity, conn_tier) combinations for a requested level."""
    if level < LEVEL_MIN or level > LEVEL_MAX:
        return []
    conn_tier = level
    return [(obscurity, conn_tier) for obscurity in _allowed_obscurities_for_level(level)]
