"""Persistent leaderboard backed by Upstash Redis (HTTP REST API).

Data model (per mode):
  ZSET  lb:scores:<mode>   member=name, score=high_score (int)
  HASH  lb:ts:<mode>       field=name,  value=epoch_seconds last submitted
  SET   lb:names           every name ever submitted (any mode) for autocomplete

Submit semantics:
  - score is upserted with ZADD GT so it only increases.
  - timestamp is always refreshed on submit (extends the 90-day window).

Read semantics:
  - top(mode, limit, days): ZREVRANGE WITHSCORES, then HMGET timestamps,
    drop entries older than `days`.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

WINDOW_DAYS = 90
MAX_NAME_LEN = 24
NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]{1,%d}$" % MAX_NAME_LEN)

VALID_MODES = ("normal", "hard")


class LeaderboardError(RuntimeError):
    pass


def _config() -> tuple[str, str] | None:
    url = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
    tok = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
    if not url or not tok:
        return None
    return url, tok


def is_configured() -> bool:
    return _config() is not None


def _post(commands: list[list[Any]]) -> Any:
    """Run a Redis pipeline via Upstash's /pipeline endpoint."""
    cfg = _config()
    if cfg is None:
        raise LeaderboardError("leaderboard not configured")
    url, tok = cfg
    r = httpx.post(
        f"{url}/pipeline",
        json=commands,
        headers={"Authorization": f"Bearer {tok}"},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


def normalise_name(name: str) -> str:
    return " ".join(name.strip().split())[:MAX_NAME_LEN]


def valid_name(name: str) -> bool:
    return bool(NAME_RE.match(name))


def submit(name: str, score: int, mode: str) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise LeaderboardError(f"invalid mode: {mode}")
    name = normalise_name(name)
    if not valid_name(name):
        raise LeaderboardError("name must be 1-24 chars (letters, digits, space, _-.)")
    if not isinstance(score, int) or score < 0:
        raise LeaderboardError("score must be a non-negative int")

    now = int(time.time())
    res = _post([
        ["ZADD", f"lb:scores:{mode}", "GT", "CH", str(score), name],
        ["HSET", f"lb:ts:{mode}", name, str(now)],
        ["SADD", "lb:names", name],
        ["ZSCORE", f"lb:scores:{mode}", name],
    ])
    # Each pipeline element is {"result": ...} or {"error": ...}.
    changed = int((res[0] or {}).get("result") or 0)
    cur = (res[3] or {}).get("result")
    cur_score = int(float(cur)) if cur is not None else score
    return {"name": name, "mode": mode, "score": cur_score, "improved": bool(changed)}


def top(mode: str, limit: int = 20, days: int = WINDOW_DAYS) -> list[dict[str, Any]]:
    if mode not in VALID_MODES:
        raise LeaderboardError(f"invalid mode: {mode}")
    # Pull a few extra so the time-window filter doesn't leave us short.
    fetch_n = max(limit * 3, 60)
    res = _post([
        ["ZRANGE", f"lb:scores:{mode}", "0", str(fetch_n - 1), "REV", "WITHSCORES"],
    ])
    raw = (res[0] or {}).get("result") or []
    if not raw:
        return []
    # raw is [name, score, name, score, ...]
    pairs: list[tuple[str, int]] = [
        (raw[i], int(float(raw[i + 1]))) for i in range(0, len(raw), 2)
    ]
    names = [p[0] for p in pairs]
    ts_res = _post([["HMGET", f"lb:ts:{mode}", *names]])
    tss = (ts_res[0] or {}).get("result") or [None] * len(names)
    cutoff = int(time.time()) - days * 86400
    out = []
    for (name, score), ts in zip(pairs, tss):
        ts_int = int(ts) if ts else 0
        if ts_int < cutoff:
            continue
        out.append({"name": name, "score": score, "ts": ts_int, "mode": mode})
        if len(out) >= limit:
            break
    return out


def names_matching(prefix: str, limit: int = 8) -> list[str]:
    """Names whose case-insensitive form starts with prefix. Cheap brute force
    over the full names set (small enough)."""
    prefix = prefix.strip().lower()
    if not prefix:
        return []
    res = _post([["SMEMBERS", "lb:names"]])
    members = (res[0] or {}).get("result") or []
    starts = [n for n in members if n.lower().startswith(prefix)]
    starts.sort(key=lambda s: s.lower())
    return starts[:limit]


def name_scores(name: str) -> dict[str, int | None]:
    """Current high score for `name` in each mode (None if not present)."""
    name = normalise_name(name)
    res = _post([
        ["ZSCORE", "lb:scores:normal", name],
        ["ZSCORE", "lb:scores:hard", name],
    ])
    def _i(x: Any) -> int | None:
        v = (x or {}).get("result")
        return int(float(v)) if v is not None else None
    return {"normal": _i(res[0]), "hard": _i(res[1])}
