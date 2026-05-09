"""FastAPI HTTP layer for Jack's Flight Club.

Endpoints:
  GET  /api/health
  GET  /api/airports           -> list[{iata, name, city, country}]
  GET  /api/airlines           -> list[{iata, name}]
  POST /api/question           {level, mode, seed?}            -> Question
  POST /api/validate           {group_id, a, b, mode, legs}    -> ValidationResult+points
  POST /api/example            {group_id, a, b, mode, seed?}   -> example legs
  POST /api/hint               {group_id}                      -> {airlines: [{iata,name}]}

The frontend tracks score/level/lives. The server is stateless.

Scoring:
  base    = level * 10
  eff     = min_stops / max(actual_stops, min_stops)  # never > 1
  hint_k  = 0.5 if hint else 1.0
  points  = round(base * eff * hint_k)
"""
from __future__ import annotations

import os
import random
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data import load_dataset
from .questions import VALID_MODES, Leg, QuestionGenerator

# ----------------------------------------------------------------- bootstrap

dataset = load_dataset()
gen = QuestionGenerator(dataset)

app = FastAPI(title="Jack's Flight Club API", version="0.1.0")

# CORS_ALLOW_ORIGINS=https://foo.netlify.app,https://bar.example.com
# Defaults to "*" for local dev.
_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
_allow_origins = (
    ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------- models

Mode = Literal["normal", "hard"]


class QuestionRequest(BaseModel):
    level: int = Field(ge=1, le=10)
    mode: Mode = "normal"
    seed: int | None = None


class LegSpec(BaseModel):
    src: str
    dst: str
    airline: str


class ValidateRequest(BaseModel):
    group_id: str
    a: str
    b: str
    mode: Mode = "normal"
    legs: list[LegSpec]
    hint_used: bool = False
    level: int = Field(ge=1, le=10)


class ExampleRequest(BaseModel):
    group_id: str
    a: str
    b: str
    mode: Mode = "normal"
    seed: int | None = None


class HintRequest(BaseModel):
    group_id: str


# -------------------------------------------------------------------- helpers

def _score(level: int, min_stops: int, actual_stops: int, hint_used: bool) -> int:
    base = level * 10
    actual = max(actual_stops, min_stops, 1)
    eff = min_stops / actual
    hint_k = 0.5 if hint_used else 1.0
    return round(base * eff * hint_k)


# -------------------------------------------------------------------- routes

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "nodes": len(dataset.nodes), "groups": len(dataset.groups)}


@app.get("/api/airports")
def airports() -> list[dict[str, Any]]:
    """Catalog for the autocomplete dropdown."""
    return [
        {
            "iata": n["iata"],
            "name": n.get("name"),
            "city": n.get("city"),
            "country": n.get("country"),
        }
        for n in dataset.nodes
    ]


@app.get("/api/airlines")
def airlines() -> list[dict[str, Any]]:
    return [{"iata": a["iata"], "name": a["name"]} for a in dataset.airlines]


@app.post("/api/question")
def question(req: QuestionRequest) -> dict[str, Any]:
    rng = random.Random(req.seed) if req.seed is not None else random
    try:
        q = gen.sample(req.level, mode=req.mode, rng=rng)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    out = q.to_dict()
    # Add coords for the map.
    out["a_lat"], out["a_lon"] = _coords(q.a)
    out["b_lat"], out["b_lon"] = _coords(q.b)
    return out


@app.post("/api/validate")
def validate(req: ValidateRequest) -> dict[str, Any]:
    try:
        q = gen.question_for(req.group_id, req.a, req.b, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    legs = [Leg(src=l.src.upper(), dst=l.dst.upper(), airline=l.airline.upper()) for l in req.legs]
    result = gen.validate_answer(q, legs)
    points = 0
    if result.valid and result.stops is not None:
        points = _score(req.level, q.min_stops, result.stops, req.hint_used)
    # Coords for plotting whatever the user typed.
    leg_coords = [
        {
            "src": l.src,
            "dst": l.dst,
            "airline": l.airline,
            "src_lat": _coords(l.src)[0],
            "src_lon": _coords(l.src)[1],
            "dst_lat": _coords(l.dst)[0],
            "dst_lon": _coords(l.dst)[1],
        }
        for l in legs
    ]
    return {
        "valid": result.valid,
        "reason": result.reason,
        "stops": result.stops,
        "min_stops": q.min_stops,
        "points": points,
        "legs": leg_coords,
    }


@app.post("/api/example")
def example(req: ExampleRequest) -> dict[str, Any]:
    rng = random.Random(req.seed) if req.seed is not None else random
    try:
        q = gen.question_for(req.group_id, req.a, req.b, mode=req.mode)
        legs = gen.example_answer(q, rng=rng)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "legs": [
            {
                "src": l.src,
                "dst": l.dst,
                "airline": l.airline,
                "airline_name": gen.airline_name(l.airline),
                "src_lat": _coords(l.src)[0],
                "src_lon": _coords(l.src)[1],
                "dst_lat": _coords(l.dst)[0],
                "dst_lon": _coords(l.dst)[1],
            }
            for l in legs
        ],
    }


class RoutesRequest(BaseModel):
    group_id: str
    a: str
    b: str
    mode: Mode = "normal"
    k: int = 11


@app.post("/api/routes")
def routes(req: RoutesRequest) -> dict[str, Any]:
    """Top-K shortest indirect routings by great-circle distance flown."""
    try:
        q = gen.question_for(req.group_id, req.a, req.b, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"routes": gen.k_shortest_routes(q, k=req.k)}


@app.post("/api/hint")
def hint(req: HintRequest) -> dict[str, Any]:
    group = next((g for g in dataset.groups if g["id"] == req.group_id), None)
    if not group:
        raise HTTPException(status_code=404, detail=f"unknown group: {req.group_id}")
    codes = list(group.get("airlines", []))
    anchor = group.get("anchor")
    if anchor and anchor not in codes:
        codes.insert(0, anchor)
    return {
        "group_id": group["id"],
        "group_name": group["name"],
        "anchor": anchor,
        "airlines": [
            {"iata": c, "name": gen.airline_name(c)} for c in codes
        ],
    }


# -------------------------------------------------------------------- helpers

_node_lookup: dict[str, dict[str, Any]] = {n["iata"]: n for n in dataset.nodes}


def _coords(iata: str) -> tuple[float | None, float | None]:
    n = _node_lookup.get(iata.upper())
    if not n:
        return (None, None)
    return (n.get("lat"), n.get("lon"))
