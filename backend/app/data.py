"""Load the static dataset built by the data pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# backend/app/data.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"


@dataclass(frozen=True)
class Dataset:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    airlines: list[dict[str, Any]]
    groups: list[dict[str, Any]]


def load_dataset(data_dir: Path | None = None) -> Dataset:
    d = data_dir or DEFAULT_DATA_DIR
    return Dataset(
        nodes=json.loads((d / "nodes.json").read_text()),
        edges=json.loads((d / "edges.json").read_text()),
        airlines=json.loads((d / "airlines.json").read_text()),
        groups=json.loads((d / "groups.json").read_text()),
    )
