# Backend

Reusable Python services for Jack's Flight Club.

- `app.data` — loads the static dataset from `../data/processed/`.
- `app.graph` — builds per-group subgraphs and runs BFS for shortest-paths.
- `app.difficulty` — computes connectivity tiers and difficulty levels.
- `app.questions` — `QuestionBank` returns a random `(group, A, B)` question for a given level.
- `app.cli` — quick CLI for testing without a webserver.

The FastAPI layer (added later) will import `QuestionBank` and expose endpoints.

## Setup
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## CLI

```bash
python -m app.cli histogram                # counts of valid questions per level
python -m app.cli example --level 1        # one random level-1 question
python -m app.cli example --level 7 -n 5   # five random level-7 questions
python -m app.cli example --level 10 --seed 42
```
