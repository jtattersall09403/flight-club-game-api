# Jack's Flight Club — API

FastAPI backend that serves trivia questions, validates routings, and computes
shortest paths over alliance subgraphs.

## Deploy on Render

1. Create a new Web Service from this repo.
2. Render auto-detects `render.yaml` (root-level). Confirm:
   - Root directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`
3. Set env var `CORS_ALLOW_ORIGINS` to your Netlify URL,
   e.g. `https://flight-club.netlify.app`.

## Local dev

```
cd backend
pip install -r requirements.txt
uvicorn app.server:app --host 127.0.0.1 --port 8765
```
