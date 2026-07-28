# Distil backend

FastAPI orchestration service for task-conditioned model and code-tool distillation.

## Run locally

```bash
cd backend
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000
```

The default configuration uses SQLite and a deterministic simulated
teacher/trainer, so the full pipeline works without GPUs or credentials. Each
stage waits 1.5 seconds by default (`PIPELINE_DELAY=0` is useful for tests).
Set `TEACHER_API_BASE` and `TEACHER_API_KEY` to use an OpenAI-compatible
teacher endpoint. For real local training, run `uv sync --extra train` and set
`DISTIL_USE_TRANSFORMERS=1`, `TEACHER_LOCAL_MODEL`, and optionally
`DISTIL_STUDENT_MODEL`. Real customer-support triage runs generate 120
diverse teacher-drafted tickets by default, use a consistent system/user/
assistant chat format, mask prompt tokens from the loss, and train for three
epochs. Tune this with `DISTIL_EXAMPLES`, `DISTIL_EPOCHS`, `DISTIL_LORA_R`,
and `DISTIL_LEARNING_RATE`. The reusable qualitative harness is
`uv run python scripts/qualitative_eval.py`.

When running the Next.js frontend locally, its `/api/v1/...` requests are
proxied to this service at `http://localhost:8000`. Set the frontend's
`BACKEND_URL` environment variable if the API runs elsewhere.

## API

- `GET /api/v1/teachers`
- `POST /api/v1/runs`
- `GET /api/v1/runs` and `GET /api/v1/runs/{id}`
- `GET /api/v1/runs/{id}/artifacts`
- `GET /api/v1/runs/{id}/artifacts/download`

Run tests with `uv run pytest`; lint with `uv run ruff check app tests`.
