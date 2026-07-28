import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .catalog import TEACHERS, teacher_by_id
from .config import get_settings
from .db import RunRecord, get_session
from .pipeline import run_pipeline
from .schemas import ArtifactResponse, Metrics, RunCreate, RunResponse, TeacherResponse


def teacher_response(teacher) -> TeacherResponse:
    colors = {
        "Meta": "#f59e0b",
        "Alibaba": "#2dd4bf",
        "Mistral AI": "#fb7185",
        "Google": "#a78bfa",
        "DeepSeek": "#60a5fa",
        "Microsoft": "#34d399",
    }
    return TeacherResponse(
        id=teacher.id,
        name=teacher.name,
        params=teacher.params_label,
        params_b=teacher.params_b,
        license=teacher.license,
        family=teacher.family,
        hf_repo_id=teacher.hf_repo_id,
        note=teacher.note,
        color=colors.get(teacher.family, "#f4b761"),
    )


def run_response(run: RunRecord) -> RunResponse:
    teacher = teacher_by_id(run.teacher_id)
    metrics = json.loads(run.metrics_json or "{}")
    metrics.setdefault("compression_ratio", teacher.params_b / run.target_params)
    return RunResponse(
        id=run.id,
        teacher_id=run.teacher_id,
        teacher=f"{teacher.name} {teacher.params_label}",
        task_prompt=run.task_prompt,
        preset=run.preset,
        target_params=run.target_params,
        target_label=run.target_label,
        output_type=run.output_type,
        status=run.status,
        progress=run.progress,
        score=run.score,
        tokens_generated=run.tokens_generated,
        metrics=Metrics(**metrics),
        created_at=run.created_at,
        updated_at=run.updated_at,
        error=run.error,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="Distil API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/teachers", response_model=list[TeacherResponse])
async def list_teachers():
    return [teacher_response(teacher) for teacher in TEACHERS]


@app.post("/api/v1/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(payload: RunCreate, session: Session = Depends(get_session)):
    teacher = teacher_by_id(payload.teacher_id)
    if teacher is None:
        raise HTTPException(status_code=404, detail="Unknown teacher_id")
    if payload.target_params >= teacher.params_b:
        raise HTTPException(status_code=422, detail="target_params must be smaller than the teacher model")
    target_label = f"{payload.target_params:g}B"
    run = RunRecord(
        id=f"run-{uuid.uuid4().hex[:10]}",
        teacher_id=teacher.id,
        task_prompt=payload.task_prompt,
        preset=payload.preset,
        target_params=payload.target_params,
        target_label=target_label,
        output_type=payload.output_type,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    asyncio.create_task(run_pipeline(run.id, teacher))
    return run_response(run)


@app.get("/api/v1/runs", response_model=list[RunResponse])
async def list_runs(session: Session = Depends(get_session)):
    return [run_response(run) for run in session.query(RunRecord).order_by(RunRecord.created_at.desc()).all()]


@app.get("/api/v1/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(RunRecord, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_response(run)


@app.get("/api/v1/runs/{run_id}/artifacts", response_model=ArtifactResponse)
async def get_artifacts(run_id: str, session: Session = Depends(get_session)):
    run = session.get(RunRecord, run_id)
    if run is None or not run.artifact_path:
        raise HTTPException(status_code=404, detail="Artifacts are not ready")
    settings = get_settings()
    path = settings.artifact_dir / run_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Artifact directory is missing")
    files = [str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()]
    return ArtifactResponse(
        run_id=run_id,
        output_type=run.output_type,
        name=run.artifact_name,
        download_url=f"/api/v1/runs/{run_id}/artifacts/download",
        files=files,
    )


@app.get("/api/v1/runs/{run_id}/artifacts/download")
async def download_artifact(run_id: str, session: Session = Depends(get_session)):
    run = session.get(RunRecord, run_id)
    if run is None or not run.artifact_path:
        raise HTTPException(status_code=404, detail="Artifacts are not ready")
    settings = get_settings()
    artifact_dir = settings.artifact_dir / run_id
    if run.output_type == "model":
        archive = shutil.make_archive(str(settings.artifact_dir / run_id), "zip", root_dir=artifact_dir)
        path = Path(archive)
    else:
        path = artifact_dir / run.artifact_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file is missing")
    return FileResponse(path, filename=path.name)
