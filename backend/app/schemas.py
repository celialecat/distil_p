from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

OutputType = Literal["model", "tool"]
RunStatus = Literal["queued", "generating-data", "training", "evaluating", "complete", "failed"]


class TeacherResponse(BaseModel):
    id: str
    name: str
    params: str
    params_b: float
    license: str
    family: str
    hf_repo_id: str
    note: str
    color: str


class RunCreate(BaseModel):
    teacher_id: str = Field(min_length=1)
    task_prompt: str = Field(min_length=10, max_length=10_000)
    preset: str | None = Field(default=None, max_length=80)
    target_params: float = Field(gt=0, le=1000)
    output_type: OutputType = "model"

    @field_validator("task_prompt")
    @classmethod
    def task_not_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 10:
            raise ValueError("task_prompt must contain at least 10 characters")
        return value


class Metrics(BaseModel):
    loss_curve: list[float] = Field(default_factory=list)
    eval_score: float | None = None
    teacher_score: float | None = None
    compression_ratio: float
    tokens_generated: int = 0
    student_base: str | None = None
    examples_generated: int = 0
    held_out_examples: int = 0
    student_before_rouge_l: float | None = None
    student_after_rouge_l: float | None = None
    student_before_token_agreement: float | None = None
    student_after_token_agreement: float | None = None
    samples: list[dict[str, str]] = Field(default_factory=list)


class RunResponse(BaseModel):
    id: str
    teacher_id: str
    teacher: str
    task_prompt: str
    preset: str | None
    target_params: float
    target_label: str
    output_type: OutputType
    status: RunStatus
    progress: int
    score: float | None
    tokens_generated: int
    metrics: Metrics
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class ArtifactResponse(BaseModel):
    run_id: str
    output_type: OutputType
    name: str | None
    download_url: str
    files: list[str]
