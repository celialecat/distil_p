from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from .client import Distil

OutputType = Literal["model", "tool"]
RunStatus = Literal[
    "queued", "generating-data", "training", "evaluating", "complete", "failed"
]


class Teacher(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    params: str
    params_b: float
    license: str
    family: str
    hf_repo_id: str
    note: str
    color: str = "#f4b761"


class Metrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


class Artifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    output_type: OutputType
    name: str | None
    download_url: str
    files: list[str]


class Run(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    teacher_id: str
    teacher: str
    task_prompt: str
    preset: str | None = None
    target_params: float
    target_label: str
    output_type: OutputType
    status: RunStatus
    progress: int
    score: float | None = None
    tokens_generated: int = 0
    metrics: Metrics
    created_at: str
    updated_at: str
    error: str | None = None
    _client: Distil | None = PrivateAttr(default=None)

    def refresh(self) -> Run:
        """Fetch the latest state from the API."""
        if self._client is None:
            raise RuntimeError("This Run is not attached to a Distil client")
        updated = self._client.get_run(self.id)
        self.__dict__.update(updated.__dict__)
        return self

    def wait(self, timeout: float = 3600, poll_interval: float = 1.0) -> Run:
        """Poll until the run is complete or failed."""
        deadline = time.monotonic() + timeout
        while self.status not in ("complete", "failed"):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Distillation run {self.id} did not finish within {timeout}s"
                )
            time.sleep(poll_interval)
            self.refresh()
        if self.status == "failed":
            raise RuntimeError(self.error or f"Distillation run {self.id} failed")
        return self

    def download(self, dest: str | Path) -> Path:
        """Download the primary artifact and return its destination path."""
        if self._client is None:
            raise RuntimeError("This Run is not attached to a Distil client")
        return self._client.download(self.id, dest)
