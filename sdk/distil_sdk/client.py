from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .models import Artifact, OutputType, Run, Teacher


class DistilError(RuntimeError):
    """Raised when the Distil API returns an error."""


class Distil:
    """Small synchronous client for the Distil API."""

    def __init__(
        self, base_url: str = "http://localhost:8000", *, timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except httpx.RequestError as exc:
            raise DistilError(f"Unable to reach Distil API at {self.base_url}: {exc}") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise DistilError(f"{response.status_code}: {detail}")
        return response

    def teachers(self) -> list[Teacher]:
        return [
            Teacher.model_validate(item)
            for item in self._request("GET", "/api/v1/teachers").json()
        ]

    def distill(
        self,
        teacher: str,
        task: str,
        target_params: float,
        output_type: OutputType = "model",
        preset: str | None = None,
    ) -> Run:
        if len(task.strip()) < 10:
            raise ValueError("task must contain at least 10 non-whitespace characters")
        if target_params <= 0:
            raise ValueError("target_params must be greater than zero")
        if output_type not in ("model", "tool"):
            raise ValueError("output_type must be 'model' or 'tool'")
        payload = {
            "teacher_id": teacher,
            "task_prompt": task,
            "target_params": target_params,
            "output_type": output_type,
        }
        if preset is not None:
            payload["preset"] = preset
        run = Run.model_validate(
            self._request("POST", "/api/v1/runs", json=payload).json()
        )
        run._client = self
        return run

    def get_run(self, run_id: str) -> Run:
        run = Run.model_validate(self._request("GET", f"/api/v1/runs/{run_id}").json())
        run._client = self
        return run

    def runs(self) -> list[Run]:
        result = []
        for item in self._request("GET", "/api/v1/runs").json():
            run = Run.model_validate(item)
            run._client = self
            result.append(run)
        return result

    def artifacts(self, run_id: str) -> Artifact:
        return Artifact.model_validate(
            self._request("GET", f"/api/v1/runs/{run_id}/artifacts").json()
        )

    def download(self, run_id: str, dest: str | Path) -> Path:
        metadata = self.artifacts(run_id)
        destination = Path(dest)
        if destination.suffix:
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            destination /= metadata.name or f"{run_id}.artifact"
        response = self._request("GET", f"/api/v1/runs/{run_id}/artifacts/download")
        destination.write_bytes(response.content)
        return destination
