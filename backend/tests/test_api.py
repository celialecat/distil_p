import time

import pytest


def wait_for_complete(client, run_id: str):
    for _ in range(100):
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"complete", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("pipeline did not finish")


def test_create_run_validates_target(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "teacher_id": "phi-4",
            "task_prompt": "Extract invoice fields",
            "target_params": 14,
            "output_type": "model",
        },
    )
    assert response.status_code == 422
    assert "smaller" in response.json()["detail"]


def test_create_run_validates_task_length(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "teacher_id": "phi-4",
            "task_prompt": "too short",
            "target_params": 1,
            "output_type": "model",
        },
    )
    assert response.status_code == 422
    assert "at least 10" in str(response.json()["detail"])


def test_response_only_labels_mask_prompt():
    from app.pipeline import _response_only_labels

    assert _response_only_labels([10, 11, 12, 13], [10, 11]) == [-100, -100, 12, 13]
    assert _response_only_labels([10, 11], [10, 11, 12]) == [-100, -100]


@pytest.mark.asyncio
async def test_real_triage_generation_uses_diverse_chat_examples(monkeypatch):
    from app import pipeline
    from app.catalog import teacher_by_id
    from app.db import RunRecord

    class FakeTeacher:
        index = 0

        async def generate(self, _teacher, prompt):
            if "Draft one realistic" in prompt:
                self.index += 1
                return f"Ticket draft {self.index}"
            return "Urgency: High\nRouting Summary: Escalate to support."

    monkeypatch.setenv("DISTIL_USE_TRANSFORMERS", "1")
    monkeypatch.setenv("DISTIL_EXAMPLES", "4")
    monkeypatch.setattr(pipeline, "teacher_client", lambda: FakeTeacher())
    examples, heldout = await pipeline.generate_examples(
        RunRecord(task_prompt="Customer support ticket triage: classify urgency and route.", target_params=1),
        teacher_by_id("phi-4"),
    )
    assert len(examples) == 3
    assert len(heldout) == 1
    assert all(example.system_prompt == pipeline.TRIAGE_SYSTEM_PROMPT for example in examples + heldout)
    assert len({example.prompt for example in examples + heldout}) == 4


def test_failed_pipeline_surfaces_error(client, monkeypatch):
    from app import pipeline

    async def fail_generation(*_args, **_kwargs):
        raise RuntimeError("teacher unavailable")

    monkeypatch.setattr(pipeline, "generate_examples", fail_generation)
    response = client.post(
        "/api/v1/runs",
        json={
            "teacher_id": "phi-4",
            "task_prompt": "Classify incoming customer support issues",
            "target_params": 1,
            "output_type": "model",
        },
    )
    run = wait_for_complete(client, response.json()["id"])
    assert run["status"] == "failed"
    assert run["error"] == "teacher unavailable"


def test_simulated_pipeline_completes(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "teacher_id": "phi-4",
            "task_prompt": "Classify incoming customer support issues",
            "target_params": 1,
            "output_type": "model",
        },
    )
    assert response.status_code == 201
    run = wait_for_complete(client, response.json()["id"])
    assert run["status"] == "complete"
    assert run["metrics"]["loss_curve"]
    assert run["metrics"]["eval_score"] > 0


def test_tool_pipeline_writes_code_artifact(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "teacher_id": "qwen-2.5-72b",
            "task_prompt": "Review Python code for correctness",
            "target_params": 3,
            "output_type": "tool",
        },
    )
    run = wait_for_complete(client, response.json()["id"])
    assert run["status"] == "complete"
    artifacts = client.get(f"/api/v1/runs/{run['id']}/artifacts")
    assert artifacts.status_code == 200
    assert "distilled_tool.py" in artifacts.json()["files"]
    download = client.get(f"/api/v1/runs/{run['id']}/artifacts/download")
    assert download.status_code == 200
    assert "def run_task" in download.text


def test_model_pipeline_downloads_zip_artifact(client):
    response = client.post(
        "/api/v1/runs",
        json={
            "teacher_id": "phi-4",
            "task_prompt": "Summarize customer support tickets",
            "target_params": 1,
            "output_type": "model",
        },
    )
    run = wait_for_complete(client, response.json()["id"])
    download = client.get(f"/api/v1/runs/{run['id']}/artifacts/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert download.content[:2] == b"PK"
