# distil-sdk

Typed Python client for the Distil API.

```bash
pip install -e sdk/
```

```python
from distil_sdk import Distil

distil = Distil("http://localhost:8000")
teachers = distil.teachers()

run = distil.distill(
    teacher="qwen-2.5-72b",
    task="Extract renewal dates and obligations from legal contracts.",
    target_params=3,
    output_type="model",
)
run.wait()
print(run.metrics.eval_score)
artifact_path = run.download("./artifacts")
```

`Run.wait()` polls the persisted pipeline state. `output_type="tool"` downloads a standalone generated Python tool instead of model weights.
Task prompts must contain at least 10 non-whitespace characters. API and
network failures raise `DistilError`; a run that reaches `failed` raises a
`RuntimeError` from `Run.wait()` with the backend error message.
