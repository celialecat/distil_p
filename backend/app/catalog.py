from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherDefinition:
    id: str
    name: str
    params_b: float
    params_label: str
    family: str
    license: str
    hf_repo_id: str
    note: str


TEACHERS = (
    TeacherDefinition(
        "llama-3.1-70b",
        "Llama 3.1",
        70,
        "70B",
        "Meta",
        "Llama 3.1 Community",
        "meta-llama/Llama-3.1-70B-Instruct",
        "Reasoning powerhouse",
    ),
    TeacherDefinition(
        "llama-3.1-8b",
        "Llama 3.1",
        8,
        "8B",
        "Meta",
        "Llama 3.1 Community",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Balanced instruct model",
    ),
    TeacherDefinition(
        "qwen-2.5-72b",
        "Qwen 2.5",
        72,
        "72B",
        "Alibaba",
        "Apache 2.0",
        "Qwen/Qwen2.5-72B-Instruct",
        "Multilingual specialist",
    ),
    TeacherDefinition(
        "qwen-2.5-14b", "Qwen 2.5", 14, "14B", "Alibaba", "Apache 2.0", "Qwen/Qwen2.5-14B-Instruct", "Fast and capable"
    ),
    TeacherDefinition(
        "qwen-2.5-7b",
        "Qwen 2.5",
        7,
        "7B",
        "Alibaba",
        "Apache 2.0",
        "Qwen/Qwen2.5-7B-Instruct",
        "Compact instruct model",
    ),
    TeacherDefinition(
        "qwen-2.5-0.5b",
        "Qwen 2.5",
        0.5,
        "0.5B",
        "Alibaba",
        "Apache 2.0",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "CPU-friendly teacher",
    ),
    TeacherDefinition(
        "mistral-large",
        "Mistral Large",
        123,
        "123B",
        "Mistral AI",
        "Mistral Research",
        "mistralai/Mistral-Large-Instruct-2411",
        "Long-context expert",
    ),
    TeacherDefinition(
        "mistral-7b",
        "Mistral",
        7,
        "7B",
        "Mistral AI",
        "Apache 2.0",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Efficient open weights",
    ),
    TeacherDefinition(
        "gemma-2-27b", "Gemma 2", 27, "27B", "Google", "Gemma Terms", "google/gemma-2-27b-it", "Efficient open weights"
    ),
    TeacherDefinition(
        "gemma-2-9b", "Gemma 2", 9, "9B", "Google", "Gemma Terms", "google/gemma-2-9b-it", "Small and capable"
    ),
    TeacherDefinition(
        "deepseek-v3",
        "DeepSeek V3",
        671,
        "671B / 37B active",
        "DeepSeek",
        "MIT",
        "deepseek-ai/DeepSeek-V3",
        "MoE reasoning",
    ),
    TeacherDefinition("phi-4", "Phi-4", 14, "14B", "Microsoft", "MIT", "microsoft/phi-4", "Synthetic-data refined"),
    TeacherDefinition(
        "mixtral-8x7b",
        "Mixtral",
        46.7,
        "8x7B",
        "Mistral AI",
        "Apache 2.0",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Sparse mixture of experts",
    ),
)

STUDENT_BASES = (
    (0.5, "HuggingFaceTB/SmolLM2-360M-Instruct", "0.5B"),
    (1.0, "Qwen/Qwen2.5-0.5B-Instruct", "1B"),
    (3.0, "Qwen/Qwen2.5-3B-Instruct", "3B"),
    (7.0, "Qwen/Qwen2.5-7B-Instruct", "7B"),
)


def teacher_by_id(teacher_id: str) -> TeacherDefinition | None:
    return next((teacher for teacher in TEACHERS if teacher.id == teacher_id), None)


def nearest_student(target_params: float) -> tuple[str, str]:
    return min(STUDENT_BASES, key=lambda student: abs(student[0] - target_params))[1:]
