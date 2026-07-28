"""Compare a teacher, base student, and LoRA student on new triage tickets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

TICKETS = (
    "My payment was charged twice and I cannot reach anyone for a refund.",
    "How do I change the email address on my account?",
    "The entire dashboard is down for our whole company and every employee is blocked.",
    "A customer says their package arrived damaged, but the product still works.",
    "I received a suspicious password-reset email and now I cannot log in to my account.",
)

SYSTEM_PROMPT = (
    "You are a customer-support triage specialist. Classify the ticket urgency "
    "as exactly one of Low, Medium, High, or Critical, then write one concise "
    "routing summary line. Return exactly:\n"
    "Urgency: <Low|Medium|High|Critical>\n"
    "Routing Summary: <one concise routing line>"
)


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, ticket: str, max_new_tokens: int) -> str:
    prompt = f"Ticket:\n{ticket}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = f"System: {SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def render_report(config: dict, results: list[dict[str, str]]) -> str:
    complete = sum(
        "urgency:" in result["student_after"].lower()
        and "routing summary:" in result["student_after"].lower()
        for result in results
    )
    if complete == len(results):
        verdict = (
            f"The adapted student produced the requested urgency and routing "
            f"structure on all {len(results)} new tickets. Classification quality "
            "still requires human review against the teacher/reference answers."
        )
    else:
        verdict = (
            f"The adapted student produced a complete urgency-plus-routing "
            f"answer on {complete}/{len(results)} new tickets. It still needs "
            "more task-specific data or training if outputs are incomplete."
        )
    lines = [
        "# Qualitative Evaluation",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Configuration",
        "",
    ]
    lines.extend(f"- **{key}:** `{value}`" for key, value in config.items())
    lines.extend(["", "## Results", ""])
    for index, result in enumerate(results, 1):
        lines.extend(
            [
                f"### {index}. {result['ticket']}",
                "",
                "**Teacher**",
                "",
                "```text",
                result["teacher"],
                "```",
                "",
                "**Student before adapter**",
                "",
                "```text",
                result["student_before"],
                "```",
                "",
                "**Student after adapter**",
                "",
                "```text",
                result["student_after"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--student", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(os.getenv("DISTIL_TORCH_THREADS", "8")))
    from peft import PeftModel

    teacher, teacher_tokenizer = load_model(args.teacher)
    student, student_tokenizer = load_model(args.student)
    adapted = PeftModel.from_pretrained(
        student,
        str(args.adapter),
        local_files_only=True,
    )
    adapted.eval()
    results = []
    for ticket in TICKETS:
        results.append(
            {
                "ticket": ticket,
                "teacher": generate(teacher, teacher_tokenizer, ticket, args.max_new_tokens),
                "student_before": generate(student, student_tokenizer, ticket, args.max_new_tokens),
                "student_after": generate(adapted, student_tokenizer, ticket, args.max_new_tokens),
            }
        )
    config = {
        "teacher": args.teacher,
        "student": args.student,
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "tickets": len(TICKETS),
    }
    args.output.write_text(render_report(config, results))
    args.output.with_suffix(".json").write_text(json.dumps({"config": config, "results": results}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
