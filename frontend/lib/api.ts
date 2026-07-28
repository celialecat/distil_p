export type RunStatus = "queued" | "generating-data" | "training" | "evaluating" | "complete" | "failed";
export type OutputType = "model" | "tool";

export type Teacher = {
  id: string;
  name: string;
  params: string;
  license: string;
  family: string;
  color: string;
  note: string;
};

export type DistillationRun = {
  id: string;
  teacher: string;
  task: string;
  target: string;
  output: OutputType;
  status: RunStatus;
  progress: number;
  created: string;
  score: string;
  tokens: string;
  error?: string;
};

export const teachers: Teacher[] = [
  { id: "llama-3.1-70b", name: "Llama 3.1", params: "70B", license: "Llama 3.1 Community", family: "Meta", color: "#f59e0b", note: "Reasoning powerhouse" },
  { id: "llama-3.1-8b", name: "Llama 3.1", params: "8B", license: "Llama 3.1 Community", family: "Meta", color: "#f59e0b", note: "Balanced instruct model" },
  { id: "qwen-2.5-72b", name: "Qwen 2.5", params: "72B", license: "Apache 2.0", family: "Alibaba", color: "#2dd4bf", note: "Multilingual specialist" },
  { id: "qwen-2.5-14b", name: "Qwen 2.5", params: "14B", license: "Apache 2.0", family: "Alibaba", color: "#2dd4bf", note: "Fast and capable" },
  { id: "qwen-2.5-7b", name: "Qwen 2.5", params: "7B", license: "Apache 2.0", family: "Alibaba", color: "#2dd4bf", note: "Compact instruct model" },
  { id: "qwen-2.5-0.5b", name: "Qwen 2.5", params: "0.5B", license: "Apache 2.0", family: "Alibaba", color: "#2dd4bf", note: "CPU-friendly teacher" },
  { id: "mistral-large", name: "Mistral Large", params: "123B", license: "Mistral Research", family: "Mistral AI", color: "#fb7185", note: "Long-context expert" },
  { id: "mistral-7b", name: "Mistral", params: "7B", license: "Apache 2.0", family: "Mistral AI", color: "#fb7185", note: "Efficient open weights" },
  { id: "gemma-2-27b", name: "Gemma 2", params: "27B", license: "Gemma Terms", family: "Google", color: "#a78bfa", note: "Efficient open weights" },
  { id: "gemma-2-9b", name: "Gemma 2", params: "9B", license: "Gemma Terms", family: "Google", color: "#a78bfa", note: "Small and capable" },
  { id: "deepseek-v3", name: "DeepSeek V3", params: "671B / 37B active", license: "MIT", family: "DeepSeek", color: "#60a5fa", note: "MoE reasoning" },
  { id: "phi-4", name: "Phi-4", params: "14B", license: "MIT", family: "Microsoft", color: "#34d399", note: "Synthetic-data refined" },
  { id: "mixtral-8x7b", name: "Mixtral", params: "8x7B", license: "Apache 2.0", family: "Mistral AI", color: "#fb7185", note: "Sparse mixture of experts" },
];

export const runs: DistillationRun[] = [
  { id: "run-0248", teacher: "Qwen 2.5 72B", task: "Legal contract extraction", target: "3B", output: "model", status: "training", progress: 68, created: "12 min ago", score: "—", tokens: "1.2M" },
  { id: "run-0247", teacher: "Llama 3.1 70B", task: "Code review assistant", target: "7B", output: "tool", status: "evaluating", progress: 91, created: "38 min ago", score: "88.4%", tokens: "842K" },
  { id: "run-0246", teacher: "Phi-4", task: "Customer support triage", target: "1B", output: "model", status: "complete", progress: 100, created: "Yesterday", score: "94.1%", tokens: "492K" },
  { id: "run-0245", teacher: "Gemma 2 27B", task: "Medical literature QA", target: "3B", output: "model", status: "complete", progress: 100, created: "Yesterday", score: "91.7%", tokens: "2.1M" },
];

const fallback = <T,>(value: T) => Promise.resolve(value);

export async function getTeachers(): Promise<Teacher[]> {
  try {
    const response = await fetch("/api/v1/teachers");
    if (!response.ok) throw new Error("Backend unavailable");
    return response.json();
  } catch {
    return fallback(teachers);
  }
}

export type CreateRunInput = {
  teacher_id: string;
  task_prompt: string;
  target_params: number;
  output_type: OutputType;
  preset?: string;
};

export async function createRun(input: CreateRunInput) {
  const response = await fetch("/api/v1/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? "Unable to launch distillation");
  }
  return response.json();
}

export function getArtifactDownloadUrl(runId: string): string {
  return `/api/v1/runs/${encodeURIComponent(runId)}/artifacts/download`;
}

export type RunsFetchResult = {
  runs: DistillationRun[];
  usingFallback: boolean;
};

export async function getRunsState(): Promise<RunsFetchResult> {
  try {
    const response = await fetch("/api/v1/runs");
    if (!response.ok) throw new Error("Backend unavailable");
    const data = await response.json();
    const mapped: DistillationRun[] = data.map((run: {
      id: string;
      teacher: string;
      task_prompt: string;
      target_label: string;
      output_type: OutputType;
      status: RunStatus;
      progress: number;
      created_at: string;
      score: number | null;
      tokens_generated: number;
      error: string | null;
    }) => ({
      id: run.id,
      teacher: run.teacher,
      task: run.task_prompt,
      target: run.target_label,
      output: run.output_type,
      status: run.status,
      progress: run.progress,
      created: new Date(run.created_at).toLocaleString(),
      score: run.score === null ? "—" : `${run.score}%`,
      tokens: `${run.tokens_generated}`,
      error: run.error ?? undefined,
    }));
    return { runs: mapped, usingFallback: false };
  } catch {
    return { runs, usingFallback: true };
  }
}

export async function getRuns(): Promise<DistillationRun[]> {
  return (await getRunsState()).runs;
}
