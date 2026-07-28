"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { ArrowRight, Check, ChevronLeft, Code2, Cpu, Info, Layers3, LoaderCircle, Sparkles } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Shell } from "../components/shell";
import { SectionHeading } from "../components/section-heading";
import { createRun, getTeachers, teachers as fallbackTeachers, type OutputType, type Teacher } from "../lib/api";

const presets = ["Legal summarization", "Medical QA", "Customer support", "Code review", "Financial extraction"];
const steps = ["Teacher", "Task signal", "Target size", "Review"];
const sizes = ["0.5B", "1B", "3B", "7B"];
const outputOptions: { value: OutputType; label: string; desc: string; Icon: LucideIcon }[] = [
  { value: "model", label: "Distilled model", desc: "Deployable weights", Icon: Cpu },
  { value: "tool", label: "Code tool", desc: "Standalone artifact", Icon: Code2 },
];

export default function Home() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [teacherList, setTeacherList] = useState<Teacher[]>(fallbackTeachers);
  const [teacher, setTeacher] = useState(fallbackTeachers[2]);
  const [task, setTask] = useState("");
  const [target, setTarget] = useState("3B");
  const [output, setOutput] = useState<OutputType>("model");
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState("");

  useEffect(() => {
    getTeachers().then((items) => {
      setTeacherList(items);
      setTeacher((current) => items.find((item) => item.id === current.id) ?? items[0] ?? current);
    });
  }, []);

  const ratio = `${Math.round(parseFloat(teacher.params) / parseFloat(target))}×`;
  const taskIsValid = task.trim().length >= 10;
  async function launch() {
    if (!taskIsValid) {
      setLaunchError("Describe the task in at least 10 characters before launching.");
      return;
    }
    setLaunching(true);
    setLaunchError("");
    try {
      await createRun({
        teacher_id: teacher.id,
        task_prompt: task.trim(),
        target_params: parseFloat(target),
        output_type: output,
      });
      router.push("/runs");
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "Unable to launch distillation");
    } finally {
      setLaunching(false);
    }
  }

  return <Shell><div className="mx-auto max-w-[1220px] px-6 py-10 md:px-10 md:py-14"><div className="mb-11 flex items-start justify-between"><SectionHeading eyebrow="New distillation" title="Turn intelligence into focus." body="Choose a teacher, define the work, and crystallize a smaller model that does one thing exceptionally well." /><div className="hidden items-center gap-3 md:flex"><span className="h-2 w-2 animate-pulse rounded-full bg-[#39d6bc]" /><span className="text-xs text-[#829291]">Lab systems operational</span></div></div>
    <div className="mb-9 flex max-w-3xl items-center">{steps.map((label, i) => <div key={label} className="flex flex-1 items-center"><div className={`flex h-8 w-8 items-center justify-center rounded-full border text-xs font-semibold ${step > i + 1 ? "border-[#39d6bc] bg-[#39d6bc] text-[#09201d]" : step === i + 1 ? "border-[#e99a3c] bg-[#e99a3c] text-[#1e1308]" : "border-white/15 text-[#758482]"}`}>{step > i + 1 ? <Check size={14} /> : `0${i + 1}`}</div><span className={`ml-2 hidden text-xs sm:block ${step === i + 1 ? "text-white" : "text-[#71817f]"}`}>{label}</span>{i < steps.length - 1 && <div className={`mx-3 h-px flex-1 ${step > i + 1 ? "bg-[#39d6bc]/60" : "bg-white/10"}`} />}</div>)}</div>
    <div className="grid gap-6 xl:grid-cols-[1fr_300px]"><div className="glass min-h-[500px] rounded-3xl p-6 md:p-9"><AnimatePresence mode="wait">{step === 1 && <motion.div key="teacher" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }}><div className="mb-7 flex items-center justify-between"><div><h2 className="text-lg font-semibold">Select a teacher model</h2><p className="mt-1 text-xs text-[#829291]">Start with a model whose strengths match your ambition.</p></div><span className="rounded-full border border-white/10 px-3 py-1 text-[10px] text-[#829291]">{teacherList.length} curated models</span></div><div className="grid gap-3 sm:grid-cols-2">{teacherList.map((item) => <button key={item.id} onClick={() => setTeacher(item)} className={`group rounded-2xl border p-4 text-left transition ${teacher.id === item.id ? "border-[#e99a3c]/70 bg-[#e99a3c]/[.08]" : "border-white/[.08] bg-white/[.018] hover:border-white/20"}`}><div className="mb-4 flex items-start justify-between"><span className="flex h-9 w-9 items-center justify-center rounded-xl text-xs font-bold" style={{ backgroundColor: `${item.color}22`, color: item.color }}>{item.family.slice(0, 2).toUpperCase()}</span>{teacher.id === item.id && <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#e99a3c] text-[#1e1308]"><Check size={12} /></span>}</div><h3 className="text-sm font-semibold">{item.name} <span className="font-normal text-[#aab5b0]">{item.params}</span></h3><p className="mt-1 text-[11px] text-[#829291]">{item.note}</p><p className="mt-4 text-[10px] text-[#60706e]">{item.license}</p></button>)}</div></motion.div>}
      {step === 2 && <motion.div key="task" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }}><h2 className="text-lg font-semibold">Define the task signal</h2><p className="mt-1 text-xs text-[#829291]">Give your teacher a clear job to teach. Distil will generate and validate examples around it.</p><div className="mt-7 flex flex-wrap gap-2">{presets.map((preset) => <button key={preset} onClick={() => setTask(preset)} className={`rounded-full border px-3 py-2 text-xs ${task === preset ? "border-[#39d6bc]/60 bg-[#39d6bc]/10 text-[#6ee4d1]" : "border-white/10 text-[#9aa9a5] hover:bg-white/[.04]"}`}>{preset}</button>)}</div><textarea value={task} onChange={(e) => setTask(e.target.value)} placeholder="Describe the specialized job your student should master..." className="mt-5 h-48 w-full resize-none rounded-2xl border border-white/10 bg-black/20 p-5 text-sm leading-6 text-white outline-none placeholder:text-[#536360] focus:border-[#39d6bc]/60" />{!taskIsValid && <p className="mt-3 text-xs text-[#f4b761]">Add at least 10 characters to continue.</p>}<div className="mt-4 flex gap-2 text-[11px] text-[#71817f]"><Sparkles size={14} className="text-[#e99a3c]" />Tip: include input format, desired output, and what “good” looks like.</div></motion.div>}
      {step === 3 && <motion.div key="size" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }}><h2 className="text-lg font-semibold">Choose your target size</h2><p className="mt-1 text-xs text-[#829291]">Smaller vessels are faster to serve. Find the right balance for your task.</p><div className="mt-12 flex items-end justify-between"><span className="text-6xl font-semibold tracking-[-.08em] text-[#f4b761]">{target}</span><span className="mb-2 rounded-full bg-[#39d6bc]/10 px-3 py-1.5 text-xs text-[#55d9c5]">{ratio} compression</span></div><input type="range" min="0" max="3" step="1" value={sizes.indexOf(target)} onChange={(e) => setTarget(sizes[Number(e.target.value)])} className="mt-10 w-full accent-[#e99a3c]" /><div className="mt-3 flex justify-between text-[10px] text-[#71817f]">{sizes.map((size) => <span key={size}>{size}</span>)}</div><div className="mt-12 border-t border-white/[.08] pt-6"><p className="mb-3 text-xs font-medium">What should we produce?</p><div className="grid gap-3 sm:grid-cols-2">{outputOptions.map(({ value, label, desc, Icon }) => <button key={value} onClick={() => setOutput(value)} className={`flex items-center gap-3 rounded-xl border p-4 text-left ${output === value ? "border-[#e99a3c]/60 bg-[#e99a3c]/[.07]" : "border-white/10"}`}><Icon size={18} className={output === value ? "text-[#f4b761]" : "text-[#71817f]"} /><span><b className="block text-xs">{label}</b><small className="text-[10px] text-[#71817f]">{desc}</small></span></button>)}</div></div></motion.div>}
      {step === 4 && <motion.div key="review" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }}><div className="mx-auto max-w-lg py-5 text-center"><div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-[#e99a3c]/15 text-[#f4b761]"><FlaskIcon /></div><h2 className="text-2xl font-semibold">Ready to distill</h2><p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-[#829291]">Your teacher will generate a focused training set, then we&apos;ll form it into a {target} {output === "model" ? "model" : "tool"}.</p><div className="mt-8 grid gap-2 text-left">{[["Teacher", `${teacher.name} · ${teacher.params}`], ["Task", task || "Add a task description"], ["Output", `${target} ${output}`], ["Est. compute", "~2.4 GPU hours"]].map(([key, value]) => <div key={key} className="flex justify-between rounded-xl bg-white/[.035] px-4 py-3 text-xs"><span className="text-[#71817f]">{key}</span><span>{value}</span></div>)}</div>{!taskIsValid && <p className="mt-5 rounded-lg border border-[#e99a3c]/20 bg-[#e99a3c]/10 px-3 py-2 text-left text-xs text-[#f4b761]">Task prompts must be at least 10 characters.</p>}{launchError && <p className="mt-3 rounded-lg border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-left text-xs text-rose-200">{launchError}</p>}</div></motion.div>}</AnimatePresence><div className="mt-9 flex justify-between border-t border-white/[.08] pt-6">{step > 1 ? <button onClick={() => setStep(step - 1)} className="button-ghost flex items-center gap-2 rounded-lg px-4 py-2.5 text-xs"><ChevronLeft size={15} />Back</button> : <span />}{step < 4 ? <button onClick={() => setStep(step + 1)} disabled={step === 2 && !taskIsValid} className="button-primary flex items-center gap-2 rounded-lg px-5 py-2.5 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-50">Continue <ArrowRight size={15} /></button> : <button onClick={launch} disabled={launching || !taskIsValid} className="button-primary flex items-center gap-2 rounded-lg px-5 py-2.5 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-70">{launching ? <LoaderCircle size={15} className="animate-spin" /> : <Sparkles size={15} />}{launching ? "Launching..." : "Launch distillation"}</button>}</div></div>
      <div className="space-y-4"><div className="glass rounded-3xl p-6"><div className="mb-5 flex items-center gap-2"><div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#39d6bc]/10 text-[#39d6bc]"><Layers3 size={14} /></div><span className="text-xs font-semibold">Your vessel</span></div><div className="flask"><div className="flask-neck" /><div className="flask-body"><div className="liquid" style={{ height: `${Math.max(18, step * 20)}%` }} /></div></div><div className="mt-5 text-center"><p className="text-xs text-[#829291]">A focused student, forming</p><p className="mt-1 text-sm font-medium text-[#f4b761]">{target} · {output === "model" ? "model weights" : "code artifact"}</p></div></div><div className="glass rounded-3xl p-5"><div className="flex gap-3"><Info size={15} className="mt-0.5 shrink-0 text-[#39d6bc]" /><p className="text-[11px] leading-5 text-[#829291]">Distil uses teacher-generated examples and evaluates the student against a held-out task set before it reaches your shelf.</p></div></div></div></div>
  </div></Shell>;
}
function FlaskIcon() { return <svg viewBox="0 0 48 48" className="h-8 w-8" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="M19 5h10M22 5v14L10 37a5 5 0 0 0 4 7h20a5 5 0 0 0 4-7L26 19V5" /><path d="M14 34h20M18 29h12" /></svg>; }
