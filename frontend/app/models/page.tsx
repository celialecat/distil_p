"use client";

import { useEffect, useState } from "react";
import { ArrowUpRight, Box, Check, Code2, Copy, Download, Gauge, Search, SlidersHorizontal } from "lucide-react";
import { Shell } from "../../components/shell";
import { SectionHeading } from "../../components/section-heading";
import { getArtifactDownloadUrl, getRunsState, type DistillationRun } from "../../lib/api";

type ModelCard = { name: string; version: string; size: string; task: string; score: string; latency: string; color: string; date: string; runId?: string };
const showcaseModels: ModelCard[] = [
  { name: "contract-compass", version: "v1.0.0", size: "3B", task: "Legal contract extraction", score: "91.8%", latency: "42ms", color: "#f4b761", date: "Today" },
  { name: "support-spark", version: "v0.8.2", size: "1B", task: "Customer support triage", score: "94.1%", latency: "18ms", color: "#39d6bc", date: "Yesterday" },
  { name: "review-forge", version: "v1.2.0", size: "7B", task: "Code review assistant", score: "88.4%", latency: "67ms", color: "#c084fc", date: "2 days ago" },
  { name: "med-lit-lens", version: "v1.0.1", size: "3B", task: "Medical literature QA", score: "91.7%", latency: "44ms", color: "#60a5fa", date: "3 days ago" },
];

export default function ModelsPage() {
  const [query, setQuery] = useState("");
  const [models, setModels] = useState<ModelCard[]>(showcaseModels);
  const [usingFallback, setUsingFallback] = useState(false);
  useEffect(() => {
    getRunsState().then(({ runs: runList, usingFallback: fallback }) => {
      const completed = runList.filter((run) => run.status === "complete" && run.output === "model").map(runToModel);
      setModels([...completed, ...showcaseModels]);
      setUsingFallback(fallback);
    });
  }, []);
  const filtered = models.filter((model) => `${model.name} ${model.task}`.toLowerCase().includes(query.toLowerCase()));
  return <Shell><div className="mx-auto max-w-[1220px] px-6 py-10 md:px-10 md:py-14"><div className="flex flex-col justify-between gap-5 md:flex-row"><SectionHeading eyebrow="Your shelf" title="Small models. Big focus." body="Finished artifacts, ready to integrate into the work they were distilled for." /><button className="button-ghost flex h-fit items-center gap-2 rounded-lg px-4 py-2.5 text-xs text-[#a7b4b1]"><SlidersHorizontal size={14} /> Sort: recently added</button></div>{usingFallback && <div className="mb-6 rounded-xl border border-[#f4b761]/20 bg-[#f4b761]/[.08] px-4 py-3 text-xs text-[#f4b761]">Backend unavailable. Showing showcase models; connect the backend to see your finished artifacts.</div>}<div className="mb-8 flex max-w-lg items-center gap-3 rounded-xl border border-white/[.09] bg-white/[.025] px-4 py-3"><Search size={15} className="text-[#71817f]" /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search your model shelf..." className="w-full bg-transparent text-xs outline-none placeholder:text-[#536360]" /></div><div className="grid gap-5 md:grid-cols-2">{filtered.map((model) => <div key={`${model.name}-${model.runId ?? model.version}`} className="glass rounded-2xl p-5 transition hover:-translate-y-1 hover:border-white/20"><div className="mb-7 flex items-start justify-between"><div className="flex items-center gap-3"><div className="flex h-11 w-11 items-center justify-center rounded-xl" style={{ background: `${model.color}18`, color: model.color }}><Box size={20} /></div><div><h3 className="text-sm font-semibold">{model.name}</h3><p className="mt-1 text-[10px] text-[#71817f]">{model.version} · added {model.date}</p></div></div><button className="text-[#71817f] hover:text-white"><ArrowUpRight size={16} /></button></div><div className="mb-5 rounded-xl bg-black/20 p-4"><p className="mb-3 text-[10px] uppercase tracking-[.14em] text-[#60706e]">Specialized for</p><p className="text-sm">{model.task}</p></div><div className="grid grid-cols-3 gap-3 border-b border-white/[.07] pb-5"><div><p className="text-[10px] text-[#60706e]">Size</p><p className="mt-1 text-sm font-semibold">{model.size}</p></div><div><p className="text-[10px] text-[#60706e]">Quality</p><p className="mt-1 text-sm font-semibold text-[#39d6bc]">{model.score}</p></div><div><p className="text-[10px] text-[#60706e]">Latency</p><p className="mt-1 flex items-center gap-1 text-sm font-semibold"><Gauge size={12} className="text-[#f4b761]" />{model.latency}</p></div></div><div className="mt-4 flex gap-2">{model.runId ? <a href={getArtifactDownloadUrl(model.runId)} download className="button-primary flex flex-1 items-center justify-center gap-2 rounded-lg py-2.5 text-xs font-semibold"><Download size={14} /> Download</a> : <button disabled className="button-ghost flex flex-1 cursor-not-allowed items-center justify-center gap-2 rounded-lg py-2.5 text-xs text-[#71817f]"><Download size={14} /> Demo artifact</button>}<button className="button-ghost flex items-center gap-2 rounded-lg px-3 text-xs"><Code2 size={14} /> Integrate</button></div></div>)}</div><div className="mt-8 glass rounded-2xl p-5"><div className="flex flex-col gap-4 md:flex-row md:items-center"><div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#39d6bc]/10 text-[#39d6bc]"><Code2 size={18} /></div><div className="flex-1"><p className="text-sm font-semibold">Quick integration</p><p className="mt-1 text-xs text-[#829291]">Drop any model from your shelf into your product with the Distil SDK.</p></div><div className="flex items-center gap-3 rounded-lg border border-white/[.08] bg-black/20 px-3 py-2 font-mono text-[10px] text-[#9cafaa]"><span><span className="text-[#c084fc]">from</span> distil_sdk <span className="text-[#c084fc]">import</span> Distil</span><Copy size={13} className="text-[#71817f]" /></div><button className="button-ghost flex items-center gap-2 rounded-lg px-3 py-2 text-xs"><Check size={13} className="text-[#39d6bc]" /> Docs</button></div></div></div></Shell>;
}

function runToModel(run: DistillationRun): ModelCard {
  return { name: run.task.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""), version: "live", size: run.target, task: run.task, score: run.score, latency: "—", color: "#39d6bc", date: run.created, runId: run.id };
}
