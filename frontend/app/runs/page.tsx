"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ArrowDownToLine, ArrowUpRight, CheckCircle2, Clock3, Code2, Database, MoreHorizontal, Play, RefreshCw, Sparkles, TrendingUp } from "lucide-react";
import { Shell } from "../../components/shell";
import { SectionHeading } from "../../components/section-heading";
import { getArtifactDownloadUrl, getRunsState, type DistillationRun, type RunStatus } from "../../lib/api";

const statusMap: Record<RunStatus, { label: string; color: string }> = {
  queued: { label: "Queued", color: "#a8b2b0" },
  "generating-data": { label: "Generating data", color: "#60a5fa" },
  training: { label: "Training", color: "#f4b761" },
  evaluating: { label: "Evaluating", color: "#c084fc" },
  complete: { label: "Complete", color: "#39d6bc" },
  failed: { label: "Failed", color: "#fb7185" },
};

export default function RunsPage() {
  const [runList, setRunList] = useState<DistillationRun[]>([]);
  const [filter, setFilter] = useState<"All" | "Active" | "Complete">("All");
  const [refreshing, setRefreshing] = useState(false);
  const [usingFallback, setUsingFallback] = useState(false);

  async function refresh() {
    setRefreshing(true);
    try {
      const result = await getRunsState();
      setRunList(result.runs);
      setUsingFallback(result.usingFallback);
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, []);

  const filtered = runList.filter((run) => filter === "All" || (filter === "Complete" ? run.status === "complete" : run.status !== "complete"));
  const tokenTotal = runList.reduce((total, run) => total + (Number(run.tokens.replace(/[^\d.]/g, "")) || 0), 0);

  return <Shell><div className="mx-auto max-w-[1220px] px-6 py-10 md:px-10 md:py-14"><div className="flex items-start justify-between"><SectionHeading eyebrow="Lab activity" title="Your distillations." body="Every run, from first prompt to finished artifact, in one clear vessel." /><button onClick={refresh} className="button-ghost flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-[#829291]" disabled={refreshing}><RefreshCw size={14} className={refreshing ? "animate-spin" : ""} /> Refresh</button></div>{usingFallback && <div className="mb-6 rounded-xl border border-[#f4b761]/20 bg-[#f4b761]/[.08] px-4 py-3 text-xs text-[#f4b761]">Backend unavailable. Showing demo runs; reconnect the backend to see live progress.</div>}<div className="mb-7 flex flex-wrap gap-2">{(["All", "Active", "Complete"] as const).map((item) => <button key={item} onClick={() => setFilter(item)} className={`button-ghost rounded-full px-4 py-2 text-xs ${filter === item ? "border-[#39d6bc]/40 bg-[#39d6bc]/[.08] text-[#6ee4d1]" : "text-[#829291]"}`}>{item}{item === "All" && <span className="ml-1 opacity-60">{runList.length}</span>}</button>)}</div><div className="grid gap-4">{filtered.map((run, index) => <RunCard key={run.id} run={run} index={index} />)}{filtered.length === 0 && <div className="glass rounded-2xl p-12 text-center text-sm text-[#829291]">No runs in this view yet.</div>}</div><div className="mt-10 grid gap-4 md:grid-cols-3"><div className="glass rounded-2xl p-5"><Sparkles size={17} className="mb-5 text-[#e99a3c]" /><div className="text-2xl font-semibold">{tokenTotal ? `${tokenTotal.toFixed(1)}K` : "4.6M"}</div><p className="mt-1 text-xs text-[#829291]">Tokens distilled</p></div><div className="glass rounded-2xl p-5"><TrendingUp size={17} className="mb-5 text-[#39d6bc]" /><div className="text-2xl font-semibold">{runList.length ? `${(runList.filter((run) => run.score !== "—").reduce((sum, run) => sum + parseFloat(run.score), 0) / Math.max(1, runList.filter((run) => run.score !== "—").length)).toFixed(1)}%` : "91.4%"}</div><p className="mt-1 text-xs text-[#829291]">Average quality score</p></div><div className="glass rounded-2xl p-5"><Play size={17} className="mb-5 text-[#a78bfa]" /><div className="text-2xl font-semibold">{runList.filter((run) => run.status !== "complete").length}</div><p className="mt-1 text-xs text-[#829291]">Runs in progress</p></div></div></div></Shell>;
}

function RunCard({ run, index }: { run: DistillationRun; index: number }) {
  const status = statusMap[run.status];
  return <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * .06 }} className="glass group rounded-2xl p-5 transition hover:border-white/20"><div className="flex flex-col gap-5 md:flex-row md:items-center"><div className="flex min-w-0 flex-1 items-center gap-4"><div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${run.output === "tool" ? "bg-[#a78bfa]/10 text-[#c084fc]" : "bg-[#e99a3c]/10 text-[#f4b761]"}`}>{run.output === "tool" ? <Code2 size={19} /> : <Database size={19} />}</div><div className="min-w-0"><div className="mb-1 flex items-center gap-2"><span className="text-[10px] text-[#60706e]">{run.id}</span><span className="text-[10px] text-[#60706e]">·</span><span className="text-[10px] text-[#60706e]">{run.created}</span></div><h3 className="truncate text-sm font-semibold">{run.task}</h3><p className="mt-1 text-xs text-[#829291]">{run.teacher} <span className="mx-1 text-[#475351]">→</span> {run.target} {run.output}</p></div></div><div className="flex items-center gap-6 md:w-[390px]"><div className="min-w-[150px] flex-1"><div className="mb-2 flex justify-between text-[10px]"><span style={{ color: status.color }}>{status.label}</span><span className="text-[#71817f]">{run.progress}%</span></div><div className="h-1.5 overflow-hidden rounded-full bg-white/[.08]"><div className="h-full rounded-full transition-all" style={{ width: `${run.progress}%`, background: status.color }} /></div></div><div className="hidden text-right sm:block"><div className="text-sm font-semibold">{run.score}</div><div className="mt-1 text-[10px] text-[#60706e]">eval score</div></div><button className="text-[#71817f] hover:text-white"><MoreHorizontal size={17} /></button></div></div>{run.status === "failed" && <div className="mt-4 border-t border-rose-400/10 pt-3 text-[10px] text-rose-200"><span className="font-semibold">Pipeline failed:</span> {run.error || "The backend did not provide an error message."}</div>}{run.status !== "complete" && run.status !== "failed" && <div className="mt-4 flex items-center gap-2 border-t border-white/[.06] pt-3 text-[10px] text-[#60706e]"><Clock3 size={12} /> {run.tokens} tokens generated <span className="mx-1">·</span> live updates enabled <span className="ml-auto flex items-center gap-1 text-[#829291]">View details <ArrowUpRight size={12} /></span></div>}{run.status === "complete" && <div className="mt-4 flex items-center gap-2 border-t border-white/[.06] pt-3 text-[10px] text-[#39d6bc]"><CheckCircle2 size={12} /> Artifact ready <a href={getArtifactDownloadUrl(run.id)} download className="ml-auto flex items-center gap-1 text-[#829291] hover:text-white">Download <ArrowDownToLine size={12} /></a></div>}</motion.div>;
}
