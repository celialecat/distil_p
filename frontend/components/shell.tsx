"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { Beaker, BookOpen, ChevronDown, CircleHelp, FlaskConical, LayoutGrid, Play, Settings2, Sparkles } from "lucide-react";
import { usePathname } from "next/navigation";

const LiquidShader = dynamic(() => import("./ui/liquid-shader"), { ssr: false });

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const links = [{ href: "/", label: "Distill", icon: FlaskConical }, { href: "/runs", label: "Runs", icon: Play }, { href: "/models", label: "Model shelf", icon: LayoutGrid }];
  const shaderState = pathname === "/runs" ? "running" : pathname === "/models" ? "complete" : "idle";
  return <div className="lab-bg"><div className="pointer-events-none fixed inset-0 z-0 overflow-hidden opacity-90"><LiquidShader state={shaderState} /></div><i className="orb one" /><i className="orb two" /><i className="orb three" />
    <aside className="fixed bottom-0 left-0 top-0 z-20 hidden w-[228px] border-r border-white/[.07] bg-[#0c1416]/80 px-5 py-7 backdrop-blur-xl lg:block">
      <Link href="/" className="mb-12 flex items-center gap-3"><span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#e99a3c] text-[#18100a]"><Beaker size={19} /></span><span className="text-[17px] font-bold tracking-[-.03em]">distil<span className="text-[#e99a3c]">.</span></span></Link>
      <div className="eyebrow mb-3 px-3">The laboratory</div><nav className="space-y-1">{links.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className={`flex items-center gap-3 rounded-xl px-3 py-3 text-sm transition ${pathname === href ? "bg-[#e99a3c]/10 text-[#f4b761]" : "text-[#829291] hover:bg-white/[.04] hover:text-white"}`}><Icon size={17} strokeWidth={1.7} />{label}</Link>)}</nav>
      <div className="eyebrow mb-3 mt-10 px-3">Resources</div><nav className="space-y-1"><a href="#docs" className="flex items-center gap-3 rounded-xl px-3 py-3 text-sm text-[#829291] hover:bg-white/[.04] hover:text-white"><BookOpen size={17} />Documentation</a><a href="#help" className="flex items-center gap-3 rounded-xl px-3 py-3 text-sm text-[#829291] hover:bg-white/[.04] hover:text-white"><CircleHelp size={17} />Help center</a></nav>
      <div className="absolute bottom-7 left-5 right-5 glass rounded-2xl p-4"><div className="mb-2 flex items-center justify-between"><span className="text-xs font-semibold">GPU credits</span><Sparkles size={14} className="text-[#e99a3c]" /></div><div className="mb-2 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full w-[64%] rounded-full bg-[#e99a3c]" /></div><p className="text-[10px] text-[#829291]">64 / 100 hours remaining</p></div>
    </aside>
    <main className="relative z-10 min-h-screen lg:pl-[228px]"><header className="flex h-[76px] items-center justify-between border-b border-white/[.07] bg-[#0b1113]/20 px-6 backdrop-blur-sm md:px-10"><div className="flex items-center gap-2 text-xs text-[#71817f]"><span className="hidden md:inline">Workspace</span><ChevronDown size={14} /><span className="text-white">Celia&apos;s lab</span></div><div className="flex items-center gap-3"><button className="button-ghost flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-[#a7b4b1]"><Settings2 size={14} /> Settings</button><div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#d48c47] text-xs font-bold text-[#20150c]">C</div></div></header>{children}</main>
  </div>;
}
