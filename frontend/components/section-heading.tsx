export function SectionHeading({ eyebrow, title, body }: { eyebrow: string; title: string; body?: string }) {
  return <div className="mb-8"><div className="eyebrow mb-3">{eyebrow}</div><h1 className="text-3xl font-semibold tracking-[-.04em] text-[#f1f4ed] md:text-[40px]">{title}</h1>{body && <p className="mt-3 max-w-xl text-sm leading-6 text-[#829291]">{body}</p>}</div>;
}
