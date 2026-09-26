import type { SurfacedSentence, Surfacing } from "../lib/types";

const STYLE: Record<SurfacedSentence["label"], string> = {
  novel: "bg-emerald-100 text-emerald-900",
  partial: "bg-amber-100 text-amber-900",
  redundant: "bg-slate-100 text-slate-500",
};

const LEGEND: [SurfacedSentence["label"], string][] = [
  ["novel", "New"],
  ["partial", "Partly covered"],
  ["redundant", "Already said"],
];

function tooltip(s: SurfacedSentence) {
  const sim = `similarity ${s.redundancy.toFixed(2)}`;
  return s.closest_id ? `Closest: ${s.closest_id} (${sim}) — "${s.closest_snippet}"` : sim;
}

export default function SurfacingView({ surfacing, lowRelevance }: { surfacing: Surfacing; lowRelevance: boolean }) {
  return (
    <div className="space-y-3 border-t border-slate-100 pt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          New information vs. existing submissions
        </h3>
        <span className="rounded-full bg-emerald-600 px-2.5 py-0.5 text-xs font-semibold text-white tabular-nums">
          Net-new: {Math.round(surfacing.net_new_ratio * 100)}%
        </span>
      </div>

      {lowRelevance && (
        <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Flagged as low relevance: these sentences are new to the existing submissions, but not relevant to the
          fixed content.
        </p>
      )}

      <p className="text-sm leading-7">
        {surfacing.sentences.map((s, i) => (
          <span key={i}>
            <span title={tooltip(s)} className={`rounded px-1 py-0.5 box-decoration-clone ${STYLE[s.label]}`}>
              {s.text}
              {s.label === "redundant" && s.closest_id && (
                <sup className="ml-1 whitespace-nowrap text-[10px] font-medium text-slate-500">≈ {s.closest_id}</sup>
              )}
            </span>{" "}
          </span>
        ))}
      </p>

      <div className="flex flex-wrap gap-3 text-xs text-slate-500">
        {LEGEND.map(([label, text]) => (
          <span key={label} className="flex items-center gap-1.5">
            <span className={`inline-block h-3 w-3 rounded-sm ${STYLE[label].split(" ")[0]}`} />
            {text}
          </span>
        ))}
        <span className="text-slate-400">Hover a sentence for its closest existing submission.</span>
      </div>
    </div>
  );
}
