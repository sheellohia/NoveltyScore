import type { Flags, ScoreResponse } from "../lib/types";
import SurfacingView from "./SurfacingView";

const pct = (v: number) => `${Math.round(v * 100)}%`;

const BREAKDOWN_LABELS: Record<keyof ScoreResponse["breakdown"], string> = {
  relevance: "Relevance to fixed content (absolute)",
  novelty: "Novelty (percentile vs corpus)",
  recombination_novelty: "Recombination (percentile vs corpus)",
};

const FLAG_LABELS: Record<keyof Flags, string> = {
  duplicate: "Duplicate",
  low_relevance: "Low relevance",
  gaming: "Gaming",
};

function Bar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-slate-600">{label}</span>
        <span className="tabular-nums text-slate-500">{pct(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: pct(value) }} />
      </div>
    </div>
  );
}

export default function ScorePanel({ result }: { result: ScoreResponse }) {
  const activeFlags = (Object.keys(FLAG_LABELS) as (keyof Flags)[]).filter((k) => result.flags[k]);

  return (
    <section className="space-y-5 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Score</h2>

      <div className="flex flex-wrap items-end gap-6">
        <div>
          <div className="text-xs font-medium text-slate-500">System score</div>
          <div className="text-6xl font-bold tabular-nums tracking-tight text-indigo-700">
            {pct(result.system_score)}
          </div>
        </div>
        <div className="flex gap-6 border-l border-slate-200 pl-6">
          <div>
            <div className="text-xs font-medium text-slate-500">Baseline</div>
            <div className="text-2xl font-semibold tabular-nums text-slate-700">{pct(result.baseline_score)}</div>
          </div>
          <div>
            <div className="text-xs font-medium text-slate-500">
              LLM judge
              {result.llm_source === "cached" && <span className="ml-1 text-[10px] text-slate-400">(cached)</span>}
            </div>
            {result.llm_source === "unavailable" ? (
              <div className="text-2xl font-semibold text-slate-400" title="The LLM judge could not be reached for this submission">
                n/a
              </div>
            ) : (
              <div className="text-2xl font-semibold tabular-nums text-slate-700">{pct(result.llm_score)}</div>
            )}
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {(Object.keys(BREAKDOWN_LABELS) as (keyof typeof BREAKDOWN_LABELS)[]).map((k) => (
          <Bar key={k} label={BREAKDOWN_LABELS[k]} value={result.breakdown[k]} />
        ))}
      </div>

      {activeFlags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {activeFlags.map((k) => (
            <span
              key={k}
              className="rounded-full border border-red-200 bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700"
            >
              {FLAG_LABELS[k]}
            </span>
          ))}
        </div>
      )}

      {result.surfacing && <SurfacingView surfacing={result.surfacing} lowRelevance={result.flags.low_relevance} />}

      <p className="border-t border-slate-100 pt-4 text-sm leading-relaxed text-slate-600">{result.rationale}</p>
    </section>
  );
}
