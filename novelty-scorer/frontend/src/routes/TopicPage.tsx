import { useEffect, useState } from "react";
import ScorePanel from "../components/ScorePanel";
import SubmissionForm, { type SubmissionValues } from "../components/SubmissionForm";
import { api } from "../lib/api";
import type { ScoreResponse, TopicDetail } from "../lib/types";
import { countWords } from "../lib/words";

export default function TopicPage({ topicId }: { topicId: string }) {
  const [topic, setTopic] = useState<TopicDetail | null>(null);
  const [topicError, setTopicError] = useState<string | null>(null);

  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getTopic(topicId)
      .then((t) => !cancelled && setTopic(t))
      .catch((e: Error) => !cancelled && setTopicError(e.message));
    return () => {
      cancelled = true;
    };
  }, [topicId]);

  async function handleSubmit(values: SubmissionValues) {
    setSubmitting(true);
    setScoreError(null);
    try {
      setResult(await api.score({ topic_id: topicId, ...values }));
    } catch (e) {
      setResult(null);
      setScoreError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  if (topicError) {
    return <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">{topicError}</div>;
  }
  if (!topic) {
    return <p className="text-sm text-slate-500">Loading topic…</p>;
  }

  return (
    <div className="space-y-6">
      <article className="rounded-xl border border-slate-200 bg-slate-100/60 p-5">
        <div className="flex items-baseline justify-between gap-4">
          <h1 className="text-xl font-semibold tracking-tight">{topic.title}</h1>
          <span className="shrink-0 text-xs tabular-nums text-slate-400">
            {countWords(topic.fixed_content)} words
          </span>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-700">{topic.fixed_content}</p>
      </article>

      <div className="grid gap-6 lg:grid-cols-2">
        <SubmissionForm submitting={submitting} onSubmit={handleSubmit} />

        <div>
          {scoreError && (
            <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
              <span className="font-medium">Scoring failed:</span> {scoreError}
            </div>
          )}
          {result ? (
            <div className={submitting ? "opacity-50 transition-opacity" : ""}>
              <ScorePanel result={result} />
            </div>
          ) : (
            !scoreError && (
              <div className="flex h-full min-h-40 items-center justify-center rounded-xl border border-dashed border-slate-300 p-5 text-sm text-slate-400">
                {submitting ? "Scoring… the LLM judge can take 10–30 seconds." : "Submit to see its novelty score."}
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
