import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useParams } from "react-router-dom";
import TopicNav from "./components/TopicNav";
import TopicPage from "./routes/TopicPage";
import { api } from "./lib/api";
import type { TopicSummary } from "./lib/types";

function KeyedTopicPage() {
  const { topicId = "" } = useParams();
  // key resets form + score state when switching topics
  return <TopicPage key={topicId} topicId={topicId} />;
}

export default function App() {
  const [topics, setTopics] = useState<TopicSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listTopics()
      .then(setTopics)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-4">
          <div className="flex items-center gap-2 pt-4">
            <span className="text-lg font-semibold tracking-tight">Novelty Scorer</span>
          </div>
          <TopicNav topics={topics ?? []} />
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">
        {error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">{error}</div>
        ) : !topics ? (
          <p className="text-sm text-slate-500">Loading topics…</p>
        ) : topics.length === 0 ? (
          <p className="text-sm text-slate-500">No topics found in backend/data.</p>
        ) : (
          <Routes>
            <Route path="/" element={<Navigate to={`/topic/${topics[0].topic_id}`} replace />} />
            <Route path="/topic/:topicId" element={<KeyedTopicPage />} />
            <Route path="*" element={<Navigate to={`/topic/${topics[0].topic_id}`} replace />} />
          </Routes>
        )}
      </main>
    </div>
  );
}
