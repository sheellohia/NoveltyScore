import type { ScoreRequest, ScoreResponse, TopicDetail, TopicSummary } from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "Can't reach the server. Is the backend running on :8000?");
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") message = body.detail;
    } catch {
      // non-JSON error body; keep default message
    }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listTopics: () => request<TopicSummary[]>("/topics"),
  getTopic: (topicId: string) => request<TopicDetail>(`/topics/${encodeURIComponent(topicId)}`),
  score: (body: ScoreRequest) =>
    request<ScoreResponse>("/score", { method: "POST", body: JSON.stringify(body) }),
};
