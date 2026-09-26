// Mirrors backend/models.py — keep in sync. This is the API contract.

export interface TopicSummary {
  topic_id: string;
  title: string;
}

export interface TopicDetail {
  topic_id: string;
  title: string;
  fixed_content: string;
}

export interface ScoreRequest {
  topic_id: string;
  header: string;
  content: string;
  takeaway: string;
}

export interface Breakdown {
  relevance: number; // absolute cos(content, fixed content), 0-1
  novelty: number; // percentile vs this topic's corpus
  recombination_novelty: number; // percentile vs this topic's corpus
}

export interface Flags {
  duplicate: boolean;
  low_relevance: boolean;
  gaming: boolean;
}

export interface SurfacedSentence {
  text: string;
  label: "novel" | "partial" | "redundant";
  redundancy: number; // max cosine to any sentence in existing submissions
  novelty_contribution: number;
  closest_id: string | null;
  closest_snippet: string | null;
}

export interface Surfacing {
  net_new_ratio: number; // share of content (by words) that is novel or partial
  thresholds: { novel_max_redundancy: number; redundant_min_redundancy: number };
  sentences: SurfacedSentence[];
}

export interface ScoreResponse {
  system_score: number;
  baseline_score: number;
  llm_score: number;
  breakdown: Breakdown;
  flags: Flags;
  rationale: string;
  surfacing: Surfacing;
  llm_source: "live" | "cached" | "unavailable"; // llm_score is meaningless when "unavailable"
}
