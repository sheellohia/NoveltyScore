"""Pydantic request/response models.

This file IS the API contract between the UI and the scoring pipeline.
Changing shapes here is a breaking change for the frontend (see frontend/src/lib/types.ts).
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_CONTENT_WORDS = 100
MAX_TAKEAWAY_CHARS = 120  # must match frontend SubmissionForm TAKEAWAY_MAX


def count_words(text: str) -> int:
    """Whitespace-delimited word count. Must match frontend/src/lib/words.ts."""
    return len(text.split())


class Submission(BaseModel):
    id: str
    header: str
    content: str
    takeaway: str


class Topic(BaseModel):
    topic_id: str
    title: str
    fixed_content: str
    submissions: list[Submission] = Field(default_factory=list)


class TopicSummary(BaseModel):
    topic_id: str
    title: str


class TopicDetail(BaseModel):
    """Topic as exposed to the UI: no submissions."""

    topic_id: str
    title: str
    fixed_content: str


class ScoreRequest(BaseModel):
    topic_id: str
    header: str
    content: str
    takeaway: str

    @field_validator("header", "takeaway", "content")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must be non-empty")
        return v


class Breakdown(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)  # ABSOLUTE: cos(content, fixed_content), clipped to 0-1
    novelty: float = Field(ge=0.0, le=1.0)  # percentile vs this topic's corpus (LOO)
    recombination_novelty: float = Field(ge=0.0, le=1.0)  # percentile vs this topic's corpus (LOO)


class Flags(BaseModel):
    duplicate: bool = False
    low_relevance: bool = False
    gaming: bool = False


class SurfacedSentence(BaseModel):
    text: str
    label: Literal["novel", "partial", "redundant"]
    redundancy: float  # max cosine to any corpus sentence
    novelty_contribution: float  # 1 - redundancy
    closest_id: str | None  # corpus submission containing the closest sentence
    closest_snippet: str | None


class Surfacing(BaseModel):
    """Redundancy-aware surfacing: which sentences are new vs already said. Explanatory only."""

    net_new_ratio: float = Field(ge=0.0, le=1.0)  # word-weighted share of novel + partial sentences
    thresholds: dict[str, float]  # {"novel_max_redundancy", "redundant_min_redundancy"}
    sentences: list[SurfacedSentence]


class ScoreResponse(BaseModel):
    system_score: float = Field(ge=0.0, le=1.0)
    baseline_score: float = Field(ge=0.0, le=1.0)
    llm_score: float = Field(ge=0.0, le=1.0)
    breakdown: Breakdown
    flags: Flags
    rationale: str
    surfacing: Surfacing
    # Where llm_score came from: "live", "cached", or "unavailable" (then llm_score is 0.0 and
    # carries no information; clients must not display it as a score).
    llm_source: Literal["live", "cached", "unavailable"]
