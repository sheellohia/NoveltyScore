"""LLM-as-judge novelty scoring via Gemini.

This is the *benchmark* the system score is compared against, not part of the
system itself. Prompts live in scoring/prompts/ so they can be tuned without
touching code.

Two protocols, one rubric (prompts/judge_system.md):
  judge()         live: one new submission vs the whole topic pool.
  judge_corpus()  offline: every corpus item vs the other N-1, in ONE call per topic
                  (listwise LOO). Used by experiments/ to get a judge score for all
                  corpus items within free-tier quota.

Every verdict is persisted to results/judge_cache.json and never re-requested.
Cache key: "{protocol}:{topic_id}:{submission_hash}:{context_hash}". The context
hash covers the prompts and the pool, so editing either invalidates stale entries.

Config (backend/.env):
  GEMINI_API_KEY   required; without it the judge raises JudgeUnavailable
  GEMINI_MODEL     optional; primary model (default below). Fallbacks are tried
                   on overload/quota errors.
"""

import hashlib
import html
import json
import logging
import os
import random
import re
import threading
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scoring import config

log = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_MODEL = "gemini-3.7-flash"
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3-flash-preview"]
TIMEOUT_MS = 45_000
CORPUS_TIMEOUT_MS = 300_000
RETRYABLE = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE", "timed out")


class JudgeUnavailable(RuntimeError):
    """Judge could not produce a verdict. `reason` is a short, user-safe label; the full
    error text stays in the exception message / logs and is never shown to users."""

    def __init__(self, message: str, reason: str = "error"):
        super().__init__(message)
        self.reason = reason


def _reason(messages: list[str]) -> str:
    text = " ".join(messages)
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return "rate-limited"
    if "DEADLINE" in text or "timed out" in text.lower() or "timeout" in text.lower():
        return "timed out"
    if "503" in text or "UNAVAILABLE" in text:
        return "overloaded"
    if "unparseable" in text:
        return "invalid output"
    return "error"


class JudgeVerdict(BaseModel):
    """Structured output for one submission (see prompts/judge_system.md)."""

    core_claim: str
    closest_submission_ids: list[str] = Field(default_factory=list)
    relevance: int = Field(ge=0, le=10)
    novelty_vs_fixed: int = Field(ge=0, le=10)
    novelty_vs_pool: int = Field(ge=0, le=10)
    duplicate: bool
    low_relevance: bool
    gaming: bool
    overall_novelty: int = Field(ge=0, le=100)
    rationale: str


class CorpusItemVerdict(BaseModel):
    id: str
    closest_submission_ids: list[str] = Field(default_factory=list)
    relevance: int = Field(ge=0, le=10)
    novelty_vs_pool: int = Field(ge=0, le=10)
    duplicate: bool
    low_relevance: bool
    gaming: bool
    overall_novelty: int = Field(ge=0, le=100)
    rationale: str


class CorpusVerdicts(BaseModel):
    items: list[CorpusItemVerdict]


class JudgeResult(BaseModel):
    score: float  # 0-1, from overall_novelty after hard caps
    verdict: JudgeVerdict
    model: str
    source: str = "live"  # "live" = produced by this call, "cache" = earlier verdict for this exact input


# --- prompts ----------------------------------------------------------------


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    return {
        name: (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
        for name in ("judge_system", "judge_user", "judge_corpus_user")
    }


def _esc(text: str) -> str:
    # Stops submitted text from closing our <tags> and posing as instructions.
    return html.escape(text.strip(), quote=False)


def _render(template: str, values: dict[str, str]) -> str:
    # Single pass, so placeholder-looking text inside user data is never expanded.
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], template)


def _format_pool(submissions: list[dict]) -> str:
    if not submissions:
        return "(none yet)"
    return "\n\n".join(
        f"[id={_esc(s['id'])}]\n"
        f"Header: {_esc(s['header'])}\n"
        f"Content: {_esc(s['content'])}\n"
        f"Takeaway: {_esc(s['takeaway'])}"
        for s in submissions
    )


def _topic_values(topic: dict, pool: list[dict]) -> dict[str, str]:
    return {
        "topic_title": _esc(topic["title"]),
        "fixed_content": _esc(topic["fixed_content"]),
        "pool_size": str(len(pool)),
        "existing_submissions": _format_pool(pool),
    }


def build_prompt(submission: dict, topic: dict) -> tuple[str, str]:
    """Return (system_instruction, user_message) for the live protocol."""
    p = _prompts()
    values = _topic_values(topic, topic.get("submissions", []))
    values.update({k: _esc(submission[k]) for k in ("header", "content", "takeaway")})
    return p["judge_system"], _render(p["judge_user"], values)


def build_corpus_prompt(topic: dict, order: list[dict]) -> tuple[str, str]:
    p = _prompts()
    return p["judge_system"], _render(p["judge_corpus_user"], _topic_values(topic, order))


def _cap(overall: int, relevance: int, duplicate: bool, gaming: bool) -> int:
    """Enforce the prompt's hard rules in code, in case the model drifts."""
    if relevance <= 3:
        overall = min(overall, 15)
    if duplicate or gaming:
        overall = min(overall, 10)
    return overall


# --- persistent cache ---------------------------------------------------------

_lock = threading.Lock()
_disk: dict | None = None


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def submission_hash(sub: dict) -> str:
    return _sha("\x1f".join(sub.get(k, "") for k in ("header", "content", "takeaway")))


def _context_hash(topic: dict) -> str:
    p = _prompts()
    pool = json.dumps(
        [{k: s[k] for k in ("id", "header", "content", "takeaway")} for s in topic.get("submissions", [])],
        sort_keys=True,
    )
    return _sha(p["judge_system"] + p["judge_user"] + p["judge_corpus_user"] + topic["fixed_content"] + pool, 10)


def _cache() -> dict:
    global _disk
    if _disk is None:
        path = config.JUDGE_CACHE_PATH
        _disk = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _disk


def _cache_put(entries: dict[str, dict]) -> None:
    """Merge entries into the on-disk cache. Several processes (API server, experiment
    runs) may write, so re-read under an exclusive file lock and merge, never overwrite."""
    import fcntl

    path = config.JUDGE_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path.with_suffix(".lock"), "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        on_disk = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        on_disk.update(entries)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(on_disk, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        _cache().update(on_disk)


def cached_corpus_with_fallback(topic: dict, run: int, allow_current: bool = True) -> tuple[dict[str, dict] | None, str]:
    """Cached corpus verdicts for a run, never calling the API.

    Returns (results, version): version "current" when produced by today's prompts and
    pool, "legacy:<context_hash>" when only an older complete set exists (e.g. verdicts
    made before a prompt edit), or (None, "missing"). Legacy sets are found by the item
    ids stored in each verdict, since their keys hashed prompts that no longer exist.
    """
    cur = cached_corpus(topic, run) if allow_current else None
    if cur is not None:
        return cur, "current"
    path = config.JUDGE_CACHE_PATH
    disk = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ids = {s["id"] for s in topic["submissions"]}
    prefix = f"corpus-r{run}:{topic['topic_id']}:"
    by_ctx: dict[str, dict[str, dict]] = {}
    for k, v in disk.items():
        if k.startswith(prefix):
            by_ctx.setdefault(k.rsplit(":", 1)[1], {})[v["verdict"]["id"]] = v
    complete = {ctx: m for ctx, m in by_ctx.items() if ids <= set(m)}
    if not complete:
        return None, "missing"
    ctx = sorted(complete)[0]
    return {sid: complete[ctx][sid] for sid in ids}, f"legacy:{ctx}"


def cached_corpus(topic: dict, run: int) -> dict[str, dict] | None:
    """Corpus-judge results for a run if fully cached, else None. Never calls the API."""
    keys = {s["id"]: corpus_key(s, topic, run) for s in topic["submissions"]}
    path = config.JUDGE_CACHE_PATH
    disk = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if all(k in disk for k in keys.values()):
        return {sid: disk[k] for sid, k in keys.items()}
    return None


def live_key(submission: dict, topic: dict) -> str:
    return f"live:{topic['topic_id']}:{submission_hash(submission)}:{_context_hash(topic)}"


def corpus_key(sub: dict, topic: dict, run: int) -> str:
    return f"corpus-r{run}:{topic['topic_id']}:{submission_hash(sub)}:{_context_hash(topic)}"


# --- API calls ------------------------------------------------------------------


@lru_cache(maxsize=1)
def _client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise JudgeUnavailable("GEMINI_API_KEY is not set in backend/.env", reason="not configured")
    from google import genai

    return genai.Client(api_key=key, http_options={"timeout": TIMEOUT_MS})


def _models() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    return [primary] + [m for m in FALLBACK_MODELS if m != primary]


def _generate(system: str, user: str, schema: type[BaseModel], timeout_ms: int) -> tuple[BaseModel, str]:
    from google.genai import types

    client = _client()
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    errors: list[str] = []
    for model in _models():
        try:
            resp = client.models.generate_content(model=model, contents=user, config=cfg)
        except Exception as e:  # SDK raises typed APIError subclasses; match on text for portability
            msg = str(e)
            errors.append(f"{model}: {msg[:120]}")
            if any(code in msg for code in RETRYABLE):
                log.warning("judge model %s unavailable, trying next: %s", model, msg[:120])
                continue
            break
        try:
            parsed = resp.parsed if isinstance(resp.parsed, schema) else schema.model_validate(json.loads(resp.text))
        except Exception as e:
            errors.append(f"{model}: unparseable output ({e.__class__.__name__})")
            continue
        return parsed, model
    raise JudgeUnavailable("; ".join(errors) or "no models configured", reason=_reason(errors))


def judge(submission: dict, topic: dict) -> JudgeResult:
    """Live protocol: one submission vs the full pool. Raises JudgeUnavailable on failure."""
    key = live_key(submission, topic)
    hit = _cache().get(key)
    if hit:
        return JudgeResult.model_validate({**hit, "source": "cache"})

    system, user = build_prompt(submission, topic)
    verdict, model = _generate(system, user, JudgeVerdict, TIMEOUT_MS)
    capped = _cap(verdict.overall_novelty, verdict.relevance, verdict.duplicate, verdict.gaming)
    result = JudgeResult(score=round(capped / 100, 3), verdict=verdict, model=model)
    log.info("judge %s -> %s %s", model, result.score, verdict.model_dump_json())
    _cache_put({key: result.model_dump(exclude={"source"})})
    return result


def corpus_reference_score(submission: dict, topic: dict) -> tuple[float, str] | None:
    """If the submission's text is exactly a corpus item, return (cached corpus-judge score,
    item id) from run 0 of the listwise protocol. Never calls the API."""
    h = submission_hash(submission)
    match = next((s for s in topic.get("submissions", []) if submission_hash(s) == h), None)
    if match is None:
        return None
    res, _version = cached_corpus_with_fallback(topic, 0)
    return (res[match["id"]]["score"], match["id"]) if res else None


def judge_corpus(topic: dict, run: int = 0) -> dict[str, dict]:
    """Listwise LOO protocol: judge every corpus item against the other N-1 in one call.

    run 0 presents items in file order; run r > 0 uses a seeded shuffle (SEED + r), so
    repeated runs measure the judge's run-to-run variance including order effects.
    Returns {item_id: {"score", "model", "verdict"}}; only calls the API if any item
    for this run is missing from the cache.
    """
    subs = topic["submissions"]
    keys = {s["id"]: corpus_key(s, topic, run) for s in subs}
    cache = _cache()
    if all(k in cache for k in keys.values()):
        return {sid: cache[k] for sid, k in keys.items()}

    order = list(subs)
    if run > 0:
        random.Random(config.SEED + run).shuffle(order)
    system, user = build_corpus_prompt(topic, order)
    out, model = _generate(system, user, CorpusVerdicts, CORPUS_TIMEOUT_MS)

    by_id = {v.id: v for v in out.items}
    missing = [sid for sid in keys if sid not in by_id]
    if missing:
        raise JudgeUnavailable(f"corpus judge omitted {len(missing)} ids for {topic['topic_id']} run {run}: {missing[:5]}")
    entries = {}
    for sid, key in keys.items():
        v = by_id[sid]
        capped = _cap(v.overall_novelty, v.relevance, v.duplicate, v.gaming)
        entries[key] = {"score": round(capped / 100, 3), "model": model, "verdict": v.model_dump()}
    _cache_put(entries)
    log.info("corpus judge %s run %d via %s: %d items", topic["topic_id"], run, model, len(entries))
    return {sid: entries[k] for sid, k in keys.items()}
