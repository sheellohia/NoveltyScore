import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from data_loader import get_topic, load_fixtures, load_topics
from models import (
    MAX_CONTENT_WORDS,
    MAX_TAKEAWAY_CHARS,
    ScoreRequest,
    ScoreResponse,
    TopicDetail,
    TopicSummary,
    count_words,
)
from scoring.novelty import score
from scoring import config, corpus_cache
from scoring.config import CORPUS_DIR, FIXTURES_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     [%(name)s] %(message)s")
log = logging.getLogger("novelty")


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.set_seeds()
    topics = load_topics()  # fail fast on bad data files
    log.info("Loaded %d topics from %s", len(topics), CORPUS_DIR)
    for t in topics.values():
        log.info("  %-20s %3d submissions  %r", t.topic_id, len(t.submissions), t.title)
    fixtures = load_fixtures()
    log.info("Loaded fixtures for %d topics from %s", len(fixtures), FIXTURES_PATH)
    corpus_cache.build_all()  # embeddings + LOO calibration + fixture validation
    yield


app = FastAPI(title="Novelty Scorer", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Vite dev server (default :5173, any local port allowed for side-by-side dev)
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Flatten pydantic errors into a single readable message; contract uses 400, not 422.
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p != "body")
        msg = err["msg"].removeprefix("Value error, ")
        parts.append(f"{field}: {msg}" if field else msg)
    return JSONResponse(status_code=400, content={"detail": "; ".join(parts)})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "topics": len(load_topics())}


@app.get("/api/topics", response_model=list[TopicSummary])
def list_topics() -> list[TopicSummary]:
    return [TopicSummary(topic_id=t.topic_id, title=t.title) for t in load_topics().values()]


@app.get("/api/topics/{topic_id}", response_model=TopicDetail)
def topic_detail(topic_id: str) -> TopicDetail:
    topic = get_topic(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Unknown topic_id {topic_id!r}")
    return TopicDetail(**topic.model_dump(exclude={"submissions"}))


@app.post("/api/score", response_model=ScoreResponse)
def score_submission(req: ScoreRequest) -> ScoreResponse:
    topic = get_topic(req.topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Unknown topic_id {req.topic_id!r}")

    words = count_words(req.content)
    if words > MAX_CONTENT_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"content is {words} words; maximum is {MAX_CONTENT_WORDS}",
        )
    if len(req.takeaway) > MAX_TAKEAWAY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"takeaway is {len(req.takeaway)} characters; maximum is {MAX_TAKEAWAY_CHARS}",
        )

    return score(req.model_dump(), topic.model_dump())
