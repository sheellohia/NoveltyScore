"""Loads topic corpora from data/corpus/*.json and fixtures from data/fixtures/.

Topics are keyed by the `topic_id` inside each file, not by filename, so adding,
removing or renaming a corpus file needs no code change. Files are read once
and cached; restart the server to pick up changes.
"""

import json
import logging
from functools import lru_cache

from models import Topic
from scoring.config import CORPUS_DIR, FIXTURES_PATH

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_topics() -> dict[str, Topic]:
    """Return topics keyed by topic_id, ordered by filename."""
    topics: dict[str, Topic] = {}
    sources: dict[str, str] = {}
    for path in sorted(CORPUS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            topic = Topic.model_validate(json.load(f))
        if topic.topic_id in topics:
            raise ValueError(
                f"Duplicate topic_id {topic.topic_id!r} in {path.name} and {sources[topic.topic_id]}"
            )
        topics[topic.topic_id] = topic
        sources[topic.topic_id] = path.name
    if not topics:
        log.warning("No corpus files found in %s", CORPUS_DIR)
    return topics


def get_topic(topic_id: str) -> Topic | None:
    return load_topics().get(topic_id)


@lru_cache(maxsize=1)
def load_fixtures() -> dict[str, dict]:
    """Test fixtures keyed by topic_id ({} if the file is absent)."""
    if not FIXTURES_PATH.exists():
        log.warning("Fixtures file not found: %s", FIXTURES_PATH)
        return {}
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        return json.load(f)
