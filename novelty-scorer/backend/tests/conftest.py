import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import load_fixtures, load_topics  # noqa: E402
from scoring import config  # noqa: E402
from scoring.corpus_cache import build_topic_cache  # noqa: E402

TOPIC_IDS = list(load_topics())


@pytest.fixture(scope="session")
def topics():
    return load_topics()


@pytest.fixture(scope="session")
def fixtures():
    return load_fixtures()


@pytest.fixture(scope="session")
def caches(topics):
    config.set_seeds()
    return {tid: build_topic_cache(t) for tid, t in topics.items()}
