"""Local text embeddings (fastembed / ONNX, no network after first model download).

All vectors are L2-normalized, so a dot product is cosine similarity. Results are
memoized per (model, text) in-process; inference is deterministic, so the memo
only saves time and never changes a result.
"""

import logging
from functools import lru_cache

import numpy as np

from scoring import config

log = logging.getLogger(__name__)

_memo: dict[tuple[str, str], np.ndarray] = {}


@lru_cache(maxsize=4)
def _model(name: str):
    from fastembed import TextEmbedding

    if name in config.CUSTOM_EMBED_MODELS and name not in {m["model"] for m in TextEmbedding.list_supported_models()}:
        from fastembed.common.model_description import ModelSource, PoolingType

        spec = config.CUSTOM_EMBED_MODELS[name]
        TextEmbedding.add_custom_model(
            model=name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=name),
            dim=spec["dim"],
            model_file=spec["model_file"],
        )
    config.EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading embedding model %s", name)
    return TextEmbedding(name, cache_dir=str(config.EMBED_CACHE_DIR))


def embed(texts: list[str], model: str | None = None) -> np.ndarray:
    """Embed texts with `model` (default config.EMBED_MODEL) -> (len(texts), dim) float32 unit vectors."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    name = model or config.EMBED_MODEL
    missing = list(dict.fromkeys(t for t in texts if (name, t) not in _memo))
    if missing:
        vecs = np.asarray(list(_model(name).embed(missing)), dtype=np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        for t, v in zip(missing, vecs):
            _memo[(name, t)] = v
    return np.stack([_memo[(name, t)] for t in texts])


def clear_memo() -> None:
    """Drop memoized vectors and loaded models (used by the determinism check)."""
    _memo.clear()
    _model.cache_clear()


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of unit vector(s) `a` against rows of `b`."""
    return b @ a
