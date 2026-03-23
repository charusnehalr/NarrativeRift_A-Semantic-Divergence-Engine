"""
embedder.py — BGE-M3 dense embedding via FlagEmbedding.

Uses only dense vectors (1024-dim) for ChromaDB storage.
Sparse and ColBERT vectors are not used — they would violate ChromaDB's
flat-scalar metadata constraint and are unnecessary for this pipeline.
"""

import logging
from typing import Optional

import numpy as np
from tqdm import tqdm

import config

logger = logging.getLogger("sde.embedder")

# ── Singleton model ────────────────────────────────────────────────────────────
_bge_model = None


def load_bge_model():
    """
    Load BAAI/bge-m3 via FlagEmbedding. Singleton — loads once, reused.

    Memory: ~4GB RAM on CPU with fp16=True.
    First load downloads ~2.3GB from HuggingFace (cached afterward).
    """
    global _bge_model
    if _bge_model is not None:
        return _bge_model

    logger.info("Loading BGE-M3 model '%s' on device='%s'...", config.BGE_MODEL_NAME, config.BGE_DEVICE)
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        raise ImportError(
            "FlagEmbedding is not installed. Run: pip install FlagEmbedding"
        )

    _bge_model = BGEM3FlagModel(
        config.BGE_MODEL_NAME,
        use_fp16=config.BGE_USE_FP16,
        device=config.BGE_DEVICE,
    )
    logger.info("BGE-M3 loaded successfully.")
    return _bge_model


def embed_texts(
    texts: list[str],
    batch_size: int = None,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of text strings using BGE-M3 dense vectors.

    Args:
        texts: list of strings to embed
        batch_size: override config.BGE_BATCH_SIZE if set
        show_progress: show tqdm progress bar

    Returns:
        list of 1024-dim float lists (ChromaDB-compatible)
    """
    if not texts:
        return []

    model = load_bge_model()
    bs = batch_size or config.BGE_BATCH_SIZE
    all_vectors = []

    batches = [texts[i: i + bs] for i in range(0, len(texts), bs)]
    iterator = tqdm(batches, desc="Embedding", unit="batch") if show_progress else batches

    for batch in iterator:
        output = model.encode(
            batch,
            batch_size=len(batch),
            max_length=config.BGE_MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # output['dense_vecs'] is a numpy array of shape (batch_size, 1024)
        dense: np.ndarray = output["dense_vecs"]
        all_vectors.extend(dense.tolist())  # convert numpy → list[list[float]]

    logger.debug("Embedded %d texts → %d vectors", len(texts), len(all_vectors))
    return all_vectors


def embed_single(text: str) -> list[float]:
    """Embed a single text string. Returns a 1024-dim float list."""
    return embed_texts([text], show_progress=False)[0]


def embed_propositions_for_article(
    article_id: str,
    propositions: list[dict],
    article_meta: Optional[dict] = None,
) -> tuple[list[str], list[list[float]], list[str], list[dict]]:
    """
    Prepare ChromaDB-ready (ids, embeddings, documents, metadatas) for one article.

    Args:
        article_id: the article identifier
        propositions: list of proposition dicts from chunker output
        article_meta: optional article metadata dict from metadata.json

    Returns:
        ids: list of prop_id strings
        embeddings: list of 1024-dim float lists
        documents: list of proposition text strings
        metadatas: list of flat-scalar metadata dicts (ChromaDB constraint)
    """
    if not propositions:
        return [], [], [], []

    texts = [p["text"] for p in propositions]
    embeddings = embed_texts(texts, show_progress=True)

    ids = []
    documents = []
    metadatas = []

    for prop, embedding in zip(propositions, embeddings):
        prop_id = prop["prop_id"]
        ids.append(prop_id)
        documents.append(prop["text"])

        # Build flat metadata — ChromaDB requires str/int/float/bool only
        meta = {
            "article_id": article_id,
            "prop_index": int(prop.get("prop_index", 0)),
            "topic_hint": str(prop.get("topic_hint", "")),
            "is_factual_claim": bool(prop.get("is_factual_claim", True)),
            "char_length": len(prop["text"]),
        }
        if article_meta:
            meta["source_name"] = str(article_meta.get("source_name", ""))
            meta["publish_date"] = str(article_meta.get("publish_date", ""))
            meta["political_lean"] = str(article_meta.get("political_lean", ""))
        metadatas.append(meta)

    return ids, embeddings, documents, metadatas
