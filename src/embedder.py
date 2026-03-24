"""
embedder.py — BGE-M3 dense embedding via sentence-transformers.

Uses only dense vectors (1024-dim) for ChromaDB storage.
Switched from FlagEmbedding to sentence-transformers due to Python 3.13
compatibility issues in FlagEmbedding. Same model, same vectors.
"""

import logging
from typing import Optional

import config

logger = logging.getLogger("sde.embedder")

# ── Singleton model ────────────────────────────────────────────────────────────
_bge_model = None


def load_bge_model():
    """
    Load BAAI/bge-m3 via sentence-transformers. Singleton — loads once, reused.

    Memory: ~2-4GB RAM. First load downloads model from HuggingFace (cached afterward).
    """
    global _bge_model
    if _bge_model is not None:
        return _bge_model

    logger.info("Loading BGE-M3 model '%s' on device='%s'...", config.BGE_MODEL_NAME, config.BGE_DEVICE)
    from sentence_transformers import SentenceTransformer
    _bge_model = SentenceTransformer(config.BGE_MODEL_NAME, device=config.BGE_DEVICE)
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

    vectors = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return vectors.tolist()  # convert numpy → list[list[float]]


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


# ── Quick test ────────────────────────────────────────────────────────────────
# Run from project root:
#   uv run python -m src.embedder --mode single --article aljazeera_001

if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging, load_propositions, load_metadata

    setup_logging()

    parser = argparse.ArgumentParser(description="Embed propositions with BGE-M3")
    parser.add_argument("--mode", choices=["single", "all"], required=True, help="single: one article | all: batch")
    parser.add_argument("--article", default="aljazeera_001", help="article_id for single mode")
    parser.add_argument("--topic", default="topic_a", help="topic key")
    args = parser.parse_args()

    if args.mode == "single":
        meta = load_metadata(args.topic)
        article_meta = next((a for a in meta["articles"] if a["article_id"] == args.article), None)
        propositions = load_propositions(args.topic, args.article)

        if not propositions:
            print(f"No propositions found for {args.article}. Run chunking first.")
        else:
            print(f"\nEmbedding {len(propositions)} propositions for '{args.article}' ...\n")
            ids, embeddings, documents, metadatas = embed_propositions_for_article(
                args.article, propositions, article_meta
            )
            print(f"\n--- Done ---")
            print(f"  Propositions embedded : {len(embeddings)}")
            print(f"  Vector dimensions     : {len(embeddings[0])}")
            print(f"  Sample vector (first 5 dims): {embeddings[0][:5]}")
