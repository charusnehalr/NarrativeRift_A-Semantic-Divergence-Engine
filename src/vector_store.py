"""
vector_store.py — ChromaDB CRUD for proposition embeddings.

Each topic has its own PersistentClient and collection.
All functions accept topic_key so they operate on the correct store.

WINDOWS PATH NOTE: ChromaDB's SQLite backend fails with Windows backslash
paths. Always use chroma_dir_str (via get_topic_config) which uses as_posix().
"""

import logging
from typing import Optional

import chromadb

import config
from src.embedder import embed_propositions_for_article, embed_single
from src.utils import (
    load_metadata,
    load_propositions,
    update_article_status,
    get_articles_by_status,
)

logger = logging.getLogger("sde.vector_store")

# ── Per-topic singletons ───────────────────────────────────────────────────────
_clients: dict[str, chromadb.PersistentClient] = {}
_collections: dict[str, chromadb.Collection] = {}


def get_chroma_client(topic_key: str) -> chromadb.PersistentClient:
    """
    Return a PersistentClient for the given topic. Singleton per topic.
    Uses posix path string to avoid Windows backslash issues.
    """
    if topic_key not in _clients:
        tc = config.get_topic_config(topic_key)
        path_str = tc["chroma_dir_str"]
        logger.info("Opening ChromaDB for %s at %s", topic_key, path_str)
        _clients[topic_key] = chromadb.PersistentClient(path=path_str)
    return _clients[topic_key]


def get_or_create_collection(topic_key: str) -> chromadb.Collection:
    """
    Get or create the propositions collection for a topic.
    hnsw:space MUST be 'cosine' — set at creation, cannot be changed later.
    """
    if topic_key not in _collections:
        tc = config.get_topic_config(topic_key)
        client = get_chroma_client(topic_key)
        _collections[topic_key] = client.get_or_create_collection(
            name=tc["chroma_collection"],
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready (%d documents)",
            tc["chroma_collection"],
            _collections[topic_key].count(),
        )
    return _collections[topic_key]


def upsert_propositions(
    topic_key: str,
    article_id: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> int:
    """
    Upsert propositions into ChromaDB for a topic.
    Uses upsert (not add) so re-runs are idempotent.
    Returns count of upserted documents.
    """
    if not ids:
        return 0
    collection = get_or_create_collection(topic_key)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Upserted %d propositions for article %s", len(ids), article_id)
    return len(ids)


def query_similar(
    topic_key: str,
    query_embedding: list[float],
    top_k: int = None,
    where_filter: Optional[dict] = None,
) -> dict:
    """
    Query ChromaDB for the most similar propositions.

    Args:
        topic_key: 'topic_a' or 'topic_b'
        query_embedding: 1024-dim float list
        top_k: number of results (defaults to config.TOP_K_RETRIEVAL)
        where_filter: optional ChromaDB metadata filter, e.g. {"source_name": "Reuters"}

    Returns:
        ChromaDB QueryResult dict with keys: ids, documents, metadatas, distances
        distances are cosine distances (0 = identical, 2 = opposite)
        cosine similarity = 1 - distance
    """
    collection = get_or_create_collection(topic_key)
    k = top_k or config.TOP_K_RETRIEVAL

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    return collection.query(**kwargs)


def query_similar_text(
    topic_key: str,
    query_text: str,
    top_k: int = None,
    where_filter: Optional[dict] = None,
) -> list[dict]:
    """
    Convenience wrapper: embed query_text then run query_similar.
    Returns a flat list of result dicts with added 'cosine_similarity' field.
    """
    embedding = embed_single(query_text)
    raw = query_similar(topic_key, embedding, top_k=top_k, where_filter=where_filter)
    return _flatten_query_results(raw)


def _flatten_query_results(raw: dict) -> list[dict]:
    """Convert ChromaDB QueryResult (nested lists) to flat list of dicts."""
    results = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    for prop_id, doc, meta, dist in zip(ids, docs, metas, dists):
        results.append({
            "prop_id": prop_id,
            "text": doc,
            "cosine_similarity": round(1.0 - dist, 4),
            **meta,
        })
    return results


def get_all_propositions(topic_key: str, article_id: Optional[str] = None) -> list[dict]:
    """
    Retrieve all propositions from ChromaDB, optionally filtered by article_id.
    Returns list of flat dicts with text + metadata fields.
    """
    collection = get_or_create_collection(topic_key)
    count = collection.count()
    if count == 0:
        return []

    kwargs = {"limit": count, "include": ["documents", "metadatas"]}
    if article_id:
        kwargs["where"] = {"article_id": article_id}

    raw = collection.get(**kwargs)
    results = []
    for prop_id, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        results.append({"prop_id": prop_id, "text": doc, **meta})
    return results


def delete_article_propositions(topic_key: str, article_id: str) -> int:
    """Delete all propositions for an article from ChromaDB. Returns count deleted."""
    collection = get_or_create_collection(topic_key)
    existing = collection.get(where={"article_id": article_id})
    ids_to_delete = existing.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    logger.info("Deleted %d propositions for article %s", len(ids_to_delete), article_id)
    return len(ids_to_delete)


def ingest_article_embeddings(topic_key: str, article_id: str) -> int:
    """
    Main entry point: load propositions JSON, embed with BGE-M3, upsert to ChromaDB.
    Updates metadata status: chunked → embedded.
    Returns proposition count.
    """
    meta = load_metadata(topic_key)
    article_meta = next(
        (a for a in meta["articles"] if a["article_id"] == article_id), None
    )
    if article_meta is None:
        raise ValueError(f"Article '{article_id}' not found in {topic_key} metadata.")

    propositions = load_propositions(topic_key, article_id)
    if not propositions:
        raise ValueError(
            f"No propositions found for {article_id}. Run chunking first."
        )

    logger.info(
        "Embedding %d propositions for %s...", len(propositions), article_id
    )

    try:
        ids, embeddings, documents, metadatas = embed_propositions_for_article(
            article_id, propositions, article_meta
        )
        count = upsert_propositions(
            topic_key, article_id, ids, embeddings, documents, metadatas
        )
    except Exception as exc:
        update_article_status(topic_key, article_id, "error:embedding")
        logger.error("Embedding failed for %s: %s", article_id, exc)
        raise

    update_article_status(topic_key, article_id, "embedded")
    logger.info("Article %s embedded (%d propositions).", article_id, count)
    return count


def ingest_all_articles(topic_key: str, skip_existing: bool = True) -> dict:
    """
    Embed all articles with status 'chunked' for a topic.
    Returns {article_id: count_or_error_string}.
    """
    meta = load_metadata(topic_key)
    skip_statuses = {"embedded"} if skip_existing else set()

    to_process = [
        a for a in meta["articles"]
        if a["status"] == "chunked"
        or (not skip_existing and a["status"] not in skip_statuses)
    ]
    # Only process chunked articles (embedding requires propositions to exist)
    to_process = [a for a in meta["articles"] if a["status"] == "chunked"]

    if not to_process:
        logger.info("No articles to embed for %s.", topic_key)
        return {}

    logger.info("Embedding %d articles for %s...", len(to_process), topic_key)
    results = {}
    for article in to_process:
        article_id = article["article_id"]
        try:
            count = ingest_article_embeddings(topic_key, article_id)
            results[article_id] = count
        except Exception as exc:
            logger.error("Skipping %s: %s", article_id, exc)
            results[article_id] = f"ERROR: {exc}"

    return results


def get_collection_stats(topic_key: str) -> dict:
    """Return basic stats about a topic's ChromaDB collection."""
    collection = get_or_create_collection(topic_key)
    count = collection.count()
    tc = config.get_topic_config(topic_key)
    return {
        "topic_key": topic_key,
        "topic_name": tc["name"],
        "collection_name": tc["chroma_collection"],
        "total_propositions": count,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
# Run from project root:
#   uv run python -m src.vector_store --mode single --article aljazeera_001
#   uv run python -m src.vector_store --mode stats
#   uv run python -m src.vector_store --mode query --query "ceasefire negotiations"

if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="ChromaDB ingestion and inspection")
    parser.add_argument("--mode", choices=["single", "all", "stats", "query"], required=True)
    parser.add_argument("--article", default="aljazeera_001", help="article_id for single mode")
    parser.add_argument("--topic", default="topic_a", help="topic key")
    parser.add_argument("--query", default="ceasefire negotiations", help="query text for query mode")
    args = parser.parse_args()

    if args.mode == "single":
        print(f"\nIngesting '{args.article}' into ChromaDB ...\n")
        count = ingest_article_embeddings(args.topic, args.article)
        print(f"\n--- Stored {count} propositions in ChromaDB ---")
        props = get_all_propositions(args.topic, article_id=args.article)
        print(f"\nSample (first 3 stored):")
        for p in props[:3]:
            print(f"  [{p['prop_id']}] {p['text'][:80]}")
            print(f"    source={p.get('source_name')} | date={p.get('publish_date')}")
            print()

    elif args.mode == "all":
        print(f"\nIngesting all chunked articles for '{args.topic}' ...\n")
        results = ingest_all_articles(args.topic, skip_existing=True)
        for article_id, result in results.items():
            print(f"  {article_id}: {result}")

    elif args.mode == "stats":
        stats = get_collection_stats(args.topic)
        print(f"\n--- ChromaDB Stats ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.mode == "query":
        print(f"\nQuerying ChromaDB for: '{args.query}'\n")
        results = query_similar_text(args.topic, args.query, top_k=5)
        for r in results:
            print(f"  [{r['prop_id']}] similarity={r['cosine_similarity']}")
            print(f"  {r['text'][:80]}")
            print(f"  source={r.get('source_name')} | date={r.get('publish_date')}")
            print()
