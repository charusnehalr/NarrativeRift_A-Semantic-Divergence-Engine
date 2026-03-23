"""
utils.py — Shared utilities: metadata I/O, logging setup, directory management.
All metadata functions accept topic_key so they work for both topics.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import config


def setup_logging(level: str = None) -> logging.Logger:
    """Configure root logger. Call once at startup."""
    log_level = getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("sde")


logger = logging.getLogger("sde.utils")


def ensure_dirs() -> None:
    """Create all required project directories for both topics."""
    for topic_key in config.TOPIC_KEYS:
        tc = config.get_topic_config(topic_key)
        tc["data_dir"].mkdir(parents=True, exist_ok=True)
        tc["articles_dir"].mkdir(parents=True, exist_ok=True)
        tc["propositions_dir"].mkdir(parents=True, exist_ok=True)
        tc["chroma_dir"].mkdir(parents=True, exist_ok=True)
        tc["ground_truth_path"].parent.mkdir(parents=True, exist_ok=True)

    # Initialise empty metadata files if they don't exist
    for topic_key in config.TOPIC_KEYS:
        tc = config.get_topic_config(topic_key)
        if not tc["metadata_path"].exists():
            _write_json(tc["metadata_path"], {"articles": []})
            logger.info("Created empty metadata.json for %s", topic_key)


def load_metadata(topic_key: str) -> dict:
    """Load and return metadata.json for a topic. Returns {'articles': []} if missing."""
    tc = config.get_topic_config(topic_key)
    path = tc["metadata_path"]
    if not path.exists():
        return {"articles": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(topic_key: str, data: dict) -> None:
    """Atomically write metadata.json (write to temp file then rename)."""
    tc = config.get_topic_config(topic_key)
    path = tc["metadata_path"]
    _write_json(path, data)


def get_article(topic_key: str, article_id: str) -> dict | None:
    """Return a single article entry from metadata, or None if not found."""
    meta = load_metadata(topic_key)
    for article in meta["articles"]:
        if article["article_id"] == article_id:
            return article
    return None


def update_article_status(topic_key: str, article_id: str, status: str) -> None:
    """Update the status field of an article and save metadata."""
    meta = load_metadata(topic_key)
    for article in meta["articles"]:
        if article["article_id"] == article_id:
            article["status"] = status
            break
    save_metadata(topic_key, meta)
    logger.debug("Article %s status → %s", article_id, status)


def update_article_field(topic_key: str, article_id: str, field: str, value) -> None:
    """Update any field of an article entry and save metadata."""
    meta = load_metadata(topic_key)
    for article in meta["articles"]:
        if article["article_id"] == article_id:
            article[field] = value
            break
    save_metadata(topic_key, meta)


def register_article(
    topic_key: str,
    filename: str,
    source_name: str,
    publish_date: str,
    political_lean: str,
    url: str = "",
    topic_tags: list[str] | None = None,
) -> str:
    """
    Register an article in metadata.json for the given topic.
    Auto-assigns article_id based on source name and sequence.
    Returns the assigned article_id.
    Raises ValueError if the filename is already registered.
    """
    meta = load_metadata(topic_key)

    # Check for duplicate filename
    existing_filenames = {a["filename"] for a in meta["articles"]}
    if filename in existing_filenames:
        raise ValueError(f"Article '{filename}' is already registered in {topic_key}")

    # Derive article_id from source name prefix + sequence
    source_prefix = source_name.lower().replace(" ", "_")[:12]
    same_source = [a for a in meta["articles"] if a["source_name"] == source_name]
    seq = len(same_source) + 1
    article_id = f"{source_prefix}_{seq:03d}"

    # Ensure unique article_id
    existing_ids = {a["article_id"] for a in meta["articles"]}
    while article_id in existing_ids:
        seq += 1
        article_id = f"{source_prefix}_{seq:03d}"

    entry = {
        "article_id": article_id,
        "filename": filename,
        "source_name": source_name,
        "publish_date": publish_date,
        "political_lean": political_lean,
        "url": url,
        "topic_tags": topic_tags or [],
        "proposition_count": 0,
        "status": "raw",
        "date_registered": datetime.now(timezone.utc).isoformat(),
    }
    meta["articles"].append(entry)
    save_metadata(topic_key, meta)
    logger.info("Registered article %s (%s) for %s", article_id, filename, topic_key)
    return article_id


def get_articles_by_status(topic_key: str, status: str) -> list[dict]:
    """Return all articles with the given status for a topic."""
    meta = load_metadata(topic_key)
    return [a for a in meta["articles"] if a["status"] == status]


def load_propositions(topic_key: str, article_id: str) -> list[dict]:
    """Load saved propositions JSON for an article. Returns [] if not found."""
    tc = config.get_topic_config(topic_key)
    path = tc["propositions_dir"] / f"props_{article_id}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("propositions", [])


def load_all_propositions(topic_key: str) -> list[dict]:
    """Load and merge all proposition JSONs for a topic into a flat list."""
    tc = config.get_topic_config(topic_key)
    all_props = []
    for path in sorted(tc["propositions_dir"].glob("props_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_props.extend(data.get("propositions", []))
    return all_props


def load_contradiction_pairs(topic_key: str) -> list[dict]:
    """Load contradiction_pairs.json for a topic. Returns [] if not found."""
    tc = config.get_topic_config(topic_key)
    path = tc["contradiction_pairs_path"]
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pairs", [])


def save_contradiction_pairs(topic_key: str, pairs: list[dict]) -> None:
    """Save contradiction pairs to JSON with a timestamp."""
    tc = config.get_topic_config(topic_key)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_pairs": len(pairs),
        "pairs": pairs,
    }
    _write_json(tc["contradiction_pairs_path"], data)
    logger.info("Saved %d contradiction pairs for %s", len(pairs), topic_key)


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _write_json(path: Path, data: dict) -> None:
    """Atomically write JSON to path using a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
