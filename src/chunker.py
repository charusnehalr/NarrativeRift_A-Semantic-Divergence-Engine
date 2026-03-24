"""
chunker.py — Proposition-level chunking via Gemini Flash.

This is atomic propositions
For each article, calls Gemini to extract atomic, self-contained factual
propositions (Dense X Retrieval style). Saves results to per-article JSON
and updates metadata status: raw → chunked.
"""

import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
from src.utils import (
    get_articles_by_status,
    load_metadata,
    update_article_field,
    update_article_status,
    _write_json,
)

logger = logging.getLogger("sde.chunker")

# ── Gemini client (initialised lazily) ────────────────────────────────────────
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.template to .env and add your key."
            )
        _gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
        logger.info("Gemini client initialised (model: %s)", config.GEMINI_MODEL_NAME)
    return _gemini_client


# ── Prompt ────────────────────────────────────────────────────────────────────

CHUNKING_PROMPT_TEMPLATE = """\
You are a proposition extractor. Extract every factual claim from the article below \
as atomic, self-contained propositions.

Rules:
- Each proposition must be a single declarative sentence containing exactly one claim.
- Each proposition must be self-contained: replace all pronouns with proper names/entities.
- Skip pure opinions, editorials, and rhetorical questions.
- If an opinion is stated as a reported fact (e.g. "The ministry said X"), include it.
- Maximum 60 propositions per article.

Return ONLY valid JSON matching this exact schema (no extra text):
{{
  "propositions": [
    {{
      "text": "<the proposition>",
      "is_factual_claim": true,
      "topic_hint": "<2-4 word topic label>",
      "source_sentence": "<verbatim sentence from article that contains this claim>"
    }}
  ]
}}

Article:
{article_text}
"""


def build_chunking_prompt(article_text: str) -> str:
    return CHUNKING_PROMPT_TEMPLATE.format(article_text=article_text.strip())


# ── Gemini call with retry ────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(config.GEMINI_MAX_RETRIES),
    reraise=True,
)
def _call_gemini_raw(prompt: str) -> str:
    """Call Gemini and return the raw text response. Retries on any error."""
    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=config.GEMINI_TEMPERATURE,
            ),
        )
        # Rate-limit: sleep to stay under 15 RPM
        time.sleep(config.GEMINI_SLEEP_BETWEEN_CALLS)
        return response.text
    except Exception as exc:
        logger.warning("Gemini call failed (%s), will retry...", exc)
        raise


def call_gemini_chunker(prompt: str) -> dict:
    """
    Call Gemini for proposition extraction.
    Returns parsed dict with 'propositions' key.
    Raises RuntimeError if JSON cannot be parsed after retries.
    """
    raw_text = _call_gemini_raw(prompt)
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Sometimes Gemini wraps JSON in markdown code fences
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise RuntimeError(
                f"Gemini returned non-JSON response: {raw_text[:300]}"
            ) from exc
    return data


# ── Proposition parsing ───────────────────────────────────────────────────────

def parse_proposition_response(raw: dict, article_id: str) -> list[dict]:
    """
    Validate and normalise Gemini's JSON output.
    Assigns prop_id = '{article_id}_prop_{i:03d}'.
    Filters out entries where is_factual_claim is explicitly False.
    Returns list of clean proposition dicts.
    """
    raw_props = raw.get("propositions", [])
    if not isinstance(raw_props, list):
        logger.warning("Unexpected 'propositions' type: %s", type(raw_props))
        return []

    result = []
    for i, item in enumerate(raw_props):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        is_factual = item.get("is_factual_claim", True)
        if is_factual is False:
            continue  # skip explicitly non-factual items

        result.append({
            "prop_id": f"{article_id}_prop_{i:03d}",
            "text": text,
            "is_factual_claim": bool(is_factual),
            "topic_hint": str(item.get("topic_hint", "")).strip(),
            "source_sentence": str(item.get("source_sentence", "")).strip(),
            "prop_index": i,
            "article_id": article_id,
        })

    logger.debug("Parsed %d propositions for article %s", len(result), article_id)
    return result


# ── Per-article entry point ───────────────────────────────────────────────────

def chunk_article(topic_key: str, article_id: str) -> list[dict]:
    """
    Main entry point: chunk one article into propositions.

    1. Reads the .txt file from data/{topic_key}/articles/
    2. Calls Gemini
    3. Saves props to data/{topic_key}/propositions/props_{article_id}.json
    4. Updates metadata status: raw → chunked

    Returns list of proposition dicts.
    Raises FileNotFoundError if article .txt does not exist.
    """
    tc = config.get_topic_config(topic_key)

    # Load article metadata to get filename
    meta = load_metadata(topic_key)
    article_meta = next(
        (a for a in meta["articles"] if a["article_id"] == article_id), None
    )
    if article_meta is None:
        raise ValueError(
            f"Article '{article_id}' not found in {topic_key} metadata. "
            "Register it first with utils.register_article()."
        )

    article_path = tc["articles_dir"] / article_meta["filename"]
    if not article_path.exists():
        raise FileNotFoundError(
            f"Article text file not found: {article_path}. "
            "Place the .txt file in data/{topic_key}/articles/."
        )

    logger.info("Chunking article %s (%s)...", article_id, article_meta["filename"])
    article_text = article_path.read_text(encoding="utf-8")

    try:
        prompt = build_chunking_prompt(article_text)
        raw = call_gemini_chunker(prompt)
        propositions = parse_proposition_response(raw, article_id)
    except Exception as exc:
        update_article_status(topic_key, article_id, "error:chunking")
        logger.error("Chunking failed for %s: %s", article_id, exc)
        raise

    # Save intermediate JSON
    props_path = tc["propositions_dir"] / f"props_{article_id}.json"
    _write_json(props_path, {
        "article_id": article_id,
        "source_name": article_meta.get("source_name", ""),
        "total_propositions": len(propositions),
        "propositions": propositions,
    })

    # Update metadata
    update_article_status(topic_key, article_id, "chunked")
    update_article_field(topic_key, article_id, "proposition_count", len(propositions))

    logger.info(
        "Chunked %s → %d propositions (saved to %s)",
        article_id,
        len(propositions),
        props_path.name,
    )
    return propositions


# ── Batch entry point ─────────────────────────────────────────────────────────

def chunk_all_articles(topic_key: str, skip_existing: bool = True, max_articles: int = None) -> dict:
    """
    Chunk all articles with status 'raw' for a topic.

    Args:
        topic_key: 'topic_a' or 'topic_b'
        skip_existing: if True, skip articles already with status 'chunked' or 'embedded'

    Returns:
        dict mapping article_id → proposition count (or error string on failure)
    """
    skip_statuses = {"chunked", "embedded"} if skip_existing else set()

    meta = load_metadata(topic_key)
    to_process = [
        a for a in meta["articles"]
        if a["status"] not in skip_statuses and not a["status"].startswith("error")
    ]

    if not to_process:
        logger.info("No articles to chunk for %s (all up-to-date or errored)", topic_key)
        return {}

    if max_articles:
        to_process = to_process[:max_articles]

    logger.info("Chunking %d articles for %s...", len(to_process), topic_key)
    results = {}
    for article in to_process:
        article_id = article["article_id"]
        try:
            props = chunk_article(topic_key, article_id)
            results[article_id] = len(props)
        except Exception as exc:
            logger.error("Skipping %s due to error: %s", article_id, exc)
            results[article_id] = f"ERROR: {exc}"

    logger.info(
        "Chunking complete for %s. Results: %s",
        topic_key,
        {k: v for k, v in results.items()},
    )
    return results


# ── Quick test ────────────────────────────────────────────────────────────────
# Run from project root:
#   python src/chunker.py

if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Chunk articles into propositions")
    parser.add_argument("--mode", choices=["single", "all"], required=True, help="single: one article | all: batch")
    parser.add_argument("--article", default="aljazeera_001", help="article_id for single mode")
    parser.add_argument("--max", type=int, default=None, help="max articles to process in all mode")
    parser.add_argument("--topic", default="topic_a", help="topic key")
    args = parser.parse_args()

    if args.mode == "single":
        print(f"\nChunking '{args.article}' ...\n")
        props = chunk_article(args.topic, args.article)
        print(f"\n--- {len(props)} propositions extracted ---\n")
        for p in props:
            print(f"[{p['prop_id']}] ({p['topic_hint']})")
            print(f"  {p['text']}")
            print()

    elif args.mode == "all":
        print(f"\nChunking articles for '{args.topic}' (max={args.max}) ...\n")
        results = chunk_all_articles(args.topic, skip_existing=True, max_articles=args.max)
        print("\n--- Results ---")
        for article_id, result in results.items():
            print(f"  {article_id}: {result}")
