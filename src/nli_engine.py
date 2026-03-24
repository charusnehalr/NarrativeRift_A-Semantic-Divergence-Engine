"""
nli_engine.py — Two-tier contradiction detection pipeline.

Tier 1: cross-encoder/nli-deberta-v3-base (local, free, fast)
Tier 2: Gemini Flash fallback for borderline cases (confidence 0.4–0.7)

Also contains the cross-encoder reranker (ms-marco-MiniLM-L-6-v2) used to
prune ChromaDB candidates before NLI classification.
"""

import json
import logging
import time
from typing import Optional

import numpy as np
from scipy.special import softmax
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config
from src.utils import (
    load_all_propositions,
    load_metadata,
    save_contradiction_pairs,
    now_iso,
)

logger = logging.getLogger("sde.nli_engine")

# ── Singleton models ───────────────────────────────────────────────────────────
_reranker = None
_deberta_nli = None
_groq_client = None


def load_reranker():
    """Load ms-marco-MiniLM reranker. Singleton."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker '%s'...", config.RERANKER_MODEL_NAME)
        _reranker = CrossEncoder(config.RERANKER_MODEL_NAME)
        logger.info("Reranker loaded.")
    return _reranker


def load_deberta_nli():
    """Load nli-deberta-v3-base as CrossEncoder. Singleton."""
    global _deberta_nli
    if _deberta_nli is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading NLI model '%s'...", config.NLI_MODEL_NAME)
        # num_labels=3 → raw logits in order [contradiction, entailment, neutral]
        _deberta_nli = CrossEncoder(config.NLI_MODEL_NAME, num_labels=3)
        logger.info("DeBERTa NLI loaded.")
    return _deberta_nli


def _get_groq_client():
    """Lazy-load Groq client for Tier 2 NLI."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env.")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("Groq client initialised for NLI Tier 2 (model: %s)", config.GROQ_MODEL_NAME)
    return _groq_client


# ── Reranker ──────────────────────────────────────────────────────────────────

def rerank_candidates(
    query_text: str,
    candidate_texts: list[str],
    candidate_ids: list[str],
    top_k: int = None,
) -> list[tuple[str, float]]:
    """
    Score query-candidate pairs with the ms-marco reranker.
    Returns list of (candidate_id, score) sorted descending, top_k items.

    NOTE: reranker scores relevance/semantic similarity, NOT contradiction.
    Purpose: prune ChromaDB top-10 down to top-5 most relevant before NLI.
    """
    if not candidate_texts:
        return []

    reranker = load_reranker()
    k = top_k or config.RERANKER_TOP_K
    pairs = [(query_text, cand) for cand in candidate_texts]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]


# ── DeBERTa NLI ───────────────────────────────────────────────────────────────

def run_deberta_nli(prop_a: str, prop_b: str) -> dict:
    """
    Score a proposition pair with DeBERTa NLI.

    IMPORTANT: CrossEncoder with num_labels=3 returns raw logits.
    We apply softmax to get probabilities.
    Label order: index 0=contradiction, 1=entailment, 2=neutral

    Returns dict with probability for each label + predicted_label + confidence.
    """
    model = load_deberta_nli()
    raw_logits = model.predict([(prop_a, prop_b)])
    probs = softmax(raw_logits[0])  # shape: (3,)

    contradiction_score = float(probs[config.NLI_CONTRADICTION_IDX])
    entailment_score = float(probs[config.NLI_ENTAILMENT_IDX])
    neutral_score = float(probs[config.NLI_NEUTRAL_IDX])

    labels = ["contradiction", "entailment", "neutral"]
    predicted_idx = int(np.argmax(probs))

    return {
        "contradiction": round(contradiction_score, 4),
        "entailment": round(entailment_score, 4),
        "neutral": round(neutral_score, 4),
        "predicted_label": labels[predicted_idx],
        "confidence": round(float(probs[predicted_idx]), 4),
    }


def classify_tier(deberta_result: dict) -> str:
    """
    Route a DeBERTa result to a tier decision.

    Returns:
        'accept'  if contradiction score > NLI_ACCEPT_THRESHOLD (0.7)
        'reject'  if contradiction score < NLI_REJECT_THRESHOLD (0.4)
        'tier2'   if score is in the borderline range [0.4, 0.7]
    """
    score = deberta_result["contradiction"]
    if score >= config.NLI_ACCEPT_THRESHOLD:
        return "accept"
    if score < config.NLI_REJECT_THRESHOLD:
        return "reject"
    return "tier2"


# ── Gemini Tier 2 NLI ─────────────────────────────────────────────────────────

TIER2_PROMPT_TEMPLATE = """\
Analyze each proposition pair below from different news sources about the same event.
For each pair, classify the relationship between the two claims.

Classification options:
- CONTRADICTS: the two claims make directly opposing factual assertions
- AGREES: both claims assert the same fact (possibly worded differently)
- NUANCE: same topic, different framing or emphasis, not directly opposing
- TEMPORAL: one claim may have been true at a different time than the other
- UNRELATED: similar words but actually about different facts

Pairs:
{pairs_text}

Return ONLY a JSON array (no extra text):
[
  {{
    "pair_index": <int>,
    "verdict": "<CONTRADICTS|AGREES|NUANCE|TEMPORAL|UNRELATED>",
    "confidence": <0.0-1.0>,
    "reasoning": "<one sentence explaining the classification>"
  }}
]
"""


def build_gemini_nli_prompt(pairs: list[dict]) -> str:
    """
    Build batched Gemini NLI prompt for up to 5 pairs.
    Each pair dict must have: pair_index, prop_a, prop_b, source_a, date_a, source_b, date_b
    """
    lines = []
    for p in pairs:
        lines.append(
            f"[{p['pair_index']}] "
            f"A ({p.get('source_a', 'unknown')}, {p.get('date_a', '?')}): \"{p['prop_a']}\"\n"
            f"    B ({p.get('source_b', 'unknown')}, {p.get('date_b', '?')}): \"{p['prop_b']}\""
        )
    return TIER2_PROMPT_TEMPLATE.format(pairs_text="\n".join(lines))


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(config.GROQ_MAX_RETRIES),
    reraise=True,
)
def _call_groq_nli_raw(prompt: str) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.GEMINI_TEMPERATURE,
    )
    time.sleep(config.GROQ_SLEEP_BETWEEN_CALLS)
    return response.choices[0].message.content


def call_groq_nli(pairs: list[dict]) -> list[dict]:
    """
    Call Groq for batched NLI classification (Tier 2).
    Returns list of verdict dicts with pair_index, verdict, confidence, reasoning.
    """
    prompt = build_gemini_nli_prompt(pairs)
    raw = _call_groq_nli_raw(prompt)
    import re
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            verdicts = parsed
        elif isinstance(parsed, dict):
            verdicts = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            verdicts = []
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            verdicts = json.loads(match.group(0))
        else:
            logger.warning("Groq NLI returned non-JSON: %s", raw[:300])
            verdicts = [
                {"pair_index": p["pair_index"], "verdict": "UNRELATED",
                 "confidence": 0.5, "reasoning": "Parse error — defaulting to UNRELATED"}
                for p in pairs
            ]
    return verdicts


# ── Full contradiction detection pipeline ─────────────────────────────────────

def detect_contradictions_for_topic(topic_key: str) -> list[dict]:
    """
    Run the full contradiction detection pipeline for a topic.

    For every proposition:
      1. Query ChromaDB for top-K similar propositions (cross-doc only)
      2. Filter by cosine similarity threshold
      3. Rerank, keep top-5
      4. Run DeBERTa NLI on each pair
      5. Route to Tier 2 (Gemini) for borderline cases
      6. Deduplicate pairs (A-B == B-A)

    Returns list of contradiction pair dicts.
    Saves to data/{topic_key}/contradiction_pairs.json.
    """
    from src.vector_store import (
        get_or_create_collection,
        query_similar,
        get_all_propositions,
    )
    from src.embedder import embed_single

    logger.info("Starting contradiction detection for %s...", topic_key)
    collection = get_or_create_collection(topic_key)
    total = collection.count()
    if total == 0:
        logger.warning("No propositions in ChromaDB for %s. Run embedding first.", topic_key)
        return []

    # Load all propositions as lookup table
    all_props = get_all_propositions(topic_key)
    prop_lookup = {p["prop_id"]: p for p in all_props}

    # Load article metadata for source/date enrichment
    meta = load_metadata(topic_key)
    article_lookup = {a["article_id"]: a for a in meta["articles"]}

    seen_pairs: set = set()
    all_results: list[dict] = []
    tier2_queue: list[dict] = []  # pairs queued for Gemini Tier 2
    tier2_context: dict = {}      # pair_id → partial result dict

    from src.embedder import embed_texts

    # Process all propositions
    logger.info("Processing %d propositions for cross-doc similarity...", len(all_props))
    prop_texts = [p["text"] for p in all_props]
    prop_ids = [p["prop_id"] for p in all_props]

    for idx, prop in enumerate(all_props):
        prop_id = prop["prop_id"]
        prop_text = prop["text"]
        prop_article_id = prop.get("article_id", "")

        # Embed and query
        embedding = embed_single(prop_text)
        raw_results = query_similar(
            topic_key, embedding, top_k=config.TOP_K_RETRIEVAL
        )

        # Flatten results
        candidate_ids = raw_results.get("ids", [[]])[0]
        candidate_docs = raw_results.get("documents", [[]])[0]
        candidate_metas = raw_results.get("metadatas", [[]])[0]
        candidate_dists = raw_results.get("distances", [[]])[0]

        # Filter: remove self, same article, below similarity threshold
        filtered = []
        for cid, cdoc, cmeta, cdist in zip(
            candidate_ids, candidate_docs, candidate_metas, candidate_dists
        ):
            if cid == prop_id:
                continue
            if cmeta.get("article_id") == prop_article_id:
                continue
            cosine_sim = 1.0 - cdist
            if cosine_sim < config.SEMANTIC_SIMILARITY_THRESHOLD:
                continue
            pair_key = frozenset({prop_id, cid})
            if pair_key in seen_pairs:
                continue
            filtered.append((cid, cdoc, cmeta, cosine_sim))

        if not filtered:
            continue

        # Rerank
        cand_ids = [f[0] for f in filtered]
        cand_texts = [f[1] for f in filtered]
        ranked = rerank_candidates(prop_text, cand_texts, cand_ids)
        top_cand_ids = {cid for cid, _ in ranked}

        # Run NLI on reranked top candidates
        for cid, reranker_score in ranked:
            pair_key = frozenset({prop_id, cid})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            cand_entry = next((f for f in filtered if f[0] == cid), None)
            if cand_entry is None:
                continue
            _, cand_text, cand_meta, cosine_sim = cand_entry

            cand_article_id = cand_meta.get("article_id", "")
            pair_id = f"{prop_id}__{cid}"

            # DeBERTa Tier 1
            nli_result = run_deberta_nli(prop_text, cand_text)
            tier_decision = classify_tier(nli_result)

            partial = {
                "pair_id": pair_id,
                "prop_a_id": prop_id,
                "prop_b_id": cid,
                "prop_a_text": prop_text,
                "prop_b_text": cand_text,
                "cosine_similarity": round(cosine_sim, 4),
                "reranker_score": round(float(reranker_score), 4),
                "deberta_contradiction_score": nli_result["contradiction"],
                "deberta_entailment_score": nli_result["entailment"],
                "deberta_neutral_score": nli_result["neutral"],
                "deberta_predicted_label": nli_result["predicted_label"],
                "tier_2_used": False,
                "gemini_verdict": None,
                "gemini_confidence": None,
                "gemini_reasoning": None,
                "final_verdict": None,
                "final_confidence": None,
                "article_a_id": prop_article_id,
                "article_b_id": cand_article_id,
                "source_a": prop.get("source_name", article_lookup.get(prop_article_id, {}).get("source_name", "")),
                "source_b": cand_meta.get("source_name", article_lookup.get(cand_article_id, {}).get("source_name", "")),
                "date_a": prop.get("publish_date", article_lookup.get(prop_article_id, {}).get("publish_date", "")),
                "date_b": cand_meta.get("publish_date", article_lookup.get(cand_article_id, {}).get("publish_date", "")),
            }

            if tier_decision in ("accept", "tier2"):
                # All non-rejected pairs go to Groq for temporal/nuance verification
                tier2_queue.append({
                    "pair_index": len(tier2_queue),
                    "prop_a": prop_text,
                    "prop_b": cand_text,
                    "source_a": partial["source_a"],
                    "source_b": partial["source_b"],
                    "date_a": partial["date_a"],
                    "date_b": partial["date_b"],
                })
                tier2_context[len(tier2_queue) - 1] = partial
            # tier_decision == 'reject' → discard

        if idx % 50 == 0:
            logger.info("Progress: %d/%d propositions processed, %d queued for Groq so far",
                        idx + 1, len(all_props), len(tier2_queue))

    # Process Tier 2 queue in batches
    if tier2_queue:
        logger.info("Running Groq Tier 2 on %d pairs for temporal/nuance verification...", len(tier2_queue))
        batch_size = config.GROQ_NLI_BATCH_SIZE
        for batch_start in range(0, len(tier2_queue), batch_size):
            batch = tier2_queue[batch_start: batch_start + batch_size]
            # Re-index pair_index within this batch
            for i, item in enumerate(batch):
                item["pair_index"] = i

            try:
                verdicts = call_groq_nli(batch)
            except Exception as exc:
                logger.error("Groq Tier 2 batch failed: %s", exc)
                verdicts = []

            verdict_map = {v["pair_index"]: v for v in verdicts}
            for i, item in enumerate(batch):
                global_idx = batch_start + i
                partial = tier2_context[global_idx]
                verdict_info = verdict_map.get(i)
                if verdict_info is None:
                    continue

                verdict = verdict_info.get("verdict", "UNRELATED").upper()
                partial["tier_2_used"] = True
                partial["gemini_verdict"] = verdict
                partial["gemini_confidence"] = verdict_info.get("confidence", 0.5)
                partial["gemini_reasoning"] = verdict_info.get("reasoning", "")

                if verdict == "CONTRADICTS":
                    partial["final_verdict"] = "contradiction"
                    partial["final_confidence"] = verdict_info.get("confidence", 0.5)
                    all_results.append(partial)
                elif verdict == "AGREES":
                    partial["final_verdict"] = "agreement"
                    partial["final_confidence"] = verdict_info.get("confidence", 0.5)
                    # Optionally include agreements — skip for now
                # NUANCE, TEMPORAL, UNRELATED → discard

    logger.info(
        "Detection complete for %s: %d contradictions from %d proposition pairs examined.",
        topic_key, len(all_results), len(seen_pairs)
    )

    save_contradiction_pairs(topic_key, all_results)
    return all_results


# ── Quick test ────────────────────────────────────────────────────────────────
# Run from project root:
#   uv run python -m src.nli_engine --topic topic_a

if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Run contradiction detection pipeline")
    parser.add_argument("--topic", default="topic_a", help="topic key")
    args = parser.parse_args()

    print(f"\nRunning contradiction detection for '{args.topic}' ...\n")
    results = detect_contradictions_for_topic(args.topic)
    print(f"\n--- Done ---")
    print(f"  Contradictions found : {len(results)}")
    if results:
        print(f"\nSample (first 3):")
        for r in results[:3]:
            print(f"\n  [{r['pair_id']}]")
            print(f"  A ({r['source_a']}): {r['prop_a_text'][:80]}")
            print(f"  B ({r['source_b']}): {r['prop_b_text'][:80]}")
            print(f"  verdict={r['final_verdict']} | confidence={r['final_confidence']}")
