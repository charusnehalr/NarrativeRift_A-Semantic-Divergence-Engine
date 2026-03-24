"""
config.py — Central configuration for the Semantic Divergence Engine.
All thresholds, model names, paths, and topic definitions live here.
Every other module imports from this file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_BASE_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")))

# ── Topic definitions ──────────────────────────────────────────────────────────
TOPICS = {
    "topic_a": {
        "name": "Russia-Ukraine War",
        "description": "Territorial gains and ceasefire negotiations",
        "data_dir": DATA_DIR / "topic_a",
        "articles_dir": DATA_DIR / "topic_a" / "articles",
        "propositions_dir": DATA_DIR / "topic_a" / "propositions",
        "metadata_path": DATA_DIR / "topic_a" / "metadata.json",
        "contradiction_pairs_path": DATA_DIR / "topic_a" / "contradiction_pairs.json",
        "graph_data_path": DATA_DIR / "topic_a" / "graph_data.json",
        "graph_html_path": DATA_DIR / "topic_a" / "graph.html",
        "ground_truth_path": DATA_DIR / "topic_a" / "evaluation" / "ground_truth.csv",
        "chroma_dir": CHROMA_BASE_DIR / "topic_a",
        "chroma_collection": os.getenv("CHROMA_COLLECTION_TOPIC_A", "propositions_topic_a"),
    },
    "topic_b": {
        "name": "Strait of Hormuz Crisis",
        "description": "Strait of Hormuz closure and Middle East energy crisis",
        "data_dir": DATA_DIR / "topic_b",
        "articles_dir": DATA_DIR / "topic_b" / "articles",
        "propositions_dir": DATA_DIR / "topic_b" / "propositions",
        "metadata_path": DATA_DIR / "topic_b" / "metadata.json",
        "contradiction_pairs_path": DATA_DIR / "topic_b" / "contradiction_pairs.json",
        "graph_data_path": DATA_DIR / "topic_b" / "graph_data.json",
        "graph_html_path": DATA_DIR / "topic_b" / "graph.html",
        "ground_truth_path": DATA_DIR / "topic_b" / "evaluation" / "ground_truth.csv",
        "chroma_dir": CHROMA_BASE_DIR / "topic_b",
        "chroma_collection": os.getenv("CHROMA_COLLECTION_TOPIC_B", "propositions_topic_b"),
    },
}

TOPIC_KEYS = list(TOPICS.keys())


def get_topic_config(topic_key: str) -> dict:
    """Return topic config dict with chroma_dir_str added for Windows path safety."""
    if topic_key not in TOPICS:
        raise ValueError(f"Unknown topic_key '{topic_key}'. Must be one of {TOPIC_KEYS}")
    cfg = TOPICS[topic_key].copy()
    # Windows fix: ChromaDB SQLite fails with backslash paths
    cfg["chroma_dir_str"] = cfg["chroma_dir"].as_posix()
    return cfg


# ── Gemini API ─────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_RPM_LIMIT = 5           # free tier: 5 requests per minute
GEMINI_SLEEP_BETWEEN_CALLS = 60 / GEMINI_RPM_LIMIT  # ~4 seconds
GEMINI_MAX_RETRIES = 3
GEMINI_TEMPERATURE = 0.1        # low temp for consistent extraction
GEMINI_NLI_BATCH_SIZE = 5       # pairs per Tier 2 Gemini call

# ── BGE-M3 Embeddings ──────────────────────────────────────────────────────────
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_EMBEDDING_DIM = 1024        # dense vector dimension
BGE_MAX_LENGTH = 512            # propositions are short; 512 is sufficient
BGE_BATCH_SIZE = 4              # safe for CPU with ~4GB RAM; increase to 16 on GPU
BGE_USE_FP16 = True             # halves memory usage; safe for inference
BGE_DEVICE = "cpu"              # set to "cuda" if GPU available

# ── NLI Models ────────────────────────────────────────────────────────────────
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# DeBERTa label order: index 0=contradiction, 1=entailment, 2=neutral
NLI_CONTRADICTION_IDX = 0
NLI_ENTAILMENT_IDX = 1
NLI_NEUTRAL_IDX = 2

# Routing thresholds for two-tier system
NLI_ACCEPT_THRESHOLD = 0.7      # contradiction score > this → accept as contradiction
NLI_REJECT_THRESHOLD = 0.4      # contradiction score < this → reject (not a contradiction)
# Between REJECT and ACCEPT thresholds → escalate to Gemini Tier 2

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 5              # keep this many candidates after reranking

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 10            # ChromaDB candidates per proposition query
SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # min cosine similarity to consider a pair

# ── Graph & Clustering ────────────────────────────────────────────────────────
LOUVAIN_RESOLUTION = 1.0        # tune for coarser (< 1.0) or finer (> 1.0) clusters
GRAPH_MIN_EDGE_CONFIDENCE = 0.0 # filter edges below this confidence in UI

# ── Streamlit ─────────────────────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "Semantic Divergence Engine"
STREAMLIT_LAYOUT = "wide"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
