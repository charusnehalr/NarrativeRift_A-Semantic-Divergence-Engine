# NarrativeRift

**Semantic contradiction detection across global news sources.**

NarrativeRift ingests articles from outlets with opposing editorial perspectives, extracts atomic factual claims, and surfaces contradictions using a multi-stage NLP pipeline — ending in an interactive graph and Streamlit dashboard.

---

## Pipeline

```
Articles (raw text)
     │
     ▼
[1] Propositional Chunking       Gemini API
     │  Break articles into atomic, self-contained factual sentences
     ▼
[2] Dense Embedding              BGE-M3 (BAAI, 1024-dim)
     │  Embed each proposition into a semantic vector
     ▼
[3] Vector Store                 ChromaDB (local, cosine similarity)
     │  Store + index all proposition vectors for fast retrieval
     ▼
[4] Contradiction Detection      Two-Tier NLI
     │  Tier 1 — DeBERTa (local, fast)  →  score > 0.85: accept
     │  Tier 2 — Groq / Llama-3.1-8b   →  filter temporal + nuance false positives
     ▼
[5] Graph Construction           NetworkX + Louvain clustering
     │  Nodes = propositions, Edges = contradictions, Communities = narrative clusters
     ▼
[6] Dashboard                    Streamlit
        Interactive graph, source analytics, contradiction browser
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Propositional chunking | Gemini 2.0 Flash (`google-genai`) |
| Embeddings | BGE-M3 · BAAI · 1024-dim dense vectors |
| Vector store | ChromaDB · PersistentClient · HNSW cosine index |
| NLI Tier 1 | DeBERTa v3 · `cross-encoder/nli-deberta-v3-base` · local |
| NLI Tier 2 | Groq API · `llama-3.1-8b-instant` · temporal/nuance filtering |
| Reranker | `ms-marco-MiniLM-L-6-v2` · pre-filter before NLI |
| Graph | NetworkX · python-louvain community detection |
| Visualization | PyVis · interactive HTML (vis.js) |
| Dashboard | Streamlit |

---

## Results (Topic: Israel-Gaza, 30 articles, 6 sources)

- **1,611** propositions extracted and embedded
- **37** contradiction pairs detected after Groq verification
- **55** graph nodes · **37** edges · **19** narrative communities
- Sources span RT, Al Jazeera, SCMP, Reuters, BBC, AP — intentionally adversarial perspectives

---

## Screenshots

<!-- Add screenshots here after running the app -->
<!-- Suggested: overview metrics, contradiction graph, browse pairs section -->

<img width="1818" height="910" alt="image" src="https://github.com/user-attachments/assets/3a493ff9-479d-4d4e-92ad-8a27f3044f68" />
<img width="1794" height="778" alt="image" src="https://github.com/user-attachments/assets/47b48e38-7ab4-45fa-9962-dd03c626196d" />
<img width="1828" height="901" alt="image" src="https://github.com/user-attachments/assets/34576e5a-da17-4bf0-9ce2-e9a21097d452" />
<img width="1791" height="721" alt="image" src="https://github.com/user-attachments/assets/ce1d4f4e-226e-456c-a81f-1f5131b42d2a" />
<img width="1800" height="881" alt="image" src="https://github.com/user-attachments/assets/727e1924-4a42-48f6-86f3-7a33dbf0bc11" />
<img width="1792" height="882" alt="image" src="https://github.com/user-attachments/assets/d7c84cec-6697-44bc-a37e-365f62761b03" />


---

## Setup

**Requirements:** Python 3.10+, [uv](https://github.com/astral-sh/uv)

```bash
# Clone and set up environment
uv venv narrativerift
uv pip install -r requirements.txt --python narrativerift/Scripts/python.exe

# Copy and fill in API keys
cp .env.example .env
```

`.env` keys needed:
```
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

---

## Running the Pipeline

```bash
# 1. Chunk articles into propositions
uv run python -m src.chunker --mode all --topic topic_a

# 2. Embed and store in ChromaDB
uv run python -m src.vector_store --mode all --topic topic_a

# 3. Detect contradictions (DeBERTa + Groq)
uv run python -m src.nli_engine --topic topic_a

# 4. Build contradiction graph
uv run python -m src.graph_builder --topic topic_a

# 5. Launch dashboard
uv run streamlit run app.py
```

---

## Project Structure

```
src/
  chunker.py        — Gemini propositional chunking
  embedder.py       — BGE-M3 embedding via sentence-transformers
  vector_store.py   — ChromaDB ingestion + semantic query
  nli_engine.py     — Two-tier NLI (DeBERTa + Groq)
  graph_builder.py  — NetworkX graph + Louvain + PyVis HTML
  utils.py          — Shared metadata helpers
app.py              — Streamlit dashboard
config.py           — All paths, thresholds, API config
data/
  topic_a/          — Articles, propositions, contradiction pairs
  topic_b/
```

---

## Design Decisions

- **Two-tier NLI** — DeBERTa handles clear cases locally (free, fast); Groq handles ambiguous pairs to filter temporal mismatches ("Sunday vs Friday = same event") and framing differences vs factual contradictions.
- **Propositional chunking over semantic chunking** — Atomic sentences (one claim each, pronouns resolved) give cleaner NLI signal than paragraph-level chunks.
- **All free APIs** — Gemini free tier (1M tokens/day), Groq free tier (14,400 RPD), all ML models run locally. Total pipeline cost: $0.
