"""
app.py — NarrativeRift Streamlit UI.
Run: uv run streamlit run app.py
"""

import json
from collections import defaultdict

import pandas as pd
import streamlit as st

import config
from src.utils import load_metadata

st.set_page_config(page_title="NarrativeRift", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  [data-testid="metric-container"] {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:16px 20px;
  }
  [data-testid="stMetricValue"] { color:#2563EB; font-weight:700; }
  [data-testid="stMetricLabel"] { color:#64748B; font-size:13px; }
  h2, h3 { color:#1E40AF !important; }
  .stExpander { border:1px solid #E2E8F0 !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_graph_data(topic_key):
    path = config.get_topic_config(topic_key)["graph_data_path"]
    return json.loads(path.read_text()) if path.exists() else {}

@st.cache_data
def load_pairs(topic_key):
    path = config.get_topic_config(topic_key)["contradiction_pairs_path"]
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("pairs", [])


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_topic = st.columns([5, 1])
with col_title:
    st.title("NarrativeRift")
    st.caption("Semantic contradiction detection across global news sources")
with col_topic:
    st.markdown("<br>", unsafe_allow_html=True)
    topic_key = st.selectbox("Topic", config.TOPIC_KEYS,
        format_func=lambda k: config.TOPICS[k]["name"], label_visibility="collapsed")

st.divider()

# ── Load ───────────────────────────────────────────────────────────────────────
graph_data = load_graph_data(topic_key)
pairs      = load_pairs(topic_key)
meta       = load_metadata(topic_key)
article_lookup = {a["article_id"]: a for a in meta["articles"]}

if not graph_data or not pairs:
    st.warning(f"No data for **{config.TOPICS[topic_key]['name']}**. Run the pipeline first.")
    st.code("uv run python -m src.nli_engine --topic " + topic_key)
    st.code("uv run python -m src.graph_builder --topic " + topic_key)
    st.stop()

nodes = graph_data.get("nodes", [])
edges = graph_data.get("edges", [])


# ── Section 1: Metrics ─────────────────────────────────────────────────────────
st.subheader("Overview")
m1, m2, m3, m4, m5 = st.columns(5)
n_sources = len({n["source_name"] for n in nodes})
avg_conf  = round(sum(e["confidence"] for e in edges) / len(edges), 2) if edges else 0
m1.metric("Total Claims",        graph_data.get("total_nodes", 0),       delta="propositions")
m2.metric("Contradictions",      graph_data.get("total_edges", 0),       delta="detected pairs")
m3.metric("Narrative Clusters",  graph_data.get("total_communities", 0), delta="communities")
m4.metric("Sources Analysed",    n_sources,                              delta="news outlets")
m5.metric("Avg Confidence",      avg_conf,                               delta="NLI score")

st.divider()


# ── Section 2: Most Contradicted Sources ──────────────────────────────────────
st.subheader("Most Contradicted Source Pairs")
st.caption("Which pairs of outlets disagree the most — ranked by number of detected contradictions.")

pair_counts: dict = defaultdict(int)
pair_conf: dict   = defaultdict(list)
for p in pairs:
    key = tuple(sorted([p.get("source_a",""), p.get("source_b","")]))
    pair_counts[key] += 1
    pair_conf[key].append(p.get("final_confidence", 0))

ranked_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:8]
max_count = ranked_pairs[0][1] if ranked_pairs else 1

for (src_a, src_b), count in ranked_pairs:
    avg = round(sum(pair_conf[(src_a,src_b)]) / len(pair_conf[(src_a,src_b)]), 2)
    col_label, col_bar, col_count = st.columns([3, 5, 1])
    col_label.markdown(f"**{src_a}** vs **{src_b}**")
    col_bar.progress(count / max_count)
    col_count.markdown(f"`{count} pairs`")

st.divider()


# ── Section 3: Analytics ───────────────────────────────────────────────────────
st.subheader("Analytics")

ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("**Contradictions by Source**")
    src_counts: dict = defaultdict(int)
    for p in pairs:
        src_counts[p.get("source_a", "")] += 1
        src_counts[p.get("source_b", "")] += 1
    src_df = pd.DataFrame(
        sorted(src_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["Source", "Contradictions"],
    ).set_index("Source")
    st.bar_chart(src_df, color="#2563EB")

with ch2:
    st.markdown("**Source vs Source — Contradiction Count**")
    matrix: dict = defaultdict(int)
    for p in pairs:
        a, b = p.get("source_a",""), p.get("source_b","")
        matrix[tuple(sorted([a, b]))] += 1
    all_srcs_m = sorted({s for k in matrix for s in k})
    mat = {s: {s2: 0 for s2 in all_srcs_m} for s in all_srcs_m}
    for (a, b), c in matrix.items():
        mat[a][b] = c; mat[b][a] = c
    st.dataframe(pd.DataFrame(mat).fillna(0).astype(int), use_container_width=True)

st.divider()


# ── Section 4: Contradiction Graph ────────────────────────────────────────────
st.subheader("Contradiction Graph")
st.caption("Node colour = source · Node size = contradiction count · Edge label = confidence · Hover for details")

html_path = config.get_topic_config(topic_key)["graph_html_path"]
if html_path.exists():
    with open(html_path, "r", encoding="latin-1") as f:
        st.components.v1.html(f.read(), height=680, scrolling=False)
else:
    st.warning("Run: `uv run python -m src.graph_builder --topic " + topic_key + "`")

st.divider()


# ── Section 5: Temporal Timeline ──────────────────────────────────────────────
st.subheader("Temporal Contradiction Timeline")
st.caption("Contradictions sorted by publication date — see how narratives diverged over time.")

dated_pairs = sorted(
    [p for p in pairs if p.get("date_a")],
    key=lambda x: x.get("date_a", "")
)

if dated_pairs:
    timeline_df = pd.DataFrame([{
        "Date":       p.get("date_a",""),
        "Source A":   p.get("source_a",""),
        "Source B":   p.get("source_b",""),
        "Confidence": round(p.get("final_confidence",0), 2),
        "Claim A":    p.get("prop_a_text","")[:80] + "...",
        "Claim B":    p.get("prop_b_text","")[:80] + "...",
    } for p in dated_pairs])
    st.dataframe(timeline_df, use_container_width=True, hide_index=True)
else:
    st.info("No date information available in the pairs.")

st.divider()


# ── Section 6: Narrative Stance ────────────────────────────────────────────────
st.subheader("Narrative Stance by Political Lean")
st.caption("Sources grouped by political lean — see which outlets are most involved in contradictions.")

lean_sources    = defaultdict(set)
lean_pair_count = defaultdict(int)

for article in meta["articles"]:
    lean = article.get("political_lean", "Unknown")
    src  = article.get("source_name", "")
    if src:
        lean_sources[lean].add(src)

for pair in pairs:
    for art_id in [pair.get("article_a_id",""), pair.get("article_b_id","")]:
        if art_id in article_lookup:
            lean = article_lookup[art_id].get("political_lean", "Unknown")
            lean_pair_count[lean] += 1

# Build summary table
rows = []
for lean, srcs in sorted(lean_sources.items()):
    rows.append({
        "Political Lean":    lean,
        "Sources":           ", ".join(sorted(srcs)),
        "Source Count":      len(srcs),
        "Contradiction Appearances": lean_pair_count.get(lean, 0),
    })

lean_df = pd.DataFrame(rows).sort_values("Contradiction Appearances", ascending=False)
st.dataframe(lean_df, use_container_width=True, hide_index=True)

# Top 3 most contradicting leans as metrics
st.markdown("**Most active in contradictions:**")
top_leans = lean_df.head(3)
top_cols = st.columns(3)
for i, (_, row) in enumerate(top_leans.iterrows()):
    top_cols[i].metric(
        row["Political Lean"],
        f"{row['Contradiction Appearances']} pairs",
        delta=row["Sources"][:40],
    )

st.divider()


# ── Section 7: Browse Pairs ────────────────────────────────────────────────────
st.subheader("Browse Contradiction Pairs")

f1, f2, f3 = st.columns([2, 2, 1])
all_srcs_list = sorted(
    {p.get("source_a","") for p in pairs} | {p.get("source_b","") for p in pairs}
)
sel_src   = f1.selectbox("Filter by source", ["All"] + all_srcs_list)
min_conf  = f2.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
groq_only = f3.checkbox("Groq-verified only")

filtered = [
    p for p in pairs
    if (sel_src == "All" or p.get("source_a") == sel_src or p.get("source_b") == sel_src)
    and p.get("final_confidence",0) >= min_conf
    and (not groq_only or p.get("tier_2_used"))
]

st.caption(f"Showing **{len(filtered)}** of **{len(pairs)}** pairs")

for pair in filtered:
    conf  = pair.get("final_confidence", 0)
    tier2 = pair.get("tier_2_used", False)
    rsn   = pair.get("gemini_reasoning","")
    src_a = pair.get("source_a","")
    src_b = pair.get("source_b","")

    if conf > 0.9:   conf_label = f"🔴 {conf:.2f}"
    elif conf > 0.75: conf_label = f"🟠 {conf:.2f}"
    else:             conf_label = f"🟡 {conf:.2f}"

    title = f"{src_a}  ↔  {src_b}   |   confidence {conf_label}"
    if tier2:
        title += "   ✅ Groq"

    with st.expander(title):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**:blue[{src_a}]** · {pair.get('date_a','')}")
            st.info(pair.get("prop_a_text",""))
        with right:
            st.markdown(f"**:orange[{src_b}]** · {pair.get('date_b','')}")
            st.warning(pair.get("prop_b_text",""))

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence",    f"{conf:.3f}")
        c2.metric("DeBERTa Score", f"{pair.get('deberta_contradiction_score',0):.3f}")
        c3.metric("Cosine Sim",    f"{pair.get('cosine_similarity',0):.3f}")

        if rsn:
            st.caption(f"Groq reasoning: {rsn}")
