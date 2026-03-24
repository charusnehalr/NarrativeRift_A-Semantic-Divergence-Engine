"""
graph_builder.py — Build contradiction graph from NLI output.

Nodes = propositions (one per factual sentence)
Edges = contradiction pairs (from contradiction_pairs.json)
Edge weight = final_confidence score from NLI

Also runs Louvain community detection to cluster related contradictions.
Saves graph_data.json (for Streamlit) and graph.html (standalone PyVis viz).
"""

import json
import logging

import networkx as nx
import community as community_louvain

import config
from src.utils import load_contradiction_pairs, _write_json

logger = logging.getLogger("sde.graph_builder")

# ── Source → colour mapping (for UI) ──────────────────────────────────────────
SOURCE_COLOURS = {
    "Al Jazeera":       "#E07B39",
    "Reuters":          "#C0392B",
    "BBC":              "#1A73E8",
    "Fox News":         "#8B0000",
    "AP News":          "#2C3E50",
    "Kyiv Independent": "#27AE60",
    "The Moscow Times": "#8E44AD",
    "SCMP":             "#F39C12",
    "Global Times":     "#D35400",
    "CSIS":             "#16A085",
    "Eco Times":        "#7F8C8D",
    "Spiegel":          "#2980B9",
    "The Independent":  "#6C3483",
}
DEFAULT_COLOUR = "#95A5A6"


def _node_colour(source_name: str) -> str:
    return SOURCE_COLOURS.get(source_name, DEFAULT_COLOUR)


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(pairs: list[dict]) -> nx.Graph:
    """
    Build a NetworkX undirected graph from contradiction pairs.

    Each node stores: prop_id, text, source_name, article_id, publish_date
    Each edge stores: pair_id, confidence, deberta_score, tier_2_used, gemini_reasoning
    """
    G = nx.Graph()

    for pair in pairs:
        prop_a_id = pair["prop_a_id"]
        prop_b_id = pair["prop_b_id"]

        # Add nodes (nx ignores duplicates, keeping first-seen attributes)
        if not G.has_node(prop_a_id):
            G.add_node(prop_a_id,
                text=pair["prop_a_text"],
                source_name=pair.get("source_a", ""),
                article_id=pair.get("article_a_id", ""),
                publish_date=pair.get("date_a", ""),
                colour=_node_colour(pair.get("source_a", "")),
            )
        if not G.has_node(prop_b_id):
            G.add_node(prop_b_id,
                text=pair["prop_b_text"],
                source_name=pair.get("source_b", ""),
                article_id=pair.get("article_b_id", ""),
                publish_date=pair.get("date_b", ""),
                colour=_node_colour(pair.get("source_b", "")),
            )

        # Add edge
        G.add_edge(
            prop_a_id,
            prop_b_id,
            pair_id=pair["pair_id"],
            confidence=round(pair.get("final_confidence", 0.0), 4),
            deberta_score=pair.get("deberta_contradiction_score", 0.0),
            tier_2_used=pair.get("tier_2_used", False),
            gemini_verdict=pair.get("gemini_verdict", ""),
            gemini_reasoning=pair.get("gemini_reasoning", ""),
            source_a=pair.get("source_a", ""),
            source_b=pair.get("source_b", ""),
        )

    logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ── Louvain clustering ─────────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> dict:
    """
    Run Louvain community detection on the graph.
    Returns dict mapping node_id → community_id (int).
    Falls back to each node in its own community if graph is empty.
    """
    if G.number_of_nodes() == 0:
        return {}
    partition = community_louvain.best_partition(G, resolution=config.LOUVAIN_RESOLUTION)
    n_communities = len(set(partition.values()))
    logger.info("Louvain detected %d communities", n_communities)
    return partition


# ── Serialise for Streamlit ────────────────────────────────────────────────────

def graph_to_dict(G: nx.Graph, partition: dict) -> dict:
    """
    Convert NetworkX graph + partition to a JSON-serialisable dict.
    Structure:
      {
        "nodes": [{"id": ..., "text": ..., "source_name": ..., "community": ...}, ...],
        "edges": [{"source": ..., "target": ..., "confidence": ..., ...}, ...]
      }
    """
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "text": attrs.get("text", ""),
            "source_name": attrs.get("source_name", ""),
            "article_id": attrs.get("article_id", ""),
            "publish_date": attrs.get("publish_date", ""),
            "colour": attrs.get("colour", DEFAULT_COLOUR),
            "community": partition.get(node_id, -1),
            "degree": G.degree(node_id),
        })

    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "pair_id": attrs.get("pair_id", ""),
            "confidence": attrs.get("confidence", 0.0),
            "deberta_score": attrs.get("deberta_score", 0.0),
            "tier_2_used": attrs.get("tier_2_used", False),
            "gemini_verdict": attrs.get("gemini_verdict", ""),
            "gemini_reasoning": attrs.get("gemini_reasoning", ""),
            "source_a": attrs.get("source_a", ""),
            "source_b": attrs.get("source_b", ""),
        })

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_communities": len(set(partition.values())) if partition else 0,
        "nodes": nodes,
        "edges": edges,
    }


# ── PyVis HTML visualisation ───────────────────────────────────────────────────

def build_pyvis_html(G: nx.Graph, partition: dict, output_path: str) -> None:
    """
    Generate a standalone interactive HTML graph using PyVis.
    Node colour = news source (so you can see which outlet each claim is from).
    Node size   = degree (more contradictions = bigger node).
    Edge colour = confidence (dark red = high confidence contradiction).
    Hover tooltip = full proposition text + source + date + reasoning.
    """
    from pyvis.network import Network

    net = Network(height="100vh", width="100%", bgcolor="#0f0f1a", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.5, spring_length=80, spring_strength=0.08, damping=0.09)

    # Build legend HTML (source -> colour), ASCII only to avoid encoding issues
    active_sources = {attrs.get("source_name") for _, attrs in G.nodes(data=True)}
    legend_items = "".join(
        '<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
        '<div style="width:12px;height:12px;border-radius:50%;background:' + colour + ';flex-shrink:0"></div>'
        '<span style="font-size:12px;color:#ddd">' + source + '</span></div>'
        for source, colour in SOURCE_COLOURS.items()
        if source in active_sources
    )

    for node_id, attrs in G.nodes(data=True):
        degree = G.degree(node_id)
        source = attrs.get("source_name", "")
        colour = attrs.get("colour", DEFAULT_COLOUR)
        text = attrs.get("text", "")
        date = attrs.get("publish_date", "")
        community_id = partition.get(node_id, -1)

        # Short label: source name only (text shown on hover)
        label = source

        # Tooltip — plain text (HTML not supported in all PyVis versions)
        title = f"{source} | {date}\n\n{text}\n\nCluster {community_id} | {degree} contradiction(s)"

        net.add_node(
            node_id,
            label=label,
            title=title,
            color={"background": colour, "border": "#ffffff33", "highlight": {"background": colour, "border": "#fff"}},
            size=14 + degree * 6,
            font={"size": 11, "color": "white"},
            borderWidth=1,
        )

    for u, v, attrs in G.edges(data=True):
        confidence = attrs.get("confidence", 0.5)
        source_a = attrs.get("source_a", "")
        source_b = attrs.get("source_b", "")
        reasoning = attrs.get("gemini_reasoning", "")
        tier2 = attrs.get("tier_2_used", False)

        # Edge colour: brighter red = higher confidence contradiction
        r = int(180 + confidence * 75)
        edge_colour = f"#{r:02x}2020"

        # Tooltip — plain text
        tooltip_lines = [
            f"Contradiction  conf={confidence:.2f}",
            f"{source_a} vs {source_b}",
        ]
        if reasoning:
            tooltip_lines.append(f"\n{reasoning}")
        if tier2:
            tooltip_lines.append("(Verified by Groq)")
        title = "\n".join(tooltip_lines)

        # Edge label: just confidence score — short and readable
        edge_label = f"{confidence:.2f}"

        net.add_edge(u, v, title=title, label=edge_label, width=1.5 + confidence * 3.5,
                     color=edge_colour, font={"size": 10, "color": "#ff9999", "align": "middle"})

    # Inject legend into HTML after generation
    net.save_graph(output_path)

    legend_html = (
        '<div id="legend" style="position:fixed;top:16px;right:16px;background:#1a1a2e;border:1px solid #333;'
        'border-radius:10px;padding:14px 18px;z-index:9999;min-width:160px;box-shadow:0 4px 20px #0008">'
        '<div style="font-weight:bold;font-size:14px;margin-bottom:10px;color:#eee">News Sources</div>'
        + legend_items +
        '<div style="margin-top:12px;border-top:1px solid #333;padding-top:10px;color:#aaa;font-size:11px">'
        'Node size = contradiction count<br>Edge label = confidence score</div></div>'
        '<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);'
        'font-family:sans-serif;font-size:20px;font-weight:bold;color:white;'
        'background:#0f0f1a99;padding:8px 24px;border-radius:8px;z-index:9999">'
        'NarrativeRift - Contradiction Graph</div>'
    )

    with open(output_path, "r", encoding="latin-1") as f:
        html = f.read()
    html = html.replace("</body>", legend_html + "\n</body>")
    with open(output_path, "w", encoding="latin-1") as f:
        f.write(html)

    logger.info("PyVis HTML saved to %s", output_path)


# ── Main entry point ───────────────────────────────────────────────────────────

def build_contradiction_graph(topic_key: str) -> dict:
    """
    Full pipeline: load pairs → build graph → cluster → save JSON + HTML.
    Returns graph_data dict.
    """
    tc = config.get_topic_config(topic_key)

    pairs = load_contradiction_pairs(topic_key)
    if not pairs:
        logger.warning("No contradiction pairs found for %s. Run NLI engine first.", topic_key)
        return {}

    logger.info("Building graph from %d contradiction pairs...", len(pairs))

    G = build_graph(pairs)
    partition = detect_communities(G)
    graph_data = graph_to_dict(G, partition)

    # Save JSON for Streamlit
    _write_json(tc["graph_data_path"], graph_data)
    logger.info("Graph data saved to %s", tc["graph_data_path"])

    # Save HTML viz
    build_pyvis_html(G, partition, str(tc["graph_html_path"]))

    return graph_data


# ── Quick test ────────────────────────────────────────────────────────────────
# Run from project root:
#   uv run python -m src.graph_builder --topic topic_a

if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Build contradiction graph")
    parser.add_argument("--topic", default="topic_a", help="topic key")
    args = parser.parse_args()

    print(f"\nBuilding contradiction graph for '{args.topic}' ...\n")
    graph_data = build_contradiction_graph(args.topic)

    if graph_data:
        print(f"\n--- Done ---")
        print(f"  Nodes (propositions) : {graph_data['total_nodes']}")
        print(f"  Edges (contradictions): {graph_data['total_edges']}")
        print(f"  Communities detected : {graph_data['total_communities']}")
        print(f"\n  graph_data.json and graph.html saved to data/{args.topic}/")
