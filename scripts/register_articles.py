"""
register_articles.py — Auto-register article .txt files into metadata.json.

Reads all .txt files from data/topic_a/articles/, parses the metadata header,
and writes data/topic_a/metadata.json.

Expected .txt file format (no separator needed — header is detected by KEY: prefix):
    SOURCE: Reuters
    DATE: 2026-03-16
    URL: https://reuters.com/...
    LEAN: western-mainstream
    TAGS: ceasefire, territorial-control
    [article body starts here — no --- needed]

Usage (run from project root):
    python scripts/register_articles.py
    python scripts/register_articles.py --dry-run
    python scripts/register_articles.py --status
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ── Paths (no external imports needed) ────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # project root
TOPIC_A_ARTICLES = BASE_DIR / "data" / "topic_a" / "articles"
TOPIC_A_METADATA = BASE_DIR / "data" / "topic_a" / "metadata.json"

# ── Header keys we recognise (order doesn't matter in the file) ───────────────
HEADER_KEYS = {"SOURCE", "DATE", "URL", "LEAN", "TAGS"}

# ── Files to skip silently ────────────────────────────────────────────────────
SKIP_FILES = {"TEMPLATE.txt"}


def parse_file(filepath: Path) -> dict:
    """
    Parse a .txt article file.

    The first lines whose content matches KEY: value are treated as header.
    Everything else (including blank lines after the header) is the body.
    Order of header lines doesn't matter; they just need to be KEY: value form.
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()

    header = {}
    body_lines = []
    header_done = False

    for line in lines:
        if header_done:
            body_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            # blank line — if we've already seen at least SOURCE, treat as end of header
            if "SOURCE" in header:
                header_done = True
            continue

        colon_pos = stripped.find(":")
        if colon_pos > 0:
            key = stripped[:colon_pos].strip().upper()
            if key in HEADER_KEYS:
                value = stripped[colon_pos + 1:].strip()
                header[key] = value
                # Once we've seen all 5 keys, header is done
                if HEADER_KEYS.issubset(header):
                    header_done = True
                continue

        # Not a header line → body starts here
        header_done = True
        body_lines.append(line)

    body = "\n".join(body_lines).strip()

    # Parse TAGS into a list
    tags_raw = header.get("TAGS", "")
    topic_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    warnings = []
    if not header.get("SOURCE"):
        warnings.append("SOURCE missing")
    if not header.get("DATE"):
        warnings.append("DATE missing")
    if len(body) < 80:
        warnings.append(f"Body very short ({len(body)} chars)")

    return {
        "source_name":   header.get("SOURCE", filepath.stem),
        "publish_date":  header.get("DATE", ""),
        "url":           header.get("URL", ""),
        "political_lean": header.get("LEAN", "independent"),
        "topic_tags":    topic_tags,
        "body_length":   len(body),
        "warnings":      warnings,
    }


def _write_json(path: Path, data: dict) -> None:
    """Atomically write JSON via temp-file + rename."""
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


def load_metadata() -> dict:
    if not TOPIC_A_METADATA.exists():
        return {"articles": []}
    with open(TOPIC_A_METADATA, encoding="utf-8") as f:
        return json.load(f)


def run(dry_run: bool = False) -> None:
    txt_files = sorted(
        f for f in TOPIC_A_ARTICLES.glob("*.txt")
        if f.name not in SKIP_FILES
    )

    if not txt_files:
        print(f"No .txt files found in {TOPIC_A_ARTICLES}")
        return

    meta = load_metadata()
    registered_filenames = {a["filename"] for a in meta["articles"]}

    new_entries = []
    skipped = []

    for filepath in txt_files:
        if filepath.name in registered_filenames:
            skipped.append(filepath.name)
            continue

        parsed = parse_file(filepath)

        if parsed["warnings"]:
            for w in parsed["warnings"]:
                print(f"  [WARN] {filepath.name}: {w}")

        article_id = filepath.stem   # e.g. 'reuters_001'

        # If id already taken (shouldn't happen), append suffix
        existing_ids = {a["article_id"] for a in meta["articles"]}
        if article_id in existing_ids:
            i = 2
            while f"{article_id}_{i}" in existing_ids:
                i += 1
            article_id = f"{article_id}_{i}"

        entry = {
            "article_id":     article_id,
            "filename":       filepath.name,
            "source_name":    parsed["source_name"],
            "publish_date":   parsed["publish_date"],
            "political_lean": parsed["political_lean"],
            "url":            parsed["url"],
            "topic_tags":     parsed["topic_tags"],
            "body_length":    parsed["body_length"],
            "proposition_count": 0,
            "status":         "raw",
            "date_registered": datetime.now(timezone.utc).isoformat(),
        }
        new_entries.append(entry)

        tag = "[DRY RUN]" if dry_run else "  Added "
        print(f"{tag}  {article_id:<22}  {parsed['source_name']:<32}  {parsed['publish_date']}")

    if not dry_run and new_entries:
        meta["articles"].extend(new_entries)
        _write_json(TOPIC_A_METADATA, meta)
        print(f"\n  Saved -> {TOPIC_A_METADATA}")

    print(f"\n  Total .txt files : {len(txt_files)}")
    print(f"  Newly registered : {len(new_entries)}")
    print(f"  Already existed  : {len(skipped)}")


def print_status() -> None:
    meta = load_metadata()
    articles = meta.get("articles", [])
    print(f"\n  {'ARTICLE ID':<22} {'SOURCE':<32} {'DATE':<12} {'LEAN':<28} STATUS")
    print(f"  {'-'*22} {'-'*32} {'-'*12} {'-'*28} {'-'*8}")
    for a in articles:
        print(
            f"  {a['article_id']:<22} {a['source_name']:<32} "
            f"{a['publish_date']:<12} {a['political_lean']:<28} {a['status']}"
        )
    if not articles:
        print("  (no articles registered yet)")
    print(f"\n  Total: {len(articles)} articles")


def main():
    parser = argparse.ArgumentParser(description="Register topic_a articles into metadata.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--status",  action="store_true", help="Show current metadata and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    print(f"Scanning: {TOPIC_A_ARTICLES}\n")
    run(dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print_status()


if __name__ == "__main__":
    main()
