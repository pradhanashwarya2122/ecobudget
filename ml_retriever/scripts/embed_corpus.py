"""Pre-computes all-MiniLM-L6-v2 embeddings for every passage in
data/corpus.jsonl and writes them back in place (Passage.embedding).

NOTE: this requires downloading model weights from huggingface.co. This
sandbox's network allowlist covers package registries (pypi, npm, crates,
github) but NOT huggingface.co, so running this script here will fail with
a connection error -- that's a sandbox limitation, not a code bug. Run
this on a machine with normal internet access (or update the sandbox's
allowed-domains network setting to include huggingface.co) after
`pip install sentence-transformers`.

Run: python scripts/embed_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "sentence-transformers is not installed. Run:\n"
            "  pip install sentence-transformers\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    corpus_path = ROOT / "data" / "corpus.jsonl"
    rows = [json.loads(line) for line in corpus_path.open(encoding="utf-8")]

    print("Loading all-MiniLM-L6-v2 (requires huggingface.co access)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [row["text"] for row in rows]
    embeddings = model.encode(texts, show_progress_bar=True)

    for row, emb in zip(rows, embeddings):
        row["embedding"] = emb.tolist()

    with corpus_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote embeddings for {len(rows)} passages back to {corpus_path}")


if __name__ == "__main__":
    main()
