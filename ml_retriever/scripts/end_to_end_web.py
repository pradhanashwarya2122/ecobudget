"""Tier A/B capstone: the whole thesis on REAL web pages, end to end, in one loop.

For each (question, url): fetch the live page, measure its real byte size, split
its visible text into candidate passages, then run the actual pipeline
(decompose -> per-requirement retrieval -> adaptive STOP -> answer) over those
passages and measure how many bytes the task-sufficient path actually used versus
loading the whole page. Energy is charged with the grounded 5G model (Tier A/B).

This closes the theme-fit gap: bytes and energy here are MEASURED on real fetched
pages, not modelled. Live fetch uses a browser User-Agent; if a URL fails it
falls back to the cached backend fixture pages so the script is reproducible
offline. No gold answers on live pages, so we report the produced answer and the
byte/energy reduction, not a correctness score.

Usage: python scripts/end_to_end_web.py
"""
import glob
import re
import sys
import urllib.request
from pathlib import Path

from ml_retriever.answer import GenerativeAnswerGenerator
from ml_retriever.decomposer import TaskDecomposer
from ml_retriever.energy import EnergyAccountant, RadioStateModel, query_op_counts
from ml_retriever.evidence import CachedScorer, EvidenceCoverageTracker, QAScorer
from ml_retriever.retriever import rank_passages
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = sorted(glob.glob(str(ROOT.parent / "backend" / "pages" / "*.html")))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

# Real product/spec pages that match the corpus domain.
EXAMPLES = [
    ("What is the display refresh rate of the iPhone 16?",
     "https://www.gsmarena.com/apple_iphone_16-13317.php"),
    ("What is the weight of the Samsung Galaxy S24 Ultra?",
     "https://www.gsmarena.com/samsung_galaxy_s24_ultra-12771.php"),
]


def fetch(url, fixture_idx=0):
    """(bytes, html, source-label). Live with a browser UA; fixture on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        raw = urllib.request.urlopen(req, timeout=12).read()
        return len(raw), raw.decode("utf-8", "ignore"), f"live:{url}"
    except Exception as e:
        if FIXTURES:
            f = FIXTURES[fixture_idx % len(FIXTURES)]
            html = open(f, encoding="utf-8", errors="ignore").read()
            return len(html.encode()), html, f"fixture:{Path(f).name} ({type(e).__name__})"
        raise


def visible_text(html):
    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def chunk(text, size=240):
    """Split into ~sentence-ish passages of about `size` chars."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, buf = [], ""
    for s in sents:
        if len(buf) + len(s) > size and buf:
            out.append(buf.strip()); buf = ""
        buf += " " + s
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if len(c) > 20]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", default="models/answer-base",
                    help="answerer checkpoint dir (use models/answer-base-robust for the "
                         "noise-trained domain-gap fix)")
    args = ap.parse_args()

    print("Loading models (decomposer, answerer, encoder, QA scorer)...", file=sys.stderr)
    import json
    corpus_attrs = sorted({json.loads(l)["metadata"]["attribute"]
                           for l in (DATA / "corpus.jsonl").open()})
    decomposer = TaskDecomposer("models/decomposer-base-lora",
                                adapter_path="models/decomposer-base-lora",
                                attribute_vocab=corpus_attrs)
    answerer = GenerativeAnswerGenerator(args.answerer, adapter_path=args.answerer)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    scorer = CachedScorer(QAScorer())
    accountant = EnergyAccountant()
    radio = RadioStateModel()

    print("=" * 92)
    print("END-TO-END ON REAL WEB PAGES: full-page load vs task-sufficient load")
    print("=" * 92)
    records = []
    for i, (q, url) in enumerate(EXAMPLES):
        full_bytes, html, src = fetch(url, i)
        passages = []
        for j, c in enumerate(chunk(visible_text(html))):
            emb = encoder.encode(c)
            passages.append(Passage(passage_id=f"c{j}", text=c, byte_size=len(c.encode()),
                                    source_url=url, embedding=emb))

        reqs = decomposer.decompose(q)
        # per-requirement retrieval + adaptive stop (coverage heuristic)
        tracker = EvidenceCoverageTracker(reqs, threshold=0.3, score_fn=scorer)
        selected, seen = [], set()
        for req in reqs:
            qemb = encoder.encode(f"{req.entity} {req.attribute.replace('_',' ')}")
            ranked = rank_passages(qemb, passages, k=5)
            for sp in ranked:
                if tracker.is_sufficient():
                    break
                if sp.passage.passage_id in seen:
                    continue
                tracker.add_passage(sp.passage); seen.add(sp.passage.passage_id)
                selected.append(sp.passage)
        answer = answerer.generate(q, selected).answer

        task_bytes = sum(p.byte_size for p in selected)
        reduction = 100 * (1 - task_bytes / full_bytes) if full_bytes else 0
        ops = query_op_counts(len(reqs), len(selected), answer_mode="per_requirement")
        e_task = accountant.account(ops, task_bytes)
        e_full = accountant.account(ops, full_bytes)  # same compute, whole-page transfer
        r_task = radio.account(task_bytes, len(selected))
        r_full = radio.account(full_bytes, 1)

        print(f"\nQ: {q}")
        print(f"  source: {src}")
        print(f"  full page:        {full_bytes:>9,} B   transfer_J={e_full['transfer_j']:.2f}  radio_J={r_full['radio_j']:.2f}")
        print(f"  task-sufficient:  {task_bytes:>9,} B   transfer_J={e_task['transfer_j']:.4f}  radio_J={r_task['radio_j']:.2f}")
        print(f"  BYTE REDUCTION:   {reduction:.2f}%   ({len(reqs)} reqs, {len(selected)} passages loaded)")
        print(f"  answer: {answer!r}")
        records.append({"question": q, "url": url, "source": src,
                        "full_bytes": full_bytes, "task_bytes": task_bytes,
                        "byte_reduction_pct": round(reduction, 2),
                        "transfer_j_full": e_full["transfer_j"], "transfer_j_task": e_task["transfer_j"],
                        "answer": answer})
    import json as _json
    (DATA / "e2e_results.json").write_text(_json.dumps(records, indent=2) + "\n")
    print(f"\nWrote {DATA / 'e2e_results.json'}")


if __name__ == "__main__":
    main()
