"""Tier A: ground the payload model in REAL measured web pages instead of
assumed constants.

Phase B originally charged the `html_page` scenario at an assumed 60 KB. That is
far below reality. This script measures the actual rendered-HTML byte sizes of
the real pages the sibling `backend/` prototype fetched with Playwright (stored
under backend/pages/*.html), plus the extracted-visible-text size per page, and
writes the distribution to data/measured_payloads.json. `energy.PayloadModel`
uses the measured median so the transfer/energy numbers are grounded in measured
pages, not assumptions.

The backend fixtures are gitignored; regenerate them with
`backend/download_pages.sh` (Playwright) before running this, or point --pages
elsewhere. Reproducible: re-run to refresh the JSON.

Usage: python scripts/measure_real_pages.py [--pages ../backend/pages]
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def strip_text_bytes(html: str) -> int:
    """Dependency-free visible-text extraction: drop script/style, strip tags,
    collapse whitespace. Approximates BeautifulSoup.get_text within a few %."""
    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = re.sub(r"\s+", " ", txt).strip()
    return len(txt.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=str(ROOT.parent / "backend" / "pages"))
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.pages, "*.html"))
                   if os.path.getsize(f) > 0)
    if not files:
        raise SystemExit(f"No pages under {args.pages}. Regenerate with "
                         "backend/download_pages.sh (Playwright), or pass --pages.")

    raws, texts, ratios = [], [], []
    for f in files:
        b = os.path.getsize(f)
        html = open(f, encoding="utf-8", errors="ignore").read()
        tb = strip_text_bytes(html)
        raws.append(b); texts.append(tb); ratios.append(tb / b)

    def dist(xs):
        xs = sorted(xs)
        return {"n": len(xs), "min": min(xs), "median": int(st.median(xs)),
                "mean": int(st.mean(xs)), "max": max(xs)}

    out = {
        "source": "real Playwright-rendered pages in backend/pages/",
        "n_pages": len(files),
        "raw_html_bytes": dist(raws),
        "extracted_text_bytes": dist(texts),
        "text_to_html_ratio_median": round(st.median(ratios), 4),
        # the value PayloadModel should use for the html_page scenario:
        "html_page_bytes_measured": int(st.median(raws)),
    }
    (DATA / "measured_payloads.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {DATA / 'measured_payloads.json'}")
    print(f"=> set PayloadModel.html_page_bytes = {out['html_page_bytes_measured']:,} "
          f"(measured median of {len(files)} real pages)")


if __name__ == "__main__":
    main()
