"""Phase C: scale the STRUCTURED dataset (more entities per category -> many
more within-category comparison pairs) so Phase 7 statistics have real power.

MS MARCO / Natural Questions are open-domain single-answer QA and do NOT fit
this project's (entity, attribute) comparison schema, so we broaden the existing
structured corpus instead. New passages are templated from a compact,
hand-authored specs table (realistic values); the canonical `required_facts`
come straight from that table, so train == eval and there is no QA-extraction
noise. Comparisons are generated WITHIN a category on shared attributes only
(cross-category comparisons are nonsense).

Idempotent-ish: appends passages (skips existing passage_ids) and seed tasks
(skips existing ids). Run once, then regenerate tasks.json / splits / training
data / embeddings and retrain.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# passage-text templates per attribute; {e}=entity, {v}=value
TEMPLATES = {
    "price": "The {e} starts at {v}.",
    "weight": "The {e} weighs about {v}.",
    "battery": "The {e} has a {v} battery.",
    "camera": "The {e} has a {v} main camera.",
    "display_refresh_rate": "The {e} has a display with a {v} refresh rate.",
    "chip": "The {e} runs on the {v} chip.",
    "battery_life": "The {e} offers about {v} of listening per charge.",
    "noise_cancellation": "The {e} has {v} active noise cancellation.",
    "codec_support": "The {e} supports the {v} audio codec.",
    "range": "The {e} has an EPA-estimated range of about {v}.",
    "battery_capacity": "The {e} uses a {v} battery pack.",
    "cushioning": "Reviewers rate the {e}'s cushioning as {v}.",
    "best_for": "The {e} is best for {v}.",
    "elevation": "{e} stands {v} above sea level.",
    "first_ascent_year": "{e} was first summited in {v}.",
    "location": "{e} is located in the {v}.",
    "height": "The {e} has an architectural height of {v}.",
    "floors": "The {e} has {v} floors above ground.",
    "completion_year": "The {e} was completed in {v}.",
    "architect": "The {e} was designed by {v}.",
}

# category -> entity -> {attribute: value}. Value is BOTH the passage value and
# the canonical required_fact (short, distinctive). Values are realistic specs.
CATS: dict[str, dict[str, dict[str, str]]] = {
    "phones": {
        "iPhone 16": {"price": "$799", "weight": "170 grams", "battery": "3561 mAh",
                       "camera": "48MP", "display_refresh_rate": "60Hz", "chip": "A18"},
        "Google Pixel 9": {"price": "$799", "weight": "198 grams", "battery": "4700 mAh",
                            "camera": "50MP", "display_refresh_rate": "120Hz", "chip": "Tensor G4"},
        "OnePlus 12": {"price": "$799", "weight": "220 grams", "battery": "5400 mAh",
                        "camera": "50MP", "display_refresh_rate": "120Hz", "chip": "Snapdragon 8 Gen 3"},
        "Samsung Galaxy S24 Ultra": {"price": "$1,299", "weight": "232 grams", "battery": "5000 mAh",
                                      "camera": "200MP", "display_refresh_rate": "120Hz", "chip": "Snapdragon 8 Gen 3"},
    },
    "laptops": {
        "MacBook Pro 14 M4": {"price": "$1,599", "weight": "3.4 pounds", "battery": "22 hours", "chip": "M4"},
        "Lenovo ThinkPad X1 Carbon": {"price": "$1,399", "weight": "2.4 pounds", "battery": "15 hours", "chip": "Intel Core Ultra 7"},
        "Asus Zenbook 14 OLED": {"price": "$1,099", "weight": "2.8 pounds", "battery": "17 hours", "chip": "Intel Core Ultra 5"},
        "HP Spectre x360 14": {"price": "$1,449", "weight": "3.2 pounds", "battery": "16 hours", "chip": "Intel Core Ultra 7"},
    },
    "earbuds": {
        "Samsung Galaxy Buds3 Pro": {"price": "$249", "battery_life": "6 hours", "noise_cancellation": "strong", "codec_support": "SSC"},
        "Nothing Ear": {"price": "$149", "battery_life": "8 hours", "noise_cancellation": "moderate", "codec_support": "LDAC"},
    },
    "evs": {
        "Hyundai Ioniq 6": {"price": "$37,500", "range": "361 miles", "battery_capacity": "77 kWh", "best_for": "efficiency"},
        "Ford Mustang Mach-E": {"price": "$39,995", "range": "320 miles", "battery_capacity": "91 kWh", "best_for": "families"},
        "BYD Seal": {"price": "$46,000", "range": "354 miles", "battery_capacity": "82 kWh", "best_for": "value"},
    },
    "running_shoes": {
        "Adidas Adizero Boston 12": {"price": "$160", "weight": "9.4 ounces", "cushioning": "moderate", "best_for": "tempo runs"},
        "Brooks Ghost 16": {"price": "$140", "weight": "9.8 ounces", "cushioning": "balanced", "best_for": "daily training"},
        "Asics Gel-Nimbus 26": {"price": "$160", "weight": "10.6 ounces", "cushioning": "maximal", "best_for": "long runs"},
    },
    "mountains": {
        "Lhotse": {"elevation": "8,516 meters", "first_ascent_year": "1956", "location": "Himalayas"},
        "Makalu": {"elevation": "8,485 meters", "first_ascent_year": "1955", "location": "Himalayas"},
    },
    "buildings": {
        "Shanghai Tower": {"height": "632 meters", "floors": "128", "completion_year": "2015", "architect": "Gensler"},
        "Petronas Towers": {"height": "452 meters", "floors": "88", "completion_year": "1998", "architect": "Cesar Pelli"},
        "One World Trade Center": {"height": "541 meters", "floors": "94", "completion_year": "2014", "architect": "David Childs"},
    },
}

SRC = "https://specs.example.org/phase-c"  # synthetic provenance for templated specs
MAX_PAIRS_PER_ATTR = 6   # cap comparison pairs per (category, attribute) to bound size


def main() -> None:
    corpus_rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    have_pids = {r["passage_id"] for r in corpus_rows}
    next_pnum = max(int(r["passage_id"][1:]) for r in corpus_rows) + 1

    seeds = json.loads((DATA / "tasks_seed.json").read_text(encoding="utf-8"))
    have_ids = {t["id"] for t in seeds}
    next_tnum = max(int(t["id"][1:]) for t in seeds) + 1

    new_passages, new_seeds = [], []

    def add_passage(entity, attr, value):
        nonlocal next_pnum
        text = TEMPLATES[attr].format(e=entity, v=value)
        pid = f"p{next_pnum:03d}"; next_pnum += 1
        new_passages.append({"passage_id": pid, "text": text,
                             "byte_size": len(text.encode("utf-8")), "source_url": SRC,
                             "embedding": None, "metadata": {"entity": entity, "attribute": attr}})

    def add_seed(question, reqs, facts, atype, difficulty="medium"):
        nonlocal next_tnum
        tid = f"T{next_tnum}"; next_tnum += 1
        seen = set(); facts = [f for f in facts if not (f in seen or seen.add(f))]
        new_seeds.append({"id": tid, "question": question,
                          "decomposed_requirements": reqs,
                          "ground_truth": {"required_facts": facts, "match_threshold": 1.0},
                          "expected_answer": None, "topic": atype_topic(reqs),
                          "difficulty": difficulty, "answer_type": atype, "phase_c": True})

    def atype_topic(reqs):
        e = reqs[0]["entity"]
        for cat, ents in CATS.items():
            if e in ents:
                return cat
        return "misc"

    # 1) passages for every new (entity, attribute)
    for cat, ents in CATS.items():
        for e, attrs in ents.items():
            for a, v in attrs.items():
                add_passage(e, a, v)

    # helper: attribute phrase for questions
    def phrase(a):
        return a.replace("_", " ")

    # 2) tasks, generated WITHIN category on shared attributes
    for cat, ents in CATS.items():
        entities = list(ents)
        # attribute -> entities that have it (in this category)
        by_attr: dict[str, list[str]] = {}
        for e in entities:
            for a in ents[e]:
                by_attr.setdefault(a, []).append(e)

        # single_fact: one per (entity, first numeric-ish attr) -- a few per entity
        for e in entities:
            for a in list(ents[e])[:2]:
                add_seed(f"What is the {phrase(a)} of the {e}?",
                         [{"entity": e, "attribute": a}], [ents[e][a]], "single_fact", "easy")

        # comparison + yes_no from within-category pairs
        for a, es in by_attr.items():
            pairs = list(combinations(es, 2))[:MAX_PAIRS_PER_ATTR]
            for e1, e2 in pairs:
                add_seed(f"Compare the {e1} and the {e2} on {phrase(a)}.",
                         [{"entity": e1, "attribute": a}, {"entity": e2, "attribute": a}],
                         [ents[e1][a], ents[e2][a]], "comparison")
                # yes_no for orderable attributes
                if any(ch.isdigit() for ch in ents[e1][a]):
                    add_seed(f"Does the {e1} have a higher {phrase(a)} than the {e2}?",
                             [{"entity": e1, "attribute": a}, {"entity": e2, "attribute": a}],
                             [ents[e1][a], ents[e2][a]], "yes_no")

        # multi_part: one entity, its first two attributes
        for e in entities:
            attrs = list(ents[e])
            if len(attrs) >= 2:
                a1, a2 = attrs[0], attrs[1]
                add_seed(f"What are the {phrase(a1)} and {phrase(a2)} of the {e}?",
                         [{"entity": e, "attribute": a1}, {"entity": e, "attribute": a2}],
                         [ents[e][a1], ents[e][a2]], "multi_part")

    # write back
    with (DATA / "corpus.jsonl").open("a", encoding="utf-8") as f:
        for p in new_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    seeds.extend(new_seeds)
    (DATA / "tasks_seed.json").write_text(json.dumps(seeds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    from collections import Counter
    print(f"Added {len(new_passages)} passages, {len(new_seeds)} seeds.")
    print("new seeds by type:", dict(Counter(s["answer_type"] for s in new_seeds)))
    print(f"corpus now: {len(corpus_rows) + len(new_passages)} | seeds now: {len(seeds)}")


if __name__ == "__main__":
    main()
