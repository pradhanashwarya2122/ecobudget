"""Phase 1 scale-up: append balanced TRAIN seed tasks to data/tasks_seed.json.

Rationale (see ../Phase-1-lever-plan.md): the decomposer sat at 44% val
exact-match against a >90% target because the seed set was small (48) and
badly skewed by answer type (yes_no 1, procedure 1, list/multi_part/narrative
2 each). The decomposer trains on question -> (entity, attribute) pairs, so
more balanced SEEDS is the lever, not more passages.

This script is idempotent and additive: it reads the canonical
data/tasks_seed.json + data/corpus.jsonl + data/splits.json, then appends new
seeds (ids T49+) built ONLY from (entity, attribute) pairs already in the
corpus -- no fabricated specs. New seeds are all TRAIN-bound and never reuse an
(entity, attribute) key that appears in a val/test seed, so they add training
coverage without leaking into evaluation. The only new passages are four
well-known film plot `summary` passages (public knowledge, not fabricated
product specs) so `narrative` can grow past its two existing films.

Re-running skips ids that already exist. Run once, then regenerate the pipeline
(generate_synthetic_tasks -> make_splits -> build_decomposer_training_data).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

from generate_synthetic_tasks import ATTRIBUTE_PHRASES  # noqa: E402

# Attributes whose values are orderable, so a "higher than" yes/no reads naturally.
_NUMERIC_ATTRS = {
    "price", "weight", "height", "floors", "elevation", "population", "area",
    "battery", "battery_capacity", "range", "rating", "room_count",
    "price_per_night", "camera", "display_refresh_rate", "first_ascent_year",
    "typical_trip_length", "completion_year",
}

# Four public-domain-knowledge film plots so `narrative` (summary) can grow
# beyond Mamma Mia / Titanic. Plots are widely known, not fabricated specs.
FILM_PASSAGES = [
    ("Inception", "A skilled thief who steals secrets by entering people's dreams is "
     "offered a chance to erase his criminal record if he can plant an idea in a "
     "target's mind, a process called inception, across nested dream layers.",
     "https://www.warnerbros.com/movies/inception"),
    ("The Lion King", "A young lion prince named Simba flees his kingdom after his "
     "father's death, blaming himself, then grows up in exile before returning to "
     "reclaim his throne from his usurping uncle Scar.",
     "https://movies.disney.com/the-lion-king"),
    ("Finding Nemo", "An anxious clownfish named Marlin crosses the ocean with a "
     "forgetful fish named Dory to rescue his son Nemo, who was captured by a diver "
     "and placed in a dentist's aquarium.",
     "https://movies.disney.com/finding-nemo"),
    ("Jurassic Park", "Scientists clone dinosaurs to populate an island theme park, "
     "but when the power fails the creatures escape, and the visitors must survive "
     "and flee the island.",
     "https://www.jurassicworld.com"),
]

def _phrase(attr: str) -> str:
    return ATTRIBUTE_PHRASES.get(attr, attr.replace("_", " "))


def _load():
    corpus = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    seeds = json.loads((DATA / "tasks_seed.json").read_text(encoding="utf-8"))
    splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    return corpus, seeds, splits


def main() -> None:
    corpus, seeds, splits = _load()

    # entity -> topic, from existing seeds (new seeds inherit their entity's topic)
    ent_topic: dict[str, str] = {}
    for t in seeds:
        for r in t["decomposed_requirements"]:
            ent_topic.setdefault(r["entity"], t["topic"])

    id_to_split = {i: s for s, ids in splits.items() for i in ids}

    def key(e, a):
        return (e.strip().lower(), a.strip().lower())

    # (entity, attribute) keys used by val/test seeds -- forbidden for new train seeds
    # tasks_seed.json holds only seeds (no "synthetic" flag; that's added in
    # tasks.json). A seed's split is looked up from splits.json by id.
    forbidden = set()
    for t in seeds:
        if id_to_split.get(t["id"]) in ("val", "test"):
            for r in t["decomposed_requirements"]:
                forbidden.add(key(r["entity"], r["attribute"]))

    existing_ids = {t["id"] for t in seeds}
    existing_questions = {t["question"].strip().lower() for t in seeds}

    # --- add film summary passages (idempotent) ---
    have_pids = {r["passage_id"] for r in corpus}
    next_pnum = max(int(r["passage_id"][1:]) for r in corpus) + 1
    film_entities = []
    new_passages = []
    for entity, text, url in FILM_PASSAGES:
        film_entities.append(entity)
        if any(r["metadata"]["entity"] == entity and r["metadata"]["attribute"] == "summary" for r in corpus):
            continue
        pid = f"p{next_pnum:03d}"
        next_pnum += 1
        new_passages.append({
            "passage_id": pid, "text": text, "byte_size": len(text.encode("utf-8")),
            "source_url": url, "embedding": None,
            "metadata": {"entity": entity, "attribute": "summary"},
        })
        ent_topic[entity] = "film"

    # corpus pairs available (all), and "safe" (not forbidden) for train seeds
    corpus_pairs = [(r["metadata"]["entity"], r["metadata"]["attribute"]) for r in corpus]
    corpus_pairs += [(e, "summary") for e in film_entities]
    from collections import defaultdict
    ents_by_attr = defaultdict(list)
    seen_pair = set()
    for e, a in corpus_pairs:
        if (e, a) in seen_pair:
            continue
        seen_pair.add((e, a))
        if key(e, a) not in forbidden:
            ents_by_attr[a].append(e)
    for a in ents_by_attr:
        ents_by_attr[a] = sorted(set(ents_by_attr[a]))
    safe_pairs = sorted({(e, a) for a in ents_by_attr for e in ents_by_attr[a]})

    new_seeds = []
    counter = {"n": 48}  # highest existing seed number is T48

    def next_id():
        counter["n"] += 1
        nid = f"T{counter['n']}"
        while nid in existing_ids:
            counter["n"] += 1
            nid = f"T{counter['n']}"
        existing_ids.add(nid)
        return nid

    def add(question, reqs, answer_type, difficulty):
        q = question.strip()
        if q.lower() in existing_questions:
            return False
        topic = ent_topic.get(reqs[0]["entity"], "misc")
        new_seeds.append({
            "id": next_id(), "question": q,
            "decomposed_requirements": reqs,
            "topic": topic, "difficulty": difficulty, "answer_type": answer_type,
            "decomposer_only": True,  # ground-truth answers TBD; used for decomposer training
        })
        existing_questions.add(q.lower())
        return True

    # Targets (new seeds to add per type)
    TARGETS = {"single_fact": 6, "comparison": 4, "yes_no": 13, "multi_part": 12,
               "list": 6, "narrative": 4}

    # --- single_fact ---
    n = 0
    for e, a in safe_pairs:
        if n >= TARGETS["single_fact"]:
            break
        if a in ("summary", "procedure"):
            continue
        if add(f"What is the {_phrase(a)} of {e}?", [{"entity": e, "attribute": a}], "single_fact", "easy"):
            n += 1

    # --- comparison (1 attribute, 2 entities) ---
    n = 0
    for a in sorted(ents_by_attr):
        if n >= TARGETS["comparison"]:
            break
        if a in ("summary", "procedure"):
            continue
        ents = ents_by_attr[a]
        if len(ents) >= 2:
            e1, e2 = ents[0], ents[1]
            reqs = [{"entity": e1, "attribute": a}, {"entity": e2, "attribute": a}]
            if add(f"Compare {e1} and {e2} on {_phrase(a)}.", reqs, "comparison", "medium"):
                n += 1

    # --- yes_no (2 entities, numeric attribute) ---
    n = 0
    for a in sorted(ents_by_attr):
        if n >= TARGETS["yes_no"]:
            break
        if a not in _NUMERIC_ATTRS:
            continue
        ents = ents_by_attr[a]
        for e1, e2 in combinations(ents, 2):
            if n >= TARGETS["yes_no"]:
                break
            reqs = [{"entity": e1, "attribute": a}, {"entity": e2, "attribute": a}]
            if add(f"Is {e1}'s {_phrase(a)} higher than {e2}'s?", reqs, "yes_no", "medium"):
                n += 1

    # --- multi_part (1 entity, 2 attributes) ---
    n = 0
    attrs_by_ent = defaultdict(list)
    for e, a in safe_pairs:
        if a not in ("summary", "procedure"):
            attrs_by_ent[e].append(a)
    for e in sorted(attrs_by_ent):
        if n >= TARGETS["multi_part"]:
            break
        attrs = sorted(set(attrs_by_ent[e]))
        if len(attrs) >= 2:
            a1, a2 = attrs[0], attrs[1]
            reqs = [{"entity": e, "attribute": a1}, {"entity": e, "attribute": a2}]
            if add(f"What are {e}'s {_phrase(a1)} and {_phrase(a2)}?", reqs, "multi_part", "medium"):
                n += 1

    # --- list (amenities / codec_support) ---
    n = 0
    for a in ("amenities", "codec_support"):
        for e in ents_by_attr.get(a, []):
            if n >= TARGETS["list"]:
                break
            if add(f"List the {_phrase(a)} of {e}.", [{"entity": e, "attribute": a}], "list", "medium"):
                n += 1

    # --- narrative (film summaries) ---
    n = 0
    for e in film_entities:
        if n >= TARGETS["narrative"]:
            break
        if key(e, "summary") in forbidden:
            continue
        if add(f"Give me a brief summary of {e}.", [{"entity": e, "attribute": "summary"}], "narrative", "medium"):
            n += 1

    # --- write back ---
    if new_passages:
        with (DATA / "corpus.jsonl").open("a", encoding="utf-8") as f:
            for p in new_passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    seeds.extend(new_seeds)
    (DATA / "tasks_seed.json").write_text(json.dumps(seeds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    from collections import Counter
    print(f"Added {len(new_passages)} film passages, {len(new_seeds)} seeds.")
    print("new seeds by type:", dict(Counter(s["answer_type"] for s in new_seeds)))
    print(f"seeds now: {len(seeds)}; corpus now: {len(corpus) + len(new_passages)}")


if __name__ == "__main__":
    main()
