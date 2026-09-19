"""
Evidence Coverage Tracker — EcoBudget v2 core.
Comparison tasks are decomposed by Flan-T5 into subject x attribute
requirements; single-fact tasks become one requirement from the question.
Ground truth is never used here.
"""

import re
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

print("[evidence] loading models...")
_t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
_t5_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
_t5_model.eval()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("[evidence] ready")


ANSWER_PATTERNS = {
    "when": re.compile(r'\b(\d{1,4}\s*(AD|BC|BCE|CE)?|[A-Z][a-z]+ \d{4}|\d{4}s?)\b'),
    "date": re.compile(r'\b(\d{1,4}\s*(AD|BC|BCE|CE)?|[A-Z][a-z]+ \d{4}|\d{4}s?)\b'),
    "where": re.compile(r'\b[A-Z][a-zA-Z\s,]{2,40}\b'),
    "who": re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
    "name": re.compile(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'),
    "how_much": re.compile(r'\b\d+[\.,]?\d*\s*(KB|MB|GB|km|m|ft|kg|lb|USD|EUR|INR|'
                           r'₹|\$|%|GHz|Mbps|W|mAh|MP|hours?|days?)?\b', re.I),
    "number": re.compile(r'\b\d+[\.,]?\d*\s*(KB|MB|GB|km|m|ft|kg|lb|USD|EUR|INR|'
                         r'₹|\$|%|GHz|Mbps|W|mAh|MP|hours?|days?)?\b', re.I),
    "what": re.compile(r'.{2,}'),
    "text": re.compile(r'.{2,}'),
    "list": re.compile(r'.{2,}'),
}

_NUMERIC_HINTS = ("price", "cost", "battery", "ram", "storage", "weight",
                  "resolution", "size", "distance", "capacity", "mah", "gb",
                  "mp", "megapixel", "hz", "screen", "display", "speed",
                  "range", "latency", "power", "rate", "camera")
_DATE_HINTS = ("date", "release", "launch", "year", "founded", "built", "completed")

_LEADING_JUNK = re.compile(r'^(and|the|a|an|both)\s+', re.IGNORECASE)
_COMPARE_RE = re.compile(r'\b(compare|comparison|versus|\bvs\b|difference between|'
                         r'better|which is)\b', re.I)


def _run_t5(prompt, max_new_tokens=64):
    inputs = _t5_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = _t5_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    return _t5_tok.decode(out[0], skip_special_tokens=True).strip()


def _split_items(raw):
    parts = re.split(r"[\n,;]+|\bvs\.?\b|\bversus\b|\s+and\s+", raw, flags=re.IGNORECASE)
    seen, items = set(), []
    for p in parts:
        p = p.strip(" .-•\t\"'?")
        p = _LEADING_JUNK.sub("", p).strip()
        if len(p) < 2:
            continue
        if p.lower() not in seen:
            seen.add(p.lower())
            items.append(p)
    return items


def _infer_qtype(text):
    t = text.lower()
    if any(h in t for h in _DATE_HINTS) or t.startswith("when"):
        return "date"
    if t.startswith("who"):
        return "who"
    if t.startswith("where"):
        return "where"
    if any(h in t for h in _NUMERIC_HINTS) or "how much" in t or "how many" in t:
        return "number"
    return "what"


def _is_comparison(question):
    if _COMPARE_RE.search(question):
        return True
    # "on A, B and C" with two capitalized subjects also implies comparison
    return bool(re.search(r'\bon\b.+,', question, re.I)) and \
        len(re.findall(r'\b[A-Z][a-zA-Z0-9]+', question)) >= 3


def _decompose_comparison(question):
    subjects_raw = _run_t5(
        "List only the product names being compared in this task, "
        f"separated by commas.\nTask: {question}\nProducts:"
    )
    attributes_raw = _run_t5(
        "List only the comparison attributes in this task, "
        f"separated by commas.\nTask: {question}\nAttributes:"
    )
    subjects = _split_items(subjects_raw)
    attributes = _split_items(attributes_raw)

    requirements = []
    if subjects and attributes:
        for subj in subjects:
            for attr in attributes:
                requirements.append({
                    "text": f"What is the {attr} of {subj}?",
                    "subject": subj,
                    "attribute": attr,
                    "qtype": _infer_qtype(attr),
                    "entities": {subj, attr},
                    "satisfied": False,
                    "answer": None,
                    "confidence": 0.0,
                })
    return requirements


def _single_fact_requirement(question):
    # Subject = capitalized noun phrases from the question, for the entity check
    caps = re.findall(r'\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b', question)
    entities = set(caps) if caps else set()
    subject = caps[0] if caps else ""
    return [{
        "text": question.rstrip("?").strip() + "?",
        "subject": subject,
        "attribute": "answer",
        "qtype": _infer_qtype(question),
        "entities": entities,
        "satisfied": False,
        "answer": None,
        "confidence": 0.0,
    }]


def extract_requirements(question):
    if _is_comparison(question):
        reqs = _decompose_comparison(question)
        if reqs:
            return reqs
    return _single_fact_requirement(question)


def passage_satisfies_requirement(passage_text, requirement, qa_model_fn,
                                  similarity_threshold=0.30,
                                  qa_confidence_threshold=0.20):
    req_emb = embed_model.encode(requirement["text"], convert_to_tensor=True)
    pas_emb = embed_model.encode(passage_text, convert_to_tensor=True)
    sim = float(util.cos_sim(req_emb, pas_emb)[0][0])
    if sim < similarity_threshold:
        return False, None, 0.0

    if requirement["entities"]:
        passage_lower = passage_text.lower()
        subject_found = False
        for entity in requirement["entities"]:
            entity_words = [w for w in entity.lower().split() if len(w) > 2]
            if entity.lower() in passage_lower or any(w in passage_lower for w in entity_words):
                subject_found = True
                break
        if not subject_found:
            return False, None, 0.0

    qa_score, answer = qa_model_fn(requirement["text"], passage_text)
    if qa_score < qa_confidence_threshold or not answer:
        return False, None, 0.0

    pattern = ANSWER_PATTERNS.get(requirement["qtype"], ANSWER_PATTERNS["what"])
    if not pattern.search(answer):
        return False, None, 0.0

    return True, answer, qa_score


class EvidenceCoverageTracker:
    def __init__(self, question):
        self.question = question
        self.requirements = extract_requirements(question)
        self.bytes_used = 0
        self.passages_checked = 0

    def n_satisfied(self):
        return sum(1 for r in self.requirements if r["satisfied"])

    def n_total(self):
        return len(self.requirements)

    def coverage(self):
        if not self.requirements:
            return 1.0
        return self.n_satisfied() / self.n_total()

    def is_sufficient(self, threshold=0.8):
        return self.coverage() >= threshold

    def frac_satisfied(self):
        return self.coverage()

    def frac_remaining(self):
        return 1.0 - self.coverage()

    def check_passage(self, passage_text, passage_bytes, qa_model_fn):
        self.bytes_used += passage_bytes
        self.passages_checked += 1
        any_satisfied = False
        for req in self.requirements:
            if req["satisfied"]:
                continue
            ok, answer, confidence = passage_satisfies_requirement(
                passage_text, req, qa_model_fn
            )
            if ok:
                req["satisfied"] = True
                req["answer"] = answer
                req["confidence"] = confidence
                any_satisfied = True
        return any_satisfied

    def best_answer(self):
        satisfied = [r for r in self.requirements if r["satisfied"]]
        if not satisfied:
            return None
        if len(satisfied) == 1:
            return satisfied[0]["answer"]
        return " | ".join(f"{r['subject']} {r['attribute']}: {r['answer']}"
                          for r in satisfied)

    def status_report(self):
        lines = []
        for r in self.requirements:
            mark = "✓" if r["satisfied"] else "✗"
            ans = f" → {r['answer']}" if r["satisfied"] else ""
            conf = f" (conf={r['confidence']:.2f})" if r["satisfied"] else ""
            lines.append(f"  {mark} [{r['subject']}] {r['attribute']}{ans}{conf}")
        return "\n".join(lines)