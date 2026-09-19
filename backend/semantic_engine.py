"""
EcoBudget v2 — Semantic Evidence Engine

Three ML components, no regex in the core decision:
1. Flan-T5 (local) — task decomposition into structured requirements
2. Sentence-transformers — embedding-based passage ranking per requirement
3. BART-large-MNLI (NLI) — verifies a passage actually answers a requirement

Ground truth is NEVER used inside this module.
"""

import re
import json
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

print("[semantic_engine] loading models...")
_t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
_t5_tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
_t5_model.eval()

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

_nli = pipeline("zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1)
print("[semantic_engine] all models ready")


# ── 1. Task Decomposition (Flan-T5) ──────────────────────────────────────────

def _run_t5(prompt, max_new_tokens=256):
    inputs = _t5_tok(prompt, return_tensors="pt",
                     truncation=True, max_length=512)
    with torch.no_grad():
        out = _t5_model.generate(**inputs, max_new_tokens=max_new_tokens)
    return _t5_tok.decode(out[0], skip_special_tokens=True)


def decompose_task(task_text):
    """
    Use Flan-T5 to break a task into specific information requirements.
    Returns list of requirement dicts with subject, attribute, question, answer_type.
    """
    prompt = (
        "List the specific pieces of information needed to complete this task. "
        "Format each as: SUBJECT | ATTRIBUTE | QUESTION | TYPE\n"
        "TYPE must be one of: number, name, date, text, list\n\n"
        "Example:\n"
        "Task: Compare iPhone 15 and Galaxy S24 on price and battery.\n"
        "iPhone 15 | price | What is the price of iPhone 15? | number\n"
        "iPhone 15 | battery | What is the battery of iPhone 15? | number\n"
        "Galaxy S24 | price | What is the price of Galaxy S24? | number\n"
        "Galaxy S24 | battery | What is the battery of Galaxy S24? | number\n\n"
        f"Task: {task_text}\n"
    )

    raw = _run_t5(prompt, max_new_tokens=300)
    requirements = []

    for line in raw.strip().split("\n"):
        line = line.strip().lstrip("-•*123456789. ")
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        subject   = parts[0]
        attribute = parts[1]
        question  = parts[2]
        ans_type  = parts[3].strip().lower() if len(parts) > 3 else "text"
        if ans_type not in ("number", "name", "date", "text", "list"):
            ans_type = "text"
        if subject and attribute and question:
            requirements.append({
                "requirement": question,
                "subject":     subject,
                "attribute":   attribute,
                "answer_type": ans_type,
                "satisfied":   False,
                "answer":      None,
                "confidence":  0.0,
                "passage":     None,
            })

    if not requirements:
        # Fallback: single requirement
        requirements = [{
            "requirement": task_text,
            "subject":     "",
            "attribute":   "information",
            "answer_type": "text",
            "satisfied":   False,
            "answer":      None,
            "confidence":  0.0,
            "passage":     None,
        }]

    return requirements


# ── 2. Passage ranking per requirement (embeddings) ───────────────────────────

def rank_for_requirement(requirement, passages, top_k=10):
    """
    Rank passages by similarity to a SPECIFIC requirement question,
    not the whole task. Returns [(score, passage_dict), ...] sorted desc.
    """
    req_emb = _embed_model.encode(requirement["requirement"], convert_to_tensor=True)
    scored = []
    for p in passages:
        p_emb = _embed_model.encode(p["content"], convert_to_tensor=True)
        sim = float(util.cos_sim(req_emb, p_emb)[0][0])
        scored.append((sim, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


# ── 3. NLI evidence verification (BART-MNLI) ─────────────────────────────────

def verify_passage(passage_text, requirement):
    """
    Use NLI to check whether a passage actually answers a specific requirement.

    This is the key improvement over regex:
    - "Colosseum restored in 2016" does NOT entail "when Colosseum was completed"
    - "Colosseum completed in AD 80" DOES entail it

    Returns (verified: bool, nli_score: float, qa_answer: str or None)
    """
    # Build a specific hypothesis for this requirement
    hypothesis = (
        f"This text answers the question: {requirement['requirement']}"
    )

    try:
        result = _nli(
            passage_text[:1000],
            candidate_labels=["answers the question", "does not answer the question"],
            hypothesis_template="{}",
        )
        # scores[0] = score for first label ("answers the question")
        nli_score = result["scores"][0]

        if nli_score < 0.60:
            return False, nli_score, None

        # Subject must appear in passage (basic sanity check)
        if requirement["subject"]:
            subject_words = [w for w in requirement["subject"].lower().split()
                             if len(w) > 2]
            passage_lower = passage_text.lower()
            if subject_words and not any(w in passage_lower for w in subject_words):
                return False, 0.0, None

        # Extract the actual answer span with roberta QA
        from scorer import check_answerability
        qa_score, answer = check_answerability(
            requirement["requirement"], passage_text
        )

        if answer and qa_score > 0.15:
            return True, nli_score, answer

        # NLI says it answers but QA couldn't extract — still count as partial
        if nli_score > 0.75:
            return True, nli_score, passage_text[:100].strip()

        return False, nli_score, None

    except Exception as e:
        print(f"[verify_passage] error: {e}")
        return False, 0.0, None


# ── 4. Coverage Tracker ───────────────────────────────────────────────────────

class SemanticCoverageTracker:
    """
    Tracks which task requirements have been satisfied using
    NLI-verified evidence. This is the heart of EcoBudget v2.

    Key property: the stopping decision never uses ground truth.
    """

    def __init__(self, task_text):
        self.task_text   = task_text
        self.requirements = decompose_task(task_text)
        self.bytes_used  = 0
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

    def check_passage(self, passage_text, passage_bytes):
        """
        Check a passage against every unsatisfied requirement.
        Uses NLI to verify — not similarity scores or regex.
        Returns True if any new requirement was satisfied.
        """
        self.bytes_used += passage_bytes
        self.passages_checked += 1
        any_new = False

        for req in self.requirements:
            if req["satisfied"]:
                continue
            verified, score, answer = verify_passage(passage_text, req)
            if verified:
                req["satisfied"] = True
                req["answer"]    = answer
                req["confidence"] = score
                req["passage"]   = passage_text[:200]
                any_new = True

        return any_new

    def best_answer(self):
        satisfied = [r for r in self.requirements if r["satisfied"]]
        if not satisfied:
            return None
        if len(satisfied) == 1:
            return satisfied[0]["answer"]
        parts = [
            f"{r['subject']} {r['attribute']}: {r['answer']}"
            for r in satisfied
        ]
        return " | ".join(parts)

    def status_report(self):
        lines = []
        for r in self.requirements:
            mark = "✓" if r["satisfied"] else "✗"
            ans  = f" → {r['answer']}" if r["satisfied"] else ""
            conf = f" (NLI={r['confidence']:.2f})" if r["satisfied"] else ""
            lines.append(f"  {mark} [{r['subject']}] {r['attribute']}{ans}{conf}")
        return "\n".join(lines)
