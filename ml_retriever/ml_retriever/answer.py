"""A minimal AnswerGenerator for Phase 5.

The bandit's reward needs *some* answer to judge at STOP. Phase 6 builds the
real generative answerer; for Phase 5 this extractive stand-in assembles an
answer from the evidence the tracker has gathered -- concatenating the evidence
passage texts. Because the corpus passages contain the actual values
("$799", "3349 mAh"), the rule-based judge (judge.py) then succeeds exactly
when retrieval has gathered enough evidence to answer -- which is the signal
the bandit should be optimizing against its byte cost. It uses ONLY the
passed-in evidence (Phase 0 AnswerGenerator contract): no ground truth, no
re-fetching.
"""

from __future__ import annotations

from typing import Callable, Optional

from .types import AnswerResult, Passage

_MAX_ANSWER_CHARS = 2000

# How the generative answerer sees the evidence. Kept in one place so the
# training-data builder and inference use the identical prompt.
ANSWER_PROMPT = (
    "Answer the question using only the evidence below. Be concise.\n"
    "Question: {question}\nEvidence: {evidence}\nAnswer:"
)


def _join_evidence(passages: list[Passage]) -> tuple[str, list[str]]:
    seen: set[str] = set()
    texts, ids = [], []
    for p in passages:
        if p.passage_id in seen:
            continue
        seen.add(p.passage_id)
        texts.append(p.text)
        ids.append(p.passage_id)
    return " ".join(texts), ids


class EvidenceAnswerGenerator:
    """Assembles an answer by concatenating gathered evidence passages.

    A deliberately simple v1 generator: it abstains when given no evidence,
    otherwise returns the joined evidence text (deduped by passage_id, bounded
    in length). Phase 6 replaces it with a trained generative model behind the
    same interface.
    """

    version = "evidence-concat-v1"

    def generate(self, question: str, evidence: list[Passage]) -> AnswerResult:
        seen: set[str] = set()
        ordered: list[Passage] = []
        for p in evidence:
            if p.passage_id in seen:
                continue
            seen.add(p.passage_id)
            ordered.append(p)

        if not ordered:
            return AnswerResult(
                answer=None, confidence=0.0, used_evidence_ids=[],
                abstained=True, generator_version=self.version,
            )

        text = " ".join(p.text for p in ordered)[:_MAX_ANSWER_CHARS]
        return AnswerResult(
            answer=text,
            confidence=1.0,
            used_evidence_ids=[p.passage_id for p in ordered],
            abstained=False,
            generator_version=self.version,
        )


class ExtractiveQAAnswerGenerator:
    """Single-fact answerer: RoBERTa extractive QA over the evidence, with
    explicit abstention.

    The plan flags the raw QA path as "unreliable-when-confident", so this
    wrapper abstains when the best answer-span confidence across the evidence
    is below `min_confidence` (or there is no evidence) rather than passing the
    confidence straight through. Works only for single-value answers; the
    generative path handles comparison/multi-part/narrative.
    """

    def __init__(self, requirement, scorer=None, min_confidence: float = 0.2):
        from .evidence import QAScorer

        self.requirement = requirement
        self.scorer = scorer or QAScorer()
        self.min_confidence = min_confidence
        self.version = f"extractive-qa-v1:min_conf={min_confidence}"

    def generate(self, question: str, evidence: list[Passage]) -> AnswerResult:
        best_span, best_conf, best_id = "", 0.0, None
        for p in evidence:
            span = self.scorer.extract_answer(self.requirement, p)
            conf = self.scorer(self.requirement, p)
            if span and conf > best_conf:
                best_span, best_conf, best_id = span, conf, p.passage_id
        if not best_span or best_conf < self.min_confidence:
            return AnswerResult(answer=None, confidence=best_conf, used_evidence_ids=[],
                                abstained=True, generator_version=self.version)
        return AnswerResult(answer=best_span, confidence=best_conf, used_evidence_ids=[best_id],
                            abstained=False, generator_version=self.version)


class GenerativeAnswerGenerator:
    """Phase 6 primary answerer: a fine-tuned local seq2seq model
    (flan-t5-small/base, optional LoRA) that synthesizes an answer from the
    retrieved evidence -- producing a judgment ("Samsung S24, 120Hz vs 60Hz")
    rather than dumping raw values, which templates cannot do.

    Abstain logic (explicit, per plan.md): abstain when there is no evidence,
    or when the model returns an empty/degenerate generation. `generator_version`
    records the frozen checkpoint so Phase 7 results are attributable. The model
    is loaded lazily; a `generate_fn(prompt)->str` can be injected to test the
    wrapper (abstain, prompt assembly, result shape) without a model.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 64,
        generate_fn: Optional[Callable[[str], str]] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._generate_fn = generate_fn
        self._tokenizer = None
        self._model = None
        self.version = f"generative:{model_name_or_path}"
        if adapter_path:
            self.version += f"+lora:{adapter_path}"

    def _generate(self, prompt: str) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(prompt)
        if self._model is None:
            import torch  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
            if self.adapter_path:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            self._model.eval()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       no_repeat_ngram_size=3)
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def generate(self, question: str, evidence: list[Passage]) -> AnswerResult:
        evidence_text, ids = _join_evidence(evidence)
        if not ids:
            return AnswerResult(answer=None, confidence=0.0, used_evidence_ids=[],
                                abstained=True, generator_version=self.version)
        prompt = ANSWER_PROMPT.format(question=question, evidence=evidence_text[:_MAX_ANSWER_CHARS])
        text = self._generate(prompt).strip()
        if not text:
            return AnswerResult(answer=None, confidence=0.0, used_evidence_ids=[],
                                abstained=True, generator_version=self.version)
        return AnswerResult(answer=text, confidence=1.0, used_evidence_ids=ids,
                            abstained=False, generator_version=self.version)
