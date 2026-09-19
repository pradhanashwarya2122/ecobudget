from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re
import spacy

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)


def extract_capitalized_terms(text):
    matches = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
    stopwords = {'What', 'When', 'Where', 'Who', 'Why', 'How', 'Which', 'Is', 'Does', 'Do'}
    return set(m.lower() for m in matches if m not in stopwords and len(m) > 2)


def entity_consistency_score(task_text, candidate_text):
    task_entities = extract_capitalized_terms(task_text)
    candidate_entities = extract_entities(candidate_text)

    if not task_entities:
        return 1.0
    if not candidate_entities:
        return 0.7

    overlap = task_entities & candidate_entities
    if overlap:
        return 1.0

    for te in task_entities:
        for ce in candidate_entities:
            if te in ce or ce in te:
                return 1.0

    return 0.3


def get_real_image_bytes(src, base_url=None):
    if not src:
        return 50_000, True
    url = src
    if src.startswith('//'):
        url = 'https:' + src
    elif base_url and not src.startswith('http'):
        url = base_url.rstrip('/') + (src if src.startswith('/') else '/' + src)
    headers = {'User-Agent': 'Mozilla/5.0 (EcoBudget Research Bot; contact: student-project)'}
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        size = resp.headers.get('Content-Length')
        if size and int(size) > 500:
            return int(size), False
    except Exception:
        pass
    return 50_000, True


def parse_resources(html, base_url=None, debug=False):
    soup = BeautifulSoup(html, 'html.parser')
    content_root = (soup.find(id='mw-content-text')
                     or soup.find(id='bodyContent')
                     or soup.find('div', class_='mw-parser-output')
                     or soup.find('body')
                     or soup)

    for ref_section in content_root.find_all(['ol', 'div'], class_=['references', 'reflist', 'refbegin']):
        ref_section.decompose()
    for cite_tag in content_root.find_all('cite'):
        cite_tag.decompose()

    infobox = content_root.find('table', class_=lambda c: c and 'infobox' in c)
    infobox_text = ""
    if infobox:
        infobox_text = infobox.get_text(separator=' | ', strip=True)
        infobox.decompose()

    for junk in content_root.find_all(['script', 'style', 'nav', 'footer', 'sup', 'table']):
        junk.decompose()

    resources = []
    seen_texts = set()

    if infobox_text:
        resources.append({"type": "text", "content": infobox_text, "bytes": len(infobox_text.encode('utf-8'))})

    for tag in content_root.find_all(['p', 'li', 'h2', 'h3']):
        text = tag.get_text(separator=' ', strip=True)
        if text.startswith('↑') or text.startswith('^'):
            continue
        if text and len(text) > 20 and text not in seen_texts:
            seen_texts.add(text)
            resources.append({"type": "text", "content": text, "bytes": len(text.encode('utf-8'))})

    for img in content_root.find_all('img'):
        alt = img.get('alt', 'image')
        src = img.get('src', '')
        real_bytes, is_estimated = get_real_image_bytes(src, base_url)
        resources.append({"type": "image", "content": alt, "bytes": real_bytes, "bytes_estimated": is_estimated})

    return resources


def compute_utility(task_text, resource_text):
    emb_task = embed_model.encode(task_text, convert_to_tensor=True)
    emb_res = embed_model.encode(resource_text, convert_to_tensor=True)
    return float(util.cos_sim(emb_task, emb_res)[0][0])


def check_answerability(task_text, text_chunk):
    if len(text_chunk.strip()) < 10:
        return 0.0, None
    try:
        result = qa_model(question=task_text, context=text_chunk, handle_impossible_answer=True)
        if not result['answer'].strip():
            return 0.0, None
        return result['score'], result['answer']
    except Exception:
        return 0.0, None


def rank_by_vpb(task_text, resources, use_entity_filter=True):
    text_resources = [r for r in resources if r['type'] == 'text']
    for r in text_resources:
        r['embed_utility'] = compute_utility(task_text, r['content'])
    candidates = sorted(text_resources, key=lambda r: r['embed_utility'], reverse=True)[:30]

    for r in candidates:
        score, answer = check_answerability(task_text, r['content'])
        r['qa_score'] = score   # raw QA confidence — never multiplied by anything
        if use_entity_filter:
            consistency = entity_consistency_score(task_text, r['content'])
            r['entity_consistency'] = consistency
            score = score * consistency
        else:
            r['entity_consistency'] = 1.0
        r['utility'] = score
        r['extracted_answer'] = answer
        r['vpb'] = r['utility'] / max(r['bytes'], 1)

    return sorted(candidates, key=lambda r: r['vpb'], reverse=True)


def find_best_answer_full_page(task_text, resources, use_entity_filter=True):
    text_resources = [r for r in resources if r['type'] == 'text']
    best_score = 0.0
    best_answer = None
    for r in text_resources:
        score, answer = check_answerability(task_text, r['content'])
        if use_entity_filter:
            score = score * entity_consistency_score(task_text, r['content'])
        if score > best_score:
            best_score = score
            best_answer = answer
    return best_score, best_answer


def check_task_success(extracted_answer, ground_truth):
    if not extracted_answer:
        return False
    ea = extracted_answer.lower().strip()
    gt = ground_truth.lower().strip()
    if gt in ea or ea in gt:
        return True
    # word-order-agnostic fallback (handles "AD 80" vs "80 AD")
    ea_tokens = set(ea.replace(',', '').split())
    gt_tokens = set(gt.replace(',', '').split())
    if gt_tokens and gt_tokens.issubset(ea_tokens):
        return True
    return False


def check_task_success_v2(extracted_answer, selected_text, ground_truth):
    """
    Handles both formats:
    - ground_truth is a str -> substring match against extracted_answer (old behavior)
    - ground_truth is a dict with 'required_facts' -> checks presence in the
      full selected_text (not just the one QA-extracted span).
    Returns (success: bool, facts_matched: int, facts_total: int)
    """
    if isinstance(ground_truth, str):
        success = check_task_success(extracted_answer, ground_truth)
        return success, (1 if success else 0), 1

    required = ground_truth["required_facts"]
    threshold = ground_truth.get("match_threshold", 1.0)
    haystack = (selected_text or "").lower()

    matched = sum(1 for fact in required if fact.lower() in haystack)
    total = len(required)
    success = (matched / total) >= threshold if total else False
    return success, matched, total


# ── Evidence Gate ──────────────────────────────────────────────────────────────

import re as _re

_QUESTION_FORMS = {
    "when":     _re.compile(r'\b(\d{1,4}s?(\s*(AD|BC|BCE|CE))?|[A-Z][a-z]+ \d{4}|\d{1,2} [A-Z][a-z]+ \d{4})\b'),
    "how_tall": _re.compile(r'\b\d+[\.,]?\d*\s*(m|ft|km|meters?|feet|kilometres?)\b', _re.I),
    "how_long": _re.compile(r'\b\d+[\.,]?\d*\s*(m|ft|km|meters?|feet|kilometres?|miles?)\b', _re.I),
    "how_many": _re.compile(r'\b\d+[\.,]?\d*\b'),
    "who":      _re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
    "where":    _re.compile(r'\b[A-Z][a-zA-Z\s,]+\b'),
    "what":     _re.compile(r'.{10,}'),
}

def _question_type(question):
    q = question.lower().strip()
    if q.startswith("when") or "what year" in q or "what date" in q:
        return "when"
    if "how tall" in q or "how high" in q:
        return "how_tall"
    if "how long" in q or "how far" in q:
        return "how_long"
    if "how many" in q:
        return "how_many"
    if q.startswith("who") or "who " in q:
        return "who"
    if q.startswith("where") or "what city" in q or "what country" in q or "what state" in q:
        return "where"
    return "what"

def _extract_question_terms(question):
    stopwords = {'What', 'When', 'Where', 'Who', 'Why', 'How', 'Which',
                 'Is', 'Does', 'Do', 'Was', 'Were', 'The', 'A', 'An',
                 'In', 'Of', 'For', 'To', 'And', 'Or'}
    tokens = question.split()
    return [t.strip('?.,') for t in tokens
            if len(t) > 2 and t[0].isupper() and t.strip('?.,') not in stopwords]

def evidence_gate(question, selected_resources, qa_answer, qa_score,
                  relevance_threshold=0.45, qa_threshold=0.40):
    """
    Returns (sufficient: bool, reason: str).
    Uses four task-agnostic signals — ground truth is never used here.
    """
    # 0. Minimum resources: never stop after just 1 resource — too likely to be
    #    an infobox snippet with a confident-but-wrong date/name
    if len(selected_resources) < 3:
        return False, f"too few resources loaded ({len(selected_resources)})"

    # 1. Relevance: at least one loaded passage is topically relevant
    max_relevance = max((r['utility'] for r in selected_resources), default=0.0)
    if max_relevance < relevance_threshold:
        return False, f"no relevant passage (max_sim={max_relevance:.2f})"

    # 2. Key term presence: loaded text contains named entities from the question
    key_terms = _extract_question_terms(question)
    loaded_text = " ".join(r['content'] for r in selected_resources).lower()
    terms_found = [t for t in key_terms if t.lower() in loaded_text]
    term_coverage = len(terms_found) / len(key_terms) if key_terms else 1.0
    if term_coverage < 0.5:
        return False, f"key terms missing ({len(terms_found)}/{len(key_terms)})"

    # 3. Answer form: extracted answer matches expected shape for this question type
    if not qa_answer:
        return False, "no answer extracted yet"
    qtype = _question_type(question)
    pattern = _QUESTION_FORMS.get(qtype, _QUESTION_FORMS["what"])
    if not pattern.search(qa_answer):
        return False, f"answer '{qa_answer[:30]}' wrong form for '{qtype}'"

    # 4. QA confidence: model is sufficiently confident
    if qa_score < qa_threshold:
        return False, f"QA confidence too low ({qa_score:.2f})"

    # 5. Answer-question coherence: answer should not be completely unrelated to question terms
    # Catches cases where QA is confident but answering a different implicit question in the passage
    question_content_words = set(
        w.lower().strip("?.,") for w in question.split()
        if len(w) > 3 and w.lower() not in {
            'what', 'when', 'where', 'which', 'does', 'called', 'scientists',
            'process', 'that', 'this', 'with', 'from', 'have', 'were', 'their'
        }
    )
    answer_words = set(qa_answer.lower().split())
    loaded_text_lower = loaded_text  # already lowercased above

    # The answer OR its context window must overlap with question content words
    if question_content_words:
        answer_overlaps = question_content_words & answer_words
        # Check if answer appears near question terms in the loaded text
        answer_in_context = False
        if qa_answer:
            ans_lower = qa_answer.lower()
            idx = loaded_text_lower.find(ans_lower)
            if idx >= 0:
                window = loaded_text_lower[max(0, idx-150):idx+150]
                context_overlaps = question_content_words & set(window.split())
                answer_in_context = len(context_overlaps) >= 1

        if not answer_overlaps and not answer_in_context:
            return False, f"answer '{qa_answer[:25]}' unrelated to question terms"

    return True, f"sufficient (sim={max_relevance:.2f} qa={qa_score:.2f} type={qtype})"
