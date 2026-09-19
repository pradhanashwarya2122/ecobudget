"""Enforces the constraint that led to scrapping teammate-1-ml-retriever/:
no hosted LLM API client may be imported or referenced anywhere in
ml_retriever/ or scripts/. See scripts/check_no_llm_api.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from check_no_llm_api import check_tree  # noqa: E402


def test_package_has_no_llm_api_usage():
    violations = check_tree(ROOT / "ml_retriever")
    assert violations == [], "Found banned LLM API usage:\n" + "\n".join(violations)


def test_scripts_have_no_llm_api_usage():
    violations = check_tree(ROOT / "scripts")
    assert violations == [], "Found banned LLM API usage:\n" + "\n".join(violations)
