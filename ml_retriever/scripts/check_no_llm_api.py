#!/usr/bin/env python3
"""Fail if any source file in ml_retriever/ imports or references a hosted
LLM API client or a known LLM API host.

This exists because `teammate-1-ml-retriever/` (the previous version of
this workstream) was scrapped for depending on an LLM API. Everything in
this package must run on local models only.

Usage:
    python scripts/check_no_llm_api.py [root_dir]

Exits non-zero and prints violations if any are found. Also importable as
a function for use directly in tests.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Package names whose import alone indicates a hosted LLM API client.
BANNED_IMPORT_MODULES = {
    "openai",
    "anthropic",
    "cohere",
    "google.generativeai",
    "mistralai",
    "replicate",
    "together",
    "groq",
}

# Substrings that indicate a hardcoded call to a hosted LLM API host, even
# if reached indirectly (e.g. via `requests` or `httpx`).
BANNED_URL_SUBSTRINGS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.ai",
    "api.mistral.ai",
    "api.together.xyz",
    "api.groq.com",
)


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        # Skip the check script itself and virtual envs / caches.
        if path.name == "check_no_llm_api.py":
            continue
        if any(part in {".venv", "venv", "__pycache__", ".git"} for part in path.parts):
            continue
        yield path


def check_file(path: Path) -> list[str]:
    """Return a list of human-readable violation strings for one file."""
    violations: list[str] = []
    source = path.read_text(encoding="utf-8", errors="replace")

    for bad_url in BANNED_URL_SUBSTRINGS:
        if bad_url in source:
            violations.append(f"{path}: references banned LLM API host '{bad_url}'")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        violations.append(f"{path}: could not parse for import check ({exc})")
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in BANNED_IMPORT_MODULES or alias.name in BANNED_IMPORT_MODULES:
                    violations.append(f"{path}: imports banned module '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in BANNED_IMPORT_MODULES or node.module in BANNED_IMPORT_MODULES:
                    violations.append(f"{path}: imports from banned module '{node.module}'")

    return violations


def check_tree(root: Path) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root):
        violations.extend(check_file(path))
    return violations


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "ml_retriever"
    violations = check_tree(root)
    if violations:
        print("LLM API usage check FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"LLM API usage check passed ({root}): no banned imports or hosts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
