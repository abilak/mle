from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


SKILL_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("dynamic_programming", ("dynamic programming", "memoization", "subproblem")),
    ("graphs", ("graph", "vertex", "edge", "shortest path", "tree", "dfs", "bfs")),
    ("strings", ("string", "substring", "character", "palindrome", "anagram")),
    ("arrays", ("array", "list", "sequence", "subarray", "matrix")),
    ("number_theory", ("prime", "divisor", "modulo", "gcd", "integer")),
    ("combinatorics", ("combination", "permutation", "count the number", "ways")),
    ("sorting_search", ("sort", "binary search", "search", "ordered")),
    ("data_structures", ("heap", "queue", "stack", "linked list", "hash")),
    ("geometry", ("geometry", "point", "polygon", "circle", "area", "distance")),
    ("regex_text", ("regular expression", "regex", "word", "text")),
    ("security_sensitive", ("crypto", "cipher", "hash", "password", "permission")),
    ("concurrency", ("thread", "concurrent", "async", "lock", "parallel")),
]


def infer_skill(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.lower())
    scores = [
        (sum(normalized.count(term) for term in terms), name)
        for name, terms in SKILL_PATTERNS
    ]
    best_score, best_name = max(scores)
    return best_name if best_score > 0 else "general"


def capability_metadata(prompt: str) -> dict[str, str | int]:
    normalized = prompt.lower()
    api_terms = {
        "collections": ("dictionary", "dict", "set", "counter", "deque"),
        "math": ("math", "sqrt", "prime", "gcd", "geometry"),
        "text": ("string", "text", "regex", "character"),
        "iterators": ("iterator", "generator", "permutation", "combination"),
        "concurrency": ("async", "thread", "lock", "concurrent"),
    }
    api_scores = {
        name: sum(normalized.count(term) for term in terms) for name, terms in api_terms.items()
    }
    api_family = max(api_scores, key=api_scores.get)
    if api_scores[api_family] == 0:
        api_family = "builtins"
    length = len(normalize_prompt(prompt).split())
    if length < 35:
        complexity = "short"
    elif length < 90:
        complexity = "medium"
    else:
        complexity = "long"
    return {
        "algorithm_type": infer_skill(prompt),
        "api_family": api_family,
        "prompt_complexity": complexity,
        "prompt_tokens_proxy": length,
    }


def common_and_rare(skills: Iterable[str], rare_fraction: float = 0.2) -> dict[str, str]:
    counts = Counter(skills)
    if not counts:
        return {}
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    cutoff = max(1, round(len(ordered) * rare_fraction))
    rare = {name for name, _ in ordered[:cutoff]}
    return {name: ("rare" if name in rare else "common") for name in counts}


def difficulty_bin(success_probability: float) -> str:
    if success_probability >= 2 / 3:
        return "initially_strong"
    if success_probability >= 1 / 3:
        return "initially_moderate"
    return "initially_weak"
