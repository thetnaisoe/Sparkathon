"""Split page text into overlapping passages and rank them with BM25."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from rank_bm25 import BM25Okapi

_WORD = re.compile(r"\S+")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was "
    "were what which who will with per each".split()
)


@dataclass(frozen=True)
class Passage:
    text: str
    start: int  # character offset into the original text
    end: int  # exclusive; text == original[start:end]
    doc: str = ""  # caller's document key (e.g. the page URL)
    score: float = 0.0


def chunk(text: str, target_tokens: int = 400, overlap: int = 50, *, doc: str = "") -> list[Passage]:
    """Overlapping windows of about `target_tokens` whitespace words with character offsets."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    overlap = max(0, min(overlap, target_tokens - 1))
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not words:
        return []
    step = target_tokens - overlap
    passages = []
    for i in range(0, len(words), step):
        window = words[i : i + target_tokens]
        start, end = window[0][0], window[-1][1]
        passages.append(Passage(text[start:end], start, end, doc))
        if i + target_tokens >= len(words):
            break
    return passages


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def rank(passages: list[Passage], query: str, k: int = 4) -> list[Passage]:
    """The `k` passages most relevant to `query` (BM25), best first, each with its score.

    Passages that share no term with the query are left out.
    """
    if not passages or k <= 0:
        return []
    q = tokenize(query)
    if not q:
        return []
    tokens = [tokenize(p.text) for p in passages]
    scores = BM25Okapi([t or [""] for t in tokens]).get_scores(q)
    qset = set(q)
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    out = []
    for i in order:
        if not qset.intersection(tokens[i]):
            continue
        out.append(replace(passages[i], score=float(scores[i])))
        if len(out) == k:
            break
    return out
