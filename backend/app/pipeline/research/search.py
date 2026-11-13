"""Web search: Tavily (needs TAVILY_API_KEY) with OpenAlex (no key) as the fallback."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import httpx

log = logging.getLogger("research.search")

TAVILY_URL = "https://api.tavily.com/search"
OPENALEX_URL = "https://api.openalex.org/works"
SNIPPET_CHARS = 300


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str = ""


SearchFn = Callable[[str], list[SearchHit]]


def tavily_search(
    query: str,
    *,
    max_results: int = 5,
    timeout: float = 8.0,
    client: Optional[httpx.Client] = None,
) -> list[SearchHit]:
    """Search the web with Tavily. Returns [] when TAVILY_API_KEY is not set; raises on HTTP errors."""
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        log.info("tavily search skipped: TAVILY_API_KEY is not set")
        return []
    body = {"query": query, "max_results": max_results, "search_depth": "basic"}
    headers = {"Authorization": f"Bearer {key}"}
    if client is None:
        with httpx.Client(timeout=timeout) as own:
            resp = own.post(TAVILY_URL, json=body, headers=headers)
    else:
        resp = client.post(TAVILY_URL, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    hits = []
    for item in resp.json().get("results") or []:
        url = (item.get("url") or "").strip()
        if url:
            hits.append(SearchHit(url, (item.get("title") or "").strip(),
                                  (item.get("content") or "")[:SNIPPET_CHARS]))
    return hits[:max_results]


def _abstract(inverted: Optional[dict[str, list[int]]]) -> str:
    """Rebuild the start of an OpenAlex abstract from its inverted index."""
    if not inverted:
        return ""
    positions = sorted((pos, word) for word, poss in inverted.items() for pos in poss)
    return " ".join(word for _, word in positions)[:SNIPPET_CHARS]


def _openalex_url(work: dict[str, Any]) -> str:
    doi = (work.get("doi") or "").strip()
    if doi:
        return doi if doi.startswith("http") else f"https://doi.org/{doi}"
    location = work.get("primary_location") or {}
    return (location.get("landing_page_url") or work.get("id") or "").strip()


def openalex_search(
    query: str,
    *,
    max_results: int = 5,
    timeout: float = 8.0,
    client: Optional[httpx.Client] = None,
) -> list[SearchHit]:
    """Search scholarly works on OpenAlex (no key). Raises on HTTP errors."""
    params = {"search": query, "per-page": max_results}
    if client is None:
        with httpx.Client(timeout=timeout) as own:
            resp = own.get(OPENALEX_URL, params=params)
    else:
        resp = client.get(OPENALEX_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    hits = []
    for work in resp.json().get("results") or []:
        url = _openalex_url(work)
        if url:
            hits.append(SearchHit(url, (work.get("display_name") or work.get("title") or "").strip(),
                                  _abstract(work.get("abstract_inverted_index"))))
    return hits[:max_results]


def default_search(query: str) -> list[SearchHit]:
    """Tavily first; OpenAlex when Tavily has no key, fails or finds nothing."""
    try:
        hits = tavily_search(query)
    except Exception as exc:  # network or API error: fall back
        log.warning("tavily search failed for %r: %s", query, exc)
        hits = []
    if hits:
        return hits
    return openalex_search(query)


def tavily_configured() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())
