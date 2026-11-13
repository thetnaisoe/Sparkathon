"""Code-side checks: exact-quote verification, credibility tiers and domain independence."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache
from typing import Optional
from urllib.parse import urlsplit

from ...models import TIER_CAP, CredibilityTier

MIN_QUOTE_CHARS = 20
FACT_EVIDENCE_MAX = 0.7

_CHAR_MAP = str.maketrans({
    **{c: "'" for c in "‘’‚‛′´`"},
    **{c: '"' for c in "“”„‟″«»"},
    **{c: "-" for c in "‐‑‒–—―−﹘﹣－"},
    **{c: " " for c in "     　"},
    **{c: None for c in "­​‌‍⁠﻿"},
})
_EDGE_JUNK = " \t\r\n\"'.…"


@lru_cache(maxsize=64)
def _normalise_with_map(text: str) -> tuple[str, str, tuple[int, ...]]:
    """Normalise `text` and keep, for each output character, its index in the NFC-composed input.

    Returns (composed input, normalised text, index map). Characters are folded one at a
    time (quotes/dashes/spaces unified, NFKC, casefold) and whitespace runs collapse to one
    space, so a match in the normalised text maps back to a span of the page text.
    """
    composed = unicodedata.normalize("NFC", text)
    out: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(composed):
        folded = unicodedata.normalize("NFKC", ch.translate(_CHAR_MAP)).translate(_CHAR_MAP).casefold()
        for c in folded:
            if c.isspace():
                if not out or out[-1] == " ":
                    continue
                c = " "
            out.append(c)
            index.append(i)
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return composed, "".join(out), tuple(index)


def normalise(text: str) -> str:
    """Casefold, unify quotes/dashes/spaces, and collapse all whitespace to single spaces."""
    return _normalise_with_map(text)[1]


def find_quote(quote: str, page_text: str) -> Optional[str]:
    """The page's own wording of `quote`, or None when the quote does not occur on the page.

    Matching ignores case, whitespace and typography (see `normalise`); surrounding quotation
    marks and ellipses are ignored. Quotes shorter than MIN_QUOTE_CHARS or without letters
    are rejected as too weak to verify. The returned span is cut from the page text (with
    whitespace collapsed), so a reader can find it on the source page.
    """
    if not quote or not page_text:
        return None
    q = normalise(quote).strip(_EDGE_JUNK)
    if len(q) < MIN_QUOTE_CHARS or not any(ch.isalpha() for ch in q):
        return None
    composed, norm, index = _normalise_with_map(page_text)
    start = norm.find(q)
    if start < 0:
        return None
    begin, end = index[start], index[start + len(q) - 1] + 1
    if quote.rstrip().endswith((".", "…")):  # keep a sentence-final period the edge strip removed
        while end < len(composed) and composed[end] in ".…":
            end += 1
    return " ".join(composed[begin:end].split())


def verify_quote(quote: str, page_text: str) -> bool:
    """True when `quote` occurs verbatim in `page_text` (up to case, whitespace and typography)."""
    return find_quote(quote, page_text) is not None


# --------------------------------------------------------------------------- credibility

_PEER_REVIEWED = (
    "doi.org", "nature.com", "sciencedirect.com", "springer.com", "wiley.com", "ieee.org",
    "acm.org", "ncbi.nlm.nih.gov", "mdpi.com", "frontiersin.org", "plos.org", "science.org",
    "cell.com", "tandfonline.com", "sagepub.com", "iop.org", "acs.org", "rsc.org",
)
_GOVERNMENT = (
    "europa.eu", "bund.de", "destatis.de", "ons.gov.uk", "census.gov", "admin.ch", "gc.ca",
    "insee.fr", "istat.it", "cbs.nl", "ine.es", "statistik.at", "scb.se", "dst.dk", "ssb.no",
    "stat.fi", "cso.ie", "abs.gov.au", "stats.govt.nz", "bundesregierung.de", "bmwk.de",
    "umweltbundesamt.de", "bafa.de", "kfw.de", "arbeitsagentur.de",
)
_GOV_LABELS = {"gov", "gouv", "gob", "govt", "gv"}
_INDUSTRY = (
    "statista.com", "iea.org", "irena.org", "mckinsey.com", "gartner.com", "windeurope.org",
    "bnef.com", "oecd.org", "worldbank.org", "imf.org", "bcg.com", "deloitte.com", "pwc.com",
    "ey.com", "kpmg.com", "rolandberger.com", "idc.com", "forrester.com", "ember-energy.org",
    "unido.org", "wto.org",
)
_BLOG_HOSTS = ("medium.com", "substack.com", "wordpress.com", "blogspot.com", "tumblr.com",
               "ghost.io", "hashnode.dev", "quora.com", "reddit.com")


_HOSTNAME = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*")


def _host(url: str) -> str:
    try:
        host = (urlsplit(url if "//" in url else f"//{url}").hostname or "").lower().rstrip(".")
    except ValueError:  # malformed URL, e.g. a broken IPv6 literal
        return ""
    if not _HOSTNAME.fullmatch(host):
        return ""
    return host[4:] if host.startswith("www.") else host


def _under(host: str, domains: Iterable[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def credibility_tier(url: str) -> CredibilityTier:
    """Heuristic tier from the URL. Company sites cannot be told apart reliably, so they get 'unknown'."""
    host = _host(url)
    if not host:
        return "unknown"
    path = urlsplit(url).path.lower()
    labels = host.split(".")
    if _under(host, _PEER_REVIEWED):
        return "peer_reviewed"
    if (
        labels[-1] in ("gov", "mil")
        or (len(labels) >= 2 and labels[-2] in _GOV_LABELS)
        or _under(host, _GOVERNMENT)
    ):
        return "government"
    if _under(host, _INDUSTRY):
        return "industry_report"
    if (
        _under(host, _BLOG_HOSTS)
        or (_under(host, ("linkedin.com",)) and path.startswith("/pulse"))
        or labels[0] == "blog"
        or "/blog/" in path + "/"
    ):
        return "blog"
    return "unknown"


def evidence_strength(tier: str) -> float:
    """Evidence strength of a verified fact from a source of this tier (0.7, capped by TIER_CAP)."""
    return min(FACT_EVIDENCE_MAX, TIER_CAP.get(tier, TIER_CAP["unknown"]))


# --------------------------------------------------------------------------- independence

_TWO_PART_SUFFIXES = frozenset(
    f"{a}.{b}"
    for b, seconds in {
        "uk": "co ac gov org ltd plc me net nhs police",
        "au": "com net org edu gov asn id",
        "nz": "co ac govt org net school",
        "jp": "co ac go or ne",
        "kr": "co ac go or ne",
        "in": "co ac gov org net res",
        "za": "co ac gov org",
        "br": "com gov org net edu",
        "cn": "com gov org net edu ac",
        "at": "co ac gv or",
        "il": "co ac gov org",
        "sg": "com gov edu org",
        "hk": "com gov edu org",
        "tw": "com gov edu org",
        "mx": "com gob org edu",
        "ar": "com gob org edu",
        "tr": "com gov org edu",
    }.items()
    for a in seconds.split()
)


def registrable_domain(url: str) -> str:
    """Approximate registrable domain: last two labels, or three under a two-part suffix like co.uk."""
    host = _host(url)
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = host.split(".")
    n = 3 if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_PART_SUFFIXES else 2
    return ".".join(labels[-n:])


def count_independent_domains(urls: Iterable[str]) -> int:
    """Number of distinct registrable domains among `urls` (empty hosts ignored)."""
    return len({d for d in (registrable_domain(u) for u in urls) if d})


_INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions"
    r"|ignore\s+all\s+prior"
    r"|disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions"
    r"|system\s+prompt",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    """True when the text contains phrases aimed at an AI model's instructions."""
    return bool(_INJECTION.search(text))
