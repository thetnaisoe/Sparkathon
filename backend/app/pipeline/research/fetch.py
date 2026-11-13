"""Fetch a web page and extract its main text (httpx + trafilatura). Never raises."""

from __future__ import annotations

import codecs
import hashlib
import html
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit

import httpx
import trafilatura

log = logging.getLogger("research.fetch")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MAX_BYTES = 3 * 1024 * 1024
_SKIP_EXT = re.compile(
    r"\.(pdf|docx?|xlsx?|pptx?|zip|gz|tar|rar|7z|csv|json|xml|rss|png|jpe?g|gif|svg|webp|"
    r"mp[34]|mov|avi|wav|exe|dmg|epub)$",
    re.IGNORECASE,
)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Page:
    url: str
    domain: str
    title: Optional[str] = None
    text: str = ""
    text_hash: Optional[str] = None
    fetched_at: datetime = field(default_factory=_now)
    ok: bool = False
    error: Optional[str] = None


FetchFn = Callable[[str], Page]


def domain_of(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_probably_html(url: str) -> bool:
    """False for non-http(s) URLs and links to PDFs, documents, media or archives."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not host:
        return False
    return not _SKIP_EXT.search(parts.path)


def _fail(url: str, error: str) -> Page:
    return Page(url=url, domain=domain_of(url), ok=False, error=error[:200])


def _read_capped(resp: httpx.Response) -> bytes:
    buf = bytearray()
    for piece in resp.iter_bytes():
        buf.extend(piece)
        if len(buf) >= MAX_BYTES:
            break
    return bytes(buf[:MAX_BYTES])


_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)
_LATIN1_ALIASES = {"latin-1", "iso8859-1", "ascii"}  # browsers decode these as windows-1252


def _charset_codec(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    try:
        codec = codecs.lookup(name).name
    except LookupError:
        return None
    return "cp1252" if codec in _LATIN1_ALIASES else codec


def _decode(raw: bytes, charset: Optional[str], *, sniff_meta: bool = False) -> str:
    """Decode with the header charset, else a <meta charset> in the first 4 KB (HTML only),
    else UTF-8 with a windows-1252 fallback for legacy pages."""
    codec = _charset_codec(charset)
    if codec is None and sniff_meta:
        match = _META_CHARSET.search(raw[:4096])
        if match:
            codec = _charset_codec(match.group(1).decode("ascii", "ignore"))
    if codec is not None:
        return raw.decode(codec, errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def fetch_page(url: str, timeout: float = 8.0, *, client: Optional[httpx.Client] = None) -> Page:
    """Download `url` and extract readable text. Failures give ok=False with a short error."""
    if not is_probably_html(url):
        return _fail(url, "skipped: not an HTML page")
    own = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"},
    )
    try:
        with http.stream("GET", url, timeout=timeout) as resp:
            final_url = str(resp.url)
            if resp.status_code >= 400:
                return _fail(final_url, f"HTTP {resp.status_code}")
            ctype = resp.headers.get("content-type", "").lower()
            mime = ctype.split(";")[0].strip()
            if mime and mime not in ("text/html", "application/xhtml+xml", "text/plain"):
                return _fail(final_url, f"skipped: content type {mime}")
            raw = _read_capped(resp)
            charset = resp.charset_encoding
    except httpx.TimeoutException:
        return _fail(url, "timed out")
    except Exception as exc:  # DNS, TLS, protocol errors, ...
        return _fail(url, f"fetch error: {type(exc).__name__}: {exc}")
    finally:
        if own:
            http.close()

    body = _decode(raw, charset, sniff_meta=mime != "text/plain")
    title = None
    if mime == "text/plain":
        text = body.strip()
    else:
        match = _TITLE.search(body[:200_000])
        if match:
            title = " ".join(html.unescape(match.group(1)).split())[:300] or None
        try:
            text = (trafilatura.extract(body, url=final_url, include_comments=False) or "").strip()
        except Exception as exc:
            return _fail(final_url, f"extraction error: {type(exc).__name__}")
    if not text:
        return _fail(final_url, "no readable text")
    return Page(url=final_url, domain=domain_of(final_url), title=title, text=text,
                text_hash=text_hash(text), ok=True)
