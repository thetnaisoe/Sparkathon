"""search.py and fetch.py with httpx.MockTransport: no network, no real keys."""

import json

import httpx
import pytest

from app.pipeline.research import search as search_mod
from app.pipeline.research.fetch import fetch_page, is_probably_html
from app.pipeline.research.search import SearchHit, default_search, openalex_search, tavily_search

HTML = (
    "<html><head><title>Machine building in Germany</title></head><body><article>"
    + "<p>There were 2,450 machine-building companies with more than 20 employees in Germany in 2023.</p>" * 4
    + "</article></body></html>"
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


# --------------------------------------------------------------------------- fetch


def test_fetch_page_extracts_text_and_hash():
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://www.example.org/new"})
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=HTML)

    page = fetch_page("https://www.example.org/old", client=_client(handler))
    assert page.ok and page.error is None
    assert page.url == "https://www.example.org/new"
    assert page.domain == "example.org"
    assert page.title == "Machine building in Germany"
    assert "2,450 machine-building companies" in page.text
    assert page.text_hash.startswith("sha256:") and len(page.text_hash) == 7 + 64
    assert page.fetched_at.tzinfo is not None


def test_fetch_page_plain_text():
    client = _client(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, text="  hello world  "))
    page = fetch_page("https://example.org/a.txt", client=client)
    assert page.ok and page.text == "hello world"


@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(404, text="nope"), "HTTP 404"),
        (httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"), "content type"),
        (httpx.Response(200, headers={"content-type": "text/html"}, text="<html><body></body></html>"), "no readable text"),
    ],
)
def test_fetch_page_failures_return_not_ok(response, error):
    page = fetch_page("https://example.org/page", client=_client(lambda r: response))
    assert not page.ok
    assert error in page.error
    assert page.text == "" and page.text_hash is None


def test_fetch_page_network_error_never_raises():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    page = fetch_page("https://example.org/page", client=_client(handler))
    assert not page.ok and "ConnectError" in page.error

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    assert fetch_page("https://example.org/page", client=_client(slow)).error == "timed out"


def test_fetch_page_skips_pdf_without_request():
    def handler(request):  # pragma: no cover - must not be called
        raise AssertionError("PDF must not be requested")

    page = fetch_page("https://example.org/report.PDF", client=_client(handler))
    assert not page.ok and page.error.startswith("skipped")
    assert not is_probably_html("ftp://example.org/x")
    assert not is_probably_html("https://example.org/data.xlsx")
    assert is_probably_html("https://example.org/report?id=3")


# --------------------------------------------------------------------------- search


def test_tavily_without_key_returns_empty(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request):  # pragma: no cover - must not be called
        raise AssertionError("no request without a key")

    assert tavily_search("wind farms Germany", client=_client(handler)) == []


def test_tavily_request_and_mapping(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"url": "https://a.example/x", "title": "A", "content": "snippet a"},
            {"url": "", "title": "no url"},
        ]})

    hits = tavily_search("wind farms Germany", max_results=3, client=_client(handler))
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"] == {"query": "wind farms Germany", "max_results": 3, "search_depth": "basic"}
    assert hits == [SearchHit("https://a.example/x", "A", "snippet a")]


def test_tavily_http_error_raises(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    with pytest.raises(httpx.HTTPStatusError):
        tavily_search("q", client=_client(lambda r: httpx.Response(401, json={"detail": "bad key"})))


def test_openalex_request_and_mapping():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"results": [
            {"doi": "https://doi.org/10.1/abc", "display_name": "Paper A",
             "abstract_inverted_index": {"Wind": [0], "farms": [1], "grow": [2]}},
            {"doi": None, "display_name": "Paper B",
             "primary_location": {"landing_page_url": "https://journal.example/b"}},
        ]})

    hits = openalex_search("wind farm maintenance", client=_client(handler))
    assert seen["params"] == {"search": "wind farm maintenance", "per-page": "5"}
    assert hits[0] == SearchHit("https://doi.org/10.1/abc", "Paper A", "Wind farms grow")
    assert hits[1].url == "https://journal.example/b" and hits[1].snippet == ""


def test_default_search_falls_back_to_openalex(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(search_mod, "openalex_search", lambda q: [SearchHit("https://doi.org/1", "P")])
    assert default_search("q") == [SearchHit("https://doi.org/1", "P")]

    def broken(q):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(search_mod, "tavily_search", broken)
    assert default_search("q")[0].url == "https://doi.org/1"


def test_fetch_page_uses_meta_charset_when_header_has_none():
    body = (
        '<html><head><meta charset="iso-8859-1"><title>Maschinenbau in Österreich</title></head><body><article>'
        + "<p>Im Jahr 2023 gab es 2.450 Unternehmen für Maschinenbau mit mehr als 20 Beschäftigten.</p>" * 4
        + "</article></body></html>"
    ).encode("latin-1")
    client = _client(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=body))
    page = fetch_page("https://example.de/maschinenbau", client=client)
    assert page.ok and "�" not in page.text
    assert "Unternehmen für Maschinenbau" in page.text and "Beschäftigten" in page.text
    assert page.title == "Maschinenbau in Österreich"


def test_fetch_page_falls_back_to_windows_1252_without_any_charset():
    body = ("<html><body><article>" + "<p>Über 2.450 Unternehmen – größer als erwartet.</p>" * 4
            + "</article></body></html>").encode("cp1252")
    client = _client(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=body))
    page = fetch_page("https://example.de/x", client=client)
    assert page.ok and "Über 2.450 Unternehmen – größer" in page.text
