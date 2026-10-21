"""Generic RSS/Atom collector.

Works against any feed URL you configure. Deliberately generic rather than
targeted at a specific publisher: no EuroLeague news feed was reachable when this
was built, and hard-coding a source that may not be licensed for reuse would be a
liability rather than a feature.

Parsed with the standard library's XML parser rather than a feed library. The
subset of RSS and Atom that matters here is small, and the dependency would carry
its own parsing quirks for no benefit.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any
from xml.etree import ElementTree

import httpx

from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.logging import get_logger
from euroleague_ai.news.base import ArticleDTO
from euroleague_ai.providers.parsing import parse_datetime

logger = get_logger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"
_DC_NS = "{http://purl.org/dc/elements/1.1/}"

#: Tags that end a line of text. Keeping paragraph breaks matters downstream:
#: injury extraction works sentence by sentence, and a paragraph boundary is the
#: most reliable sentence boundary there is.
_BLOCK_TAGS = frozenset({"p", "br", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"})


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str | None) -> str | None:
    """Readable text from an HTML fragment: tags dropped, entities decoded,
    one line per paragraph."""
    if not html:
        return None
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).split("\n")]
    text = "\n".join(line for line in lines if line)
    return text or None


def _text(element: Any) -> str | None:
    if element is None:
        return None
    text = (element.text or "").strip()
    return text or None


def _parse_rfc2822(raw: str | None) -> Any:
    """Parse an RSS ``pubDate``, which is RFC-2822 rather than ISO-8601."""
    if not raw:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


def parse_feed(xml: str) -> list[ArticleDTO]:
    """Parse RSS 2.0 or Atom into articles.

    Pure function over a string, so it is testable against a committed sample
    without any network access.
    """
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        logger.warning("could not parse feed", extra={"error": str(exc)})
        return []

    articles: list[ArticleDTO] = []

    for item in root.iter("item"):  # RSS 2.0
        link = _text(item.find("link"))
        title = _text(item.find("title"))
        if not link or not title:
            continue
        articles.append(
            ArticleDTO(
                external_id=_text(item.find("guid")),
                url=link,
                title=title,
                summary=html_to_text(_text(item.find("description"))),
                # WordPress feeds carry the whole article here; without it the
                # injury details in the body would never be seen.
                body=html_to_text(_text(item.find(f"{_CONTENT_NS}encoded"))),
                author=_text(item.find("author")) or _text(item.find(f"{_DC_NS}creator")),
                published_at=_parse_rfc2822(_text(item.find("pubDate"))),
            )
        )

    for entry in root.iter(f"{_ATOM_NS}entry"):  # Atom
        link_el = entry.find(f"{_ATOM_NS}link")
        link = link_el.get("href") if link_el is not None else None
        title = _text(entry.find(f"{_ATOM_NS}title"))
        if not link or not title:
            continue
        articles.append(
            ArticleDTO(
                external_id=_text(entry.find(f"{_ATOM_NS}id")),
                url=link,
                title=title,
                summary=_text(entry.find(f"{_ATOM_NS}summary")),
                published_at=parse_datetime(
                    _text(entry.find(f"{_ATOM_NS}updated"))
                    or _text(entry.find(f"{_ATOM_NS}published"))
                ),
            )
        )

    return articles


class RssNewsCollector:
    """Fetches and parses RSS/Atom feeds."""

    collector_key = "rss"

    def __init__(
        self, settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.news_fetch_timeout_seconds),
            headers={"User-Agent": self._settings.euroleague_user_agent},
            follow_redirects=True,
        )

    async def collect(self, feed_url: str) -> list[ArticleDTO]:
        try:
            response = await self._client.get(feed_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # One dead feed must not abort collection from the others.
            logger.warning("feed fetch failed", extra={"url": feed_url, "error": str(exc)})
            return []
        return parse_feed(response.text)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
