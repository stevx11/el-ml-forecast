"""News collection.

Article collection is implemented behind a protocol. Structured event extraction
is planned -- see docs/ml-roadmap.md, phase 5.
"""

from euroleague_ai.news.base import ArticleDTO, NewsCollector
from euroleague_ai.news.rss import RssNewsCollector

__all__ = ["ArticleDTO", "NewsCollector", "RssNewsCollector"]
