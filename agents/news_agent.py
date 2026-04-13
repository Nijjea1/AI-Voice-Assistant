"""
News Agent
===========
Fetches news based on what the user asked for.

THREE MODES:
  1. WORLD NEWS — "What's happening in the world?"
     → RSS feeds (BBC, CNBC, NYT, Al Jazeera) + opens worldmonitor.app
     
  2. DEFAULT NEWS — "Any news?" / "Brief me"
     → Google News RSS for Canada/Ontario
     
  3. SPECIFIC NEWS — "Tech news" / "News about AI" / "USA news"
     → Google News RSS for that specific topic

WHY GOOGLE NEWS RSS?
  DuckDuckGo aggressively rate-limits automated requests (403 errors).
  Google News RSS is free, has no API key, no rate limit, and supports
  any search query. The URL format is simple:
    https://news.google.com/rss/search?q=TOPIC&hl=en-CA&gl=CA
  It returns standard RSS XML that we parse the same way as BBC/NYT feeds.
"""

import json
import webbrowser
import datetime
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from agents.base import BaseAgent, FunctionDef, AgentResult
from config import OLLAMA_URL, RESPONDER_MODEL, GRAY, RESET, CYAN, GREEN, YELLOW


# ─────────────────────────────────────────────
# RSS Feed Sources (world news only)
# ─────────────────────────────────────────────
RSS_FEEDS = [
    ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml", "World"),
    ("CNBC", "https://www.cnbc.com/id/100727362/device/rss/rss.html", "Business"),
    ("NYT", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "World"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", "World"),
]

# Google News RSS base URL — works for any search query
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-CA&gl=CA&ceid=CA:en"

# Default search when user doesn't specify a topic
DEFAULT_SEARCH = "Canada Ontario"

# Cache duration
CACHE_DURATION = datetime.timedelta(minutes=15)

# Keywords that trigger world news mode
WORLD_KEYWORDS = ["world", "global", "international", "worldwide", "around the world"]


class NewsAgent(BaseAgent):
    """Fetches news based on what the user asks for."""

    name = "news"
    description = "Fetches news headlines — world, regional, or topic-specific"

    def __init__(self):
        super().__init__()
        self._cache = {}
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Jarvis-AI/1.0"})

    def get_functions(self) -> List[FunctionDef]:
        return [
            FunctionDef(
                name="get_news",
                description=(
                    "Fetch news headlines. Supports world news, regional news, "
                    "or topic-specific news based on the user's query."
                ),
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The user's original request",
                    }
                },
                required_params=[],
            ),
        ]

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        if func_name == "get_news":
            query = params.get("query", "")
            return self._get_news(query)
        return AgentResult(False, f"Unknown function: {func_name}")

    # ─────────────────────────────────────────
    # Main Logic — Decide Which Mode
    # ─────────────────────────────────────────

    def _get_news(self, query: str) -> AgentResult:
        """
        Decide what kind of news to fetch.
        
        "What's happening in the world?" → world mode (RSS + monitor)
        "Any news?" → default (Canada/Ontario via Google News)
        "Tech news" → search "tech" via Google News
        "News about AI" → search "AI" via Google News
        """
        lower = query.lower()
        is_world = any(word in lower for word in WORLD_KEYWORDS)

        if is_world:
            return self._world_news()
        else:
            search_query = self._extract_topic(lower)
            return self._search_news(search_query)

    def _extract_topic(self, lower_query: str) -> str:
        """
        Figure out what specific news the user wants.
        
        "tech news" → "tech"
        "news about AI" → "AI"  
        "what's happening in Japan" → "Japan"
        "any news?" → "Canada Ontario" (default)
        """
        # Words to strip — they don't tell us WHAT to search for
        strip_words = [
            "news", "headlines", "brief me", "catch me up", "briefing",
            "what's happening", "whats happening", "what is happening",
            "what did i miss", "any", "the", "latest", "recent", "current",
            "tell me about", "tell me", "give me", "get me", "show me",
            "what's going on", "whats going on", "what is going on",
            "in", "about", "for", "on", "with", "regarding",
            "update", "updates", "today", "right now", "please",
        ]

        remaining = lower_query
        for word in strip_words:
            remaining = remaining.replace(word, "")

        # Clean up whitespace and punctuation
        remaining = re.sub(r"[?.!,]", "", remaining)
        remaining = " ".join(remaining.split()).strip()

        # If something specific remains, use it
        # Changed from > 2 to > 1 so "AI" (2 chars) works
        if remaining and len(remaining) > 1:
            return remaining
        else:
            return DEFAULT_SEARCH

    # ─────────────────────────────────────────
    # Mode 1: World News (RSS feeds + world monitor)
    # ─────────────────────────────────────────

    def _world_news(self) -> AgentResult:
        """Fetch world news from RSS feeds and open the world monitor."""
        cache_key = "_world_"

        if self._is_cache_valid(cache_key):
            print(f"{GRAY}[News] Using cached world briefing{RESET}")
            self._open_world_monitor()
            return AgentResult(
                success=True,
                message=self._format_briefing(self._cache[cache_key]["articles"], is_world=True),
                data=self._cache[cache_key]["articles"],
            )

        print(f"{CYAN}[News] Fetching world headlines from {len(RSS_FEEDS)} sources...{RESET}")
        raw_articles = self._fetch_all_rss()

        if not raw_articles:
            return AgentResult(False, "Could not fetch world news right now.")

        print(f"{CYAN}[News] Got {len(raw_articles)} articles, curating...{RESET}")
        curated = self._curate_with_ai(raw_articles)
        if not curated:
            curated = self._format_raw_fallback(raw_articles)

        self._cache[cache_key] = {
            "time": datetime.datetime.now(),
            "articles": curated,
        }

        self._open_world_monitor()

        print(f"{GREEN}[News] ✓ World briefing ready ({len(curated)} stories){RESET}")
        return AgentResult(
            success=True,
            message=self._format_briefing(curated, is_world=True),
            data=curated,
        )

    # ─────────────────────────────────────────
    # Mode 2 & 3: Search News (Google News RSS)
    # ─────────────────────────────────────────

    def _search_news(self, search_topic: str) -> AgentResult:
        """
        Search for news using Google News RSS.
        
        Google News RSS works like a search engine for news:
          https://news.google.com/rss/search?q=AI&hl=en-CA&gl=CA
        
        Returns standard RSS XML we can parse just like BBC/NYT feeds.
        No API key needed, no rate limiting.
        """
        cache_key = search_topic.lower().strip()

        if self._is_cache_valid(cache_key):
            print(f"{GRAY}[News] Using cached results for '{search_topic}'{RESET}")
            return AgentResult(
                success=True,
                message=self._format_briefing(self._cache[cache_key]["articles"], is_world=False),
                data=self._cache[cache_key]["articles"],
            )

        print(f"{CYAN}[News] Searching Google News: '{search_topic}'...{RESET}")

        try:
            # Build the Google News RSS URL with the search query
            # We URL-encode the query by replacing spaces with +
            encoded_query = search_topic.replace(" ", "+")
            url = GOOGLE_NEWS_RSS.format(query=encoded_query)

            response = self._http.get(url, timeout=10)

            if response.status_code != 200:
                return AgentResult(
                    False,
                    f"Google News returned status {response.status_code}.",
                )

            # Parse the RSS XML — same format as any RSS feed
            root = ET.fromstring(response.content)
            articles = []
            seen_titles = set()

            for item in root.findall(".//item")[:10]:
                title = item.findtext("title", "")

                # Google News titles often end with " - Source Name"
                # Split to extract the source
                source = "Google News"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0].strip()
                    source = parts[1].strip()

                # Skip duplicates
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")

                # Get description if available
                description = item.findtext("description", "")
                if description:
                    description = re.sub(r"<[^>]+>", "", description).strip()

                articles.append({
                    "title": title,
                    "source": source,
                    "category": "News",
                    "url": link,
                    "summary": description[:200] if description else "",
                    "date": pub_date,
                })

            # Limit to 6
            articles = articles[:6]

            if not articles:
                return AgentResult(False, f"No news found for '{search_topic}'.")

            # Cache
            self._cache[cache_key] = {
                "time": datetime.datetime.now(),
                "articles": articles,
            }

            print(f"{GREEN}[News] ✓ Found {len(articles)} stories for '{search_topic}'{RESET}")
            return AgentResult(
                success=True,
                message=self._format_briefing(articles, is_world=False),
                data=articles,
            )

        except Exception as e:
            print(f"{GRAY}[News] Search failed: {e}{RESET}")
            return AgentResult(False, f"News search failed: {e}")

    # ─────────────────────────────────────────
    # RSS Fetching (world news only)
    # ─────────────────────────────────────────

    def _fetch_all_rss(self) -> List[Dict]:
        """Fetch all RSS feeds simultaneously."""
        all_articles = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_feed = {
                executor.submit(self._fetch_single_feed, name, url, category): name
                for name, url, category in RSS_FEEDS
            }

            for future in as_completed(future_to_feed):
                feed_name = future_to_feed[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"{GRAY}[News] {feed_name} feed failed: {e}{RESET}")

        return all_articles

    def _fetch_single_feed(self, source_name: str, url: str, category: str) -> List[Dict]:
        """Fetch and parse a single RSS feed."""
        try:
            response = self._http.get(url, timeout=5)
            if response.status_code != 200:
                return []

            root = ET.fromstring(response.content)
            articles = []

            for item in root.findall(".//item")[:5]:
                title = item.findtext("title", "")
                description = item.findtext("description", "")
                link = item.findtext("link", "")

                if description:
                    description = re.sub(r"<[^>]+>", "", description).strip()

                articles.append({
                    "source": source_name,
                    "title": title,
                    "summary": description[:200] if description else "",
                    "category": category,
                    "url": link,
                })

            return articles
        except Exception:
            return []

    # ─────────────────────────────────────────
    # AI Curation (world news only)
    # ─────────────────────────────────────────

    def _curate_with_ai(self, raw_articles: List[Dict]) -> Optional[List[Dict]]:
        """Send raw headlines to Ollama to pick the best 6."""
        news_input = [
            {"id": i, "source": a["source"], "title": a["title"]}
            for i, a in enumerate(raw_articles)
        ]

        prompt = f"""You are a news editor. Here are raw headlines:

{json.dumps(news_input, indent=2)}

Pick the 6 most important and diverse stories. Rewrite each title to be punchy and under 10 words. Return ONLY a JSON array. No markdown, no explanation.

Format: [{{"id": 0, "title": "Short punchy title"}}]"""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/chat",
                json={
                    "model": RESPONDER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=60,
            )

            if response.status_code != 200:
                return None

            content = response.json()["message"]["content"]

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            selected = json.loads(content)

            curated = []
            for item in selected:
                idx = item.get("id", 0)
                if idx < len(raw_articles):
                    original = raw_articles[idx]
                    curated.append({
                        "title": item.get("title", original["title"]),
                        "source": original["source"],
                        "category": original.get("category", "World"),
                        "url": original["url"],
                        "summary": original["summary"],
                    })

            return curated if curated else None

        except Exception as e:
            print(f"{GRAY}[News] AI curation failed: {e}{RESET}")
            return None

    def _format_raw_fallback(self, raw_articles: List[Dict]) -> List[Dict]:
        """Fallback without AI curation."""
        seen = set()
        unique = []
        for a in raw_articles:
            if a["title"] not in seen:
                seen.add(a["title"])
                unique.append(a)
        return unique[:6]

    # ─────────────────────────────────────────
    # Formatting
    # ─────────────────────────────────────────

    def _format_briefing(self, articles: List[Dict], is_world: bool) -> str:
        """Format articles for the LLM to rephrase naturally."""
        if not articles:
            return "No news available right now."

        lines = ["Here are the top stories:\n"]
        for i, article in enumerate(articles, 1):
            source = article.get("source", "Unknown")
            title = article.get("title", "No title")
            lines.append(f"{i}. [{source}] {title}")

        if is_world:
            lines.append("\nYou also opened the world monitor in the browser for a visual overview.")

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # World Monitor (world news only)
    # ─────────────────────────────────────────

    def _open_world_monitor(self):
        """Open worldmonitor.app — ONLY called for world news."""
        try:
            webbrowser.open("https://worldmonitor.app/")
            print(f"{CYAN}[News] Opened world monitor in browser{RESET}")
        except Exception as e:
            print(f"{GRAY}[News] Could not open world monitor: {e}{RESET}")

    # ─────────────────────────────────────────
    # Cache
    # ─────────────────────────────────────────

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self._cache:
            return False
        return datetime.datetime.now() - self._cache[cache_key]["time"] < CACHE_DURATION

    # ─────────────────────────────────────────
    # System Info
    # ─────────────────────────────────────────

    def get_system_info(self) -> Optional[Dict]:
        if not self._cache:
            return None
        latest_key = max(self._cache.keys(), key=lambda k: self._cache[k]["time"])
        articles = self._cache[latest_key]["articles"]
        return {
            "recent_headlines": [
                {"title": a["title"], "source": a["source"]}
                for a in articles[:3]
            ]
        }

    def shutdown(self):
        self._http.close()


    