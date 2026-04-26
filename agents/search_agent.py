"""
Search Agent
=============
Searches the web using Google News RSS and DuckDuckGo.

WHY TWO SOURCES?
  Google News RSS is great for news-related queries but doesn't cover
  general knowledge searches like "How tall is the Eiffel Tower?" or
  "Python list comprehension tutorial."
  
  For general searches, we use DuckDuckGo's text search which covers
  everything. If DDG rate-limits us, we fall back to Google News RSS.

HOW IT WORKS:
  1. User asks "Search for Python tutorials"
  2. Agent extracts the search query ("Python tutorials")
  3. Searches DuckDuckGo for text results
  4. Returns titles + snippets for the LLM to summarize
  5. The Dispatcher asks Ollama to phrase the results naturally
"""

import re
from typing import List, Dict, Any, Optional

import requests
import xml.etree.ElementTree as ET

from agents.base import BaseAgent, FunctionDef, AgentResult
from config import GRAY, RESET, CYAN, GREEN, YELLOW


# Google search RSS (fallback if DDG fails)
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-CA&gl=CA&ceid=CA:en"


class SearchAgent(BaseAgent):
    """Searches the web for information."""

    name = "search"
    description = "Searches the web for information on any topic"

    def __init__(self):
        super().__init__()
        self._http = requests.Session()
        self._http.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Jarvis-AI/1.0"
        })

    def get_functions(self) -> List[FunctionDef]:
        return [
            FunctionDef(
                name="web_search",
                description=(
                    "Search the web for information. Use when the user says: "
                    "'Search for Python tutorials', 'Look up the Eiffel Tower', "
                    "'Google how to make pasta', 'Find information about black holes', "
                    "'What is quantum computing?', 'Search the web for...'"
                ),
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                required_params=["query"],
            ),
        ]

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        if func_name == "web_search":
            query = params.get("query", "")
            return self._search(query)
        return AgentResult(False, f"Unknown function: {func_name}")

    # ─────────────────────────────────────────
    # Main Search
    # ─────────────────────────────────────────

    def _search(self, query: str) -> AgentResult:
        """
        Search the web. Tries DuckDuckGo first, falls back to Google News RSS.
        """
        # Clean the query — remove search prefixes
        clean_query = self._clean_query(query)

        if not clean_query:
            return AgentResult(False, "No search query provided.")

        print(f"{CYAN}[Search] Searching: '{clean_query}'...{RESET}")

        # Try DuckDuckGo first
        result = self._search_ddg(clean_query)

        # If DDG fails (rate limit, import error), fall back to Google News RSS
        if result is None:
            print(f"{GRAY}[Search] DDG failed, falling back to Google News RSS{RESET}")
            result = self._search_google_rss(clean_query)

        if result is None:
            return AgentResult(False, f"Could not find results for '{clean_query}'.")

        return result

    def _search_ddg(self, query: str) -> Optional[AgentResult]:
        """
        Search using DuckDuckGo text search.
        Returns None if it fails (so we can fall back).
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return None

            # Format results for the LLM
            lines = [f"Search results for '{query}':\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "")[:150]
                source = r.get("href", "")
                lines.append(f"{i}. {title}")
                if body:
                    lines.append(f"   {body}")

            print(f"{GREEN}[Search] ✓ Found {len(results)} results via DuckDuckGo{RESET}")

            return AgentResult(
                success=True,
                message="\n".join(lines),
                data=results,
            )

        except ImportError:
            print(f"{YELLOW}[Search] duckduckgo-search not installed{RESET}")
            return None
        except Exception as e:
            print(f"{GRAY}[Search] DDG error: {e}{RESET}")
            return None

    def _search_google_rss(self, query: str) -> Optional[AgentResult]:
        """
        Fallback search using Google News RSS.
        Not as comprehensive as DDG for general queries, but reliable.
        """
        try:
            encoded = query.replace(" ", "+")
            url = GOOGLE_NEWS_RSS.format(query=encoded)
            response = self._http.get(url, timeout=10)

            if response.status_code != 200:
                return None

            root = ET.fromstring(response.content)
            articles = []

            for item in root.findall(".//item")[:5]:
                title = item.findtext("title", "")
                # Google News titles end with " - Source"
                source = "Google News"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0].strip()
                    source = parts[1].strip()

                description = item.findtext("description", "")
                if description:
                    description = re.sub(r"<[^>]+>", "", description).strip()

                articles.append({
                    "title": title,
                    "source": source,
                    "summary": description[:150] if description else "",
                })

            if not articles:
                return None

            lines = [f"Search results for '{query}':\n"]
            for i, a in enumerate(articles, 1):
                lines.append(f"{i}. [{a['source']}] {a['title']}")
                if a["summary"]:
                    lines.append(f"   {a['summary']}")

            print(f"{GREEN}[Search] ✓ Found {len(articles)} results via Google News{RESET}")

            return AgentResult(
                success=True,
                message="\n".join(lines),
                data=articles,
            )

        except Exception as e:
            print(f"{GRAY}[Search] Google RSS error: {e}{RESET}")
            return None

    # ─────────────────────────────────────────
    # Query Cleaning
    # ─────────────────────────────────────────

    def _clean_query(self, text: str) -> str:
        """
        Extract the actual search query from natural language.
        
        "Search for Python tutorials" → "Python tutorials"
        "Google how to make pasta" → "how to make pasta"
        "Look up the Eiffel Tower" → "the Eiffel Tower"
        """
        lower = text.lower()

        prefixes = [
            "search for", "search the web for", "search",
            "look up", "lookup", "google", "find information about",
            "find info about", "find info on", "find",
            "tell me about", "what is", "what are", "who is",
            "how to", "how do i",
        ]

        for prefix in prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        return text.strip()

    def shutdown(self):
        self._http.close()