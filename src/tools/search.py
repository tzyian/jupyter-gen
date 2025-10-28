from typing import Any, Dict, List, Optional

import arxiv
from fastmcp import FastMCP
from tavily import TavilyClient

from config import settings

mcp = FastMCP("mcp2-search-tools")


@mcp.tool
def search_arxiv_tool(
    query: str, surveys_only: bool = True, max_results: int = 3
) -> Dict[str, Any]:
    return search_arxiv_papers(query, surveys_only, max_results)


@mcp.tool
def search_tavily_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    return search_tavily(query, max_results)


def search_arxiv_papers(
    query: str, surveys_only: bool = True, max_results: int = 3
) -> Dict[str, Any]:
    """
    Search arXiv for papers matching `query`. If `surveys_only` is True,
    bias toward surveys/overviews. Returns list of metadata.
    """
    query = f"all:{query}"

    if surveys_only:
        query = f"{query} AND (all:survey OR all:overview OR all:review OR all:meta-analysis)"

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client()

        items: List[Dict[str, Any]] = []
        for r in client.results(search):
            items.append(
                {
                    "title": r.title,
                    "url": r.entry_id,
                    "summary": r.summary,
                    "published": r.published.isoformat() if r.published else None,
                    "authors": [a.name for a in r.authors],
                    "primary_category": r.primary_category,
                    "categories": r.categories,
                }
            )

        return {"results": items}
    except Exception as exc:
        return {
            "results": [],
            "error": f"arXiv search failed: {exc}",
        }


def search_tavily(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Use Tavily to search the web with a focus on research. If `surveys_only` is True,
    bias the query toward surveys/meta-analyses.
    """

    if not settings.tavily_api_key:
        return {
            "results": [],
            "skipped": True,
            "message": "TAVILY_API_KEY not configured",
        }
    try:
        api_key = settings.tavily_api_key.get_secret_value()

        client = TavilyClient(api_key=api_key)
        result = client.search(query=query, max_results=max_results)

        items: List[Dict[str, Optional[str]]] = []
        for r in result.get("results", []):
            items.append(
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                }
            )

        return {"results": items}
    except Exception as exc:
        return {
            "results": [],
            "error": f"Tavily search failed: {exc}",
        }


if __name__ == "__main__":
    mcp.run()
