# Semantic Scholar API, Link Scoring & PDF Downloader
import os
import requests

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

PAYWALLED_DOMAINS = [
    "ieeexplore.ieee.org",
    "link.springer.com",
    "sciencedirect.com",
    "wiley.com",
    "dl.acm.org",
]


def score_pdf_url(url: str) -> int:
    """Ranks PDF sources so reliable open-access links (e.g., ArXiv) are attempted first."""
    if not url:
        return 0
    url_lower = url.lower()
    if "arxiv.org" in url_lower:
        return 100  # Highest priority
    if any(domain in url_lower for domain in PAYWALLED_DOMAINS):
        return 10   # Lowest priority (paywalled/anti-bot)
    return 50       # Standard open-access repository


def download_pdf_bytes(pdf_url: str) -> bytes:
    """Downloads PDF bytes using realistic browser headers and verifies PDF format."""
    response = requests.get(pdf_url, headers=DOWNLOAD_HEADERS, timeout=25, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} error fetching PDF from {pdf_url}")

    content_type = response.headers.get("Content-Type", "").lower()
    content_peek = response.content[:10]

    # Verify that response is a genuine PDF
    if "application/pdf" not in content_type and not content_peek.startswith(b"%PDF"):
        raise RuntimeError(f"URL did not return a valid PDF (Content-Type: {content_type})")

    return response.content


def search_semantic_scholar(query: str, limit: int = 10) -> list:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    response = requests.get(
        url,
        headers=headers,
        params={
            "query": query,
            "limit": limit,
            "fields": "title,year,citationCount,externalIds,url,openAccessPdf",
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Semantic Scholar API error {response.status_code}: {response.text}")

    return response.json().get("data", [])


def get_papers_from_semantic_scholar(search_query: str, limit: int = 10) -> list:
    results = search_semantic_scholar(search_query, limit=limit)
    if not results:
        return []

    print("\n[SEMANTIC SCHOLAR CANDIDATES]")
    papers = []
    seen_titles = set()

    for result in results:
        title = result.get("title", "")
        year = result.get("year")
        citations = result.get("citationCount", 0)
        arxiv_id = result.get("externalIds", {}).get("ArXiv")

        pdf_url = None
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif result.get("openAccessPdf"):
            pdf_url = result["openAccessPdf"].get("url")

        if not pdf_url:
            continue

        normalized_title = title.lower().strip()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)

        papers.append({
            "title": title,
            "year": year,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id,
            "citations": citations,
            "score": score_pdf_url(pdf_url),
        })

    # Prioritize Open Access score first, then citation count
    papers.sort(key=lambda p: (p["score"], p["citations"]), reverse=True)

    for idx, p in enumerate(papers, start=1):
        print(f"{idx}. score={p['score']} | citations={p['citations']} | {p['title']} ({p['year']}) | pdf={p['pdf_url']}")

    return papers