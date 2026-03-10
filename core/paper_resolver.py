"""Resolve a URL/DOI/PMID to paper metadata + open-access PDF URL."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from .base_client import BaseClient
from .models import Paper

CACHE_DIR = Path("cache/api")

# Patterns for identifying input types
DOI_PATTERN = re.compile(r"(10\.\d{4,}/[^\s]+)")
PMID_PATTERN = re.compile(r"(?:PMID[:\s]*)?(\d{7,8})\b")
DOI_URL_PATTERN = re.compile(r"(?:doi\.org|dx\.doi\.org)/(10\.\d{4,}/[^\s?#]+)")

# Publisher URL → DOI mapping patterns
PUBLISHER_DOI_PATTERNS = [
    # Nature: nature.com/articles/s41586-020-1234-5 → 10.1038/s41586-020-1234-5
    (re.compile(r"nature\.com/articles/(s\d+[\w\-]+)"), "10.1038/{}"),
    # Science: science.org/doi/10.1126/science.xxx
    (re.compile(r"science\.org/doi/(10\.\d{4,}/[^\s?#]+)"), "{}"),
    # Cell/Elsevier: cell.com/*/fulltext/S0092-8674(20)30001-1 → use PII
    # Wiley: onlinelibrary.wiley.com/doi/10.xxxx/xxx
    (re.compile(r"onlinelibrary\.wiley\.com/doi/(10\.\d{4,}/[^\s?#]+)"), "{}"),
    # PLOS: journals.plos.org/plosone/article?id=10.1371/...
    (re.compile(r"journals\.plos\.org/\w+/article\?id=(10\.\d{4,}/[^\s&#]+)"), "{}"),
    # PNAS: pnas.org/doi/10.1073/pnas.xxx
    (re.compile(r"pnas\.org/doi/(10\.\d{4,}/[^\s?#]+)"), "{}"),
    # ScienceDirect / Elsevier PII → resolve via CrossRef
    # sciencedirect.com/science/article/pii/S0006899310003926
    # linkinghub.elsevier.com/retrieve/pii/S0006899310003926
    (re.compile(r"(?:sciencedirect\.com/science/article|linkinghub\.elsevier\.com/retrieve)/pii/(S\d{16,})"), "pii:{}"),
    # BMJ/Lancet/etc via doi in URL path
    (re.compile(r"/(?:doi|article|full)/(10\.\d{4,}/[^\s?#]+)"), "{}"),
]


def _make_paper_id(identifier: str) -> str:
    return hashlib.sha256(identifier.encode()).hexdigest()[:12]


def _doi_from_publisher_url(url: str) -> str | None:
    """Extract DOI from known publisher URL patterns.
    Returns 'pii:SXXX' for ScienceDirect PII (needs async resolution)."""
    for pattern, doi_template in PUBLISHER_DOI_PATTERNS:
        m = pattern.search(url)
        if m:
            captured = m.group(1)
            doi = doi_template.format(captured)
            return doi
    return None


def parse_input(raw: str) -> tuple[str, str]:
    """Parse user input into (type, identifier). Returns ('doi', '10.xxx/yyy') etc."""
    raw = raw.strip()

    # DOI URL
    m = DOI_URL_PATTERN.search(raw)
    if m:
        return "doi", m.group(1)

    # Raw DOI
    m = DOI_PATTERN.match(raw)
    if m:
        return "doi", m.group(1)

    # PMID
    m = PMID_PATTERN.match(raw)
    if m:
        return "pmid", m.group(1)

    # PubMed URL
    pm_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", raw)
    if pm_match:
        return "pmid", pm_match.group(1)

    # PMC URL
    pmc_match = re.search(r"ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)", raw)
    if pmc_match:
        return "pmc", pmc_match.group(1)

    # Publisher URL patterns (Nature, Science, Wiley, PLOS, ScienceDirect, etc.)
    doi = _doi_from_publisher_url(raw)
    if doi:
        if doi.startswith("pii:"):
            return "pii", doi[4:]
        return "doi", doi

    # Generic URL — try to find DOI in the URL
    m = DOI_PATTERN.search(raw)
    if m:
        return "doi", m.group(1)

    # If nothing matches, treat as a URL to scrape for DOI
    if raw.startswith("http"):
        return "url", raw

    raise ValueError(f"Cannot parse input: {raw}")


class PaperResolver:
    def __init__(self):
        self.crossref = BaseClient(
            base_url="https://api.crossref.org",
            rate_interval=0.5,
            cache_dir=CACHE_DIR / "crossref",
            user_agent="GetMeTheData/0.1.0 (mailto:research@imperial.ac.uk)",
        )
        self.unpaywall = BaseClient(
            base_url="https://api.unpaywall.org/v2",
            rate_interval=0.5,
            cache_dir=CACHE_DIR / "unpaywall",
        )
        self.ncbi = BaseClient(
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_interval=0.35,
            cache_dir=CACHE_DIR / "ncbi",
        )

    async def close(self):
        await self.crossref.close()
        await self.unpaywall.close()
        await self.ncbi.close()

    async def resolve(self, raw_input: str) -> Paper:
        input_type, identifier = parse_input(raw_input)
        doi = None

        if input_type == "doi":
            doi = identifier
        elif input_type == "pmid":
            doi = await self._pmid_to_doi(identifier)
        elif input_type == "pmc":
            doi = await self._pmc_to_doi(identifier)
        elif input_type == "pii":
            doi = await self._pii_to_doi(identifier)
        elif input_type == "url":
            doi = await self._url_to_doi(identifier)

        if not doi:
            raise ValueError(f"Could not resolve DOI from input: {raw_input}")

        paper = await self._crossref_metadata(doi)
        paper.source_url = raw_input

        if input_type == "pmid":
            paper.pmid = identifier

        # Try Unpaywall for OA PDF
        pdf_url = await self._unpaywall_pdf(doi)
        if pdf_url:
            paper.pdf_url = pdf_url

        return paper

    async def _crossref_metadata(self, doi: str) -> Paper:
        data = await self.crossref.get(f"works/{doi}")
        msg = data.get("message", {})

        authors = []
        for a in msg.get("author", []):
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)

        title_list = msg.get("title", [])
        title = title_list[0] if title_list else None

        year = None
        date_parts = msg.get("published-print", msg.get("published-online", {})).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        journal_list = msg.get("container-title", [])
        journal = journal_list[0] if journal_list else None

        return Paper(
            paper_id=_make_paper_id(doi),
            doi=doi,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
        )

    async def _unpaywall_pdf(self, doi: str) -> Optional[str]:
        try:
            data = await self.unpaywall.get(
                f"{doi}",
                params={"email": "research@imperial.ac.uk"},
            )
            best = data.get("best_oa_location", {})
            if best:
                return best.get("url_for_pdf") or best.get("url")
        except Exception:
            pass
        return None

    async def _pii_to_doi(self, pii: str) -> Optional[str]:
        """Resolve a ScienceDirect PII to DOI via CrossRef search."""
        try:
            # CrossRef can search by "alternative-id" which includes PII
            data = await self.crossref.get("works", params={
                "filter": f"alternative-id:{pii}",
                "rows": "1",
            })
            items = data.get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI")
        except Exception:
            pass

        # Fallback: scrape the ScienceDirect page for DOI meta tag
        try:
            import httpx
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.get(
                    f"https://www.sciencedirect.com/science/article/pii/{pii}",
                    headers={"User-Agent": "GetMeTheData/0.1.0", "Accept": "text/html"},
                )
                import re as _re
                meta = _re.search(
                    r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\'](10\.\d{4,}/[^"\']+)["\']',
                    resp.text, _re.IGNORECASE,
                )
                if meta:
                    return meta.group(1)
                # Also try dc.identifier
                meta = _re.search(
                    r'<meta[^>]+name=["\']dc\.identifier["\'][^>]+content=["\']doi:(10\.\d{4,}/[^"\']+)["\']',
                    resp.text, _re.IGNORECASE,
                )
                if meta:
                    return meta.group(1)
        except Exception:
            pass
        return None

    async def _pmid_to_doi(self, pmid: str) -> Optional[str]:
        try:
            data = await self.ncbi.get("esummary.fcgi", params={
                "db": "pubmed", "id": pmid, "retmode": "json",
            })
            result = data.get("result", {}).get(pmid, {})
            article_ids = result.get("articleids", [])
            for aid in article_ids:
                if aid.get("idtype") == "doi":
                    return aid["value"]
        except Exception:
            pass
        return None

    async def _pmc_to_doi(self, pmc_id: str) -> Optional[str]:
        try:
            data = await self.ncbi.get("esearch.fcgi", params={
                "db": "pubmed", "term": f"{pmc_id}[pmc]", "retmode": "json",
            })
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if id_list:
                return await self._pmid_to_doi(id_list[0])
        except Exception:
            pass
        return None

    async def _url_to_doi(self, url: str) -> Optional[str]:
        """Try to extract DOI from a publisher page by fetching HTML."""
        try:
            import httpx
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.get(url, headers={
                    "User-Agent": "GetMeTheData/0.1.0",
                    "Accept": "text/html",
                })
                text = resp.text

                # Check meta tags first (most reliable)
                import re as _re
                meta_doi = _re.search(
                    r'<meta[^>]+(?:name|property)=["\'](?:citation_doi|dc\.identifier|DOI)["\'][^>]+content=["\'](10\.\d{4,}/[^"\']+)["\']',
                    text, _re.IGNORECASE,
                )
                if meta_doi:
                    return meta_doi.group(1)

                # Fallback: find DOI anywhere in the page
                m = DOI_PATTERN.search(text)
                if m:
                    return m.group(1)
        except Exception:
            pass
        return None
