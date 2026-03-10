"""Detect supplementary data links and data repositories."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .models import SupplementaryItem
from .text_extractor import extract_full_text

# Repository URL patterns
REPO_PATTERNS = [
    (r"github\.com/[\w\-]+/[\w\-]+", "github"),
    (r"zenodo\.org/record(?:s)?/\d+", "zenodo"),
    (r"doi\.org/10\.5281/zenodo\.\d+", "zenodo"),
    (r"figshare\.com/(?:articles|s|ndownloader)/\S+", "figshare"),
    (r"doi\.org/10\.6084/m9\.figshare\.\d+", "figshare"),
    (r"datadryad\.org/stash/dataset/\S+", "dryad"),
    (r"doi\.org/10\.5061/dryad\.\w+", "dryad"),
    (r"ncbi\.nlm\.nih\.gov/geo/query/acc\.cgi\?acc=GSE\d+", "geo"),
    (r"ebi\.ac\.uk/arrayexpress/experiments/E-\w+-\d+", "arrayexpress"),
    (r"proteomecentral\.proteomexchange\.org/\S+", "proteomexchange"),
    (r"ebi\.ac\.uk/pride/archive/projects/PXD\d+", "pride"),
    (r"synapse\.org/#!Synapse:syn\d+", "synapse"),
    (r"osf\.io/\w{5}", "osf"),
    (r"ebi\.ac\.uk/ena/browser/view/PRJ\w+", "ena"),
    (r"ncbi\.nlm\.nih\.gov/bioproject/PRJ\w+", "sra"),
]

# Accession number patterns
ACCESSION_PATTERNS = [
    (r"\b(GSE\d{4,})\b", "geo", "GEO"),
    (r"\b(GPL\d{4,})\b", "geo", "GEO Platform"),
    (r"\b(PRJNA\d{4,})\b", "sra", "BioProject"),
    (r"\b(PRJEB\d{4,})\b", "ena", "ENA Project"),
    (r"\b(SRP\d{4,})\b", "sra", "SRA"),
    (r"\b(PXD\d{4,})\b", "proteomexchange", "ProteomeXchange"),
    (r"\b(E-MTAB-\d+)\b", "arrayexpress", "ArrayExpress"),
    (r"\b(syn\d{6,})\b", "synapse", "Synapse"),
    (r"\b(EGAD\d{8,})\b", "ega", "EGA Dataset"),
]

# Supplement link keywords
SUPP_KEYWORDS = [
    "supplement", "supporting info", "additional file", "appendix",
    "extended data", "source data", "supplemental", "supp table",
    "supp figure", "data availability", "code availability",
]


async def find_supplementary(doi: str | None, pdf_path: Path | None) -> list[SupplementaryItem]:
    """Find supplementary data from publisher page and PDF text."""
    items: list[SupplementaryItem] = []

    # Scan PDF text for repository URLs and accession numbers
    if pdf_path and pdf_path.exists():
        items.extend(_scan_pdf_text(pdf_path))

    # Fetch publisher HTML and parse supplement links
    if doi:
        items.extend(await _scan_publisher_page(doi))
        items.extend(await _scan_pmc(doi))

    # Deduplicate by URL (or label if no URL)
    seen: set[str] = set()
    unique: list[SupplementaryItem] = []
    for item in items:
        key = item.url or item.label
        # Normalize URL for dedup
        key = key.rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def _scan_pdf_text(pdf_path: Path) -> list[SupplementaryItem]:
    """Scan PDF full text for repository URLs and accession numbers."""
    text = extract_full_text(pdf_path)
    items = []

    for pattern, source in REPO_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            url = match.group(0)
            if not url.startswith("http"):
                url = f"https://{url}"
            items.append(SupplementaryItem(
                label=f"Data repository ({source})",
                item_type="repository",
                source=source,
                url=url,
            ))

    for pattern, source, label_prefix in ACCESSION_PATTERNS:
        for match in re.finditer(pattern, text):
            accession = match.group(1)
            items.append(SupplementaryItem(
                label=f"{label_prefix}: {accession}",
                item_type="accession",
                source=source,
                url=_accession_url(accession, source),
            ))

    # Look for URLs near "data availability" or "code availability" sections
    da_match = re.search(
        r"(?:data|code)\s+availability[:\s]*(.*?)(?:\n\n|\.\s+[A-Z])",
        text, re.IGNORECASE | re.DOTALL,
    )
    if da_match:
        section = da_match.group(1)
        urls = re.findall(r"https?://\S+", section)
        for url in urls:
            url = url.rstrip(".,;)")
            if not any(url.lower() in (i.url or "").lower() for i in items):
                items.append(SupplementaryItem(
                    label="Data availability link",
                    item_type="data_link",
                    source="paper_text",
                    url=url,
                ))

    return items


def _accession_url(accession: str, source: str) -> str:
    urls = {
        "geo": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        "sra": f"https://www.ncbi.nlm.nih.gov/bioproject/{accession}",
        "ena": f"https://www.ebi.ac.uk/ena/browser/view/{accession}",
        "proteomexchange": f"http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID={accession}",
        "arrayexpress": f"https://www.ebi.ac.uk/arrayexpress/experiments/{accession}",
        "synapse": f"https://www.synapse.org/#!Synapse:{accession}",
        "ega": f"https://ega-archive.org/datasets/{accession}",
    }
    return urls.get(source, "")


async def _scan_publisher_page(doi: str) -> list[SupplementaryItem]:
    """Fetch the DOI landing page and look for supplement links."""
    items = []
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(
                f"https://doi.org/{doi}",
                headers={"User-Agent": "GetMeTheData/0.1.0", "Accept": "text/html"},
            )
            if resp.status_code != 200:
                return items

            soup = BeautifulSoup(resp.text, "html.parser")

            # Look for supplementary material links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True).lower()

                # Skip fragment-only, reference, and self-anchors
                if href.startswith("#"):
                    continue
                if "#ref-" in href or "#CR" in href or "#Fn" in href:
                    continue

                # Supplement links
                if any(kw in text for kw in SUPP_KEYWORDS):
                    full_url = href if href.startswith("http") else f"https://{resp.url.host}{href}"
                    items.append(SupplementaryItem(
                        label=link.get_text(strip=True)[:100],
                        item_type="supplementary_file",
                        source="publisher",
                        url=full_url,
                    ))

                # Repository links
                for pattern, source in REPO_PATTERNS:
                    if re.search(pattern, href, re.IGNORECASE):
                        full_url = href if href.startswith("http") else f"https://{href}"
                        items.append(SupplementaryItem(
                            label=f"Data ({source})",
                            item_type="repository",
                            source=source,
                            url=full_url,
                        ))
                        break

            # Data availability section in HTML
            for heading in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
                heading_text = heading.get_text(strip=True).lower()
                if any(kw in heading_text for kw in ["data availability", "code availability", "data and code"]):
                    # Gather text + links from sibling elements
                    section_el = heading.find_next_sibling()
                    if not section_el:
                        section_el = heading.parent
                    if section_el:
                        for a in section_el.find_all("a", href=True):
                            url = a["href"]
                            # Skip fragment-only links (#ref-CR22, #MOESM4, etc.)
                            if url.startswith("#"):
                                continue
                            # Skip same-article anchor links (references, footnotes, supplements)
                            if re.search(r"#(ref-|CR|Fn|Sec|MOESM)", url):
                                continue
                            # Skip links back to the same article with anchors
                            if doi and doi in url and "#" in url:
                                continue
                            # Skip numeric-only link text (superscript references like "7", "22")
                            link_text = a.get_text(strip=True)
                            if re.fullmatch(r"\d{1,3}", link_text):
                                continue
                            if not url.startswith("http"):
                                url = f"https://{resp.url.host}{url}"
                            items.append(SupplementaryItem(
                                label=a.get_text(strip=True)[:100] or "Data availability link",
                                item_type="data_link",
                                source="publisher",
                                url=url,
                            ))

    except Exception:
        pass

    return items


async def _scan_pmc(doi: str) -> list[SupplementaryItem]:
    """Check PubMed Central for supplementary files."""
    items = []
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            # Try to find PMC ID from DOI
            resp = await client.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={"ids": doi, "format": "json", "tool": "getmethedata"},
            )
            if resp.status_code != 200:
                return items

            data = resp.json()
            records = data.get("records", [])
            if not records or "pmcid" not in records[0]:
                return items

            pmcid = records[0]["pmcid"]

            # Fetch PMC page for supplementary materials
            pmc_resp = await client.get(
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
                headers={"User-Agent": "GetMeTheData/0.1.0", "Accept": "text/html"},
            )
            if pmc_resp.status_code != 200:
                return items

            soup = BeautifulSoup(pmc_resp.text, "html.parser")

            # Find supplementary material section
            supp_sections = soup.find_all(["div", "section"], class_=re.compile(r"supp|supplementary", re.I))
            if not supp_sections:
                supp_sections = []
                for h in soup.find_all(["h2", "h3"]):
                    if "supplement" in h.get_text(strip=True).lower():
                        supp_sections.append(h.parent)

            for section in supp_sections:
                for link in section.find_all("a", href=True):
                    href = link["href"]
                    if not href.startswith("http"):
                        href = f"https://www.ncbi.nlm.nih.gov{href}"
                    text = link.get_text(strip=True)
                    if text:
                        items.append(SupplementaryItem(
                            label=text[:100],
                            item_type="supplementary_file",
                            source="pmc",
                            url=href,
                        ))

            # Add PMC link itself
            items.append(SupplementaryItem(
                label=f"PMC Full Text ({pmcid})",
                item_type="full_text",
                source="pmc",
                url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
            ))

    except Exception:
        pass

    return items
