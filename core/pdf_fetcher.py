"""Download open-access PDFs to cache."""

from __future__ import annotations

from pathlib import Path

import httpx

CACHE_DIR = Path("cache/pdfs")


async def fetch_pdf(pdf_url: str, paper_id: str) -> Path:
    """Download PDF and return local path. Returns cached version if exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = CACHE_DIR / f"{paper_id}.pdf"

    if pdf_path.exists():
        return pdf_path

    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(pdf_url, headers={
            "User-Agent": "GetMeTheData/0.1.0",
            "Accept": "application/pdf",
        })
        resp.raise_for_status()

        pdf_path.write_bytes(resp.content)

    return pdf_path


def save_uploaded_pdf(content: bytes, paper_id: str) -> Path:
    """Save an uploaded PDF file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = CACHE_DIR / f"{paper_id}.pdf"
    pdf_path.write_bytes(content)
    return pdf_path
