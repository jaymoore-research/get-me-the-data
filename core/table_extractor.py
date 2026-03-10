"""Extract tables from PDFs using pdfplumber and optionally refine with Claude."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import anthropic
import pdfplumber

from .models import Confidence, TableData

logger = logging.getLogger(__name__)

# Match "Table 1.", "Table S1:", etc.
TABLE_CAPTION_PATTERN = re.compile(
    r"((?:Table)\s+S?\d+[a-zA-Z]?\s*[.:\-—].*?)(?=(?:Table)\s+S?\d+[a-zA-Z]?\s*[.:\-—]|\Z)",
    re.DOTALL | re.IGNORECASE,
)

TABLE_REF_PATTERN = re.compile(
    r"Table\s+(S?\d+[a-zA-Z]?)",
    re.IGNORECASE,
)


def _extract_table_captions(pdf_path: Path) -> dict[str, str]:
    """Extract table captions keyed by table number (e.g., '1', 'S1')."""
    full_text = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    text = "\n\n".join(full_text)

    captions = {}
    for match in TABLE_CAPTION_PATTERN.finditer(text):
        caption_text = match.group(1).strip()
        caption_text = re.sub(r"\s+", " ", caption_text)
        ref_match = TABLE_REF_PATTERN.match(caption_text)
        if ref_match:
            table_num = ref_match.group(1)
            # Truncate long captions
            captions[table_num] = caption_text[:500]
    return captions


def extract_tables(pdf_path: Path, paper_id: str) -> list[TableData]:
    """Extract all tables from a PDF using pdfplumber."""
    tables = []
    captions = _extract_table_captions(pdf_path)

    with pdfplumber.open(str(pdf_path)) as pdf:
        # Pre-extract all page texts for caption lookups
        all_page_texts = [p.extract_text() or "" for p in pdf.pages]

        table_counter = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            page_text = all_page_texts[page_num - 1]

            for tbl_idx, raw_table in enumerate(page_tables):
                if not raw_table or len(raw_table) < 2:
                    continue

                table_counter += 1

                # Clean cells
                cleaned = []
                for row in raw_table:
                    cleaned.append([
                        (cell or "").strip().replace("\n", " ")
                        for cell in row
                    ])

                # First row is typically headers
                headers = cleaned[0]
                rows = cleaned[1:]

                # Skip tables that are all empty
                has_data = any(
                    any(cell for cell in row)
                    for row in rows
                )
                if not has_data:
                    continue

                # Skip tables with too few columns (likely misdetected text blocks)
                non_empty_headers = [h for h in headers if h.strip()]
                if len(non_empty_headers) < 2:
                    continue

                # Try to match caption — check multiple strategies
                caption = None
                # 1. Find "Table N" references on this page
                refs = re.findall(r"Table\s+(S?\d+[a-zA-Z]?)", page_text, re.IGNORECASE)
                for ref in refs:
                    if ref in captions:
                        caption = captions[ref]
                        break
                # 2. Fall back to sequential table number
                if not caption and str(table_counter) in captions:
                    caption = captions[str(table_counter)]
                # 3. If still no caption, try previous page (caption may span pages)
                if not caption and page_num > 1:
                    prev_refs = re.findall(r"Table\s+(S?\d+[a-zA-Z]?)", all_page_texts[page_num - 2], re.IGNORECASE)
                    for ref in prev_refs:
                        if ref in captions:
                            caption = captions[ref]
                            break

                table_id = f"{paper_id}_table_{table_counter}"
                tables.append(TableData(
                    table_id=table_id,
                    paper_id=paper_id,
                    page_number=page_num,
                    caption=caption,
                    headers=headers,
                    rows=rows,
                    confidence=Confidence.HIGH,
                    source="pdf_text",
                ))

    return tables


async def refine_table_with_claude(table: TableData) -> TableData:
    """Use Claude to clean up and validate an extracted table.

    Fixes OCR errors, merges split cells, identifies numeric columns, etc.
    """
    client = anthropic.AsyncAnthropic()

    # Build a text representation of the table
    table_text = ""
    if table.caption:
        table_text += f"Caption: {table.caption}\n\n"
    table_text += "Headers: " + " | ".join(table.headers) + "\n"
    for row in table.rows:
        table_text += " | ".join(row) + "\n"

    prompt = f"""\
You are a scientific data extraction specialist. Clean up this table extracted from a PDF.

{table_text}

Fix any issues:
1. Merge cells that were incorrectly split across columns
2. Fix OCR artifacts or encoding issues
3. Standardize number formats (remove spaces in numbers, fix decimal points)
4. If a header row was missed, identify it
5. Remove completely empty rows/columns

Return a JSON object:
{{
  "headers": ["col1", "col2", ...],
  "rows": [["val1", "val2", ...], ...],
  "confidence": "high|medium|low",
  "notes": "any issues found or data quality notes"
}}

Return ONLY the JSON object."""

    try:
        response = await client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        table.headers = data.get("headers", table.headers)
        table.rows = data.get("rows", table.rows)
        table.notes = data.get("notes")
        try:
            table.confidence = Confidence(data.get("confidence", "medium"))
        except ValueError:
            pass

    except Exception as e:
        logger.warning(f"Claude refinement failed for {table.table_id}: {e}")
        table.notes = f"Auto-refinement failed: {e}"

    return table


async def refine_tables(
    tables: list[TableData], max_concurrent: int = 3
) -> list[TableData]:
    """Refine multiple tables with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _refine(tbl: TableData) -> TableData:
        async with semaphore:
            return await refine_table_with_claude(tbl)

    results = await asyncio.gather(*[_refine(t) for t in tables])
    return list(results)
