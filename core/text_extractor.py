"""Extract text and figure legends from PDFs using pdfplumber."""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

# Match "Figure 1.", "Fig. 2:", "Figure S1.", "Fig. 2 |", "Extended Data Fig. 3 |", etc.
LEGEND_PATTERN = re.compile(
    r"((?:Extended\s+Data\s+)?(?:Figure|Fig\.?)\s+S?\d+[a-zA-Z]?\s*[.:\-—|].*?)(?=(?:Extended\s+Data\s+)?(?:Figure|Fig\.?)\s+S?\d+[a-zA-Z]?\s*[.:\-—|]|\Z)",
    re.DOTALL | re.IGNORECASE,
)

FIGURE_REF_PATTERN = re.compile(
    r"((?:Extended\s+Data\s+)?(?:Supplementary\s+)?)?(?:Figure|Fig\.?)\s+(S?\d+[a-zA-Z]?)",
    re.IGNORECASE,
)

# For extracting a short title from a legend sentence
# e.g., "Fig. 2 | Benchmarking DeepRVAT for gene discovery." → "Benchmarking DeepRVAT for gene discovery"
TITLE_SPLIT = re.compile(
    r"(?:Extended\s+Data\s+)?(?:Figure|Fig\.?)\s+S?\d+[a-zA-Z]?\s*[.:\-—|]\s*",
    re.IGNORECASE,
)


def extract_full_text(pdf_path: Path) -> str:
    """Extract all text from PDF."""
    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def extract_legends(pdf_path: Path) -> dict[str, str]:
    """Extract figure legends keyed by figure number.

    Keys: '1', '2', 'S1', 'ED1' (Extended Data), 'Supp1' (Supplementary).
    """
    full_text = extract_full_text(pdf_path)
    legends = {}

    for match in LEGEND_PATTERN.finditer(full_text):
        legend_text = match.group(1).strip()
        # Clean up whitespace
        legend_text = re.sub(r"\s+", " ", legend_text)

        ref_match = FIGURE_REF_PATTERN.match(legend_text)
        if ref_match:
            prefix = (ref_match.group(1) or "").strip().lower()
            fig_num = ref_match.group(2)
            # Build a unique key
            if "extended" in prefix:
                key = "ED" + fig_num
            elif "supplementary" in prefix:
                key = "Supp" + fig_num
            else:
                key = fig_num
            legends[key] = legend_text

    return legends


def legend_short_title(legend: str) -> str:
    """Extract a short, readable title from a figure legend.

    "Fig. 2 | Benchmarking DeepRVAT for gene discovery. We applied..."
    → "Fig. 2 — Benchmarking DeepRVAT for gene discovery"

    "Figure 1. Expression of 20S proteasome alpha-subunits in the substantia nigra..."
    → "Figure 1 — Expression of 20S proteasome alpha-subunits"
    """
    if not legend:
        return ""

    # Get the figure number prefix
    ref_match = FIGURE_REF_PATTERN.match(legend)
    if not ref_match:
        # No standard prefix — return first sentence truncated
        first_sentence = re.split(r"[.!?]\s", legend, maxsplit=1)[0]
        return first_sentence[:80]

    # Extract text after "Fig. N |" / "Figure N." prefix
    remainder = TITLE_SPLIT.sub("", legend, count=1).strip()

    # Take first sentence — split on ". " but also on common noise markers
    # like "a," "b," (sub-panel descriptions) which indicate title has ended
    first_sentence = re.split(r"\.\s|(?<=[a-z])\s[a-g],\s", remainder, maxsplit=1)[0].rstrip(".")

    # Truncate if very long
    if len(first_sentence) > 80:
        first_sentence = first_sentence[:77] + "..."

    # Build readable prefix from matched groups
    prefix_text = (ref_match.group(1) or "").strip()
    fig_num = ref_match.group(2)
    if prefix_text:
        fig_prefix = prefix_text.title() + " Fig. " + fig_num
    else:
        fig_prefix = "Fig. " + fig_num

    return fig_prefix + " — " + first_sentence


def extract_text_by_page(pdf_path: Path) -> list[str]:
    """Extract text per page."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def find_figure_mentions(full_text: str, fig_num: str) -> list[str]:
    """Find sentences in the paper body that reference a specific figure.

    Args:
        full_text: Full paper text.
        fig_num: Figure number key like '1', '2', 'S1', 'ED1'.

    Returns:
        List of unique sentences mentioning this figure (max 10).
    """
    if not full_text or not fig_num:
        return []

    # Build pattern to match "Fig. 1", "Figure 1", "Fig 1a", etc.
    # Handle extended data / supplementary prefixes
    prefix = ""
    num = fig_num
    if fig_num.startswith("ED"):
        prefix = r"(?:Extended\s+Data\s+)"
        num = fig_num[2:]
    elif fig_num.startswith("Supp"):
        prefix = r"(?:Supplementary\s+)"
        num = fig_num[4:]

    pattern = re.compile(
        prefix + r"(?:Figure|Fig\.?)\s+" + re.escape(num) + r"[a-z]?\b",
        re.IGNORECASE,
    )

    # Split text into sentences (rough but effective)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", full_text)

    mentions = []
    seen = set()
    for sent in sentences:
        if pattern.search(sent):
            # Clean up and deduplicate
            clean = re.sub(r"\s+", " ", sent).strip()
            if len(clean) < 20 or len(clean) > 500:
                continue
            # Skip the legend itself (starts with "Figure N." pattern)
            if re.match(r"^(?:Extended\s+Data\s+)?(?:Figure|Fig\.?)\s+\S+\s*[.:\-—|]", clean):
                continue
            key = clean[:80].lower()
            if key not in seen:
                seen.add(key)
                mentions.append(clean)
            if len(mentions) >= 10:
                break

    return mentions
