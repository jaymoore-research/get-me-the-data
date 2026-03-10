"""Extract figures from PDF using PyMuPDF, filter decorations, classify with Claude."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from .models import Confidence, Figure, PlotType
from .text_extractor import extract_legends, extract_text_by_page, legend_short_title

logger = logging.getLogger(__name__)

# Minimum dimensions to filter out icons/logos
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_AREA = 20000
MAX_ASPECT_RATIO = 8.0


def _merge_nearby_rects(
    rects: list[tuple[int, fitz.Rect]], page_rect: fitz.Rect, gap: float = 20.0
) -> list[tuple[list[int], fitz.Rect]]:
    """Merge image rects that are close together into composite figure regions.

    Returns list of (xref_indices, merged_rect) tuples.
    ``gap`` is the max distance (in points) between rects to merge them.
    """
    if not rects:
        return []

    # Sort by vertical then horizontal position
    entries = sorted(rects, key=lambda r: (r[1].y0, r[1].x0))
    groups: list[tuple[list[int], fitz.Rect]] = [
        ([entries[0][0]], fitz.Rect(entries[0][1]))
    ]

    for idx, rect in entries[1:]:
        merged = False
        for group in groups:
            # Check if this rect is close to the group's bounding rect
            grect = group[1]
            expanded = fitz.Rect(
                grect.x0 - gap, grect.y0 - gap, grect.x1 + gap, grect.y1 + gap
            )
            if expanded.intersects(rect):
                group[0].append(idx)
                group[1] |= rect  # union
                merged = True
                break
        if not merged:
            groups.append(([idx], fitz.Rect(rect)))

    return groups


# Padding (in points) around detected image region to capture axes/labels
# 50pt ≈ 0.7 inches — enough for tick labels, axis titles, and legends
REGION_PADDING = 50.0


def _extract_raw_images(pdf_path: Path, dpi: int = 250) -> list[dict]:
    """Extract figures by rendering page regions around embedded images.

    Instead of extracting raw bitmaps (which miss vector axes/labels),
    this finds where images are placed on the page, groups nearby images
    into composite figures, then renders those page regions at high DPI.
    """
    images = []
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        # Collect placement rects for each embedded image
        img_rects: list[tuple[int, fitz.Rect]] = []
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                # Use the first (usually only) placement rect
                rect = rects[0]
                # Filter tiny images (icons, logos)
                w_pts, h_pts = rect.width, rect.height
                if w_pts < 50 or h_pts < 50:
                    continue
                aspect = max(w_pts, h_pts) / max(min(w_pts, h_pts), 1)
                if aspect > MAX_ASPECT_RATIO:
                    continue
                img_rects.append((img_index, rect))
            except Exception:
                continue

        if not img_rects:
            continue

        # Merge nearby images into composite figures
        groups = _merge_nearby_rects(img_rects, page.rect)

        for group_idx, (indices, region) in enumerate(groups):
            # Add padding to capture axes, labels, tick marks
            clip = fitz.Rect(
                max(page.rect.x0, region.x0 - REGION_PADDING),
                max(page.rect.y0, region.y0 - REGION_PADDING),
                min(page.rect.x1, region.x1 + REGION_PADDING),
                min(page.rect.y1, region.y1 + REGION_PADDING),
            )

            # Render just this region at high DPI
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_data = pix.tobytes("png")

            img = Image.open(io.BytesIO(img_data))

            # Filter by pixel dimensions after rendering
            if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
                continue
            if img.width * img.height < MIN_AREA:
                continue

            # Resize if large (keep under ~1MB for Claude vision)
            if img.width > 1500 or img.height > 1500:
                img.thumbnail((1500, 1500), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            img_data = buf.getvalue()

            images.append({
                "page_number": page_num + 1,
                "image_index": indices[0] if len(indices) == 1 else group_idx,
                "width": img.width,
                "height": img.height,
                "image_bytes": img_data,
            })

    doc.close()
    return images


def _render_pages_as_images(pdf_path: Path, dpi: int = 200) -> list[dict]:
    """Fallback: render each PDF page as an image (for vector-based figures)."""
    images = []
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        img_data = pix.tobytes("png")

        # Resize for Claude
        img = Image.open(io.BytesIO(img_data))
        if img.width > 1500 or img.height > 1500:
            img.thumbnail((1500, 1500), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        img_data = buf.getvalue()

        images.append({
            "page_number": page_num + 1,
            "image_index": 0,
            "width": img.width,
            "height": img.height,
            "image_bytes": img_data,
        })

    doc.close()
    return images


def _match_legend_to_image(
    page_num: int, img_index: int, legends: dict[str, str], page_texts: list[str]
) -> str | None:
    """Try to match a legend to an image by page proximity."""
    if not legends:
        return None

    page_text = page_texts[page_num - 1] if page_num <= len(page_texts) else ""

    # Find figure references on this page, including Extended Data prefix
    has_extended = bool(re.search(r"extended\s+data", page_text, re.IGNORECASE))

    refs = re.findall(r"(?:Figure|Fig\.?)\s+(S?\d+[a-zA-Z]?)", page_text, re.IGNORECASE)

    for ref in refs:
        # Try with Extended Data prefix first if page mentions it
        if has_extended and ("ED" + ref) in legends:
            return legends["ED" + ref]
        if ref in legends:
            return legends[ref]

    return None


async def extract_figures(
    pdf_path: Path, paper_id: str, classify: bool = True
) -> list[Figure]:
    """Extract figures from PDF, filter, match legends, optionally classify.

    First tries embedded image extraction. If few/no images found,
    falls back to rendering pages (for vector-based figures).
    """
    raw_images = _extract_raw_images(pdf_path)

    # Fallback: if <2 embedded images, render pages as images
    if len(raw_images) < 2:
        logger.info(f"Only {len(raw_images)} embedded images found, rendering pages")
        page_images = _render_pages_as_images(pdf_path)
        # Merge — prefer embedded images, add rendered pages
        raw_images.extend(page_images)

    legends = extract_legends(pdf_path)
    page_texts = extract_text_by_page(pdf_path)

    figures = []
    for img in raw_images:
        b64 = base64.b64encode(img["image_bytes"]).decode("utf-8")
        legend = _match_legend_to_image(
            img["page_number"], img["image_index"], legends, page_texts
        )
        title = legend_short_title(legend) if legend else None

        fig = Figure(
            figure_id=f"{paper_id}_p{img['page_number']}_i{img['image_index']}",
            paper_id=paper_id,
            page_number=img["page_number"],
            image_index=img["image_index"],
            width=img["width"],
            height=img["height"],
            image_base64=b64,
            title=title,
            legend=legend,
            plot_type=PlotType.OTHER,
            plot_type_confidence=Confidence.LOW,
        )
        figures.append(fig)

    if classify and figures:
        figures = await _classify_figures(figures)

    return figures


async def _classify_figures(figures: list[Figure], max_concurrent: int = 5) -> list[Figure]:
    """Use Claude vision to classify each figure's plot type (with concurrency)."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _classify_one(fig: Figure) -> None:
        async with semaphore:
            try:
                response = await client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": fig.image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Classify this image from a scientific paper. "
                                    "Respond with ONLY one word from this list:\n"
                                    "scatter, bar, line, box, violin, histogram, heatmap, "
                                    "forest, kaplan_meier, dot_strip, stacked_bar, funnel, "
                                    "roc, volcano, waterfall, bland_altman, paired, "
                                    "bubble, area, dose_response, manhattan, "
                                    "correlation_matrix, error_bar, "
                                    "table, other, non_data\n\n"
                                    "'dot_strip' = individual data points per group (no bars, just dots).\n"
                                    "'stacked_bar' = bar chart with stacked colored segments.\n"
                                    "'funnel' = meta-analysis funnel plot (effect size vs SE).\n"
                                    "'roc' = receiver operating characteristic curve.\n"
                                    "'volcano' = log fold-change vs -log10 p-value.\n"
                                    "'waterfall' = ordered bars showing response per sample.\n"
                                    "'bland_altman' = difference vs mean (method comparison).\n"
                                    "'paired' = before-after points connected by lines.\n"
                                    "'bubble' = scatter plot where point SIZE encodes a third variable.\n"
                                    "'area' = filled area chart or stacked area chart.\n"
                                    "'dose_response' = sigmoidal/log-linear dose-response curve.\n"
                                    "'manhattan' = genomic positions vs -log10(p), by chromosome.\n"
                                    "'correlation_matrix' = pairwise correlation matrix / correlogram.\n"
                                    "'error_bar' = means with error bars only (no bars underneath).\n"
                                    "'table' = a data table with rows and columns of numbers/text.\n"
                                    "'non_data' = diagram, photo, schematic, flowchart, "
                                    "or any image without quantitative data.\n"
                                    "'other' = a quantitative plot that doesn't fit the above types."
                                ),
                            },
                        ],
                    }],
                )
                label = response.content[0].text.strip().lower().replace(" ", "_")
                try:
                    fig.plot_type = PlotType(label)
                    fig.plot_type_confidence = Confidence.HIGH
                except ValueError:
                    fig.plot_type = PlotType.OTHER
                    fig.plot_type_confidence = Confidence.LOW
            except Exception as e:
                logger.warning(f"Classification failed for {fig.figure_id}: {e}")

    await asyncio.gather(*[_classify_one(fig) for fig in figures])
    return figures
