"""Pydantic models for Get Me The Data."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PlotType(str, Enum):
    SCATTER = "scatter"
    BAR = "bar"
    LINE = "line"
    BOX = "box"
    VIOLIN = "violin"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    FOREST = "forest"
    KAPLAN_MEIER = "kaplan_meier"
    DOT_STRIP = "dot_strip"
    STACKED_BAR = "stacked_bar"
    FUNNEL = "funnel"
    ROC = "roc"
    VOLCANO = "volcano"
    WATERFALL = "waterfall"
    BLAND_ALTMAN = "bland_altman"
    PAIRED = "paired"
    BUBBLE = "bubble"
    AREA = "area"
    DOSE_RESPONSE = "dose_response"
    MANHATTAN = "manhattan"
    CORRELATION_MATRIX = "correlation_matrix"
    ERROR_BAR = "error_bar"
    TABLE = "table"
    OTHER = "other"
    NON_DATA = "non_data"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Paper(BaseModel):
    paper_id: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    source_url: Optional[str] = None


class Figure(BaseModel):
    figure_id: str
    paper_id: str
    page_number: int
    image_index: int
    width: int
    height: int
    image_base64: str
    title: Optional[str] = None
    legend: Optional[str] = None
    plot_type: PlotType = PlotType.OTHER
    plot_type_confidence: Confidence = Confidence.MEDIUM


class DataSeries(BaseModel):
    name: str = "Series 1"
    x_values: list[float | str] = Field(default_factory=list)
    y_values: list[float] = Field(default_factory=list)
    error_bars_lower: list[float | None] = Field(default_factory=list)
    error_bars_upper: list[float | None] = Field(default_factory=list)


class ExtractedData(BaseModel):
    figure_id: str
    plot_type: PlotType
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_scale: Optional[str] = None  # "linear" or "log"
    y_scale: Optional[str] = None  # "linear" or "log"
    series: list[DataSeries] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    notes: Optional[str] = None
    # Context metadata — what legend/text context was used during extraction
    legend_text: Optional[str] = None  # Figure legend/caption from PDF
    text_mentions: list[str] = Field(default_factory=list)  # Sentences mentioning this figure


class SupplementaryItem(BaseModel):
    label: str
    item_type: str  # "data_file", "repository", "supplementary_figure", etc.
    source: str  # "publisher", "github", "zenodo", "figshare", etc.
    url: Optional[str] = None
    description: Optional[str] = None


class TableData(BaseModel):
    table_id: str
    paper_id: str
    page_number: int
    caption: Optional[str] = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    notes: Optional[str] = None
    source: str = "pdf_text"  # "pdf_text" or "image"


class ResolveRequest(BaseModel):
    url: str


class ExtractRequest(BaseModel):
    paper_id: str
    figure_ids: list[str]


class ExtractSingleRequest(BaseModel):
    paper_id: str
    figure_id: str
    # Optional crop region as percentages (0-100) of the image dimensions
    crop_x_pct: Optional[float] = None
    crop_y_pct: Optional[float] = None
    crop_w_pct: Optional[float] = None
    crop_h_pct: Optional[float] = None


class TableExtractRequest(BaseModel):
    paper_id: str
    table_ids: list[str]
