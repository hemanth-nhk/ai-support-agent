"""
Deterministic, server-side chart rendering engine for Tabular Analytics.
Guarantees:
1. LLM never generates Python/Matplotlib/style/path code.
2. OO Figure/Canvas per-call isolation; no shared global pyplot state; explicit cleanup.
3. Independent result-shape validation (no blind coercion of explicit requests).
4. Session-isolated storage under data/charts/{session_id}/{chart_id}.png with path traversal defenses.
5. Strict artifact bounds (dimensions, DPI, row counts, PNG bytes, title/label lengths).
"""
import io
import logging
from pathlib import Path
import re
from typing import Any, Optional
import uuid

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from app.config import settings
from app.analytics.schemas import AnalysisResult, ChartArtifact
from app.analytics.chart_style import (
    FIG_WIDTH_INCHES,
    FIG_HEIGHT_INCHES,
    DPI,
    WIDTH_PX,
    HEIGHT_PX,
    MAX_ROWS,
    MAX_BAR_CATEGORIES,
    MAX_PNG_BYTES,
    MAX_TITLE_LENGTH,
    MAX_LABEL_LENGTH,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    PALETTE,
    BG_COLOR,
    GRID_COLOR,
    TEXT_MAIN,
    TEXT_MUTED,
    SPINE_COLOR,
    FONT_FAMILY,
    FONT_SIZE_TITLE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    truncate_text,
)

logger = logging.getLogger("app.analytics.visualizer")

PNG_MAGIC_BYTES = b"\x89PNG\r\n\x1a\n"
UUID_PATTERN = re.compile(r"^[0-9a-fA-F\-]{8,64}$")


class ChartRenderError(Exception):
    """Raised when chart generation or validation fails."""
    pass


class ChartIncompatibleError(Exception):
    """Raised when requested chart type is incompatible with result shape."""
    pass


class ChartSecurityError(Exception):
    """Raised when traversal or unauthorized path access is detected."""
    pass


def _is_numeric_value(val: Any) -> bool:
    """Checks whether a scalar value is numeric (int/float, excluding bool)."""
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return True
    try:
        float(str(val))
        return True
    except (ValueError, TypeError):
        return False


def _to_float(val: Any) -> float:
    """Converts a value to float, defaulting to 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _is_temporal_string(val: Any) -> bool:
    """Heuristic for ISO dates or year strings like '2023-01-01' or '2023-Q1'."""
    if not isinstance(val, str):
        return False
    s = val.strip()
    return bool(
        re.match(r"^\d{4}-\d{2}-\d{2}", s)
        or re.match(r"^\d{4}-\d{2}", s)
        or re.match(r"^\d{4}/", s)
        or re.match(r"^\d{4}$", s)
        or re.match(r"^\d{4}-q\d", s, re.IGNORECASE)
    )


def classify_result_columns(result: AnalysisResult) -> dict[str, list[str]]:
    """
    Independently inspects actual AnalysisResult rows to classify columns into:
    numeric, temporal, categorical.
    """
    if not result.rows or not result.columns:
        return {"numeric": [], "temporal": [], "categorical": []}

    sample_rows = result.rows[: min(50, len(result.rows))]
    numeric_cols: list[str] = []
    temporal_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in result.columns:
        values = [row.get(col) for row in sample_rows if row.get(col) is not None]
        if not values:
            categorical_cols.append(col)
            continue

        num_count = sum(1 for v in values if _is_numeric_value(v))
        if num_count / len(values) >= 0.8:
            numeric_cols.append(col)
        elif all(_is_temporal_string(v) for v in values):
            temporal_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "temporal": temporal_cols,
        "categorical": categorical_cols,
    }


def validate_chart_compatibility(
    result: AnalysisResult,
    requested_type: Optional[str] = None,
    is_explicit: bool = False,
) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Validates whether the DuckDB result can be rendered as a chart.
    Returns: (is_compatible, effective_chart_type, error_reason)

    Invariants:
    - Zero rows -> no chart.
    - Explicit incompatible chart -> fails with structured reason; NO silent coercion.
    - Implicit chart -> deterministic compatible selection based on result shape.
    """
    if result.row_count == 0 or not result.rows:
        return False, None, "The query returned 0 rows. A chart cannot be rendered from an empty result."

    col_classes = classify_result_columns(result)
    num_cols = col_classes["numeric"]
    cat_cols = col_classes["categorical"]
    temp_cols = col_classes["temporal"]
    dim_cols = cat_cols + temp_cols

    # Evaluation per chart type
    def _can_render_bar() -> bool:
        return len(dim_cols) >= 1 and len(num_cols) >= 1

    def _can_render_line() -> bool:
        # Preferred with temporal dimension, but acceptable with 1 ordered dim + 1 numeric
        return (len(temp_cols) >= 1 or len(cat_cols) >= 1) and len(num_cols) >= 1

    def _can_render_scatter() -> bool:
        # Requires at least 2 numeric columns
        return len(num_cols) >= 2

    def _can_render_histogram() -> bool:
        # Requires at least 1 numeric column
        return len(num_cols) >= 1

    type_checks = {
        "bar": _can_render_bar,
        "line": _can_render_line,
        "scatter": _can_render_scatter,
        "histogram": _can_render_histogram,
    }

    if requested_type and is_explicit:
        req = requested_type.lower()
        if req not in type_checks:
            return False, None, f"Unsupported chart type '{requested_type}'. Supported types: bar, line, scatter, histogram."
        if type_checks[req]():
            return True, req, None
        else:
            shape_desc = f"{len(dim_cols)} dimensions ({', '.join(dim_cols) or 'none'}) and {len(num_cols)} numeric metrics ({', '.join(num_cols) or 'none'})"
            return (
                False,
                None,
                f"Requested '{req}' chart is incompatible with result shape: found {shape_desc}.",
            )

    # Implicit selection (or requested_type is a suggestion without explicit user demand)
    if requested_type and requested_type.lower() in type_checks and type_checks[requested_type.lower()]():
        return True, requested_type.lower(), None

    # Deterministic automatic selection
    if len(temp_cols) >= 1 and len(num_cols) >= 1:
        return True, "line", None
    if len(dim_cols) >= 1 and len(num_cols) >= 1:
        return True, "bar", None
    if len(num_cols) >= 2:
        return True, "scatter", None
    if len(num_cols) == 1:
        return True, "histogram", None

    return False, None, f"Could not determine a compatible chart type for columns: {result.columns}."


def _render_bar_chart(
    ax: Any,
    rows: list[dict[str, Any]],
    cat_col: str,
    num_col: str,
    title: str,
) -> None:
    # Bound categories
    clamped_rows = rows[:MAX_BAR_CATEGORIES]
    labels = [truncate_text(str(r.get(cat_col, "")), MAX_LABEL_LENGTH) for r in clamped_rows]
    values = [_to_float(r.get(num_col, 0)) for r in clamped_rows]

    # Bar colors
    colors = PALETTE[: len(clamped_rows)] if len(clamped_rows) <= len(PALETTE) else PRIMARY_COLOR
    bars = ax.bar(range(len(clamped_rows)), values, color=colors, width=0.6, edgecolor=SPINE_COLOR, linewidth=0.5)

    ax.set_xticks(range(len(clamped_rows)))
    rotation = 35 if len(clamped_rows) > 5 else 0
    ha = "right" if rotation > 0 else "center"
    ax.set_xticklabels(labels, rotation=rotation, ha=ha, fontsize=FONT_SIZE_TICK, color=TEXT_MAIN)
    ax.set_ylabel(truncate_text(num_col, MAX_LABEL_LENGTH), fontsize=FONT_SIZE_LABEL, color=TEXT_MAIN)
    ax.set_title(truncate_text(title, MAX_TITLE_LENGTH), fontsize=FONT_SIZE_TITLE, fontweight="bold", color=TEXT_MAIN, pad=12)


def _render_line_chart(
    ax: Any,
    rows: list[dict[str, Any]],
    x_col: str,
    y_col: str,
    title: str,
) -> None:
    # Sort by x_col if temporal/ordered
    sorted_rows = rows[:MAX_ROWS]
    labels = [truncate_text(str(r.get(x_col, "")), MAX_LABEL_LENGTH) for r in sorted_rows]
    values = [_to_float(r.get(y_col, 0)) for r in sorted_rows]

    ax.plot(
        range(len(sorted_rows)),
        values,
        color=PRIMARY_COLOR,
        marker="o",
        markersize=5,
        linewidth=2,
        label=truncate_text(y_col, MAX_LABEL_LENGTH),
    )

    # Subsample x-ticks if there are many rows
    step = max(1, len(sorted_rows) // 10)
    tick_indices = list(range(0, len(sorted_rows), step))
    if (len(sorted_rows) - 1) not in tick_indices:
        tick_indices.append(len(sorted_rows) - 1)

    ax.set_xticks(tick_indices)
    ax.set_xticklabels([labels[i] for i in tick_indices], rotation=35, ha="right", fontsize=FONT_SIZE_TICK, color=TEXT_MAIN)
    ax.set_ylabel(truncate_text(y_col, MAX_LABEL_LENGTH), fontsize=FONT_SIZE_LABEL, color=TEXT_MAIN)
    ax.set_title(truncate_text(title, MAX_TITLE_LENGTH), fontsize=FONT_SIZE_TITLE, fontweight="bold", color=TEXT_MAIN, pad=12)


def _render_scatter_chart(
    ax: Any,
    rows: list[dict[str, Any]],
    x_col: str,
    y_col: str,
    title: str,
) -> None:
    clamped_rows = rows[:MAX_ROWS]
    x_vals = [_to_float(r.get(x_col, 0)) for r in clamped_rows]
    y_vals = [_to_float(r.get(y_col, 0)) for r in clamped_rows]

    ax.scatter(
        x_vals,
        y_vals,
        color=PRIMARY_COLOR,
        alpha=0.75,
        edgecolors=SPINE_COLOR,
        linewidth=0.5,
        s=40,
    )
    ax.set_xlabel(truncate_text(x_col, MAX_LABEL_LENGTH), fontsize=FONT_SIZE_LABEL, color=TEXT_MAIN)
    ax.set_ylabel(truncate_text(y_col, MAX_LABEL_LENGTH), fontsize=FONT_SIZE_LABEL, color=TEXT_MAIN)
    ax.set_title(truncate_text(title, MAX_TITLE_LENGTH), fontsize=FONT_SIZE_TITLE, fontweight="bold", color=TEXT_MAIN, pad=12)


def _render_histogram_chart(
    ax: Any,
    rows: list[dict[str, Any]],
    num_col: str,
    title: str,
) -> None:
    clamped_rows = rows[:MAX_ROWS]
    values = [_to_float(r.get(num_col, 0)) for r in clamped_rows if _is_numeric_value(r.get(num_col))]

    bins = min(20, max(5, len(values) // 5)) if values else 10
    ax.hist(
        values,
        bins=bins,
        color=SECONDARY_COLOR,
        edgecolor=BG_COLOR,
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_xlabel(truncate_text(num_col, MAX_LABEL_LENGTH), fontsize=FONT_SIZE_LABEL, color=TEXT_MAIN)
    ax.set_ylabel("Frequency", fontsize=FONT_SIZE_LABEL, color=TEXT_MAIN)
    ax.set_title(truncate_text(title, MAX_TITLE_LENGTH), fontsize=FONT_SIZE_TITLE, fontweight="bold", color=TEXT_MAIN, pad=12)


def render_chart(
    result: AnalysisResult,
    chart_type: str,
    session_id: str,
    title: Optional[str] = None,
) -> ChartArtifact:
    """
    Renders an AnalysisResult into a bounded PNG artifact and persists it under
    data/charts/{session_id}/{chart_id}.png.

    Thread safety / State isolation:
    - Each render owns its Figure and Canvas instance.
    - Explicit cleanup via fig.clf().
    """
    if not session_id or not UUID_PATTERN.match(session_id.strip()):
        raise ChartSecurityError(f"Invalid session_id format: '{session_id}'")

    clean_session_id = session_id.strip()
    col_classes = classify_result_columns(result)
    num_cols = col_classes["numeric"]
    dim_cols = col_classes["categorical"] + col_classes["temporal"]

    # Deduce columns to plot
    chart_type = chart_type.lower()
    default_title = f"{chart_type.capitalize()} Chart"
    if result.plan and result.plan.aggregations:
        agg = result.plan.aggregations[0]
        agg_name = agg.alias or (f"{agg.function.value}({agg.column})" if agg.column else "Count")
        if result.plan.group_by:
            gb_name = result.plan.group_by[0].column
            default_title = f"{agg_name} by {gb_name}"
        else:
            default_title = agg_name

    final_title = title or default_title

    # Create isolated figure instance
    fig = Figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI, facecolor=BG_COLOR)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Style axes chrome
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, linestyle="--", alpha=0.6, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.8)

    try:
        if chart_type == "bar":
            cat_col = dim_cols[0] if dim_cols else (result.columns[0] if result.columns else "dimension")
            num_col = num_cols[0] if num_cols else (result.columns[1] if len(result.columns) > 1 else "value")
            _render_bar_chart(ax, result.rows, cat_col, num_col, final_title)

        elif chart_type == "line":
            temp_col = col_classes["temporal"][0] if col_classes["temporal"] else (dim_cols[0] if dim_cols else result.columns[0])
            num_col = num_cols[0] if num_cols else (result.columns[1] if len(result.columns) > 1 else "value")
            _render_line_chart(ax, result.rows, temp_col, num_col, final_title)

        elif chart_type == "scatter":
            x_col = num_cols[0] if len(num_cols) >= 1 else result.columns[0]
            y_col = num_cols[1] if len(num_cols) >= 2 else (result.columns[1] if len(result.columns) > 1 else result.columns[0])
            _render_scatter_chart(ax, result.rows, x_col, y_col, final_title)

        elif chart_type == "histogram":
            num_col = num_cols[0] if num_cols else result.columns[0]
            _render_histogram_chart(ax, result.rows, num_col, final_title)

        else:
            raise ChartRenderError(f"Unsupported chart_type '{chart_type}'.")

        fig.tight_layout()

        # Render to in-memory buffer
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        png_bytes = buffer.getvalue()

    finally:
        # Explicit release to prevent GDI / memory leak
        fig.clf()

    # Verify PNG magic bytes
    if not png_bytes.startswith(PNG_MAGIC_BYTES):
        raise ChartRenderError("Generated chart artifact failed PNG magic byte validation.")

    # Verify PNG artifact size limit
    if len(png_bytes) > MAX_PNG_BYTES:
        raise ChartRenderError(f"Chart artifact size ({len(png_bytes)} bytes) exceeds maximum {MAX_PNG_BYTES} bytes.")

    # Session-isolated storage
    chart_id = str(uuid.uuid4())
    charts_base = Path(settings.charts_dir).resolve()
    session_charts_dir = (charts_base / clean_session_id).resolve()

    # Traversal security check
    if not session_charts_dir.is_relative_to(charts_base):
        raise ChartSecurityError(f"Session path escape attempt: '{session_id}'")

    session_charts_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_charts_dir / f"{chart_id}.png"

    with open(file_path, "wb") as f:
        f.write(png_bytes)

    return ChartArtifact(
        chart_id=chart_id,
        chart_type=chart_type,  # type: ignore[arg-type]
        filename=f"{chart_id}.png",
        title=truncate_text(final_title, MAX_TITLE_LENGTH),
        width_px=WIDTH_PX,
        height_px=HEIGHT_PX,
        url=f"/charts/{clean_session_id}/{chart_id}",
    )


def resolve_chart_file(session_id: str, chart_id: str) -> Path:
    """
    Authoritative server-controlled chart resolution with strict traversal prevention.
    Validates:
    - session_id format
    - chart_id format
    - path remains strictly inside session chart directory
    - file exists
    - .png extension only
    """
    if not session_id or not UUID_PATTERN.match(session_id.strip()):
        raise ChartSecurityError("Invalid session_id format.")
    if not chart_id or not UUID_PATTERN.match(chart_id.strip()):
        raise ChartSecurityError("Invalid chart_id format.")

    clean_session_id = session_id.strip()
    clean_chart_id = chart_id.strip()

    charts_base = Path(settings.charts_dir).resolve()
    session_charts_dir = (charts_base / clean_session_id).resolve()

    if not session_charts_dir.is_relative_to(charts_base):
        raise ChartSecurityError("Path traversal detected in session_id.")

    target_file = (session_charts_dir / f"{clean_chart_id}.png").resolve()

    if not target_file.is_relative_to(session_charts_dir):
        raise ChartSecurityError("Path traversal detected in chart_id.")

    if target_file.suffix.lower() != ".png":
        raise ChartSecurityError("Only .png chart artifacts are permitted.")

    if not target_file.is_file():
        raise FileNotFoundError(f"Chart artifact '{clean_chart_id}' not found.")

    return target_file
