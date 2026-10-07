"""
Centralized design tokens, color palettes, and bounding constants for deterministic chart rendering.
All styling is application-owned: neither the LLM nor the end user can inject arbitrary styling directives.
"""

# Figure geometry and resolution
FIG_WIDTH_INCHES: float = 8.0
FIG_HEIGHT_INCHES: float = 4.5
DPI: int = 150
WIDTH_PX: int = int(FIG_WIDTH_INCHES * DPI)   # 1200 px
HEIGHT_PX: int = int(FIG_HEIGHT_INCHES * DPI)  # 675 px

# Resource and artifact bounds
MAX_ROWS: int = 500
MAX_BAR_CATEGORIES: int = 20
MAX_PNG_BYTES: int = 2 * 1024 * 1024  # 2 MB hard limit
MAX_TITLE_LENGTH: int = 80
MAX_LABEL_LENGTH: int = 25

# Color Palette (curated, modern, high-contrast)
PRIMARY_COLOR: str = "#4F46E5"     # Indigo
SECONDARY_COLOR: str = "#059669"   # Emerald
ACCENT_AMBER: str = "#D97706"      # Amber
ACCENT_ROSE: str = "#E11D48"       # Rose
ACCENT_VIOLET: str = "#7C3AED"     # Violet
ACCENT_CYAN: str = "#0891B2"       # Cyan

# Categorical palette sequence
PALETTE: list[str] = [
    "#4F46E5",  # Indigo
    "#059669",  # Emerald
    "#D97706",  # Amber
    "#E11D48",  # Rose
    "#7C3AED",  # Violet
    "#0891B2",  # Cyan
    "#2563EB",  # Blue
    "#DB2777",  # Pink
]

# Structural / Chrome Styling
BG_COLOR: str = "#FFFFFF"
GRID_COLOR: str = "#E2E8F0"        # Slate-200
TEXT_MAIN: str = "#1E293B"         # Slate-800
TEXT_MUTED: str = "#64748B"        # Slate-500
SPINE_COLOR: str = "#CBD5E1"       # Slate-300

# Typography
FONT_FAMILY: str = "sans-serif"
FONT_SIZE_TITLE: int = 13
FONT_SIZE_LABEL: int = 10
FONT_SIZE_TICK: int = 9


def truncate_text(text: str, max_length: int) -> str:
    """Safely truncates text and adds ellipsis if over max length."""
    if not text:
        return ""
    s = str(text).strip()
    if len(s) > max_length:
        return s[: max_length - 3] + "..."
    return s
