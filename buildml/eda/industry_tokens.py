# ruff: noqa: E501
"""Shared Industry design tokens for Static EDA and the Teaching Studio App.

Single source for the steel / blueprint palette (``#5980a6``), Barlow stacks,
square corners, and hairline spacing. Static HTML inlines these into
``cockpit_style.COCKPIT_CSS``; the dashboard App maps the same values into
``tokens.css`` and Plotly palettes so the two surfaces stay aligned without
forking a second Industry theme.
"""

from __future__ import annotations

from typing import Any

# CSS custom properties only (no selectors beyond :root). Offline-safe: no CDN.
INDUSTRY_ROOT_CSS = """
:root {
  color-scheme: light;
  --color-bg: #f2f2f3;
  --color-surface: #e9e9ea;
  --color-text: #1d1f20;
  --color-accent: #5980a6;
  --color-accent-2: #728fab;
  --color-divider: color-mix(in srgb, #1d1f20 16%, transparent);
  --color-neutral-100: #f5f5f8;
  --color-neutral-200: #e7e7ea;
  --color-neutral-300: #d4d4d7;
  --color-neutral-400: #b7b7ba;
  --color-neutral-500: #98989b;
  --color-neutral-600: #7a7a7d;
  --color-neutral-700: #5d5d60;
  --color-neutral-800: #424244;
  --color-neutral-900: #2b2b2d;
  --color-accent-100: #eef6ff;
  --color-accent-200: #d6ebff;
  --color-accent-300: #b5d9fd;
  --color-accent-400: #94bce3;
  --color-accent-500: #749dc4;
  --color-accent-600: #597ea3;
  --color-accent-700: #416180;
  --color-accent-800: #2c455d;
  --color-accent-900: #1d2d3d;
  --font-heading: "Barlow Condensed", "Arial Narrow", "Helvetica Neue Condensed",
    "Franklin Gothic Medium", "Segoe UI", sans-serif;
  --font-heading-weight: 600;
  --font-body: "Barlow", "Segoe UI", "Helvetica Neue", system-ui, sans-serif;
  --font-mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --space-1: 3.4px;
  --space-2: 6.8px;
  --space-3: 10.2px;
  --space-4: 13.6px;
  --space-6: 20.4px;
  --space-8: 27.2px;
  --radius-sm: 0;
  --radius-md: 0;
  --radius-lg: 0;
  --shadow-sm: 0 1px 2px color-mix(in srgb, #2b2b2d 14%, transparent);
  --bml-max: 1180px;
}
""".strip()

INDUSTRY_ACCENT = "#5980a6"
INDUSTRY_TEXT = "#1d1f20"
INDUSTRY_BG = "#f2f2f3"

# Plotly / App chart palettes keyed off the same steel ramp.
INDUSTRY_CHART_LIGHT: dict[str, Any] = {
    "accent": "#5980a6",
    "accent_soft": "#749dc4",
    "ink": "#1d1f20",
    "muted": "#5d5d60",
    "warn": "#b45309",
    "critical": "#b91c1c",
    "info": "#416180",
    "grid": "#d4d4d7",
    "hover_bg": "#f5f5f8",
    "hover_ink": "#1d1f20",
    "annotation": "#5d5d60",
    "series": [
        "#5980a6",
        "#2c455d",
        "#b45309",
        "#416180",
        "#728fab",
        "#9f1239",
        "#424244",
        "#a16207",
    ],
    "heatmap": [
        [0.0, "#2c455d"],
        [0.5, "#f5f5f8"],
        [1.0, "#5980a6"],
    ],
    "gauge_steps": [
        {"range": [0, 5], "color": "#eef6ff"},
        {"range": [5, 10], "color": "#fff1e0"},
    ],
    "gauge_hot": "#fde8e8",
}

INDUSTRY_CHART_DARK: dict[str, Any] = {
    "accent": "#94bce3",
    "accent_soft": "#749dc4",
    "ink": "#e8eef5",
    "muted": "#9aa8b8",
    "warn": "#f0a35a",
    "critical": "#f07178",
    "info": "#94bce3",
    "grid": "#2a3340",
    "hover_bg": "#171c24",
    "hover_ink": "#e8eef5",
    "annotation": "#9aa8b8",
    "series": [
        "#94bce3",
        "#5980a6",
        "#f0a35a",
        "#b5d9fd",
        "#728fab",
        "#fb7185",
        "#98989b",
        "#fbbf24",
    ],
    "heatmap": [
        [0.0, "#94bce3"],
        [0.5, "#1d2430"],
        [1.0, "#5980a6"],
    ],
    "gauge_steps": [
        {"range": [0, 5], "color": "#1d2d3d"},
        {"range": [5, 10], "color": "#3a2716"},
    ],
    "gauge_hot": "#3a1719",
}
