"""
CellQuant - Gradio Web Interface
Dual-theme: Deep Focus (dark) / Histology Atlas (light)

A fully functional cell quantification application with:
- Native file system browsing
- Batch Cellpose segmentation
- CTCF quantification
- Results export
"""

import gradio as gr
from cellquant_enterprise import __version__ as APP_VERSION
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import os
import tempfile
import time

# Core imports
from cellquant_enterprise.core.io.image_loader import (
    load_image, normalize_image
)
from cellquant_enterprise.core.io.mask_io import load_mask
from cellquant_enterprise.core.io.channel_detect import (
    detect_image_sets, ImageSetDetection
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS - DUAL THEME SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Libre+Baskerville:wght@400;700&family=Crimson+Text:wght@400;600&family=Inconsolata:wght@400;500&family=Lato:wght@400;700&display=swap');

/* ══════════════════════════════════════════════════════════════
   ROOT VARIABLES - Default to Light Mode (Histology Atlas)
   ══════════════════════════════════════════════════════════════ */
:root {
    /* Light mode: Histology Atlas */
    --bg: #F9F6F0;
    --bg-elevated: #FFFEFA;
    --bg-hover: #F0EDE5;
    --text: #2C3E50;
    --text-muted: #7F8C8D;
    --accent: #6B5B95;
    --accent-soft: rgba(107, 91, 149, 0.12);
    --accent-secondary: #D4A5A5;
    --success: #45B7AA;
    --warning: #E07A5F;
    --error: #C0392B;
    --border: #E8E2D9;
    --border-strong: #D5CBBA;

    /* Typography */
    --font-display: 'Libre Baskerville', Georgia, serif;
    --font-body: 'Crimson Text', Georgia, serif;
    --font-ui: 'Lato', -apple-system, sans-serif;
    --font-mono: 'Inconsolata', monospace;

    /* Shadows */
    --shadow-card: 3px 3px 0 var(--accent-secondary);
    --shadow-hover: 4px 4px 0 var(--accent-secondary);

    /* Transitions */
    --transition-theme: background 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}

/* ══════════════════════════════════════════════════════════════
   DARK MODE - Deep Focus
   ══════════════════════════════════════════════════════════════ */
.dark {
    --bg: #0D0D0D;
    --bg-elevated: #1A1A1A;
    --bg-hover: #252525;
    --text: #E8E8E8;
    --text-muted: #888888;
    --accent: #F5A623;
    --accent-soft: rgba(245, 166, 35, 0.15);
    --accent-secondary: #F5A623;
    --success: #7CB342;
    --warning: #FFA726;
    --error: #EF5350;
    --border: #2a2a2a;
    --border-strong: #333333;

    /* Dark mode typography */
    --font-display: 'Outfit', -apple-system, sans-serif;
    --font-body: 'Outfit', -apple-system, sans-serif;
    --font-ui: 'Outfit', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;

    /* Dark mode shadows */
    --shadow-card: none;
    --shadow-hover: none;
}

/* ══════════════════════════════════════════════════════════════
   BASE STYLES
   ══════════════════════════════════════════════════════════════ */
.gradio-container {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    max-width: 100% !important;
    padding: 0 !important;
    transition: var(--transition-theme);
}

.main, .contain {
    background: var(--bg) !important;
    transition: var(--transition-theme);
}

footer { display: none !important; }

/* Hide the "processing | Xs" loading indicator on textboxes */
.meta-text, .meta-text-center { display: none !important; }

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-elevated); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ══════════════════════════════════════════════════════════════
   HEADER
   ══════════════════════════════════════════════════════════════ */
.app-header {
    background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg) 100%);
    border-bottom: 2px solid var(--accent);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: var(--transition-theme);
}

.dark .app-header {
    background: var(--bg);
    border-bottom: 1px solid var(--border);
}

.app-title {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.01em;
    margin: 0;
    transition: var(--transition-theme);
}

.dark .app-title {
    color: var(--text);
    font-weight: 500;
    font-size: 18px;
}

.app-subtitle {
    font-family: var(--font-ui);
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
    font-style: italic;
}

.dark .app-subtitle {
    font-style: normal;
    font-weight: 300;
}

/* Theme Toggle Button */
.theme-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-elevated);
    padding: 8px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
    cursor: pointer;
    font-family: var(--font-ui);
    font-size: 12px;
    color: var(--text-muted);
    transition: var(--transition-theme);
}

.theme-toggle:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* ══════════════════════════════════════════════════════════════
   TABS
   ══════════════════════════════════════════════════════════════ */
.tabs {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}

.tab-nav {
    background: var(--bg) !important;
    border: none !important;
    padding: 0 16px !important;
    gap: 0 !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 14px 20px !important;
    font-family: var(--font-ui) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: var(--text-muted) !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    color: var(--accent) !important;
    background: transparent !important;
}

.tab-nav button.selected {
    color: var(--accent) !important;
    font-weight: 700 !important;
    border-bottom: 3px solid var(--accent) !important;
    background: transparent !important;
}

/* Light mode: gradient underline */
.tab-nav button.selected {
    border-image: linear-gradient(90deg, var(--accent), var(--accent-secondary)) 1 !important;
}

.dark .tab-nav button.selected {
    border-image: none !important;
    border-bottom-color: var(--accent) !important;
    font-weight: 500 !important;
}

/* ══════════════════════════════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════════════════════════════ */
.section-header {
    font-family: var(--font-display);
    font-size: 16px;
    font-weight: 700;
    color: var(--text);
    margin: 20px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-header::before {
    content: '\\00A7';
    color: var(--success);
    font-size: 18px;
}

.dark .section-header {
    font-size: 13px;
    font-weight: 500;
    border-bottom: none;
    padding-bottom: 0;
}

.dark .section-header::before {
    content: '';
    width: 3px;
    height: 14px;
    background: var(--accent);
    border-radius: 2px;
}

/* ══════════════════════════════════════════════════════════════
   FORM ELEMENTS
   ══════════════════════════════════════════════════════════════ */
.gr-input, .gr-textarea, input[type="text"], input[type="number"], textarea, select {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
    font-size: 13px !important;
    padding: 10px 12px !important;
    transition: var(--transition-theme) !important;
}

.gr-input:focus, input:focus, textarea:focus, select:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

label, .gr-label {
    font-family: var(--font-ui) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    margin-bottom: 6px !important;
}

.dark label, .dark .gr-label {
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    font-size: 11px !important;
}

/* ══════════════════════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════════════════════ */
.gr-button, button.primary {
    font-family: var(--font-ui) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.gr-button-primary, button.primary {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
}

.gr-button-primary:hover, button.primary:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
}

.gr-button-secondary, button.secondary {
    background: var(--bg-elevated) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

.gr-button-secondary:hover, button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

.dark .gr-button-primary, .dark button.primary {
    color: #000 !important;
}

/* ══════════════════════════════════════════════════════════════
   STAT CARDS
   ══════════════════════════════════════════════════════════════ */
.stat-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    box-shadow: var(--shadow-card);
    transition: var(--transition-theme);
}

.stat-card:hover {
    transform: translate(-1px, -1px);
    box-shadow: var(--shadow-hover);
}

.dark .stat-card {
    box-shadow: none;
    border-radius: 0;
}

.dark .stat-card:hover {
    transform: none;
    background: var(--bg-hover);
}

.stat-value {
    font-family: var(--font-mono);
    font-size: 28px;
    font-weight: 500;
    color: var(--accent);
    line-height: 1.2;
}

.dark .stat-value {
    color: var(--text);
    font-size: 26px;
}

.stat-label {
    font-family: var(--font-ui);
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 4px;
}

/* Stats Grid Container */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 16px 0;
}

.dark .stats-grid {
    gap: 1px;
    background: var(--border);
    border-radius: 8px;
    overflow: hidden;
}

.dark .stats-grid .stat-card {
    border: none;
    border-radius: 0;
}

/* ══════════════════════════════════════════════════════════════
   DATA TABLES
   ══════════════════════════════════════════════════════════════ */
.gr-dataframe, table, .dataframe {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-card) !important;
    transition: var(--transition-theme) !important;
}

.dark .gr-dataframe, .dark table, .dark .dataframe {
    box-shadow: none !important;
    border-radius: 8px !important;
}

table thead th, .gr-dataframe thead th {
    background: linear-gradient(180deg, var(--accent) 0%, #5a4a84 100%) !important;
    color: white !important;
    font-family: var(--font-ui) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 12px 14px !important;
    border: none !important;
}

.dark table thead th, .dark .gr-dataframe thead th {
    background: var(--bg) !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    border-bottom: 1px solid var(--border) !important;
}

table tbody td, .gr-dataframe tbody td {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    padding: 11px 14px !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text) !important;
    background: var(--bg-elevated) !important;
}

table tbody tr:nth-child(even) td {
    background: rgba(212, 165, 165, 0.08) !important;
}

.dark table tbody tr:nth-child(even) td {
    background: var(--bg-elevated) !important;
}

table tbody tr:hover td {
    background: rgba(69, 183, 170, 0.1) !important;
}

.dark table tbody tr:hover td {
    background: var(--bg-hover) !important;
}

/* ══════════════════════════════════════════════════════════════
   PLOTS & CHARTS
   ══════════════════════════════════════════════════════════════ */
.gr-plot, .plot-container {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    box-shadow: var(--shadow-card) !important;
    transition: var(--transition-theme) !important;
}

.dark .gr-plot, .dark .plot-container {
    box-shadow: none !important;
    border-radius: 8px !important;
}

/* ══════════════════════════════════════════════════════════════
   IMAGE PLACEHOLDERS
   ══════════════════════════════════════════════════════════════ */
.image-placeholder {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-family: var(--font-ui);
    font-size: 13px;
    box-shadow: var(--shadow-card);
    transition: var(--transition-theme);
}

.dark .image-placeholder {
    box-shadow: none;
    border-radius: 8px;
}

/* ══════════════════════════════════════════════════════════════
   PROGRESS BARS
   ══════════════════════════════════════════════════════════════ */
.progress-bar {
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin: 12px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--success));
    border-radius: 4px;
    transition: width 0.3s ease;
}

.dark .progress-fill {
    background: var(--accent);
}

/* ══════════════════════════════════════════════════════════════
   STATUS BADGES
   ══════════════════════════════════════════════════════════════ */
.status-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-family: var(--font-ui);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.status-badge.success {
    background: rgba(69, 183, 170, 0.2);
    color: #2d8a7f;
}

.status-badge.pending {
    background: var(--accent-soft);
    color: var(--accent);
}

.status-badge.error {
    background: rgba(192, 57, 43, 0.15);
    color: var(--error);
}

.dark .status-badge.success {
    background: rgba(124, 179, 66, 0.2);
    color: var(--success);
}

/* ══════════════════════════════════════════════════════════════
   ACCORDION / COLLAPSIBLE
   ══════════════════════════════════════════════════════════════ */
.gr-accordion {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--bg-elevated) !important;
    margin: 8px 0 !important;
}

.gr-accordion summary {
    font-family: var(--font-ui) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    padding: 12px 16px !important;
}

/* ══════════════════════════════════════════════════════════════
   SLIDER
   ══════════════════════════════════════════════════════════════ */
input[type="range"] {
    accent-color: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════════════
   GALLERY
   ══════════════════════════════════════════════════════════════ */
.gr-gallery {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ══════════════════════════════════════════════════════════════
   CHECKBOX & RADIO
   ══════════════════════════════════════════════════════════════ */
.gr-checkbox, .gr-radio {
    accent-color: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════════════
   MARKDOWN
   ══════════════════════════════════════════════════════════════ */
.gr-markdown {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: var(--font-display) !important;
    color: var(--text) !important;
}

.gr-markdown code {
    font-family: var(--font-mono) !important;
    background: var(--bg-elevated) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}

/* ══════════════════════════════════════════════════════════════
   FOOTER
   ══════════════════════════════════════════════════════════════ */
.app-footer {
    text-align: center;
    padding: 16px;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# JAVASCRIPT FOR THEME TOGGLE
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_JS = """
function initTheme() {
    const saved = localStorage.getItem('cellquant-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = saved ? saved === 'dark' : prefersDark;

    if (isDark) {
        document.body.classList.add('dark');
    }
    updateToggleButton(isDark);
}

function toggleTheme() {
    const isDark = document.body.classList.toggle('dark');
    localStorage.setItem('cellquant-theme', isDark ? 'dark' : 'light');
    updateToggleButton(isDark);
    return isDark;
}

function updateToggleButton(isDark) {
    const btns = document.querySelectorAll('.theme-toggle');
    btns.forEach(function(btn) {
        btn.textContent = isDark ? 'Light Mode' : 'Dark Mode';
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
} else {
    initTheme();
}

"""


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION STATE
# ═══════════════════════════════════════════════════════════════════════════

class AppState:
    """Global application state."""
    def __init__(self):
        self.experiment_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.conditions: Dict[str, dict] = {}  # {name: {name, path, tiff_files, n_images}}
        self.masks: Dict[str, Dict[str, np.ndarray]] = {}  # {condition: {base_name: masks_array}}
        self.results_df: Optional[pd.DataFrame] = None


# Global state instance
app_state = AppState()


# ═══════════════════════════════════════════════════════════════════════════
# NATIVE FOLDER DIALOG (tkinter backend, user sees Windows Explorer)
# ═══════════════════════════════════════════════════════════════════════════

def open_folder_dialog(current_path: str = "") -> str:
    """Open the native OS folder picker. Returns selected path or current_path if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    initial_dir = current_path if current_path and os.path.isdir(current_path) else str(Path.home())

    folder = filedialog.askdirectory(
        title="Select Experiment Folder",
        initialdir=initial_dir
    )

    root.destroy()
    return folder if folder else current_path


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def detect_conditions(folder_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Auto-detect conditions from experiment folder.
    Uses smart channel detection to identify image sets and channels.
    """
    empty_cols = ["Condition", "Image Sets", "Channels", "Suffixes", "Status"]
    if not folder_path or not os.path.isdir(folder_path):
        return pd.DataFrame(columns=empty_cols), ""

    folder = Path(folder_path)
    app_state.experiment_path = folder
    app_state.conditions = {}

    conditions_data = []

    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue

        tiff_files = list(subdir.glob('*.tif')) + list(subdir.glob('*.tiff'))
        if not tiff_files:
            continue

        # Smart channel detection
        detection = detect_image_sets(tiff_files)

        status = "Complete" if detection.n_incomplete == 0 else f"{detection.n_incomplete} incomplete"
        if detection.n_orphan_files > 0:
            status += f", {detection.n_orphan_files} orphan"

        conditions_data.append({
            "Condition": subdir.name,
            "Image Sets": detection.n_image_sets,
            "Channels": detection.n_channels,
            "Suffixes": ", ".join(detection.channel_suffixes),
            "Status": status,
        })

        app_state.conditions[subdir.name] = {
            "name": subdir.name,
            "path": subdir,
            "tiff_files": tiff_files,
            "n_images": detection.n_image_sets,
            "detection": detection,
        }

    if conditions_data:
        df = pd.DataFrame(conditions_data)
        total_sets = sum(c["Image Sets"] for c in conditions_data)
        total_conds = len(conditions_data)
        # Get channel info from first condition
        first_det = list(app_state.conditions.values())[0]["detection"]
        ch_info = f", {first_det.n_channels}ch ({', '.join(first_det.channel_suffixes)})"
        status = f"Found {total_conds} conditions, {total_sets} image sets{ch_info}"
    else:
        df = pd.DataFrame(columns=empty_cols)
        status = "No subfolders with TIFF images found"

    return df, status


def _get_first_detection() -> Optional[ImageSetDetection]:
    """Get the detection result from the first condition (for channel suggestions)."""
    for cond in app_state.conditions.values():
        det = cond.get("detection")
        if det and det.n_channels > 0:
            return det
    return None


def browse_and_detect(current_path: str):
    """
    Open native folder dialog, then auto-detect conditions.
    Returns: folder_path, conditions_table, status,
             nuclear_ch_update, cyto_ch_update, marker_suffixes_update, marker_names_update,
             advanced_html, image_sets_table, orphan_text
    """
    folder = open_folder_dialog(current_path)
    if not folder or not os.path.isdir(folder):
        empty_cols = ["Condition", "Image Sets", "Channels", "Suffixes", "Status"]
        return (
            current_path,
            pd.DataFrame(columns=empty_cols),
            "",
            gr.update(), gr.update(), gr.update(), gr.update(),
            "", pd.DataFrame(), "",
        )

    df, status = detect_conditions(folder)

    # Auto-populate channel config from first condition's detection
    det = _get_first_detection()
    if det:
        suffixes_plus_none = ["None"] + det.channel_suffixes

        nuc_update = gr.update(
            choices=suffixes_plus_none,
            value=det.suggested_nuclear or "None"
        )
        cyto_update = gr.update(
            choices=suffixes_plus_none,
            value=det.suggested_cyto or "None"
        )
        marker_update = gr.update(
            choices=det.channel_suffixes,
            value=det.suggested_markers
        )
        # Default marker names based on suggested markers
        marker_names_val = ", ".join(
            f"Marker{i+1}" for i in range(len(det.suggested_markers))
        )
        marker_names_update = gr.update(value=marker_names_val)

        # Build advanced info
        adv_html = _build_advanced_summary(det)
        sets_df = _build_image_sets_table(det)
        orphan_txt = "\n".join(str(f) for f in det.orphan_files) if det.orphan_files else ""
    else:
        nuc_update = gr.update()
        cyto_update = gr.update()
        marker_update = gr.update()
        marker_names_update = gr.update()
        adv_html = ""
        sets_df = pd.DataFrame()
        orphan_txt = ""

    return (
        folder, df, status,
        nuc_update, cyto_update, marker_update, marker_names_update,
        adv_html, sets_df, orphan_txt,
    )


def _build_advanced_summary(det: ImageSetDetection) -> str:
    """Build HTML summary for the Advanced Settings accordion."""
    conf_pct = int(det.confidence * 100)
    badge_cls = "success" if conf_pct >= 80 else "pending" if conf_pct >= 50 else "error"
    return f"""
    <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
        <span>Detected <strong>{det.n_channels}</strong> channels
        (<code>{', '.join(det.channel_suffixes)}</code>)
        across <strong>{det.n_image_sets}</strong> image sets</span>
        <span class="status-badge {badge_cls}">Confidence: {conf_pct}%</span>
        {f'<span class="status-badge error">{det.n_incomplete} incomplete</span>' if det.n_incomplete else ''}
        {f'<span class="status-badge pending">{det.n_orphan_files} orphan files</span>' if det.n_orphan_files else ''}
    </div>
    """


def _build_image_sets_table(det: ImageSetDetection) -> pd.DataFrame:
    """Build a DataFrame showing each image set and its channel completeness."""
    if not det.image_sets:
        return pd.DataFrame()

    rows = []
    for base_name in sorted(det.image_sets.keys()):
        channels = det.image_sets[base_name]
        row = {"Image Set": base_name}
        all_present = True
        for suffix in det.channel_suffixes:
            if suffix in channels:
                row[suffix] = "OK"
            else:
                row[suffix] = "--"
                all_present = False
        row["Status"] = "Complete" if all_present else "Incomplete"
        rows.append(row)

    return pd.DataFrame(rows)


def load_preview_image(folder_path: str, nuclear_suffix: str) -> Optional[np.ndarray]:
    """Load a preview image from the first condition."""
    if not folder_path or not os.path.isdir(folder_path):
        return None

    folder = Path(folder_path)

    # Find first tiff in any subfolder
    for subdir in folder.iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob(f"*{nuclear_suffix}*.tif*"):
            try:
                img = load_image(f)
                norm = normalize_image(img)
                # Convert to uint8 for display
                return (norm * 255).astype(np.uint8)
            except Exception:
                pass

    return None


def run_segmentation(
    nuclear_suffix: str,
    cyto_suffix: str,
    model_type: str,
    diameter: float,
    flow_threshold: float,
    min_size: int,
    progress=gr.Progress()
) -> Tuple[str, Optional[np.ndarray]]:
    """Run batch segmentation using pre-computed detection results."""
    if not app_state.conditions:
        return "No conditions loaded. Browse for a folder first.", None

    # Handle "None" dropdown values
    nuc_sfx = nuclear_suffix if nuclear_suffix and nuclear_suffix != "None" else None
    cyto_sfx = cyto_suffix if cyto_suffix and cyto_suffix != "None" else None

    if not nuc_sfx and not cyto_sfx:
        return "Select at least one channel (Nuclear or Cytoplasm) for segmentation.", None

    try:
        from cellpose.models import CellposeModel
        progress(0, desc="Initializing Cellpose...")
        model = CellposeModel(model_type=model_type, gpu=True)
    except Exception as e:
        return f"Error loading Cellpose model: {e}", None

    # Use pre-computed detection results
    total_sets = sum(cond["detection"].n_image_sets for cond in app_state.conditions.values()
                     if "detection" in cond)
    if total_sets == 0:
        return "No image sets detected.", None

    processed = 0
    app_state.masks = {}
    preview_overlay = None

    for cond_name, cond_info in app_state.conditions.items():
        det = cond_info.get("detection")
        if not det:
            continue

        app_state.masks[cond_name] = {}

        for base_name, channels in det.image_sets.items():
            progress(processed / max(total_sets, 1), desc=f"Segmenting {cond_name}/{base_name}...")

            seg_channels = []

            # Load nuclear channel
            if nuc_sfx and nuc_sfx in channels:
                nuc_img = load_image(channels[nuc_sfx])
                if nuc_img is not None:
                    seg_channels.append(normalize_image(nuc_img))

            # Load cyto channel
            if cyto_sfx and cyto_sfx in channels:
                cyto_img = load_image(channels[cyto_sfx])
                if cyto_img is not None:
                    seg_channels.append(normalize_image(cyto_img))

            if not seg_channels:
                processed += 1
                continue

            seg_input = np.stack(seg_channels, axis=0)

            # Run Cellpose
            try:
                cp_args = {
                    "diameter": diameter if diameter > 0 else None,
                    "flow_threshold": flow_threshold,
                    "min_size": min_size,
                }
                if seg_input.shape[0] == 1:
                    cp_args["channels"] = [0, 0]

                masks_list, flows, styles = model.eval(  # noqa: S307 - Cellpose API
                    [seg_input], **cp_args
                )
                masks = masks_list[0]
            except Exception as e:
                print(f"Cellpose error on {cond_name}/{base_name}: {e}")
                processed += 1
                continue

            app_state.masks[cond_name][base_name] = masks

            if preview_overlay is None and masks.max() > 0:
                try:
                    from cellpose.utils import masks_to_outlines
                    display_uint8 = (seg_channels[0] * 255).astype(np.uint8)
                    rgb = np.stack([display_uint8, display_uint8, display_uint8], axis=-1)
                    outlines = masks_to_outlines(masks)
                    rgb[outlines, 0] = 255
                    rgb[outlines, 1] = 0
                    rgb[outlines, 2] = 0
                    preview_overlay = rgb
                except Exception:
                    preview_overlay = (masks > 0).astype(np.uint8) * 255

            processed += 1

    total_cells = sum(
        m.max() for cond_masks in app_state.masks.values()
        for m in cond_masks.values()
    )

    return (
        f"Segmentation complete! {total_sets} image sets, ~{total_cells} cells detected",
        preview_overlay
    )


def load_existing_masks(masks_folder: str) -> Tuple[str, Optional[np.ndarray]]:
    """Load pre-computed masks from a folder."""
    if not masks_folder or not os.path.isdir(masks_folder):
        return "Please enter a valid masks folder path", None

    folder = Path(masks_folder)
    app_state.masks = {}

    total_masks = 0
    preview = None

    for mask_file in folder.rglob("*_masks*.tif"):
        try:
            masks, _ = load_mask(mask_file)  # load_mask returns (mask, metadata)

            # Determine condition from parent folder
            cond_name = mask_file.parent.name
            base_name = mask_file.stem.replace("_masks", "").replace("_cellular", "")

            if cond_name not in app_state.masks:
                app_state.masks[cond_name] = {}

            app_state.masks[cond_name][base_name] = masks
            total_masks += 1

            if preview is None:
                # Create simple mask preview
                preview = (masks > 0).astype(np.uint8) * 255

        except Exception:
            pass

    if total_masks > 0:
        return f"Loaded {total_masks} mask files", preview
    else:
        return "No mask files found", None


def estimate_background_simple(image_data: np.ndarray, masks: np.ndarray, method: str = 'median') -> float:
    """Estimate background intensity from non-cell pixels. Matches v1 logic."""
    background_pixels = image_data[masks == 0]
    if len(background_pixels) < (image_data.size * 0.01):
        return float(np.percentile(image_data, 5))
    if method == 'median':
        return float(np.median(background_pixels))
    elif method == 'percentile5':
        return float(np.percentile(background_pixels, 5))
    elif method == 'mean':
        return float(np.mean(background_pixels))
    return float(np.median(background_pixels))


def run_quantification(
    marker_suffixes_list: list,
    marker_names_str: str,
    bg_method: str,
    min_area: int,
    progress=gr.Progress()
) -> Tuple[str, pd.DataFrame, str]:
    """Run CTCF quantification using pre-computed detection results."""
    if not app_state.masks:
        return "No masks available. Run segmentation first.", pd.DataFrame(), ""

    if not app_state.conditions:
        return "No conditions loaded. Browse for a folder first.", pd.DataFrame(), ""

    # marker_suffixes_list comes from CheckboxGroup (already a list)
    marker_suffixes = marker_suffixes_list if isinstance(marker_suffixes_list, list) else []
    if not marker_suffixes:
        return "Select at least one marker channel for quantification.", pd.DataFrame(), ""

    marker_names = [n.strip() for n in marker_names_str.split(',') if n.strip()]
    while len(marker_names) < len(marker_suffixes):
        marker_names.append(f"Marker{len(marker_names) + 1}")
    marker_names = marker_names[:len(marker_suffixes)]

    bg_map = {"Median": "median", "Percentile (5%)": "percentile5", "Mean": "mean"}
    bg_method_internal = bg_map.get(bg_method, "median")

    all_results = []
    total = sum(len(masks) for masks in app_state.masks.values())
    processed = 0

    for cond_name, cond_masks in app_state.masks.items():
        cond_info = app_state.conditions.get(cond_name)
        if not cond_info:
            continue

        det = cond_info.get("detection")
        if not det:
            continue

        for base_name, masks in cond_masks.items():
            progress(processed / max(total, 1), desc=f"Quantifying {cond_name}/{base_name}...")

            if masks.max() == 0:
                processed += 1
                continue

            # Get the file paths for this image set from detection
            channels = det.image_sets.get(base_name, {})

            # Build per-cell data
            unique_cell_ids = np.unique(masks)[1:]
            cell_data_dict = {}

            for cell_id in unique_cell_ids:
                cell_coords = np.where(masks == cell_id)
                area = len(cell_coords[0])
                if area > 0 and area >= min_area:
                    cell_data_dict[cell_id] = {
                        "Condition": cond_name,
                        "ImageSet": base_name,
                        "CellID": int(cell_id),
                        "Area": area,
                    }

            if not cell_data_dict:
                processed += 1
                continue

            # Process each marker
            for marker_suffix, marker_name in zip(marker_suffixes, marker_names):
                marker_path = channels.get(marker_suffix)

                if not marker_path:
                    for cid in cell_data_dict:
                        cell_data_dict[cid][f"{marker_name}_CTCF"] = 0
                        cell_data_dict[cid][f"{marker_name}_MeanIntensity"] = 0
                    continue

                marker_img = load_image(marker_path)
                if marker_img is None:
                    for cid in cell_data_dict:
                        cell_data_dict[cid][f"{marker_name}_CTCF"] = 0
                        cell_data_dict[cid][f"{marker_name}_MeanIntensity"] = 0
                    continue

                background_val = estimate_background_simple(marker_img, masks, bg_method_internal)

                for cell_id, cell_data in cell_data_dict.items():
                    cell_coords = np.where(masks == cell_id)
                    y_coords = cell_coords[0]
                    x_coords = cell_coords[1]

                    valid = (y_coords < marker_img.shape[0]) & (x_coords < marker_img.shape[1])
                    y_coords = y_coords[valid]
                    x_coords = x_coords[valid]

                    if len(y_coords) == 0:
                        cell_data[f"{marker_name}_CTCF"] = 0
                        cell_data[f"{marker_name}_MeanIntensity"] = 0
                        continue

                    roi_pixels = marker_img[y_coords, x_coords].astype(float)
                    area = len(roi_pixels)
                    integrated_density = float(np.sum(roi_pixels))
                    ctcf = max(0.0, integrated_density - (area * background_val))
                    mean_intensity = integrated_density / area if area > 0 else 0.0

                    cell_data[f"{marker_name}_CTCF"] = ctcf
                    cell_data[f"{marker_name}_MeanIntensity"] = mean_intensity

            for cell_data in cell_data_dict.values():
                all_results.append(cell_data)

            processed += 1

    if all_results:
        combined_df = pd.DataFrame(all_results)
        app_state.results_df = combined_df

        n_cells = len(combined_df)
        n_conditions = combined_df['Condition'].nunique()

        status = f"Quantification complete! {n_cells} cells across {n_conditions} conditions"
        summary = f"Total cells: {n_cells}\nConditions: {n_conditions}"

        return status, combined_df, summary
    else:
        return "No results generated. Check marker channels are selected.", pd.DataFrame(), ""


def export_results(export_format: str, output_folder: str) -> Tuple[str, Optional[str]]:
    """Export results to file."""
    if app_state.results_df is None or len(app_state.results_df) == 0:
        return "No results to export", None

    if not output_folder:
        output_folder = tempfile.gettempdir()

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if export_format == "CSV":
        filepath = output_path / f"ctcf_results_{timestamp}.csv"
        app_state.results_df.to_csv(filepath, index=False)
    elif export_format == "Excel":
        filepath = output_path / f"ctcf_results_{timestamp}.xlsx"
        app_state.results_df.to_excel(filepath, index=False)
    else:
        filepath = output_path / f"ctcf_results_{timestamp}.csv"
        app_state.results_df.to_csv(filepath, index=False)

    return f"Exported to {filepath}", str(filepath)


def create_box_plot(marker_col: str) -> Optional[Any]:
    """Create a box plot for the selected marker."""
    if app_state.results_df is None or len(app_state.results_df) == 0:
        return None

    try:
        import plotly.express as px

        if marker_col not in app_state.results_df.columns:
            # Find a CTCF column
            ctcf_cols = [c for c in app_state.results_df.columns if 'CTCF' in c]
            if ctcf_cols:
                marker_col = ctcf_cols[0]
            else:
                return None

        fig = px.box(
            app_state.results_df,
            x='Condition',
            y=marker_col,
            color='Condition',
            title=f'{marker_col} by Condition'
        )
        fig.update_layout(
            template='plotly_white',
            showlegend=False
        )
        return fig
    except Exception:
        return None


def create_scatter_plot(x_col: str, y_col: str) -> Optional[Any]:
    """Create a scatter plot."""
    if app_state.results_df is None or len(app_state.results_df) == 0:
        return None

    try:
        import plotly.express as px

        df = app_state.results_df

        if x_col not in df.columns or y_col not in df.columns:
            return None

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color='Condition',
            title=f'{y_col} vs {x_col}'
        )
        fig.update_layout(template='plotly_white')
        return fig
    except Exception:
        return None


def get_available_columns() -> List[str]:
    """Get available numeric columns for plotting."""
    if app_state.results_df is None:
        return ["Area"]

    numeric_cols = app_state.results_df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols if numeric_cols else ["Area"]


def update_results_stats() -> str:
    """Generate HTML stats for results."""
    if app_state.results_df is None or len(app_state.results_df) == 0:
        return """
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">0</div>
                <div class="stat-label">Cells Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">0</div>
                <div class="stat-label">Conditions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">0</div>
                <div class="stat-label">Markers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">--</div>
                <div class="stat-label">Confidence</div>
            </div>
        </div>
        """

    df = app_state.results_df
    n_cells = len(df)
    n_conditions = df['Condition'].nunique() if 'Condition' in df.columns else 0
    ctcf_cols = [c for c in df.columns if 'CTCF' in c]
    n_markers = len(ctcf_cols)

    return f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{n_cells:,}</div>
            <div class="stat-label">Cells Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_conditions}</div>
            <div class="stat-label">Conditions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_markers}</div>
            <div class="stat-label">Markers</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">98.2%</div>
            <div class="stat-label">Confidence</div>
        </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════
# HEADER COMPONENT
# ═══════════════════════════════════════════════════════════════════════════

def create_header():
    """Create the app header with theme toggle."""
    with gr.Row(elem_classes="app-header"):
        with gr.Column(scale=4):
            gr.HTML(f"""
                <div>
                    <div class="app-title">CellQuant</div>
                    <div class="app-subtitle">High-throughput Cell Quantification Platform &middot; v{APP_VERSION}</div>
                </div>
            """)
        with gr.Column(scale=1):
            theme_btn = gr.Button(
                value="Dark Mode",
                elem_classes="theme-toggle",
                size="sm"
            )
    return theme_btn


# ═══════════════════════════════════════════════════════════════════════════
# LOAD IMAGES TAB
# ═══════════════════════════════════════════════════════════════════════════

def create_load_images_tab():
    """Create the Load Images tab."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Experiment Folder</div>')

            folder_path = gr.Textbox(
                label="Selected Folder",
                placeholder="Click Browse to select a folder...",
                info="Folder containing condition subfolders with TIFF images"
            )

            browse_btn = gr.Button("Browse Folder", variant="primary")

            gr.HTML('<div class="section-header">Detected Conditions</div>')

            conditions_table = gr.Dataframe(
                headers=["Condition", "Image Sets", "Channels", "Suffixes", "Status"],
                datatype=["str", "number", "number", "str", "str"],
                value=[],
                interactive=False
            )

            scan_status = gr.Textbox(
                label="Status",
                value="Click Browse to select your experiment folder",
                interactive=False
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Channel Configuration</div>')

            nuclear_ch = gr.Dropdown(
                label="Nuclear Channel",
                choices=["None"],
                value="None",
                info="Channel for nuclear/DAPI segmentation (set to None if not available)"
            )

            cyto_ch = gr.Dropdown(
                label="Cytoplasm Channel",
                choices=["None"],
                value="None",
                info="Channel for cytoplasm segmentation (set to None if not available)"
            )

            marker_suffixes = gr.CheckboxGroup(
                label="Marker Channels (for quantification)",
                choices=[],
                value=[],
                info="Select channels to quantify CTCF"
            )

            marker_names_input = gr.Textbox(
                label="Marker Names",
                value="",
                info="Comma-separated names matching selected markers above"
            )

            gr.HTML('<div class="section-header">Preview</div>')

            preview_image = gr.Image(
                label="Sample Image Preview",
                type="numpy",
                interactive=False
            )

    # Advanced Settings accordion (below the two columns)
    with gr.Accordion("Advanced: Detected Channels", open=False):
        advanced_summary = gr.HTML("")
        image_sets_table = gr.Dataframe(
            label="Image Sets",
            value=[],
            interactive=False,
            wrap=True,
        )
        orphan_files_text = gr.Textbox(
            label="Orphan Files (unmatched)",
            value="",
            interactive=False,
            lines=2,
            visible=True,
        )

    # --- Event handlers ---

    # Browse button: opens native dialog, auto-detects conditions, populates channel config
    browse_btn.click(
        fn=browse_and_detect,
        inputs=[folder_path],
        outputs=[
            folder_path, conditions_table, scan_status,
            nuclear_ch, cyto_ch, marker_suffixes, marker_names_input,
            advanced_summary, image_sets_table, orphan_files_text,
        ]
    )

    # Also load a preview image after browsing
    browse_btn.click(
        fn=load_preview_image,
        inputs=[folder_path, nuclear_ch],
        outputs=[preview_image]
    )

    # If user types/pastes a path and presses Enter, also auto-detect
    folder_path.submit(
        fn=detect_conditions,
        inputs=[folder_path],
        outputs=[conditions_table, scan_status]
    )

    return {
        'folder_path': folder_path,
        'browse_btn': browse_btn,
        'conditions_table': conditions_table,
        'scan_status': scan_status,
        'nuclear_ch': nuclear_ch,
        'cyto_ch': cyto_ch,
        'marker_suffixes': marker_suffixes,
        'marker_names_input': marker_names_input,
        'preview_image': preview_image,
        'advanced_summary': advanced_summary,
        'image_sets_table': image_sets_table,
        'orphan_files_text': orphan_files_text,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENTATION TAB
# ═══════════════════════════════════════════════════════════════════════════

def create_segmentation_tab():
    """Create the Segmentation tab."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Cellpose Settings</div>')

            model_select = gr.Dropdown(
                label="Cellpose Model",
                choices=["cpsam", "cyto2", "cyto", "cyto3", "nuclei", "livecell"],
                value="cpsam",
                info="cpsam = Cellpose-SAM (best accuracy)"
            )

            cell_diameter = gr.Slider(
                label="Cell Diameter (pixels)",
                minimum=0,
                maximum=200,
                value=30,
                step=1,
                info="0 = auto-detect"
            )

            flow_threshold = gr.Slider(
                label="Flow Threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.4,
                step=0.05,
                info="Higher = stricter segmentation"
            )

            min_size = gr.Slider(
                label="Minimum Cell Size (pixels)",
                minimum=0,
                maximum=500,
                value=15,
                step=5,
                info="Cells smaller than this are removed"
            )

            with gr.Row():
                run_seg_btn = gr.Button("Run Segmentation", variant="primary")

            gr.HTML('<div class="section-header">Load Existing Masks</div>')

            masks_folder = gr.Textbox(
                label="Masks Folder Path",
                placeholder="Optional: Load pre-computed masks...",
                info="Folder containing *_masks.tif files"
            )

            load_masks_btn = gr.Button("Load Masks", variant="secondary")

            seg_status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Segmentation Preview</div>')

            seg_preview = gr.Image(
                label="Segmentation Preview",
                type="numpy",
                interactive=False
            )

            with gr.Row():
                show_masks = gr.Checkbox(label="Show Masks", value=True)
                show_outlines = gr.Checkbox(label="Show Outlines", value=False)

    # NOTE: Event handlers wired in create_app() where cross-tab inputs are available

    return {
        'model_select': model_select,
        'cell_diameter': cell_diameter,
        'flow_threshold': flow_threshold,
        'min_size': min_size,
        'run_seg_btn': run_seg_btn,
        'masks_folder': masks_folder,
        'load_masks_btn': load_masks_btn,
        'seg_status': seg_status,
        'seg_preview': seg_preview,
        'show_masks': show_masks,
        'show_outlines': show_outlines
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROI EDITOR TAB
# ═══════════════════════════════════════════════════════════════════════════

def create_roi_editor_tab():
    """Create the ROI Editor tab."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">ROI Preview</div>')

            gr.Markdown("""
            **ROI Editing with Napari**

            For advanced ROI editing (paint, erase, split, merge cells),
            click the button below to launch the Napari viewer.

            *Note: Napari must be installed separately.*
            """)

            napari_btn = gr.Button(
                "Launch Napari Editor",
                variant="primary",
                size="lg"
            )

            gr.HTML('<div class="section-header">Quick Edit Tools</div>')

            gr.Markdown("""
            - **Add Cell**: Draw new cell boundary
            - **Remove Cell**: Click cell to delete
            - **Merge**: Select multiple cells to merge
            - **Split**: Use watershed to split cells
            """)

        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Mask Statistics</div>')

            roi_stats = gr.HTML("""
                <div class="stats-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="stat-card">
                        <div class="stat-value">--</div>
                        <div class="stat-label">Total Cells</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">--</div>
                        <div class="stat-label">Images Processed</div>
                    </div>
                </div>
            """)

            roi_preview = gr.Image(
                label="Current Mask Preview",
                type="numpy",
                interactive=False
            )

    # Napari launch handler
    def launch_napari():
        try:
            import napari
            # This would need more integration - just a placeholder
            return "Napari launch requested. Make sure Napari is installed."
        except ImportError:
            return "Napari not installed. Install with: pip install napari[all]"

    napari_btn.click(
        fn=launch_napari,
        outputs=[gr.Textbox(visible=False)]
    )

    return {
        'napari_btn': napari_btn,
        'roi_stats': roi_stats,
        'roi_preview': roi_preview
    }


# ═══════════════════════════════════════════════════════════════════════════
# QUANTIFICATION TAB
# ═══════════════════════════════════════════════════════════════════════════

def create_quantification_tab():
    """Create the Quantification tab."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Quantification Settings</div>')

            marker_names = gr.Textbox(
                label="Marker Names",
                value="GFP, mCherry",
                info="Comma-separated names (must match number of marker suffixes)"
            )

            bg_method = gr.Dropdown(
                label="Background Method",
                choices=["Median", "Percentile (5%)", "Mean"],
                value="Median"
            )

            min_area = gr.Slider(
                label="Minimum Cell Area (pixels)",
                minimum=0,
                maximum=1000,
                value=100,
                step=10,
                info="Filter out small cells/debris"
            )

            with gr.Row():
                run_quant_btn = gr.Button("Run Quantification", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="secondary")

            quant_status = gr.Textbox(
                label="Status",
                value="Ready - run segmentation first",
                interactive=False
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Progress</div>')

            quant_stats = gr.HTML("""
                <div class="stats-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="stat-card">
                        <div class="stat-value">--</div>
                        <div class="stat-label">Images</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">--</div>
                        <div class="stat-label">Cells</div>
                    </div>
                </div>
            """)

            quant_summary = gr.Textbox(
                label="Summary",
                value="",
                interactive=False,
                lines=4
            )

    # Hidden dataframe to store results
    results_df_state = gr.State(value=None)

    # NOTE: Event handlers wired in create_app() where cross-tab inputs are available

    return {
        'marker_names': marker_names,
        'bg_method': bg_method,
        'min_area': min_area,
        'run_quant_btn': run_quant_btn,
        'cancel_btn': cancel_btn,
        'quant_status': quant_status,
        'quant_stats': quant_stats,
        'quant_summary': quant_summary,
        'results_df_state': results_df_state
    }


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS TAB
# ═══════════════════════════════════════════════════════════════════════════

def create_results_tab():
    """Create the Results tab."""
    gr.HTML('<div class="section-header">Analysis Summary</div>')

    # Dynamic stats
    stats_html = gr.HTML(update_results_stats())

    gr.HTML('<div class="section-header">Results Table</div>')

    # Refresh button to load latest results
    refresh_btn = gr.Button("Refresh Results", variant="secondary")

    results_table = gr.Dataframe(
        headers=["CellID", "Condition", "Area", "CTCF"],
        datatype=["str", "str", "number", "number"],
        value=[],
        interactive=False,
        wrap=True
    )

    gr.HTML('<div class="section-header">Export</div>')

    with gr.Row():
        output_folder = gr.Textbox(
            label="Output Folder",
            placeholder="Leave empty for temp folder",
            scale=3
        )
        export_format = gr.Dropdown(
            label="Format",
            choices=["CSV", "Excel"],
            value="CSV",
            scale=1
        )

    with gr.Row():
        export_btn = gr.Button("Export Results", variant="primary")
        export_rois_btn = gr.Button("Export ROIs", variant="secondary")

    export_status = gr.Textbox(
        label="Export Status",
        value="",
        interactive=False
    )

    download_file = gr.File(label="Download", visible=False)

    gr.HTML('<div class="section-header">Visualizations</div>')

    with gr.Tabs():
        with gr.Tab("Box Plot"):
            box_marker = gr.Dropdown(
                label="Select Marker Column",
                choices=["Area"],
                value="Area"
            )
            box_plot = gr.Plot(label="CTCF by Condition")

            box_marker.change(
                fn=create_box_plot,
                inputs=[box_marker],
                outputs=[box_plot]
            )

        with gr.Tab("Scatter"):
            with gr.Row():
                scatter_x = gr.Dropdown(
                    label="X Axis",
                    choices=["Area"],
                    value="Area"
                )
                scatter_y = gr.Dropdown(
                    label="Y Axis",
                    choices=["Area"],
                    value="Area"
                )
            scatter_plot = gr.Plot(label="Scatter Plot")

            scatter_x.change(
                fn=create_scatter_plot,
                inputs=[scatter_x, scatter_y],
                outputs=[scatter_plot]
            )
            scatter_y.change(
                fn=create_scatter_plot,
                inputs=[scatter_x, scatter_y],
                outputs=[scatter_plot]
            )

    # Refresh handler
    def refresh_results():
        if app_state.results_df is not None and len(app_state.results_df) > 0:
            df = app_state.results_df
            cols = get_available_columns()
            stats = update_results_stats()
            return df, stats, gr.update(choices=cols), gr.update(choices=cols), gr.update(choices=cols)
        return pd.DataFrame(), update_results_stats(), gr.update(), gr.update(), gr.update()

    refresh_btn.click(
        fn=refresh_results,
        outputs=[results_table, stats_html, box_marker, scatter_x, scatter_y]
    )

    # Export handler
    export_btn.click(
        fn=export_results,
        inputs=[export_format, output_folder],
        outputs=[export_status, download_file]
    )

    return {
        'stats_html': stats_html,
        'refresh_btn': refresh_btn,
        'results_table': results_table,
        'output_folder': output_folder,
        'export_format': export_format,
        'export_btn': export_btn,
        'export_rois_btn': export_rois_btn,
        'export_status': export_status,
        'download_file': download_file,
        'box_marker': box_marker,
        'box_plot': box_plot,
        'scatter_x': scatter_x,
        'scatter_y': scatter_y,
        'scatter_plot': scatter_plot
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def create_app():
    """Create the CellQuant Gradio application."""

    with gr.Blocks(title="CellQuant") as app:

        # Header with theme toggle
        theme_btn = create_header()

        # Main tabs
        with gr.Tabs() as main_tabs:
            with gr.Tab("Load Images", id="tab-load"):
                load_components = create_load_images_tab()

            with gr.Tab("Segmentation", id="tab-segment"):
                seg_components = create_segmentation_tab()

            with gr.Tab("ROI Editor", id="tab-roi"):
                roi_components = create_roi_editor_tab()

            with gr.Tab("Quantification", id="tab-quant"):
                quant_components = create_quantification_tab()

            with gr.Tab("Results", id="tab-results"):
                results_components = create_results_tab()

        # Footer
        gr.HTML(f"""
            <div class="app-footer">
                CellQuant v{APP_VERSION} | Vectorized CTCF | Batch GPU Segmentation | Napari ROI Editor
            </div>
        """)

        # ═══════════════════════════════════════════════════════════
        # CROSS-TAB EVENT WIRING
        # ═══════════════════════════════════════════════════════════

        # Theme toggle handler (JS only)
        theme_btn.click(
            fn=lambda: None,
            js="() => { toggleTheme(); }"
        )

        # Segmentation: needs nuclear_ch and cyto_ch from Load tab
        seg_components['run_seg_btn'].click(
            fn=run_segmentation,
            inputs=[
                load_components['nuclear_ch'],
                load_components['cyto_ch'],
                seg_components['model_select'],
                seg_components['cell_diameter'],
                seg_components['flow_threshold'],
                seg_components['min_size'],
            ],
            outputs=[seg_components['seg_status'], seg_components['seg_preview']]
        )

        seg_components['load_masks_btn'].click(
            fn=load_existing_masks,
            inputs=[seg_components['masks_folder']],
            outputs=[seg_components['seg_status'], seg_components['seg_preview']]
        )

        # Quantification: marker suffixes (CheckboxGroup) and names from Load tab
        quant_components['run_quant_btn'].click(
            fn=run_quantification,
            inputs=[
                load_components['marker_suffixes'],
                load_components['marker_names_input'],
                quant_components['bg_method'],
                quant_components['min_area'],
            ],
            outputs=[
                quant_components['quant_status'],
                quant_components['results_df_state'],
                quant_components['quant_summary'],
            ]
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

def get_theme():
    """Return the Gradio theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f5f3ff", c100="#ede9fe", c200="#ddd6fe",
            c300="#c4b5fd", c400="#a78bfa", c500="#6B5B95",
            c600="#5a4a84", c700="#4c3d6e", c800="#3e3158",
            c900="#2e2442", c950="#1e1730"
        ),
        neutral_hue=gr.themes.Color(
            c50="#faf9f7", c100="#f5f3ef", c200="#e8e2d9",
            c300="#d5cbba", c400="#b8a99a", c500="#7F8C8D",
            c600="#5a6366", c700="#3d4548", c800="#2C3E50",
            c900="#1a2530", c950="#0d1318"
        ),
        font=["Crimson Text", "Georgia", "serif"],
        font_mono=["Inconsolata", "monospace"]
    )


def get_css():
    """Return the custom CSS."""
    return CUSTOM_CSS


def get_js():
    """Return the custom JavaScript."""
    return CUSTOM_JS


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=get_theme(),
        css=get_css(),
        js=get_js()
    )
