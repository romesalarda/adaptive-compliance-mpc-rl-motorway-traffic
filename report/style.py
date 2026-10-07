"""Matplotlib style helpers targeting IEEE double-column paper conventions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# IEEE column widths (inches), per the standard two-column template.
SINGLE_COL_WIDTH = 3.45
DOUBLE_COL_WIDTH = 7.16

IEEE_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Consistent colour per scenario alias and linestyle per controller, reused across all figures.
SCENARIO_COLORS = {"scenario_1": "#1b9e77", "scenario_2": "#d95f02", "scenario_3": "#7570b3"}
CONTROLLER_STYLES = {"static": {"linestyle": "-", "hatch": None}, "rl": {"linestyle": "--", "hatch": "//"}}


def apply_ieee_style() -> None:
    plt.rcParams.update(IEEE_RCPARAMS)


def save_figure(fig: Figure, path_no_ext: Path, formats: list[str], dpi: int) -> None:
    """Save one figure under multiple formats (e.g. pdf + png) sharing the same base filename."""
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = path_no_ext.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, format=fmt)
