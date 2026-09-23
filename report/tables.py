"""Overview table (TTS/collisions/TWT/TDT/RMSE, static vs RL, with % difference) mirroring the figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from macroscopic.logger import LOGGER
from report.data_loader import ReportConfig, ScenarioConfig

Results = dict[str, dict[str, dict[str, Any]]]

# (result_key, display_name, source) — source selects how to pull {mean,lower,upper} out of an aggregate dict.
_SCALAR_METRICS = [
    ("total_time_spent", "TTS (veh\u00b7h)", "summary"),
    ("total_collisions", "Collisions (count)", "summary"),
    ("total_waiting", "TWT (veh\u00b7h)", "summary"),
    ("total_distance_travelled", "TDT (veh\u00b7km)", "summary"),
    ("density", "RMSE density 1-step (veh/km)", "rmse_summary"),
    ("speed", "RMSE speed 1-step (km/h)", "rmse_summary"),
    ("rmse_rho", "RMSE density full-horizon (veh/km)", "rmse_timeseries"),
    ("rmse_v", "RMSE speed full-horizon (km/h)", "rmse_timeseries"),
]


def _extract(agg: dict[str, Any], key: str, source: str) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for one metric, or NaNs if unavailable in this aggregate."""
    if source == "summary":
        v = agg["summary"][key]
        return v["mean"], v["lower"], v["upper"]
    if source == "rmse_summary":
        rs = agg.get("rmse_summary")
        if rs is None or key not in rs:
            return np.nan, np.nan, np.nan
        v = rs[key]
        return float(np.nanmean(v["mean"])), float(np.nanmean(v["lower"])), float(np.nanmean(v["upper"]))
    if source == "rmse_timeseries":
        if key not in agg:
            return np.nan, np.nan, np.nan
        v = agg[key]
        return float(np.nanmean(v["mean"])), float(np.nanmean(v["lower"])), float(np.nanmean(v["upper"]))
    raise ValueError(f"Unknown metric source: {source}")


def build_overview_table(results: Results, scenarios: list[ScenarioConfig], cfg: ReportConfig) -> pd.DataFrame:
    """One row per scenario x metric: static value, RL value, and %-difference (RL vs static)."""
    LOGGER.info("Building overview table")
    rows = []
    for sc in scenarios:
        static_agg = results[sc.id]["static"]
        rl_agg = results[sc.id]["rl"]
        for key, display_name, source in _SCALAR_METRICS:
            s_mean, s_lo, s_hi = _extract(static_agg, key, source)
            r_mean, r_lo, r_hi = _extract(rl_agg, key, source)
            pct_diff = (r_mean - s_mean) / s_mean * 100.0 if s_mean and not np.isnan(s_mean) else np.nan
            rows.append({
                "Scenario": sc.label,
                "Metric": display_name,
                "Static (mean)": s_mean,
                "Static (95% CI)": f"[{s_lo:.2f}, {s_hi:.2f}]" if not np.isnan(s_lo) else "n/a",
                "RL (mean)": r_mean,
                "RL (95% CI)": f"[{r_lo:.2f}, {r_hi:.2f}]" if not np.isnan(r_lo) else "n/a",
                "% Diff (RL vs static)": pct_diff,
            })
    df = pd.DataFrame(rows)
    for col in ("Static (mean)", "RL (mean)", "% Diff (RL vs static)"):
        df[col] = df[col].round(3)
    return df


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    """Minimal GFM table writer (avoids depending on the optional `tabulate` package)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table(df: pd.DataFrame, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_no_ext.with_suffix(".csv"), index=False)
    _write_markdown(df, path_no_ext.with_suffix(".md"))
    df.to_latex(path_no_ext.with_suffix(".tex"), index=False, float_format="%.2f", longtable=True)
    LOGGER.info(f"Wrote overview table to {path_no_ext}.{{csv,md,tex}}")


def build_q_table_frame(
    q_table: dict[tuple, Any],
    state_order: list[tuple[str, ...]],
    action_values: list[float],
    episode: int,
) -> pd.DataFrame:
    """Dump the shared mixed-pool Q-table snapshot: one row per state, one column per action + best action."""
    rows = []
    for key in state_order:
        q_values = q_table.get(key)
        row: dict[str, Any] = {"State": "|".join(key), "Episode": episode}
        values = q_values if q_values is not None else [np.nan] * len(action_values)
        for a, q in zip(action_values, values):
            row[f"Q(alpha={a:g})"] = round(float(q), 4) if not np.isnan(q) else np.nan
        row["Best alpha"] = action_values[int(np.nanargmax(q_values))] if q_values is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
