"""Figure builders for the IEEE report: demand profiles, TTS/collisions, RMSE, RL alpha selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from macroscopic.logger import LOGGER
from report.data_loader import ReportConfig, ScenarioConfig
from report.style import (
    CONTROLLER_STYLES,
    DOUBLE_COL_WIDTH,
    SCENARIO_COLORS,
    SINGLE_COL_WIDTH,
    apply_ieee_style,
    save_figure,
)

# results[scenario.id] = {"static": <RunFolderBootstrapper.aggregate() dict>, "rl": <...>}
Results = dict[str, dict[str, dict[str, Any]]]


def _bar_errors(means: list[float], los: list[float], his: list[float]) -> list[list[float]]:
    return [[m - l for m, l in zip(means, los)], [h - m for h, m in zip(his, means)]]


def plot_demand_profiles(
    scenarios: list[ScenarioConfig],
    demand_profiles: dict[str, dict[str, Any]],
    cfg: ReportConfig,
) -> None:
    """Single-column figure, one stacked subplot per scenario, showing mainline + on-ramp demand."""
    LOGGER.info("Building figure: demand profiles")
    apply_ieee_style()
    fig, axes = plt.subplots(
        len(scenarios), 1, figsize=(SINGLE_COL_WIDTH, 1.35 * len(scenarios)), sharex=True
    )
    for ax, sc in zip(axes, scenarios):
        demand = demand_profiles[sc.id]
        xp, fp = demand["xp"], demand["fp"]
        color = SCENARIO_COLORS[sc.id]
        ax.plot(xp, fp[0], marker="o", ms=2.5, color=color, linestyle="-", label="Mainline")
        ax.plot(xp, fp[1], marker="s", ms=2.5, color=color, linestyle="--", label="On-ramp")
        ax.set_ylabel("Demand\n(veh/h)")
        ax.set_title(f"{sc.label} ({demand.get('descriptor', '')})", loc="left", fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[-1].set_xlabel("Time (h)")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / "demand_profiles", cfg.formats, cfg.dpi)
    plt.close(fig)


def plot_tts_collisions(results: Results, scenarios: list[ScenarioConfig], cfg: ReportConfig) -> None:
    """Merged figure: TTS and collision-count comparisons as two side-by-side grouped bar panels."""
    LOGGER.info("Building figure: TTS + collisions comparison")
    apply_ieee_style()
    fig, (ax_tts, ax_col) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.4))
    x = np.arange(len(scenarios))
    width = 0.35
    labels = [sc.label for sc in scenarios]

    for ax, key, ylabel in (
        (ax_tts, "total_time_spent", "TTS (veh\u00b7h)"),
        (ax_col, "total_collisions", "Collisions (count)"),
    ):
        static_summary = [results[sc.id]["static"]["summary"][key] for sc in scenarios]
        rl_summary = [results[sc.id]["rl"]["summary"][key] for sc in scenarios]
        static_mean = [s["mean"] for s in static_summary]
        rl_mean = [s["mean"] for s in rl_summary]
        ax.bar(
            x - width / 2, static_mean, width,
            yerr=_bar_errors(static_mean, [s["lower"] for s in static_summary], [s["upper"] for s in static_summary]),
            capsize=2, label=r"Static $\alpha$=0.01", color="#999999",
        )
        ax.bar(
            x + width / 2, rl_mean, width,
            yerr=_bar_errors(rl_mean, [s["lower"] for s in rl_summary], [s["upper"] for s in rl_summary]),
            capsize=2, label="RL", color="#4477aa",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # headroom so the shared legend never overlaps a bar
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    handles, legend_labels = ax_tts.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / "tts_collisions", cfg.formats, cfg.dpi)
    plt.close(fig)


def plot_rmse(metric: str, results: Results, scenarios: list[ScenarioConfig], cfg: ReportConfig) -> None:
    """Two-panel figure for one metric ('density' or 'speed'): 1-step (bar) + full-horizon (line) RMSE."""
    LOGGER.info(f"Building figure: RMSE ({metric})")
    label_map = {"density": "Density RMSE (veh/km)", "speed": "Speed RMSE (km/h)"}
    ts_key = {"density": "rmse_rho", "speed": "rmse_v"}[metric]

    apply_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.4))
    x = np.arange(len(scenarios))
    width = 0.35

    # --- panel 1: 1-step-ahead RMSE, averaged across segments, static vs RL ---
    for offset, controller, color, label in (
        (-width / 2, "static", "#999999", r"Static $\alpha$=0.01"),
        (width / 2, "rl", "#4477aa", "RL"),
    ):
        means, los, his = [], [], []
        for sc in scenarios:
            rmse_summary = results[sc.id][controller].get("rmse_summary")
            if rmse_summary is None or metric not in rmse_summary:
                LOGGER.warning(f"No 1-step RMSE data for {sc.id}/{controller}; plotting as gap")
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            arr = rmse_summary[metric]
            means.append(float(np.nanmean(arr["mean"])))
            los.append(float(np.nanmean(arr["lower"])))
            his.append(float(np.nanmean(arr["upper"])))
        ax1.bar(x + offset, means, width, yerr=_bar_errors(means, los, his), capsize=2, color=color, label=label)
    ax1.set_xticks(x)
    ax1.set_xticklabels([sc.label for sc in scenarios])
    ax1.set_ylabel(f"1-step {label_map[metric]}")
    ax1.set_title("1-step-ahead", fontsize=8)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax1.legend(frameon=False)

    # --- panel 2: full-horizon RMSE (mean over Np prediction steps) vs cycle time ---
    for sc in scenarios:
        color = SCENARIO_COLORS[sc.id]
        for controller in ("static", "rl"):
            agg = results[sc.id][controller]
            if ts_key not in agg:
                LOGGER.warning(f"No full-horizon RMSE data for {sc.id}/{controller}; skipping line")
                continue
            controller_label = r"Static $\alpha$=0.01" if controller == "static" else "RL"
            ax2.plot(
                agg["rmse_cycle_t"], agg[ts_key]["mean"],
                color=color, linestyle=CONTROLLER_STYLES[controller]["linestyle"],
                label=f"{sc.label} ({controller_label})",
            )
    ax2.set_xlabel("Simulation time (s)")
    ax2.set_ylabel(f"Full-horizon {label_map[metric]}")
    ax2.set_title("Full-horizon (mean over prediction steps)", fontsize=8)
    ax2.grid(alpha=0.3, linewidth=0.4)
    ax2.legend(frameon=False, fontsize=5.5, ncol=1)

    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / f"rmse_{metric}", cfg.formats, cfg.dpi)
    plt.close(fig)


def plot_rmse_full_horizon_combined(results: Results, scenarios: list[ScenarioConfig], cfg: ReportConfig) -> None:
    """Single figure, density + speed full-horizon RMSE side by side (1-step is covered by the overview table)."""
    LOGGER.info("Building figure: full-horizon RMSE (density + speed combined)")
    label_map = {"density": "Density RMSE (veh/km)", "speed": "Speed RMSE (km/h)"}
    ts_key_map = {"density": "rmse_rho", "speed": "rmse_v"}

    apply_ieee_style()
    fig, (ax_density, ax_speed) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.4))

    for ax, metric in ((ax_density, "density"), (ax_speed, "speed")):
        ts_key = ts_key_map[metric]
        for sc in scenarios:
            color = SCENARIO_COLORS[sc.id]
            for controller in ("static", "rl"):
                agg = results[sc.id][controller]
                if ts_key not in agg:
                    LOGGER.warning(f"No full-horizon RMSE data for {sc.id}/{controller}; skipping line")
                    continue
                controller_label = r"Static $\alpha$=0.01" if controller == "static" else "RL"
                ax.plot(
                    agg["rmse_cycle_t"], agg[ts_key]["mean"],
                    color=color, linestyle=CONTROLLER_STYLES[controller]["linestyle"],
                    label=f"{sc.label} ({controller_label})",
                )
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel(f"Full-horizon {label_map[metric]}")
        ax.set_title(f"{label_map[metric].split(' RMSE')[0]}", fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.4)

    handles, legend_labels = ax_density.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / "rmse_full_horizon_combined", cfg.formats, cfg.dpi)
    plt.close(fig)


def _count_for_alpha(counts: dict[str, int], alpha: float) -> int:
    return sum(n for k, n in counts.items() if abs(float(k) - alpha) < 1e-9)

def plot_rl_alpha_selection(
    action_counts: dict[str, dict[str, int]],
    scenarios: list[ScenarioConfig],
    action_values: list[float],
    static_alpha: float,
    cfg: ReportConfig,
) -> None:
    """Compact single-panel figure: RL alpha-selection frequency (%) grouped by scenario."""
    LOGGER.info("Building figure: RL alpha-selection frequency")
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.2))

    n_scen = len(scenarios)
    width = 0.8 / n_scen
    x_base = np.arange(len(action_values))
    for i, sc in enumerate(scenarios):
        counts = action_counts[sc.id]
        total = sum(counts.values()) or 1
        freqs = [100.0 * _count_for_alpha(counts, a) / total for a in action_values]
        offset = (i - (n_scen - 1) / 2) * width
        ax.bar(x_base + offset, freqs, width, color=SCENARIO_COLORS[sc.id], label=sc.label)

    static_idx = min(range(len(action_values)), key=lambda i: abs(action_values[i] - static_alpha))
    ax.axvspan(static_idx - 0.45, static_idx + 0.45, color="red", alpha=0.08, zorder=0)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{a:g}" for a in action_values])
    ax.set_xlabel(r"VSL compliance $\alpha$ chosen by RL agent")
    ax.set_ylabel("Selection frequency (%)")
    ax.annotate(
        "static baseline", xy=(static_idx, ax.get_ylim()[1]), xytext=(static_idx, ax.get_ylim()[1] * 1.02),
        ha="center", va="bottom", fontsize=6, color="red", annotation_clip=False,
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / "rl_alpha_selection", cfg.formats, cfg.dpi)
    plt.close(fig)


def plot_q_table_heatmap(
    q_table: dict[tuple, np.ndarray],
    state_order: list[tuple[str, ...]],
    action_values: list[float],
    episode: int,
    cfg: ReportConfig,
) -> None:
    """Heatmap of the shared mixed-pool Q-table snapshot (one table trained jointly across all 3 scenarios)."""
    LOGGER.info(f"Building figure: Q-table heatmap (episode {episode})")
    apply_ieee_style()
    n_states, n_actions = len(state_order), len(action_values)
    matrix = np.full((n_states, n_actions), np.nan)
    for i, key in enumerate(state_order):
        if key in q_table:
            matrix[i, : len(q_table[key])] = q_table[key]
        else:
            LOGGER.warning(f"State {key} missing from Q-table snapshot (never visited during training)")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 0.4 * n_states + 1.2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels([f"{a:g}" for a in action_values])
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(["|".join(k) for k in state_order])
    ax.set_xlabel(r"Action ($\alpha$)")
    ax.set_ylabel("State (density|TTS trend)")
    ax.set_title(f"Learned Q-table (episode {episode})", fontsize=8)
    for i in range(n_states):
        for j in range(n_actions):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, label="Q-value", shrink=0.8)
    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / "q_table_heatmap", cfg.formats, cfg.dpi)
    plt.close(fig)
