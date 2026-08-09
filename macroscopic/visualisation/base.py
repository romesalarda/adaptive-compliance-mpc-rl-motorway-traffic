"""
Visualisation utilities for METANET traffic simulation results.

Replaces the deprecated free-standing `plot_dynamics_results`, `plot_rm_results`,
`plot_coordinated_control_results` and `plot_rm_vsl_results` functions with a single
cohesive `METANETVisualizer` class, backed by small typed dataclasses instead of
long positional argument lists.

Usage
-----
    params  = NetworkParams(T=10/3600, L=0.5, lanes=3, rho_crit=33.5, v_min=40, v_max=120)
    metrics = SimulationMetrics(tts=123.4, total_wait_time=12.3, total_distance=4567.8,
                                 congestion_time=0.42, excess_congestion=3.1,
                                 total_speed_variance=210.0, computation_time=1.8)
    results = SimulationResults(time=t, RHO=RHO, V=V, W=W, Q=Q, Q_o=Q_o,
                                 demands=demands, Tfin=Tfin, R=R)

    viz = METANETVisualizer()
    fig = viz.plot_ramp_metering(results, params, metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from macroscopic.statistics.base import MacroOverviewSimulationResults, MacroSimulationResults
from macroscopic.simulation.parameters import BaseParameters, VslParameters
from macroscopic.simulation.environment import MacroEnvironment

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Visualizer
# --------------------------------------------------------------------------- #

class MacroscopicVisualiser:
    """Cohesive plotting utility for METANET ramp-metering / VSL simulation results.

    One instance can render any of the four standard result layouts
    (uncontrolled dynamics, ramp metering only, coordinated RM+VSL, RM+VSL
    with explicit VSL rate panel) by sharing the same styling, colour
    palettes and AUC-annotated legend logic.
    """

    ORIGIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    METERING_COLOR = "#d62728"

    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize_scale: float = 1.0, dpi: int = 100):
        self.style = style
        self.figsize_scale = figsize_scale
        self.dpi = dpi

    # -- internal helpers ---------------------------------------------------

    def _figure(self, nrows: int, ncols: int, figsize: tuple, title: str):
        plt.style.use(self.style)
        fig, axs = plt.subplots(
            nrows, ncols, constrained_layout=True, sharex=True,
            figsize=(figsize[0] * self.figsize_scale, figsize[1] * self.figsize_scale),
            dpi=self.dpi,
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig, axs

    @staticmethod
    def _auc(y: np.ndarray, time: np.ndarray) -> float:
        return float(np.trapezoid(y, time))

    @staticmethod
    def _segment_names(n: int) -> list[str]:
        return [f"Seg {i + 1}" for i in range(n)]

    def _segment_colors(self, n: int):
        return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    def _vsl_colors(self, n: int):
        return plt.cm.viridis(np.linspace(0, 1, max(n, 1)))

    def _plot_series(self, ax: Axes, time, data, colors, names, ylabel, title,
                      step: bool = False, fill: bool = True, fill_base: float = 0.0,
                      alpha: float = 0.12, ncol: int = 1, linewidth: float = 1.5) -> Axes:
        """Plot one or more series on a single axis, annotating each legend
        entry with its area-under-curve (AUC), mirroring the original
        functions' behaviour but without repeating the loop everywhere."""
        data2 = data if data.ndim > 1 else data.reshape(-1, 1)
        n = data2.shape[1]
        for i in range(n):
            y = data2[:, i]
            auc = self._auc(y, time)
            auc_str = f"{auc:.2f}" if abs(auc) < 10 else f"{auc:.1f}"
            label = f"{names[i]} (AUC={auc_str})"
            if step:
                ax.step(time, y, where="post", color=colors[i], linewidth=linewidth, label=label)
                ax.fill_between(time, fill_base, y, step="post", color=colors[i], alpha=alpha)
            else:
                ax.plot(time, y, color=colors[i], linewidth=linewidth, label=label)
                if fill:
                    ax.fill_between(time, fill_base, y, color=colors[i], alpha=alpha)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.legend(loc="best", fontsize=8, ncol=ncol)
        ax.grid(True, alpha=0.3)
        return ax

    def _density_panel(self, ax: Axes, results: MacroSimulationResults, params: BaseParameters,
                        seg_colors, seg_names) -> None:
        self._plot_series(ax, params.time, results.rho, seg_colors, seg_names,
                           "Density [veh/km/lane]", "Density Profile", ncol=2)
        ax.axhline(y=params.rho_crit, color="r", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Critical ρ={params.rho_crit}")
        ax.legend(loc="best", fontsize=8, ncol=2)

    def _vsl_panel(self, ax: Axes, results: MacroSimulationResults, params: VslParameters,
                   vsl_colors, vsl_names) -> None:
        self._plot_series(ax, params.time, results.v_ctrl, vsl_colors, vsl_names,
                           "VSL Speed [km/h]", "Variable Speed Limits",
                           step=True, fill_base=params.minimum_speed_limit, alpha=0.15)
        ax.axhline(y=params.minimum_speed_limit, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(y=params.maximum_speed_limit, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_ylim(params.minimum_speed_limit - 5, params.maximum_speed_limit + 5)

    def _metering_panel(self, ax: Axes, results: MacroSimulationResults, params: BaseParameters) -> None:
        if results.r is None or results.r.size == 0:
            ax.set_title("Ramp Metering Control", fontweight="bold", fontsize=10)
            ax.text(0.5, 0.5, "No ramp metering data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xlabel("Time [h]", fontweight="bold")
            return
        r = results.r if results.r.ndim > 1 else results.r.reshape(-1, 1)
        n_ramps = r.shape[1]
        colors = [self.METERING_COLOR] if n_ramps == 1 else self._vsl_colors(n_ramps)
        names = ["Metering Rate"] if n_ramps == 1 else [f"Ramp {i+1}" for i in range(n_ramps)]
        self._plot_series(ax, params.time, r, colors, names,
                        "Metering Rate", "Ramp Metering Control",
                        step=True, alpha=0.15, linewidth=2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time [h]", fontweight="bold")

    def _metrics_box(self, fig: Figure, metrics: MacroOverviewSimulationResults, include_computation: bool = True) -> None:
        fig.text(0.5, 0.01, f"Metrics: {metrics.as_text(include_computation)}",
                  ha="center", fontsize=8.5,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", edgecolor="gray", alpha=0.8))

    def _palette(self, results: MacroSimulationResults):
        seg_colors = self._segment_colors(results.v.shape[1])
        seg_names = self._segment_names(results.v.shape[1])
        orig_colors = self.ORIGIN_COLORS[:results.w.shape[1]]
        orig_names = list(results.origin_names)
        return seg_colors, seg_names, orig_colors, orig_names

    @staticmethod
    def _finish(fig: Figure, axs, results: MacroSimulationResults, show: bool) -> Figure:
        axs.flat[0].set_xlim(0, results.parameters.Tfin)
        if show:
            plt.show(block=True)
        return fig

    # -- public API -----------------------------------------------------------

    def plot_dynamics(self, results: MacroSimulationResults, params: BaseParameters,
                       metrics: MacroOverviewSimulationResults, show: bool = False) -> Figure:
        """Plot uncontrolled network dynamics (no active control)."""
        fig, axs = self._figure(3, 2, (14, 8), "Network Dynamics - Natural Traffic Flow (No Active Control)")
        seg_colors, seg_names, orig_colors, orig_names = self._palette(results)

        self._plot_series(axs[0, 0], params.time, results.v, seg_colors, seg_names,
                           "Speed [km/h]", "Speed Profile", ncol=2)
        self._plot_series(axs[0, 1], params.time, results.q, seg_colors, seg_names,
                           "Flow [veh/h]", "Flow Profile", ncol=2)
        self._density_panel(axs[1, 0], results, params, seg_colors, seg_names)
        self._plot_series(axs[1, 1], params.time, results.demands, orig_colors, orig_names,
                           "Origin Demands [veh/h]", "Origin Demands", linewidth=2, alpha=0.15)
        self._plot_series(axs[2, 0], params.time, results.q_on, orig_colors, orig_names,
                           "Origin Flow [veh/h]", "Origin Flow", linewidth=2, alpha=0.15)
        axs[2, 0].set_xlabel("Time [h]", fontweight="bold")
        self._plot_series(axs[2, 1], params.time, results.w, orig_colors, orig_names,
                           "Queue [veh]", "Queue Length", linewidth=2, alpha=0.15)
        axs[2, 1].set_xlabel("Time [h]", fontweight="bold")

        self._metrics_box(fig, metrics, include_computation=False)
        return self._finish(fig, axs, results, show)

    def plot_ramp_metering(self, results: MacroSimulationResults, params: BaseParameters,
                            metrics: MacroOverviewSimulationResults, show: bool = False) -> Figure:
        """Plot MPC ramp-metering-only control results."""
        if results.R is None:
            raise ValueError("results.R (metering rate) is required for the ramp-metering plot")
        fig, axs = self._figure(4, 2, (14, 10), "MPC Ramp Metering Control - Traffic System Behavior")
        seg_colors, seg_names, orig_colors, orig_names = self._palette(results)

        self._plot_series(axs[0, 0], params.time, results.v, seg_colors, seg_names,
                           "Speed [km/h]", "Speed Profile", ncol=2)
        self._plot_series(axs[0, 1], params.time, results.q, seg_colors, seg_names,
                           "Flow [veh/h]", "Flow Profile", ncol=2)
        self._density_panel(axs[1, 0], results, params, seg_colors, seg_names)
        self._plot_series(axs[1, 1], params.time, results.demands, orig_colors, orig_names,
                           "Origin Demands [veh/h]", "Origin Demands", linewidth=2, alpha=0.15)
        self._plot_series(axs[2, 0], params.time, results.q_on, orig_colors, orig_names,
                           "Origin Flow [veh/h]", "Origin Flow", linewidth=2, alpha=0.15)
        self._plot_series(axs[2, 1], params.time, results.w, orig_colors, orig_names,
                           "Queue [veh]", "Queue Length", linewidth=2, alpha=0.15)
        self._metering_panel(axs[3, 0], results, params)

        summary_lines = [
            "Performance Metrics:", "━━━━━━━━━━━━━━━━━━━━",
            f"TTS: {metrics.total_time_spent:.2f} veh·h",
            f"Wait Time: {metrics.total_wait_time:.2f} veh·h",
            f"Distance: {metrics.total_distance_traveled:.1f} veh·km",
            f"Cong. Time: {metrics.congestion_time:.3f} h",
            f"Excess ρ: {metrics.excess_congestion_time:.2f}",
            f"Speed Var: {metrics.total_speed_variance:.1f}",
            "━━━━━━━━━━━━━━━━━━━━", "Simulation params:",
            f"T = {params.T * 3600:.0f} sec",
            f"Duration = {results.parameters.Tfin:.1f} h",
            f"Segments = {results.rho.shape[1]}",
            f"Lanes = {params.lanes}",
        ]
        # if metrics.computation_time is not None:
        #     summary_lines.append(f"Comp. Time = {metrics.computation_time:.1f} s")
        axs[3, 1].text(0.1, 0.5, "\n".join(summary_lines), fontsize=10, verticalalignment="center",
                       family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        axs[3, 1].set_axis_off()

        return self._finish(fig, axs, results, show)

    def plot_coordinated_control(self, results: MacroSimulationResults, params: BaseParameters,
                                  metrics: MacroOverviewSimulationResults, title: Optional[str] = None,
                                  show: bool = False) -> Figure:
        """Plot coordinated RM + VSL control results (compact layout)."""

        fig, axs = self._figure(4, 2, (14, 10), title or "MPC Coordinated Control - Ramp Metering + VSL")
        seg_colors, seg_names, orig_colors, orig_names = self._palette(results)
        vsl_colors = self._vsl_colors(results.v_ctrl.shape[1])
        vsl_names = [f"VSL Seg {i}" for i in range(results.v_ctrl.shape[1])]

        self._plot_series(axs[0, 0], params.time, results.v, seg_colors, seg_names,
                           "Speed [km/h]", "Speed Profile", ncol=2)
        self._plot_series(axs[0, 1], params.time, results.q, seg_colors, seg_names,
                           "Flow [veh/h]", "Flow Profile", ncol=2)
        self._density_panel(axs[1, 0], results, params, seg_colors, seg_names)
        self._plot_series(axs[1, 1], params.time, results.w, orig_colors, orig_names,
                           "Queue [veh]", "Queue Length", linewidth=2, alpha=0.15)
        self._plot_series(axs[2, 0], params.time, results.q_on, orig_colors, orig_names,
                            "Origin Flow [veh/h]", "Origin Flow", linewidth=2, alpha=0.15)
        self._plot_series(axs[2, 0], params.time, results.q_on, orig_colors, orig_names,
                           "Origin Flow [veh/h]", "Origin Flow", linewidth=2, alpha=0.15)
        self._vsl_panel(axs[2, 1], results, params, vsl_colors, vsl_names)

        if results.r is not None and np.any(results.r):
            self._metering_panel(axs[3, 0], results, params)
        else:
            axs[3, 0].set_xlabel("Time [h]", fontweight="bold")

        self._plot_series(axs[3, 1], params.time, results.demands, orig_colors, orig_names,
                           "Origin Demands [veh/h]", "Origin Demands", linewidth=2, alpha=0.15)
        axs[3, 1].set_xlabel("Time [h]", fontweight="bold")

        self._metrics_box(fig, metrics, include_computation=False)
        return self._finish(fig, axs, results, show)

    def plot_rm_vsl(self, results: MacroSimulationResults, params: BaseParameters,
                     metrics: MacroOverviewSimulationResults, show: bool = False) -> Figure:
        """Plot combined MPC RM + VSL control results, with an explicit VSL-rate panel."""
        if results.r is None or results.v_ctrl is None:
            raise ValueError("results.r and results.v_ctrl are required for this plot")
        fig, axs = self._figure(4, 2, (14, 10), "MPC Ramp Metering + VSL Control - Traffic System Behavior")
        seg_colors, seg_names, orig_colors, orig_names = self._palette(results)
        vsl_colors = self._vsl_colors(results.v_ctrl.shape[1])
        vsl_names = self._segment_names(results.v_ctrl.shape[1])

        self._plot_series(axs[0, 0], params.time, results.v, seg_colors, seg_names,
                           "Speed [km/h]", "Speed Profile", ncol=2)
        self._plot_series(axs[0, 1], params.time, results.q, seg_colors, seg_names,
                           "Flow [veh/h]", "Flow Profile", ncol=2)
        self._density_panel(axs[1, 0], results, params, seg_colors, seg_names)
        self._plot_series(axs[1, 1], params.time, results.w, orig_colors, orig_names,
                           "Queue [veh]", "Queue Length", linewidth=2, alpha=0.15)
        self._plot_series(axs[2, 0], params.time, results.q_on, orig_colors, orig_names,
                           "Origin Flow [veh/h]", "Origin Flow", linewidth=2, alpha=0.15)
        self._vsl_panel(axs[2, 1], results, params, vsl_colors, vsl_names)
        self._metering_panel(axs[3, 0], results, params)

        # self._plot_series(axs[3, 1], params.time, results.B, vsl_colors, vsl_names,
        #                    "VSL Rate", "VSL Control Rate", step=True, alpha=0.15)
        axs[3, 1].set_ylim(-0.05, 1.05)
        axs[3, 1].set_xlabel("Time [h]", fontweight="bold")

        self._metrics_box(fig, metrics, include_computation=True)
        return self._finish(fig, axs, results, show)