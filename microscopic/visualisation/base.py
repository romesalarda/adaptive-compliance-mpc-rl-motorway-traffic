"""
Visualisation for microscopic simulation results.

MicroVisualiser produces:
  - Per-edge time-series of density, speed, and queue
  - Overlay comparison of macro-predicted vs micro-actual state
  - History of applied VSL and RM controls
  - Combined Plotly HTML dashboard

Mirrors the structure of macroscopic/visualisation/base.py so results can be
shown side-by-side without coupling the two modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from microscopic.statistics.base import MicroSimulationResults
    from macroscopic.statistics.base import MacroSimulationResults


class MicroVisualiser:

    def __init__(
        self,
        micro_results: MicroSimulationResults,
        macro_results: Optional[MacroSimulationResults] = None,
        edge_labels: Optional[list[str]] = None,
    ):
        self.micro = micro_results
        self.macro = macro_results
        # Human-readable labels for each segment; fall back to seg1..segN
        n = micro_results.density.shape[1] if micro_results.density is not None else 0
        self.labels = edge_labels or [f"seg{i+1}" for i in range(n)]

    # ------------------------------------------------------------------
    # Individual plot helpers
    # ------------------------------------------------------------------

    def plot_timeseries(
        self,
        segment_indices: Optional[list[int]] = None,
        show: bool = False,
    ):
        """
        Line plots of density, speed, and queue over simulation time.
        Returns a matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        times = np.asarray(self.micro.run.timestamps) / 3600  # hours
        idxs = segment_indices or list(range(len(self.labels)))

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].set_ylabel("Density (veh/km/lane)")
        axes[1].set_ylabel("Speed (km/h)")
        axes[2].set_ylabel("Queue (vehicles)")
        axes[2].set_xlabel("Time (h)")

        if self.micro.density is not None:
            for i in idxs:
                axes[0].plot(times, self.micro.density[:, i], label=self.labels[i])
                axes[1].plot(times, self.micro.speed[:, i], label=self.labels[i])

        if self.micro.queue is not None:
            for j in range(self.micro.queue.shape[1]):
                axes[2].plot(times, self.micro.queue[:, j], label=f"origin {j}")

        for ax in axes:
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Micro simulation — {self.micro.scenario_id} [{self.micro.solver_type}]"
        )
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_macro_micro_comparison(self, show: bool = False):
        """
        Side-by-side comparison of macro-predicted and micro-actual density and speed.
        Returns a matplotlib Figure (or None if no macro results provided).
        """
        import matplotlib.pyplot as plt

        if self.macro is None or self.macro.rho is None:
            return None

        n_seg = self.micro.density.shape[1]
        fig, axes = plt.subplots(2, n_seg, figsize=(4 * n_seg, 6), squeeze=False)

        macro_times = np.arange(self.macro.rho.shape[0]) * float(
            getattr(self.macro.parameters, "T", 10) / 3600
        )  # hours
        micro_times = np.asarray(self.micro.run.timestamps) / 3600

        for i in range(n_seg):
            ax_rho = axes[0][i]
            ax_v = axes[1][i]
            ax_rho.set_title(self.labels[i], fontsize=9)

            ax_rho.plot(macro_times, self.macro.rho[:, i], label="macro", linestyle="--", alpha=0.8)
            ax_rho.plot(micro_times, self.micro.density[:, i], label="micro", alpha=0.8)

            ax_v.plot(macro_times, self.macro.v[:, i], linestyle="--", alpha=0.8)
            ax_v.plot(micro_times, self.micro.speed[:, i], alpha=0.8)

            if i == 0:
                ax_rho.set_ylabel("Density (veh/km/lane)")
                ax_v.set_ylabel("Speed (km/h)")
            ax_rho.legend(fontsize=7)
            ax_rho.grid(True, alpha=0.3)
            ax_v.grid(True, alpha=0.3)

        axes[1][n_seg // 2].set_xlabel("Time (h)")
        fig.suptitle("Macro predicted vs Micro actual")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_controls_history(self, show: bool = False):
        """
        Step-plot of applied VSL and ramp metering rate over time.
        Returns a matplotlib Figure (or None if no controls were recorded).
        """
        import matplotlib.pyplot as plt

        run = self.micro.run
        if not run.solve_steps:
            return None

        solve_times = np.asarray(run.solve_steps) * (
            run.timestamps[1] - run.timestamps[0] if len(run.timestamps) > 1 else 1.0
        ) / 3600

        has_vsl = bool(run.v_ctrl_applied)
        has_rm = bool(run.r_applied)
        n_rows = int(has_vsl) + int(has_rm)
        if n_rows == 0:
            return None

        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True, squeeze=False)
        row = 0

        if has_vsl:
            v_ctrl = np.array(run.v_ctrl_applied)
            for k in range(v_ctrl.shape[1]):
                axes[row][0].step(solve_times, v_ctrl[:, k], label=f"VSL {self.labels[k]}", where="post")
            axes[row][0].set_ylabel("Speed limit (km/h)")
            axes[row][0].legend(fontsize=7)
            axes[row][0].grid(True, alpha=0.3)
            row += 1

        if has_rm:
            r_arr = np.array(run.r_applied)
            for k in range(r_arr.shape[1]):
                axes[row][0].step(solve_times, r_arr[:, k], label=f"RM rate {k}", where="post")
            axes[row][0].set_ylabel("Metering rate")
            axes[row][0].set_ylim(0, 1.05)
            axes[row][0].legend(fontsize=7)
            axes[row][0].grid(True, alpha=0.3)

        axes[-1][0].set_xlabel("Time (h)")
        fig.suptitle("Applied controls history")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Interactive dashboard
    # ------------------------------------------------------------------

    def plot_dashboard(
        self,
        filename: str = "micro_dashboard.html",
        output_dir: str = ".",
        sumo_data: Optional[dict] = None,
    ) -> str:
        """
        Build a Plotly HTML dashboard combining all panels.
        Returns the path to the written file.
        """
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for dashboard generation. pip install plotly")

        times = np.asarray(self.micro.run.timestamps) / 3600
        solve_times = (
            np.asarray(self.micro.run.solve_steps)
            * (times[1] - times[0] if len(times) > 1 else 1.0)
            if self.micro.run.solve_steps
            else np.array([])
        )

        n_seg = self.micro.density.shape[1] if self.micro.density is not None else 0
        run = self.micro.run

        row_titles = ["Density", "Speed", "Queue / origin", "VSL controls", "RM rate"]
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=row_titles,
                            vertical_spacing=0.05)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        # Density
        if self.micro.density is not None:
            for i in range(n_seg):
                c = colors[i % len(colors)]
                fig.add_trace(go.Scatter(x=times, y=self.micro.density[:, i],
                                         name=self.labels[i], line=dict(color=c), legendgroup=f"seg{i}",
                                         showlegend=True), row=1, col=1)
                # macro overlay
                if self.macro is not None and self.macro.rho is not None:
                    macro_t = np.arange(self.macro.rho.shape[0]) * float(
                        getattr(self.macro.parameters, "T", 10)) / 3600
                    fig.add_trace(go.Scatter(x=macro_t, y=self.macro.rho[:, i],
                                             name=f"{self.labels[i]} (macro)", line=dict(color=c, dash="dash"),
                                             legendgroup=f"seg{i}", showlegend=True), row=1, col=1)

        # Speed
        if self.micro.speed is not None:
            for i in range(n_seg):
                fig.add_trace(go.Scatter(x=times, y=self.micro.speed[:, i],
                                         name=self.labels[i], line=dict(color=colors[i % len(colors)]),
                                         legendgroup=f"seg{i}", showlegend=False), row=2, col=1)

        # Queue
        if self.micro.queue is not None:
            for j in range(self.micro.queue.shape[1]):
                fig.add_trace(go.Scatter(x=times, y=self.micro.queue[:, j],
                                         name=f"origin {j}"), row=3, col=1)

        # VSL
        if run.v_ctrl_applied:
            v_ctrl = np.array(run.v_ctrl_applied)
            for k in range(v_ctrl.shape[1]):
                fig.add_trace(go.Scatter(x=solve_times, y=v_ctrl[:, k], mode="lines+markers",
                                         name=f"VSL {self.labels[k]}", line_shape="hv"), row=4, col=1)

        # RM
        if run.r_applied:
            r_arr = np.array(run.r_applied)
            for k in range(r_arr.shape[1]):
                fig.add_trace(go.Scatter(x=solve_times, y=r_arr[:, k], mode="lines+markers",
                                         name=f"RM rate {k}", line_shape="hv"), row=5, col=1)

        ov = self.micro.overview
        kpi_text = (
            f"TTS: {ov.total_time_spent:.1f} veh·h | "
            f"TTD: {ov.total_distance_travelled:.0f} veh·km | "
            f"Waiting: {ov.total_waiting:.1f} veh·h | "
            f"Collisions: {ov.total_collisions}"
        )
        if sumo_data and sumo_data.get("tripinfo"):
            td = sumo_data["tripinfo"]
            kpi_text += (
                f" | SUMO TTS: {td.get('tts_s', 0)/3600:.1f} h | "
                f"Vehicles: {td.get('n_vehicles', 0)}"
            )

        fig.update_layout(
            title=dict(text=(
                f"Micro simulation dashboard — {self.micro.scenario_id} [{self.micro.solver_type}]<br>"
                f"<sup>{kpi_text}</sup>"
            )),
            height=1400,
            legend=dict(orientation="v", x=1.01),
        )
        fig.update_xaxes(title_text="Time (h)", row=5, col=1)

        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)
