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
    from microscopic.statistics.async_stats import AsyncMPCStats
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

    # ------------------------------------------------------------------
    # Async MPC detail dashboard
    # ------------------------------------------------------------------

    def plot_async_dashboard(
        self,
        async_stats: "AsyncMPCStats",
        filename: str = "async_dashboard.html",
        output_dir: str = ".",
    ) -> str:
        """
        Rich Plotly dashboard for async MPC diagnostics.

        Panels:
          1. Solve timeline (Gantt): push→receive bar per cycle, queue-full markers
          2. Solve latency per cycle (receive_sim - push_sim)
          3. Solve time distributions: IPOPT reported vs wall push→receive
          4. State pushed to MPC per cycle (rho / v / w per segment)
          5. Predicted rho vs actual micro density per segment (when available)
          6. Cumulative queue-full event count
        """
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required. pip install plotly")

        records = async_stats.records
        if not records:
            raise ValueError("No completed async MPC records to visualise.")

        n_seg = len(self.labels)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f"]

        push_sim  = np.array([r.push_time_sim  for r in records])
        recv_sim  = np.array([r.receive_time_sim for r in records])
        solve_rep = np.array([r.solve_time_reported for r in records])
        solve_wall = np.array([r.solve_time_wall for r in records])
        cycle_ids  = np.array([r.cycle_id for r in records])
        latency    = recv_sim - push_sim

        has_predictions = any(r.rho_pred for r in records)
        has_actual      = self.micro.density is not None
        show_pred_vs_actual = has_predictions and has_actual

        n_rows = 6
        row_titles = [
            "Solve Timeline (push → receive, per cycle)",
            "Solve Latency per Cycle (s)",
            "Solve Time Distribution",
            "State Pushed to MPC — Density / Speed / Queue",
            f"Predicted rho vs Actual Micro Density {'(per segment)' if show_pred_vs_actual else '(no predictions recorded)'}",
            "Cumulative Queue-Full Events",
        ]
        row_heights = [0.18, 0.12, 0.12, 0.20, 0.24, 0.14]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=False,
            subplot_titles=row_titles,
            vertical_spacing=0.055,
            row_heights=row_heights,
        )

        # ── Row 1: Gantt timeline ────────────────────────────────────────────
        for i, r in enumerate(records):
            c = colors[i % len(colors)]
            fig.add_trace(go.Bar(
                x=[r.receive_time_sim - r.push_time_sim],
                y=[f"cycle {r.cycle_id}"],
                base=[r.push_time_sim],
                orientation="h",
                marker_color=c,
                marker_opacity=0.55,
                name=f"cycle {r.cycle_id}",
                hovertemplate=(
                    f"Cycle {r.cycle_id}<br>"
                    f"Push: {r.push_time_sim:.1f}s<br>"
                    f"Receive: {r.receive_time_sim:.1f}s<br>"
                    f"Latency: {r.receive_time_sim - r.push_time_sim:.2f}s<br>"
                    f"Solve time: {r.solve_time_reported:.3f}s<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=1)
            # inner bar showing IPOPT solve time only
            fig.add_trace(go.Bar(
                x=[r.solve_time_reported],
                y=[f"cycle {r.cycle_id}"],
                base=[r.push_time_sim],
                orientation="h",
                marker_color=c,
                marker_opacity=0.95,
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)

        # queue-full markers on timeline
        if async_stats.queue_events:
            qf_times = [ev["time_sim"] for ev in async_stats.queue_events]
            # find closest cycle y label for vertical placement
            for qt in qf_times:
                fig.add_vline(x=qt, line_dash="dot", line_color="red",
                              annotation_text="Q-full", annotation_font_size=8, row=1, col=1)

        fig.update_xaxes(title_text="Simulation time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Cycle", autorange="reversed", row=1, col=1)
        fig.update_layout(barmode="overlay")

        # ── Row 2: Latency line ──────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=push_sim, y=latency,
            mode="lines+markers",
            name="Latency (s)",
            line=dict(color="#1f77b4"),
            hovertemplate="Push t=%.1fs<br>Latency: %.3fs<extra></extra>" % (0, 0),
        ), row=2, col=1)
        # solve_interval reference line
        solve_interval_s = float(np.median(np.diff(push_sim))) if len(push_sim) > 1 else 0
        if solve_interval_s > 0:
            fig.add_hline(y=solve_interval_s, line_dash="dash", line_color="orange",
                          annotation_text=f"Solve interval ({solve_interval_s:.0f}s)",
                          annotation_font_size=9, row=2, col=1)
        fig.update_xaxes(title_text="Push time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Latency (s)", row=2, col=1)

        # ── Row 3: Histograms ────────────────────────────────────────────────
        fig.add_trace(go.Histogram(
            x=solve_rep, name="IPOPT reported (s)",
            opacity=0.75, nbinsx=20,
            marker_color="#2ca02c",
        ), row=3, col=1)
        fig.add_trace(go.Histogram(
            x=solve_wall, name="Wall push→receive (s)",
            opacity=0.55, nbinsx=20,
            marker_color="#d62728",
        ), row=3, col=1)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)

        # ── Row 4: State pushed to MPC ───────────────────────────────────────
        state_rho = np.array([r.state_rho for r in records if r.state_rho])   # (n_cyc, n_seg)
        state_v   = np.array([r.state_v   for r in records if r.state_v])
        state_w   = np.array([r.state_w   for r in records if r.state_w])
        push_times_valid = push_sim[: len(state_rho)]

        if state_rho.ndim == 2:
            for i in range(min(state_rho.shape[1], n_seg)):
                c = colors[i % len(colors)]
                lbl = self.labels[i] if i < len(self.labels) else f"seg{i+1}"
                fig.add_trace(go.Scatter(
                    x=push_times_valid, y=state_rho[:, i],
                    mode="lines+markers", marker_size=4,
                    name=f"rho pushed {lbl}", line=dict(color=c),
                    legendgroup=f"pushed_{i}", showlegend=True,
                    customdata=cycle_ids[: len(state_rho)],
                    hovertemplate="t=%{x:.1f}s cycle=%{customdata}<br>rho=%{y:.2f}<extra></extra>",
                ), row=4, col=1)
        if state_v.ndim == 2:
            for i in range(min(state_v.shape[1], n_seg)):
                c = colors[i % len(colors)]
                lbl = self.labels[i] if i < len(self.labels) else f"seg{i+1}"
                fig.add_trace(go.Scatter(
                    x=push_times_valid, y=state_v[:, i],
                    mode="lines", line=dict(color=c, dash="dash"),
                    name=f"v pushed {lbl}", legendgroup=f"pushed_{i}", showlegend=False,
                ), row=4, col=1)
        if state_w.ndim == 2:
            for j in range(state_w.shape[1]):
                fig.add_trace(go.Scatter(
                    x=push_times_valid, y=state_w[:, j],
                    mode="lines+markers", marker_size=3, marker_symbol="square",
                    name=f"w pushed origin{j}",
                    line=dict(color="#8c564b", dash="dot"),
                    showlegend=True,
                ), row=4, col=1)
        fig.update_xaxes(title_text="Push time (s)", row=4, col=1)
        fig.update_yaxes(title_text="rho (solid) / v (dash) / w (dot)", row=4, col=1)

        # ── Row 5: Predicted rho vs actual density ───────────────────────────
        if show_pred_vs_actual:
            micro_times = np.asarray(self.micro.run.timestamps)  # seconds
            step_len_s = float(micro_times[1] - micro_times[0]) if len(micro_times) > 1 else 1.0

            for seg_i in range(n_seg):
                c = colors[seg_i % len(colors)]
                lbl = self.labels[seg_i] if seg_i < len(self.labels) else f"seg{seg_i+1}"

                # Actual micro density for this segment
                fig.add_trace(go.Scatter(
                    x=micro_times, y=self.micro.density[:, seg_i],
                    mode="lines", line=dict(color=c, width=1.5),
                    name=f"actual {lbl}", legendgroup=f"pred_{seg_i}", showlegend=True,
                ), row=5, col=1)

                # Predicted rollouts from each cycle (semi-transparent)
                for r in records:
                    if not r.rho_pred:
                        continue
                    pred = np.array(r.rho_pred)  # (horizon, n_seg)
                    if seg_i >= pred.shape[1]:
                        continue
                    horizon = pred.shape[0]
                    pred_times = r.push_time_sim + np.arange(1, horizon + 1) * step_len_s
                    fig.add_trace(go.Scatter(
                        x=pred_times, y=pred[:, seg_i],
                        mode="lines", line=dict(color=c, width=0.8, dash="dot"),
                        opacity=0.35,
                        name=f"pred {lbl} c{r.cycle_id}",
                        legendgroup=f"pred_{seg_i}", showlegend=False,
                        hovertemplate=f"Cycle {r.cycle_id}<br>rho_pred={seg_i}<br>%{{y:.2f}}<extra></extra>",
                    ), row=5, col=1)
        else:
            fig.add_annotation(
                text="No prediction data available (solver did not return rho_pred)",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=12),
                row=5, col=1,
            )
        fig.update_xaxes(title_text="Simulation time (s)", row=5, col=1)
        fig.update_yaxes(title_text="Density (veh/km/lane)", row=5, col=1)

        # ── Row 6: Cumulative queue-full events ──────────────────────────────
        if async_stats.queue_events:
            qf_sim_times = sorted(ev["time_sim"] for ev in async_stats.queue_events)
            cum_counts = list(range(1, len(qf_sim_times) + 1))
            fig.add_trace(go.Scatter(
                x=qf_sim_times, y=cum_counts,
                mode="lines+markers",
                name="Cumulative queue-full events",
                line=dict(color="red", shape="hv"),
                hovertemplate="t=%{x:.1f}s<br>Cumulative: %{y}<extra></extra>",
            ), row=6, col=1)
        else:
            fig.add_annotation(
                text="No queue-full events recorded",
                xref="paper", yref="paper", x=0.5, y=0.04,
                showarrow=False, font=dict(size=11),
            )
        fig.update_xaxes(title_text="Simulation time (s)", row=6, col=1)
        fig.update_yaxes(title_text="Count", row=6, col=1)

        # ── Layout ───────────────────────────────────────────────────────────
        n_completed = sum(1 for r in records if r.status == "completed")
        mean_ipopt = float(np.mean(solve_rep)) if len(solve_rep) else 0
        mean_lat   = float(np.mean(latency)) if len(latency) else 0
        subtitle = (
            f"{n_completed} completed cycles | "
            f"mean IPOPT: {mean_ipopt:.3f}s | "
            f"mean latency: {mean_lat:.2f}s | "
            f"queue-full events: {len(async_stats.queue_events)}"
        )
        fig.update_layout(
            title=dict(text=(
                f"Async MPC diagnostics — {self.micro.scenario_id} [{self.micro.solver_type}]<br>"
                f"<sup>{subtitle}</sup>"
            )),
            height=1800,
            legend=dict(orientation="v", x=1.01, font=dict(size=9)),
            barmode="overlay",
        )

        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)

