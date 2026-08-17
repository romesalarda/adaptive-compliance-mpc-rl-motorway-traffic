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
    # Prediction alignment helper (shared by dashboard + analysis)
    # ------------------------------------------------------------------

    def _align_predictions(self, async_stats: "AsyncMPCStats", macro_T_s: float) -> dict:
        """
        Align solver predictions against actual micro state.
        Returns 1-step-ahead comparison arrays and per-cycle full-horizon RMSE.
        macro_T_s: macro time step in seconds (push interval).
        """
        micro_times = np.asarray(self.micro.run.timestamps)
        micro_step_s = float(micro_times[1] - micro_times[0]) if len(micro_times) > 1 else 0.1
        macro_step_micro = max(1, int(round(macro_T_s / micro_step_s)))

        records = [r for r in async_stats.records if r.rho_pred]

        pred_t: list = []
        pred_rho_1: list = []
        pred_v_1: list = []
        pred_w_1: list = []
        act_rho_1: list = []
        act_v_1: list = []
        act_w_1: list = []
        cycle_t: list = []
        rmse_rho: list = []
        rmse_v: list = []
        rmse_w: list = []

        for r in records:
            push_idx = int(round(r.push_time_sim / micro_step_s))
            pred_rho = np.array(r.rho_pred)   # (horizon, n_seg)
            horizon  = pred_rho.shape[0]

            idx_1 = push_idx + macro_step_micro
            if idx_1 < len(micro_times):
                pred_t.append(r.push_time_sim + macro_T_s)
                pred_rho_1.append(pred_rho[0])
                if r.v_pred:
                    pred_v_1.append(np.array(r.v_pred)[0])
                if r.w_pred:
                    pred_w_1.append(np.array(r.w_pred)[0])
                act_rho_1.append(self.micro.density[idx_1])
                act_v_1.append(self.micro.speed[idx_1])
                if self.micro.queue is not None:
                    act_w_1.append(self.micro.queue[idx_1])

            # Full-horizon RMSE: compare every macro step in the prediction
            indices = [push_idx + (k + 1) * macro_step_micro for k in range(horizon)]
            valid = [(k, i) for k, i in enumerate(indices) if i < len(micro_times)]
            if valid:
                ks, idxs = zip(*valid)
                ks, idxs = list(ks), list(idxs)
                rho_err = pred_rho[ks] - self.micro.density[idxs]
                rmse_rho.append(float(np.sqrt(np.mean(rho_err ** 2))))
                if r.v_pred:
                    v_err = np.array(r.v_pred)[ks] - self.micro.speed[idxs]
                    rmse_v.append(float(np.sqrt(np.mean(v_err ** 2))))
                if r.w_pred and self.micro.queue is not None:
                    w_err = np.array(r.w_pred)[ks] - self.micro.queue[idxs]
                    rmse_w.append(float(np.sqrt(np.mean(w_err ** 2))))
                cycle_t.append(r.push_time_sim)

        def _arr(lst, ndim2=False):
            if not lst:
                return np.empty((0, 1)) if ndim2 else np.empty((0,))
            return np.array(lst, dtype=np.float32)

        return {
            "pred_t":        _arr(pred_t),
            "pred_rho_1":    _arr(pred_rho_1, ndim2=True),
            "pred_v_1":      _arr(pred_v_1,   ndim2=True),
            "pred_w_1":      _arr(pred_w_1,   ndim2=True),
            "act_rho_1":     _arr(act_rho_1,  ndim2=True),
            "act_v_1":       _arr(act_v_1,    ndim2=True),
            "act_w_1":       _arr(act_w_1,    ndim2=True),
            "cycle_t":       _arr(cycle_t),
            "rmse_rho":      _arr(rmse_rho),
            "rmse_v":        _arr(rmse_v),
            "rmse_w":        _arr(rmse_w),
            "records":       records,
            "macro_T_s":     macro_T_s,
            "macro_step_micro": macro_step_micro,
        }

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
        macro_T_s  = float(np.median(np.diff(push_sim))) if len(push_sim) > 1 else 10.0

        has_predictions = any(r.rho_pred for r in records)
        has_actual      = self.micro.density is not None and self.micro.speed is not None
        aln = self._align_predictions(async_stats, macro_T_s) if has_predictions and has_actual else None

        n_rows = 9
        row_titles = [
            "Solve Timeline (push → receive, per cycle)",
            "Solve Latency per Cycle (s)",
            "Solve Time Distribution",
            "State Pushed to MPC — Density / Speed / Queue",
            "Density: 1-step-ahead Predicted vs Actual",
            "Speed: 1-step-ahead Predicted vs Actual",
            "Queue: 1-step-ahead Predicted vs Actual",
            "Per-Cycle Prediction RMSE (full horizon)",
            "Cumulative Queue-Full Events",
        ]
        row_heights = [0.13, 0.09, 0.09, 0.14, 0.11, 0.11, 0.09, 0.11, 0.13]

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

        # ── Rows 5–7: 1-step-ahead predictions vs actual ─────────────────────
        micro_times_s = np.asarray(self.micro.run.timestamps)
        if aln is not None and aln["pred_t"].size > 0:
            # Row 5: Density
            for seg_i in range(n_seg):
                c = colors[seg_i % len(colors)]
                lbl = self.labels[seg_i] if seg_i < len(self.labels) else f"seg{seg_i+1}"
                fig.add_trace(go.Scatter(
                    x=micro_times_s, y=self.micro.density[:, seg_i],
                    mode="lines", line=dict(color=c, width=1.2),
                    name=f"actual {lbl}", legendgroup=f"d5_{seg_i}", showlegend=True,
                ), row=5, col=1)
                if aln["pred_rho_1"].ndim == 2 and seg_i < aln["pred_rho_1"].shape[1]:
                    fig.add_trace(go.Scatter(
                        x=aln["pred_t"], y=aln["pred_rho_1"][:, seg_i],
                        mode="markers+lines",
                        line=dict(color=c, dash="dash", width=1.0),
                        marker=dict(size=4, symbol="circle-open"),
                        name=f"pred {lbl}", legendgroup=f"d5_{seg_i}", showlegend=True,
                    ), row=5, col=1)
            fig.update_xaxes(title_text="Simulation time (s)", row=5, col=1)
            fig.update_yaxes(title_text="Density (veh/km/lane)", row=5, col=1)

            # Row 6: Speed
            if aln["pred_v_1"].ndim == 2 and aln["pred_v_1"].shape[0] > 0:
                for seg_i in range(n_seg):
                    c = colors[seg_i % len(colors)]
                    lbl = self.labels[seg_i] if seg_i < len(self.labels) else f"seg{seg_i+1}"
                    fig.add_trace(go.Scatter(
                        x=micro_times_s, y=self.micro.speed[:, seg_i],
                        mode="lines", line=dict(color=c, width=1.2),
                        name=f"actual v {lbl}", legendgroup=f"d6_{seg_i}", showlegend=True,
                    ), row=6, col=1)
                    if seg_i < aln["pred_v_1"].shape[1]:
                        fig.add_trace(go.Scatter(
                            x=aln["pred_t"], y=aln["pred_v_1"][:, seg_i],
                            mode="markers+lines",
                            line=dict(color=c, dash="dash", width=1.0),
                            marker=dict(size=4, symbol="circle-open"),
                            name=f"pred v {lbl}", legendgroup=f"d6_{seg_i}", showlegend=True,
                        ), row=6, col=1)
            fig.update_xaxes(title_text="Simulation time (s)", row=6, col=1)
            fig.update_yaxes(title_text="Speed (km/h)", row=6, col=1)

            # Row 7: Queue
            if aln["pred_w_1"].ndim == 2 and aln["pred_w_1"].shape[0] > 0 and self.micro.queue is not None:
                n_orig = self.micro.queue.shape[1]
                for j in range(n_orig):
                    c = colors[j % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=micro_times_s, y=self.micro.queue[:, j],
                        mode="lines", line=dict(color=c, width=1.2),
                        name=f"actual w{j}", legendgroup=f"d7_{j}", showlegend=True,
                    ), row=7, col=1)
                    if j < aln["pred_w_1"].shape[1]:
                        fig.add_trace(go.Scatter(
                            x=aln["pred_t"], y=aln["pred_w_1"][:, j],
                            mode="markers+lines",
                            line=dict(color=c, dash="dash", width=1.0),
                            marker=dict(size=4, symbol="circle-open"),
                            name=f"pred w{j}", legendgroup=f"d7_{j}", showlegend=True,
                        ), row=7, col=1)
            fig.update_xaxes(title_text="Simulation time (s)", row=7, col=1)
            fig.update_yaxes(title_text="Queue (vehicles)", row=7, col=1)
        else:
            fig.add_annotation(
                text="No prediction data available",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=12),
            )

        # ── Row 8: Per-cycle RMSE ─────────────────────────────────────────────
        if aln is not None and aln["rmse_rho"].size > 0:
            fig.add_trace(go.Scatter(
                x=aln["cycle_t"], y=aln["rmse_rho"],
                mode="lines+markers", name="RMSE density (veh/km/lane)",
                line=dict(color="#1f77b4"), marker_size=5,
            ), row=8, col=1)
            if aln["rmse_v"].size > 0:
                fig.add_trace(go.Scatter(
                    x=aln["cycle_t"], y=aln["rmse_v"],
                    mode="lines+markers", name="RMSE speed (km/h)",
                    line=dict(color="#ff7f0e"), marker_size=5,
                ), row=8, col=1)
            if aln["rmse_w"].size > 0:
                fig.add_trace(go.Scatter(
                    x=aln["cycle_t"], y=aln["rmse_w"],
                    mode="lines+markers", name="RMSE queue (veh)",
                    line=dict(color="#2ca02c"), marker_size=5,
                ), row=8, col=1)
        fig.update_xaxes(title_text="Push time (s)", row=8, col=1)
        fig.update_yaxes(title_text="RMSE", row=8, col=1)

        # ── Row 9: Cumulative queue-full events ───────────────────────────────
        if async_stats.queue_events:
            qf_sim_times = sorted(ev["time_sim"] for ev in async_stats.queue_events)
            cum_counts = list(range(1, len(qf_sim_times) + 1))
            fig.add_trace(go.Scatter(
                x=qf_sim_times, y=cum_counts,
                mode="lines+markers",
                name="Cumulative queue-full events",
                line=dict(color="red", shape="hv"),
                hovertemplate="t=%{x:.1f}s<br>Cumulative: %{y}<extra></extra>",
            ), row=9, col=1)
        else:
            fig.add_annotation(
                text="No queue-full events recorded",
                xref="paper", yref="paper", x=0.5, y=0.04,
                showarrow=False, font=dict(size=11),
            )
        fig.update_xaxes(title_text="Simulation time (s)", row=9, col=1)
        fig.update_yaxes(title_text="Count", row=9, col=1)

        # ── Layout ───────────────────────────────────────────────────────────
        n_completed = sum(1 for r in records if r.status == "completed")
        mean_ipopt = float(np.mean(solve_rep)) if len(solve_rep) else 0
        mean_lat   = float(np.mean(latency)) if len(latency) else 0
        mean_rmse_rho = float(np.mean(aln["rmse_rho"])) if aln is not None and aln["rmse_rho"].size > 0 else float("nan")
        subtitle = (
            f"{n_completed} completed cycles | "
            f"mean IPOPT: {mean_ipopt:.3f}s | "
            f"mean latency: {mean_lat:.2f}s | "
            f"mean density RMSE: {mean_rmse_rho:.3f} | "
            f"queue-full events: {len(async_stats.queue_events)}"
        )
        fig.update_layout(
            title=dict(text=(
                f"Async MPC diagnostics — {self.micro.scenario_id} [{self.micro.solver_type}]<br>"
                f"<sup>{subtitle}</sup>"
            )),
            height=2600,
            legend=dict(orientation="v", x=1.01, font=dict(size=9)),
            barmode="overlay",
        )

        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)

    # ------------------------------------------------------------------
    # Detailed prediction vs actual analysis dashboard
    # ------------------------------------------------------------------

    def plot_prediction_analysis(
        self,
        async_stats: "AsyncMPCStats",
        filename: str = "prediction_analysis.html",
        output_dir: str = ".",
    ) -> str:
        """
        Full-detail prediction vs actual dashboard.

        Panels:
          1. Density  — actual (solid) + 1-step-ahead predicted (dashed open circles) +
                        full rollout overlays every 5th cycle (faint dots)
          2. Speed    — same layout as density
          3. Queue    — same layout
          4. Per-cycle RMSE over time (full prediction horizon, all variables)
          5. RMSE summary table (per-segment 1-step + per-cycle mean + overall means)
        """
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required. pip install plotly")

        push_times = np.array([r.push_time_sim for r in async_stats.records])
        macro_T_s  = float(np.median(np.diff(push_times))) if len(push_times) > 1 else 10.0
        aln        = self._align_predictions(async_stats, macro_T_s)

        if aln["pred_t"].size == 0:
            raise ValueError("No prediction data in async_stats records.")

        n_seg   = len(self.labels)
        micro_t = np.asarray(self.micro.run.timestamps)
        colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f"]

        specs = [[{"type": "scatter"}]] * 4 + [[{"type": "table"}]]
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=[
                "Density — actual (solid) vs 1-step-ahead predicted (dashed ○) + full rollouts (dot)",
                "Speed — actual vs 1-step-ahead predicted + full rollouts",
                "Queue — actual vs 1-step-ahead predicted + full rollouts",
                "Per-Cycle RMSE over time (full prediction horizon)",
                "RMSE Summary",
            ],
            row_heights=[0.22, 0.22, 0.15, 0.18, 0.23],
            vertical_spacing=0.06,
            specs=specs,
        )

        def _plot_var(row, actual_arr, pred_1_arr, rollout_attr, ylabel):
            """Plot actual + 1-step + rollout traces for one variable."""
            n_vars = actual_arr.shape[1] if actual_arr.ndim == 2 else 1
            for i in range(n_vars):
                c   = colors[i % len(colors)]
                lbl = self.labels[i] if i < len(self.labels) else f"seg{i+1}"
                y_act = actual_arr[:, i] if actual_arr.ndim == 2 else actual_arr
                # Actual full trace
                fig.add_trace(go.Scatter(
                    x=micro_t, y=y_act,
                    mode="lines", line=dict(color=c, width=1.5),
                    name=f"actual {lbl}",
                    legendgroup=f"r{row}_{i}", showlegend=True,
                ), row=row, col=1)
                # 1-step-ahead predicted
                if pred_1_arr.ndim == 2 and i < pred_1_arr.shape[1]:
                    fig.add_trace(go.Scatter(
                        x=aln["pred_t"], y=pred_1_arr[:, i],
                        mode="markers+lines",
                        line=dict(color=c, dash="dash", width=1.2),
                        marker=dict(size=5, symbol="circle-open"),
                        name=f"1-step pred {lbl}",
                        legendgroup=f"r{row}_{i}", showlegend=True,
                    ), row=row, col=1)
                # Full rollout overlays — every 5th cycle to limit trace count
                for k, r_rec in enumerate(aln["records"]):
                    if k % 5 != 0:
                        continue
                    raw = getattr(r_rec, rollout_attr)
                    if not raw:
                        continue
                    p = np.array(raw)
                    if p.ndim < 2 or i >= p.shape[1]:
                        continue
                    p_times = r_rec.push_time_sim + np.arange(1, p.shape[0] + 1) * macro_T_s
                    fig.add_trace(go.Scatter(
                        x=p_times, y=p[:, i],
                        mode="lines", line=dict(color=c, width=0.6, dash="dot"),
                        opacity=0.25, showlegend=False,
                        hovertemplate=f"c{r_rec.cycle_id} {lbl}<br>t=%{{x:.0f}}s<br>%{{y:.2f}}<extra></extra>",
                    ), row=row, col=1)
            fig.update_xaxes(title_text="Simulation time (s)", row=row, col=1)
            fig.update_yaxes(title_text=ylabel, row=row, col=1)

        # Rows 1-3
        _plot_var(1, self.micro.density, aln["pred_rho_1"], "rho_pred", "Density (veh/km/lane)")
        if aln["pred_v_1"].shape[0] > 0:
            _plot_var(2, self.micro.speed, aln["pred_v_1"], "v_pred", "Speed (km/h)")
        if self.micro.queue is not None and aln["pred_w_1"].shape[0] > 0:
            _plot_var(3, self.micro.queue, aln["pred_w_1"], "w_pred", "Queue (vehicles)")

        # Row 4: Per-cycle RMSE
        if aln["rmse_rho"].size > 0:
            fig.add_trace(go.Scatter(
                x=aln["cycle_t"], y=aln["rmse_rho"],
                mode="lines+markers", name="Density RMSE (veh/km/lane)",
                line=dict(color="#1f77b4"), marker_size=5,
                hovertemplate="t=%{x:.0f}s<br>RMSE=%{y:.3f}<extra></extra>",
            ), row=4, col=1)
        if aln["rmse_v"].size > 0:
            fig.add_trace(go.Scatter(
                x=aln["cycle_t"], y=aln["rmse_v"],
                mode="lines+markers", name="Speed RMSE (km/h)",
                line=dict(color="#ff7f0e"), marker_size=5,
            ), row=4, col=1)
        if aln["rmse_w"].size > 0:
            fig.add_trace(go.Scatter(
                x=aln["cycle_t"], y=aln["rmse_w"],
                mode="lines+markers", name="Queue RMSE (veh)",
                line=dict(color="#2ca02c"), marker_size=5,
            ), row=4, col=1)
        fig.update_xaxes(title_text="Push time (s)", row=4, col=1)
        fig.update_yaxes(title_text="RMSE", row=4, col=1)

        # Row 5: RMSE summary table
        def _seg_rmse_1step(pred_arr, act_arr, n):
            if pred_arr.ndim < 2 or pred_arr.shape[0] == 0:
                return ["N/A"] * n
            sz = min(pred_arr.shape[0], act_arr.shape[0])
            return [
                f"{np.sqrt(np.mean((pred_arr[:sz, i] - act_arr[:sz, i])**2)):.4f}"
                if i < pred_arr.shape[1] and i < act_arr.shape[1] else "N/A"
                for i in range(n)
            ]

        seg_names    = list(self.labels[:n_seg]) + ["MEAN (all seg)"]
        rho_1step    = _seg_rmse_1step(aln["pred_rho_1"], aln["act_rho_1"], n_seg)
        v_1step      = _seg_rmse_1step(aln["pred_v_1"],   aln["act_v_1"],   n_seg)
        mean_rho_cyc = f"{float(np.mean(aln['rmse_rho'])):.4f}" if aln["rmse_rho"].size > 0 else "N/A"
        mean_v_cyc   = f"{float(np.mean(aln['rmse_v'])):.4f}"   if aln["rmse_v"].size  > 0 else "N/A"
        mean_w_cyc   = f"{float(np.mean(aln['rmse_w'])):.4f}"   if aln["rmse_w"].size  > 0 else "N/A"
        # Append mean row
        rho_1step.append(mean_rho_cyc)
        v_1step.append(mean_v_cyc)
        # Queue: overall cycle RMSE (not per-segment) — pad with dash for segment rows
        w_col = ["—"] * n_seg + [mean_w_cyc]

        row_fill = [["#2c3e50"] + ["#ecf0f1" if i % 2 == 0 else "#ffffff" for i in range(n_seg + 1)]]
        font_col = [["white"] + ["black"] * (n_seg + 1)]

        fig.add_trace(go.Table(
            header=dict(
                values=["Segment",
                        "Density RMSE 1-step (veh/km/lane)",
                        "Speed RMSE 1-step (km/h)",
                        "Queue RMSE mean/cycle (veh)"],
                fill_color="#2c3e50", font=dict(color="white", size=11), align="left",
                height=30,
            ),
            cells=dict(
                values=[seg_names, rho_1step, v_1step, w_col],
                fill_color=row_fill,
                font=dict(color=font_col, size=10),
                align="left", height=25,
            ),
        ), row=5, col=1)

        n_pred   = len(aln["records"])
        mr       = float(np.mean(aln["rmse_rho"])) if aln["rmse_rho"].size > 0 else float("nan")
        mv       = float(np.mean(aln["rmse_v"]))   if aln["rmse_v"].size  > 0 else float("nan")
        fig.update_layout(
            title=dict(text=(
                f"MPC Prediction Analysis — {self.micro.scenario_id} [{self.micro.solver_type}]<br>"
                f"<sup>{n_pred} prediction cycles | macro T={macro_T_s:.1f}s | "
                f"mean density RMSE: {mr:.3f} veh/km/lane | mean speed RMSE: {mv:.3f} km/h</sup>"
            )),
            height=1900,
            legend=dict(orientation="v", x=1.01, font=dict(size=9)),
        )

        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)

