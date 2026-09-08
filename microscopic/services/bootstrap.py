"""
Bootstrap analysis across a folder of repeated micro-sim runs.

Expects `folder` to contain repeated runs of the SAME scenario + solver type,
each in its own subdirectory with `simulation.h5`, `meta.json`, and
`micro_config.json` (the layout produced by micro_main.py), e.g.:

    scenarios/a3_motorway/micro-sim-runs/default/rm/
        rm_001_.../ (simulation.h5, meta.json, micro_config.json, ...)
        rm_002_.../
        rm_003_.../

`RunFolderBootstrapper.run()` loads every run, validates they share a
scenario_id + run_type, computes bootstrap (case-resampling across runs)
confidence bands for density/speed/queue/controls/RMSE, and writes a Plotly
HTML dashboard + JSON summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from macroscopic.logger import LOGGER
from macroscopic.statistics.dump import numpy_to_python
from microscopic.statistics.async_stats import AsyncMPCStats, load_async_stats_hdf5
from microscopic.statistics.base import MicroSimulationResults
from microscopic.statistics.dump import load_hdf5
from microscopic.visualisation.base import MicroVisualiser


class ScenarioMismatchError(ValueError):
    """Raised when runs in the folder don't share the same scenario_id/run_type."""


class InsufficientRunsError(ValueError):
    """Raised when fewer than 2 valid runs are found in the folder."""


@dataclass
class LoadedRun:
    run_id: str
    run_type: str
    scenario_id: str
    path: Path
    results: MicroSimulationResults
    config: dict
    async_stats: Optional[AsyncMPCStats] = None
    labels: list[str] = field(default_factory=list)


def _bootstrap_ci(
    samples: np.ndarray,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Case-resample the first axis (runs) of `samples` with replacement.

    Returns {"mean", "lower", "upper"} arrays with shape samples.shape[1:].
    """
    n_runs = samples.shape[0]
    boot_means = np.empty((n_boot,) + samples.shape[1:], dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot_means[b] = samples[idx].mean(axis=0)

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": samples.mean(axis=0),
        "lower": np.quantile(boot_means, alpha, axis=0),
        "upper": np.quantile(boot_means, 1.0 - alpha, axis=0),
    }


class RunFolderBootstrapper:
    """Load, bootstrap-aggregate, and plot a folder of repeated micro-sim runs."""

    def __init__(
        self,
        folder: str | Path,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: Optional[int] = None,
    ):
        self.folder = Path(folder)
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self.rng = np.random.default_rng(seed)

        self.runs: list[LoadedRun] = []
        self.aggregated: dict = {}

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------

    def load_runs(self) -> list[LoadedRun]:
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {self.folder}")

        run_dirs = sorted(
            d for d in self.folder.iterdir()
            if d.is_dir() and (d / "simulation.h5").exists()
        )
        if len(run_dirs) < 2:
            raise InsufficientRunsError(
                f"Need at least 2 runs to bootstrap; found {len(run_dirs)} under {self.folder}"
            )

        loaded: list[LoadedRun] = []
        for run_dir in run_dirs:
            meta_path = run_dir / "meta.json"
            config_path = run_dir / "micro_config.json"
            if not meta_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"Missing meta.json/micro_config.json in {run_dir}")

            with meta_path.open(encoding="utf-8") as fh:
                meta = json.load(fh)
            with config_path.open(encoding="utf-8") as fh:
                config = json.load(fh)

            results = load_hdf5(run_dir / "simulation.h5")

            async_path = run_dir / "async_mpc_stats.h5"
            async_stats = load_async_stats_hdf5(async_path) if async_path.exists() else None

            edge_map = config.get("edge_to_segment_map", {})
            labels = [name for name, _ in sorted(edge_map.items(), key=lambda kv: kv[1])]

            loaded.append(LoadedRun(
                run_id=meta.get("run_id", run_dir.name),
                run_type=meta.get("run_type", ""),
                scenario_id=meta.get("scenario_id", ""),
                path=run_dir,
                results=results,
                config=config,
                async_stats=async_stats,
                labels=labels,
            ))

        scenario_ids = {r.scenario_id for r in loaded}
        run_types = {r.run_type for r in loaded}
        if len(scenario_ids) > 1 or len(run_types) > 1:
            offenders = "\n".join(f"  {r.path} -> scenario={r.scenario_id!r} run_type={r.run_type!r}" for r in loaded)
            raise ScenarioMismatchError(
                f"Runs under {self.folder} do not share the same scenario_id/run_type:\n{offenders}"
            )

        self.runs = loaded
        return loaded

    # ------------------------------------------------------------------
    # 2. Aggregate
    # ------------------------------------------------------------------

    def aggregate(self) -> dict:
        if not self.runs:
            self.load_runs()

        runs = self.runs
        n_boot, ci, rng = self.n_bootstrap, self.ci, self.rng

        # --- per-step arrays: density, speed, queue ---
        n_steps = min(r.results.density.shape[0] for r in runs)
        density = np.stack([r.results.density[:n_steps] for r in runs])  # (n_runs, n_steps, n_seg)
        speed = np.stack([r.results.speed[:n_steps] for r in runs])
        queue = np.stack([r.results.queue[:n_steps] for r in runs])
        timestamps = runs[0].results.run.timestamps[:n_steps]

        aggregated: dict = {
            "labels": runs[0].labels,
            "timestamps": np.asarray(timestamps),
            "density": _bootstrap_ci(density, n_boot, ci, rng),
            "speed": _bootstrap_ci(speed, n_boot, ci, rng),
            "queue": _bootstrap_ci(queue, n_boot, ci, rng),
        }

        # --- sparse control arrays: VSL / RM (only if every run has them) ---
        vsl_lists = [r.results.run.v_ctrl_applied for r in runs]
        if all(vsl_lists):
            n_solves = min(len(v) for v in vsl_lists)
            v_ctrl = np.stack([np.array(v[:n_solves]) for v in vsl_lists])
            aggregated["v_ctrl"] = _bootstrap_ci(v_ctrl, n_boot, ci, rng)
            aggregated["v_ctrl_solve_t"] = np.asarray(runs[0].results.run.solve_timestamps[:n_solves])

        rm_lists = [r.results.run.r_applied for r in runs]
        if all(rm_lists):
            n_solves = min(len(v) for v in rm_lists)
            r_applied = np.stack([np.array(v[:n_solves]) for v in rm_lists])
            aggregated["r_applied"] = _bootstrap_ci(r_applied, n_boot, ci, rng)
            aggregated["r_applied_solve_t"] = np.asarray(runs[0].results.run.solve_timestamps[:n_solves])

        # --- MPC prediction RMSE (only for runs that have async_mpc_stats.h5) ---
        rmse_rho_list, rmse_v_list, rmse_t = [], [], None
        for r in runs:
            if r.async_stats is None or not r.async_stats.records:
                continue
            push_sim = np.array([rec.push_time_sim for rec in r.async_stats.records])
            macro_T_s = float(np.median(np.diff(push_sim))) if len(push_sim) > 1 else 10.0
            vis = MicroVisualiser(r.results, macro_results=None, edge_labels=r.labels)
            aln = vis._align_predictions(r.async_stats, macro_T_s)
            if aln["rmse_rho"].size > 0:
                rmse_rho_list.append(aln["rmse_rho"])
                rmse_v_list.append(aln["rmse_v"])
                rmse_t = aln["cycle_t"] if rmse_t is None else rmse_t

        if len(rmse_rho_list) >= 2:
            n_cycles = min(a.shape[0] for a in rmse_rho_list)
            rmse_rho = np.stack([a[:n_cycles] for a in rmse_rho_list])
            aggregated["rmse_rho"] = _bootstrap_ci(rmse_rho, n_boot, ci, rng)
            aggregated["rmse_cycle_t"] = rmse_t[:n_cycles]
            if all(a.size > 0 and a.shape[0] >= n_cycles for a in rmse_v_list):
                rmse_v = np.stack([a[:n_cycles] for a in rmse_v_list])
                aggregated["rmse_v"] = _bootstrap_ci(rmse_v, n_boot, ci, rng)
        elif runs and any(r.async_stats is not None for r in runs):
            LOGGER.warning("Fewer than 2 runs have async_mpc_stats.h5 with predictions; skipping RMSE bootstrap panel.")

        # --- scalar overview summary (bootstrap CI on run-level averages) ---
        summary = {}
        for field_name in (
            "total_time_spent", "total_distance_travelled", "total_waiting",
            "mean_density", "mean_speed", "mean_mpc_solve_time",
        ):
            vals = np.array([getattr(r.results.overview, field_name) for r in runs], dtype=np.float64)
            summary[field_name] = _bootstrap_ci(vals, n_boot, ci, rng)
        aggregated["summary"] = summary
        aggregated["n_runs"] = len(runs)
        aggregated["run_ids"] = [r.run_id for r in runs]
        aggregated["scenario_id"] = runs[0].scenario_id
        aggregated["run_type"] = runs[0].run_type

        self.aggregated = aggregated
        return aggregated

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------

    def plot(
        self,
        output_dir: Optional[str | Path] = None,
        filename: str = "bootstrap_analysis.html",
    ) -> str:
        if not self.aggregated:
            self.aggregate()

        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for dashboard generation. pip install plotly")

        agg = self.aggregated
        labels = agg["labels"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        t_h = agg["timestamps"] / 3600.0

        row_titles = ["Density bootstrap (veh/km/lane)", "Speed bootstrap (km/h)", "Queue bootstrap (vehicles)"]
        row_keys = ["density", "speed", "queue"]
        has_vsl = "v_ctrl" in agg
        has_rm = "r_applied" in agg
        has_rmse = "rmse_rho" in agg

        if has_rmse:
            row_titles.append("RMSE bootstrap (density + speed)")
        if has_vsl:
            row_titles.append("VSL control bootstrap (km/h)")
        if has_rm:
            row_titles.append("RM control bootstrap (rate)")
        row_titles.append("Summary of aggregated data")

        n_rows = len(row_titles)
        specs = [[{"type": "scatter"}]] * (n_rows - 1) + [[{"type": "table"}]]
        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=False,
            subplot_titles=row_titles, specs=specs,
            vertical_spacing=0.045,
        )

        def _add_band(row, x, mean, lower, upper, name, color):
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself", fillcolor=color, opacity=0.18,
                line=dict(width=0), hoverinfo="skip",
                legendgroup=name, showlegend=False,
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines", name=name,
                line=dict(color=color), legendgroup=name,
            ), row=row, col=1)

        # Rows 1-3: density / speed / queue
        for i, key in enumerate(row_keys, start=1):
            data = agg[key]
            n_channels = data["mean"].shape[1]
            for c in range(n_channels):
                color = colors[c % len(colors)]
                name = labels[c] if key != "queue" and c < len(labels) else f"origin {c}" if key == "queue" else f"ch{c}"
                _add_band(i, t_h, data["mean"][:, c], data["lower"][:, c], data["upper"][:, c], name, color)
            fig.update_yaxes(title_text=row_titles[i - 1], row=i, col=1)
        fig.update_xaxes(title_text="Time (h)", row=len(row_keys), col=1)

        row = len(row_keys) + 1

        if has_rmse:
            rmse_t = agg["rmse_cycle_t"]
            _add_band(row, rmse_t, agg["rmse_rho"]["mean"], agg["rmse_rho"]["lower"], agg["rmse_rho"]["upper"],
                       "RMSE density", colors[0])
            if "rmse_v" in agg:
                _add_band(row, rmse_t, agg["rmse_v"]["mean"], agg["rmse_v"]["lower"], agg["rmse_v"]["upper"],
                          "RMSE speed", colors[1])
            fig.update_xaxes(title_text="Push time (s)", row=row, col=1)
            fig.update_yaxes(title_text="RMSE", row=row, col=1)
            row += 1

        if has_vsl:
            solve_t = agg["v_ctrl_solve_t"] / 3600.0
            data = agg["v_ctrl"]
            for c in range(data["mean"].shape[1]):
                color = colors[c % len(colors)]
                name = labels[c] if c < len(labels) else f"vsl{c}"
                _add_band(row, solve_t, data["mean"][:, c], data["lower"][:, c], data["upper"][:, c], f"VSL {name}", color)
            fig.update_xaxes(title_text="Time (h)", row=row, col=1)
            fig.update_yaxes(title_text="Speed limit (km/h)", row=row, col=1)
            row += 1

        if has_rm:
            solve_t = agg["r_applied_solve_t"] / 3600.0
            data = agg["r_applied"]
            for c in range(data["mean"].shape[1]):
                color = colors[c % len(colors)]
                _add_band(row, solve_t, data["mean"][:, c], data["lower"][:, c], data["upper"][:, c], f"RM rate {c}", color)
            fig.update_xaxes(title_text="Time (h)", row=row, col=1)
            fig.update_yaxes(title_text="Metering rate", row=row, col=1)
            row += 1

        # Summary table
        summary = agg["summary"]
        metric_names = list(summary.keys())
        means = [f"{summary[m]['mean']:.3f}" for m in metric_names]
        lowers = [f"{summary[m]['lower']:.3f}" for m in metric_names]
        uppers = [f"{summary[m]['upper']:.3f}" for m in metric_names]
        fig.add_trace(go.Table(
            header=dict(values=["Metric", "Mean", f"{int(self.ci*100)}% CI lower", f"{int(self.ci*100)}% CI upper"],
                        fill_color="#2c3e50", font=dict(color="white", size=11), align="left"),
            cells=dict(values=[metric_names, means, lowers, uppers], align="left"),
        ), row=row, col=1)

        fig.update_layout(
            title=dict(text=(
                f"Bootstrap analysis — {agg['scenario_id']} [{agg['run_type']}]<br>"
                f"<sup>{agg['n_runs']} runs: {', '.join(agg['run_ids'])} | "
                f"{self.n_bootstrap} bootstrap resamples | {int(self.ci*100)}% CI</sup>"
            )),
            height=280 * n_rows,
            legend=dict(orientation="v", x=1.01),
        )

        out_dir = Path(output_dir) if output_dir is not None else self.folder
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / filename
        fig.write_html(str(html_path))

        summary_path = out_dir / "bootstrap_summary.json"
        summary_json = {
            "scenario_id": agg["scenario_id"],
            "run_type": agg["run_type"],
            "n_runs": agg["n_runs"],
            "run_ids": agg["run_ids"],
            "n_bootstrap": self.n_bootstrap,
            "ci": self.ci,
            "summary": {m: {k: v for k, v in summary[m].items()} for m in metric_names},
        }
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(numpy_to_python(summary_json), fh, indent=2)

        LOGGER.info(f"Bootstrap dashboard written to {html_path}")
        LOGGER.info(f"Bootstrap summary written to {summary_path}")
        return str(html_path)

    # ------------------------------------------------------------------

    def run(self, output_dir: Optional[str | Path] = None, filename: str = "bootstrap_analysis.html") -> str:
        self.load_runs()
        self.aggregate()
        return self.plot(output_dir=output_dir, filename=filename)
