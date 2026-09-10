"""RL training overview: aggregates all episodes for one experiment + control type."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _decode(v: Any) -> Any:
    """Coerce h5py bytes attrs to str so json.loads works correctly."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _load_meta(run_path: Path) -> dict | None:
    """Load /meta group attributes from rl_stats.h5; None when file is absent."""
    h5_path = run_path / "rl_stats.h5"
    if not h5_path.exists():
        print(f"[warn] skipping {run_path.name}: rl_stats.h5 not found")
        return None

    meta: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        if "meta" not in f:
            return {}
        for k, v in f["meta"].attrs.items():
            v = _decode(v)
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
            meta[k] = v
    return meta


def _load_steps(run_path: Path) -> dict[str, Any] | None:
    """Load /steps datasets and /reward_breakdown from rl_stats.h5."""
    h5_path = run_path / "rl_stats.h5"
    if not h5_path.exists():
        return None

    out: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        if "steps" in f:
            for name in f["steps"]:
                out[name] = f["steps"][name][()]
        if "reward_breakdown" in f:
            out["_breakdown"] = {k: f["reward_breakdown"][k][()] for k in f["reward_breakdown"]}
    return out


def _state_key_to_label(key_tuple: tuple, features_config: list[dict]) -> str:
    """Map a discrete state-key tuple (strings) to a human-readable label."""
    parts: list[str] = []
    for i, element in enumerate(key_tuple):
        if i < len(features_config):
            parts.append(f"{features_config[i]['name']}={element}")
        else:
            parts.append(str(element))
    return ", ".join(parts) if parts else str(key_tuple)


def _savefig(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RLOverviewVisualiser:
    """
    Produces multi-episode diagnostic plots for a Q-learning training run.

    Usage:
        vis = RLOverviewVisualiser(experiment_dir, control_type="coordinated")
        vis.load()
        paths = vis.render()   # saves to rl-overview/<control>/<timestamp>/
        # or: vis.render(out_dir=some_path)
    """

    def __init__(self, experiment_dir: str | Path, control_type: str) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.control_type = control_type
        self.control_dir = self.experiment_dir / control_type

        # Populated by load()
        self._runs: list[dict] = []
        self._metas: list[dict] = []
        self._steps: list[dict | None] = []
        self._episode_numbers: list[int] = []
        self._q_table: dict[tuple, np.ndarray] | None = None
        self._action_values: list[float] = []
        self._state_labels: list[str] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self) -> "RLOverviewVisualiser":
        """Load index.json, all rl_stats.h5 files, and q_table.npy. Returns self."""
        runs = self._read_index_runs(self.experiment_dir, self.control_type)
        runs.sort(key=lambda r: r.get("timestamp") or "")
        self._load_runs(runs)
        self._load_q_table()

        print(f"Loaded {len(self._runs)} episodes for '{self.control_type}'")
        return self

    def load_pool(
        self, experiment_dirs: list[str | Path], pool_control_dir: str | Path
    ) -> "RLOverviewVisualiser":
        """
        Load runs merged across a scenario pool's experiment dirs, sorted globally by
        timestamp, with the shared Q-table read from pool_control_dir instead of
        experiment_dir/control_type.
        """
        runs: list[dict] = []
        for exp_dir in experiment_dirs:
            for run in self._read_index_runs(Path(exp_dir), self.control_type):
                run = dict(run)
                run["_source_experiment_dir"] = str(exp_dir)
                runs.append(run)
        runs.sort(key=lambda r: r.get("timestamp") or "")

        self._load_runs(runs)
        self.control_dir = Path(pool_control_dir)
        self._load_q_table()

        print(f"Loaded {len(self._runs)} episodes for '{self.control_type}' across {len(experiment_dirs)} pooled scenarios")
        return self

    @staticmethod
    def _read_index_runs(experiment_dir: Path, control_type: str) -> list[dict]:
        """Read index.json and return raw run entries filtered by control_type (unsorted)."""
        index_path = experiment_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"index.json not found: {index_path}")

        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)

        return [r for r in index.get("runs", []) if r.get("run_type") == control_type]

    def _load_runs(self, runs: list[dict]) -> None:
        """Load meta/steps for each run entry and populate instance state."""
        metas: list[dict | None] = []
        steps: list[dict | None] = []
        for run in runs:
            run_path = Path(run["path"])
            metas.append(_load_meta(run_path))
            steps.append(_load_steps(run_path))

        # Filter to episodes where meta loaded successfully
        valid = [(run, m, s) for run, m, s in zip(runs, metas, steps) if m is not None]
        if not valid:
            raise ValueError("No valid rl_stats.h5 files found in any run folder.")

        self._runs, self._metas, self._steps = zip(*valid)  # type: ignore[assignment]
        self._runs = list(self._runs)
        self._metas = list(self._metas)
        self._steps = list(self._steps)
        self._episode_numbers = list(range(1, len(self._runs) + 1))

    def _load_q_table(self) -> None:
        qt_path = self.control_dir / "q_table.npy"
        desc_path = self.control_dir / "rl_descriptor.json"

        if not qt_path.exists():
            print(f"[warn] q_table.npy not found in {self.control_dir}")
            return

        payload = np.load(str(qt_path), allow_pickle=True).item()
        self._q_table = payload.get("q_table", payload)
        qt_meta = payload.get("meta", {})
        self._action_values = [float(a) for a in qt_meta.get("action_values", [])]

        features_config: list[dict] = []
        if desc_path.exists():
            with open(desc_path, encoding="utf-8") as fh:
                desc = json.load(fh)
            features_config = desc.get("state", {}).get("features", [])
            if not self._action_values:
                self._action_values = [float(a) for a in desc.get("actions", {}).get("alpha_values", [])]

        # Fall back to first episode meta if still empty
        if not self._action_values:
            for m in self._metas:
                cfg = m.get("initial_config", {})
                if cfg and cfg.get("action_values"):
                    self._action_values = [float(a) for a in cfg["action_values"]]
                    break

        if self._q_table is not None:
            self._state_labels = [_state_key_to_label(k, features_config) for k in self._q_table]

    # ------------------------------------------------------------------
    # Render entrypoint
    # ------------------------------------------------------------------

    def render(self, out_dir: Path | None = None) -> list[Path]:
        """
        Generate all plots and summary.json.

        Args:
            out_dir: destination directory. Defaults to
                     <experiment_dir>/rl-overview/<control>/<YYYY-MM-DDTHH-MM-SS>/
        Returns:
            List of Path objects for every file written.
        """
        if not self._runs:
            raise RuntimeError("Call load() before render().")

        if out_dir is None:
            ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            out_dir = self.experiment_dir / "rl-overview" / self.control_type / ts

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []

        saved.append(self._plot_epsilon_decay(out_dir))
        saved.append(self._plot_total_reward(out_dir))

        if self._q_table is not None and self._action_values:
            saved.append(self._plot_q_table_heatmap(out_dir))
        else:
            print("[warn] Q-table heatmap skipped (q_table.npy missing or no action values)")

        saved.append(self._plot_exploration_rate(out_dir))
        saved.append(self._plot_tts_per_episode(out_dir))

        if self._action_values:
            saved.append(self._plot_action_distribution(out_dir))

        rc_path = self._plot_reward_components(out_dir)
        if rc_path:
            saved.append(rc_path)
        else:
            print("[info] reward_components.png skipped (no reward_breakdown data)")

        saved.append(self._save_summary_json(out_dir))

        return saved

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def _plot_epsilon_decay(self, out_dir: Path) -> Path:
        epsilons = [float(m.get("final_epsilon", float("nan"))) for m in self._metas]

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self._episode_numbers, epsilons, color="#ff7f0e",
                    linewidth=1.8, marker="o", markersize=3)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Final ε")
            ax.set_title("Epsilon decay across episodes")
            valid = [e for e in epsilons if not np.isnan(e)]
            if valid:
                ax.set_ylim(0, max(valid) * 1.05)
            fig.tight_layout()

        return _savefig(fig, out_dir, "epsilon_decay.png")

    def _plot_total_reward(self, out_dir: Path, rolling_window: int = 5) -> Path:
        rewards = np.array([float(m.get("total_reward", float("nan"))) for m in self._metas])
        valid_mask = ~np.isnan(rewards)

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self._episode_numbers, rewards, color="#1f77b4",
                    linewidth=1.2, alpha=0.5, label="Total reward")

            if valid_mask.sum() >= rolling_window:
                kernel = np.ones(rolling_window) / rolling_window
                valid_rewards = rewards[valid_mask]
                smoothed = np.convolve(valid_rewards, kernel, mode="valid")
                valid_indices = np.where(valid_mask)[0]
                # align to the end of each window
                rolling_line = np.full_like(rewards, np.nan)
                rolling_line[valid_indices[rolling_window - 1:]] = smoothed
                ax.plot(self._episode_numbers, rolling_line, color="#d62728",
                        linewidth=2.0, label=f"Rolling mean (w={rolling_window})")

            ax.set_xlabel("Episode")
            ax.set_ylabel("Total reward")
            ax.set_title("Total reward per episode (convergence)")
            ax.legend()
            fig.tight_layout()

        return _savefig(fig, out_dir, "total_reward.png")

    def _plot_q_table_heatmap(self, out_dir: Path) -> Path:
        q_table = self._q_table
        n_states = len(q_table)
        n_actions = len(self._action_values)

        q_matrix = np.full((n_states, n_actions), np.nan)
        for row, key in enumerate(q_table):
            vals = q_table[key]
            q_matrix[row, : len(vals)] = vals

        abs_max = float(np.nanmax(np.abs(q_matrix))) if not np.all(np.isnan(q_matrix)) else 1.0

        with plt.rc_context({**_STYLE, "axes.grid": False}):
            fig, ax = plt.subplots(figsize=(max(6, n_actions * 1.5), max(3, n_states * 0.8)))
            im = ax.imshow(q_matrix, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(n_actions))
            ax.set_xticklabels([f"α={a:.2f}" for a in self._action_values], fontsize=9)
            ax.set_yticks(range(n_states))
            ax.set_yticklabels(self._state_labels, fontsize=9)
            ax.set_xlabel("Action (alpha compliance)")
            ax.set_ylabel("State")
            ax.set_title("Q-table heatmap (final)")

            for r in range(n_states):
                for c in range(n_actions):
                    v = q_matrix[r, c]
                    if not np.isnan(v):
                        # dark text for low-magnitude cells, white for extreme values
                        colour = "white" if abs(v) > abs_max * 0.7 else "black"
                        ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color=colour)

            fig.colorbar(im, ax=ax, label="Q-value")
            fig.tight_layout()

        return _savefig(fig, out_dir, "q_table_heatmap.png")

    def _plot_exploration_rate(self, out_dir: Path) -> Path:
        rates = [float(m.get("exploration_rate", float("nan"))) for m in self._metas]

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(self._episode_numbers, rates, color="#9467bd", width=0.7)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Exploration rate")
            ax.set_ylim(0, 1.05)
            ax.set_title("Exploration rate per episode")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            fig.tight_layout()

        return _savefig(fig, out_dir, "exploration_rate.png")

    def _plot_tts_per_episode(self, out_dir: Path) -> Path:
        tts_totals: list[float] = []
        for steps in self._steps:
            if steps is not None and "tts_raw" in steps:
                tts_totals.append(float(np.nansum(steps["tts_raw"])))
            else:
                tts_totals.append(float("nan"))

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self._episode_numbers, tts_totals, color="#2ca02c",
                    linewidth=1.8, marker="o", markersize=3)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total TTS (veh·h)")
            ax.set_title("Total time spent (TTS) per episode")
            fig.tight_layout()

        return _savefig(fig, out_dir, "tts_per_episode.png")

    def _plot_action_distribution(self, out_dir: Path) -> Path:
        n_episodes = len(self._episode_numbers)
        colour_cycle = plt.cm.tab10(np.linspace(0, 0.9, max(len(self._action_values), 1)))
        counts_per_action: dict[float, list[float]] = {a: [] for a in self._action_values}

        for m in self._metas:
            raw = m.get("action_counts", {})
            if not isinstance(raw, dict):
                raw = {}
            # JSON round-trip stores dict keys as strings; normalise to float
            normalised = {float(k): int(v) for k, v in raw.items()}
            for a in self._action_values:
                # find closest key to handle float precision edge cases
                matched = next((v for k, v in normalised.items() if abs(k - a) < 1e-9), 0)
                counts_per_action[a].append(float(matched))

        with plt.rc_context(_STYLE):
            fig_w = max(10, n_episodes * 0.25)
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            bottom = np.zeros(n_episodes)
            for idx, a in enumerate(self._action_values):
                vals = np.array(counts_per_action[a])
                ax.bar(self._episode_numbers, vals, bottom=bottom,
                       label=f"α={a:.2f}", color=colour_cycle[idx], width=0.7)
                bottom = bottom + vals

            ax.set_xlabel("Episode")
            ax.set_ylabel("Steps selected")
            ax.set_title("Action distribution per episode")
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()

        return _savefig(fig, out_dir, "action_distribution.png")

    def _plot_reward_components(self, out_dir: Path) -> Path | None:
        comp_names: list[str] = []
        for steps in self._steps:
            if steps and "_breakdown" in steps:
                comp_names = list(steps["_breakdown"].keys())
                break

        if not comp_names:
            return None

        colour_cycle = plt.cm.tab10(np.linspace(0, 0.9, max(len(comp_names), 1)))
        totals: dict[str, list[float]] = {c: [] for c in comp_names}

        for steps in self._steps:
            if steps and "_breakdown" in steps:
                for c in comp_names:
                    totals[c].append(float(np.nansum(steps["_breakdown"].get(c, [0.0]))))
            else:
                for c in comp_names:
                    totals[c].append(float("nan"))

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 4))
            for idx, c in enumerate(comp_names):
                ax.plot(self._episode_numbers, totals[c], linewidth=1.8, marker="o",
                        markersize=3, label=c, color=colour_cycle[idx])
            ax.set_xlabel("Episode")
            ax.set_ylabel("Sum of reward component")
            ax.set_title("Reward components per episode")
            ax.legend(fontsize=8)
            fig.tight_layout()

        return _savefig(fig, out_dir, "reward_components.png")

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------

    def _save_summary_json(self, out_dir: Path) -> Path:
        records = []
        for ep_num, (run, meta, steps) in enumerate(
            zip(self._runs, self._metas, self._steps), start=1
        ):
            tts_total = (
                float(np.nansum(steps["tts_raw"])) if steps and "tts_raw" in steps else None
            )
            action_counts = meta.get("action_counts", {})
            if not isinstance(action_counts, dict):
                action_counts = {}

            records.append({
                "episode": ep_num,
                "run_id": run.get("run_id"),
                "timestamp": run.get("timestamp"),
                "total_reward": float(meta.get("total_reward", float("nan"))),
                "mean_reward": float(meta.get("mean_reward", float("nan"))),
                "min_reward": float(meta.get("min_reward", float("nan"))),
                "max_reward": float(meta.get("max_reward", float("nan"))),
                "exploration_rate": float(meta.get("exploration_rate", float("nan"))),
                "final_epsilon": float(meta.get("final_epsilon", float("nan"))),
                "n_steps": int(meta.get("n_steps", 0)),
                "q_table_size": int(meta.get("q_table_size", 0)),
                "tts_total": tts_total,
                "action_counts": {str(k): int(v) for k, v in action_counts.items()},
            })

        path = out_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"n_episodes": len(records), "episodes": records}, fh, indent=2)
        return path
