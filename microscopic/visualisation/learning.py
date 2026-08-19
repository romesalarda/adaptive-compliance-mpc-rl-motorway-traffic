"""RL episode visualisation — all 7 panels as matplotlib figures and a Plotly dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

if TYPE_CHECKING:
    from microscopic.statistics.rl_stats import RLEpisodeStats

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required for plot_dashboard. pip install plotly")

class RLVisualiser:
    """
    Produces per-episode diagnostic plots for the Q-learning agent.

    Panels:
      1. Cumulative reward over solve intervals
      2. Epsilon decay curve
      3. Q-table heatmap (state × action)
      4. Action selection frequency (bar chart)
      5. Reward distribution per alpha value (boxplot)
      6. TTS timeline annotated with action-change markers
      7. Exploration vs exploitation ratio over time (rolling)
    """

    def __init__(self, rl_stats: "RLEpisodeStats") -> None:
        self.stats = rl_stats

    # ------------------------------------------------------------------
    # Matplotlib figures
    # ------------------------------------------------------------------

    def plot_cumulative_reward(self, show: bool = False):
        """
        Plot cumulative reward over the episode. Returns the matplotlib figure object.

        Args:
            show: bool, whether to plt.show() the figure
        """
        rewards = [r.reward_total for r in self.stats.records]
        cumulative = np.cumsum(rewards)
        times = [r.timestamp / 3600 for r in self.stats.records]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, cumulative, color="#1f77b4")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("Cumulative reward over episode")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_epsilon_decay(self, show: bool = False):
        '''
        Plot epsilon decay over the episode. Returns the matplotlib figure object.
        Args:
            show: bool, whether to plt.show() the figure
        
        '''
        times = [r.timestamp / 3600 for r in self.stats.records]
        epsilons = [r.epsilon for r in self.stats.records]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, epsilons, color="#ff7f0e")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Epsilon")
        ax.set_ylim(0, 1.05)
        ax.set_title("Epsilon decay over episode")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_q_table_heatmap(self, show: bool = False):
        """
        Mean Q-value per (state_label, alpha) pair across all recorded steps.
        Returns the matplotlib figure object.
        """

        if not self.stats.records:
            return None

        state_labels = list(dict.fromkeys(r.state_label for r in self.stats.records))
        alphas = sorted(dict.fromkeys(r.alpha for r in self.stats.records))

        # Accumulate all Q-values for each (state_label, alpha_index) pair
        buckets: dict[tuple, list[float]] = {}
        for r in self.stats.records:
            if not r.q_values_at_state:
                continue
            s_idx = state_labels.index(r.state_label)
            for a_idx, alpha in enumerate(alphas):
                if a_idx < len(r.q_values_at_state):
                    buckets.setdefault((s_idx, a_idx), []).append(r.q_values_at_state[a_idx])

        q_matrix = np.full((len(state_labels), len(alphas)), np.nan)
        for (s_idx, a_idx), vals in buckets.items():
            q_matrix[s_idx, a_idx] = float(np.nanmean(vals))

        fig, ax = plt.subplots(figsize=(max(6, len(alphas) * 1.5), max(3, len(state_labels) * 1.2)))
        im = ax.imshow(q_matrix, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"α={a:.2f}" for a in alphas])
        ax.set_yticks(range(len(state_labels)))
        ax.set_yticklabels(state_labels)
        ax.set_xlabel("Action (alpha compliance)")
        ax.set_ylabel("State")
        ax.set_title("Q-table heatmap (mean Q-value)")
        fig.colorbar(im, ax=ax, label="Q-value")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_action_frequency(self, show: bool = False):
        '''
        Plot a bar chart of how many times each alpha value was selected during the episode.
        Returns the matplotlib figure object.
        '''
        alphas = sorted(dict.fromkeys(r.alpha for r in self.stats.records))
        counts = [sum(1 for r in self.stats.records if r.alpha == a) for a in alphas]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([f"α={a:.2f}" for a in alphas], counts, color="#2ca02c")
        ax.set_xlabel("Alpha value")
        ax.set_ylabel("Times selected")
        ax.set_title("Action selection frequency")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_reward_by_action(self, show: bool = False):
        '''
        Plot a boxplot of rewards grouped by alpha value.
        Returns the matplotlib figure object.
        '''
        alphas = sorted(dict.fromkeys(r.alpha for r in self.stats.records))
        groups = [[r.reward_total for r in self.stats.records if r.alpha == a] for a in alphas]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(groups, labels=[f"α={a:.2f}" for a in alphas], patch_artist=True)
        ax.set_xlabel("Alpha value")
        ax.set_ylabel("Reward")
        ax.set_title("Reward distribution per alpha value")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_tts_timeline(self, show: bool = False):
        """
        TTS over time; vertical markers at action changes, colour-coded by alpha.
        Returns the matplotlib figure object.
        """

        times = [r.timestamp / 3600 for r in self.stats.records]
        tts = [r.tts_raw for r in self.stats.records]
        alphas_unique = sorted(dict.fromkeys(r.alpha for r in self.stats.records))
        cmap = cm.get_cmap("tab10", len(alphas_unique))
        alpha_colour = {a: cmap(i) for i, a in enumerate(alphas_unique)}

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, tts, color="steelblue", linewidth=1.5, label="TTS")

        prev_alpha = None
        for r in self.stats.records:
            if r.alpha != prev_alpha:
                ax.axvline(r.timestamp / 3600, color=alpha_colour[r.alpha], alpha=0.5, linewidth=0.8)
                prev_alpha = r.alpha

        for a in alphas_unique:
            ax.plot([], [], color=alpha_colour[a], label=f"α={a:.2f}")

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("TTS (veh·h)")
        ax.set_title("TTS over time with action change markers")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_exploration_ratio(self, window: int = 10, show: bool = False):
        """
        Rolling exploration fraction (was_exploration=True per window steps).

        Args:
            window: int, rolling window size in steps
            show: bool, whether to plt.show() the figure
        
        """

        explored = [float(r.was_exploration) for r in self.stats.records]
        times = [r.timestamp / 3600 for r in self.stats.records]
        rolling = [
            float(np.mean(explored[max(0, i - window):i + 1]))
            for i in range(len(explored))
        ]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, rolling, color="#9467bd")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Exploration rate (rolling)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Exploration vs exploitation (rolling window={window})")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Plotly dashboard
    # ------------------------------------------------------------------

    def plot_dashboard(
        self,
        filename: str = "rl_dashboard.html",
        output_dir: str = ".",
    ) -> str:
        """
        Render all 7 panels into a single Plotly HTML dashboard. Returns output path.
        
        Args:
            filename: str, output filename (HTML)
            output_dir: str, directory to save the HTML file
        Returns:
            str: path to the saved HTML file
        """
        

        records = self.stats.records
        if not records:
            raise ValueError("No RL step records to visualise.")

        times_h = [r.timestamp / 3600 for r in records]
        alphas_unique = sorted(dict.fromkeys(r.alpha for r in records))
        _colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=False,
            subplot_titles=[
                "Cumulative reward",
                "Epsilon decay",
                "Action selection frequency",
                "TTS with action markers",
                "Exploration ratio (rolling)",
            ],
            vertical_spacing=0.07,
        )

        # Row 1: cumulative reward
        cumulative = np.cumsum([r.reward_total for r in records])
        fig.add_trace(go.Scatter(x=times_h, y=cumulative.tolist(), name="Cumul. reward",
                                 line=dict(color="#1f77b4")), row=1, col=1)

        # Row 2: epsilon
        fig.add_trace(go.Scatter(x=times_h, y=[r.epsilon for r in records], name="Epsilon",
                                 line=dict(color="#ff7f0e")), row=2, col=1)

        # Row 3: action frequency
        freq = {a: sum(1 for r in records if r.alpha == a) for a in alphas_unique}
        fig.add_trace(go.Bar(
            x=[f"α={a:.2f}" for a in alphas_unique],
            y=[freq[a] for a in alphas_unique],
            name="Action freq", marker_color="#2ca02c",
        ), row=3, col=1)

        # Row 4: TTS + alpha-change markers
        fig.add_trace(go.Scatter(x=times_h, y=[r.tts_raw for r in records],
                                 name="TTS", line=dict(color="steelblue")), row=4, col=1)
        prev_alpha = None
        for r in records:
            if r.alpha != prev_alpha:
                a_idx = alphas_unique.index(r.alpha) if r.alpha in alphas_unique else 0
                fig.add_vline(
                    x=r.timestamp / 3600,
                    line_color=_colours[a_idx % len(_colours)],
                    line_width=1, opacity=0.5, row=4, col=1,
                )
                prev_alpha = r.alpha

        # Row 5: rolling exploration ratio
        explored = [float(r.was_exploration) for r in records]
        window = 10
        rolling = [float(np.mean(explored[max(0, i - window):i + 1])) for i in range(len(explored))]
        fig.add_trace(go.Scatter(x=times_h, y=rolling, name="Explore ratio",
                                 line=dict(color="#9467bd")), row=5, col=1)

        summary = self.stats.episode_summary
        kpi = (
            f"Steps: {summary.get('n_steps', '?')} | "
            f"Total reward: {summary.get('total_reward', 0.0):.3f} | "
            f"Explore rate: {summary.get('exploration_rate', 0.0):.1%} | "
            f"Final ε: {summary.get('final_epsilon', 0.0):.4f} | "
            f"Q-table states: {summary.get('q_table_size', '?')}"
        )
        fig.update_layout(
            title=dict(text=f"RL Episode Dashboard<br><sup>{kpi}</sup>"),
            height=1600,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Time (h)", row=5, col=1)

        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        return str(out_path)
