from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import h5py


@dataclass
class RLStepRecord:
    """
    Statistics captured at every RL decision point (each solve interval).
    
    Attributes:
        timestamp: float, simulation time (s)
        state_key: tuple, discrete state key (from StateDiscretiser)
        state_label: str, human-readable state label (from StateDiscretiser)
        action_idx: int, index of the selected action in the action_values list
        alpha: float, the selected alpha value (action)
        reward_total: float, total reward received for this step
        reward_breakdown: dict[str, float], per-component reward contributions
        epsilon: float, exploration rate at this step
        was_exploration: bool, whether the action was exploratory
        q_values_at_state: list[float], Q-values for all actions at this state
        tts_raw: float, raw TTS increase observed in this step
        collisions_raw: int, number of collisions observed in this step
    
    """

    timestamp: float = 0.0
    state_key: tuple = field(default_factory=tuple)
    state_label: str = ""
    action_idx: int = 0
    alpha: float = 0.0
    reward_total: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    epsilon: float = 1.0
    was_exploration: bool = False
    q_values_at_state: list[float] = field(default_factory=list)
    tts_raw: float = 0.0
    collisions_raw: int = 0


@dataclass
class RLEpisodeStats:
    """
    Accumulates per-step RL records and computes an episode summary on finalise().
    
    Attributes:
        records: list[RLStepRecord], all recorded steps in the episode
        episode_summary: dict, summary statistics computed at finalise()
        _finalised: bool, whether finalise() has been called
    
    """

    records: list[RLStepRecord] = field(default_factory=list)
    episode_summary: dict = field(default_factory=dict)
    _finalised: bool = field(default=False, repr=False)

    def record_step(self, record: RLStepRecord) -> None:
        self.records.append(record)

    def finalise(self, agent) -> None:
        """Compute episode summary — call once after the simulation ends."""
        if not self.records:
            self.episode_summary = {}
            self._finalised = True
            return

        rewards = [r.reward_total for r in self.records]
        n = len(self.records)
        n_explore = sum(1 for r in self.records if r.was_exploration)
        action_counts = {}
        for r in self.records:
            action_counts[float(r.alpha)] = action_counts.get(float(r.alpha), 0) + 1

        self.episode_summary = {
            "n_steps": n,
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "exploration_rate": float(n_explore / n) if n > 0 else 0.0,
            "final_epsilon": float(agent.epsilon),
            "q_table_size": len(agent.q_table),
            "total_agent_steps": int(agent._total_steps),
            "total_episodes": int(agent._total_episodes),
            "action_counts": action_counts,
            "initial_config": {
                "epsilon": float(agent._initial_epsilon),
                "learning_rate": float(agent._initial_lr),
                "discount_factor": float(agent._initial_discount),
                "epsilon_decay": float(agent._initial_decay),
                "epsilon_min": float(agent._initial_min),
                "epsilon_decay_enabled": bool(agent._initial_decay_enabled),
                "epsilon_decay_mode": str(agent._initial_decay_mode),
                "action_values": [float(a) for a in agent.action_values],
            },
        }
        self._finalised = True


# ---------------------------------------------------------------------------
# HDF5 serialisation
# ---------------------------------------------------------------------------

def save_rl_stats_hdf5(stats: RLEpisodeStats, path: str | Path) -> None:
    '''
    Save RLEpisodeStats to an HDF5 file. The structure is:
    /meta: attributes for episode summary
    /steps: datasets for each step attribute (timestamp, state_label, action_idx, etc.)
    /reward_breakdown: group with datasets for each reward component (if present)
    
    Args:
        stats: RLEpisodeStats, the statistics to save
        path: str or Path, the output file path
    
    '''

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        mg = f.create_group("meta")
        for k, v in stats.episode_summary.items():
            if isinstance(v, dict):
                # store dicts as JSON strings
                mg.attrs[k] = json.dumps(v)
            else:
                mg.attrs[k] = v

        if not stats.records:
            return

        sg = f.create_group("steps")
        sg.create_dataset(
            "timestamp",
            data=np.array([r.timestamp for r in stats.records], dtype=np.float64),
        )
        sg.create_dataset(
            "action_idx",
            data=np.array([r.action_idx for r in stats.records], dtype=np.int32),
        )
        sg.create_dataset(
            "alpha",
            data=np.array([r.alpha for r in stats.records], dtype=np.float64),
        )
        sg.create_dataset(
            "reward_total",
            data=np.array([r.reward_total for r in stats.records], dtype=np.float64),
        )
        sg.create_dataset(
            "epsilon",
            data=np.array([r.epsilon for r in stats.records], dtype=np.float64),
        )
        sg.create_dataset(
            "was_exploration",
            data=np.array([r.was_exploration for r in stats.records], dtype=bool),
        )
        sg.create_dataset(
            "tts_raw",
            data=np.array([r.tts_raw for r in stats.records], dtype=np.float64),
        )
        sg.create_dataset(
            "collisions_raw",
            data=np.array([r.collisions_raw for r in stats.records], dtype=np.int32),
        )

        str_dt = h5py.string_dtype()
        sg.create_dataset(
            "state_label",
            data=[r.state_label for r in stats.records],
            dtype=str_dt,
        )

        # Q-values: (n_steps, n_actions)
        q_arrays = [r.q_values_at_state for r in stats.records]
        if any(q_arrays):
            max_len = max(len(q) for q in q_arrays)
            padded = np.full((len(q_arrays), max_len), np.nan, dtype=np.float64)
            for i, q in enumerate(q_arrays):
                if q:
                    padded[i, : len(q)] = q
            sg.create_dataset("q_values", data=padded, compression="gzip")

        # per-component reward breakdown — stored as separate datasets
        if stats.records[0].reward_breakdown:
            rg = f.create_group("reward_breakdown")
            for comp_name in stats.records[0].reward_breakdown:
                rg.create_dataset(
                    comp_name,
                    data=np.array(
                        [r.reward_breakdown.get(comp_name, 0.0) for r in stats.records],
                        dtype=np.float64,
                    ),
                )


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def save_rl_stats_json(stats: RLEpisodeStats, path: str | Path) -> None:
    '''
    Save RLEpisodeStats to a JSON file.

    Args:
        stats: RLEpisodeStats, the statistics to save
        path: str or Path, the output file path
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": stats.episode_summary,
        "steps": [
            {
                "timestamp": r.timestamp,
                "state_label": r.state_label,
                "action_idx": r.action_idx,
                "alpha": r.alpha,
                "reward_total": r.reward_total,
                "reward_breakdown": r.reward_breakdown,
                "epsilon": r.epsilon,
                "was_exploration": r.was_exploration,
                "q_values_at_state": r.q_values_at_state,
                "tts_raw": r.tts_raw,
                "collisions_raw": r.collisions_raw,
            }
            for r in stats.records
        ],
    }

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_rl_descriptor(rl_config: dict, path: str | Path) -> None:
    """Write the RL config (agent hyperparams, actions, reward) to a JSON descriptor.

    Saved once per experiment folder alongside q_table.npy so the config that
    produced the Q-table is always recoverable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rl_config, f, indent=2)
