from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from microscopic.agents.state_features import StateDiscretiser
    from microscopic.agents.reward_components import WeightedRewardCalculator

logger = logging.getLogger(__name__)


class QLearningAgent:
    """
    Tabular Q-learning agent that maps a discrete state key to an alpha compliance value.

    State discretisation and reward calculation are injected — the agent itself
    knows nothing about SUMO or traffic-specific quantities.
    """

    def __init__(
        self,
        state_discretiser: "StateDiscretiser",
        reward_calculator: "WeightedRewardCalculator",
        action_values: list[float],
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        epsilon_decay_enabled: bool = True,
        epsilon_decay_mode: str = "per_step",
    ) -> None:
        '''
        Args:
            state_discretiser: StateDiscretiser, maps state dict to discrete key
            reward_calculator: WeightedRewardCalculator, computes total reward and breakdown
            action_values: list of float, the discrete alpha values the agent can select
            learning_rate: float, Q-learning learning rate (alpha)
            discount_factor: float, Q-learning discount factor (gamma)
            epsilon: float, initial exploration rate
            epsilon_decay: float, multiplicative decay factor for epsilon after each step
            epsilon_min: float, minimum value for epsilon
            epsilon_decay_enabled: bool, whether to apply epsilon decay
            epsilon_decay_mode: str, when to decay — "per_step" (each solve interval) or "per_episode" (once per run)
        
        '''
        self.state_discretiser = state_discretiser
        self.reward_calculator = reward_calculator
        self.action_values = list(action_values)
        self.n_actions = len(action_values)

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_decay_enabled = epsilon_decay_enabled
        self.epsilon_decay_mode = epsilon_decay_mode

        # Frozen snapshot of construction-time hyperparameters for logging/reporting
        self._initial_epsilon = epsilon
        self._initial_lr = learning_rate
        self._initial_discount = discount_factor
        self._initial_decay = epsilon_decay
        self._initial_min = epsilon_min
        self._initial_decay_enabled = epsilon_decay_enabled
        self._initial_decay_mode = epsilon_decay_mode

        self.q_table: defaultdict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        self.last_was_exploration: bool = False
        self._episode_id: str = uuid.uuid4().hex
        self._total_steps: int = 0
        self._total_episodes: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state_data: dict) -> tuple[int, float]:
        """Epsilon-greedy: returns (action_idx, alpha_value)."""
        state_key = self.state_discretiser.key(state_data)
        self.last_was_exploration = bool(np.random.random() < self.epsilon)
        if self.last_was_exploration:
            idx = int(np.random.randint(0, self.n_actions))
        else:
            idx = int(np.argmax(self.q_table[state_key]))
        return idx, self.action_values[idx]

    def select_best_action(self, state_data: dict) -> tuple[int, float]:
        """Greedy — no exploration."""
        state_key = self.state_discretiser.key(state_data)
        self.last_was_exploration = False
        idx = int(np.argmax(self.q_table[state_key]))
        return idx, self.action_values[idx]

    # ------------------------------------------------------------------
    # Q-table update
    # ------------------------------------------------------------------

    def update(
        self,
        prev_state_key: tuple,
        action_idx: int,
        state_data: dict,
        done: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute reward, perform TD update.
        Returns (total_reward, breakdown_dict) for logging.

        Args:
            prev_state_key: tuple, the discrete state key before taking the action
            action_idx: int, index of the action taken
            state_data: dict, the current state dictionary after taking the action
            done: bool, whether this is a terminal step (episode end)
        Returns:
            total_reward: float, the total reward received for this step
            breakdown_dict: dict[str, float], per-component reward contributions

        """
        total_reward, breakdown = self.reward_calculator.calculate(state_data)

        next_state_key = self.state_discretiser.key(state_data)
        current_q = self.q_table[prev_state_key][action_idx]

        if done:
            target_q = total_reward
        else:
            max_next_q = float(np.max(self.q_table[next_state_key]))
            target_q = total_reward + self.gamma * max_next_q

        self.q_table[prev_state_key][action_idx] += self.lr * (target_q - current_q)
        self._total_steps += 1

        logger.debug(
            "[QL] update s=%s a=%d r=%.4f Q: %.4f → %.4f",
            prev_state_key, action_idx, total_reward,
            current_q, self.q_table[prev_state_key][action_idx],
        )
        return total_reward, breakdown

    # ------------------------------------------------------------------
    # Epsilon management
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Per-step decay — no-op when disabled or mode is per_episode."""
        if not self.epsilon_decay_enabled or self.epsilon_decay_mode != "per_step":
            return
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_epsilon_episode(self) -> None:
        """Per-episode decay — call once at the end of each run."""
        if not self.epsilon_decay_enabled or self.epsilon_decay_mode != "per_episode":
            return
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values_for_state(self, state_data: dict) -> np.ndarray:
        key = self.state_discretiser.key(state_data)
        return self.q_table[key].copy()

    def display_q_table(self) -> None:
        """Print a formatted summary of the current Q-table to stdout."""
        n_states = len(self.q_table)
        action_header = "  ".join(f"α={a:.2f}" for a in self.action_values)
        print(f"\n[QL] Q-Table ({n_states} state(s) | {action_header})")
        if n_states == 0:
            print("  (empty — no states visited yet)")
            print()
            return
        print(f"  {'State key':<30}  {'Best α':>8}  Q-values")
        print(f"  {'-'*30}  {'-'*8}  {'-'*56}")
        for key, q_arr in sorted(self.q_table.items(), key=lambda x: str(x[0])):
            best_idx = int(np.argmax(q_arr))
            best_alpha = self.action_values[best_idx]
            q_str = "  ".join(f"{q:+.4f}" for q in q_arr)
            print(f"  {str(key):<30}  {best_alpha:>8.2f}  [{q_str}]")
        print()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_q_table(self, path: str | Path) -> None:
        """Save Q-table and agent metadata as a .npy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": dict(self.q_table),
            "meta": {
                "epsilon": self.epsilon,
                "total_steps": self._total_steps,
                "total_episodes": self._total_episodes,
                "action_values": self.action_values,
                "episode_id": self._episode_id,
            },
        }
        np.save(str(path), payload)
        logger.info("[QL] Q-table saved to %s (%d states)", path, len(self.q_table))

    def load_q_table(self, path: str | Path | None) -> None:
        if path is None:
            return
        path = Path(path)
        if not path.exists():
            logger.warning("[QL] Q-table not found at %s — starting fresh.", path)
            return
        payload = np.load(str(path), allow_pickle=True).item()
        loaded = payload.get("q_table", payload)  # back-compat: plain dict
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64), loaded
        )
        meta = payload.get("meta", {})
        if "epsilon" in meta:
            self.epsilon = float(meta["epsilon"])
        if "total_steps" in meta:
            self._total_steps = int(meta["total_steps"])
        if "total_episodes" in meta:
            self._total_episodes = int(meta["total_episodes"])
        logger.info(
            "[QL] Loaded Q-table from %s (%d states, eps=%.4f, steps=%d)",
            path, len(self.q_table), self.epsilon, self._total_steps,
        )
        self.display_q_table()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset episode-specific state; preserves Q-table and epsilon."""
        self._episode_id = uuid.uuid4().hex
        self._total_episodes += 1

    @classmethod
    def from_config(
        cls,
        rl_config: dict,
        state_discretiser: "StateDiscretiser",
        reward_calculator: "WeightedRewardCalculator",
    ) -> "QLearningAgent":
        agent_cfg = rl_config.get("agent", {})
        action_values: list[float] = [
            float(v) for v in rl_config["actions"]["alpha_values"]
        ]
        return cls(
            state_discretiser=state_discretiser,
            reward_calculator=reward_calculator,
            action_values=action_values,
            learning_rate=float(agent_cfg.get("learning_rate", 0.1)),
            discount_factor=float(agent_cfg.get("discount_factor", 0.95)),
            epsilon=float(agent_cfg.get("epsilon", 1.0)),
            epsilon_decay=float(agent_cfg.get("epsilon_decay", 0.995)),
            epsilon_min=float(agent_cfg.get("epsilon_min", 0.01)),
            epsilon_decay_enabled=bool(agent_cfg.get("epsilon_decay_enabled", True)),
            epsilon_decay_mode=str(agent_cfg.get("epsilon_decay_mode", "per_step")),
        )
