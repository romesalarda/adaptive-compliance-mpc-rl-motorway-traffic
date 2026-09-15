from __future__ import annotations

import logging
import types as _types
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from microscopic.controller.async_mpc import AsyncMPCController
from microscopic.controller.solver_pool import SolverPoolMixin, solver_swap_mixin_from_config
from microscopic.agents.qlearning import QLearningAgent
from microscopic.agents.state_features import StateDiscretiser
from microscopic.agents.reward_components import WeightedRewardCalculator, reward_from_config
from microscopic.statistics.rl_stats import RLEpisodeStats, RLStepRecord


if TYPE_CHECKING:
    from macroscopic.simulation.environment import MacroEnvironment
    from microscopic.simulation.environment import MicroEnvironment
    from macroscopic.solvers.base import BaseMPCSolver
    from typing import Union

logger = logging.getLogger(__name__)


def _build_rl_controller_class(rebuild_strategy: str) -> type:
    """Compose the correct swap mixin with AsyncMPCController at runtime."""
    swap_mixin = solver_swap_mixin_from_config(rebuild_strategy)

    class _RLAsyncMPCController(swap_mixin, AsyncMPCController):
        pass

    _RLAsyncMPCController.__name__ = "RLAsyncMPCController"
    return _RLAsyncMPCController


class RLAsyncMPCController:
    """
    Async MPC controller whose alpha compliance value is chosen by a Q-learning agent.

    Architecture:
    - Inherits AsyncMPCController for the non-blocking solve thread.
    - Composes SolverPoolMixin (or RebuildSolverMixin) for hot-swapping solvers
      under _solver_lock before each state push.
    - On every solve interval: augments state with RL-derived metrics, performs a
      Q-table update (if learning), selects next alpha, and swaps the solver.

    Usage:
        controller = RLAsyncMPCController.from_config(
            rl_config=..., solver_factory=..., macro_env=..., micro_env=...,
            solve_interval_steps=600, step_length=0.1, learning=True,
            q_table_path=Path("..."),
        )
        controller.setup_rl()
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "Use RLAsyncMPCController.from_config() to build a controller instance."
        )

    @classmethod
    def from_config(
        cls,
        rl_config: dict,
        solver_factory: Callable[[float], "BaseMPCSolver"],
        macro_env: "MacroEnvironment",
        micro_env: "MicroEnvironment",
        solve_interval_steps: int = 600,
        step_length: float = 0.1,
        learning: bool = True,
        q_table_path: Path | None = None,
    ) -> "AsyncMPCController":
        """
        Build and return a fully-wired RLAsyncMPCController instance.
        The returned object is an AsyncMPCController subclass with RL extensions
        bound directly onto it.
        """
        rebuild_strategy = rl_config.get("solver_pool", {}).get("rebuild_strategy", "pool")
        controller_class = _build_rl_controller_class(rebuild_strategy)

        state_disc = StateDiscretiser.from_config(rl_config["state"])
        reward_calc = reward_from_config(rl_config["reward"])
        agent = QLearningAgent.from_config(rl_config, state_disc, reward_calc)

        alpha_values: list[float] = [float(a) for a in rl_config["actions"]["alpha_values"]]
        mid_alpha = alpha_values[len(alpha_values) // 2]
        initial_solver = solver_factory(mid_alpha)

        instance = controller_class.__new__(controller_class)
        AsyncMPCController.__init__(
            instance,
            macro_env=macro_env,
            micro_env=micro_env,
            solver=initial_solver,
            solve_interval_steps=solve_interval_steps,
            step_length=step_length,
        )

        # RL attributes
        instance._rl_config = rl_config
        instance._rl_solver_factory = solver_factory
        instance._rl_alpha_values = alpha_values
        instance._rl_agent = agent
        instance._rl_learning = learning
        instance._rl_q_table_path = q_table_path
        instance._rl_stats = RLEpisodeStats()

        instance._rl_prev_state_key = None
        instance._rl_prev_action_id = None
        instance._rl_prev_alpha = mid_alpha
        instance._rl_step_count = 0

        window_s = _max_reward_window(rl_config)
        window_intervals = max(1, int(window_s / (solve_interval_steps * step_length)))
        instance._rl_tts_deque = deque(maxlen=window_intervals + 1)
        instance._rl_collision_deque = deque()
        instance._rl_collision_window_s = float(window_s)

        # Bind RL methods directly onto the instance
        instance.setup_rl = _types.MethodType(_setup_rl, instance)
        instance.on_step = _types.MethodType(_rl_on_step, instance)
        instance.stop = _types.MethodType(_rl_stop, instance)
        instance.get_rl_stats = lambda: instance._rl_stats
        instance.print_pre_run_status = _types.MethodType(_rl_print_pre_run_status, instance)

        return instance


# ---------------------------------------------------------------------------
# RL methods — bound onto instances by from_config()
# ---------------------------------------------------------------------------

def _setup_rl(self) -> None:
    """Build the solver pool. Must be called once before the SUMO loop starts."""
    logger.info("[RL] Setting up RL controller with %d alpha values ...", len(self._rl_alpha_values))
    logger.info("\n[RL] Building solver pool with rebuild strategy: %s", self._rl_config.get("solver_pool", {}).get("rebuild_strategy", "pool"))

    self.build_solver_pool(self._rl_alpha_values, self._rl_solver_factory)
    _validate_solver_pool_timing(self)
    if self._rl_q_table_path is not None:
        self._rl_agent.load_q_table(self._rl_q_table_path)
    else:
        self._rl_agent.display_q_table()
    if not self._rl_learning:
        self._rl_agent.epsilon = 0.0
        logger.info("[RL] Evaluation mode: epsilon fixed at 0.")
    logger.info(
        "\n[RL]setup_rl() complete — %d alpha values, learning=%s",
        len(self._rl_alpha_values), self._rl_learning,
    )


def _rl_compute_tts(self, state: dict) -> float:
    """TTS via macro formula: T*sum(rho*L*lanes)+sum(w), consistent with macro objective."""
    params = self.macro_env.parameters
    rho = np.asarray(state["rho"], dtype=float).flatten()
    w = np.asarray(state["w"], dtype=float).flatten()
    return float(params.T) * float(np.sum(rho * float(params.L) * float(params.lanes))) + float(np.sum(w))


def _rl_get_collision_count(self) -> int:
    """Best-effort collision count via traci at the current simulation step."""
    try:
        import traci
        if hasattr(traci.simulation, "getCollisions"):
            return len(traci.simulation.getCollisions())
        if hasattr(traci.simulation, "getCollidingVehiclesNumber"):
            return int(traci.simulation.getCollidingVehiclesNumber())
    except Exception:
        pass
    return 0


def _rl_on_step(self, step: int, env: "MicroEnvironment", state: dict) -> None:
    '''
    RL extension of AsyncMPCController.on_step(). Called every micro step. Computes RL metrics, updates Q-table, selects next alpha, and swaps the solver

    Args:
        step: int, current micro step
        env: MicroEnvironment, the micro environment
        state: dict, the current state dictionary from the micro environment

    '''
    sim_time = step * self.step_length

    if sim_time - self._last_push_sim >= self._solve_interval_s:
        # Collect RL metrics
        current_tts = _rl_compute_tts(self, state)
        self._rl_tts_deque.append((sim_time, current_tts))
        tts_increase = (
            self._rl_tts_deque[-1][1] - self._rl_tts_deque[0][1]
            if len(self._rl_tts_deque) >= 2
            else 0.0
        )

        collision_now = _rl_get_collision_count(self)
        self._rl_collision_deque.append((sim_time, collision_now))
        prune_before = sim_time - self._rl_collision_window_s
        while self._rl_collision_deque and self._rl_collision_deque[0][0] < prune_before:
            self._rl_collision_deque.popleft()
        collisions_in_window = int(sum(c for _, c in self._rl_collision_deque))

        rl_state = {**state, "tts_increase": tts_increase, "collisions_in_window": collisions_in_window}

        state_key = self._rl_agent.state_discretiser.key(rl_state)
        state_label = self._rl_agent.state_discretiser.label(rl_state)
        epsilon_before = float(self._rl_agent.epsilon)

        # Q-update using previous state as the transition origin
        reward_total = 0.0
        reward_breakdown: dict[str, float] = {}
        if self._rl_prev_state_key is not None and self._rl_learning:
            reward_total, reward_breakdown = self._rl_agent.update(
                prev_state_key=self._rl_prev_state_key,
                action_idx=self._rl_prev_action_idx,
                state_data=rl_state,
            )

        # Action selection
        if self._rl_learning:
            action_idx, alpha = self._rl_agent.select_action(rl_state)
            was_exploration = bool(np.random.random() < epsilon_before)
        else:
            action_idx, alpha = self._rl_agent.select_best_action(rl_state)
            was_exploration = False

        q_values = self._rl_agent.get_q_values_for_state(rl_state).tolist()

        logger.info(
            "[RL] t=%.1fs  state=%s  α=%.2f (%s)  reward=%.4f  ε=%.4f",
            sim_time, state_label, alpha,
            "explore" if was_exploration else "exploit",
            reward_total, epsilon_before,
        )

        # Swap solver under lock so the worker thread can't start a stale solve.
        # Skip the swap entirely when the policy re-selects the same alpha, so the solver
        # pointer (and any in-flight solve) is left completely undisturbed.
        if alpha != self._rl_prev_alpha:
            with self._solver_lock:
                self.swap_solver(alpha)
        else:
            logger.debug("[RL] alpha unchanged (%.2f) — skipping solver swap", alpha)

        if self._rl_learning:
            self._rl_agent.decay_epsilon()

        self._rl_stats.record_step(RLStepRecord(
            timestamp=sim_time,
            state_key=state_key,
            state_label=state_label,
            action_idx=action_idx,
            alpha=alpha,
            reward_total=reward_total,
            reward_breakdown=reward_breakdown,
            epsilon=epsilon_before,
            was_exploration=was_exploration,
            q_values_at_state=q_values,
            tts_raw=current_tts,
            collisions_raw=collision_now,
        ))

        self._rl_prev_state_key = state_key
        self._rl_prev_action_idx = action_idx
        self._rl_prev_alpha = alpha
        self._rl_step_count += 1

    # Delegate state push + control receive to AsyncMPCController unchanged
    AsyncMPCController.on_step(self, step, env, state)


def _rl_stop(self, timeout: float = 5.0) -> None:
    """
    Finalise RL stats, save Q-table, then shut down the worker thread.

    Args:
        timeout: float, seconds to wait for the worker thread to finish
    """
    if self._rl_learning:
        self._rl_agent.decay_epsilon_episode()
    self._rl_stats.finalise(self._rl_agent)
    if self._rl_learning and self._rl_q_table_path is not None:
        self._rl_agent._total_episodes += 1
        self._rl_agent.save_q_table(self._rl_q_table_path)
        logger.info("[RL] Q-table saved to %s", self._rl_q_table_path)

        # per-episode snapshot for tracking Q-table evolution over training
        snapshot_path = (
            self._rl_q_table_path.parent / "q-tables"
            / f"q-table-ep-{self._rl_agent._total_episodes}.npy"
        )
        self._rl_agent.save_q_table(snapshot_path)
        logger.info("[RL] Q-table snapshot saved to %s", snapshot_path)
    AsyncMPCController.stop(self, timeout=timeout)


def _rl_print_pre_run_status(self) -> None:
    """Print RL agent status summary — call before manual_input_check."""
    agent = self._rl_agent
    cfg = agent._rl_config if hasattr(agent, "_rl_config") else {}
    print("\n" + "=" * 60)
    print("[RL] Agent Status")
    print("=" * 60)
    print(f"  Mode            : {'LEARNING' if self._rl_learning else 'EVALUATION (greedy)'}")
    print(f"  Epsilon (now)   : {agent.epsilon:.4f}  (initial: {agent._initial_epsilon:.4f})")
    print(f"  LR / gamma      : {agent._initial_lr}  /  {agent._initial_discount}")
    print(f"  Decay           : {agent._initial_decay} (min={agent._initial_min}, "
          f"enabled={agent._initial_decay_enabled}, mode={agent._initial_decay_mode})")
    print(f"  Actions (α)     : {agent.action_values}")
    print(f"  Total steps     : {agent._total_steps}")
    print(f"  Total episodes  : {agent._total_episodes}")
    print(f"  Q-table states  : {len(agent.q_table)}")
    if self._rl_q_table_path:
        print(f"  Q-table path    : {self._rl_q_table_path}")
    agent.display_q_table()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_reward_window(rl_config: dict) -> float:
    components = rl_config.get("reward", {}).get("components", {})
    windows = [float(v.get("window_seconds", 300)) for v in components.values()]
    return max(windows) if windows else 300.0


def _validate_solver_pool_timing(self) -> None:
    """Warn if any pooled solver's Np/Nc/M/solve_interval_steps disagree — they must share one MPC timing scheme."""
    pool = getattr(self, "_solver_pool", None)
    if not pool:
        return
    expected_steps = self._solve_interval_s / self.step_length
    for alpha, solver in pool.items():
        sc = getattr(solver.environment, "solver_config", {}) or {}
        got_steps = sc.get("solve_interval_steps")
        if got_steps is not None and not np.isclose(got_steps, expected_steps):
            logger.warning(
                "[RL] pooled solver \u03b1=%.2f has solve_interval_steps=%s but controller expects %.1f — "
                "pool members must share identical solve timing or MPC predictions will misalign.",
                alpha, got_steps, expected_steps,
            )
        for attr in ("Np", "Nc", "M"):
            expected_val = getattr(solver, attr, None)
            reference_val = getattr(next(iter(pool.values())), attr, None)
            if expected_val != reference_val:
                logger.warning(
                    "[RL] pooled solver \u03b1=%.2f has %s=%s, differing from other pool members (%s) — "
                    "check solver_factory consistency.",
                    alpha, attr, expected_val, reference_val,
                )