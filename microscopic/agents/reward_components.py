from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardComponent(Protocol):
    name: str
    weight: float

    def calculate(self, state_data: dict) -> float: ...


@dataclass
class TTSRewardComponent:
    """
    Penalises the Total Time Spent accrued over the last decision interval.
    state_data must contain 'tts_interval' (veh?h): the time-integral of vehicles in the
    system (mainline + ramp queues) between the previous and the current decision.
    """

    weight: float = 1.0
    name: str = "tts"
    scale: float = 1.0

    def calculate(self, state_data: dict) -> float:
        return -float(state_data.get("tts_interval", 0.0)) / self.scale


@dataclass
class CollisionRewardComponent:
    """
    Penalises collisions that occurred during the last decision interval.
    state_data must contain 'collisions_interval' (int), counted every simulation step.
    The count is clipped at 'clip' so rare pile-ups cannot swamp the TTS term.
    """

    weight: float = 0.5
    name: str = "collisions"
    scale: float = 1.0
    clip: float | None = None

    def calculate(self, state_data: dict) -> float:
        collisions = float(state_data.get("collisions_interval", 0))
        if self.clip is not None:
            collisions = min(collisions, self.clip)
        return -collisions / self.scale


def reward_from_config(cfg: dict) -> "WeightedRewardCalculator":
    """Build a WeightedRewardCalculator from the 'reward' block in rl config."""
    _registry: dict[str, type] = {
        "tts": TTSRewardComponent,
        "collisions": CollisionRewardComponent,
    }
    components: list[RewardComponent] = []
    for comp_name, comp_cfg in cfg.get("components", {}).items():
        if comp_name not in _registry:
            raise ValueError(
                f"Unknown reward component '{comp_name}'. Supported: {list(_registry)}"
            )
        klass = _registry[comp_name]
        weight = float(comp_cfg.get("weight", 1.0))
        kwargs = {"weight": weight}
        if "scale" in comp_cfg:
            kwargs["scale"] = float(comp_cfg["scale"])
        if "clip" in comp_cfg and klass is CollisionRewardComponent:
            kwargs["clip"] = float(comp_cfg["clip"])
        components.append(klass(**kwargs))

    return WeightedRewardCalculator(components)


class WeightedRewardCalculator:
    """Sums weighted reward components; returns total and per-component breakdown."""

    def __init__(self, components: list[RewardComponent]) -> None:
        if not components:
            raise ValueError("WeightedRewardCalculator requires at least one component.")
        self.components = components

    def calculate(self, state_data: dict) -> tuple[float, dict[str, float]]:
        """Returns (total_reward, {component_name: weighted_value})."""
        breakdown: dict[str, float] = {}
        total = 0.0
        for comp in self.components:
            raw = comp.calculate(state_data)
            weighted = comp.weight * raw
            breakdown[comp.name] = weighted
            total += weighted
        return total, breakdown
