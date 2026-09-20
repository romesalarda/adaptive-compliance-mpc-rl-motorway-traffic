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
    Penalises increase in Total Time Spent over the solve window.
    state_data must contain 'tts_increase' (veh·h), pre-computed by the controller.
    Negative because we minimise TTS growth; reward is negated cost.
    """

    weight: float = 1.0
    name: str = "tts"
    scale: float = 1.0

    def calculate(self, state_data: dict) -> float:
        tts_increase = float(state_data.get("tts_increase", 0.0))
        # normalise by 'scale' so magnitude is comparable across scenarios with different baseline TTS
        return -tts_increase / self.scale


@dataclass
class CollisionRewardComponent:
    """
    Penalises collisions detected within the observation window.
    state_data must contain 'collisions_in_window' (int), pre-computed by the controller.
    """

    weight: float = 0.5
    name: str = "collisions"
    scale: float = 1.0

    def calculate(self, state_data: dict) -> float:
        collisions = float(state_data.get("collisions_in_window", 0))
        # normalise by 'scale' so magnitude is comparable to other reward components
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
        if "scale" in comp_cfg and klass in (TTSRewardComponent, CollisionRewardComponent):
            kwargs["scale"] = float(comp_cfg["scale"])
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
