from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StateFeature(Protocol):
    """Extracts a scalar from the on_step state dict and discretises it to a label."""

    name: str
    thresholds: list[float]
    labels: list[str]

    def extract(self, state_data: dict) -> float: ...
    def discretise(self, value: float) -> str: ...


@dataclass
class ThresholdFeature:
    """
    Discretises a scalar into named bins using half-open intervals:
    (-inf, t0) → labels[0], [t0, t1) → labels[1], ..., [t_{n-1}, +inf) → labels[n].
    len(labels) must equal len(thresholds) + 1.
    """

    name: str
    thresholds: list[float]
    labels: list[str]
    _extract_fn: Callable[[dict], float] = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.thresholds) + 1:
            raise ValueError(
                f"Feature '{self.name}': need len(labels) == len(thresholds)+1, "
                f"got {len(self.labels)} labels and {len(self.thresholds)} thresholds."
            )

    def extract(self, state_data: dict) -> float:
        return float(self._extract_fn(state_data))

    def discretise(self, value: float) -> str:
        for threshold, label in zip(self.thresholds, self.labels):
            if value < threshold:
                return label
        return self.labels[-1]

    def discretise_from_state(self, state_data: dict) -> str:
        return self.discretise(self.extract(state_data))


class MeanDensityFeature(ThresholdFeature):
    """Mean density over a configurable set of segment indices."""

    def __init__(
        self,
        segment_indices: list[int],
        thresholds: list[float],
        labels: list[str],
        name: str = "mean_critical_density",
    ) -> None:
        super().__init__(
            name=name,
            thresholds=thresholds,
            labels=labels,
            _extract_fn=lambda s: float(
                np.mean(np.asarray(s["rho"], dtype=float).flatten()[segment_indices])
            ),
        )
        self.segment_indices = segment_indices


def feature_from_config(cfg: dict) -> ThresholdFeature:
    """
    Build a ThresholdFeature from a config dict.

    Supported 'source' values:
      - "mean_density": mean of rho over segment_indices
      - "mean_speed":   mean of v over segment_indices
      - "ramp_queue":   sum of w over origin_indices (default [1])
      - "tts_trend":    sign/magnitude of tts_increase (already computed by the controller for reward)
    """
    source: str = cfg["source"]
    thresholds: list[float] = [float(t) for t in cfg["thresholds"]]
    labels: list[str] = list(cfg["labels"])
    name: str = cfg.get("name", source)

    if source == "mean_density":
        indices = [int(i) for i in cfg["segment_indices"]]
        return MeanDensityFeature(
            segment_indices=indices,
            thresholds=thresholds,
            labels=labels,
            name=name,
        )

    if source == "mean_speed":
        indices = [int(i) for i in cfg["segment_indices"]]
        return ThresholdFeature(
            name=name,
            thresholds=thresholds,
            labels=labels,
            _extract_fn=lambda s, idx=indices: float(
                np.mean(np.asarray(s["v"], dtype=float).flatten()[idx])
            ),
        )

    if source == "ramp_queue":
        indices = [int(i) for i in cfg.get("origin_indices", [1])]
        return ThresholdFeature(
            name=name,
            thresholds=thresholds,
            labels=labels,
            _extract_fn=lambda s, idx=indices: float(
                np.sum(np.asarray(s["w"], dtype=float).flatten()[idx])
            ),
        )

    if source == "tts_trend":
        # relies on 'tts_increase' already being injected into state_data by the RL controller
        return ThresholdFeature(
            name=name,
            thresholds=thresholds,
            labels=labels,
            _extract_fn=lambda s: float(s.get("tts_increase", 0.0)),
        )

    raise ValueError(
        f"Unknown state feature source '{source}'. "
        "Supported: 'mean_density', 'mean_speed', 'ramp_queue', 'tts_trend'."
    )


class StateDiscretiser:
    """
    Converts a state_data dict into a hashable tuple key using one or more features.
    Each element of the tuple is the discretised label for that feature dimension.
    """

    def __init__(self, features: list[ThresholdFeature]) -> None:
        if not features:
            raise ValueError("StateDiscretiser requires at least one feature.")
        self.features = features

    @classmethod
    def from_config(cls, state_cfg: dict) -> "StateDiscretiser":
        features = [feature_from_config(f) for f in state_cfg["features"]]
        return cls(features)

    def key(self, state_data: dict) -> tuple[str, ...]:
        return tuple(f.discretise_from_state(state_data) for f in self.features)

    def label(self, state_data: dict) -> str:
        """Human-readable compound label, e.g. 'low' or 'medium|high'."""
        parts = [f.discretise_from_state(state_data) for f in self.features]
        return "|".join(parts)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def all_labels(self) -> list[list[str]]:
        return [f.labels for f in self.features]
