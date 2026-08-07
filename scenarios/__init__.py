from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from macroscopic.simulation.environment import Environment
from macroscopic.simulation.parameters import BaseParameters, VslParameters

if TYPE_CHECKING:
    from sym_metanet import Network

_SCENARIOS_DIR = Path(__file__).parent

# Maps control type to the Parameters class that should be instantiated
_PARAMS_REGISTRY: dict[str, type[BaseParameters]] = {
    "base": BaseParameters,
    "rm": BaseParameters,
    "vsl": VslParameters,
}


class Scenario:
    """Loads a scenario folder and builds an Environment for a given control type."""

    def __init__(self, name: str) -> None:
        self.path = _SCENARIOS_DIR / name
        if not self.path.is_dir():
            available = ", ".join(Scenario.list_scenarios())
            raise FileNotFoundError(
                f"Scenario '{name}' not found. Available: {available}"
            )

        config_path = self.path / "scenario.json"
        with config_path.open() as f:
            self._config: dict = json.load(f)

        self.name = name
        self.display_name: str = self._config.get("name", name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_environment(self, control_type: str) -> Environment:
        """Merge parameters, instantiate the right Parameters class, build network and demands."""
        if control_type not in _PARAMS_REGISTRY:
            raise ValueError(
                f"Unknown control type '{control_type}'. "
                f"Choose from: {list(_PARAMS_REGISTRY)}"
            )

        params_cls = _PARAMS_REGISTRY[control_type]
        params_dict = dict(self._config.get("parameters", {}))
        overrides = self._config.get("control_parameters", {}).get(control_type, {})
        params_dict.update(overrides)

        parameters = params_cls(**params_dict)

        network = self._build_network(parameters, control_type)
        demands = self._build_demands(parameters.time)

        return Environment(parameters=parameters, network=network, demands=demands)

    @property
    def demands_meta(self) -> dict:
        """Raw demands config for use with RunSet.new_run(demands_meta=...)."""
        return self._config["demands"]

    @staticmethod
    def list_scenarios() -> list[str]:
        return [
            d.name
            for d in _SCENARIOS_DIR.iterdir()
            if d.is_dir() and (d / "scenario.json").exists()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_network(self, parameters: BaseParameters, control_type: str) -> "Network":
        network_path = self.path / "network.py"
        spec = importlib.util.spec_from_file_location(
            f"scenarios.{self.name}.network", network_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_network(parameters, control_type)

    def _build_demands(self, time: np.ndarray) -> np.ndarray:
        cfg = self._config["demands"]
        if cfg["type"] != "interp":
            raise NotImplementedError(f"Demand type '{cfg['type']}' is not supported.")
        sources = [np.interp(time, cfg["xp"], fp) for fp in cfg["fp"]]
        return np.stack(sources).T
