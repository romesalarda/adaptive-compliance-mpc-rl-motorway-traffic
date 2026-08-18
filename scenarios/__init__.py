from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from macroscopic.simulation.environment import MacroEnvironment
from macroscopic.simulation.parameters import BaseParameters, VslParameters

if TYPE_CHECKING:
    from sym_metanet import Network, Link, LinkWithVsl
    from microscopic.simulation.environment import MicroEnvironment

_SCENARIOS_DIR = Path(__file__).parent

# Maps control type to the Parameters class that should be instantiated
_PARAMS_REGISTRY: dict[str, type[BaseParameters]] = {
    "base": BaseParameters,
    "rm": BaseParameters,
    "vsl": VslParameters,
    "coordinated": VslParameters,
}

# Keys within a control_parameters block that are NOT model parameters
_RESERVED_CONTROL_KEYS = {"objective_weights", "solver_config", "link_class"}

# Maps link_class string names to their implementations
def _get_link_class_registry() -> dict:
    from macroscopic.vsl import CarlsonLinkWithVsl, FrejoLinkWithVsl
    import casadi as cs
    from sym_metanet import Link
    return {
        "CarlsonLinkWithVsl": CarlsonLinkWithVsl,
        "FrejoLinkWithVsl": FrejoLinkWithVsl,
        "Link": Link[cs.SX],
        "LinkWithVsl": LinkWithVsl[cs.SX],
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

    @property
    def micro_config(self) -> dict:
        """Raw micro block from scenario.json; raises if missing."""
        cfg = self._config.get("micro")
        if cfg is None:
            raise KeyError(f"Scenario '{self.name}' has no 'micro' block in scenario.json")
        return cfg

    def build_micro_environment(self) -> "MicroEnvironment":
        """Instantiate a MicroEnvironment from the scenario's 'micro' block."""
        from microscopic.simulation.environment import MicroEnvironment

        cfg = self.micro_config
        return MicroEnvironment(
            edge_to_segment_map={k: int(v) for k, v in cfg["edge_to_segment_map"].items()},
            origin_w_map={k: int(v) for k, v in cfg["origin_w_map"].items()},
            n_segments=int(cfg["n_segments"]),
            n_origins=int(cfg["n_origins"]),
            step_length=float(cfg["step_length"]),
            vsl_edge_ids=list(cfg.get("vsl_edge_ids", [])),
            ramp_tl_map=dict(cfg.get("ramp_tl_map", {})),
        )

    def build_environment(self, control_type: str, **kwargs) -> MacroEnvironment:
        """Merge parameters, instantiate the right Parameters class, build network and demands."""
        if control_type not in _PARAMS_REGISTRY:
            raise ValueError(
                f"Unknown control type '{control_type}'. "
                f"Choose from: {list(_PARAMS_REGISTRY)}"
            )

        params_cls = _PARAMS_REGISTRY[control_type]
        params_dict = dict(self._config.get("parameters", {}))
        overrides = dict(self._config.get("control_parameters", {}).get(control_type, {}))

        objective_weights = overrides.pop("objective_weights", {})
        solver_config = overrides.pop("solver_config", {})
        link_class_name: str | None = overrides.pop("link_class", None)

        params_dict.update(overrides)
        parameters = params_cls(**params_dict)

        link_class = None
        if link_class_name is not None:
            registry = _get_link_class_registry()
            if link_class_name not in registry:
                raise ValueError(
                    f"Unknown link_class '{link_class_name}'. "
                    f"Available: {list(registry)}"
                )
            link_class = registry[link_class_name]

        network = self._build_network(parameters, control_type, link_class=link_class, **kwargs)
        demands = self._build_demands(parameters.time)

        return MacroEnvironment(
            parameters=parameters,
            network=network,
            demands=demands,
            objective_weights=objective_weights,
            solver_config=solver_config,
            link_class_name=link_class_name,
        )

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

    def _build_network(self, parameters: BaseParameters, control_type: str, link_class=None, **kwargs) -> "Network":
        network_path = self.path / "network.py"
        spec = importlib.util.spec_from_file_location(
            f"scenarios.{self.name}.network", network_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_network(parameters, control_type, linkKlass=link_class, **kwargs)

    def _build_demands(self, time: np.ndarray) -> np.ndarray:
        cfg = self._config["demands"]
        if cfg["type"] != "interp":
            raise NotImplementedError(f"Demand type '{cfg['type']}' is not supported.")
        sources = [np.interp(time, cfg["xp"], fp) for fp in cfg["fp"]]
        return np.stack(sources).T
