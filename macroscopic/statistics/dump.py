from __future__ import annotations

import dataclasses
import datetime
import json
import time as _time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from macroscopic.statistics.base import MacroSimulationResults

# ---------------------------------------------------------------------------
# JSON-safe type conversion
# ---------------------------------------------------------------------------

def _numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy / CasADi types to JSON-serialisable Python types."""
    # CasADi DM (numeric dense matrix)
    try:
        import casadi as cs
        if isinstance(obj, cs.DM):
            return np.array(obj).tolist()
    except ImportError:
        pass

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------

_BASE_PARAM_KEYS = [
    "T", "Tfin", "L", "lanes", "C", "tau", "kappa", "eta",
    "rho_max", "delta", "rho_crit", "v_free", "a",
    "rho_init", "v_free_init", "w_init", "r_init",
]

_VSL_PARAM_KEYS = [
    "alpha_compliance", "A", "E", "V", "v_ctrl",
    "segments_with_vsl", "minimum_speed_limit", "maximum_speed_limit",
]

# Link attributes present on every sym_metanet Link
_LINK_BASE_ATTRS = [
    "N", "L", "lam",
    "rho_max", "rho_crit", "v_free", "a",
    "turnrate"
]

# Optional VSL-specific attributes on LinkWithVsl / subclasses
_LINK_VSL_ATTRS = ["segments_with_vsl", "alpha", "V", "A", "E"]

# Candidate capacity attribute names on origin elements
_ORIGIN_CAPACITY_ATTRS = ["capacity", "C", "max_flow"]

_DEMANDS_SUBSAMPLE_N = 50


def _serialize_parameters(parameters) -> dict:
    param_type = type(parameters).__name__
    keys = _BASE_PARAM_KEYS[:]
    if param_type == "VslParameters":
        keys += _VSL_PARAM_KEYS

    data: dict = {"_type": param_type}
    for key in keys:
        data[key] = _numpy_to_python(getattr(parameters, key, None))
    return data


def _serialize_network(network) -> dict:
    links: dict = {}
    for name, link in network.links_by_name.items():
        link_data: dict = {"_type": type(link).__name__}
        for attr in _LINK_BASE_ATTRS:
            link_data[attr] = _numpy_to_python(getattr(link, attr, None))
        for attr in _LINK_VSL_ATTRS:
            val = getattr(link, attr, None)
            if val is not None:
                link_data[attr] = _numpy_to_python(val)
        links[name] = link_data

    nodes: dict = {
        name: {"_type": type(node).__name__}
        for name, node in network.nodes_by_name.items()
    }

    origins: dict = {}
    for name, origin in network.origins_by_name.items():
        origin_data: dict = {"_type": type(origin).__name__}
        for attr in _ORIGIN_CAPACITY_ATTRS:
            val = getattr(origin, attr, None)
            if val is not None:
                origin_data["capacity"] = _numpy_to_python(val)
                break
        origins[name] = origin_data

    destinations: dict = {
        name: {"_type": type(dest).__name__}
        for name, dest in network.destinations_by_name.items()
    }

    # Topology edges as ordered list
    edges: list = []
    for u_node, d_node, link in network.links:
        edges.append({
            "from": u_node.name,
            "link": link.name,
            "to": d_node.name,
        })

    # Origins attached to which node
    origin_nodes: dict = {}
    for origin, node in network.origins.items():
        origin_nodes[origin.name] = node.name

    return {
        "name": getattr(network, "name", "unknown"),
        "links": links,
        "nodes": nodes,
        "origins": origins,
        "origin_nodes": origin_nodes,
        "destinations": destinations,
        "topology": edges,
    }


def _serialize_demands(demands: np.ndarray, demands_meta: Optional[dict]) -> dict:
    """
    Stores demands compactly. If `demands_meta` is provided (e.g. interp keypoints),
    that dict is stored verbatim. Otherwise the array is subsampled to at most
    _DEMANDS_SUBSAMPLE_N evenly-spaced rows.
    """
    if demands_meta is not None:
        return {"_source": "keypoints", **demands_meta}

    n = demands.shape[0]
    step = max(1, n // _DEMANDS_SUBSAMPLE_N)
    indices = list(range(0, n, step))
    return {
        "_source": "subsampled",
        "original_shape": list(demands.shape),
        "subsample_count": len(indices),
        "indices": indices,
        "values": demands[indices].tolist(),
    }


def _channel_summary(arr: Any) -> Optional[dict]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return None
    return {
        "shape": list(a.shape),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _serialize_statistics(sim_results, overview_results, run_results) -> dict:
    overview: dict = {}
    for field in dataclasses.fields(overview_results):
        overview[field.name] = _numpy_to_python(getattr(overview_results, field.name))

    run: dict = {}
    for field in dataclasses.fields(run_results):
        run[field.name] = _numpy_to_python(getattr(run_results, field.name))

    channels: dict = {}
    for channel, arr in (
        ("rho", sim_results.rho),
        ("v", sim_results.v),
        ("w", sim_results.w),
        ("q", sim_results.q),
        ("q_on", sim_results.q_on),
        ("v_ctrl", sim_results.v_ctrl),
        ("r", sim_results.r),
    ):
        channels[channel] = _channel_summary(arr)

    return {"overview": overview, "run": run, "channels": channels}


# ---------------------------------------------------------------------------
# HDF5 timeseries dump / load
# ---------------------------------------------------------------------------

# Channels stored verbatim; absent (None / empty) channels are skipped.
_HDF5_CHANNELS = ("rho", "v", "w", "q", "q_on", "v_ctrl", "r")


def _dump_timeseries_hdf5(sim_results, path: Path) -> None:
    """Write all MacroSimulationResults arrays to an HDF5 file for later replay."""
    import h5py

    with h5py.File(path, "w") as hf:
        ch = hf.create_group("channels")
        for name in _HDF5_CHANNELS:
            arr = getattr(sim_results, name, None)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float64)
            if a.size == 0:
                continue
            ch.create_dataset(name, data=a, compression="gzip", compression_opts=4)

        demands = getattr(sim_results, "demands", None)
        if demands is not None:
            a = np.asarray(demands, dtype=np.float64)
            if a.size > 0:
                hf.create_dataset("demands", data=a, compression="gzip", compression_opts=4)

        meta = hf.create_group("metadata")
        origin_names = getattr(sim_results, "origin_names", None) or []
        # h5py requires a fixed-length or variable-length string dtype
        dt = h5py.string_dtype()
        meta.create_dataset("origin_names", data=np.array(origin_names, dtype=object), dtype=dt)


def load_timeseries(run_dir: str) -> MacroSimulationResults:
    """
    Reconstruct a :class:`MacroSimulationResults` from a run's ``timeseries.h5`` file.

    Args:
        run_dir: Path to a run folder produced by :meth:`RunSet.new_run`.

    Returns:
        Populated ``MacroSimulationResults`` with all squeezed arrays restored.
    """
    import h5py

    path = Path(run_dir) / "timeseries.h5"
    if not path.exists():
        raise FileNotFoundError(f"timeseries.h5 not found in '{run_dir}'")

    kwargs: dict = {}
    with h5py.File(path, "r") as hf:
        ch = hf.get("channels", {})
        for name in _HDF5_CHANNELS:
            if name in ch:
                kwargs[name] = np.array(ch[name])

        if "demands" in hf:
            kwargs["demands"] = np.array(hf["demands"])

        origin_names = []
        if "metadata" in hf and "origin_names" in hf["metadata"]:
            origin_names = [n.decode() if isinstance(n, bytes) else n
                            for n in hf["metadata"]["origin_names"]]

    result = MacroSimulationResults(**kwargs)
    result.origin_names = origin_names
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# RunSet
# ---------------------------------------------------------------------------

class RunSet:
    """
    Manages a named collection of simulation runs stored under ``base_dir/name/``.

    On first use the folder is created. Subsequent uses append to the existing
    ``index.json`` so runs accumulate a sequential identifier per run type
    (e.g. ``rm_001``, ``rm_002``, ``vsl_001`` …).

    Example usage::

        demands_meta = {
            "type": "interp",
            "sources": ["mainline", "on_ramp"],
            "xp": [0.0, 0.5, 1.0, 1.5, 2.5],
            "fp": [[2800, 3000, 3500, 3000, 2800], [200, 600, 800, 600, 200]],
        }

        run_set = RunSet("mpc_tests")
        run_set.new_run(
            solver=solver,
            run_type="rm",
            figures={"ramp_metering": fig},
            demands_meta=demands_meta,
        )
    """

    _INDEX_FILENAME = "index.json"

    def __init__(self, name: str, base_dir: str = "results") -> None:
        self.name = name
        self.base_dir = Path(base_dir) / name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict = self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.base_dir / self._INDEX_FILENAME

    def _load_index(self) -> dict:
        path = self._index_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"name": self.name, "runs": []}

    def _save_index(self) -> None:
        _write_json(self._index_path(), self._index)

    def _next_run_id(self, run_type: str) -> str:
        count = sum(
            1 for r in self._index["runs"]
            if r.get("run_type") == run_type
        )
        return f"{run_type}_{count + 1:03d}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_run(
        self,
        solver: Any,
        run_type: str,
        figures: Optional[dict] = None,
        demands_meta: Optional[dict] = None,
        extra_meta: Optional[dict] = None,
        save_objective_weights: bool = True,
    ) -> str:
        """
        Serialise a completed simulation run to disk.

        Args:
            solver:        Any ``BaseSimulation`` subclass after ``.simulate()`` has been called.
            run_type:      Short label used for folder naming, e.g. ``"rm"``, ``"vsl"``, ``"base"``.
            figures:       ``{name: matplotlib.figure.Figure}`` — saved as ``figures/<name>.png``.
            demands_meta:  Optional dict of demand keypoints. Falls back to 50-point subsample.
            extra_meta:    Optional extra fields merged into ``meta.json`` for extensibility.

        Returns:
            Absolute path to the run folder as a string.
        """
        t0 = _time.perf_counter()

        run_id = self._next_run_id(run_type)
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        env = solver.environment
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # ---- meta.json ----
        meta: dict = {
            "run_id": run_id,
            "run_type": run_type,
            "timestamp": timestamp,
            "solver_class": type(solver).__name__,
            "environment_id": env.id,
            "network_name": getattr(env.network, "name", "unknown"),
            "simulation_time_s": solver.run_results.total_simulation_time,

            "solver_config": _numpy_to_python({
                "Np": solver.Np,
                "Nc": solver.Nc,
                "M": solver.M,
            }),
        }
        if extra_meta:
            meta.update(extra_meta)

        if save_objective_weights and hasattr(solver, "objective_weights"):
            meta["objective_weights"] = _numpy_to_python(solver.objective_weights)
            
        _write_json(run_dir / "meta.json", meta)

        # ---- parameters.json ----            
        _write_json(run_dir / "parameters.json", _serialize_parameters(env.parameters))

        # ---- network.json ----
        _write_json(run_dir / "network.json", _serialize_network(env.network))

        # ---- demands.json ----
        _write_json(run_dir / "demands.json", _serialize_demands(env.demands, demands_meta))

        # ---- statistics.json ----
        _write_json(
            run_dir / "statistics.json",
            _serialize_statistics(
                solver.simulation_results,
                solver.overview_results,
                solver.run_results,
            ),
        )

        # ---- timeseries.h5 ----
        _dump_timeseries_hdf5(solver.simulation_results, run_dir / "timeseries.h5")

        # ---- figures/ ----
        if figures:
            figures_dir = run_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            for fig_name, fig in figures.items():
                fig.savefig(
                    figures_dir / f"{fig_name}.png",
                    dpi=150,
                    bbox_inches="tight",
                )

        # ---- update index ----
        self._index["runs"].append({
            "run_id": run_id,
            "run_type": run_type,
            "timestamp": timestamp,
            "solver_class": type(solver).__name__,
            "path": str(run_dir.resolve()),
        })
        self._save_index()

        elapsed = _time.perf_counter() - t0
        print(f"[RunSet] '{run_id}' saved to '{run_dir}' ({elapsed:.2f}s)")

        return str(run_dir.resolve())
