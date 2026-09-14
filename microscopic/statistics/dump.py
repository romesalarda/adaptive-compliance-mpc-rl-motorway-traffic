"""
Serialisation helpers for micro simulation results.

HDF5 layout mirrors the macroscopic statistics format for easy combined analysis:
  /run/density        (n_steps, n_segments)  float32
  /run/speed          (n_steps, n_segments)  float32
  /run/queue          (n_steps, n_origins)   float32
  /run/timestamps     (n_steps,)             float32
  /run/steps          (n_steps,)             int32
  /run/v_ctrl_applied (n_solves, n_vsl)      float32  [if present]
  /run/r_applied      (n_solves, n_metered)  float32  [if present]
  /run/mpc_solve_times (n_solves,)           float32  [if present]
  /run/solve_steps     (n_solves,)           int32    [if present]
  /overview/<key>     scalar
  /meta/<key>         string / scalar

SUMO output XML parsers return plain dicts to keep things dependency-light.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from microscopic.statistics.base import MicroSimulationResults


# ---------------------------------------------------------------------------
# HDF5 save / load
# ---------------------------------------------------------------------------

def save_hdf5(results: MicroSimulationResults, path: str | Path) -> None:
    """Serialise a MicroSimulationResults to HDF5."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # --- run group ---
        rg = f.create_group("run")
        _write_array(rg, "steps", results.run.steps, np.int32)
        _write_array(rg, "timestamps", results.run.timestamps, np.float32)

        if results.density is not None:
            rg.create_dataset("density", data=results.density.astype(np.float32), compression="gzip")
        if results.speed is not None:
            rg.create_dataset("speed", data=results.speed.astype(np.float32), compression="gzip")
        if results.queue is not None:
            rg.create_dataset("queue", data=results.queue.astype(np.float32), compression="gzip")

        if results.run.v_ctrl_applied:
            _write_array(rg, "v_ctrl_applied", results.run.v_ctrl_applied, np.float32, stack=True)
        if results.run.r_applied:
            _write_array(rg, "r_applied", results.run.r_applied, np.float32, stack=True)
        if results.run.mpc_solve_times:
            _write_array(rg, "mpc_solve_times", results.run.mpc_solve_times, np.float32)
        if results.run.solve_steps:
            _write_array(rg, "solve_steps", results.run.solve_steps, np.int32)
        if results.run.solve_timestamps:
            _write_array(rg, "solve_timestamps", results.run.solve_timestamps, np.float64)

        # --- overview group ---
        og = f.create_group("overview")
        ov = results.overview
        for attr in (
            "total_time_spent", "total_distance_travelled", "total_waiting",
            "total_collisions", "mean_density", "mean_speed",
            "mean_mpc_solve_time", "runtime_s",
        ):
            og.attrs[attr] = getattr(ov, attr)

        # --- meta group ---
        mg = f.create_group("meta")
        mg.attrs["scenario_id"] = results.scenario_id
        mg.attrs["solver_type"] = results.solver_type
        mg.attrs["run_id"] = results.run_id
        mg.attrs["timestamp"] = results.timestamp


def load_hdf5(path: str | Path) -> MicroSimulationResults:
    """Deserialise a MicroSimulationResults from HDF5."""
    import h5py
    from microscopic.statistics.base import (
        MicroSimulationResults,
        MicroRunSimulationResults,
        MicroOverviewSimulationResults,
    )

    path = Path(path)
    results = MicroSimulationResults()

    with h5py.File(path, "r") as f:
        rg = f["run"]
        results.density = rg["density"][:] if "density" in rg else None
        results.speed = rg["speed"][:] if "speed" in rg else None
        results.queue = rg["queue"][:] if "queue" in rg else None

        run = MicroRunSimulationResults()
        run.steps = rg["steps"][:].tolist() if "steps" in rg else []
        run.timestamps = rg["timestamps"][:].tolist() if "timestamps" in rg else []
        run.mpc_solve_times = rg["mpc_solve_times"][:].tolist() if "mpc_solve_times" in rg else []
        run.solve_steps = rg["solve_steps"][:].tolist() if "solve_steps" in rg else []
        run.solve_timestamps = rg["solve_timestamps"][:].tolist() if "solve_timestamps" in rg else []
        results.run = run

        ov = MicroOverviewSimulationResults()
        for attr, val in f["overview"].attrs.items():
            setattr(ov, attr, val)
        results.overview = ov

        meta = f["meta"].attrs
        results.scenario_id = str(meta.get("scenario_id", ""))
        results.solver_type = str(meta.get("solver_type", ""))
        results.run_id = str(meta.get("run_id", ""))
        results.timestamp = str(meta.get("timestamp", ""))

    return results


def _write_array(group, name: str, data, dtype, stack: bool = False) -> None:
    if not data:
        return
    arr = np.stack(data).astype(dtype) if stack else np.asarray(data, dtype=dtype)
    group.create_dataset(name, data=arr, compression="gzip")


# ---------------------------------------------------------------------------
# SUMO XML output parsers
# ---------------------------------------------------------------------------

def parse_tripinfo(xml_path: str | Path) -> dict:
    """
    Parse SUMO tripinfo.xml output.

    Returns:
        tts_s          – total time spent (sum of durations) in seconds
        ttd_m          – total distance travelled in metres
        total_waiting_s – total waiting time in seconds
        n_vehicles     – number of completed trips
        mean_duration_s
        mean_speed_ms
    """
    path = Path(xml_path)
    if not path.exists():
        return {}

    tree = ET.parse(path)
    root = tree.getroot()

    durations, distances, waiting = [], [], []
    for trip in root.iter("tripinfo"):
        durations.append(float(trip.get("duration", 0)))
        distances.append(float(trip.get("routeLength", 0)))
        waiting.append(float(trip.get("waitingTime", 0)))

    n = len(durations)
    return {
        "tts_s": float(sum(durations)),
        "ttd_m": float(sum(distances)),
        "total_waiting_s": float(sum(waiting)),
        "n_vehicles": n,
        "mean_duration_s": float(np.mean(durations)) if n else 0.0,
        "mean_speed_ms": float(sum(distances) / sum(durations)) if sum(durations) > 0 else 0.0,
    }


def parse_collision(xml_path: str | Path) -> dict:
    """
    Parse SUMO collision output XML.

    Returns:
        num_collisions – total number of collision events
        events         – list of {time, type, collider, victim}
    """
    path = Path(xml_path)
    if not path.exists():
        return {"num_collisions": 0, "events": []}

    tree = ET.parse(path)
    root = tree.getroot()

    events = [
        {
            "time": float(col.get("time", 0)),
            "type": col.get("type", ""),
            "collider": col.get("collider", ""),
            "victim": col.get("victim", ""),
        }
        for col in root.iter("collision")
    ]
    return {"num_collisions": len(events), "events": events}


def parse_queue(xml_path: str | Path) -> dict:
    """
    Parse SUMO queue output XML (from --queue-output).

    Returns:
        times     – list of simulation times
        queues    – dict {lane_id: list of queue lengths over time}
    """
    path = Path(xml_path)
    if not path.exists():
        return {"times": [], "queues": {}}

    tree = ET.parse(path)
    root = tree.getroot()

    times: list[float] = []
    queues: dict[str, list[float]] = {}

    for data in root.iter("data"):
        t = float(data.get("timestep", 0))
        times.append(t)
        for lane in data.iter("lane"):
            lid = lane.get("id", "")
            q = float(lane.get("queueing_length", 0))
            queues.setdefault(lid, []).append(q)

    return {"times": times, "queues": queues}


# ---------------------------------------------------------------------------
# Run metadata / index  (mirrors macro RunSet conventions)
# ---------------------------------------------------------------------------


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def save_run_metadata(
    out_dir: Path,
    stats,
    macro_env,
    micro_cfg: dict,
    experiment: str = "default",
    solver=None,
    heterogeneity: str | None = None,
    demand_descriptor: str | None = None,
) -> None:
    """Save meta.json, parameters.json, network.json, demands.json, micro_config.json."""
    from macroscopic.statistics.dump import (
        serialize_parameters, serialize_network, serialize_demands, numpy_to_python,
    )
    out_dir = Path(out_dir)

    meta: dict = {
        "run_id": stats.run_id,
        "run_type": stats.solver_type,
        "timestamp": stats.timestamp,
        "scenario_id": stats.scenario_id,
        "experiment": experiment,
        "solver_class": type(solver).__name__ if solver is not None else None,
        "heterogeneity": heterogeneity,
        "demand_descriptor": demand_descriptor,
    }
    if solver is not None:
        meta["solver_config"] = numpy_to_python({
            "Np": getattr(solver, "Np", None),
            "Nc": getattr(solver, "Nc", None),
            "M": getattr(solver, "M", None),
        })
        if hasattr(solver, "objective_weights"):
            meta["objective_weights"] = numpy_to_python(solver.objective_weights)

    _write_json(out_dir / "meta.json", meta)
    _write_json(out_dir / "parameters.json", serialize_parameters(macro_env.parameters))
    _write_json(out_dir / "network.json", serialize_network(macro_env.network))
    _write_json(out_dir / "demands.json", serialize_demands(macro_env.demands, None))
    _write_json(out_dir / "micro_config.json", micro_cfg)


def save_error_report(
    out_dir: Path,
    exc: BaseException,
    scenario_id: str,
    control_type: str,
    run_id: str,
    experiment: str = "default",
    errors_root: str | Path = "Errors",
) -> None:
    """Record a crash's info in the run folder and in a top-level Errors/ folder.

    Called from the exception handler around a run so the run's failure (incl.
    KeyboardInterrupt) is still recoverable/traceable after the process exits.
    """
    import traceback
    from datetime import datetime

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario_id": scenario_id,
        "control_type": control_type,
        "run_id": run_id,
        "experiment": experiment,
        "run_path": str(Path(out_dir).resolve()),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "error.json", report)

    errors_dir = Path(errors_root)
    errors_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    _write_json(errors_dir / f"{ts_slug}_{scenario_id}_{control_type}_{run_id}.json", report)


def update_micro_index(experiment_dir: Path, entry: dict) -> None:
    """Append a run entry to the experiment-level index.json, creating it if absent."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    index_path = experiment_dir / "index.json"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
    else:
        index = {
            "name": entry.get("scenario_id", ""),
            "experiment": entry.get("experiment", ""),
            "runs": [],
        }

    index["runs"].append(entry)
    _write_json(index_path, index)
