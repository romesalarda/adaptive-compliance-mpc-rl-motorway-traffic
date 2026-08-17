"""
Entry point for closed-loop SUMO-MPC microscopic simulations.

Usage examples
--------------
Single run (ramp metering):
    python micro_main.py --scenario a2_motorway --solver rm --no-gui

Batch (2 workers, VSL + coordinated):
    python micro_main.py --scenario a2_motorway --solver vsl coordinated --batch --workers 2

Each run saves results to:
    runs/<scenario>/<solver>_<run_id>/
        simulation.h5    – per-step data and KPIs
        dashboard.html   – interactive Plotly dashboard
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
import uuid
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from microscopic.statistics.base import MicroSimulationResults

# ---- solver registry -------------------------------------------------------

def _build_solver(macro_env, control_type: str):
    """Instantiate and setup the macro solver for the given control type."""
    from macroscopic.solvers import MpcRMSimulation, MpcVSLSimulation, MpcCoordinatedSimulation, DummyMPCSolver

    common = dict(Np=7, Nc=3, M=6, function_compactness=2)

    if control_type == "rm":
        solver = MpcRMSimulation(environment=macro_env, **common)
    elif control_type == "vsl":
        solver = MpcVSLSimulation(environment=macro_env, **common)
    elif control_type == "coordinated":
        solver = MpcCoordinatedSimulation(environment=macro_env, Np=7, Nc=5, M=6,
                                           function_compactness=2,
                                           objective_weights=macro_env.objective_weights)
    elif control_type == "base":
        solver = DummyMPCSolver(environment=macro_env, **common)

    else:
        raise ValueError(f"Unknown solver type: {control_type!r}")

    solver.setup()
    return solver


# ---- single-run wiring ------------------------------------------------------

def run_single(
    scenario_name: str,
    control_type: str,
    no_gui: bool = True,
    output_root: str = "runs",
    port: int = 8813,
    show_dashboard: bool = True,
    use_async_controller: bool = True,
) -> "MicroSimulationResults":
    """
    Build and run one closed-loop SUMO-MPC simulation.
    Returns MicroSimulationResults after post-processing.
    """
    from scenarios import Scenario
    from microscopic.simulation.base import SimulationConfig, SUMOBaseSimulation
    from microscopic.controller.mpc import MPCController
    from microscopic.controller.async_mpc import AsyncMPCController
    from microscopic.statistics.base import MicroSimulationResults
    from microscopic.statistics.dump import save_hdf5, parse_tripinfo, parse_collision, parse_queue
    from microscopic.statistics.async_stats import save_async_stats_hdf5
    from microscopic.visualisation.base import MicroVisualiser

    scenario = Scenario(scenario_name)
    micro_cfg = scenario.micro_config

    macro_env = scenario.build_environment(control_type)
    micro_env = scenario.build_micro_environment()

    solver = _build_solver(macro_env, control_type)

    run_id = uuid.uuid4().hex[:8]
    out_dir = Path(output_root) / scenario_name / f"{control_type}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_config = SimulationConfig.from_scenario(
        scenario_path=str(Path("scenarios") / scenario_name),
        no_gui=no_gui,
        port=port,
    )
    # absolute paths so SUMO writes named outputs directly into the run directory
    sim_config.extra_args = [
        "--tripinfo-output", str(out_dir.resolve() / "tripinfo.xml"),
        "--collision-output", str(out_dir.resolve() / "collision.xml"),
        "--queue-output", str(out_dir.resolve() / "queue.xml"),
    ]

    stats = MicroSimulationResults(
        scenario_id=scenario_name,
        solver_type=control_type,
        run_id=run_id,
    )

    if use_async_controller:
        controller = AsyncMPCController(
            macro_env=macro_env,
            micro_env=micro_env,
            solver=solver,
            solve_interval_steps=micro_cfg["solve_interval_steps"],
            step_length=sim_config.step_length,
        )
    else:
        controller = MPCController(
            macro_env=macro_env,
            micro_env=micro_env,
            solver=solver,
            solve_interval_steps=micro_cfg["solve_interval_steps"],
        )

    sim = SUMOBaseSimulation(sim_config, micro_env, controller, stats)

    t0 = time.perf_counter()
    sim.setup()
    sim.run()
    sim.post_run()
    wall_time = time.perf_counter() - t0

    # finalize with geometry (all A2 segments are 1 km, 2 lanes)
    n_seg = micro_cfg["n_segments"]
    stats.finalize(
        step_length=sim_config.step_length,
        n_steps=sim._step,
        segment_lengths_km=[1.0] * n_seg,
        segment_lanes=[2] * n_seg,
        runtime_s=wall_time,
    )

    # parse SUMO output files
    sumo_data = {
        "tripinfo": parse_tripinfo(out_dir / "tripinfo.xml"),
        "collision": parse_collision(out_dir / "collision.xml"),
        "queue": parse_queue(out_dir / "queue.xml"),
    }
    stats.overview.total_collisions = sumo_data["collision"]["num_collisions"]

    save_hdf5(stats, out_dir / "simulation.h5")

    if use_async_controller and isinstance(controller, AsyncMPCController):
        async_path = out_dir / "async_mpc_stats.h5"
        save_async_stats_hdf5(controller.async_stats, async_path)
        n_completed = sum(1 for r in controller.async_stats.records if r.status == "completed")
        mean_solve = (
            sum(r.solve_time_reported for r in controller.async_stats.records) / n_completed
            if n_completed else 0.0
        )
        print(f"[async-mpc] {n_completed} cycles | mean solve={mean_solve:.3f}s | stats → {async_path}")

    if show_dashboard:
        vis = MicroVisualiser(stats, edge_labels=list(micro_cfg["edge_to_segment_map"].keys()))
        dash_path = vis.plot_dashboard(
            filename="dashboard.html",
            output_dir=str(out_dir),
            sumo_data=sumo_data,
        )
        print(f"Dashboard saved to {dash_path}")

        if use_async_controller and isinstance(controller, AsyncMPCController):
            async_dash_path = vis.plot_async_dashboard(
                async_stats=controller.async_stats,
                filename="async_dashboard.html",
                output_dir=str(out_dir),
            )
            print(f"Async dashboard saved to {async_dash_path}")

    print(
        f"[{scenario_name}/{control_type}] "
        f"TTS={stats.overview.total_time_spent:.1f} veh·h | "
        f"Collisions={stats.overview.total_collisions} | "
        f"Wall time={wall_time:.1f}s"
    )
    return stats


# ---- batch run --------------------------------------------------------------

def run_batch(
    scenario_name: str,
    control_types: list[str],
    n_workers: int = 2,
    base_port: int = 8813,
    output_root: str = "runs",
    no_gui: bool = True,
    use_async_controller: bool = True,
) -> list:
    """
    Run multiple control types in parallel using isolated processes.
    Returns list of MicroSimulationResults (collected via Queue).
    """
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    processes = []

    for i, control_type in enumerate(control_types):
        port = base_port + i

        # Factory must be module-level (picklable) for Windows spawn
        # We pass all args via partial-like wrapper defined below
        factory = _SimFactory(
            scenario_name=scenario_name,
            control_type=control_type,
            port=port,
            output_root=output_root,
            no_gui=no_gui,
            use_async_controller=use_async_controller,
        )
        p = multiprocessing.Process(
            target=_batch_worker,
            args=(factory, result_queue),
        )
        p.start()
        processes.append(p)
        # stagger launches slightly to avoid port conflicts
        time.sleep(0.5)

    results = []
    for p in processes:
        p.join()
        try:
            results.append(result_queue.get_nowait())
        except Exception:
            pass

    return results


class _SimFactory:
    """Picklable factory that creates and runs one simulation."""

    def __init__(self, scenario_name, control_type, port, output_root, no_gui, use_async_controller=True):
        self.scenario_name = scenario_name
        self.control_type = control_type
        self.port = port
        self.output_root = output_root
        self.no_gui = no_gui
        self.use_async_controller = use_async_controller

    def __call__(self):
        return run_single(
            scenario_name=self.scenario_name,
            control_type=self.control_type,
            no_gui=self.no_gui,
            output_root=self.output_root,
            port=self.port,
            show_dashboard=True,
            use_async_controller=self.use_async_controller,
        )


def _batch_worker(factory: _SimFactory, result_queue: multiprocessing.Queue) -> None:
    result = factory()
    result_queue.put(result)


# ---- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SUMO-MPC closed-loop simulation")
    parser.add_argument("--scenario", default="a2_motorway", help="Scenario folder name")
    parser.add_argument("--solver", nargs="+", default=["rm"],
                        choices=["rm", "vsl", "coordinated", "base"],
                        help="Solver type(s). Multiple values trigger --batch automatically.")
    parser.add_argument("--no-gui", action="store_true", help="Run SUMO without GUI")
    parser.add_argument("--batch", action="store_true", help="Run solver types in parallel")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--port", type=int, default=8813, help="Base TraCI port")
    parser.add_argument("--output", default="runs", help="Root output directory")
    parser.add_argument(
        "--no-async-controller",
        dest="use_async_controller",
        action="store_false",
        default=True,
        help="Use synchronous (blocking) MPCController instead of AsyncMPCController",
    )
    args = parser.parse_args()

    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME is not set. Please configure it before running.")

    if args.batch or len(args.solver) > 1:
        run_batch(
            scenario_name=args.scenario,
            control_types=args.solver,
            n_workers=args.workers,
            base_port=args.port,
            output_root=args.output,
            no_gui=args.no_gui,
            use_async_controller=args.use_async_controller,
        )
    else:
        run_single(
            scenario_name=args.scenario,
            control_type=args.solver[0],
            no_gui=args.no_gui,
            output_root=args.output,
            port=args.port,
            use_async_controller=args.use_async_controller,
        )


if __name__ == "__main__":
    # Required on Windows for multiprocessing
    multiprocessing.freeze_support()
    main()
