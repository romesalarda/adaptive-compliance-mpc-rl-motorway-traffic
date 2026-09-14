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
import random
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from typing import TYPE_CHECKING

from microscopic.controller.rl_async_mpc import RLAsyncMPCController
from microscopic.statistics.rl_stats import save_rl_stats_hdf5, save_rl_stats_json, update_rl_runs_log
from microscopic.visualisation.learning import RLVisualiser
from microscopic.statistics.dump import save_error_report

from macroscopic.solvers import MpcRMSimulation, MpcVSLSimulation, MpcCoordinatedSimulation, DummyMPCSolver

if TYPE_CHECKING:
    from microscopic.statistics.base import MicroSimulationResults
    from macroscopic.solvers.base import BaseMPCSolver
    from scenarios import Scenario

# ---- run directory helpers -------------------------------------------------

def _resolve_run_dir(
    scenario_name: str,
    control_type: str,
    experiment: str = "default",
    output_root: str | None = None,
    heterogeneity: str | None = None,
    demand_descriptor: str | None = None,
    verbose_descriptor: bool = False,
) -> tuple[Path, str]:
    """
    Return (out_dir, run_id). Uses scenario-relative path unless output_root overrides.
    
    Args:
        scenario_name: str, name of the scenario folder
        control_type: str, solver type (rm, vsl, coordinated, base)
        experiment: str, sub-folder under micro-sim-runs/ for grouping runs
        output_root: str or None, if provided overrides the default run path
        heterogeneity: str or None, scenario heterogeneity descriptor (e.g. "low")
        demand_descriptor: str or None, scenario demand profile slug (e.g. "hourly-peak")
        verbose_descriptor: bool, if True append heterogeneity/demand slugs to the folder name

    Returns:
        tuple[Path, str]: (output directory, run ID)
    """
    if output_root is not None:
        run_id = uuid.uuid4().hex[:8]
        return Path(output_root) / scenario_name / f"{control_type}_{run_id}", run_id

    base = Path("scenarios") / scenario_name / "micro-sim-runs" / experiment / control_type
    counter = 1
    if base.exists():
        pattern = re.compile(rf"^{re.escape(control_type)}_(\d+)")
        for d in base.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    counter = max(counter, int(m.group(1)) + 1)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_id = f"{control_type}_{counter:03d}"
    folder_name = f"{run_id}_{ts}"
    if verbose_descriptor and heterogeneity and demand_descriptor:
        folder_name += f"_{heterogeneity}_het__{demand_descriptor}"
    return base / folder_name, run_id


def _copy_detectors(scenario_name: str, out_dir: Path) -> None:
    """
    Copy SUMO detector outputs from micronet/detectors/ into the run folder.
    
    Args:
        scenario_name: str, name of the scenario folder
        out_dir: Path, the output directory for the current run
    Returns:
        None
    
    """
    src = Path("scenarios") / scenario_name / "micronet" / "detectors"
    if not src.exists():
        return
    dst = out_dir / "detectors"
    dst.mkdir(parents=True, exist_ok=True)
    for xml_file in src.glob("*.xml"):
        shutil.copy2(xml_file, dst / xml_file.name)


# ---- solver registry -------------------------------------------------------

def _build_solver(macro_env, control_type: str):
    """
    Instantiate and setup the macro solver for the given control type.
    
    Args:
        macro_env: MacroEnvironment, the macroscopic environment
        control_type: str, one of "rm", "vsl", "coordinated", "base"
    Returns:
        BaseMPCSolver, the fully setup solver instance
    """

    sc = macro_env.solver_config
    common = dict(
        Np=sc.get("Np", 7),
        Nc=sc.get("Nc", 3),
        M=sc.get("M", 6),
        function_compactness=sc.get("function_compactness", 2),
        objective_weights=macro_env.objective_weights,
    )
    if control_type == "rm":
        solver = MpcRMSimulation(environment=macro_env, **common)
    elif control_type == "vsl":
        solver = MpcVSLSimulation(
            environment=macro_env, 
            **common,
            initial_conditions={
                macro_env.network.origins_by_name["O1"]: {"v_ctrl": float("inf")},
            },
        )
    elif control_type == "coordinated":
        solver = MpcCoordinatedSimulation(
            environment=macro_env,
            **common,
            initial_conditions={
                macro_env.network.origins_by_name["O1"]: {"v_ctrl": float("inf")},
            }
        )
    elif control_type == "base":
        solver = DummyMPCSolver(environment=macro_env, **common)

    else:
        raise ValueError(f"Unknown solver type: {control_type!r}")

    solver.setup()
    return solver


def _build_solver_with_alpha(scenario: "Scenario", control_type: str, alpha: float) -> "BaseMPCSolver":
    """
    Build one fully-setup solver with alpha_compliance overridden; used by the solver pool factory.
    
    Args:
        scenario: Scenario, the scenario object
        control_type: str, one of "rm", "vsl", "coordinated", "base"
        alpha: float, the alpha_compliance value to set for this solver

    Returns:
        BaseMPCSolver, the fully setup solver instance with alpha_compliance set
    
    """
    from macroscopic.solvers import MpcCoordinatedSimulation

    macro_env = scenario.build_environment_with_alpha(control_type, alpha)
    sc = macro_env.solver_config
    common = dict(
        Np=sc.get("Np", 7),
        Nc=sc.get("Nc", 3),
        M=sc.get("M", 6),
        function_compactness=sc.get("function_compactness", 2),
        objective_weights=macro_env.objective_weights,
    )
    if control_type == "coordinated":
        solver = MpcCoordinatedSimulation(
            environment=macro_env,
            **common,
            initial_conditions={
                macro_env.network.origins_by_name["O1"]: {"v_ctrl": float("inf")},
            },
        )
    else:
        raise ValueError(f"RL solver pool only supports 'coordinated'; got '{control_type}'.")
    solver.setup(show_logs=False)
    return solver


# ---- scenario pool helpers ---------------------------------------------------

def _validate_pool_rl_configs(scenario_names: list[str], rl_config_name: str) -> dict:
    """
    Ensure every scenario in the pool shares an identical 'state'/'actions' rl-config schema
    (required since they all train the same Q-table). Returns the first scenario's rl_config.
    """
    from scenarios import Scenario

    reference_name = scenario_names[0]
    reference_cfg = Scenario(reference_name).get_rl_config(rl_config_name)
    for name in scenario_names[1:]:
        cfg = Scenario(name).get_rl_config(rl_config_name)
        for key in ("state", "actions"):
            if cfg.get(key) != reference_cfg.get(key):
                raise ValueError(
                    f"Scenario pool mismatch: '{name}' has a different '{key}' rl-config block "
                    f"than '{reference_name}'. All pooled scenarios must share the same "
                    f"'state'/'actions' schema so they can train one Q-table."
                )
    return reference_cfg


def _build_scenario_schedule(scenario_names: list[str], n: int, seed: int | None) -> list[str]:
    """Stratified schedule: each scenario appears floor(n/k) or ceil(n/k) times, order shuffled."""
    k = len(scenario_names)
    base, remainder = divmod(n, k)
    rng = random.Random(seed)

    schedule: list[str] = []
    for name in scenario_names:
        schedule.extend([name] * base)

    extra_names = rng.sample(scenario_names, remainder)
    schedule.extend(extra_names)

    rng.shuffle(schedule)
    return schedule


def _save_pool_scenarios_descriptor(q_table_dir: Path, pool_name: str) -> None:
    """Write pool_scenarios.json listing each pooled scenario's heterogeneity/demand descriptor."""
    import json
    from scenarios import Scenario

    entries = []
    for name in pool_name.split("+"):
        sc = Scenario(name)
        entries.append({
            "scenario": name,
            "heterogeneity": sc.heterogeneity,
            "demand_descriptor": sc.demand_descriptor,
        })
    with open(q_table_dir / "pool_scenarios.json", "w", encoding="utf-8") as fh:
        json.dump({"pool_name": pool_name, "scenarios": entries}, fh, indent=2)


# ---- single-run wiring ------------------------------------------------------

def run_single(
    scenario_name: str,
    control_type: str,
    no_gui: bool = True,
    output_root: str | None = None,
    port: int = 8813,
    show_dashboard: bool = True,
    use_async_controller: bool = True,
    wait_for_input: bool = True,
    experiment: str = "default",
    redirect_detectors: bool = False,
    use_rl: bool = False,
    rl_config_name: str = "default",
    learning: bool = True,
    pool_name: str | None = None,
    verbose_descriptor: bool = False,

) -> "MicroSimulationResults":
    """
    Build and run one closed-loop SUMO-MPC simulation.
    Returns MicroSimulationResults after post-processing.
    """
    from scenarios import Scenario
    from microscopic.simulation.base import SimulationConfig, SUMOBaseSimulation
    from microscopic.controller.mpc import SynchronousMPCController
    from microscopic.controller.async_mpc import AsyncMPCController
    from microscopic.statistics.base import MicroSimulationResults
    from microscopic.statistics.dump import save_hdf5, parse_tripinfo, parse_collision, parse_queue, save_run_metadata, update_micro_index
    from microscopic.statistics.async_stats import save_async_stats_hdf5
    from microscopic.statistics.rl_stats import save_rl_descriptor
    from microscopic.visualisation.base import MicroVisualiser

    scenario = Scenario(scenario_name)
    micro_cfg = scenario.micro_config

    macro_env = scenario.build_environment(control_type)
    micro_env = scenario.build_micro_environment()

    if not use_rl:
        solver = _build_solver(macro_env, control_type)

    out_dir, run_id = _resolve_run_dir(
        scenario_name,
        control_type,
        experiment,
        output_root,
        heterogeneity=scenario.heterogeneity,
        demand_descriptor=scenario.demand_descriptor,
        verbose_descriptor=verbose_descriptor,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
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

        _solve_steps = macro_env.solver_config.get("solve_interval_steps", micro_cfg["solve_interval_steps"])

        if use_rl:

            rl_config = scenario.get_rl_config(rl_config_name)
            # Q-table lives at the experiment/control_type level (or pool level when pool_name is set)
            if pool_name is not None:
                q_table_dir = Path("scenarios") / "_pools" / pool_name / experiment / control_type
            else:
                q_table_dir = Path("scenarios") / scenario_name / "micro-sim-runs" / experiment / control_type
            q_table_dir.mkdir(parents=True, exist_ok=True)
            q_table_path = q_table_dir / "q_table.npy"

            # Save descriptor once per experiment+solver folder so the config is always recoverable
            save_rl_descriptor(rl_config, q_table_dir / "rl_descriptor.json")
            if pool_name is not None:
                _save_pool_scenarios_descriptor(q_table_dir, pool_name)

            controller = RLAsyncMPCController.from_config(
                rl_config=rl_config,
                solver_factory=lambda alpha, _sc=scenario, _ct=control_type: _build_solver_with_alpha(_sc, _ct, alpha),
                macro_env=macro_env,
                micro_env=micro_env,
                solve_interval_steps=_solve_steps,
                step_length=sim_config.step_length,
                learning=learning,
                q_table_path=q_table_path,
            )
            controller.setup_rl()

        elif use_async_controller:
            controller = AsyncMPCController(
                macro_env=macro_env,
                micro_env=micro_env,
                solver=solver,
                solve_interval_steps=_solve_steps,
                step_length=sim_config.step_length,
            )
        else:
            controller = SynchronousMPCController(
                macro_env=macro_env,
                micro_env=micro_env,
                solver=solver,
                solve_interval_steps=_solve_steps,
            )

        sim = SUMOBaseSimulation(sim_config, micro_env, controller, stats)

        t0 = time.perf_counter()
        sim.setup()

        if wait_for_input:
            if use_rl:
                controller.print_pre_run_status()
            sim.manual_input_check()

        sim.run()
        sim.post_run()

        if redirect_detectors:
            _copy_detectors(scenario_name, out_dir)

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

        # TODO: fix negative predictions

        # parse SUMO output files
        sumo_data = {
            "tripinfo": parse_tripinfo(out_dir / "tripinfo.xml"),
            "collision": parse_collision(out_dir / "collision.xml"),
            "queue": parse_queue(out_dir / "queue.xml"),
        }
        stats.overview.total_collisions = sumo_data["collision"]["num_collisions"]

        save_hdf5(stats, out_dir / "simulation.h5")

        _active_solver = None if use_rl else solver
        save_run_metadata(
            out_dir=out_dir,
            stats=stats,
            macro_env=macro_env,
            micro_cfg=micro_cfg,
            experiment=experiment,
            solver=_active_solver,
            heterogeneity=scenario.heterogeneity,
            demand_descriptor=scenario.demand_descriptor,
        )

        if output_root is None:
            experiment_dir = Path("scenarios") / scenario_name / "micro-sim-runs" / experiment
            update_micro_index(
                experiment_dir=experiment_dir,
                entry={
                    "run_id": run_id,
                    "run_type": control_type,
                    "timestamp": stats.timestamp,
                    "solver_class": type(_active_solver).__name__ if _active_solver else "RLAsyncMPC",
                    "scenario_id": scenario_name,
                    "experiment": experiment,
                    "path": str(out_dir.resolve()),
                    "heterogeneity": scenario.heterogeneity,
                    "demand_descriptor": scenario.demand_descriptor,
                    "pool_name": pool_name,
                },
            )

        if use_rl:
            rl_stats = controller.get_rl_stats()
            save_rl_stats_hdf5(rl_stats, out_dir / "rl_stats.h5")
            save_rl_stats_json(rl_stats, out_dir / "rl_stats.json")
            print(f"[rl] {len(rl_stats.records)} RL steps | Q-table → {q_table_path}")
            update_rl_runs_log(
                q_table_dir,
                entry={
                    "run_id": run_id,
                    "scenario_id": scenario_name,
                    "control_type": control_type,
                    "experiment": experiment,
                    "pool_name": pool_name,
                    "path": str(out_dir.resolve()),
                    "timestamp": stats.timestamp,
                    "learning": learning,
                    **{
                        k: rl_stats.episode_summary.get(k)
                        for k in ("final_epsilon", "q_table_size", "total_agent_steps", "total_episodes", "n_steps", "mean_reward")
                    },
                },
            )
            if show_dashboard and rl_stats.records:
                try:
                    rl_dash = RLVisualiser(rl_stats).plot_dashboard(
                        filename="rl_dashboard.html", output_dir=str(out_dir)
                    )
                    print(f"RL dashboard saved to {rl_dash}")
                except Exception as e:
                    print(f"RL dashboard skipped: {e}")

        if (use_async_controller or use_rl) and isinstance(controller, AsyncMPCController):
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
                try:
                    pred_dash_path = vis.plot_prediction_analysis(
                        async_stats=controller.async_stats,
                        filename="prediction_analysis.html",
                        output_dir=str(out_dir),
                    )
                    print(f"Prediction analysis saved to {pred_dash_path}")
                except ValueError as e:
                    print(f"Prediction analysis skipped: {e}")

        print(
            f"[{scenario_name}/{control_type}] "
            f"TTS={stats.overview.total_time_spent:.1f} veh·h | "
            f"Collisions={stats.overview.total_collisions} | "
            f"Wall time={wall_time:.1f}s"
        )
        return stats
    except (Exception, KeyboardInterrupt) as e:
        save_error_report(
            out_dir=out_dir,
            exc=e,
            scenario_id=scenario_name,
            control_type=control_type,
            run_id=run_id,
            experiment=experiment,
        )
        raise


# ---- batch run --------------------------------------------------------------

def run_batch(
    scenario_name: str,
    control_types: list[str],
    n_workers: int = 2,
    base_port: int = 8813,
    output_root: str | None = None,
    no_gui: bool = True,
    use_async_controller: bool = True,
    experiment: str = "default",
    redirect_detectors: bool = False,
    verbose_descriptor: bool = False,
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
            experiment=experiment,
            redirect_detectors=redirect_detectors,
            verbose_descriptor=verbose_descriptor,
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

    def __init__(self, scenario_name, control_type, port, output_root, no_gui, use_async_controller=True, experiment="default", redirect_detectors=False, verbose_descriptor=False):
        self.scenario_name = scenario_name
        self.control_type = control_type
        self.port = port
        self.output_root = output_root
        self.no_gui = no_gui
        self.use_async_controller = use_async_controller
        self.experiment = experiment
        self.redirect_detectors = redirect_detectors
        self.verbose_descriptor = verbose_descriptor

    def __call__(self):
        return run_single(
            scenario_name=self.scenario_name,
            control_type=self.control_type,
            no_gui=self.no_gui,
            output_root=self.output_root,
            port=self.port,
            show_dashboard=True,
            use_async_controller=self.use_async_controller,
            experiment=self.experiment,
            redirect_detectors=self.redirect_detectors,
            verbose_descriptor=self.verbose_descriptor,
        )


def _batch_worker(factory: _SimFactory, result_queue: multiprocessing.Queue) -> None:
    result = factory()
    result_queue.put(result)


# ---- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SUMO-MPC closed-loop simulation")
    parser.add_argument("--scenario", default="a2_motorway", help="Scenario folder name")
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated pool of scenario names for RL training (e.g. 'a3_motorway_a,a3_motorway_b'). "
             "Requires --rl; each --repeat run draws from the pool via a stratified shuffle.",
    )
    parser.add_argument(
        "--pool-name",
        dest="pool_name",
        default=None,
        help="Override the auto-generated pool folder name (default: sorted scenario names joined by '+')",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for the --scenarios stratified shuffle (default: random)")
    parser.add_argument("--solver", nargs="+", default=["rm"],
                        choices=["rm", "vsl", "coordinated", "base"],
                        help="Solver type(s). Multiple values trigger --batch automatically.")
    parser.add_argument("--no-gui", action="store_true", help="Run SUMO without GUI")
    parser.add_argument("--batch", action="store_true", help="Run solver types in parallel")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--port", type=int, default=8813, help="Base TraCI port")
    parser.add_argument("--output", default=None, help="Override output root (uses legacy runs/<scenario>/<solver>_<hex>/ layout)")
    parser.add_argument("--experiment", default="default", help="Named sub-folder under micro-sim-runs/ (default: 'default')")
    parser.add_argument(
        "--redirect-detectors",
        dest="redirect_detectors",
        action="store_true",
        default=False,
        help="Copy SUMO detector outputs from micronet/detectors/ into the run folder after simulation",
    )
    parser.add_argument(
        "--verbose-descriptor",
        dest="verbose_descriptor",
        action="store_true",
        default=False,
        help="Append scenario heterogeneity/demand descriptor slugs to the run folder name",
    )
    parser.add_argument(
        "--no-async-controller",
        dest="use_async_controller",
        action="store_false",
        default=True,
        help="Use synchronous (blocking) MPCController instead of AsyncMPCController",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait_for_input",
        action="store_false",
        default=True,
        help="Do not wait for user input before starting simulation",
    )
    parser.add_argument(
        "--rl",
        dest="use_rl",
        action="store_true",
        default=False,
        help="Use the RL Q-learning controller (requires --solver coordinated)",
    )
    parser.add_argument(
        "--rl-config-name",
        dest="rl_config_name",
        default="default",
        help="Named RL config from scenario.json 'rl.configs' block (default: 'default')",
    )
    parser.add_argument(
        "--no-learning",
        dest="learning",
        action="store_false",
        default=True,
        help="Run RL agent in evaluation mode (greedy, no Q-table updates)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run the same single simulation N times in sequence (RL Q-table persists between runs; all other state resets each run)",
    )
    args = parser.parse_args()

    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME is not set. Please configure it before running.")

    scenario_pool: list[str] | None = None
    if args.scenarios is not None:
        scenario_pool = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if not args.use_rl:
            sys.exit("--scenarios requires --rl (scenario pooling only applies to RL training).")
        if args.batch or len(args.solver) > 1:
            sys.exit("--scenarios cannot be combined with --batch or multiple --solver values.")
        if not scenario_pool:
            sys.exit("--scenarios must list at least one scenario name.")

    if args.batch or len(args.solver) > 1:
        run_batch(
            scenario_name=args.scenario,
            control_types=args.solver,
            n_workers=args.workers,
            base_port=args.port,
            output_root=args.output,
            no_gui=args.no_gui,
            use_async_controller=args.use_async_controller,
            experiment=args.experiment,
            redirect_detectors=args.redirect_detectors,
            verbose_descriptor=args.verbose_descriptor,
        )
    else:
        repeat = args.repeat if not (args.batch or len(args.solver) > 1) else 1
        wait_for_input = args.wait_for_input if not (args.batch or len(args.solver) > 1 or repeat > 1) else False

        if scenario_pool is not None:
            from scenarios import pool_name_for

            _validate_pool_rl_configs(scenario_pool, args.rl_config_name)
            pool_name = args.pool_name or pool_name_for(scenario_pool)
            schedule = _build_scenario_schedule(scenario_pool, repeat, args.seed)
            print(f"[pool] '{pool_name}' schedule ({repeat} runs): {schedule}")
        else:
            pool_name = None
            schedule = [args.scenario] * repeat

        for i in range(repeat):
            if repeat > 1:
                print(f"\n[REPEAT] Run {i + 1}/{repeat}")
            run_single(
                scenario_name=schedule[i],
                control_type=args.solver[0],
                no_gui=args.no_gui,
                output_root=args.output,
                port=args.port,
                use_async_controller=args.use_async_controller,
                wait_for_input=wait_for_input,
                experiment=args.experiment,
                redirect_detectors=args.redirect_detectors,
                use_rl=args.use_rl,
                rl_config_name=args.rl_config_name,
                learning=args.learning,
                pool_name=pool_name,
                verbose_descriptor=args.verbose_descriptor,
            )


if __name__ == "__main__":
    # Required on Windows for multiprocessing
    multiprocessing.freeze_support()
    main()
