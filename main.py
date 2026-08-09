import argparse
import numpy as np
import matplotlib.pyplot as plt

from macroscopic.solvers import MpcRMSimulation, MpcVSLSimulation, MpcCoordinatedSimulation
from macroscopic.simulation.model import BaseModelSimulation
from macroscopic.visualisation.base import MacroscopicVisualiser
from macroscopic.statistics import RunSet
from macroscopic.vsl import CarlsonLinkWithVsl, FrejoLinkWithVsl

from scenarios import Scenario

from macroscopic.logger import LOGGER


def _run(scenario: Scenario, control_type: str, visualiser: MacroscopicVisualiser, run_set: RunSet, **kwargs) -> None:
    env = scenario.build_environment(control_type, **kwargs)
    network = env.network

    extra_meta = {
        "linkKlass": kwargs.get("linkKlass", None).__name__ if kwargs.get("linkKlass", None) else None,
        "control_type": control_type,
    }

    if control_type == "vsl":
        solver = MpcVSLSimulation(
            environment=env,
            Np=7,
            Nc=3,
            M=6,
            function_compactness=2,
            initial_conditions={
                network.origins_by_name["O1"]: {"v_ctrl": np.inf},  # effectively unrestricted
            },
        )
        
        solver.setup()
        solver.simulate()
        fig = visualiser.plot_rm_vsl(
            results=solver.simulation_results,
            params=solver.environment.parameters,
            metrics=solver.overview_results,
            show=False,
        )
        run_set.new_run(solver, run_type=control_type, figures={"rm_vsl": fig}, demands_meta=scenario.demands_meta, extra_meta=extra_meta)

    elif control_type == "rm":
        solver = MpcRMSimulation(
            environment=env,
            Np=7,
            Nc=3,
            M=6,
            function_compactness=2,
            objective_weights={"temporal_smoothness": 0.4},
            initial_conditions={
                network.origins_by_name["O1"]: {"v_ctrl": np.inf, "r": 1},  # effectively unrestricted
            },
        )
        solver.setup()
        solver.simulate()
        fig = visualiser.plot_ramp_metering(
            results=solver.simulation_results,
            params=solver.environment.parameters,
            metrics=solver.overview_results,
            show=False,
        )
        run_set.new_run(solver, run_type=control_type, figures={"ramp_metering": fig}, demands_meta=scenario.demands_meta, extra_meta=extra_meta)

    elif control_type == "coordinated":
        solver = MpcCoordinatedSimulation(
            environment=env,
            Np=5,
            Nc=4,
            M=30,
            function_compactness=2,
            initial_conditions={
                network.origins_by_name["O1"]: {"v_ctrl": np.inf},  # effectively unrestricted
            },
            objective_weights=env.objective_weights,
        )
        solver.setup()
        solver.simulate()
        fig = visualiser.plot_coordinated_control(
            results=solver.simulation_results,
            params=solver.environment.parameters,
            metrics=solver.overview_results,
            show=False,
        )
        run_set.new_run(solver, run_type=control_type, figures={"coordinated": fig}, demands_meta=scenario.demands_meta, extra_meta=extra_meta)

    elif control_type == "base":
        solver = BaseModelSimulation(
            environment=env,
            function_compactness=1,
            initial_conditions={
                network.origins_by_name["O1"]: {"v_ctrl": np.inf, "r": 1},  # effectively unrestricted
            },
        )
        solver.setup()
        solver.parameters.pprint()
        solver.simulate()
        fig = visualiser.plot_dynamics(
            results=solver.simulation_results,
            params=solver.environment.parameters,
            metrics=solver.overview_results,
            show=False,
        )
        run_set.new_run(solver, run_type=control_type, figures={"dynamics": fig}, demands_meta=scenario.demands_meta, extra_meta=extra_meta)

    LOGGER.info("Displaying all figures.")
    plt.show()


if __name__ == "__main__":
    # py main.py

    # # Single run, no prompts
    # py main.py --scenario a1_motorway --control base

    # # Batch run
    # py main.py --scenario a1_motorway --batch

    # # Partially specified — prompts only for missing pieces
    # py main.py --scenario a1_motorway

    parser = argparse.ArgumentParser(description="Run macroscopic traffic simulations.")
    parser.add_argument("--scenario", metavar="NAME", help="Scenario folder name")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--control", choices=["vsl", "rm", "base", "coordinated"], metavar="TYPE", help="Single run: control type (vsl | rm | base | coordinated)")
    mode_group.add_argument("--batch", action="store_true", help="Batch run: execute all control types in sequence")
    args = parser.parse_args()

    available_scenarios = Scenario.list_scenarios()

    if args.scenario:
        scenario_name = args.scenario
    else:
        LOGGER.info(f"Available scenarios: {', '.join(available_scenarios)}")
        scenario_name = input("Enter scenario name: ").strip()
        if not scenario_name and available_scenarios:
            scenario_name = available_scenarios[0]
    if not scenario_name:
        raise ValueError("No scenario name provided and no available scenarios found.")

    scenario = Scenario(scenario_name)

    if args.batch:
        mode = "batch"
    elif args.control:
        mode = "single"
        control_type = args.control
    else:
        mode = input("Run mode (single | batch): ").strip().lower()
        if mode not in ("single", "batch"):
            raise ValueError("Mode must be 'single' or 'batch'.")
        if mode == "single":
            control_type = input("Enter control type (vsl | rm | base): ").strip().lower()
            if control_type not in ("vsl", "rm", "base", "coordinated"):
                raise ValueError("Invalid control type. Choose from 'vsl', 'rm', 'base', 'coordinated'.")

    visualiser = MacroscopicVisualiser()
    run_set = RunSet(
        name=scenario_name,
        base_dir=scenario.path / "macro-sim-runs",
    )

    if mode == "single":
        _run(scenario, control_type, visualiser, run_set, linkKlass=FrejoLinkWithVsl)
    else:
        for control_type in ("base", "rm", "vsl", "coordinated"):
            LOGGER.info(f"\n--- Running: {control_type} ---")
            _run(scenario, control_type, visualiser, run_set)

