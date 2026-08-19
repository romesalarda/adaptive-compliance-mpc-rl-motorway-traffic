from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import traci

if TYPE_CHECKING:
    from microscopic.controller.base import BaseSUMOController
    from microscopic.simulation.environment import MicroEnvironment
    from microscopic.statistics.base import MicroSimulationResults


@dataclass
class SimulationConfig:
    """All parameters needed to launch and run one SUMO simulation."""

    sumo_cfg: str          # absolute or relative path to .sumocfg
    duration_steps: int    # total simulation steps (sumo_duration_s / step_length)
    step_length: float = 0.1
    seed: int = 42
    use_gui: bool = False
    port: int = 8813       # TraCI TCP port — must be unique across concurrent runs
    # dirs to create relative to the sumocfg before SUMO starts (e.g. detector output dirs)
    sumo_output_subdirs: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)

    @property
    def sumo_cmd(self) -> list[str]:
        binary = "sumo-gui" if self.use_gui else "sumo"
        return [
            binary,
            "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--seed", str(self.seed),
            "--no-step-log", "true",
            *self.extra_args,
        ]

    def prepare_output_dirs(self) -> None:
        """Create output sub-directories next to the sumocfg before SUMO starts."""
        sumo_dir = Path(self.sumo_cfg).parent
        for sub in self.sumo_output_subdirs:
            (sumo_dir / sub).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_scenario(
        cls,
        scenario_path: str,
        no_gui: bool = False,
        port: int = 8813,
        **overrides,
    ) -> "SimulationConfig":
        """
        Build config from a scenario folder's scenario.json micro block.
        
        Args:
            scenario_path: Path to a scenario folder containing scenario.json.
            no_gui: If True, force SUMO to run in headless mode.
            port: TraCI TCP port for this simulation (must be unique across concurrent runs).
            **overrides: Any SimulationConfig fields to override from the micro block.
        Returns:
            SimulationConfig instance.
        """
        config_path = Path(scenario_path) / "scenario.json"
        with config_path.open() as f:
            config = json.load(f)
        micro = config["micro"]
        sumo_cfg = str(Path(scenario_path) / micro["sumo_config"])
        return cls(
            sumo_cfg=sumo_cfg,
            duration_steps=int(micro.get("duration_steps", round(9000 / micro["step_length"]))),
            step_length=float(micro["step_length"]),
            use_gui=not no_gui,
            port=port,
            sumo_output_subdirs=list(micro.get("output_subdirs", [])),
            **overrides,
        )


class SUMOBaseSimulation:
    '''
    Base class for running a SUMO simulation with a given environment and controller.
    '''
    def __init__(
        self,
        config: SimulationConfig,
        env: MicroEnvironment,
        controller: BaseSUMOController,
        stats: MicroSimulationResults,
    ):
        '''
        Base class for running a SUMO simulation with a given environment and controller.

        Args:
            config: SimulationConfig instance with SUMO parameters.
            env: MicroEnvironment instance representing the simulation environment.
            controller: BaseSUMOController instance to control the simulation.
            stats: MicroSimulationResults instance to collect statistics.
        
        '''
        self.config = config
        self.env = env
        self.controller = controller
        self.stats = stats
        self.simulation_time: float = 0.0
        self._step: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        '''
        Checks prerequisites, prepares output directories, starts SUMO, and sets up the environment and controller.
        '''
        self.check()
        self.config.prepare_output_dirs()
        traci.start(self.config.sumo_cmd, port=self.config.port)
        self.env.setup()
        self.env.check()

        self.controller.setup()

    def check(self) -> None:
        '''
        Raise errors if SUMO_HOME is not set or the sumo_cfg file does not exist.
        '''
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("SUMO_HOME environment variable is not set.")
        if not Path(self.config.sumo_cfg).exists():
            raise FileNotFoundError(f"SUMO config not found: {self.config.sumo_cfg}")

    def step(self) -> None:
        traci.simulationStep()
        self.env.update()
        self.env.step_metering()

        state = self.env.collect_state()
        self.controller.on_step(self._step, self.env, state)

        received = self.controller._controls_received_this_step
        self.stats.record_step(
            step=self._step,
            time=self.simulation_time,
            state=state,
            controls=self.controller._last_controls if received else None,
            solve_time=self.controller._last_solve_time if received else None,
        )

        self.simulation_time += self.config.step_length
        self._step += 1

    def manual_input_check(self) -> None:
        '''
        Pretty-print the environment's configuration and state for debugging.
        Wait for user input to continue or quit.
        '''
        macro_env = self.controller.macro_env
        macro_env.parameters.pprint(ignore=["_time"])
        self.controller.micro_env.pprint()

        link_class_name = getattr(macro_env, "link_class_name", None) or "Link (default)"
        solver_config = getattr(macro_env, "solver_config", {})
        print(f"\n  Link class : {link_class_name}")
        if solver_config:
            cfg_str = ", ".join(f"{k}={v}" for k, v in solver_config.items())
            print(f"  Solver cfg : {cfg_str}")

        ready = input("\nPress Enter to continue, or type 'q' to quit: ")
        if ready.lower() == "q":
            raise KeyboardInterrupt("User requested exit.")
        

    def run(self) -> None:
        '''
        Continuously step the simulation until the duration is reached or SUMO signals that all vehicles have left.
        '''
        for _ in range(self.config.duration_steps):
            # stop early only if SUMO signals all vehicles have left and cleared
            if self._step > 0 and traci.simulation.getMinExpectedNumber() == 0:
                break
            self.step()

    def post_run(self) -> None:
        '''
        Clean up the simulation, close SUMO, stop the controller, and finalize statistics.
        '''
        traci.close()
        self.controller.stop()
        self.stats.finalize(
            step_length=self.config.step_length,
            n_steps=self._step,
        )

    # ------------------------------------------------------------------
    # Batch / parallel execution
    # ------------------------------------------------------------------

    @classmethod
    def run_in_process(
        cls,
        factory: Callable[[], "SUMOBaseSimulation"],
        result_queue: Optional[multiprocessing.Queue] = None,
    ) -> multiprocessing.Process:
        """
        Spawn a simulation in an isolated process.

        `factory` must be a module-level callable (picklable on Windows spawn).
        Each process owns its TraCI connection on the port set in SimulationConfig.

        Args:
            factory: A callable that returns a SUMOBaseSimulation instance.
            result_queue: Optional multiprocessing.Queue to receive the simulation stats after completion.
        Returns:
            The multiprocessing.Process object for the spawned simulation.
        """
        p = multiprocessing.Process(
            target=_simulation_worker,
            args=(factory, result_queue),
        )
        p.start()
        return p


# Module-level worker required for multiprocessing.spawn on Windows
def _simulation_worker(
    factory: Callable[[], SUMOBaseSimulation],
    result_queue: Optional[multiprocessing.Queue],
) -> None:
    """
    Module-level worker function to run a SUMOBaseSimulation in a separate process.

    Args:
        factory: A callable that returns a SUMOBaseSimulation instance.
        result_queue: Optional multiprocessing.Queue to receive the simulation stats after completion.  

    Raises:
        Any exceptions raised during simulation setup or execution will propagate to the parent process.
    """
    sim = factory()
    sim.setup()
    sim.run()
    sim.post_run()
    if result_queue is not None:
        result_queue.put(sim.stats)
