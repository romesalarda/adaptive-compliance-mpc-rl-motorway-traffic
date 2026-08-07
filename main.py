from macroscopic.simulation.environment import Environment
from macroscopic.simulation.parameters import BaseParameters, VslParameters
from sym_metanet import Network

from dataclasses import asdict

# from macroscopic.simulation.demands import Demands
from macroscopic.solvers import MpcRMSimulation, MpcVSLSimulation
from macroscopic.simulation.model import BaseModelSimulation

from macroscopic.vsl import CarlsonLinkWithVsl
from sym_metanet import (
    Destination,
    Link,
    LinkWithVsl,
    MeteredOnRamp,
    Network,
    Node,
    Origin,
    MainstreamOrigin,   
)

import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

from macroscopic.visualisation.base import METANETVisualiser
from macroscopic.statistics import RunSet

def define_network(parameters: BaseParameters, L1_klass: Link, **kwargs) -> Network:
    params = parameters

    N1 = Node(name="N1")
    N2 = Node(name="N2")
    N3 = Node(name="N3")
    
    # O1 = MeteredOnRamp[cs.SX](params.C[0], name="O1")
    O2 = MeteredOnRamp[cs.SX](params.C[1], name="O2")
    O1 = MainstreamOrigin[cs.SX](name="O1")
    # O2 = MainstreamOrigin[cs.SX](name="O2")
    D1 = Destination[cs.SX](name="D1")

    L1 = L1_klass(
        nb_segments=4,
        lanes=params.lanes,
        length=params.L,
        maximum_density=params.rho_max,
        critical_density=params.rho_crit,
        free_flow_velocity=params.v_free,
        name="L1",
        **kwargs
    )
    
    L2 = Link[cs.SX](
        nb_segments=2, 
        lanes=params.lanes, 
        length=params.L, 
        maximum_density=params.rho_max, 
        critical_density=params.rho_crit, 
        free_flow_velocity=params.v_free, 
        a=params.a,
        name="L2")
                
    return (
        Network(name="A1")
        .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        .add_origin(O2, N2)
    )


# Compact representation of demands for run dumps — keypoints used by create_demands()
_DEMANDS_META = {
    "type": "interp",
    "sources": ["mainline", "on_ramp"],
    "xp": [0.0, 0.5, 1.0, 1.5, 2.5],
    "fp": [
        [2800, 3000, 3500, 3000, 2800],
        [200, 600, 800, 600, 200],
    ],
}

def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
            (
                # Mainline demand
                np.interp(
                    time,
                    # (0.0, 0.5, 1.0, 1.5, 2.5),
                    _DEMANDS_META["xp"],
                    # (2800, 3000, 3500, 3000, 2800)
                    _DEMANDS_META["fp"][0]
                ),
                # On-ramp demand
                np.interp(
                    time,
                    _DEMANDS_META["xp"],
                    _DEMANDS_META["fp"][1]

                ),
            )
        )

if __name__ == "__main__":


    initial_conditions = {
        "rho_init": [10, 10, 10, 10, 10, 10],
        "v_free_init": [112, 112, 112, 112, 112, 112],
        "w_init": [0, 0],
        "r_init": [1],
    }

    test_type = input("Enter test type (vsl | rm | base): ").strip().lower()
    if test_type not in ["vsl", "rm", "base"]:
        raise ValueError("Invalid test type. Please choose from 'vsl', 'rm', or 'base'.")
    
    visualiser = METANETVisualiser()
    run_set = RunSet("mpc_tests")

    if test_type == "vsl":

        initial_conditions.update(
            {
                "alpha_compliance": 0.5,
                "A": 0.8,
                "E": 3.0,
                "V": 120,
                "v_ctrl": [120, 120, 120, 120],
                "segments_with_vsl": [0, 1, 2, 3],
                "minimum_speed_limit": 30,
                "maximum_speed_limit": 120,
                "a": 1.867,
            }
        )
    
        parameters = VslParameters(**initial_conditions)
        network = define_network(
            parameters, 
            CarlsonLinkWithVsl,
            a=parameters.a,
            segments_with_vsl=initial_conditions["segments_with_vsl"],
            alpha=initial_conditions["alpha_compliance"],
            V=initial_conditions["V"],
            A=initial_conditions["A"],
            E=initial_conditions["E"],
            )
        solver = MpcVSLSimulation(
            environment=Environment(
                parameters=parameters,
                network=network,
                demands=create_demands(parameters.time).T,
            ),
            Np=7,
            Nc=3,
            M=6,
            function_compactness=2,
            initial_conditions={
                network.origins_by_name["O1"]: {"v_ctrl": np.inf},  # effectively unrestricted
            }
        )
        solver.setup()
        solver.simulate()

        fig = visualiser.plot_rm_vsl(
            results=solver.simulation_results,
            params=solver.environment.parameters,
            metrics=solver.overview_results,
            show=False
        )
        run_set.new_run(solver, run_type=test_type, figures={"rm_vsl": fig}, demands_meta=_DEMANDS_META)
        plt.show()

    elif test_type == "rm":

        parameters = BaseParameters(**initial_conditions)
        parameters.a = 1.867
        network = define_network(
            parameters, 
            Link,
            a=parameters.a,
            )
        solver = MpcRMSimulation(
            environment=Environment(
                parameters=parameters,
                network=network,
                demands=create_demands(parameters.time).T,
            ),
            Np=7,
            Nc=3,
            M=6,
            function_compactness=2,
            objective_weights={
                "temporal_smoothness": 0.4     
            },
            initial_conditions={
                network.origins_by_name["O1"]: {"v_ctrl": np.inf, "r": 1},  # effectively unrestricted

            }
        )
        solver.setup()
        solver.simulate()

        fig = visualiser.plot_ramp_metering(
            results=solver.simulation_results,
            params=solver.environment.parameters,
            metrics=solver.overview_results,
            show=False
        )
        run_set.new_run(solver, run_type=test_type, figures={"ramp_metering": fig}, demands_meta=_DEMANDS_META)
        plt.show()

    elif test_type == "base":
        
        parameters = BaseParameters(**initial_conditions)
        parameters.a = 1.867
        network = define_network(
            parameters, 
            Link,
            a=parameters.a,            
            )
        
        solver = BaseModelSimulation(
            environment=Environment(
                parameters=parameters,
                network=network,
                demands=create_demands(parameters.time).T,
            ),
            function_compactness=1,
            initial_conditions={
                network.origins_by_name["O1"]: {"v_ctrl": np.inf, "r": 1},  # effectively unrestricted
            }
        )
        solver.setup()
        solver.parameters.pprint()
        solver.simulate()

        fig = visualiser.plot_dynamics(
            results=solver.simulation_results,
            params=solver.environment.parameters,
            metrics=solver.overview_results,
            show=False
        )
        run_set.new_run(solver, run_type=test_type, figures={"dynamics": fig}, demands_meta=_DEMANDS_META)
        plt.show()



    