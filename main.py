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
    MeteredOnRamp,
    Network,
    Node,
    Origin,
    MainstreamOrigin,   
)

import casadi as cs
import numpy as np

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


def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
            (
                # Mainline demand
                np.interp(
                    time,
                    (0.0, 0.5, 1.0, 1.5, 2.5),
                    (2800, 3200, 4000, 4200, 4200)
                ),
                # On-ramp demand
                np.interp(
                    time,
                    (0.0, 0.5, 1.0, 1.5, 2.5),
                    (200, 300, 300, 300, 200)

                ),
            )
        )

if __name__ == "__main__":


    initial_conditions = {
        "rho_init": [10, 10, 10, 10, 10, 10],
        "v_free_init": [102, 102, 102, 102, 102, 102],
        "w_init": [0, 0],
        "r_init": [0],

     
    }

    test_type = input("Enter test type (vsl | rm | base): ").strip().lower()
    if test_type not in ["vsl", "rm", "base"]:
        raise ValueError("Invalid test type. Please choose from 'vsl', 'rm', or 'base'.")

    if test_type == "vsl":

        initial_conditions.update(
            {
                "alpha_compliance": 0.5,
                "A": 0.5,
                "E": 0.5,
                "V": 120,
                "v_ctrl": [120, 120, 120, 120],
                "segments_with_vsl": [0, 1, 2, 3],
                "minimum_speed_limit": 60,
                "maximum_speed_limit": 120,
            }
        )
    
        parameters = VslParameters(**initial_conditions)
        network = define_network(
            parameters, 
            CarlsonLinkWithVsl,
            a=0.5,
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
        )
        solver.setup()
        solver.simulate()

    elif test_type == "rm":

        parameters = BaseParameters(**initial_conditions)
        network = define_network(
            parameters, 
            Link,
            a=0.5,
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
        )
        solver.setup()
        solver.simulate()

    elif test_type == "base":
        

        parameters = BaseParameters(**initial_conditions)
        network = define_network(
            parameters, 
            Link,
            a=0.5,
            )
        
        solver = BaseModelSimulation(
            environment=Environment(
                parameters=parameters,
                network=network,
                demands=create_demands(parameters.time).T,
            ),
            function_compactness=1,
            initial_conditions={
                network.origins_by_name["O1"]: {"r": 1, "v_ctrl": 102}
            }
        )
        solver.setup()
        solver.simulate()

    print(asdict(solver.overview_results))
