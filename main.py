

# macroscopic 

# parameters = BaseParameters()
# network = Network()
# demands = Demands()

# environment = Environment(parameters, network, demands)
# solver = Solver(environment) 

if __name__ == "__main__":
    from macroscopic.simulation.environment import Environment
    from macroscopic.simulation.parameters import BaseParameters
    from sym_metanet import Network

    # from macroscopic.simulation.demands import Demands
    from macroscopic.solvers.base import BaseMacroscopicSolver

    initial_conditions = {
        "rho_init": [10, 10, 10, 10, 10],
        "v_free_init": [102, 102, 102, 102, 102],
        "w_init": [0, 0],
        "r_init": [0],
    }

    solver = BaseMacroscopicSolver(
        environment=Environment(
            parameters=BaseParameters(**initial_conditions),
            network=Network(),
            demands=None,  # Replace with actual demands if available
        ),
        Np=10,
        Nc=5,
        M=1,
    )
    solver.setup()