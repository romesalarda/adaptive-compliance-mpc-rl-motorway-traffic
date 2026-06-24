import time
import sym_metanet as smn

from typing import TypeVar
from csnlp.wrappers import Mpc
from csnlp import Nlp

import casadi as cs
import numpy as np

SymType = TypeVar("SymType", cs.SX, cs.MX)


from macroscopic.simulation.environment import Environment
from macroscopic.simulation.statistics import (
    MacroSimulationResults, 
    MacroOverviewSimulationResults, 
    MacroRunSimulationResults
)
from macroscopic.logger import LOGGER
class BaseMPCSolver:
    """
    Base class for macroscopic solvers. Sets up the environment, initializes the MPC solver, 
    and provides methods for solving the macroscopic traffic flow problem.
    """

    DEFAULT_SOLVER_OPTIONS = {
        "ipopt": {  
            "expand" : True,
            "print_time" : False,
            "ipopt": {
                "print_level" : 0,
                "max_iter" : 1000,
                "tol" : 1e-6,
                "acceptable_tol" : 1e-5,
                "acceptable_obj_change_tol" : 1e-4,
            }
        }
    }

    DEFAULT_SOLVER = "ipopt"

    def __init__(self, 
            environment: Environment,
            Np=None,
            Nc=None,
            M=None,
            **kwargs    
        ):

        '''
        Args:
            environment (Environment): The simulation environment containing parameters, network, and demands.
            Np (int): Prediction horizon length for the solver.
            Nc (int): Control horizon length for the solver.
            M (int): Input spacing for the solver.
            kwargs: Additional keyword arguments for solver configuration.

        Raises:
            ValueError: If Np, Nc, or M are not positive integers.

        '''
        self.environment = environment
        self._Np = Np
        self._Nc = Nc
        self._M = M
        self._MPC = None  # Placeholder for the MPC object, to be defined in subclasses

        self._rho: cs.SX = None  # Placeholder for the density state, to be defined in subclasses
        self._v: cs.SX = None  # Placeholder for the speed state, to be defined in subclasses
        self._w: cs.SX = None  # Placeholder for the congestion queue state, to be defined in subclasses
        self._r: cs.SX = None  # Placeholder for the ramp metering rate state, to be defined in subclasses
        self._r_last: cs.SX = None  # Placeholder for the last ramp metering rate state, to be defined in subclasses

        self.simulation_results: MacroSimulationResults = MacroSimulationResults()  # Placeholder for simulation results, to be defined in subclasses
        self.overview_results: MacroOverviewSimulationResults = MacroOverviewSimulationResults()  # Placeholder for overview results, to be defined in subclasses
        self.run_results: MacroRunSimulationResults = MacroRunSimulationResults()  # Placeholder for run results, to be defined in subclasses

        self.number_of_segments = sum(link.N for _, _, link in self.environment.network.links)
        self.number_of_origins = len(self.environment.network.origins)

        self.kwargs = kwargs  # Store any additional keyword arguments for future use
        self.queue_upper_bound = kwargs.get("queue_upper_bound", None)  # Upper bound for the congestion queue, can be adjusted as needed 
        self.function_compactness = kwargs.get("function_compactness", 1)  # Compactness level for the function representation, can be adjusted as needed
    
    @property
    def Np(self):
        '''
        Prediction horizon length for the solver.
        '''
        return self._Np 
    
    @Np.setter
    def Np(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Np must be a positive integer.")
        self._Np = value
    
    @property
    def Nc(self):
        '''
        Control horizon length for the solver.
        '''
        return self._Nc
    
    @Nc.setter
    def Nc(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Nc must be a positive integer.")
        self._Nc = value

    @property
    def M(self):
        '''
        Input spacing for the solver.
        '''
        return self._M

    @M.setter
    def M(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("M must be a positive integer.")
        self._M = value

    @property
    def MPC(self):
        '''
        Model Predictive Control (MPC) object for the solver.
        '''
        return self._MPC
    
    @MPC.setter
    def MPC(self, value):
        if not isinstance(value, (Mpc, Nlp, type(None))):
            raise ValueError("MPC must be an instance of Mpc, Nlp, or None.")
        self._MPC = value

    @property
    def rho(self):  
        
        '''
        Density state for the solver.
        '''
        return self._rho
    
    @rho.setter
    def rho(self, value):
        if not isinstance(value, (cs.SX, type(None))):
            raise ValueError("rho must be a CasADi SX variable or None.")
        self._rho = value
    
    @property
    def v(self):
        '''
        Speed state for the solver.
        '''
        return self._v
    
    @v.setter
    def v(self, value):
        if not isinstance(value, (cs.SX, type(None))):
            raise ValueError("v must be a CasADi SX variable or None.")
        self._v = value
    
    @property
    def w(self):
        '''
        Congestion queue state for the solver.
        '''
        return self._w
    
    @w.setter
    def w(self, value):
        if not isinstance(value, (cs.SX, type(None))):
            raise ValueError("w must be a CasADi SX variable or None.")
        self._w = value

    @property
    def r(self):
        '''
        Ramp metering rate state for the solver.
        '''
        return self._r
    
    @r.setter
    def r(self, value):
        if not isinstance(value, (cs.SX, type(None))):
            raise ValueError("r must be a CasADi SX variable or None.")
        self._r = value

    @property
    def r_last(self) -> SymType:
        '''
        Last ramp metering rate state for the solver.
        '''
        return self._r_last
    
    @r_last.setter
    def r_last(self, value: SymType):
        if not isinstance(value, (cs.SX, type(None))):
            raise ValueError("r_last must be a CasADi SX variable or None.")
        self._r_last = value

    #### mpc methods

    def get_function(self, more_out: bool = True, compact: int = None):
        '''
        Returns a function that can be used to evaluate the macroscopic traffic flow model.

        Args:
            more_out (bool, optional): Whether to include additional outputs in the function. Defaults to True.
            compact (int, optional): Whether to return a compact version of the function. Defaults to None.

        Returns:
            function: A function that is used to represent the dynamics of the macroscopic traffic flow model.

        '''
        if self.function_compactness is not None and compact is None:
            compact = self.function_compactness

        self.environment.check()  

        return smn.engine.to_function(
            net=self.environment.network,
            more_out=more_out,
            compact=compact,
            T=self.environment.parameters.T,
        )


    def pre_init_mpc(self):
        '''
        override this method to perform any necessary pre-initialization steps for the MPC solver.
        '''
        self.environment.ready(initial_conditions=self.kwargs.get("initial_conditions", {}))

    def init_mpc(self):
        '''
        Initialize the Model Predictive Control (MPC) solver with the specified parameters.
        This method sets up the MPC solver using the provided prediction horizon (Np), control horizon (Nc), and input spacing (M). It creates an instance of the MPC solver 
        and configures it for the macroscopic traffic flow problem.  

        Returns:
            Mpc: An instance of the Model Predictive Control (MPC) solver.      
        '''
        self.environment.check()  # Ensure the environment is ready before initializing the MPC solver
        
        self.MPC = Mpc[cs.SX](
            nlp=Nlp[cs.SX](sym_type="SX"),
            prediction_horizon=self.Np * self.M,
            control_horizon=self.Nc * self.M,
            input_spacing=self.M,
        )
        return self.MPC

    def init_base_states(self):
        '''
        Initialize the base states for the MPC solver.
        '''
        n_seg = self.number_of_segments
        kwargs = self.kwargs

        self.rho, _ = self.MPC.state("rho", self.number_of_segments, lb=0)
        self.v, _ = self.MPC.state("v", self.number_of_segments, lb=0)

        if self.queue_upper_bound is not None:
            assert len(self.queue_upper_bound) == self.number_of_origins, "Queue upper bound must match the number of origins."
        
        self.w, _ = self.MPC.state("w", self.number_of_origins, lb=0, ub=[[np.inf], [100]])

    def init_control_states(self):
        '''
        Initialize the control states for the MPC solver. Base class implementation initializes the ramp metering rate control state. Subclasses can override this method to add additional control states as needed.
        '''
        self.r, _ = self.MPC.action("r", lb=0, ub=1)
        self.r_last = self.MPC.parameter("r_last", (self.r.size1(), 1))


    def init_constraints(self):
        '''
        Initialize the constraints for the MPC solver. This method sets up the constraints that govern the macroscopic traffic flow model, ensuring that the optimization problem adheres to the physical and operational limits of the system.
        '''
        pass
    
    def init_objective(self):
        '''
        Initialize the objective function for the MPC solver. This method sets up the objective function that will be minimized during the optimization process. The objective function is defined based on the traffic density, congestion queue, and ramp metering rates, with optional weighting for the ramp metering rates.
        '''
        self.MPC.minimize(
            self.environment.parameters.T * cs.sum2(cs.sum1(self.rho * self.environment.parameters.L * self.environment.parameters.lanes) + cs.sum1(self.w))
            + self.kwargs.get("ramp_weight", 0.4) * cs.sumsqr(cs.diff(cs.horzcat(self.r_last, self.r)))
        )

    def init_disturbances(self):
        '''
        Initialize the disturbances for the MPC solver. This method sets up the disturbances that affect the macroscopic traffic flow model, such as external inputs or environmental factors. The disturbances are defined based on the number of origins in the network.
        '''
        self.MPC.disturbance("d", self.number_of_origins)
    
    def init_non_linear_dynamics(self):
        '''
        Initialize the non-linear dynamics for the MPC solver. This method sets up the non-linear dynamics
        that govern the macroscopic traffic flow model. It retrieves the function representing the dynamics and configures the MPC solver to use this function for its non-linear dynamics.
        '''
        F = self.get_function()
        
        self.MPC.set_nonlinear_dynamics(lambda x, u, d: F(x, u, d)[0])
    
    def post_init_mpc(self):
        pass

    def init_solver(self, solver: str = DEFAULT_SOLVER, opts: dict = None):
        '''
        Initialize the solver for the MPC problem. This method sets up the solver that will be used to solve the optimization problem defined by the MPC formulation. The solver is configured with appropriate settings and options for efficient computation.
        '''
        if opts is None:
            opts = self.DEFAULT_SOLVER_OPTIONS.get(solver, {})

        self.MPC.init_solver(solver=solver, opts=opts)


    def setup(self, suppress=False):
        """
        Setup the macroscopic solver by initializing the MPC solver and other necessary components.
        """
        LOGGER.info("Setting up the macroscopic solver...")
        try:
            LOGGER.info("Starting the setup process for the macroscopic solver...")
            self.pre_init_mpc()
            LOGGER.info("\n1. Pre-initialization of the macroscopic solver completed successfully. ✓")
            LOGGER.info("Proceeding with the initialization of the MPC solver and related components...")
            self.init_mpc()
            LOGGER.info("MPC solver initialization completed successfully. ✓")
            LOGGER.info("\n2. Proceeding with the initialization of base states, control states, constraints, disturbances, objective function, non-linear dynamics, and solver...")
            self.init_base_states()
            LOGGER.info("Base states initialization completed successfully. ✓")
            LOGGER.info("\n3. Proceeding with the initialization of control states...")
            self.init_control_states()
            LOGGER.info("Control states initialization completed successfully. ✓")
            LOGGER.info("\n4. Proceeding with the initialization of constraints...")
            self.init_constraints()
            LOGGER.info("Constraints initialization completed successfully. ✓")
            LOGGER.info("\n5. Proceeding with the initialization of disturbances...")
            self.init_disturbances()
            LOGGER.info("Disturbances initialization completed successfully. ✓")
            LOGGER.info("\n6. Proceeding with the initialization of the objective function...")
            self.init_objective()
            LOGGER.info("Objective function initialization completed successfully. ✓")
            LOGGER.info("\n7. Proceeding with the initialization of non-linear dynamics...")
            self.init_non_linear_dynamics()
            LOGGER.info("Non-linear dynamics initialization completed successfully. ✓")
            LOGGER.info("\n8. Proceeding with the initialization of the solver...")
            self.init_solver()
            LOGGER.info("Solver initialization completed successfully. ✓")
            LOGGER.info("All components of the macroscopic solver have been initialized successfully. ✓")
            self.post_init_mpc()
            LOGGER.info("Post-initialization of the macroscopic solver completed successfully. ✓")
        except Exception as e:
            LOGGER.error(f"Solver Setup Failed: An error occurred during the setup of the macroscopic solver: {e}")
            if not suppress:
                raise e
            return False
        LOGGER.info("Macroscopic solver setup complete.")
        return True
    
    def check(self):
        """
        Check if the macroscopic solver is properly set up and ready for simulation.
        Raises an error if the solver is not ready.
        """
        if self.MPC is None:
            raise RuntimeError("MPC solver is not initialized. Call the 'setup' method first.")
        
    def pre_simulate(self):
        self.check()  # Ensure the solver is properly set up before starting the simulation
        LOGGER.info("Solver setup verified. Proceeding with pre-simulation steps...")
    
    def simulate(self, suppress=True):
        '''
        Simulate the traffic flow dynamics over time using the MPC solver. This method orchestrates the entire simulation process, including pre-simulation steps, the main simulation loop, and post-simulation steps. It handles exceptions and logs relevant information about the simulation progress.
        '''
        try:
                
            LOGGER.info("\nStarting the simulation process...")
            self.pre_simulate()
            LOGGER.info("Pre-simulation steps completed successfully. ✓")
            LOGGER.info("Proceeding with the simulation of traffic flow dynamics over time...")
            self._simulate()
            LOGGER.info("Simulation of traffic flow dynamics completed successfully. ✓")
            LOGGER.info("Proceeding with post-simulation steps...")
            self.post_simulate()
            LOGGER.info("Post-simulation steps completed successfully. ✓")

        except KeyboardInterrupt:
            LOGGER.warning("Simulation interrupted by user.")

        except Exception as e:
            LOGGER.error(f"Simulation Failed: An error occurred during the simulation process: {e}")
            if not suppress:
                raise e
            
    def _simulate(self):    
        '''
        Simulate the traffic flow dynamics over time using the MPC solver. This method is intended to be overridden by subclasses to implement specific simulation logic. It should contain the main simulation loop that iteratively solves the MPC problem, updates the state variables, and stores the simulation results.
        '''

        raise NotImplementedError("The '_simulate' method must be implemented in subclasses of BaseMPCSolver.")
    
    def store_step(self, rho, v, w, q, q_on, r_last):
        '''
        Store the simulation results for a single time step. This method appends the current state variables and outputs to the simulation results object, allowing for later analysis and visualization of the traffic flow dynamics.
        '''
        self.simulation_results.rho.append(rho)
        self.simulation_results.v.append(v)
        self.simulation_results.w.append(w)
        self.simulation_results.q.append(q)
        self.simulation_results.q_on.append(q_on)
        self.simulation_results.r.append(r_last)

    def calculate_demands_over_horizon(self, k):
        '''
        Calculate the predicted demand over the prediction horizon. This method extracts the demand values for the current time step and pads them if necessary to match the prediction horizon.
        '''
        d_hat = self.environment.demands[k : k + self.Np * self.M, :]
        if d_hat.shape[0] < self.Np * self.M:
            d_hat = np.pad(d_hat, ((0, self.Np * self.M - d_hat.shape[0]), (0, 0)), "edge")
        return d_hat
    
    def get_next_states(self, rho, v, w, r_last, k, F):
        '''
        Calculate the next states of the traffic flow variables given the current states, ramp metering rates, and demands.
        Args:
            rho (cs.SX): Current traffic density state.
            v (cs.SX): Current traffic speed state.
            w (cs.SX): Current congestion queue state.
            r_last (cs.SX): Last ramp metering rate state.
            k (int): Current time step index.
            F (function): Function representing the dynamics of the macroscopic traffic flow model.
        Returns:
            tuple: Next states of traffic density, speed, congestion queue, and flow variables.
        '''

        x_next, q_all = F(cs.vertcat(rho, v, w), r_last, self.environment.demands[k, :])
        rho, v, w = cs.vertsplit(x_next, (0, self.number_of_segments, 2 * self.number_of_segments, 2 * self.number_of_segments + self.number_of_origins))
        q, q_on = cs.vertsplit(q_all, (0, self.number_of_segments, self.number_of_segments + self.number_of_origins))
        return rho, v, w, q, q_on
    
    def post_simulate(self):
        '''
        Post-process the simulation results after the simulation is complete. This method can be used to compute additional statistics, generate plots, or perform any other necessary post-simulation analysis.
        '''
        RHO, V, W, Q, Q_o, R = (np.squeeze(o) for o in (
            self.simulation_results.rho, 
            self.simulation_results.v, 
            self.simulation_results.w,
            self.simulation_results.q, 
            self.simulation_results.q_on, 
            self.simulation_results.r
            )
        ) 
        parameters = self.environment.parameters
        T = parameters.T
        L = parameters.L
        lanes = parameters.lanes
        rho_crit = parameters.rho_crit
        v_free = parameters.v_free

        self.overview_results.total_time_spent = T * sum((rho * L * lanes).sum() + w.sum() for rho, w in zip(RHO, W))
        self.overview_results.total_wait_time = T * sum(w.sum() for w in W)
        self.overview_results.total_distance_traveled = T * L * sum(q.sum() for q in Q)
        self.overview_results.congestion_time = T * sum(1 for rho in RHO if rho.sum() > rho_crit * RHO.shape[1])
        self.overview_results.excess_congestion_time = T * sum(np.maximum(0, rho - rho_crit).sum() for rho in RHO)
        self.overview_results.total_speed_variance = T * sum(((v - v_free) ** 2).sum() for v in V)

        return self.overview_results