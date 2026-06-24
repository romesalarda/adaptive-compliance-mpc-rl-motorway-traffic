import time
import sym_metanet as smn

from typing import Any, Optional, Sequence, TypeVar, Union, Literal
from csnlp.wrappers import Mpc
from csnlp import Nlp
import casadi as cs

SymType = TypeVar("SymType", cs.SX, cs.MX)


from macroscopic.simulation.environment import Environment
from macroscopic.simulation.statistics import (
    MacroSimulationResults, 
    MacroOverviewSimulationResults, 
    MacroRunSimulationResults
)
from macroscopic.logger import LOGGER



class BaseMacroscopicSolver:
    """
    Base class for macroscopic solvers.
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

        self.rho, _ = self.MPC.state("rho", n_seg, lb=kwargs.get("rho_lb", 0), ub=kwargs.get("rho_ub", 1e6))
        self.v, _ = self.MPC.state("v", n_seg, lb=kwargs.get("v_lb", 0), ub=kwargs.get("v_ub", 1e6))

        if self.queue_upper_bound is not None:
            assert len(self.queue_upper_bound) == self.number_of_origins, "Queue upper bound must match the number of origins."
        self.w, _ = self.MPC.state("w", self.number_of_origins, lb=kwargs.get("w_lb", 0), ub=kwargs.get("w_ub", self.queue_upper_bound))

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
            LOGGER.info("\n3. Proceeding with the initialization of constraints...")
            self.init_constraints()
            LOGGER.info("Constraints initialization completed successfully. ✓")
            LOGGER.info("\n4. Proceeding with the initialization of disturbances...")
            self.init_disturbances()
            LOGGER.info("Disturbances initialization completed successfully. ✓")
            LOGGER.info("\n5. Proceeding with the initialization of the objective function...")
            self.init_objective()
            LOGGER.info("Objective function initialization completed successfully. ✓")
            LOGGER.info("\n6. Proceeding with the initialization of non-linear dynamics...")
            self.init_non_linear_dynamics()
            LOGGER.info("Non-linear dynamics initialization completed successfully. ✓")
            LOGGER.info("\n7. Proceeding with the initialization of the solver...")
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

    def pre_solve(self):
        pass

    def solve(self):
        """
        Solve the macroscopic traffic flow problem.
        This method should be implemented by subclasses.
        """
        self.pre_solve()
        self._solve()
        self.post_solve()

    def _solve(self):
        
        rho = self.environment.parameters.rho_init
        v = self.environment.parameters.v_free_init
        w  = self.environment.parameters.w_init        
    
    def post_solve(self):
        pass