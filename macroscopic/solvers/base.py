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

    def get_function(self, more_out: bool = True, compact: int = 1):
        '''
        Returns a function that can be used to evaluate the macroscopic traffic flow model.

        Args:
            more_out (bool, optional): Whether to include additional outputs in the function. Defaults to True.
            compact (int, optional): Whether to return a compact version of the function. Defaults to 1.

        Returns:
            function: A function that is used to represent the dynamics of the macroscopic traffic flow model.

        '''
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
        pass

    def init_mpc(self):
        '''
        Initialize the Model Predictive Control (MPC) solver with the specified parameters.
        This method sets up the MPC solver using the provided prediction horizon (Np), control horizon (Nc), and input spacing (M). It creates an instance of the MPC solver 
        and configures it for the macroscopic traffic flow problem.  

        Returns:
            Mpc: An instance of the Model Predictive Control (MPC) solver.      
        '''
        LOGGER.info("Initializing the MPC solver...")
        self.MPC = Mpc[cs.SX](
            nlp=Nlp[cs.SX](sym_type="SX"),
            prediction_horizon=self.Np * self.M,
            control_horizon=self.Nc * self.M,
            input_spacing=self.M,
        )
        LOGGER.info("MPC solver initialized successfully. ✓")
        return self.MPC

    def init_base_states(self):
        '''
        Initialize the base states for the MPC solver.
        '''
        LOGGER.info("Initializing base states for the MPC solver...")
        n_seg = self.number_of_segments
        kwargs = self.kwargs

        self.rho, _ = self.MPC.state("rho", n_seg, lb=kwargs.get("rho_lb", 0), ub=kwargs.get("rho_ub", 1e6))
        self.v, _ = self.MPC.state("v", n_seg, lb=kwargs.get("v_lb", 0), ub=kwargs.get("v_ub", 1e6))

        if self.queue_upper_bound is not None:
            assert len(self.queue_upper_bound) == self.number_of_origins, "Queue upper bound must match the number of origins."
        self.w, _ = self.MPC.state("w", self.number_of_origins, lb=kwargs.get("w_lb", 0), ub=kwargs.get("w_ub", self.queue_upper_bound))
        LOGGER.info("Base states initialized successfully for the MPC solver. ✓")

    def init_control_states(self):
        '''
        Initialize the control states for the MPC solver.
        '''

        self.r, _ = self.MPC.action("r", lb=0, ub=1)
        self.r_last = self.MPC.parameter("r_last", (self.r.size1(), 1))


    def init_constraints(self):
        pass
    
    def init_objective(self):
        
        self.MPC.minimize(
            self.environment.parameters.T * cs.sum2(cs.sum1(self.rho * self.environment.parameters.L * self.environment.parameters.lanes) + cs.sum1(self.w))
            + self.kwargs.get("ramp_weight", 0.4) * cs.sumsqr(cs.diff(cs.horzcat(self.r_last, self.r)))
        )

    def init_disturbances(self):
        '''
        Initialize the disturbances for the MPC solver. This method sets up the disturbances that affect the macroscopic traffic flow model, such as external inputs or environmental factors. The disturbances are defined based on the number of origins in the network.
        '''
        
        return self.MPC.disturbance("d", self.number_of_origins)
    
    def init_non_linear_dynamics(self):
        '''
        Initialize the non-linear dynamics for the MPC solver. This method sets up the non-linear dynamics
        that govern the macroscopic traffic flow model. It retrieves the function representing the dynamics and configures the MPC solver to use this function for its non-linear dynamics.
        '''
        LOGGER.info("Initializing non-linear dynamics for the MPC solver...")
        F = self.get_function()
        
        self.MPC.set_nonlinear_dynamics(lambda x, u, d: F(x, u, d)[0])

        LOGGER.info("Non-linear dynamics initialized successfully for the MPC solver. ✓")
    
    def post_init_mpc(self):
        pass

    def init_solver(self, solver: str = DEFAULT_SOLVER, opts: dict = None):
        '''
        Initialize the solver for the MPC problem. This method sets up the solver that will be used to solve the optimization problem defined by the MPC formulation. The solver is configured with appropriate settings and options for efficient computation.
        '''
        LOGGER.info(f"Initializing the solver '{solver}' for the MPC problem...")
        if opts is None:
            opts = self.DEFAULT_SOLVER_OPTIONS.get(solver, {})

        self.MPC.init_solver(solver=solver, opts=opts)

        LOGGER.info(f"Solver '{solver}' initialized successfully for the MPC problem. ✓")

    def setup(self):
        """
        Setup the macroscopic solver by initializing the MPC solver and other necessary components.
        """
        LOGGER.info("Setting up the macroscopic solver...")

        self.pre_init_mpc()
        self.init_mpc()
        self.init_base_states()
        self.init_control_states()
        self.init_constraints()
        self.init_disturbances()
        self.init_objective()
        self.init_non_linear_dynamics()
        self.init_solver()
        self.post_init_mpc()

        LOGGER.info("Macroscopic solver setup complete.")

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