from macroscopic.simulation.environment import MacroEnvironment
from macroscopic.statistics.base import (
    MacroSimulationResults,
    MacroOverviewSimulationResults,
    MacroRunSimulationResults,
)
from macroscopic.simulation.parameters import BaseParameters, VslParameters
from macroscopic.logger import LOGGER

import sym_metanet as smn
import copy
import numpy as np

import typing

class BaseSimulation:
    '''
    Base class for macroscopic simulation. This class provides a common interface for different types of simulations, 
    including MPC and non-MPC simulations. It is designed to be extended by specific simulation classes that implement 
    the setup and simulate methods.
    '''

    def __init__(self, environment: MacroEnvironment, function_compactness: int = None):
        '''
        Initializes the BaseSimulation class.

        Args:
            environment (Environment): The simulation environment containing parameters, network, and demands.
            function_compactness (int, optional): The compactness level for the function representation. Defaults to None.
        '''
        self.environment = environment
        self.function_compactness = function_compactness

        self.simulation_results: MacroSimulationResults = MacroSimulationResults()  # Placeholder for simulation results, to be defined in subclasses
        self.overview_results: MacroOverviewSimulationResults = MacroOverviewSimulationResults()  # Placeholder for overview results, to be defined in subclasses
        self.run_results: MacroRunSimulationResults = MacroRunSimulationResults()  # Placeholder for run results, to be defined in subclasses


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
        F = smn.engine.to_function(
            net=self.environment.network,
            more_out=more_out,
            compact=compact,
            T=self.environment.parameters.T,
        )

        for el, acts in self.environment.network.actions.items():
            LOGGER.info(f"Network element '{el.name}' has {len(acts)} actions: {acts}")

        return F
    
    def store_step(self, rho, v, w, q, q_on, omit_if_none=False, **kwargs):
        '''
        Store the simulation results for a single time step. This method appends the current state variables and outputs to the simulation results object, allowing for later analysis and visualization of the traffic flow dynamics.
        '''
        self.simulation_results.RHO.append(rho)
        self.simulation_results.V.append(v)
        self.simulation_results.W.append(w)
        self.simulation_results.Q.append(q)
        self.simulation_results.Q_o.append(q_on)

        if not omit_if_none: # store even if None 
            self.simulation_results.R.append(kwargs.get("r_last", None))
            self.simulation_results.V_CTRL.append(kwargs.get("v_ctrl_last", None))
            return

        # store only if not None
        if kwargs.get("r_last", None) is not None:
            self.simulation_results.R.append(kwargs.get("r_last", None))
        if kwargs.get("v_ctrl_last", None) is not None:
            self.simulation_results.V_CTRL.append(kwargs.get("v_ctrl_last", None))

    def post_simulate(self):
        '''
        Post-process the simulation results after the simulation is complete. This method can be used to compute additional statistics, generate plots, or perform any other necessary post-simulation analysis.
        '''
        RHO, V, W, Q, Q_o, R = (np.squeeze(o) for o in (
            self.simulation_results.RHO, 
            self.simulation_results.V, 
            self.simulation_results.W,
            self.simulation_results.Q, 
            self.simulation_results.Q_o, 
            self.simulation_results.R
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

        self.simulation_results.rho = RHO
        self.simulation_results.v = V
        self.simulation_results.w = W
        self.simulation_results.q = Q
        self.simulation_results.q_on = Q_o
        self.simulation_results.r = R
        self.simulation_results.v_ctrl = np.squeeze(np.array(self.simulation_results.V_CTRL)) if self.simulation_results.V_CTRL else None

        self.simulation_results.origin_names = list(self.environment.network.origins_by_name.keys())
        self.simulation_results.demands = self.environment.demands
        self.simulation_results.parameters = self.environment.parameters

        return self.overview_results

class BaseModelSimulation(BaseSimulation):
    '''
    Base model simulation, no MPC, just pure simulation of the model with given parameters and initial conditions.
    '''
    def __init__(self, 
                 environment: MacroEnvironment, 
                 function_compactness: int = None, **kwargs):
        super().__init__(environment, function_compactness)
        self.kwargs = kwargs  # Store additional keyword arguments for use in the simulation
        self.parameters: typing.Union[BaseParameters, VslParameters] = environment.parameters  # Store the parameters from the environment for easy access
  
    def setup(self):
        self.environment.ready(self.kwargs.get("initial_conditions", None)) 

    def pre_simulate(self):
        pass

    def _simulate(self):

        self.environment.check()  # Ensure the environment is ready before simulation
        
        F = self.get_function(more_out=True, compact=self.function_compactness)

        rho = copy.copy(self.parameters.rho_init)
        v = copy.copy(self.parameters.v_free_init)
        w = copy.copy(self.parameters.w_init)
        
        r = copy.copy(self.parameters.r_init)
        
        if hasattr(self.parameters, "v_ctrl"):
            v_ctrl = copy.copy(self.parameters.v_ctrl)
        else:
            v_ctrl = None

        for t_idx, d in enumerate(self.environment.demands):
    
            if v_ctrl is not None:
                rho, v, w, q, q_o = F(rho, v, w, v_ctrl, r, d)   
            else:
                rho, v, w, q, q_o = F(rho, v, w, r, d)
            # print(f"step {t_idx} of {self.environment.demands.shape[0]}")
            # print(f"rho: {rho}, v: {v}, w: {w}, q: {q}, q_o: {q_o}, r: {r}, v_ctrl: {v_ctrl}")
            self.store_step(rho, v, w, q, q_o, omit_if_none=False, r_last=r, v_ctrl_last=v_ctrl)

    def simulate(self):
        '''
        Runs the complete simulation process, including setup, pre-simulation steps, the main simulation loop, and post-simulation steps.
        '''
        self.setup()
        self.pre_simulate()
        self._simulate()
        self.post_simulate()
        


        
