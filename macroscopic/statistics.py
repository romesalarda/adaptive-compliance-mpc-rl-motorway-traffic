from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class MacroSimulationResults:
    """
    Class to store statistics of a macroscopic simulation run.
    """
    # physical quantities
    rho: list = None  # Density array 
    v: list = None  # Speed array
    w: list = None  # queue
    q: list = None  # Flow array
    q_on: list = None  # On-ramp flow array
    v_ctrl: list = None  # Control speed array
    r: list = None  # Ramp metering rate array

    def __post_init__(self):
        # Initialize lists if they are None
        if self.rho is None:
            self.rho = []
        if self.v is None:
            self.v = []
        if self.w is None:
            self.w = []
        if self.q is None:
            self.q = []
        if self.q_on is None:
            self.q_on = []
        if self.v_ctrl is None:
            self.v_ctrl = []
        if self.r is None:
            self.r = []
    

@dataclass(slots=True)
class MacroOverviewSimulationResults:
    
    total_time_spent: float = 0.0  # Total time spent
    total_distance_traveled: float = 0.0  # Total distance traveled
    congestion_time: float = 0.0  # Total congestion time
    excess_congestion_time: float = 0.0  # Total excess congestion time
    total_speed_variance: float = 0.0  # Total speed variance
    total_wait_time: float = 0.0  # Total wait time
    

@dataclass(slots=True)  
class MacroRunSimulationResults:

    total_simulation_time: float = 0.0  # Total simulation time