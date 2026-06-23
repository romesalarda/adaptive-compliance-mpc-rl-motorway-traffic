from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class MacroSimulationResults:
    """
    Class to store statistics of a macroscopic simulation run.
    """
    # physical quantities
    rho: np.ndarray = None  # Density array 
    v: np.ndarray = None  # Speed array
    w: np.ndarray = None  # queue
    q: np.ndarray = None  # Flow array
    q_on: np.ndarray = None  # On-ramp flow array
    v_ctrl: np.ndarray = None  # Control speed array
    r: np.ndarray = None  # Ramp metering rate array
    

@dataclass(slots=True)
class MacroOverviewSimulationResults:
    
    total_travel_time: float = 0.0  # Total travel time
    total_time_spent: float = 0.0  # Total time spent
    total_distance_traveled: float = 0.0  # Total distance traveled
    congestion_time: float = 0.0  # Total congestion time
    excess_congestion_time: float = 0.0  # Total excess congestion time
    total_speed_variance: float = 0.0  # Total speed variance
    total_vehicles: int = 0  # Total number of vehicles

@dataclass(slots=True)  
class MacroRunSimulationResults:

    total_simulation_time: float = 0.0  # Total simulation time