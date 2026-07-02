from dataclasses import dataclass
import numpy as np
from macroscopic.simulation.parameters import BaseParameters

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

    RHO: list = None  # List of density arrays for each time step
    V: list = None  # List of speed arrays for each time step
    W: list = None  # List of queue arrays for each time step
    Q: list = None  # List of flow arrays for each time step
    Q_o: list = None  # List of on-ramp flow arrays for each time step
    V_CTRL: list = None  # List of control speed arrays for each time step
    R: list = None  # List of ramp metering rate arrays for each time step

    origin_names: list = None  # List of origin names

    demands: np.ndarray = None  # Demands array

    parameters: BaseParameters = None  # Simulation parameters

    def __post_init__(self):
        # Initialize arrays if they are None
        if self.rho is None:
            self.rho = np.array([])
        if self.v is None:
            self.v = np.array([])
        if self.w is None:
            self.w = np.array([])
        if self.q is None:
            self.q = np.array([])
        if self.q_on is None:
            self.q_on = np.array([])
        if self.v_ctrl is None:
            self.v_ctrl = np.array([])
        if self.r is None:
            self.r = np.array([])

        self.RHO = []
        self.V = []
        self.W = []
        self.Q = []
        self.Q_o = []
        self.V_CTRL = []
        self.R = []

        
    

@dataclass(slots=True)
class MacroOverviewSimulationResults:
    
    total_time_spent: float = 0.0  # Total time spent
    total_distance_traveled: float = 0.0  # Total distance traveled
    congestion_time: float = 0.0  # Total congestion time
    excess_congestion_time: float = 0.0  # Total excess congestion time
    total_speed_variance: float = 0.0  # Total speed variance
    total_wait_time: float = 0.0  # Total wait time
    
    def as_text(self, include_computation: bool = True) -> str:
        parts = [
            f"TTS={self.total_time_spent:.2f} veh·h",
            f"Wait={self.total_wait_time:.2f} veh·h",
            f"Distance={self.total_distance_traveled:.1f} veh·km",
            f"Cong.Time={self.congestion_time:.3f} h",
            f"Excess ρ={self.excess_congestion_time:.2f}",
            f"Speed Var={self.total_speed_variance:.1f}",
        ]
        if include_computation and self.computation_time is not None:
            parts.append(f"Comp.Time={self.computation_time:.1f} s")
        return " | ".join(parts)

@dataclass(slots=True)  
class MacroRunSimulationResults:

    total_simulation_time: float = 0.0  # Total simulation time