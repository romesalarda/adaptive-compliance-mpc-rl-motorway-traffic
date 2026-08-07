from macroscopic.statistics.base import (
    MacroSimulationResults, MacroOverviewSimulationResults, MacroRunSimulationResults
)

class MPCSimulationVisualisation:

    def __init__(self, 
                 simulation_results: MacroSimulationResults, 
                 overview_results: MacroOverviewSimulationResults, 
                 run_results: MacroRunSimulationResults
                 ):
        self.simulation_results = simulation_results
        self.overview_results = overview_results
        self.run_results = run_results