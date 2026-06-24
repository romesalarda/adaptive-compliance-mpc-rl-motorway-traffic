from .parameters import BaseParameters
from sym_metanet import Network, engines, Network

import uuid


class Environment:
    '''
    Encapsulates the environment for the macroscopic simulation, including parameters, network, and demands.
    '''
    def __init__(self,
            parameters: BaseParameters,
            network: Network,
            demands,         
        ):
        self.parameters = parameters
        self.network = network
        self.demands = demands
        self.id = str(uuid.uuid4())  # Unique identifier for the environment instance
        self.is_ready = False  # Flag to indicate if the environment is ready for simulation

    def __hash__(self):
        return hash(self.id)
    
    def reset(self):
        '''
        Resets the environment to its initial state.
        '''
        self.is_ready = False
    
    def ready(self, initial_conditions: dict):

        self.reset() 
        
        engines.use("casadi", sym_type='SX')

        self.network.is_valid(raises=True)

        self.network.step(
            T=self.parameters.T,
            tau=self.parameters.tau,
            eta=self.parameters.eta,
            kappa=self.parameters.kappa,
            delta=self.parameters.delta,
            initial_conditions=initial_conditions
        )

        self.is_ready = True  # Set the environment as ready after successful setup
    
    def check(self):
        if not self.is_ready:
            raise RuntimeError("Environment is not ready. Call the 'ready' method first.")