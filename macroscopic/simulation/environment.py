from .parameters import BaseParameters
from sym_metanet import Network

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

    def __hash__(self):
        return hash(self.id)
    
    def check(self):
        pass