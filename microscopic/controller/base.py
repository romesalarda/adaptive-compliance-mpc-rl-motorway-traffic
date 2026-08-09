
from macroscopic.simulation.environment import MacroEnvironment
from microscopic.simulation.environment import MicroEnvironment


class BaseSUMOController:
    def __init__(self, 
        macroscopic_env: MacroEnvironment,
        microscopic_env: MicroEnvironment
        ):
        self.macro_env = macroscopic_env
        self.micro_env = microscopic_env

    