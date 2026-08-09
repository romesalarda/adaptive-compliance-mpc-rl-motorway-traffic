import typing
import traci
import traci.constants as tc


class BaseMicroEnvironment:
    """
    Encapsulates the environment for the microscopic simulation, including parameters, network, and demands.
    """
    def __init__(self):
        self.is_ready = False  # Flag to indicate if the environment is ready for simulation

    def get_edge_traffic_density(self, edge_id: str) -> float:
        pass

    def get_average_edge_speed(self, edge_id: str) -> float:
        pass

    def get_queue_at_origin(self, edge_id: str) -> int:
        pass

    def collect_state(self) -> typing.Dict[str, typing.Any]:
        """
        Collects the current state of the microscopic simulation environment.
        Returns a dictionary containing relevant state information.
        """
        pass

    def setup(self):
        pass

    def reset(self):
        pass

    def check(self):
        pass

class VariableSpeedLimitMixin:

    def set_variable_speed_limit(self, edge_id: str, speed_limit: float):
        """
        Set the variable speed limit for a specific edge in the microscopic simulation.
        """
        traci.edge.setMaxSpeed(edge_id, speed_limit)

class RampMeteringMixin:

    def set_ramp_metering_rate(self, edge_id: str, metering_rate: float):
        """
        Set the ramp metering rate for a specific edge in the microscopic simulation.
        """
        pass


class AggregateStateMixin:
    '''
    Mixin that allows the recording of states over time and the aggregation of these states into a single state representation.
    '''
    # in setup, define varibles
    # setup methods that allow cache recording
    # to be implemented later on, low priority

class MicroEnvironment(BaseMicroEnvironment, VariableSpeedLimitMixin, RampMeteringMixin):
    pass
