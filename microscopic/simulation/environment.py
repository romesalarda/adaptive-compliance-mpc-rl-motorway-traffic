from __future__ import annotations

import typing

import numpy as np
import traci
import traci.constants as tc


# Variables subscribed per edge on each simulation step
_EDGE_SUB_VARS = (
    tc.LAST_STEP_VEHICLE_NUMBER,  # int
    tc.LAST_STEP_MEAN_SPEED,      # float m/s; -1 when no vehicles
    tc.LAST_STEP_VEHICLE_NUMBER,  # int (speed < 0.1 m/s)
    tc.LAST_STEP_OCCUPANCY,       # float %
)


class BaseMicroEnvironment:
    """
    Encapsulates environment state for one microscopic simulation run.
    Communicates with SUMO via TraCI subscriptions (batch queries, one round-trip per step).
    """

    def __init__(
        self,
        edge_to_segment_map: dict[str, int],
        origin_w_map: dict[str, int],
        n_segments: int,
        n_origins: int,
        step_length: float = 0.1,
        vsl_edge_ids: list[str] | None = None,
        ramp_tl_map: dict[str, str] | None = None,
    ):
        self.edge_to_segment_map = edge_to_segment_map  # {sumo_edge_id: macro_segment_index}
        self.origin_w_map = origin_w_map                # {sumo_edge_id: w_vector_index}
        self.n_segments = n_segments
        self.n_origins = n_origins
        self.step_length = step_length
        self.vsl_edge_ids: list[str] = vsl_edge_ids or []
        self.ramp_tl_map: dict[str, str] = ramp_tl_map or {}  # origin_edge → TL id

        self.is_ready = False

        # Cached edge properties (populated during setup())
        self._edge_lengths_km: dict[str, float] = {}
        self._edge_n_lanes: dict[str, int] = {}
        self._edge_free_flow_kmh: dict[str, float] = {}

        # Latest subscription results
        self._edge_sub_results: dict[str, dict] = {}

        # RampMeteringMixin state (stored here so MRO-safe across subclasses)
        self._meter_state: dict[str, dict] = {}
        self._manual_override: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Subscribe to all relevant edges. Must be called after traci.start()."""
        all_edges = list(self.edge_to_segment_map) + list(self.origin_w_map)
        for edge_id in dict.fromkeys(all_edges):  # preserve order, deduplicate
            traci.edge.subscribe(edge_id, _EDGE_SUB_VARS)
            n_lanes = traci.edge.getLaneNumber(edge_id)
            self._edge_n_lanes[edge_id] = n_lanes
            self._edge_lengths_km[edge_id] = traci.lane.getLength(f"{edge_id}_0") / 1000.0
            self._edge_free_flow_kmh[edge_id] = traci.lane.getMaxSpeed(f"{edge_id}_0") * 3.6
        self.is_ready = True

    def update(self) -> None:
        """Pull subscription results from TraCI. Call once per simulation step."""
        for edge_id in list(self.edge_to_segment_map) + list(self.origin_w_map):
            result = traci.edge.getSubscriptionResults(edge_id)
            if result is not None:
                self._edge_sub_results[edge_id] = result

    def reset(self) -> None:
        self._edge_sub_results.clear()
        self._meter_state.clear()
        self._manual_override.clear()
        self.is_ready = False

    def check(self) -> None:
        if not self.is_ready:
            raise RuntimeError("MicroEnvironment not ready — call setup() after traci.start().")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_edge_traffic_density(self, edge_id: str) -> float:
        """Density in veh/km/lane."""
        results = self._edge_sub_results.get(edge_id, {})
        n_veh = results.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
        length_km = self._edge_lengths_km.get(edge_id, 1.0)
        lanes = self._edge_n_lanes.get(edge_id, 1)
        denom = length_km * lanes
        return n_veh / denom if denom > 0 else 0.0

    def get_average_edge_speed(self, edge_id: str) -> float:
        """Mean speed in km/h; returns free-flow speed when the edge is empty."""
        results = self._edge_sub_results.get(edge_id, {})
        speed_ms = results.get(tc.LAST_STEP_MEAN_SPEED, -1.0)
        if speed_ms < 0:
            return self._edge_free_flow_kmh.get(edge_id, 0.0)
        return speed_ms * 3.6

    def get_queue_at_origin(self, edge_id: str) -> int:
        """Number of vehicles on the origin edge (proxy for macro queue w)."""
        results = self._edge_sub_results.get(edge_id, {})
        return int(results.get(tc.LAST_STEP_VEHICLE_NUMBER, 0))

    def collect_state(self) -> dict[str, np.ndarray]:
        """
        Aggregate edge-level data into macro-compatible state arrays.
        Returns {'rho': (n_segments,), 'v': (n_segments,), 'w': (n_origins,)}
        """
        rho = np.zeros(self.n_segments, dtype=float)
        v = np.zeros(self.n_segments, dtype=float)
        w = np.zeros(self.n_origins, dtype=float)

        for edge_id, seg_idx in self.edge_to_segment_map.items():
            rho[seg_idx] = self.get_edge_traffic_density(edge_id)
            v[seg_idx] = self.get_average_edge_speed(edge_id)

        for edge_id, w_idx in self.origin_w_map.items():
            w[w_idx] = self.get_queue_at_origin(edge_id)

        return {"rho": rho, "v": v, "w": w}


class VariableSpeedLimitMixin:

    def set_variable_speed_limit(self, edge_id: str, speed_limit_kmh: float) -> None:
        """Apply speed limit to every lane on the edge."""
        speed_ms = speed_limit_kmh / 3.6
        n_lanes = traci.edge.getLaneNumber(edge_id)
        for lane_idx in range(n_lanes):
            traci.lane.setMaxSpeed(f"{edge_id}_{lane_idx}", speed_ms)


class RampMeteringMixin:
    """
    Controls ramp metering via TraCI traffic-light state.
    Each MPC solve updates green/red phase durations for the next cycle.
    Phase transitions are driven by step_metering(), called every simulation step.
    """

    # Default metering cycle length in seconds
    _DEFAULT_CYCLE_S: float = 60.0

    def set_ramp_metering_rate(self, tl_id: str, rate: float) -> None:
        """
        Set metering rate ∈ [0, 1]: fraction of the cycle that the signal is green.
        No-op when manual_override is active for this TL.
        """
        if self._manual_override.get(tl_id, False):
            return

        rate = float(np.clip(rate, 0.0, 1.0))
        step_length: float = getattr(self, "step_length", 0.1)
        cycle_steps = max(2, round(self._DEFAULT_CYCLE_S / step_length))
        green_steps = max(1, round(rate * cycle_steps))
        red_steps = max(1, cycle_steps - green_steps)

        if tl_id not in self._meter_state:
            self._meter_state[tl_id] = {
                "phase": "G",
                "timer": 0,
                "green_steps": green_steps,
                "red_steps": red_steps,
            }
            self._apply_tl_state(tl_id, "G")
        else:
            state = self._meter_state[tl_id]
            state["green_steps"] = green_steps
            state["red_steps"] = red_steps

    def step_metering(self) -> None:
        """Advance all phase state machines. Must be called once per simulation step."""
        for tl_id, state in self._meter_state.items():
            if self._manual_override.get(tl_id, False):
                continue
            state["timer"] += 1
            phase_limit = state["green_steps"] if state["phase"] == "G" else state["red_steps"]
            if state["timer"] >= phase_limit:
                new_phase = "r" if state["phase"] == "G" else "G"
                state["phase"] = new_phase
                state["timer"] = 0
                self._apply_tl_state(tl_id, new_phase)

    def _apply_tl_state(self, tl_id: str, phase: str) -> None:
        n_links = len(traci.trafficlight.getControlledLinks(tl_id))
        state_str = phase * max(1, n_links)
        
        traci.trafficlight.setRedYellowGreenState(tl_id, state_str)

    def surrender_ramp_to_manual(self, tl_id: str) -> None:
        """Bypass MPC control; external code must drive this TL directly."""
        self._manual_override[tl_id] = True

    def restore_ramp_control(self, tl_id: str) -> None:
        self._manual_override[tl_id] = False


class AggregateStateMixin:
    """Rolling-window state aggregation for smoothed MPC inputs. Stub for future use."""
    pass


class MicroEnvironment(BaseMicroEnvironment, VariableSpeedLimitMixin, RampMeteringMixin):
    pass
