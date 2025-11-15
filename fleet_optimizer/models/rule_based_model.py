"""
Rule-based optimization model.

Simple heuristic-based decision making using if/then rules.
"""

from typing import List, Dict
from .base_model import BaseOptimizationModel


class RuleBasedModel(BaseOptimizationModel):
    """
    Rule-based optimization model.

    Uses simple heuristics:
    1. If battery < low_battery_threshold, go charge
    2. If available and requests pending, pick up nearest request
    3. Otherwise, idle
    """

    def __init__(self, config: dict = None):
        """
        Initialize rule-based model.

        Config parameters:
            - low_battery_threshold: Battery SOC below which vehicle should charge (default: 0.3)
            - charge_target_soc: Target SOC when charging (default: 0.9)
            - max_pickup_distance_km: Maximum distance to consider for pickup (default: 10.0)
        """
        super().__init__(config)
        self.low_battery_threshold = self.config.get('low_battery_threshold', 0.3)
        self.charge_target_soc = self.config.get('charge_target_soc', 0.9)
        self.max_pickup_distance_km = self.config.get('max_pickup_distance_km', 10.0)

    def make_decisions(self, state: dict) -> List[dict]:
        """Make decisions using simple rules"""
        fleet = state['fleet']
        depots = state['depots']
        active_requests = state['active_requests']
        distance_func = state['distance_func']

        decisions = []

        # Get available vehicles
        available_vehicles = fleet.get_available_vehicles()

        # Get pending requests
        pending_requests = [r for r in active_requests.values() if r.is_pending]

        for vehicle in available_vehicles:
            decision = None

            # Rule 1: Low battery? Go charge!
            if vehicle.battery_soc < self.low_battery_threshold:
                decision = self._decide_charging(vehicle, depots, distance_func)

            # Rule 2: Requests pending and enough battery? Pick up!
            elif pending_requests:
                decision = self._decide_pickup(
                    vehicle,
                    pending_requests,
                    distance_func
                )

            # Rule 3: Otherwise idle
            else:
                decision = {
                    'vehicle_id': vehicle.id,
                    'action': 'IDLE'
                }

            if decision:
                decisions.append(decision)

        return decisions

    def _decide_charging(self, vehicle, depots: Dict, distance_func) -> dict:
        """
        Decide which depot to charge at.

        Chooses nearest depot with available slots.
        """
        best_depot = None
        min_distance = float('inf')

        for depot in depots.values():
            if depot.has_available_slot():
                distance = distance_func(vehicle.location, depot.location)
                if distance < min_distance:
                    min_distance = distance
                    best_depot = depot

        if best_depot:
            return {
                'vehicle_id': vehicle.id,
                'action': 'CHARGING',
                'depot_id': best_depot.id
            }

        # No available depot, stay idle
        return {
            'vehicle_id': vehicle.id,
            'action': 'IDLE'
        }

    def _decide_pickup(self, vehicle, pending_requests: List, distance_func) -> dict:
        """
        Decide which request to pick up.

        Chooses nearest request that vehicle can complete.
        """
        best_request = None
        min_distance = float('inf')

        for request in pending_requests:
            # Calculate distance to pickup
            pickup_distance = distance_func(vehicle.location, request.pickup_location)

            # Skip if too far
            if pickup_distance > self.max_pickup_distance_km:
                continue

            # Calculate trip distance
            trip_distance = distance_func(
                request.pickup_location,
                request.dropoff_location
            )

            # Check if vehicle can complete the trip
            total_distance = pickup_distance + trip_distance
            if not vehicle.can_complete_trip(total_distance):
                continue

            # Check if this is the closest
            if pickup_distance < min_distance:
                min_distance = pickup_distance
                best_request = request

        if best_request:
            return {
                'vehicle_id': vehicle.id,
                'action': 'PICKUP_PASSENGER',
                'request_id': best_request.request_id
            }

        # No suitable request found
        return {
            'vehicle_id': vehicle.id,
            'action': 'IDLE'
        }

    def __repr__(self) -> str:
        return (f"RuleBasedModel(low_battery={self.low_battery_threshold:.1%}, "
                f"max_pickup_dist={self.max_pickup_distance_km}km)")
