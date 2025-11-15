"""
Greedy optimization model.

More sophisticated decision making using greedy assignment based on
profitability and efficiency metrics.
"""

from typing import List, Dict, Tuple
import numpy as np
from .base_model import BaseOptimizationModel


class GreedyModel(BaseOptimizationModel):
    """
    Greedy optimization model.

    Makes decisions by optimizing for immediate profitability:
    1. Calculate profit/cost for each vehicle-request pair
    2. Assign vehicles to requests greedily (best match first)
    3. Send low-battery vehicles to charge during off-peak hours
    4. Reposition idle vehicles to high-demand areas
    """

    def __init__(self, config: dict = None):
        """
        Initialize greedy model.

        Config parameters:
            - low_battery_threshold: Battery SOC below which vehicle should charge (default: 0.25)
            - critical_battery_threshold: Critical battery level (default: 0.15)
            - charge_target_soc: Target SOC when charging (default: 0.9)
            - max_pickup_distance_km: Maximum distance to consider for pickup (default: 15.0)
            - base_fare: Base fare per trip (default: 3.0)
            - per_km_rate: Rate per kilometer (default: 1.5)
            - energy_cost_per_kwh: Cost of energy (default: 0.12)
            - enable_repositioning: Enable repositioning to high demand areas (default: False)
        """
        super().__init__(config)
        self.low_battery_threshold = self.config.get('low_battery_threshold', 0.25)
        self.critical_battery_threshold = self.config.get('critical_battery_threshold', 0.15)
        self.charge_target_soc = self.config.get('charge_target_soc', 0.9)
        self.max_pickup_distance_km = self.config.get('max_pickup_distance_km', 15.0)
        self.base_fare = self.config.get('base_fare', 3.0)
        self.per_km_rate = self.config.get('per_km_rate', 1.5)
        self.energy_cost_per_kwh = self.config.get('energy_cost_per_kwh', 0.12)
        self.enable_repositioning = self.config.get('enable_repositioning', False)

    def make_decisions(self, state: dict) -> List[dict]:
        """Make decisions using greedy optimization"""
        fleet = state['fleet']
        depots = state['depots']
        active_requests = state['active_requests']
        distance_func = state['distance_func']
        current_time = state['current_time']

        decisions = []

        # Get available vehicles
        available_vehicles = fleet.get_available_vehicles()

        # Get pending requests
        pending_requests = [r for r in active_requests.values() if r.is_pending]

        # Separate vehicles by battery level
        critical_battery_vehicles = []
        low_battery_vehicles = []
        normal_battery_vehicles = []

        for vehicle in available_vehicles:
            if vehicle.battery_soc < self.critical_battery_threshold:
                critical_battery_vehicles.append(vehicle)
            elif vehicle.battery_soc < self.low_battery_threshold:
                low_battery_vehicles.append(vehicle)
            else:
                normal_battery_vehicles.append(vehicle)

        # Priority 1: Send critical battery vehicles to charge immediately
        for vehicle in critical_battery_vehicles:
            decision = self._decide_charging(vehicle, depots, distance_func, current_time)
            if decision:
                decisions.append(decision)

        # Priority 2: Assign normal battery vehicles to requests greedily
        if pending_requests and normal_battery_vehicles:
            assignments = self._greedy_assignment(
                normal_battery_vehicles,
                pending_requests,
                distance_func
            )
            decisions.extend(assignments)

            # Mark assigned vehicles/requests
            assigned_vehicle_ids = {d['vehicle_id'] for d in assignments}
            assigned_request_ids = {d.get('request_id') for d in assignments if 'request_id' in d}

            normal_battery_vehicles = [
                v for v in normal_battery_vehicles
                if v.id not in assigned_vehicle_ids
            ]
            pending_requests = [
                r for r in pending_requests
                if r.request_id not in assigned_request_ids
            ]

        # Priority 3: Send low battery vehicles to charge (preferably during off-peak)
        for vehicle in low_battery_vehicles:
            # Check if it's off-peak hour (cheaper charging)
            hour = int((current_time / 60) % 24)
            is_off_peak = hour < 9 or hour > 18

            if is_off_peak or vehicle.battery_soc < self.low_battery_threshold:
                decision = self._decide_charging(vehicle, depots, distance_func, current_time)
                if decision:
                    decisions.append(decision)

        # Priority 4: Idle vehicles - either reposition or stay idle
        idle_vehicles = normal_battery_vehicles + [
            v for v in low_battery_vehicles
            if v.id not in {d['vehicle_id'] for d in decisions}
        ]

        for vehicle in idle_vehicles:
            if self.enable_repositioning and pending_requests:
                # Reposition to area with high demand
                decision = self._decide_reposition(vehicle, pending_requests, distance_func)
            else:
                decision = {'vehicle_id': vehicle.id, 'action': 'IDLE'}

            if decision:
                decisions.append(decision)

        return decisions

    def _calculate_profit(
        self,
        vehicle,
        request,
        pickup_distance_km: float,
        trip_distance_km: float
    ) -> float:
        """
        Calculate expected profit for a vehicle-request assignment.

        Args:
            vehicle: Vehicle object
            request: RideRequest object
            pickup_distance_km: Distance to pickup location
            trip_distance_km: Distance from pickup to dropoff

        Returns:
            Expected profit (revenue - cost)
        """
        # Revenue
        revenue = self.base_fare + (trip_distance_km * self.per_km_rate)

        # Cost (energy for deadhead + trip)
        total_distance = pickup_distance_km + trip_distance_km
        energy_kwh = total_distance * vehicle.specs.efficiency_kwh_per_km * 1.05  # 5% overhead
        energy_cost = energy_kwh * self.energy_cost_per_kwh

        # Profit
        profit = revenue - energy_cost

        # Penalty for long deadhead (reduce profit)
        if pickup_distance_km > 0:
            deadhead_penalty = pickup_distance_km * 0.5  # $0.5 per km deadhead
            profit -= deadhead_penalty

        return profit

    def _greedy_assignment(
        self,
        vehicles: List,
        requests: List,
        distance_func
    ) -> List[dict]:
        """
        Perform greedy assignment of vehicles to requests.

        Creates all possible vehicle-request pairs, sorts by profit,
        and assigns greedily ensuring each vehicle and request is assigned once.

        Args:
            vehicles: List of available vehicles
            requests: List of pending requests
            distance_func: Distance calculation function

        Returns:
            List of assignment decisions
        """
        # Calculate profit for all vehicle-request pairs
        assignments = []

        for vehicle in vehicles:
            for request in requests:
                pickup_distance = distance_func(vehicle.location, request.pickup_location)

                # Skip if too far
                if pickup_distance > self.max_pickup_distance_km:
                    continue

                trip_distance = distance_func(
                    request.pickup_location,
                    request.dropoff_location
                )

                # Check if vehicle can complete the trip
                total_distance = pickup_distance + trip_distance
                if not vehicle.can_complete_trip(total_distance):
                    continue

                # Calculate profit
                profit = self._calculate_profit(
                    vehicle,
                    request,
                    pickup_distance,
                    trip_distance
                )

                assignments.append({
                    'vehicle_id': vehicle.id,
                    'request_id': request.request_id,
                    'profit': profit,
                    'pickup_distance': pickup_distance,
                })

        # Sort by profit (descending)
        assignments.sort(key=lambda x: x['profit'], reverse=True)

        # Greedy assignment
        assigned_vehicles = set()
        assigned_requests = set()
        decisions = []

        for assignment in assignments:
            vehicle_id = assignment['vehicle_id']
            request_id = assignment['request_id']

            if vehicle_id not in assigned_vehicles and request_id not in assigned_requests:
                decisions.append({
                    'vehicle_id': vehicle_id,
                    'action': 'PICKUP_PASSENGER',
                    'request_id': request_id
                })
                assigned_vehicles.add(vehicle_id)
                assigned_requests.add(request_id)

        return decisions

    def _decide_charging(
        self,
        vehicle,
        depots: Dict,
        distance_func,
        current_time: float
    ) -> dict:
        """
        Decide which depot to charge at.

        Considers:
        - Distance to depot
        - Availability of charging slots
        - Electricity pricing (prefer off-peak)
        """
        hour = int((current_time / 60) % 24)

        best_depot = None
        best_score = float('-inf')

        for depot in depots.values():
            if not depot.has_available_slot():
                continue

            distance = distance_func(vehicle.location, depot.location)

            # Can vehicle reach the depot?
            if not vehicle.can_complete_trip(distance):
                continue

            # Calculate score (higher is better)
            # Factors: closer is better, cheaper electricity is better
            distance_score = 1.0 / (distance + 1.0)  # Avoid division by zero
            price = depot.pricing.get_price(hour)
            price_score = 1.0 / price

            # Weighted score
            score = (distance_score * 0.7) + (price_score * 0.3)

            if score > best_score:
                best_score = score
                best_depot = depot

        if best_depot:
            return {
                'vehicle_id': vehicle.id,
                'action': 'CHARGING',
                'depot_id': best_depot.id
            }

        return None

    def _decide_reposition(
        self,
        vehicle,
        pending_requests: List,
        distance_func
    ) -> dict:
        """
        Decide where to reposition vehicle.

        Repositions to centroid of pending requests.
        """
        if not pending_requests:
            return {'vehicle_id': vehicle.id, 'action': 'IDLE'}

        # Calculate centroid of pending requests
        lats = [r.pickup_location[0] for r in pending_requests]
        lons = [r.pickup_location[1] for r in pending_requests]

        centroid_lat = np.mean(lats)
        centroid_lon = np.mean(lons)
        target_location = (centroid_lat, centroid_lon)

        return {
            'vehicle_id': vehicle.id,
            'action': 'REPOSITION',
            'target_location': target_location
        }

    def __repr__(self) -> str:
        return (f"GreedyModel(low_battery={self.low_battery_threshold:.1%}, "
                f"reposition={self.enable_repositioning})")
