"""
Feature extraction utilities for ML training.

Converts simulation state into feature vectors suitable for ML models.
"""

import numpy as np
from typing import Dict, List, Any, Tuple


class StateFeatureExtractor:
    """
    Extracts feature vectors from simulation state.

    Converts complex nested state dictionaries into flat numpy arrays
    suitable for ML models.
    """

    def __init__(self, max_vehicles: int = 200, max_requests: int = 100):
        """
        Initialize feature extractor.

        Args:
            max_vehicles: Maximum number of vehicles to handle
            max_requests: Maximum number of pending requests to handle
        """
        self.max_vehicles = max_vehicles
        self.max_requests = max_requests

    def extract_state_features(self, state: dict) -> np.ndarray:
        """
        Extract features from full simulation state.

        Args:
            state: State dictionary from simulation

        Returns:
            Flattened feature vector as numpy array
        """
        features = []

        # Temporal features
        features.extend(self._extract_temporal_features(state))

        # Fleet aggregate features
        features.extend(self._extract_fleet_features(state))

        # Depot features
        features.extend(self._extract_depot_features(state))

        # Request features
        features.extend(self._extract_request_features(state))

        return np.array(features, dtype=np.float32)

    def _extract_temporal_features(self, state: dict) -> List[float]:
        """Extract time-related features"""
        current_time = state.get('current_time', 0.0)
        hour = int((current_time / 60) % 24)

        # Cyclical encoding of hour (sin/cos to handle 23->0 wraparound)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Day progress (0-1)
        day_progress = (current_time % 1440) / 1440

        return [hour_sin, hour_cos, day_progress]

    def _extract_fleet_features(self, state: dict) -> List[float]:
        """Extract fleet-level aggregate features"""
        fleet = state.get('fleet')
        if not fleet:
            return [0.0] * 15

        vehicles = list(fleet.vehicles.values())

        if not vehicles:
            return [0.0] * 15

        # Battery statistics
        battery_socs = [v.battery_soc for v in vehicles]
        avg_battery = np.mean(battery_socs)
        min_battery = np.min(battery_socs)
        max_battery = np.max(battery_socs)
        std_battery = np.std(battery_socs)

        # Status distribution (as ratios)
        total_vehicles = len(vehicles)
        idle_ratio = len([v for v in vehicles if v.is_available]) / total_vehicles
        charging_ratio = len([v for v in vehicles if 'CHARG' in v.status.value.upper()]) / total_vehicles
        busy_ratio = len([v for v in vehicles if v.current_passenger_id is not None]) / total_vehicles

        # Location spread (simplified - could use more sophisticated spatial features)
        lats = [v.location[0] for v in vehicles]
        lons = [v.location[1] for v in vehicles]
        lat_spread = np.std(lats) if lats else 0.0
        lon_spread = np.std(lons) if lons else 0.0

        # Utilization metrics
        total_distance = sum(v.total_distance_km for v in vehicles)
        distance_with_passenger = sum(v.total_distance_with_passenger_km for v in vehicles)
        utilization = distance_with_passenger / total_distance if total_distance > 0 else 0.0

        return [
            avg_battery, min_battery, max_battery, std_battery,
            idle_ratio, charging_ratio, busy_ratio,
            lat_spread, lon_spread,
            utilization,
            total_vehicles / self.max_vehicles,  # Normalized fleet size
            sum(v.completed_trips for v in vehicles) / (total_vehicles * 10),  # Normalized trips
        ]

    def _extract_depot_features(self, state: dict) -> List[float]:
        """Extract depot-level features"""
        depots = state.get('depots', {})

        if not depots:
            return [0.0] * 8

        depot_list = list(depots.values())

        # Aggregate metrics
        total_slots = sum(d.total_slots for d in depot_list)
        available_slots = sum(d.available_slots for d in depot_list)
        total_queue = sum(d.queue_length for d in depot_list)

        avg_utilization = np.mean([d.utilization for d in depot_list])
        max_utilization = np.max([d.utilization for d in depot_list])

        # Current time pricing (use first depot as representative)
        current_time = state.get('current_time', 0.0)
        hour = int((current_time / 60) % 24)
        current_price = depot_list[0].pricing.get_price(hour) if depot_list else 0.12

        return [
            available_slots / max(total_slots, 1),  # Ratio of available slots
            total_queue / max(total_slots, 1),  # Queue to capacity ratio
            avg_utilization,
            max_utilization,
            current_price,
            len(depot_list),  # Number of depots
        ]

    def _extract_request_features(self, state: dict) -> List[float]:
        """Extract request-level features"""
        active_requests = state.get('active_requests', {})

        pending_requests = [
            r for r in active_requests.values()
            if hasattr(r, 'is_pending') and r.is_pending
        ]

        if not pending_requests:
            return [0.0] * 10

        current_time = state.get('current_time', 0.0)

        # Request count features
        num_pending = len(pending_requests)

        # Wait time statistics
        wait_times = [current_time - r.request_time for r in pending_requests]
        avg_wait = np.mean(wait_times)
        max_wait = np.max(wait_times)

        # Urgency features (requests close to timeout)
        urgent_count = sum(1 for r in pending_requests if r.has_exceeded_max_wait(current_time) * 0.8)

        # Geographic spread of requests
        pickup_lats = [r.pickup_location[0] for r in pending_requests]
        pickup_lons = [r.pickup_location[1] for r in pending_requests]
        spatial_spread = np.std(pickup_lats) + np.std(pickup_lons)

        return [
            num_pending / self.max_requests,  # Normalized count
            avg_wait,
            max_wait,
            urgent_count / max(num_pending, 1),
            spatial_spread,
        ]

    def extract_vehicle_features(self, vehicle, state: dict) -> np.ndarray:
        """
        Extract features for a specific vehicle.

        Args:
            vehicle: Vehicle object
            state: Full simulation state

        Returns:
            Feature vector for this vehicle
        """
        features = []

        # Vehicle-specific features
        features.extend([
            vehicle.battery_soc,
            vehicle.remaining_range_km / 400.0,  # Normalized by typical max range
            1.0 if vehicle.is_available else 0.0,
            1.0 if vehicle.current_passenger_id else 0.0,
            vehicle.total_distance_km / 1000.0,  # Normalized
            vehicle.completed_trips / 50.0,  # Normalized
        ])

        # Location features (normalized)
        lat, lon = vehicle.location
        features.extend([
            (lat - 37.7) / 0.1,  # Normalized to service area
            (lon + 122.45) / 0.1,
        ])

        # Context features (same for all vehicles)
        features.extend(self._extract_temporal_features(state))

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability"""
        return [
            # Temporal (3)
            'hour_sin', 'hour_cos', 'day_progress',
            # Fleet (12)
            'avg_battery', 'min_battery', 'max_battery', 'std_battery',
            'idle_ratio', 'charging_ratio', 'busy_ratio',
            'lat_spread', 'lon_spread', 'utilization',
            'fleet_size_norm', 'trips_norm',
            # Depot (6)
            'available_slots_ratio', 'queue_ratio', 'avg_depot_util',
            'max_depot_util', 'electricity_price', 'num_depots',
            # Request (5)
            'pending_requests_norm', 'avg_wait', 'max_wait',
            'urgent_ratio', 'request_spatial_spread',
        ]

    def get_feature_dim(self) -> int:
        """Get total dimension of feature vector"""
        return len(self.get_feature_names())


def action_to_vector(action: dict) -> np.ndarray:
    """
    Convert action dictionary to one-hot encoded vector.

    Args:
        action: Action dictionary with 'action' key

    Returns:
        One-hot encoded action vector
    """
    action_types = ['IDLE', 'PICKUP_PASSENGER', 'CHARGING', 'REPOSITION']
    action_name = action.get('action', 'IDLE')

    vector = np.zeros(len(action_types), dtype=np.float32)
    if action_name in action_types:
        idx = action_types.index(action_name)
        vector[idx] = 1.0

    return vector


def actions_to_matrix(actions: List[dict]) -> np.ndarray:
    """
    Convert list of actions to matrix.

    Args:
        actions: List of action dictionaries

    Returns:
        Matrix of one-hot encoded actions
    """
    return np.array([action_to_vector(a) for a in actions])
