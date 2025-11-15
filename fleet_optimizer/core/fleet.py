"""
Fleet management module.

Defines the Fleet class which manages a collection of vehicles.
"""

from typing import List, Dict, Optional
import numpy as np
from .vehicle import Vehicle, VehicleStatus, VehicleSpecs


class Fleet:
    """
    Manages a collection of vehicles.

    Provides methods for fleet-level operations and queries.
    """

    def __init__(self, vehicles: Optional[List[Vehicle]] = None):
        """
        Initialize a fleet.

        Args:
            vehicles: List of Vehicle objects (optional)
        """
        self.vehicles: Dict[str, Vehicle] = {}
        if vehicles:
            for vehicle in vehicles:
                self.vehicles[vehicle.id] = vehicle

    def add_vehicle(self, vehicle: Vehicle):
        """Add a vehicle to the fleet"""
        self.vehicles[vehicle.id] = vehicle

    def remove_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        """Remove a vehicle from the fleet"""
        return self.vehicles.pop(vehicle_id, None)

    def get_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        """Get a vehicle by ID"""
        return self.vehicles.get(vehicle_id)

    @property
    def size(self) -> int:
        """Total number of vehicles in fleet"""
        return len(self.vehicles)

    def get_available_vehicles(self) -> List[Vehicle]:
        """Get all available (idle) vehicles"""
        return [v for v in self.vehicles.values() if v.is_available]

    def get_vehicles_by_status(self, status: VehicleStatus) -> List[Vehicle]:
        """Get all vehicles with a specific status"""
        return [v for v in self.vehicles.values() if v.status == status]

    def get_vehicles_needing_charge(self, threshold: float = 0.3) -> List[Vehicle]:
        """
        Get vehicles below a battery threshold.

        Args:
            threshold: Battery SOC threshold (0-1)

        Returns:
            List of vehicles below threshold
        """
        return [v for v in self.vehicles.values() if v.battery_soc < threshold]

    def get_vehicles_in_area(
        self,
        center: tuple,
        radius_km: float,
        distance_func
    ) -> List[Vehicle]:
        """
        Get vehicles within a radius of a location.

        Args:
            center: (latitude, longitude) center point
            radius_km: Radius in km
            distance_func: Function to calculate distance between two points

        Returns:
            List of vehicles within radius
        """
        vehicles_in_area = []
        for vehicle in self.vehicles.values():
            distance = distance_func(vehicle.location, center)
            if distance <= radius_km:
                vehicles_in_area.append(vehicle)
        return vehicles_in_area

    def get_closest_vehicle(
        self,
        location: tuple,
        distance_func,
        filter_func=None
    ) -> Optional[Vehicle]:
        """
        Get closest vehicle to a location.

        Args:
            location: (latitude, longitude) target location
            distance_func: Function to calculate distance
            filter_func: Optional function to filter vehicles (e.g., only available)

        Returns:
            Closest vehicle or None
        """
        candidates = self.vehicles.values()
        if filter_func:
            candidates = [v for v in candidates if filter_func(v)]

        if not candidates:
            return None

        closest = min(
            candidates,
            key=lambda v: distance_func(v.location, location)
        )
        return closest

    def get_fleet_stats(self) -> dict:
        """
        Get fleet-level statistics.

        Returns:
            Dictionary with fleet statistics
        """
        if not self.vehicles:
            return {
                'total_vehicles': 0,
                'available_vehicles': 0,
                'avg_battery_soc': 0.0,
                'avg_utilization': 0.0,
            }

        vehicles_list = list(self.vehicles.values())

        # Count by status
        status_counts = {}
        for status in VehicleStatus:
            count = len(self.get_vehicles_by_status(status))
            status_counts[status.value] = count

        # Battery statistics
        battery_socs = [v.battery_soc for v in vehicles_list]
        avg_battery_soc = np.mean(battery_socs)
        min_battery_soc = np.min(battery_socs)
        max_battery_soc = np.max(battery_socs)

        # Distance and utilization
        total_distance = sum(v.total_distance_km for v in vehicles_list)
        total_distance_with_passenger = sum(
            v.total_distance_with_passenger_km for v in vehicles_list
        )
        total_distance_empty = sum(v.total_distance_empty_km for v in vehicles_list)

        utilization = (
            total_distance_with_passenger / total_distance
            if total_distance > 0 else 0.0
        )

        # Energy statistics
        total_energy_consumed = sum(
            v.total_energy_consumed_kwh for v in vehicles_list
        )
        avg_energy_per_vehicle = total_energy_consumed / len(vehicles_list)

        # Trip statistics
        total_trips = sum(v.completed_trips for v in vehicles_list)

        return {
            'total_vehicles': len(vehicles_list),
            'available_vehicles': len(self.get_available_vehicles()),
            'status_counts': status_counts,
            'avg_battery_soc': avg_battery_soc,
            'min_battery_soc': min_battery_soc,
            'max_battery_soc': max_battery_soc,
            'total_distance_km': total_distance,
            'total_distance_with_passenger_km': total_distance_with_passenger,
            'total_distance_empty_km': total_distance_empty,
            'utilization': utilization,
            'empty_miles_ratio': total_distance_empty / total_distance if total_distance > 0 else 0.0,
            'total_energy_consumed_kwh': total_energy_consumed,
            'avg_energy_per_vehicle_kwh': avg_energy_per_vehicle,
            'total_completed_trips': total_trips,
        }

    @staticmethod
    def create_homogeneous_fleet(
        num_vehicles: int,
        specs: VehicleSpecs,
        initial_locations: List[tuple],
        initial_battery_soc: float = 1.0,
        id_prefix: str = "VEH"
    ) -> 'Fleet':
        """
        Create a fleet of identical vehicles.

        Args:
            num_vehicles: Number of vehicles to create
            specs: VehicleSpecs for all vehicles
            initial_locations: List of (lat, lon) tuples for initial positions
            initial_battery_soc: Initial battery state of charge
            id_prefix: Prefix for vehicle IDs

        Returns:
            Fleet object with created vehicles
        """
        if len(initial_locations) != num_vehicles:
            raise ValueError(
                f"Number of locations ({len(initial_locations)}) must match "
                f"number of vehicles ({num_vehicles})"
            )

        vehicles = []
        for i in range(num_vehicles):
            vehicle = Vehicle(
                vehicle_id=f"{id_prefix}_{i:04d}",
                initial_location=initial_locations[i],
                specs=specs,
                initial_battery_soc=initial_battery_soc
            )
            vehicles.append(vehicle)

        return Fleet(vehicles)

    def __repr__(self) -> str:
        return f"Fleet(size={self.size}, available={len(self.get_available_vehicles())})"
