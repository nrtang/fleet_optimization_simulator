"""
Vehicle module for the fleet optimization simulator.

Defines the Vehicle class which represents an individual electric autonomous vehicle
with state tracking and behavior modeling.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class VehicleStatus(Enum):
    """Vehicle operational status"""
    IDLE = "idle"
    DRIVING_TO_PICKUP = "driving_to_pickup"
    DRIVING_WITH_PASSENGER = "driving_with_passenger"
    DRIVING_TO_DEPOT = "driving_to_depot"
    REPOSITIONING = "repositioning"
    CHARGING = "charging"
    QUEUED_FOR_CHARGING = "queued_for_charging"


@dataclass
class VehicleSpecs:
    """Technical specifications for a vehicle"""
    battery_capacity_kwh: float  # Total battery capacity in kWh
    range_km: float  # Maximum range on full charge
    charging_rate_kw: float  # Maximum charging rate in kW
    passenger_capacity: int  # Number of passengers
    efficiency_kwh_per_km: float  # Energy consumption per km

    def __post_init__(self):
        """Validate specifications"""
        if self.battery_capacity_kwh <= 0:
            raise ValueError("Battery capacity must be positive")
        if self.range_km <= 0:
            raise ValueError("Range must be positive")
        if self.charging_rate_kw <= 0:
            raise ValueError("Charging rate must be positive")


class Vehicle:
    """
    Represents an electric autonomous vehicle in the fleet.

    Tracks vehicle state including location, battery level, passenger status,
    and operational state. Provides methods for state updates based on actions.
    """

    def __init__(
        self,
        vehicle_id: str,
        initial_location: Tuple[float, float],
        specs: VehicleSpecs,
        initial_battery_soc: float = 1.0
    ):
        """
        Initialize a vehicle.

        Args:
            vehicle_id: Unique identifier for the vehicle
            initial_location: (latitude, longitude) tuple
            specs: VehicleSpecs object with technical specifications
            initial_battery_soc: Initial battery state of charge (0-1)
        """
        self.id = vehicle_id
        self.location = initial_location
        self.specs = specs
        self.battery_soc = initial_battery_soc  # State of charge (0-1)
        self.status = VehicleStatus.IDLE
        self.current_passenger_id: Optional[str] = None
        self.assigned_request_id: Optional[str] = None
        self.assigned_depot_id: Optional[str] = None

        # Tracking metrics
        self.total_distance_km = 0.0
        self.total_distance_with_passenger_km = 0.0
        self.total_distance_empty_km = 0.0
        self.total_energy_consumed_kwh = 0.0
        self.total_charging_time_minutes = 0.0
        self.total_idle_time_minutes = 0.0
        self.completed_trips = 0

    @property
    def battery_kwh(self) -> float:
        """Current battery level in kWh"""
        return self.battery_soc * self.specs.battery_capacity_kwh

    @property
    def remaining_range_km(self) -> float:
        """Estimated remaining range in km"""
        return self.battery_kwh / self.specs.efficiency_kwh_per_km

    @property
    def is_available(self) -> bool:
        """Check if vehicle is available for assignment"""
        return self.status == VehicleStatus.IDLE and self.current_passenger_id is None

    def can_complete_trip(self, distance_km: float, buffer_factor: float = 1.2) -> bool:
        """
        Check if vehicle has enough battery to complete a trip.

        Args:
            distance_km: Distance of the trip in km
            buffer_factor: Safety buffer multiplier (default 1.2 = 20% buffer)

        Returns:
            True if vehicle can complete the trip
        """
        required_range = distance_km * buffer_factor
        return self.remaining_range_km >= required_range

    def drive(self, distance_km: float, with_passenger: bool = False) -> float:
        """
        Update vehicle state after driving a distance.

        Args:
            distance_km: Distance driven in km
            with_passenger: Whether the vehicle had a passenger

        Returns:
            Energy consumed in kWh
        """
        # Calculate energy consumption with passenger load factor
        load_factor = 1.1 if with_passenger else 1.0
        energy_consumed = distance_km * self.specs.efficiency_kwh_per_km * load_factor

        # Update battery
        self.battery_soc -= energy_consumed / self.specs.battery_capacity_kwh
        self.battery_soc = max(0.0, self.battery_soc)  # Ensure non-negative

        # Update tracking metrics
        self.total_distance_km += distance_km
        self.total_energy_consumed_kwh += energy_consumed
        if with_passenger:
            self.total_distance_with_passenger_km += distance_km
        else:
            self.total_distance_empty_km += distance_km

        return energy_consumed

    def charge(self, duration_minutes: float, charging_power_kw: float) -> float:
        """
        Update vehicle state after charging.

        Args:
            duration_minutes: Charging duration in minutes
            charging_power_kw: Actual charging power delivered in kW

        Returns:
            Energy added in kWh
        """
        # Apply charging curve - charging slows down as battery fills
        # Using a simple model: full speed until 80%, then taper to 20% at 100%
        if self.battery_soc < 0.8:
            effective_power = charging_power_kw
        else:
            # Linear taper from 100% power at 80% SOC to 20% power at 100% SOC
            taper_factor = 1.0 - 0.8 * ((self.battery_soc - 0.8) / 0.2)
            effective_power = charging_power_kw * taper_factor

        # Limit by vehicle's maximum charging rate
        effective_power = min(effective_power, self.specs.charging_rate_kw)

        # Calculate energy added
        energy_added_kwh = (effective_power * duration_minutes) / 60.0

        # Update battery SOC
        new_soc = self.battery_soc + (energy_added_kwh / self.specs.battery_capacity_kwh)
        self.battery_soc = min(1.0, new_soc)  # Cap at 100%

        # Update tracking
        self.total_charging_time_minutes += duration_minutes

        return energy_added_kwh

    def update_location(self, new_location: Tuple[float, float]):
        """Update vehicle location"""
        self.location = new_location

    def assign_passenger(self, passenger_id: str):
        """Assign a passenger to the vehicle"""
        self.current_passenger_id = passenger_id
        self.status = VehicleStatus.DRIVING_WITH_PASSENGER

    def drop_off_passenger(self):
        """Complete passenger drop-off"""
        self.current_passenger_id = None
        self.completed_trips += 1
        self.status = VehicleStatus.IDLE

    def get_state_dict(self) -> dict:
        """
        Get current vehicle state as a dictionary.

        Returns:
            Dictionary with current vehicle state
        """
        return {
            'id': self.id,
            'location': self.location,
            'battery_soc': self.battery_soc,
            'battery_kwh': self.battery_kwh,
            'remaining_range_km': self.remaining_range_km,
            'status': self.status.value,
            'current_passenger_id': self.current_passenger_id,
            'assigned_request_id': self.assigned_request_id,
            'assigned_depot_id': self.assigned_depot_id,
            'is_available': self.is_available,
            'total_distance_km': self.total_distance_km,
            'total_distance_with_passenger_km': self.total_distance_with_passenger_km,
            'total_distance_empty_km': self.total_distance_empty_km,
            'total_energy_consumed_kwh': self.total_energy_consumed_kwh,
            'completed_trips': self.completed_trips,
        }

    def __repr__(self) -> str:
        return (f"Vehicle({self.id}, location={self.location}, "
                f"battery={self.battery_soc:.1%}, status={self.status.value})")
