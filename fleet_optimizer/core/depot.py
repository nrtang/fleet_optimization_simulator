"""
Depot module for charging infrastructure management.

Defines the Depot class which represents a charging depot with multiple charging
slots, queue management, and dynamic pricing.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import deque
import numpy as np


@dataclass
class ChargingSlot:
    """Represents a single charging slot at a depot"""
    slot_id: str
    charging_power_kw: float  # Charging power in kW
    occupied: bool = False
    vehicle_id: Optional[str] = None
    charge_start_time: Optional[float] = None


@dataclass
class ElectricityPricing:
    """Electricity pricing structure"""
    base_rate_per_kwh: float  # Base rate in $/kWh
    peak_hours: List[int]  # Hours considered peak (e.g., [9, 10, 11, ..., 17])
    peak_multiplier: float  # Multiplier for peak hours
    demand_charge_per_kw: float = 0.0  # Demand charge in $/kW (optional)

    def get_price(self, hour: int) -> float:
        """Get electricity price for a given hour"""
        if hour in self.peak_hours:
            return self.base_rate_per_kwh * self.peak_multiplier
        return self.base_rate_per_kwh


class Depot:
    """
    Represents a charging depot with multiple charging slots.

    Manages charging queue, slot allocation, power constraints, and pricing.
    """

    def __init__(
        self,
        depot_id: str,
        location: tuple,
        num_fast_chargers: int,
        num_slow_chargers: int,
        fast_charger_power_kw: float = 150.0,
        slow_charger_power_kw: float = 50.0,
        max_power_capacity_kw: Optional[float] = None,
        pricing: Optional[ElectricityPricing] = None,
    ):
        """
        Initialize a charging depot.

        Args:
            depot_id: Unique identifier for the depot
            location: (latitude, longitude) tuple
            num_fast_chargers: Number of fast charging slots
            num_slow_chargers: Number of slow charging slots
            fast_charger_power_kw: Power rating of fast chargers
            slow_charger_power_kw: Power rating of slow chargers
            max_power_capacity_kw: Maximum total power capacity (grid constraint)
            pricing: ElectricityPricing object
        """
        self.id = depot_id
        self.location = location
        self.fast_charger_power_kw = fast_charger_power_kw
        self.slow_charger_power_kw = slow_charger_power_kw

        # Create charging slots
        self.slots: List[ChargingSlot] = []

        # Add fast chargers
        for i in range(num_fast_chargers):
            self.slots.append(
                ChargingSlot(
                    slot_id=f"{depot_id}_fast_{i}",
                    charging_power_kw=fast_charger_power_kw
                )
            )

        # Add slow chargers
        for i in range(num_slow_chargers):
            self.slots.append(
                ChargingSlot(
                    slot_id=f"{depot_id}_slow_{i}",
                    charging_power_kw=slow_charger_power_kw
                )
            )

        self.max_power_capacity_kw = max_power_capacity_kw or float('inf')
        self.pricing = pricing or ElectricityPricing(
            base_rate_per_kwh=0.12,
            peak_hours=list(range(9, 18)),  # 9 AM - 5 PM
            peak_multiplier=1.5
        )

        # Queue management (FIFO)
        self.queue: deque = deque()

        # Tracking metrics
        self.total_energy_dispensed_kwh = 0.0
        self.total_charging_sessions = 0
        self.total_revenue = 0.0
        self.queue_wait_times: List[float] = []

    @property
    def total_slots(self) -> int:
        """Total number of charging slots"""
        return len(self.slots)

    @property
    def available_slots(self) -> int:
        """Number of available (unoccupied) charging slots"""
        return sum(1 for slot in self.slots if not slot.occupied)

    @property
    def occupied_slots(self) -> int:
        """Number of currently occupied charging slots"""
        return sum(1 for slot in self.slots if slot.occupied)

    @property
    def current_power_usage_kw(self) -> float:
        """Current total power being drawn"""
        return sum(slot.charging_power_kw for slot in self.slots if slot.occupied)

    @property
    def queue_length(self) -> int:
        """Number of vehicles in queue"""
        return len(self.queue)

    @property
    def utilization(self) -> float:
        """Charging slot utilization (0-1)"""
        return self.occupied_slots / self.total_slots if self.total_slots > 0 else 0.0

    def has_available_slot(self) -> bool:
        """Check if there's an available charging slot"""
        return self.available_slots > 0

    def can_accept_vehicle(self, vehicle_charging_rate_kw: float) -> bool:
        """
        Check if depot can accept a vehicle given power constraints.

        Args:
            vehicle_charging_rate_kw: Vehicle's charging rate requirement

        Returns:
            True if vehicle can be accepted
        """
        if not self.has_available_slot():
            return False

        # Check if adding this vehicle would exceed power capacity
        projected_usage = self.current_power_usage_kw + vehicle_charging_rate_kw
        return projected_usage <= self.max_power_capacity_kw

    def get_available_slot(self, prefer_fast: bool = True) -> Optional[ChargingSlot]:
        """
        Get an available charging slot.

        Args:
            prefer_fast: Prefer fast chargers if available

        Returns:
            Available ChargingSlot or None
        """
        available = [slot for slot in self.slots if not slot.occupied]
        if not available:
            return None

        if prefer_fast:
            # Sort by charging power (descending)
            available.sort(key=lambda s: s.charging_power_kw, reverse=True)

        return available[0]

    def assign_vehicle_to_slot(
        self,
        vehicle_id: str,
        current_time: float,
        prefer_fast: bool = True
    ) -> Optional[ChargingSlot]:
        """
        Assign a vehicle to an available charging slot.

        Args:
            vehicle_id: ID of the vehicle to assign
            current_time: Current simulation time
            prefer_fast: Prefer fast chargers

        Returns:
            Assigned ChargingSlot or None if no slots available
        """
        slot = self.get_available_slot(prefer_fast)
        if slot is None:
            return None

        slot.occupied = True
        slot.vehicle_id = vehicle_id
        slot.charge_start_time = current_time
        self.total_charging_sessions += 1

        return slot

    def release_vehicle_from_slot(self, vehicle_id: str) -> Optional[ChargingSlot]:
        """
        Release a vehicle from its charging slot.

        Args:
            vehicle_id: ID of the vehicle to release

        Returns:
            Released ChargingSlot or None if vehicle not found
        """
        for slot in self.slots:
            if slot.vehicle_id == vehicle_id:
                slot.occupied = False
                slot.vehicle_id = None
                slot.charge_start_time = None
                return slot
        return None

    def add_to_queue(self, vehicle_id: str, arrival_time: float):
        """
        Add a vehicle to the charging queue.

        Args:
            vehicle_id: ID of the vehicle
            arrival_time: Time when vehicle arrived at queue
        """
        self.queue.append((vehicle_id, arrival_time))

    def remove_from_queue(self, vehicle_id: str) -> bool:
        """
        Remove a vehicle from the queue.

        Args:
            vehicle_id: ID of the vehicle to remove

        Returns:
            True if vehicle was in queue and removed
        """
        for i, (vid, _) in enumerate(self.queue):
            if vid == vehicle_id:
                del self.queue[i]
                return True
        return False

    def process_queue(self, current_time: float) -> List[tuple]:
        """
        Process queue and assign available slots.

        Args:
            current_time: Current simulation time

        Returns:
            List of (vehicle_id, slot) tuples for newly assigned vehicles
        """
        assigned = []
        while self.queue and self.has_available_slot():
            vehicle_id, arrival_time = self.queue.popleft()
            slot = self.assign_vehicle_to_slot(vehicle_id, current_time)
            if slot:
                wait_time = current_time - arrival_time
                self.queue_wait_times.append(wait_time)
                assigned.append((vehicle_id, slot))
            else:
                # Put back in queue if assignment failed
                self.queue.appendleft((vehicle_id, arrival_time))
                break

        return assigned

    def calculate_charging_cost(
        self,
        energy_kwh: float,
        hour: int
    ) -> float:
        """
        Calculate cost of charging.

        Args:
            energy_kwh: Energy delivered in kWh
            hour: Hour of day (0-23)

        Returns:
            Cost in dollars
        """
        price_per_kwh = self.pricing.get_price(hour)
        cost = energy_kwh * price_per_kwh
        self.total_revenue += cost
        self.total_energy_dispensed_kwh += energy_kwh
        return cost

    def get_state_dict(self) -> dict:
        """Get current depot state as a dictionary"""
        return {
            'id': self.id,
            'location': self.location,
            'total_slots': self.total_slots,
            'available_slots': self.available_slots,
            'occupied_slots': self.occupied_slots,
            'queue_length': self.queue_length,
            'utilization': self.utilization,
            'current_power_usage_kw': self.current_power_usage_kw,
            'max_power_capacity_kw': self.max_power_capacity_kw,
            'total_energy_dispensed_kwh': self.total_energy_dispensed_kwh,
            'total_charging_sessions': self.total_charging_sessions,
        }

    def __repr__(self) -> str:
        return (f"Depot({self.id}, slots={self.occupied_slots}/{self.total_slots}, "
                f"queue={self.queue_length})")
