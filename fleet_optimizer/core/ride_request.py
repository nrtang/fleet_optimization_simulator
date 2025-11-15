"""
Ride request module for demand modeling.

Defines the RideRequest class which represents a passenger ride request.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional


class RequestStatus(Enum):
    """Status of a ride request"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class RideRequest:
    """
    Represents a passenger ride request.

    Attributes:
        request_id: Unique identifier for the request
        pickup_location: (latitude, longitude) for pickup
        dropoff_location: (latitude, longitude) for dropoff
        request_time: Time when request was made (in simulation time)
        passengers: Number of passengers
        max_wait_time: Maximum acceptable wait time in minutes
        status: Current status of the request
        assigned_vehicle_id: ID of assigned vehicle (if any)
        pickup_time: Time when vehicle picked up passenger
        dropoff_time: Time when passenger was dropped off
        fare: Fare for the trip in dollars
    """
    request_id: str
    pickup_location: Tuple[float, float]
    dropoff_location: Tuple[float, float]
    request_time: float
    passengers: int = 1
    max_wait_time: float = 10.0  # minutes
    status: RequestStatus = RequestStatus.PENDING
    assigned_vehicle_id: Optional[str] = None
    pickup_time: Optional[float] = None
    dropoff_time: Optional[float] = None
    fare: float = 0.0

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending"""
        return self.status == RequestStatus.PENDING

    @property
    def is_active(self) -> bool:
        """Check if request is active (assigned or in progress)"""
        return self.status in [RequestStatus.ASSIGNED, RequestStatus.IN_PROGRESS]

    @property
    def is_completed(self) -> bool:
        """Check if request is completed"""
        return self.status == RequestStatus.COMPLETED

    def assign_vehicle(self, vehicle_id: str, current_time: float):
        """Assign a vehicle to this request"""
        self.assigned_vehicle_id = vehicle_id
        self.status = RequestStatus.ASSIGNED

    def start_trip(self, current_time: float):
        """Start the trip (pickup occurred)"""
        self.pickup_time = current_time
        self.status = RequestStatus.IN_PROGRESS

    def complete_trip(self, current_time: float):
        """Complete the trip (dropoff occurred)"""
        self.dropoff_time = current_time
        self.status = RequestStatus.COMPLETED

    def cancel(self):
        """Cancel the request"""
        self.status = RequestStatus.CANCELLED

    def timeout(self):
        """Mark request as timed out"""
        self.status = RequestStatus.TIMEOUT

    def get_wait_time(self, current_time: float) -> float:
        """
        Get current wait time in minutes.

        Args:
            current_time: Current simulation time

        Returns:
            Wait time in minutes since request
        """
        if self.pickup_time is not None:
            return self.pickup_time - self.request_time
        return current_time - self.request_time

    def has_exceeded_max_wait(self, current_time: float) -> bool:
        """Check if request has exceeded maximum wait time"""
        return self.get_wait_time(current_time) > self.max_wait_time

    def calculate_fare(self, distance_km: float, base_fare: float = 3.0,
                      per_km_rate: float = 1.5, per_minute_rate: float = 0.3) -> float:
        """
        Calculate trip fare.

        Args:
            distance_km: Trip distance in km
            base_fare: Base fare in dollars
            per_km_rate: Rate per kilometer
            per_minute_rate: Rate per minute

        Returns:
            Calculated fare in dollars
        """
        if self.pickup_time is None or self.dropoff_time is None:
            return 0.0

        trip_duration_minutes = self.dropoff_time - self.pickup_time
        self.fare = base_fare + (distance_km * per_km_rate) + (trip_duration_minutes * per_minute_rate)
        return self.fare

    def get_state_dict(self) -> dict:
        """Get current request state as a dictionary"""
        return {
            'request_id': self.request_id,
            'pickup_location': self.pickup_location,
            'dropoff_location': self.dropoff_location,
            'request_time': self.request_time,
            'passengers': self.passengers,
            'max_wait_time': self.max_wait_time,
            'status': self.status.value,
            'assigned_vehicle_id': self.assigned_vehicle_id,
            'pickup_time': self.pickup_time,
            'dropoff_time': self.dropoff_time,
            'fare': self.fare,
        }

    def __repr__(self) -> str:
        return (f"RideRequest({self.request_id}, status={self.status.value}, "
                f"vehicle={self.assigned_vehicle_id})")
