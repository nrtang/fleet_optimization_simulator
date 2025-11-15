"""
Simulation engine module.

Implements discrete event simulation for the fleet optimization system.
"""

from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import heapq

from .vehicle import Vehicle, VehicleStatus
from .fleet import Fleet
from .depot import Depot
from .ride_request import RideRequest, RequestStatus
from .demand_generator import DemandGenerator


class EventType(Enum):
    """Types of simulation events"""
    VEHICLE_ARRIVES_AT_PICKUP = "vehicle_arrives_at_pickup"
    VEHICLE_ARRIVES_AT_DROPOFF = "vehicle_arrives_at_dropoff"
    VEHICLE_ARRIVES_AT_DEPOT = "vehicle_arrives_at_depot"
    VEHICLE_ARRIVES_AT_REPOSITION = "vehicle_arrives_at_reposition"
    VEHICLE_CHARGING_COMPLETE = "vehicle_charging_complete"
    REQUEST_TIMEOUT = "request_timeout"
    GENERATE_DEMAND = "generate_demand"
    OPTIMIZATION_STEP = "optimization_step"
    SIMULATION_END = "simulation_end"


@dataclass
class Event:
    """Represents a discrete simulation event"""
    time: float  # Event time in minutes
    event_type: EventType
    data: dict  # Event-specific data

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.time < other.time


class SimulationEngine:
    """
    Discrete event simulation engine for the fleet optimization system.

    Manages the simulation clock, event queue, and coordinates interactions
    between vehicles, depots, and ride requests.
    """

    def __init__(
        self,
        fleet: Fleet,
        depots: List[Depot],
        demand_generator: DemandGenerator,
        optimization_model,  # Will be defined in models module
        time_step_minutes: float = 1.0,
        simulation_duration_hours: float = 24.0,
        distance_func: Optional[Callable] = None,
        travel_speed_kmh: float = 30.0,
    ):
        """
        Initialize simulation engine.

        Args:
            fleet: Fleet object
            depots: List of Depot objects
            demand_generator: DemandGenerator object
            optimization_model: Optimization model for decision-making
            time_step_minutes: Time step for optimization decisions
            simulation_duration_hours: Total simulation duration
            distance_func: Function to calculate distance between coordinates
            travel_speed_kmh: Average travel speed
        """
        self.fleet = fleet
        self.depots = {depot.id: depot for depot in depots}
        self.demand_generator = demand_generator
        self.optimization_model = optimization_model
        self.time_step_minutes = time_step_minutes
        self.simulation_duration_minutes = simulation_duration_hours * 60
        self.distance_func = distance_func or self._haversine_distance
        self.travel_speed_kmh = travel_speed_kmh

        # Simulation state
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.active_requests: Dict[str, RideRequest] = {}
        self.completed_requests: List[RideRequest] = []
        self.cancelled_requests: List[RideRequest] = []

        # Vehicle assignments
        self.vehicle_destinations: Dict[str, Tuple[float, float]] = {}
        self.vehicle_arrival_times: Dict[str, float] = {}

        # Initialize event queue
        self._initialize_events()

    @staticmethod
    def _haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two coordinates in km"""
        lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
        lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        radius_earth_km = 6371.0

        return radius_earth_km * c

    def _initialize_events(self):
        """Initialize the event queue with recurring events"""
        # Schedule demand generation events
        time = 0.0
        while time < self.simulation_duration_minutes:
            self._add_event(Event(
                time=time,
                event_type=EventType.GENERATE_DEMAND,
                data={}
            ))
            time += self.time_step_minutes

        # Schedule optimization steps
        time = 0.0
        while time < self.simulation_duration_minutes:
            self._add_event(Event(
                time=time,
                event_type=EventType.OPTIMIZATION_STEP,
                data={}
            ))
            time += self.time_step_minutes

        # Schedule simulation end
        self._add_event(Event(
            time=self.simulation_duration_minutes,
            event_type=EventType.SIMULATION_END,
            data={}
        ))

    def _add_event(self, event: Event):
        """Add an event to the priority queue"""
        heapq.heappush(self.event_queue, event)

    def _get_next_event(self) -> Optional[Event]:
        """Get next event from the queue"""
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        return None

    def calculate_travel_time(self, distance_km: float) -> float:
        """Calculate travel time in minutes"""
        return (distance_km / self.travel_speed_kmh) * 60.0

    def send_vehicle_to_location(
        self,
        vehicle: Vehicle,
        destination: Tuple[float, float],
        event_type: EventType,
        event_data: dict
    ):
        """
        Send a vehicle to a destination and schedule arrival event.

        Args:
            vehicle: Vehicle to send
            destination: (latitude, longitude) destination
            event_type: Type of arrival event
            event_data: Data for arrival event
        """
        distance_km = self.distance_func(vehicle.location, destination)
        travel_time = self.calculate_travel_time(distance_km)

        # Update vehicle state
        with_passenger = vehicle.current_passenger_id is not None
        vehicle.drive(distance_km, with_passenger)
        vehicle.update_location(destination)

        # Track destination
        self.vehicle_destinations[vehicle.id] = destination
        arrival_time = self.current_time + travel_time
        self.vehicle_arrival_times[vehicle.id] = arrival_time

        # Schedule arrival event
        self._add_event(Event(
            time=arrival_time,
            event_type=event_type,
            data={**event_data, 'vehicle_id': vehicle.id}
        ))

    def _handle_generate_demand(self, event: Event):
        """Handle demand generation event"""
        new_requests = self.demand_generator.generate_requests(
            self.current_time,
            self.time_step_minutes
        )

        for request in new_requests:
            self.active_requests[request.request_id] = request

            # Schedule timeout event
            timeout_time = request.request_time + request.max_wait_time
            self._add_event(Event(
                time=timeout_time,
                event_type=EventType.REQUEST_TIMEOUT,
                data={'request_id': request.request_id}
            ))

    def _handle_optimization_step(self, event: Event):
        """Handle optimization decision-making"""
        # Get current state
        state = {
            'current_time': self.current_time,
            'fleet': self.fleet,
            'depots': self.depots,
            'active_requests': self.active_requests,
            'distance_func': self.distance_func,
        }

        # Get decisions from optimization model
        decisions = self.optimization_model.make_decisions(state)

        # Execute decisions
        for decision in decisions:
            self._execute_decision(decision)

    def _execute_decision(self, decision: dict):
        """Execute a decision from the optimization model"""
        action = decision['action']
        vehicle_id = decision['vehicle_id']
        vehicle = self.fleet.get_vehicle(vehicle_id)

        if vehicle is None or not vehicle.is_available:
            return

        if action == 'PICKUP_PASSENGER':
            request_id = decision['request_id']
            request = self.active_requests.get(request_id)

            if request and request.is_pending:
                # Assign vehicle to request
                request.assign_vehicle(vehicle_id, self.current_time)
                vehicle.assigned_request_id = request_id
                vehicle.status = VehicleStatus.DRIVING_TO_PICKUP

                # Send vehicle to pickup location
                self.send_vehicle_to_location(
                    vehicle,
                    request.pickup_location,
                    EventType.VEHICLE_ARRIVES_AT_PICKUP,
                    {'request_id': request_id}
                )

        elif action == 'CHARGING':
            depot_id = decision['depot_id']
            depot = self.depots.get(depot_id)

            if depot and depot.has_available_slot():
                # Assign vehicle to depot
                vehicle.assigned_depot_id = depot_id
                vehicle.status = VehicleStatus.DRIVING_TO_DEPOT

                # Send vehicle to depot
                self.send_vehicle_to_location(
                    vehicle,
                    depot.location,
                    EventType.VEHICLE_ARRIVES_AT_DEPOT,
                    {'depot_id': depot_id}
                )

        elif action == 'REPOSITION':
            target_location = decision['target_location']
            vehicle.status = VehicleStatus.REPOSITIONING

            # Send vehicle to reposition location
            self.send_vehicle_to_location(
                vehicle,
                target_location,
                EventType.VEHICLE_ARRIVES_AT_REPOSITION,
                {}
            )

        elif action == 'IDLE':
            vehicle.status = VehicleStatus.IDLE

    def _handle_vehicle_arrives_at_pickup(self, event: Event):
        """Handle vehicle arrival at pickup location"""
        vehicle_id = event.data['vehicle_id']
        request_id = event.data['request_id']

        vehicle = self.fleet.get_vehicle(vehicle_id)
        request = self.active_requests.get(request_id)

        if vehicle and request and request.status == RequestStatus.ASSIGNED:
            # Pickup passenger
            request.start_trip(self.current_time)
            vehicle.assign_passenger(request_id)

            # Send vehicle to dropoff
            self.send_vehicle_to_location(
                vehicle,
                request.dropoff_location,
                EventType.VEHICLE_ARRIVES_AT_DROPOFF,
                {'request_id': request_id}
            )

    def _handle_vehicle_arrives_at_dropoff(self, event: Event):
        """Handle vehicle arrival at dropoff location"""
        vehicle_id = event.data['vehicle_id']
        request_id = event.data['request_id']

        vehicle = self.fleet.get_vehicle(vehicle_id)
        request = self.active_requests.get(request_id)

        if vehicle and request:
            # Calculate fare
            distance_km = self.distance_func(
                request.pickup_location,
                request.dropoff_location
            )
            request.calculate_fare(distance_km)

            # Complete trip
            request.complete_trip(self.current_time)
            vehicle.drop_off_passenger()

            # Move request to completed
            del self.active_requests[request_id]
            self.completed_requests.append(request)

            # Clear assignments
            vehicle.assigned_request_id = None

    def _handle_vehicle_arrives_at_depot(self, event: Event):
        """Handle vehicle arrival at depot"""
        vehicle_id = event.data['vehicle_id']
        depot_id = event.data['depot_id']

        vehicle = self.fleet.get_vehicle(vehicle_id)
        depot = self.depots.get(depot_id)

        if vehicle and depot:
            # Try to assign to a slot
            slot = depot.assign_vehicle_to_slot(vehicle_id, self.current_time)

            if slot:
                # Start charging
                vehicle.status = VehicleStatus.CHARGING

                # Calculate charging time to reach 90% SOC
                target_soc = 0.9
                energy_needed = (target_soc - vehicle.battery_soc) * vehicle.specs.battery_capacity_kwh
                charging_time = (energy_needed / slot.charging_power_kw) * 60  # minutes

                # Schedule charging complete event
                self._add_event(Event(
                    time=self.current_time + charging_time,
                    event_type=EventType.VEHICLE_CHARGING_COMPLETE,
                    data={'vehicle_id': vehicle_id, 'depot_id': depot_id, 'slot_id': slot.slot_id}
                ))
            else:
                # Add to queue
                depot.add_to_queue(vehicle_id, self.current_time)
                vehicle.status = VehicleStatus.QUEUED_FOR_CHARGING

    def _handle_vehicle_charging_complete(self, event: Event):
        """Handle vehicle charging completion"""
        vehicle_id = event.data['vehicle_id']
        depot_id = event.data['depot_id']

        vehicle = self.fleet.get_vehicle(vehicle_id)
        depot = self.depots.get(depot_id)

        if vehicle and depot:
            # Calculate energy added and cost
            slot = depot.release_vehicle_from_slot(vehicle_id)
            if slot:
                charge_duration = self.current_time - slot.charge_start_time
                energy_added = vehicle.charge(charge_duration, slot.charging_power_kw)

                # Calculate cost
                hour = int((self.current_time / 60) % 24)
                depot.calculate_charging_cost(energy_added, hour)

            # Vehicle becomes available
            vehicle.status = VehicleStatus.IDLE
            vehicle.assigned_depot_id = None

            # Process queue
            depot.process_queue(self.current_time)

    def _handle_request_timeout(self, event: Event):
        """Handle request timeout"""
        request_id = event.data['request_id']
        request = self.active_requests.get(request_id)

        if request and request.is_pending:
            request.timeout()
            del self.active_requests[request_id]
            self.cancelled_requests.append(request)

    def step(self) -> bool:
        """
        Execute one simulation step (process next event).

        Returns:
            True if simulation should continue, False if ended
        """
        event = self._get_next_event()

        if event is None or event.event_type == EventType.SIMULATION_END:
            return False

        # Advance simulation clock
        self.current_time = event.time

        # Handle event
        if event.event_type == EventType.GENERATE_DEMAND:
            self._handle_generate_demand(event)
        elif event.event_type == EventType.OPTIMIZATION_STEP:
            self._handle_optimization_step(event)
        elif event.event_type == EventType.VEHICLE_ARRIVES_AT_PICKUP:
            self._handle_vehicle_arrives_at_pickup(event)
        elif event.event_type == EventType.VEHICLE_ARRIVES_AT_DROPOFF:
            self._handle_vehicle_arrives_at_dropoff(event)
        elif event.event_type == EventType.VEHICLE_ARRIVES_AT_DEPOT:
            self._handle_vehicle_arrives_at_depot(event)
        elif event.event_type == EventType.VEHICLE_CHARGING_COMPLETE:
            self._handle_vehicle_charging_complete(event)
        elif event.event_type == EventType.REQUEST_TIMEOUT:
            self._handle_request_timeout(event)

        return True

    def run(self, progress_callback: Optional[Callable] = None) -> dict:
        """
        Run the full simulation.

        Args:
            progress_callback: Optional callback function(current_time, total_time)

        Returns:
            Simulation results dictionary
        """
        while self.step():
            if progress_callback and self.current_time % 60 == 0:  # Update every hour
                progress_callback(self.current_time, self.simulation_duration_minutes)

        return self.get_results()

    def get_results(self) -> dict:
        """Get simulation results"""
        total_requests = len(self.completed_requests) + len(self.cancelled_requests)
        completed_requests = len(self.completed_requests)

        return {
            'simulation_duration_hours': self.simulation_duration_minutes / 60,
            'total_requests': total_requests,
            'completed_requests': completed_requests,
            'cancelled_requests': len(self.cancelled_requests),
            'service_level': completed_requests / total_requests if total_requests > 0 else 0.0,
            'fleet_stats': self.fleet.get_fleet_stats(),
            'depot_stats': {depot_id: depot.get_state_dict() for depot_id, depot in self.depots.items()},
            'requests': {
                'completed': [r.get_state_dict() for r in self.completed_requests],
                'cancelled': [r.get_state_dict() for r in self.cancelled_requests],
            }
        }
