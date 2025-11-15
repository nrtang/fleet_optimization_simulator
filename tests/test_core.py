"""
Unit tests for core simulation components.
"""

import pytest
import numpy as np
from fleet_optimizer.core.vehicle import Vehicle, VehicleSpecs, VehicleStatus
from fleet_optimizer.core.fleet import Fleet
from fleet_optimizer.core.depot import Depot, ElectricityPricing
from fleet_optimizer.core.ride_request import RideRequest, RequestStatus
from fleet_optimizer.core.demand_generator import DemandGenerator, DemandZone


class TestVehicle:
    """Test Vehicle class"""

    @pytest.fixture
    def vehicle_specs(self):
        """Create vehicle specifications"""
        return VehicleSpecs(
            battery_capacity_kwh=75.0,
            range_km=400.0,
            charging_rate_kw=150.0,
            passenger_capacity=4,
            efficiency_kwh_per_km=0.19
        )

    @pytest.fixture
    def vehicle(self, vehicle_specs):
        """Create a test vehicle"""
        return Vehicle(
            vehicle_id="TEST_001",
            initial_location=(37.75, -122.45),
            specs=vehicle_specs,
            initial_battery_soc=0.8
        )

    def test_vehicle_initialization(self, vehicle, vehicle_specs):
        """Test vehicle is initialized correctly"""
        assert vehicle.id == "TEST_001"
        assert vehicle.battery_soc == 0.8
        assert vehicle.status == VehicleStatus.IDLE
        assert vehicle.is_available is True
        assert vehicle.specs == vehicle_specs

    def test_battery_calculations(self, vehicle):
        """Test battery-related calculations"""
        assert vehicle.battery_kwh == 60.0  # 75 * 0.8
        expected_range = 60.0 / 0.19
        assert abs(vehicle.remaining_range_km - expected_range) < 0.01

    def test_drive_reduces_battery(self, vehicle):
        """Test driving reduces battery"""
        initial_soc = vehicle.battery_soc
        vehicle.drive(10.0, with_passenger=False)
        assert vehicle.battery_soc < initial_soc
        assert vehicle.total_distance_km == 10.0
        assert vehicle.total_distance_empty_km == 10.0

    def test_drive_with_passenger(self, vehicle):
        """Test driving with passenger uses more energy"""
        vehicle1 = Vehicle("V1", (37.75, -122.45), vehicle.specs, 1.0)
        vehicle2 = Vehicle("V2", (37.75, -122.45), vehicle.specs, 1.0)

        energy1 = vehicle1.drive(10.0, with_passenger=False)
        energy2 = vehicle2.drive(10.0, with_passenger=True)

        assert energy2 > energy1
        assert vehicle1.total_distance_with_passenger_km == 0.0
        assert vehicle2.total_distance_with_passenger_km == 10.0

    def test_charging_increases_battery(self, vehicle):
        """Test charging increases battery"""
        vehicle.battery_soc = 0.3
        initial_soc = vehicle.battery_soc

        energy_added = vehicle.charge(60.0, 150.0)  # 60 minutes at 150 kW

        assert vehicle.battery_soc > initial_soc
        assert energy_added > 0
        assert vehicle.total_charging_time_minutes == 60.0

    def test_can_complete_trip(self, vehicle):
        """Test trip feasibility check"""
        assert vehicle.can_complete_trip(50.0) is True
        assert vehicle.can_complete_trip(5000.0) is False

    def test_passenger_assignment(self, vehicle):
        """Test passenger assignment and drop-off"""
        assert vehicle.current_passenger_id is None

        vehicle.assign_passenger("PASS_001")
        assert vehicle.current_passenger_id == "PASS_001"
        assert vehicle.status == VehicleStatus.DRIVING_WITH_PASSENGER

        vehicle.drop_off_passenger()
        assert vehicle.current_passenger_id is None
        assert vehicle.completed_trips == 1


class TestFleet:
    """Test Fleet class"""

    @pytest.fixture
    def fleet(self):
        """Create a test fleet"""
        specs = VehicleSpecs(75.0, 400.0, 150.0, 4, 0.19)
        locations = [(37.75 + i*0.01, -122.45) for i in range(10)]
        return Fleet.create_homogeneous_fleet(10, specs, locations)

    def test_fleet_size(self, fleet):
        """Test fleet size"""
        assert fleet.size == 10

    def test_get_available_vehicles(self, fleet):
        """Test getting available vehicles"""
        available = fleet.get_available_vehicles()
        assert len(available) == 10

        # Mark one as busy
        vehicle = list(fleet.vehicles.values())[0]
        vehicle.status = VehicleStatus.DRIVING_TO_PICKUP
        available = fleet.get_available_vehicles()
        assert len(available) == 9

    def test_get_vehicles_needing_charge(self, fleet):
        """Test getting vehicles needing charge"""
        # Set some vehicles to low battery
        vehicles = list(fleet.vehicles.values())
        vehicles[0].battery_soc = 0.2
        vehicles[1].battery_soc = 0.15

        needing_charge = fleet.get_vehicles_needing_charge(threshold=0.3)
        assert len(needing_charge) == 2


class TestDepot:
    """Test Depot class"""

    @pytest.fixture
    def depot(self):
        """Create a test depot"""
        pricing = ElectricityPricing(0.12, list(range(9, 18)), 1.5)
        return Depot(
            "DEPOT_1",
            (37.75, -122.45),
            num_fast_chargers=5,
            num_slow_chargers=3,
            pricing=pricing
        )

    def test_depot_initialization(self, depot):
        """Test depot initialization"""
        assert depot.id == "DEPOT_1"
        assert depot.total_slots == 8
        assert depot.available_slots == 8
        assert depot.occupied_slots == 0

    def test_slot_assignment(self, depot):
        """Test vehicle slot assignment"""
        slot = depot.assign_vehicle_to_slot("VEH_001", 0.0)
        assert slot is not None
        assert slot.occupied is True
        assert slot.vehicle_id == "VEH_001"
        assert depot.occupied_slots == 1
        assert depot.available_slots == 7

    def test_slot_release(self, depot):
        """Test slot release"""
        depot.assign_vehicle_to_slot("VEH_001", 0.0)
        released = depot.release_vehicle_from_slot("VEH_001")

        assert released is not None
        assert released.occupied is False
        assert depot.occupied_slots == 0

    def test_queue_management(self, depot):
        """Test charging queue"""
        # Fill all slots
        for i in range(8):
            depot.assign_vehicle_to_slot(f"VEH_{i:03d}", 0.0)

        assert depot.has_available_slot() is False

        # Add to queue
        depot.add_to_queue("VEH_100", 10.0)
        assert depot.queue_length == 1

        # Release a slot and process queue
        depot.release_vehicle_from_slot("VEH_000")
        assigned = depot.process_queue(20.0)

        assert len(assigned) == 1
        assert assigned[0][0] == "VEH_100"
        assert depot.queue_length == 0

    def test_pricing(self, depot):
        """Test electricity pricing"""
        # Off-peak (hour 6)
        price_off_peak = depot.pricing.get_price(6)
        assert price_off_peak == 0.12

        # Peak (hour 12)
        price_peak = depot.pricing.get_price(12)
        assert price_peak == 0.18  # 0.12 * 1.5


class TestRideRequest:
    """Test RideRequest class"""

    @pytest.fixture
    def request(self):
        """Create a test ride request"""
        return RideRequest(
            request_id="REQ_001",
            pickup_location=(37.75, -122.45),
            dropoff_location=(37.76, -122.44),
            request_time=10.0
        )

    def test_request_initialization(self, request):
        """Test request initialization"""
        assert request.request_id == "REQ_001"
        assert request.status == RequestStatus.PENDING
        assert request.is_pending is True

    def test_request_lifecycle(self, request):
        """Test request state transitions"""
        # Assign vehicle
        request.assign_vehicle("VEH_001", 15.0)
        assert request.status == RequestStatus.ASSIGNED
        assert request.assigned_vehicle_id == "VEH_001"

        # Start trip
        request.start_trip(20.0)
        assert request.status == RequestStatus.IN_PROGRESS
        assert request.pickup_time == 20.0

        # Complete trip
        request.complete_trip(30.0)
        assert request.status == RequestStatus.COMPLETED
        assert request.dropoff_time == 30.0

    def test_wait_time(self, request):
        """Test wait time calculation"""
        wait = request.get_wait_time(15.0)
        assert wait == 5.0  # 15.0 - 10.0

        request.pickup_time = 20.0
        wait = request.get_wait_time(25.0)
        assert wait == 10.0  # 20.0 - 10.0

    def test_timeout(self, request):
        """Test request timeout"""
        request.max_wait_time = 5.0
        assert request.has_exceeded_max_wait(10.0) is False
        assert request.has_exceeded_max_wait(20.0) is True


class TestDemandGenerator:
    """Test DemandGenerator class"""

    @pytest.fixture
    def demand_generator(self):
        """Create a test demand generator"""
        service_area = ((37.7, -122.5), (37.8, -122.4))
        zone = DemandZone("downtown", (37.75, -122.45), 2.0, 100.0)

        return DemandGenerator(
            service_area_bounds=service_area,
            base_demand_per_hour=50.0,
            demand_zones=[zone],
            random_seed=42
        )

    def test_demand_generation(self, demand_generator):
        """Test demand request generation"""
        requests = demand_generator.generate_requests(
            current_time_minutes=60.0,
            time_step_minutes=10.0
        )

        # Should generate some requests
        assert len(requests) >= 0
        for request in requests:
            assert request.request_id.startswith("REQ_")
            assert request.request_time >= 60.0
            assert request.request_time < 70.0

    def test_demand_pattern(self, demand_generator):
        """Test hourly demand patterns"""
        # Peak hour (e.g., hour 17)
        rate_peak = demand_generator.calculate_demand_rate(
            17 * 60,  # 5 PM
            (37.75, -122.45)
        )

        # Off-peak hour (e.g., hour 3)
        rate_off_peak = demand_generator.calculate_demand_rate(
            3 * 60,  # 3 AM
            (37.75, -122.45)
        )

        # Peak should be higher than off-peak
        assert rate_peak > rate_off_peak


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
