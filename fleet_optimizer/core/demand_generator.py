"""
Demand generation module.

Provides demand generation with time-based patterns, geographic distribution,
and special events modeling.
"""

from typing import List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from .ride_request import RideRequest


@dataclass
class DemandPattern:
    """Defines hourly demand patterns"""
    hourly_multipliers: List[float]  # 24 values, one per hour

    def __post_init__(self):
        if len(self.hourly_multipliers) != 24:
            raise ValueError("Must provide 24 hourly multipliers")

    def get_multiplier(self, hour: int) -> float:
        """Get demand multiplier for a specific hour"""
        return self.hourly_multipliers[hour % 24]


@dataclass
class SpecialEvent:
    """Defines a special event that affects demand"""
    event_id: str
    location: Tuple[float, float]  # (lat, lon)
    start_time: float  # Start time in simulation time (minutes)
    duration: float  # Duration in minutes
    peak_demand_multiplier: float  # Demand surge at peak
    radius_km: float  # Affected area radius
    decay_rate: float = 0.5  # How quickly demand decays after peak (0-1)

    def is_active(self, current_time: float) -> bool:
        """Check if event is currently active"""
        return self.start_time <= current_time < (self.start_time + self.duration)

    def get_multiplier(self, current_time: float, distance_from_center_km: float) -> float:
        """
        Get demand multiplier for a location and time.

        Args:
            current_time: Current simulation time
            distance_from_center_km: Distance from event center

        Returns:
            Demand multiplier (1.0 = normal demand)
        """
        if not self.is_active(current_time):
            return 1.0

        # Spatial decay - demand decreases with distance
        if distance_from_center_km > self.radius_km:
            return 1.0

        spatial_factor = 1.0 - (distance_from_center_km / self.radius_km)

        # Temporal pattern - demand builds up and decays
        elapsed = current_time - self.start_time
        progress = elapsed / self.duration

        if progress < 0.5:
            # Building up to peak
            temporal_factor = progress * 2
        else:
            # Decaying from peak
            temporal_factor = 1.0 - ((progress - 0.5) * 2 * self.decay_rate)

        multiplier = 1.0 + (self.peak_demand_multiplier - 1.0) * spatial_factor * temporal_factor
        return max(1.0, multiplier)


@dataclass
class DemandZone:
    """Defines a geographic zone with specific demand characteristics"""
    zone_id: str
    center: Tuple[float, float]  # (lat, lon)
    radius_km: float
    base_demand_rate: float  # Requests per hour


class DemandGenerator:
    """
    Generates synthetic ride requests based on patterns, zones, and events.
    """

    def __init__(
        self,
        service_area_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        base_demand_per_hour: float = 100.0,
        demand_pattern: Optional[DemandPattern] = None,
        demand_zones: Optional[List[DemandZone]] = None,
        special_events: Optional[List[SpecialEvent]] = None,
        distance_func: Optional[Callable] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize demand generator.

        Args:
            service_area_bounds: ((min_lat, min_lon), (max_lat, max_lon))
            base_demand_per_hour: Base demand rate (requests per hour)
            demand_pattern: DemandPattern for hourly variation
            demand_zones: List of demand zones with specific characteristics
            special_events: List of special events
            distance_func: Function to calculate distance between coordinates
            random_seed: Random seed for reproducibility
        """
        self.service_area_bounds = service_area_bounds
        self.base_demand_per_hour = base_demand_per_hour
        self.demand_pattern = demand_pattern or self._create_default_pattern()
        self.demand_zones = demand_zones or []
        self.special_events = special_events or []
        self.distance_func = distance_func or self._haversine_distance

        self.rng = np.random.RandomState(random_seed)
        self.request_counter = 0

        # Cache for efficiency
        self._zone_weights = self._calculate_zone_weights()

    @staticmethod
    def _create_default_pattern() -> DemandPattern:
        """Create a default weekday demand pattern"""
        # Typical weekday pattern: low at night, peaks during commute hours
        multipliers = [
            0.3, 0.2, 0.2, 0.2, 0.3, 0.5,  # 0-5 AM: very low
            0.8, 1.3, 1.5, 1.2, 1.0, 1.0,  # 6-11 AM: morning peak
            1.1, 1.0, 1.0, 1.1, 1.2, 1.5,  # 12-5 PM: lunch and afternoon
            1.6, 1.4, 1.2, 1.0, 0.8, 0.5,  # 6-11 PM: evening peak and decline
        ]
        return DemandPattern(hourly_multipliers=multipliers)

    def _calculate_zone_weights(self) -> np.ndarray:
        """Calculate weights for zone-based sampling"""
        if not self.demand_zones:
            return np.array([1.0])
        weights = np.array([zone.base_demand_rate for zone in self.demand_zones])
        return weights / weights.sum()

    @staticmethod
    def _haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate haversine distance between two coordinates.

        Args:
            coord1: (latitude, longitude) tuple
            coord2: (latitude, longitude) tuple

        Returns:
            Distance in kilometers
        """
        lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
        lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        radius_earth_km = 6371.0

        return radius_earth_km * c

    def _sample_location_in_area(self) -> Tuple[float, float]:
        """Sample a random location within the service area"""
        (min_lat, min_lon), (max_lat, max_lon) = self.service_area_bounds

        if self.demand_zones:
            # Sample from a zone based on weights
            zone = self.rng.choice(self.demand_zones, p=self._zone_weights)

            # Sample within zone radius
            angle = self.rng.uniform(0, 2 * np.pi)
            radius = self.rng.uniform(0, zone.radius_km)

            # Approximate conversion: 1 degree ≈ 111 km at equator
            dlat = (radius * np.cos(angle)) / 111.0
            dlon = (radius * np.sin(angle)) / (111.0 * np.cos(np.radians(zone.center[0])))

            lat = zone.center[0] + dlat
            lon = zone.center[1] + dlon

            # Ensure within service area bounds
            lat = np.clip(lat, min_lat, max_lat)
            lon = np.clip(lon, min_lon, max_lon)

            return (lat, lon)
        else:
            # Uniform sampling across service area
            lat = self.rng.uniform(min_lat, max_lat)
            lon = self.rng.uniform(min_lon, max_lon)
            return (lat, lon)

    def _sample_trip_destination(self, origin: Tuple[float, float]) -> Tuple[float, float]:
        """
        Sample a trip destination given an origin.

        Uses a distance distribution: most trips are short, some are long.
        """
        # Sample trip distance from exponential distribution (most trips are short)
        mean_trip_distance_km = 5.0
        max_trip_distance_km = 30.0
        distance_km = min(
            self.rng.exponential(mean_trip_distance_km),
            max_trip_distance_km
        )

        # Random direction
        angle = self.rng.uniform(0, 2 * np.pi)

        # Convert to lat/lon offset
        dlat = (distance_km * np.cos(angle)) / 111.0
        dlon = (distance_km * np.sin(angle)) / (111.0 * np.cos(np.radians(origin[0])))

        destination = (origin[0] + dlat, origin[1] + dlon)

        # Ensure within service area
        (min_lat, min_lon), (max_lat, max_lon) = self.service_area_bounds
        destination = (
            np.clip(destination[0], min_lat, max_lat),
            np.clip(destination[1], min_lon, max_lon)
        )

        return destination

    def calculate_demand_rate(self, current_time_minutes: float, location: Tuple[float, float]) -> float:
        """
        Calculate demand rate for a specific time and location.

        Args:
            current_time_minutes: Current time in minutes since start
            location: (latitude, longitude)

        Returns:
            Demand rate in requests per hour
        """
        # Time-based multiplier
        hour = int((current_time_minutes / 60) % 24)
        time_multiplier = self.demand_pattern.get_multiplier(hour)

        # Base demand
        demand_rate = self.base_demand_per_hour * time_multiplier

        # Apply special event multipliers
        for event in self.special_events:
            distance = self.distance_func(location, event.location)
            event_multiplier = event.get_multiplier(current_time_minutes, distance)
            demand_rate *= event_multiplier

        return demand_rate

    def generate_requests(
        self,
        current_time_minutes: float,
        time_step_minutes: float
    ) -> List[RideRequest]:
        """
        Generate ride requests for a time step.

        Args:
            current_time_minutes: Current simulation time in minutes
            time_step_minutes: Duration of time step in minutes

        Returns:
            List of RideRequest objects
        """
        requests = []

        # Sample a location to estimate demand rate (simplified)
        sample_location = self._sample_location_in_area()
        demand_rate = self.calculate_demand_rate(current_time_minutes, sample_location)

        # Calculate expected number of requests in this time step
        expected_requests = (demand_rate / 60.0) * time_step_minutes

        # Sample actual number of requests from Poisson distribution
        num_requests = self.rng.poisson(expected_requests)

        # Generate individual requests
        for _ in range(num_requests):
            # Sample request time within the time step
            request_time = current_time_minutes + self.rng.uniform(0, time_step_minutes)

            # Sample pickup location
            pickup_location = self._sample_location_in_area()

            # Sample dropoff location
            dropoff_location = self._sample_trip_destination(pickup_location)

            # Create request
            request = RideRequest(
                request_id=f"REQ_{self.request_counter:06d}",
                pickup_location=pickup_location,
                dropoff_location=dropoff_location,
                request_time=request_time,
                passengers=1,  # Simplified: always 1 passenger
                max_wait_time=10.0,  # 10 minutes max wait
            )

            requests.append(request)
            self.request_counter += 1

        return requests

    def add_special_event(self, event: SpecialEvent):
        """Add a special event to the generator"""
        self.special_events.append(event)

    def add_demand_zone(self, zone: DemandZone):
        """Add a demand zone and recalculate weights"""
        self.demand_zones.append(zone)
        self._zone_weights = self._calculate_zone_weights()

    def __repr__(self) -> str:
        return (f"DemandGenerator(base_rate={self.base_demand_per_hour}/hr, "
                f"zones={len(self.demand_zones)}, events={len(self.special_events)})")
