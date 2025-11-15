"""
Simple example simulation.

Demonstrates how to set up and run a basic simulation programmatically.
"""

import numpy as np
from fleet_optimizer.core.vehicle import Vehicle, VehicleSpecs
from fleet_optimizer.core.fleet import Fleet
from fleet_optimizer.core.depot import Depot, ElectricityPricing
from fleet_optimizer.core.demand_generator import DemandGenerator, DemandZone
from fleet_optimizer.core.simulation_engine import SimulationEngine
from fleet_optimizer.models import RuleBasedModel
from fleet_optimizer.utils.metrics import MetricsAnalyzer


def main():
    """Run a simple simulation"""
    print("Setting up simulation...")

    # Define service area (San Francisco area)
    service_area_bounds = ((37.7, -122.5), (37.8, -122.4))

    # Create vehicle specifications
    specs = VehicleSpecs(
        battery_capacity_kwh=75.0,
        range_km=400.0,
        charging_rate_kw=150.0,
        passenger_capacity=4,
        efficiency_kwh_per_km=0.19
    )

    # Create fleet of 50 vehicles
    print("Creating fleet of 50 vehicles...")
    rng = np.random.RandomState(42)
    initial_locations = [
        (
            rng.uniform(37.7, 37.8),
            rng.uniform(-122.5, -122.4)
        )
        for _ in range(50)
    ]

    fleet = Fleet.create_homogeneous_fleet(
        num_vehicles=50,
        specs=specs,
        initial_locations=initial_locations,
        initial_battery_soc=0.8
    )

    # Create depot
    print("Creating charging depot...")
    pricing = ElectricityPricing(
        base_rate_per_kwh=0.12,
        peak_hours=list(range(9, 18)),
        peak_multiplier=1.5
    )

    depot = Depot(
        depot_id="main_depot",
        location=(37.75, -122.45),
        num_fast_chargers=10,
        num_slow_chargers=5,
        pricing=pricing
    )

    # Create demand generator
    print("Creating demand generator...")
    downtown_zone = DemandZone(
        zone_id="downtown",
        center=(37.75, -122.45),
        radius_km=2.0,
        base_demand_rate=100.0
    )

    demand_generator = DemandGenerator(
        service_area_bounds=service_area_bounds,
        base_demand_per_hour=100.0,
        demand_zones=[downtown_zone],
        random_seed=42
    )

    # Create optimization model
    print("Creating rule-based optimization model...")
    model = RuleBasedModel(config={
        'low_battery_threshold': 0.3,
        'max_pickup_distance_km': 10.0
    })

    # Create simulation engine
    print("Creating simulation engine...")
    engine = SimulationEngine(
        fleet=fleet,
        depots=[depot],
        demand_generator=demand_generator,
        optimization_model=model,
        time_step_minutes=1.0,
        simulation_duration_hours=8.0,  # Short 8-hour simulation
        travel_speed_kmh=30.0
    )

    # Run simulation
    print("\nRunning simulation (8 hours)...")
    print("-" * 50)

    def progress_callback(current_time, total_time):
        if int(current_time) % 60 == 0:  # Print every hour
            hours = current_time / 60
            print(f"Simulated {hours:.0f} hours...")

    results = engine.run(progress_callback)

    # Analyze results
    print("\nAnalyzing results...")
    analyzer = MetricsAnalyzer(results)
    metrics = analyzer.calculate_all_metrics()

    # Print summary
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(analyzer.generate_summary())

    # Print additional details
    print("\n" + "=" * 50)
    print("ADDITIONAL DETAILS")
    print("=" * 50)

    fleet_stats = metrics['fleet']
    print(f"\nFleet Performance:")
    print(f"  Total Distance: {fleet_stats['total_distance_km']:.1f} km")
    print(f"  Distance with Passenger: {fleet_stats['total_distance_with_passenger_km']:.1f} km")
    print(f"  Distance Empty: {fleet_stats['total_distance_empty_km']:.1f} km")
    print(f"  Energy Consumed: {fleet_stats['total_energy_consumed_kwh']:.1f} kWh")

    depot_metrics = metrics['depot']
    print(f"\nDepot Performance:")
    for depot_id, depot_data in depot_metrics.items():
        print(f"  {depot_id}:")
        print(f"    Utilization: {depot_data['utilization']:.1%}")
        print(f"    Energy Dispensed: {depot_data['total_energy_dispensed_kwh']:.1f} kWh")
        print(f"    Charging Sessions: {depot_data['total_charging_sessions']}")

    print("\nSimulation complete!")


if __name__ == '__main__':
    main()
