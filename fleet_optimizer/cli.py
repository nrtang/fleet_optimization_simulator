"""
Command-line interface for the fleet optimization simulator.

Provides commands to run simulations, compare scenarios, and analyze results.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from fleet_optimizer.core.vehicle import Vehicle, VehicleSpecs
from fleet_optimizer.core.fleet import Fleet
from fleet_optimizer.core.depot import Depot, ElectricityPricing
from fleet_optimizer.core.demand_generator import DemandGenerator, DemandZone, DemandPattern, SpecialEvent
from fleet_optimizer.core.simulation_engine import SimulationEngine
from fleet_optimizer.models import RuleBasedModel, GreedyModel
from fleet_optimizer.utils.config import Config, save_default_config
from fleet_optimizer.utils.metrics import MetricsAnalyzer


def create_fleet_from_config(config: Config) -> Fleet:
    """Create fleet from configuration"""
    fleet_params = config.get_fleet_params()
    service_area = config.get_service_area_bounds()

    num_vehicles = fleet_params['num_vehicles']
    initial_soc = fleet_params.get('initial_battery_soc', 0.8)

    # Create vehicle specs
    specs_dict = fleet_params['vehicle_specs']
    specs = VehicleSpecs(
        battery_capacity_kwh=specs_dict['battery_capacity_kwh'],
        range_km=specs_dict['range_km'],
        charging_rate_kw=specs_dict['charging_rate_kw'],
        passenger_capacity=specs_dict['passenger_capacity'],
        efficiency_kwh_per_km=specs_dict['efficiency_kwh_per_km']
    )

    # Generate random initial locations within service area
    (min_lat, min_lon), (max_lat, max_lon) = service_area
    rng = np.random.RandomState(config.get('simulation.random_seed', 42))

    initial_locations = [
        (
            rng.uniform(min_lat, max_lat),
            rng.uniform(min_lon, max_lon)
        )
        for _ in range(num_vehicles)
    ]

    # Create fleet
    fleet = Fleet.create_homogeneous_fleet(
        num_vehicles=num_vehicles,
        specs=specs,
        initial_locations=initial_locations,
        initial_battery_soc=initial_soc
    )

    return fleet


def create_depots_from_config(config: Config) -> list:
    """Create depots from configuration"""
    depot_params = config.get_depot_params()
    pricing_params = config.get('pricing', {})

    # Create pricing
    pricing = ElectricityPricing(
        base_rate_per_kwh=pricing_params.get('base_rate_per_kwh', 0.12),
        peak_hours=pricing_params.get('peak_hours', list(range(9, 18))),
        peak_multiplier=pricing_params.get('peak_multiplier', 1.5)
    )

    depots = []
    for depot_config in depot_params:
        depot = Depot(
            depot_id=depot_config['depot_id'],
            location=tuple(depot_config['location']),
            num_fast_chargers=depot_config['num_fast_chargers'],
            num_slow_chargers=depot_config['num_slow_chargers'],
            fast_charger_power_kw=depot_config.get('fast_charger_power_kw', 150.0),
            slow_charger_power_kw=depot_config.get('slow_charger_power_kw', 50.0),
            max_power_capacity_kw=depot_config.get('max_power_capacity_kw', None),
            pricing=pricing
        )
        depots.append(depot)

    return depots


def create_demand_generator_from_config(config: Config) -> DemandGenerator:
    """Create demand generator from configuration"""
    demand_params = config.get_demand_params()
    service_area = config.get_service_area_bounds()

    # Create demand zones
    demand_zones = []
    for zone_config in demand_params.get('demand_zones', []):
        zone = DemandZone(
            zone_id=zone_config['zone_id'],
            center=tuple(zone_config['center']),
            radius_km=zone_config['radius_km'],
            base_demand_rate=zone_config['base_demand_rate']
        )
        demand_zones.append(zone)

    # Create special events
    special_events = []
    for event_config in demand_params.get('special_events', []):
        event = SpecialEvent(
            event_id=event_config['event_id'],
            location=tuple(event_config['location']),
            start_time=event_config['start_time'],
            duration=event_config['duration'],
            peak_demand_multiplier=event_config['peak_demand_multiplier'],
            radius_km=event_config['radius_km'],
            decay_rate=event_config.get('decay_rate', 0.5)
        )
        special_events.append(event)

    # Create demand generator
    demand_generator = DemandGenerator(
        service_area_bounds=service_area,
        base_demand_per_hour=demand_params['base_demand_per_hour'],
        demand_zones=demand_zones,
        special_events=special_events,
        random_seed=config.get('simulation.random_seed', 42)
    )

    return demand_generator


def create_optimization_model_from_config(config: Config):
    """Create optimization model from configuration"""
    model_params = config.get_model_params()
    model_type = model_params.get('type', 'rule_based')
    model_config = model_params.get('config', {})

    if model_type == 'rule_based':
        return RuleBasedModel(model_config)
    elif model_type == 'greedy':
        return GreedyModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_simulation(config_path: str, verbose: bool = True):
    """
    Run a simulation from a configuration file.

    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print progress
    """
    # Load configuration
    if verbose:
        print(f"Loading configuration from {config_path}...")
    config = Config.from_yaml(config_path)

    # Create components
    if verbose:
        print("Creating fleet...")
    fleet = create_fleet_from_config(config)

    if verbose:
        print("Creating depots...")
    depots = create_depots_from_config(config)

    if verbose:
        print("Creating demand generator...")
    demand_generator = create_demand_generator_from_config(config)

    if verbose:
        print("Creating optimization model...")
    optimization_model = create_optimization_model_from_config(config)

    # Create simulation engine
    sim_params = config.get_simulation_params()

    if verbose:
        print("Initializing simulation engine...")
    engine = SimulationEngine(
        fleet=fleet,
        depots=depots,
        demand_generator=demand_generator,
        optimization_model=optimization_model,
        time_step_minutes=sim_params['time_step_minutes'],
        simulation_duration_hours=sim_params['duration_hours'],
        travel_speed_kmh=sim_params.get('travel_speed_kmh', 30.0)
    )

    # Run simulation
    if verbose:
        print(f"\nRunning simulation ({sim_params['duration_hours']} hours)...")
        print(f"Fleet size: {fleet.size} vehicles")
        print(f"Depots: {len(depots)}")
        print(f"Optimization model: {optimization_model}")
        print()

        # Progress bar
        pbar = tqdm(total=int(sim_params['duration_hours'] * 60), unit='min')

        def progress_callback(current_time, total_time):
            pbar.update(int(current_time - pbar.n))

        results = engine.run(progress_callback)
        pbar.close()
    else:
        results = engine.run()

    # Analyze results
    if verbose:
        print("\nAnalyzing results...")
    analyzer = MetricsAnalyzer(results)
    metrics = analyzer.calculate_all_metrics()

    # Print summary
    if verbose:
        print("\n" + analyzer.generate_summary())

    # Save results if configured
    output_config = config.get('output', {})
    if output_config.get('save_results', False):
        output_dir = output_config.get('output_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"\nSaving results to {output_dir}...")

        if output_config.get('save_csv', False):
            analyzer.export_to_csv(output_dir)
            if verbose:
                print(f"  - Saved CSV files")

        if output_config.get('save_json', False):
            json_path = os.path.join(output_dir, 'results.json')
            analyzer.export_to_json(json_path)
            if verbose:
                print(f"  - Saved {json_path}")

    return results, metrics


def generate_default_config_command(args):
    """Generate default configuration file"""
    output_path = args.output
    save_default_config(output_path)
    print(f"Default configuration saved to {output_path}")


def run_simulation_command(args):
    """Run simulation command"""
    run_simulation(args.config, verbose=not args.quiet)


def compare_scenarios_command(args):
    """Compare multiple scenarios"""
    print(f"Comparing {len(args.configs)} scenarios...")

    results_list = []
    names = []

    for config_path in args.configs:
        print(f"\nRunning {config_path}...")
        results, _ = run_simulation(config_path, verbose=False)
        results_list.append(results)

        # Extract name from config path
        name = Path(config_path).stem
        names.append(name)

    # Compare results
    from fleet_optimizer.utils.metrics import compare_scenarios
    comparison_df = compare_scenarios(results_list, names)

    print("\n=== SCENARIO COMPARISON ===\n")
    print(comparison_df.to_string(index=False))

    # Save comparison if requested
    if args.output:
        comparison_df.to_csv(args.output, index=False)
        print(f"\nComparison saved to {args.output}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="EV Autonomous Fleet Optimization Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run a simulation')
    run_parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    run_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output'
    )
    run_parser.set_defaults(func=run_simulation_command)

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple scenarios')
    compare_parser.add_argument(
        'configs',
        nargs='+',
        type=str,
        help='Paths to configuration files to compare'
    )
    compare_parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output CSV file for comparison results'
    )
    compare_parser.set_defaults(func=compare_scenarios_command)

    # Generate config command
    config_parser = subparsers.add_parser(
        'generate-config',
        help='Generate default configuration file'
    )
    config_parser.add_argument(
        '-o', '--output',
        type=str,
        default='config.yaml',
        help='Output path for configuration file (default: config.yaml)'
    )
    config_parser.set_defaults(func=generate_default_config_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
