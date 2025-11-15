"""
Metrics and analysis module.

Provides comprehensive metrics calculation and analysis for simulation results.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


class MetricsAnalyzer:
    """
    Analyzes simulation results and calculates performance metrics.
    """

    def __init__(self, simulation_results: dict):
        """
        Initialize metrics analyzer.

        Args:
            simulation_results: Results dictionary from simulation engine
        """
        self.results = simulation_results
        self.metrics = {}

    def calculate_all_metrics(self) -> dict:
        """
        Calculate all metrics.

        Returns:
            Dictionary with all calculated metrics
        """
        self.metrics = {
            'primary': self._calculate_primary_metrics(),
            'secondary': self._calculate_secondary_metrics(),
            'tertiary': self._calculate_tertiary_metrics(),
            'fleet': self._calculate_fleet_metrics(),
            'depot': self._calculate_depot_metrics(),
        }

        return self.metrics

    def _calculate_primary_metrics(self) -> dict:
        """Calculate primary performance metrics"""
        fleet_stats = self.results['fleet_stats']
        total_requests = self.results['total_requests']
        completed_requests = self.results['completed_requests']

        # Fleet utilization
        utilization = fleet_stats.get('utilization', 0.0)

        # Revenue per mile
        completed_request_data = self.results['requests']['completed']
        total_revenue = sum(r['fare'] for r in completed_request_data)
        total_miles = fleet_stats['total_distance_km'] * 0.621371  # Convert to miles
        revenue_per_mile = total_revenue / total_miles if total_miles > 0 else 0.0

        # Service level
        service_level = self.results['service_level']

        return {
            'fleet_utilization': utilization,
            'revenue_per_mile': revenue_per_mile,
            'service_level': service_level,
            'total_revenue': total_revenue,
            'total_miles': total_miles,
        }

    def _calculate_secondary_metrics(self) -> dict:
        """Calculate secondary performance metrics"""
        fleet_stats = self.results['fleet_stats']
        completed_request_data = self.results['requests']['completed']

        # Energy cost per mile
        total_energy_kwh = fleet_stats['total_energy_consumed_kwh']
        avg_energy_cost_per_kwh = 0.12  # Simplified, should come from depot pricing
        total_energy_cost = total_energy_kwh * avg_energy_cost_per_kwh
        total_miles = fleet_stats['total_distance_km'] * 0.621371

        energy_cost_per_mile = total_energy_cost / total_miles if total_miles > 0 else 0.0

        # Empty miles ratio
        empty_miles_ratio = fleet_stats.get('empty_miles_ratio', 0.0)

        # Average wait time
        wait_times = []
        for request in completed_request_data:
            if request['pickup_time'] is not None:
                wait_time = request['pickup_time'] - request['request_time']
                wait_times.append(wait_time)

        avg_wait_time = np.mean(wait_times) if wait_times else 0.0

        # Vehicle miles between charges (simplified)
        total_distance_km = fleet_stats['total_distance_km']
        num_vehicles = fleet_stats['total_vehicles']
        avg_distance_per_vehicle = total_distance_km / num_vehicles if num_vehicles > 0 else 0.0

        return {
            'energy_cost_per_mile': energy_cost_per_mile,
            'total_energy_cost': total_energy_cost,
            'empty_miles_ratio': empty_miles_ratio,
            'avg_wait_time_minutes': avg_wait_time,
            'avg_distance_per_vehicle_km': avg_distance_per_vehicle,
        }

    def _calculate_tertiary_metrics(self) -> dict:
        """Calculate tertiary/nice-to-have metrics"""
        depot_stats = self.results.get('depot_stats', {})
        fleet_stats = self.results['fleet_stats']
        completed_request_data = self.results['requests']['completed']

        # Depot utilization
        depot_utilizations = {}
        total_energy_dispensed = 0
        total_charging_sessions = 0

        for depot_id, depot_data in depot_stats.items():
            depot_utilizations[depot_id] = depot_data.get('utilization', 0.0)
            total_energy_dispensed += depot_data.get('total_energy_dispensed_kwh', 0.0)
            total_charging_sessions += depot_data.get('total_charging_sessions', 0)

        avg_depot_utilization = np.mean(list(depot_utilizations.values())) if depot_utilizations else 0.0

        # Revenue per vehicle per day
        total_revenue = sum(r['fare'] for r in completed_request_data)
        num_vehicles = fleet_stats['total_vehicles']
        simulation_hours = self.results['simulation_duration_hours']
        days = simulation_hours / 24.0

        revenue_per_vehicle_per_day = total_revenue / (num_vehicles * days) if num_vehicles > 0 and days > 0 else 0.0

        # Completed trips per vehicle
        total_trips = fleet_stats['total_completed_trips']
        trips_per_vehicle = total_trips / num_vehicles if num_vehicles > 0 else 0.0

        return {
            'depot_utilizations': depot_utilizations,
            'avg_depot_utilization': avg_depot_utilization,
            'total_energy_dispensed_kwh': total_energy_dispensed,
            'total_charging_sessions': total_charging_sessions,
            'revenue_per_vehicle_per_day': revenue_per_vehicle_per_day,
            'trips_per_vehicle': trips_per_vehicle,
        }

    def _calculate_fleet_metrics(self) -> dict:
        """Calculate detailed fleet metrics"""
        fleet_stats = self.results['fleet_stats']

        return {
            'total_vehicles': fleet_stats['total_vehicles'],
            'avg_battery_soc': fleet_stats['avg_battery_soc'],
            'min_battery_soc': fleet_stats['min_battery_soc'],
            'max_battery_soc': fleet_stats['max_battery_soc'],
            'total_distance_km': fleet_stats['total_distance_km'],
            'total_distance_with_passenger_km': fleet_stats['total_distance_with_passenger_km'],
            'total_distance_empty_km': fleet_stats['total_distance_empty_km'],
            'total_energy_consumed_kwh': fleet_stats['total_energy_consumed_kwh'],
            'avg_energy_per_vehicle_kwh': fleet_stats['avg_energy_per_vehicle_kwh'],
            'status_counts': fleet_stats['status_counts'],
        }

    def _calculate_depot_metrics(self) -> dict:
        """Calculate detailed depot metrics"""
        depot_stats = self.results.get('depot_stats', {})

        depot_details = {}
        for depot_id, depot_data in depot_stats.items():
            depot_details[depot_id] = {
                'total_slots': depot_data.get('total_slots', 0),
                'utilization': depot_data.get('utilization', 0.0),
                'total_energy_dispensed_kwh': depot_data.get('total_energy_dispensed_kwh', 0.0),
                'total_charging_sessions': depot_data.get('total_charging_sessions', 0),
            }

        return depot_details

    def generate_summary(self) -> str:
        """
        Generate a text summary of key metrics.

        Returns:
            Formatted string with metric summary
        """
        if not self.metrics:
            self.calculate_all_metrics()

        primary = self.metrics['primary']
        secondary = self.metrics['secondary']
        tertiary = self.metrics['tertiary']

        summary = f"""
=== SIMULATION RESULTS SUMMARY ===

Duration: {self.results['simulation_duration_hours']:.1f} hours

--- Primary Metrics ---
Fleet Utilization: {primary['fleet_utilization']:.1%}
Revenue per Mile: ${primary['revenue_per_mile']:.2f}
Service Level: {primary['service_level']:.1%}
Total Revenue: ${primary['total_revenue']:.2f}

--- Secondary Metrics ---
Energy Cost per Mile: ${secondary['energy_cost_per_mile']:.2f}
Empty Miles Ratio: {secondary['empty_miles_ratio']:.1%}
Average Wait Time: {secondary['avg_wait_time_minutes']:.1f} minutes

--- Tertiary Metrics ---
Revenue per Vehicle per Day: ${tertiary['revenue_per_vehicle_per_day']:.2f}
Trips per Vehicle: {tertiary['trips_per_vehicle']:.1f}
Average Depot Utilization: {tertiary['avg_depot_utilization']:.1%}

--- Request Statistics ---
Total Requests: {self.results['total_requests']}
Completed Requests: {self.results['completed_requests']}
Cancelled Requests: {self.results['cancelled_requests']}

--- Fleet Statistics ---
Total Distance: {self.metrics['fleet']['total_distance_km']:.1f} km
Distance with Passenger: {self.metrics['fleet']['total_distance_with_passenger_km']:.1f} km
Energy Consumed: {self.metrics['fleet']['total_energy_consumed_kwh']:.1f} kWh
        """

        return summary.strip()

    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Export metrics to pandas DataFrames.

        Returns:
            Dictionary of DataFrames with different metric categories
        """
        if not self.metrics:
            self.calculate_all_metrics()

        dataframes = {}

        # Primary metrics
        dataframes['primary'] = pd.DataFrame([self.metrics['primary']])

        # Secondary metrics
        dataframes['secondary'] = pd.DataFrame([self.metrics['secondary']])

        # Tertiary metrics (excluding nested dicts)
        tertiary_flat = {
            k: v for k, v in self.metrics['tertiary'].items()
            if not isinstance(v, dict)
        }
        dataframes['tertiary'] = pd.DataFrame([tertiary_flat])

        # Fleet metrics
        fleet_flat = {
            k: v for k, v in self.metrics['fleet'].items()
            if not isinstance(v, dict)
        }
        dataframes['fleet'] = pd.DataFrame([fleet_flat])

        # Depot metrics
        if self.metrics['depot']:
            depot_records = []
            for depot_id, depot_data in self.metrics['depot'].items():
                record = {'depot_id': depot_id, **depot_data}
                depot_records.append(record)
            dataframes['depot'] = pd.DataFrame(depot_records)

        # Request details
        completed = self.results['requests']['completed']
        if completed:
            dataframes['completed_requests'] = pd.DataFrame(completed)

        cancelled = self.results['requests']['cancelled']
        if cancelled:
            dataframes['cancelled_requests'] = pd.DataFrame(cancelled)

        return dataframes

    def export_to_csv(self, output_dir: str):
        """
        Export metrics to CSV files.

        Args:
            output_dir: Directory to save CSV files
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        dataframes = self.export_to_dataframe()

        for name, df in dataframes.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)

    def export_to_json(self, filepath: str):
        """
        Export metrics to JSON file.

        Args:
            filepath: Path to JSON file
        """
        import json

        if not self.metrics:
            self.calculate_all_metrics()

        with open(filepath, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'results': self.results,
            }, f, indent=2)


def compare_scenarios(
    results_list: List[dict],
    scenario_names: List[str]
) -> pd.DataFrame:
    """
    Compare metrics across multiple scenarios.

    Args:
        results_list: List of simulation results dictionaries
        scenario_names: List of scenario names

    Returns:
        DataFrame with comparison of key metrics
    """
    comparison_data = []

    for results, name in zip(results_list, scenario_names):
        analyzer = MetricsAnalyzer(results)
        metrics = analyzer.calculate_all_metrics()

        row = {
            'scenario': name,
            'fleet_utilization': metrics['primary']['fleet_utilization'],
            'revenue_per_mile': metrics['primary']['revenue_per_mile'],
            'service_level': metrics['primary']['service_level'],
            'total_revenue': metrics['primary']['total_revenue'],
            'energy_cost_per_mile': metrics['secondary']['energy_cost_per_mile'],
            'empty_miles_ratio': metrics['secondary']['empty_miles_ratio'],
            'avg_wait_time': metrics['secondary']['avg_wait_time_minutes'],
            'revenue_per_vehicle_per_day': metrics['tertiary']['revenue_per_vehicle_per_day'],
        }

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)
