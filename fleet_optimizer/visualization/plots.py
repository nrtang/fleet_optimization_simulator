"""
Visualization module for simulation results.

Provides plotting functions for analyzing and comparing simulation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_fleet_metrics(metrics: dict, output_path: str = None):
    """
    Plot fleet-level metrics.

    Args:
        metrics: Metrics dictionary from MetricsAnalyzer
        output_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    primary = metrics['primary']
    secondary = metrics['secondary']
    fleet = metrics['fleet']

    # Plot 1: Key performance metrics
    ax = axes[0, 0]
    kpis = {
        'Fleet\nUtilization': primary['fleet_utilization'],
        'Service\nLevel': primary['service_level'],
        'Empty\nMiles\nRatio': secondary['empty_miles_ratio'],
    }
    colors = ['#2ecc71' if v > 0.7 else '#e74c3c' if v < 0.4 else '#f39c12' for v in kpis.values()]
    ax.bar(kpis.keys(), [v * 100 for v in kpis.values()], color=colors, alpha=0.7)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Key Performance Indicators')
    ax.set_ylim(0, 100)
    for i, (k, v) in enumerate(kpis.items()):
        ax.text(i, v * 100 + 2, f'{v:.1%}', ha='center', fontweight='bold')

    # Plot 2: Revenue and costs
    ax = axes[0, 1]
    revenue = primary['total_revenue']
    energy_cost = secondary['total_energy_cost']
    profit = revenue - energy_cost

    categories = ['Revenue', 'Energy\nCost', 'Profit']
    values = [revenue, energy_cost, profit]
    colors_money = ['#2ecc71', '#e74c3c', '#3498db']
    ax.bar(categories, values, color=colors_money, alpha=0.7)
    ax.set_ylabel('Dollars ($)')
    ax.set_title('Revenue and Costs')
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f'${v:.2f}', ha='center', fontweight='bold')

    # Plot 3: Distance breakdown
    ax = axes[1, 0]
    with_passenger = fleet['total_distance_with_passenger_km']
    empty = fleet['total_distance_empty_km']

    distances = [with_passenger, empty]
    labels = ['With Passenger', 'Empty']
    colors_dist = ['#3498db', '#95a5a6']
    wedges, texts, autotexts = ax.pie(
        distances,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors_dist,
        startangle=90
    )
    ax.set_title('Distance Breakdown')

    # Plot 4: Battery statistics
    ax = axes[1, 1]
    battery_data = {
        'Average': fleet['avg_battery_soc'],
        'Minimum': fleet['min_battery_soc'],
        'Maximum': fleet['max_battery_soc'],
    }
    colors_battery = ['#f39c12', '#e74c3c', '#2ecc71']
    ax.barh(list(battery_data.keys()), [v * 100 for v in battery_data.values()], color=colors_battery, alpha=0.7)
    ax.set_xlabel('State of Charge (%)')
    ax.set_title('Fleet Battery Statistics')
    ax.set_xlim(0, 100)
    for i, (k, v) in enumerate(battery_data.items()):
        ax.text(v * 100 + 2, i, f'{v:.1%}', va='center', fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(comparison_df: pd.DataFrame, output_path: str = None):
    """
    Plot comparison of multiple scenarios.

    Args:
        comparison_df: DataFrame from compare_scenarios()
        output_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    metrics_to_plot = [
        ('fleet_utilization', 'Fleet Utilization', '%', 100),
        ('service_level', 'Service Level', '%', 100),
        ('revenue_per_mile', 'Revenue per Mile', '$', 1),
        ('energy_cost_per_mile', 'Energy Cost per Mile', '$', 1),
        ('empty_miles_ratio', 'Empty Miles Ratio', '%', 100),
        ('avg_wait_time', 'Avg Wait Time', 'min', 1),
    ]

    for idx, (metric, title, unit, multiplier) in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = comparison_df[metric].values * multiplier
        scenarios = comparison_df['scenario'].values

        bars = ax.bar(range(len(scenarios)), values, alpha=0.7)

        # Color bars based on performance
        if 'ratio' in metric or 'cost' in metric or 'wait' in metric:
            # Lower is better
            colors = ['#2ecc71' if v < np.median(values) else '#e74c3c' for v in values]
        else:
            # Higher is better
            colors = ['#2ecc71' if v > np.median(values) else '#e74c3c' for v in values]

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_ylabel(unit)
        ax.set_title(title)

        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_time_series(results: dict, output_path: str = None):
    """
    Plot time series of key metrics (if time-series data available).

    Note: Current implementation doesn't track time-series data,
    this is a placeholder for future enhancement.
    """
    # This would require modifying the simulation engine to track
    # metrics over time. For now, this is a placeholder.
    pass


def create_all_plots(metrics: dict, output_dir: str):
    """
    Create all standard plots and save to directory.

    Args:
        metrics: Metrics dictionary from MetricsAnalyzer
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Fleet metrics plot
    plot_fleet_metrics(
        metrics,
        output_path=os.path.join(output_dir, 'fleet_metrics.png')
    )


def plot_depot_utilization(metrics: dict, output_path: str = None):
    """
    Plot depot utilization comparison.

    Args:
        metrics: Metrics dictionary from MetricsAnalyzer
        output_path: Optional path to save plot
    """
    depot_metrics = metrics.get('depot', {})

    if not depot_metrics:
        print("No depot data available for plotting")
        return

    depot_ids = list(depot_metrics.keys())
    utilizations = [depot_metrics[d]['utilization'] * 100 for d in depot_ids]
    energy_dispensed = [depot_metrics[d]['total_energy_dispensed_kwh'] for d in depot_ids]
    charging_sessions = [depot_metrics[d]['total_charging_sessions'] for d in depot_ids]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Utilization
    ax = axes[0]
    ax.bar(depot_ids, utilizations, alpha=0.7, color='#3498db')
    ax.set_ylabel('Utilization (%)')
    ax.set_title('Depot Utilization')
    ax.set_ylim(0, 100)
    for i, v in enumerate(utilizations):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center')

    # Energy dispensed
    ax = axes[1]
    ax.bar(depot_ids, energy_dispensed, alpha=0.7, color='#2ecc71')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Total Energy Dispensed')
    for i, v in enumerate(energy_dispensed):
        ax.text(i, v + max(energy_dispensed) * 0.02, f'{v:.0f}', ha='center')

    # Charging sessions
    ax = axes[2]
    ax.bar(depot_ids, charging_sessions, alpha=0.7, color='#f39c12')
    ax.set_ylabel('Number of Sessions')
    ax.set_title('Total Charging Sessions')
    for i, v in enumerate(charging_sessions):
        ax.text(i, v + max(charging_sessions) * 0.02, f'{int(v)}', ha='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
