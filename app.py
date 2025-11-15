"""
Streamlit UI for Fleet Optimization Simulator

A simple web interface to run simulations, compare models, and visualize results.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import tempfile
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_optimizer.cli import run_simulation
from fleet_optimizer.utils.metrics import MetricsAnalyzer, compare_scenarios


# Set page configuration
st.set_page_config(
    page_title="Fleet Optimization Simulator",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []


def create_temp_config(params):
    """Create a temporary config file from parameters"""
    config = {
        'simulation': {
            'duration_hours': params['duration_hours'],
            'optimization_mode': params['optimization_mode'],
            'time_step_minutes': params['time_step_minutes'],
            'travel_speed_kmh': params['travel_speed_kmh'],
            'random_seed': params['random_seed'],
        },
        'service_area': {
            'name': 'San Francisco',
            'min_lat': 37.7,
            'max_lat': 37.8,
            'min_lon': -122.5,
            'max_lon': -122.4,
        },
        'fleet': {
            'num_vehicles': params['num_vehicles'],
            'initial_battery_soc': params['initial_battery_soc'],
            'vehicle_specs': {
                'battery_capacity_kwh': 75.0,
                'range_km': 400.0,
                'charging_rate_kw': 150.0,
                'passenger_capacity': 4,
                'efficiency_kwh_per_km': 0.19,
            },
        },
        'depots': [
            {
                'depot_id': 'depot_1',
                'location': [37.75, -122.45],
                'num_fast_chargers': params['num_fast_chargers'],
                'num_slow_chargers': params['num_slow_chargers'],
                'fast_charger_power_kw': 150.0,
                'slow_charger_power_kw': 50.0,
                'max_power_capacity_kw': 3500.0,
            },
        ],
        'demand': {
            'base_demand_per_hour': params['demand_per_hour'],
            'use_default_pattern': True,
            'demand_zones': [
                {
                    'zone_id': 'downtown',
                    'center': [37.75, -122.45],
                    'radius_km': 2.0,
                    'base_demand_rate': 100.0,
                },
            ],
            'special_events': [],
        },
        'pricing': {
            'base_rate_per_kwh': 0.12,
            'peak_hours': list(range(9, 18)),
            'peak_multiplier': 1.5,
        },
        'model': {
            'type': params['model_type'],
            'config': params['model_config'],
        },
        'output': {
            'save_results': False,
            'output_dir': 'results',
            'save_csv': False,
            'save_json': False,
            'generate_plots': False,
        },
    }

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()

    return temp_file.name


def plot_comparison_metrics(comparison_df):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metrics = [
        ('fleet_utilization', 'Fleet Utilization (%)', 100),
        ('service_level', 'Service Level (%)', 100),
        ('revenue_per_mile', 'Revenue per Mile ($)', 1),
        ('energy_cost_per_mile', 'Energy Cost per Mile ($)', 1),
        ('empty_miles_ratio', 'Empty Miles Ratio (%)', 100),
        ('avg_wait_time', 'Avg Wait Time (min)', 1),
    ]

    for idx, (metric, title, multiplier) in enumerate(metrics):
        ax = axes[idx]
        values = comparison_df[metric].values * multiplier
        scenarios = comparison_df['scenario'].values

        bars = ax.bar(range(len(scenarios)), values, alpha=0.7)

        # Color bars
        if 'ratio' in metric or 'cost' in metric or 'wait' in metric:
            # Lower is better
            colors = ['#2ecc71' if v < values.mean() else '#e74c3c' for v in values]
        else:
            # Higher is better
            colors = ['#2ecc71' if v > values.mean() else '#e74c3c' for v in values]

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_title(title)

        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    return fig


# Main UI
st.title("🚗 Fleet Optimization Simulator")
st.markdown("Compare different optimization algorithms and configurations")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Optimization Model",
    ["rule_based", "greedy"],
    help="Rule-based: Simple heuristics | Greedy: Profit optimization"
)

# Optimization mode
optimization_mode = st.sidebar.selectbox(
    "Optimization Mode",
    ["fixed_interval", "event_driven"],
    help="Fixed interval: Decide every N minutes | Event-driven: React immediately to state changes"
)

# Model-specific parameters
st.sidebar.subheader("Model Parameters")

if model_type == "rule_based":
    low_battery = st.sidebar.slider("Low Battery Threshold", 0.1, 0.5, 0.3, 0.05)
    max_pickup_dist = st.sidebar.slider("Max Pickup Distance (km)", 5.0, 20.0, 10.0, 1.0)
    model_config = {
        'low_battery_threshold': low_battery,
        'max_pickup_distance_km': max_pickup_dist,
    }
else:  # greedy
    low_battery = st.sidebar.slider("Low Battery Threshold", 0.1, 0.5, 0.25, 0.05)
    max_pickup_dist = st.sidebar.slider("Max Pickup Distance (km)", 5.0, 25.0, 15.0, 1.0)
    enable_reposition = st.sidebar.checkbox("Enable Repositioning", False)
    model_config = {
        'low_battery_threshold': low_battery,
        'max_pickup_distance_km': max_pickup_dist,
        'enable_repositioning': enable_reposition,
    }

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")

duration_hours = st.sidebar.slider("Duration (hours)", 1, 24, 8, 1)
num_vehicles = st.sidebar.slider("Number of Vehicles", 10, 200, 50, 10)
demand_per_hour = st.sidebar.slider("Demand (requests/hour)", 50, 500, 200, 50)

# Advanced parameters (collapsible)
with st.sidebar.expander("Advanced Parameters"):
    time_step_minutes = st.slider("Time Step (minutes)", 0.5, 5.0, 1.0, 0.5)
    travel_speed_kmh = st.slider("Travel Speed (km/h)", 20, 50, 30, 5)
    initial_battery_soc = st.slider("Initial Battery SOC", 0.5, 1.0, 0.8, 0.05)
    num_fast_chargers = st.slider("Fast Chargers", 5, 50, 20, 5)
    num_slow_chargers = st.slider("Slow Chargers", 5, 50, 10, 5)
    random_seed = st.number_input("Random Seed", 0, 9999, 42, 1)

# Create scenario name
scenario_name = st.sidebar.text_input(
    "Scenario Name",
    f"{model_type}_{optimization_mode}_{num_vehicles}v"
)

# Run button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    # Create parameters dictionary
    params = {
        'model_type': model_type,
        'model_config': model_config,
        'optimization_mode': optimization_mode,
        'duration_hours': duration_hours,
        'time_step_minutes': time_step_minutes,
        'travel_speed_kmh': travel_speed_kmh,
        'num_vehicles': num_vehicles,
        'initial_battery_soc': initial_battery_soc,
        'demand_per_hour': demand_per_hour,
        'num_fast_chargers': num_fast_chargers,
        'num_slow_chargers': num_slow_chargers,
        'random_seed': random_seed,
    }

    # Create temp config
    config_path = create_temp_config(params)

    # Run simulation
    with st.spinner(f"Running simulation '{scenario_name}'..."):
        try:
            results, metrics = run_simulation(config_path, verbose=False)

            # Store results
            st.session_state.results.append({
                'name': scenario_name,
                'results': results,
                'metrics': metrics,
                'params': params,
            })
            st.session_state.scenarios.append(scenario_name)

            # Clean up temp file
            os.unlink(config_path)

            st.success(f"✅ Simulation '{scenario_name}' completed!")

        except Exception as e:
            st.error(f"❌ Error running simulation: {e}")
            os.unlink(config_path)

# Clear button
if st.sidebar.button("🗑️ Clear All Results"):
    st.session_state.results = []
    st.session_state.scenarios = []
    st.rerun()

# Main content area
if len(st.session_state.results) == 0:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to get started!")

    st.markdown("""
    ### Quick Start Guide

    1. **Select a Model**: Choose between rule-based (simple) or greedy (profit-optimizing)
    2. **Choose Optimization Mode**: Fixed interval or event-driven
    3. **Configure Parameters**: Adjust fleet size, demand, duration, etc.
    4. **Run Simulation**: Click the button to run
    5. **Compare Results**: Run multiple scenarios and compare

    ### Model Descriptions

    **Rule-Based Model**
    - Simple if/then logic
    - Charges when battery < threshold
    - Picks up nearest request
    - Fast and predictable

    **Greedy Model**
    - Profit-based optimization
    - Considers revenue vs costs
    - Global vehicle-request matching
    - More sophisticated decisions

    **Optimization Modes**
    - **Fixed Interval**: Decisions every N minutes (e.g., every 1 minute)
    - **Event-Driven**: React immediately when vehicles become available or requests arrive
    """)

else:
    # Display results
    st.header("Results")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Comparison", "📈 Detailed Metrics", "🔍 Individual Results"])

    with tab1:
        st.subheader("Scenario Comparison")

        # Create comparison dataframe
        comparison_data = []
        for scenario in st.session_state.results:
            metrics = scenario['metrics']
            row = {
                'scenario': scenario['name'],
                'fleet_utilization': metrics['primary']['fleet_utilization'],
                'revenue_per_mile': metrics['primary']['revenue_per_mile'],
                'service_level': metrics['primary']['service_level'],
                'total_revenue': metrics['primary']['total_revenue'],
                'energy_cost_per_mile': metrics['secondary']['energy_cost_per_mile'],
                'empty_miles_ratio': metrics['secondary']['empty_miles_ratio'],
                'avg_wait_time': metrics['secondary']['avg_wait_time_minutes'],
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Display comparison plot
        if len(comparison_df) > 1:
            st.pyplot(plot_comparison_metrics(comparison_df))
        else:
            st.info("Run at least 2 scenarios to see comparison plots")

        # Display comparison table
        st.subheader("Metrics Table")

        # Format for display
        display_df = comparison_df.copy()
        display_df['fleet_utilization'] = (display_df['fleet_utilization'] * 100).round(1).astype(str) + '%'
        display_df['service_level'] = (display_df['service_level'] * 100).round(1).astype(str) + '%'
        display_df['revenue_per_mile'] = '$' + display_df['revenue_per_mile'].round(2).astype(str)
        display_df['total_revenue'] = '$' + display_df['total_revenue'].round(2).astype(str)
        display_df['energy_cost_per_mile'] = '$' + display_df['energy_cost_per_mile'].round(2).astype(str)
        display_df['empty_miles_ratio'] = (display_df['empty_miles_ratio'] * 100).round(1).astype(str) + '%'
        display_df['avg_wait_time'] = display_df['avg_wait_time'].round(1).astype(str) + ' min'

        st.dataframe(display_df, use_container_width=True)

        # Download button
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            "📥 Download Comparison CSV",
            csv,
            "comparison.csv",
            "text/csv",
        )

    with tab2:
        st.subheader("Detailed Metrics")

        selected_scenario = st.selectbox(
            "Select Scenario",
            [r['name'] for r in st.session_state.results]
        )

        # Find selected result
        result = next(r for r in st.session_state.results if r['name'] == selected_scenario)
        metrics = result['metrics']

        # Display in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fleet Utilization", f"{metrics['primary']['fleet_utilization']:.1%}")
            st.metric("Service Level", f"{metrics['primary']['service_level']:.1%}")
            st.metric("Total Revenue", f"${metrics['primary']['total_revenue']:.2f}")

        with col2:
            st.metric("Revenue per Mile", f"${metrics['primary']['revenue_per_mile']:.2f}")
            st.metric("Energy Cost per Mile", f"${metrics['secondary']['energy_cost_per_mile']:.2f}")
            st.metric("Empty Miles Ratio", f"{metrics['secondary']['empty_miles_ratio']:.1%}")

        with col3:
            st.metric("Avg Wait Time", f"{metrics['secondary']['avg_wait_time_minutes']:.1f} min")
            st.metric("Revenue per Vehicle/Day", f"${metrics['tertiary']['revenue_per_vehicle_per_day']:.2f}")
            st.metric("Trips per Vehicle", f"{metrics['tertiary']['trips_per_vehicle']:.1f}")

        # Fleet details
        st.subheader("Fleet Statistics")
        fleet_stats = metrics['fleet']

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Battery Statistics**")
            st.write(f"- Average: {fleet_stats['avg_battery_soc']:.1%}")
            st.write(f"- Minimum: {fleet_stats['min_battery_soc']:.1%}")
            st.write(f"- Maximum: {fleet_stats['max_battery_soc']:.1%}")

        with col2:
            st.write("**Distance Statistics**")
            st.write(f"- Total: {fleet_stats['total_distance_km']:.1f} km")
            st.write(f"- With Passenger: {fleet_stats['total_distance_with_passenger_km']:.1f} km")
            st.write(f"- Empty: {fleet_stats['total_distance_empty_km']:.1f} km")

        st.write("**Energy Consumption**")
        st.write(f"- Total: {fleet_stats['total_energy_consumed_kwh']:.1f} kWh")
        st.write(f"- Average per Vehicle: {fleet_stats['avg_energy_per_vehicle_kwh']:.1f} kWh")

    with tab3:
        st.subheader("Individual Results")

        selected_scenario = st.selectbox(
            "Select Scenario ",
            [r['name'] for r in st.session_state.results],
            key='individual_select'
        )

        result = next(r for r in st.session_state.results if r['name'] == selected_scenario)

        # Configuration
        st.write("**Configuration:**")
        st.json(result['params'])

        # Raw results
        with st.expander("View Raw Results"):
            st.json(result['results'])
