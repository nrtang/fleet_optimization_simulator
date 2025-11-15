# EV Autonomous Fleet Optimization Simulator

A comprehensive, modular Python-based simulator for optimizing electric autonomous vehicle (EV) fleets. Compare different optimization strategies, test fleet sizing scenarios, and analyze the impact of charging infrastructure on fleet operations.

## Features

### Data Simulation Engine
- **Vehicle State Management**: Track individual vehicle states including location, battery level, passenger status
- **Realistic Demand Generation**:
  - Time-based patterns (hourly, daily variations)
  - Geographic distribution with demand zones
  - Special event modeling (concerts, sports events)
- **Charging Infrastructure**:
  - Multiple depot locations with configurable charging capacity
  - Fast and slow chargers
  - Dynamic electricity pricing (peak/off-peak)
  - Queue management
- **Discrete Event Simulation**: Accurate modeling of travel times, battery depletion, and charging curves

### Fleet Optimization Models
- **Rule-Based Model**: Simple heuristic baseline (charge when low, pick up nearest request)
- **Greedy Model**: Profit-optimizing assignments with smart charging decisions
- **Extensible Architecture**: Easy to add new optimization models (RL, MIP, etc.)

### Comprehensive Analytics
- **Primary Metrics**: Fleet utilization, revenue per mile, service level
- **Secondary Metrics**: Energy cost, empty miles ratio, wait times
- **Tertiary Metrics**: Depot utilization, revenue per vehicle, trip statistics
- **Exportable Results**: CSV, JSON, and visualization plots

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd fleet_optimization_simulator

# Install the package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### 1. Generate a Default Configuration

```bash
python -m fleet_optimizer.cli generate-config -o my_config.yaml
```

### 2. Run a Simulation

```bash
python -m fleet_optimizer.cli run configs/default.yaml
```

### 3. Compare Different Strategies

```bash
python -m fleet_optimizer.cli compare configs/rule_based.yaml configs/default.yaml -o comparison.csv
```

## Configuration

Simulations are configured using YAML files. See `configs/default.yaml` for a complete example.

### Key Configuration Sections

#### Simulation Parameters
```yaml
simulation:
  duration_hours: 24        # Simulation duration
  time_step_minutes: 1.0    # Decision-making frequency
  travel_speed_kmh: 30.0    # Average travel speed
  random_seed: 42           # For reproducibility
```

#### Service Area
```yaml
service_area:
  name: San Francisco
  min_lat: 37.7
  max_lat: 37.8
  min_lon: -122.5
  max_lon: -122.4
```

#### Fleet Configuration
```yaml
fleet:
  num_vehicles: 100
  initial_battery_soc: 0.8
  vehicle_specs:
    battery_capacity_kwh: 75.0
    range_km: 400.0
    charging_rate_kw: 150.0
    passenger_capacity: 4
    efficiency_kwh_per_km: 0.19
```

#### Depots
```yaml
depots:
  - depot_id: depot_1
    location: [37.75, -122.45]
    num_fast_chargers: 20
    num_slow_chargers: 10
    fast_charger_power_kw: 150.0
    slow_charger_power_kw: 50.0
    max_power_capacity_kw: 3500.0
```

#### Demand Generation
```yaml
demand:
  base_demand_per_hour: 200.0
  use_default_pattern: true
  demand_zones:
    - zone_id: downtown
      center: [37.75, -122.45]
      radius_km: 2.0
      base_demand_rate: 100.0

  special_events:
    - event_id: concert
      location: [37.76, -122.43]
      start_time: 1140  # 7 PM (19:00 * 60 minutes)
      duration: 180     # 3 hours
      peak_demand_multiplier: 3.0
      radius_km: 1.5
```

#### Optimization Model
```yaml
model:
  type: greedy  # or 'rule_based'
  config:
    low_battery_threshold: 0.25
    charge_target_soc: 0.9
    max_pickup_distance_km: 15.0
    enable_repositioning: false
```

## Usage Examples

### Running a Basic Simulation

```python
from fleet_optimizer.cli import run_simulation

# Run simulation and get results
results, metrics = run_simulation('configs/default.yaml', verbose=True)

# Access key metrics
print(f"Fleet Utilization: {metrics['primary']['fleet_utilization']:.1%}")
print(f"Service Level: {metrics['primary']['service_level']:.1%}")
print(f"Revenue per Mile: ${metrics['primary']['revenue_per_mile']:.2f}")
```

### Comparing Optimization Strategies

```python
from fleet_optimizer.utils.metrics import compare_scenarios
from fleet_optimizer.cli import run_simulation

# Run multiple scenarios
results_rule = run_simulation('configs/rule_based.yaml', verbose=False)[0]
results_greedy = run_simulation('configs/default.yaml', verbose=False)[0]

# Compare
comparison = compare_scenarios(
    [results_rule, results_greedy],
    ['Rule-Based', 'Greedy']
)

print(comparison)
```

### Creating Custom Optimization Models

```python
from fleet_optimizer.models.base_model import BaseOptimizationModel

class MyCustomModel(BaseOptimizationModel):
    def make_decisions(self, state: dict) -> list:
        """
        Implement your optimization logic here.

        Args:
            state: Contains fleet, depots, active_requests, etc.

        Returns:
            List of decision dictionaries
        """
        decisions = []

        # Your logic here
        for vehicle in state['fleet'].get_available_vehicles():
            decision = {
                'vehicle_id': vehicle.id,
                'action': 'IDLE'  # or PICKUP_PASSENGER, CHARGING, REPOSITION
            }
            decisions.append(decision)

        return decisions
```

## Project Structure

```
fleet_optimization_simulator/
├── fleet_optimizer/
│   ├── core/                    # Core simulation components
│   │   ├── vehicle.py           # Vehicle class and state management
│   │   ├── fleet.py             # Fleet management
│   │   ├── depot.py             # Charging infrastructure
│   │   ├── ride_request.py      # Demand representation
│   │   ├── demand_generator.py  # Synthetic demand generation
│   │   └── simulation_engine.py # Discrete event simulation
│   ├── models/                  # Optimization models
│   │   ├── base_model.py        # Abstract base class
│   │   ├── rule_based_model.py  # Simple heuristic model
│   │   └── greedy_model.py      # Greedy optimization model
│   ├── utils/                   # Utilities
│   │   ├── config.py            # Configuration management
│   │   └── metrics.py           # Metrics calculation and analysis
│   ├── visualization/           # Plotting and visualization
│   │   └── plots.py
│   └── cli.py                   # Command-line interface
├── configs/                     # Example configurations
│   ├── default.yaml
│   └── rule_based.yaml
├── tests/                       # Unit tests
├── examples/                    # Example scripts
├── requirements.txt
├── setup.py
└── README.md
```

## Metrics Reference

### Primary Metrics
- **Fleet Utilization**: Percentage of total fleet time spent with paying passengers
- **Revenue per Mile**: Total revenue divided by total miles driven
- **Service Level**: Percentage of ride requests fulfilled within time threshold

### Secondary Metrics
- **Energy Cost per Mile**: Total electricity cost divided by total miles
- **Empty Miles Ratio**: Deadhead miles divided by total miles
- **Average Wait Time**: Mean customer wait time from request to pickup

### Tertiary Metrics
- **Depot Utilization**: Charging slot occupancy over time
- **Revenue per Vehicle per Day**: Average daily revenue per vehicle
- **Trips per Vehicle**: Average number of completed trips per vehicle

## Extending the Simulator

### Adding a New Optimization Model

1. Create a new file in `fleet_optimizer/models/`
2. Inherit from `BaseOptimizationModel`
3. Implement `make_decisions()` method
4. Register in `fleet_optimizer/models/__init__.py`
5. Update config loader in `cli.py`

### Adding Special Events

```yaml
demand:
  special_events:
    - event_id: stadium_game
      location: [37.78, -122.44]
      start_time: 1080  # 6 PM
      duration: 240     # 4 hours
      peak_demand_multiplier: 4.0
      radius_km: 2.0
      decay_rate: 0.6
```

### Custom Demand Patterns

```python
from fleet_optimizer.core.demand_generator import DemandPattern

# Create custom hourly pattern
weekend_pattern = DemandPattern(
    hourly_multipliers=[
        0.5, 0.3, 0.2, 0.2, 0.3, 0.4,  # Late night / early morning
        0.6, 0.9, 1.2, 1.3, 1.4, 1.5,  # Morning / midday
        1.4, 1.3, 1.2, 1.3, 1.5, 1.8,  # Afternoon / evening
        2.0, 1.8, 1.5, 1.2, 0.9, 0.7,  # Night
    ]
)
```

## Performance Considerations

- **Fleet Size**: Tested with up to 1000 vehicles
- **Simulation Duration**: 24-hour simulations typically complete in 1-5 minutes
- **Time Step**: 1-minute steps provide good balance of accuracy and speed
- **Demand Rate**: Handles 100-500 requests/hour efficiently

## Troubleshooting

### Common Issues

**Import errors**
```bash
# Ensure package is installed
pip install -e .
```

**Simulation runs slowly**
- Increase `time_step_minutes` in configuration
- Reduce fleet size or simulation duration
- Disable unnecessary output/logging

**No results generated**
- Check `output.save_results` is `true` in config
- Verify output directory exists and is writable

## Contributing

Contributions welcome! Areas for enhancement:
- Additional optimization models (MIP, RL, etc.)
- More sophisticated charging curves
- Real-time traffic integration
- Multi-modal fleet support
- Web-based dashboard

## License

MIT License - see LICENSE file for details

## Citation

If you use this simulator in your research, please cite:

```
@software{ev_fleet_optimizer,
  title = {EV Autonomous Fleet Optimization Simulator},
  year = {2024},
  url = {https://github.com/yourusername/fleet_optimization_simulator}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.
