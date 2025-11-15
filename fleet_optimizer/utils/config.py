"""
Configuration management module.

Handles loading and validation of YAML configuration files.
"""

import yaml
from typing import Dict, Any, List, Tuple
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


class Config:
    """Configuration manager for simulation parameters"""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration.

        Args:
            config_dict: Dictionary with configuration parameters
        """
        self.config = config_dict
        self._validate()

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            Config object
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def _validate(self):
        """Validate configuration parameters"""
        required_sections = ['simulation', 'fleet', 'depots', 'demand']

        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required section: {section}")

        # Validate simulation parameters
        sim = self.config['simulation']
        if 'duration_hours' not in sim:
            raise ConfigurationError("Missing simulation.duration_hours")
        if 'time_step_minutes' not in sim:
            raise ConfigurationError("Missing simulation.time_step_minutes")

        # Validate fleet parameters
        fleet = self.config['fleet']
        if 'num_vehicles' not in fleet:
            raise ConfigurationError("Missing fleet.num_vehicles")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'simulation.duration_hours')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_simulation_params(self) -> Dict[str, Any]:
        """Get simulation parameters"""
        return self.config.get('simulation', {})

    def get_fleet_params(self) -> Dict[str, Any]:
        """Get fleet parameters"""
        return self.config.get('fleet', {})

    def get_depot_params(self) -> List[Dict[str, Any]]:
        """Get depot parameters"""
        return self.config.get('depots', [])

    def get_demand_params(self) -> Dict[str, Any]:
        """Get demand parameters"""
        return self.config.get('demand', {})

    def get_model_params(self) -> Dict[str, Any]:
        """Get optimization model parameters"""
        return self.config.get('model', {})

    def get_service_area_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get service area bounds.

        Returns:
            ((min_lat, min_lon), (max_lat, max_lon))
        """
        area = self.config.get('service_area', {})
        return (
            (area['min_lat'], area['min_lon']),
            (area['max_lat'], area['max_lon'])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self.config.copy()

    def save(self, filepath: str):
        """
        Save configuration to YAML file.

        Args:
            filepath: Path to save YAML file
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __repr__(self) -> str:
        return f"Config(sections={list(self.config.keys())})"


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration.

    Returns:
        Dictionary with default configuration
    """
    return {
        'simulation': {
            'duration_hours': 24,
            'time_step_minutes': 1.0,
            'travel_speed_kmh': 30.0,
            'random_seed': 42,
        },

        'service_area': {
            'name': 'San Francisco',
            'min_lat': 37.7,
            'max_lat': 37.8,
            'min_lon': -122.5,
            'max_lon': -122.4,
        },

        'fleet': {
            'num_vehicles': 100,
            'initial_battery_soc': 0.8,
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
                'num_fast_chargers': 20,
                'num_slow_chargers': 10,
                'fast_charger_power_kw': 150.0,
                'slow_charger_power_kw': 50.0,
                'max_power_capacity_kw': 3500.0,
            },
            {
                'depot_id': 'depot_2',
                'location': [37.77, -122.43],
                'num_fast_chargers': 15,
                'num_slow_chargers': 15,
                'fast_charger_power_kw': 150.0,
                'slow_charger_power_kw': 50.0,
                'max_power_capacity_kw': 3000.0,
            },
        ],

        'demand': {
            'base_demand_per_hour': 200.0,
            'use_default_pattern': True,
            'demand_zones': [
                {
                    'zone_id': 'downtown',
                    'center': [37.75, -122.45],
                    'radius_km': 2.0,
                    'base_demand_rate': 100.0,
                },
                {
                    'zone_id': 'mission',
                    'center': [37.76, -122.42],
                    'radius_km': 1.5,
                    'base_demand_rate': 80.0,
                },
            ],
            'special_events': [],
        },

        'pricing': {
            'base_rate_per_kwh': 0.12,
            'peak_hours': [9, 10, 11, 12, 13, 14, 15, 16, 17],
            'peak_multiplier': 1.5,
        },

        'model': {
            'type': 'greedy',  # 'rule_based' or 'greedy'
            'config': {
                'low_battery_threshold': 0.25,
                'charge_target_soc': 0.9,
                'max_pickup_distance_km': 15.0,
                'enable_repositioning': False,
            },
        },

        'output': {
            'save_results': True,
            'output_dir': 'results',
            'save_csv': True,
            'save_json': True,
            'generate_plots': True,
        },
    }


def save_default_config(filepath: str):
    """
    Save default configuration to YAML file.

    Args:
        filepath: Path to save YAML file
    """
    config = create_default_config()
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
