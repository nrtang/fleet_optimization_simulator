"""
Quick test script to verify ML models work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all models can be imported."""
    print("=" * 80)
    print("Testing ML Model Imports")
    print("=" * 80)

    try:
        from fleet_optimizer.models import (
            NeuralNetworkModel,
            DQNModel,
            EnsembleModel,
            AdaptiveEnsembleModel
        )
        print("✓ All models imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\n" + "=" * 80)
    print("Testing Model Instantiation")
    print("=" * 80)

    from fleet_optimizer.models import (
        NeuralNetworkModel,
        DQNModel,
        EnsembleModel
    )

    success = True

    # Test Neural Network Model
    try:
        nn_config = {
            'model_path': 'trained_models/neural_network_model.pt',
            'low_battery_threshold': 0.25,
            'max_pickup_distance_km': 15.0
        }
        nn_model = NeuralNetworkModel(nn_config)
        print("✓ Neural Network Model instantiated successfully")
    except Exception as e:
        print(f"✗ Neural Network Model failed: {e}")
        success = False

    # Test DQN Model
    try:
        dqn_config = {
            'model_path': 'trained_models/dqn_model.pt',
            'training_mode': False,
            'epsilon': 0.1
        }
        dqn_model = DQNModel(dqn_config)
        print("✓ DQN Model instantiated successfully")
    except Exception as e:
        print(f"✗ DQN Model failed: {e}")
        success = False

    # Test Ensemble Model
    try:
        ensemble_config = {
            'aggregation_strategy': 'voting',
            'use_greedy': True,
            'use_neural_network': False,
            'use_dqn': False
        }
        ensemble_model = EnsembleModel(ensemble_config)
        print("✓ Ensemble Model instantiated successfully")
    except Exception as e:
        print(f"✗ Ensemble Model failed: {e}")
        success = False

    return success


def test_feature_extraction():
    """Test that feature extraction works."""
    print("\n" + "=" * 80)
    print("Testing Feature Extraction")
    print("=" * 80)

    try:
        from fleet_optimizer.utils.feature_extraction import StateFeatureExtractor
        from fleet_optimizer.core.fleet import Fleet
        from fleet_optimizer.core.depot import Depot
        from fleet_optimizer.utils.distance import haversine_distance

        # Create feature extractor
        extractor = StateFeatureExtractor()

        # Create mock state
        fleet = Fleet(num_vehicles=10, vehicle_specs={
            'battery_capacity_kwh': 75.0,
            'range_km': 400.0,
            'charging_rate_kw': 150.0,
            'passenger_capacity': 4,
            'efficiency_kwh_per_km': 0.19
        })

        depot = Depot(
            depot_id='test_depot',
            latitude=37.75,
            longitude=-122.45,
            num_fast_chargers=10,
            num_slow_chargers=5,
            fast_charger_power_kw=150.0,
            slow_charger_power_kw=50.0
        )

        state = {
            'current_time': 0,
            'fleet': fleet,
            'depots': {'test_depot': depot},
            'active_requests': {},
            'distance_func': haversine_distance
        }

        # Extract features
        features = extractor.extract_state_features(state)

        print(f"✓ Feature extraction successful")
        print(f"  - Extracted {len(features)} features")
        print(f"  - Feature shape: {features.shape}")
        print(f"  - Feature range: [{features.min():.3f}, {features.max():.3f}]")

        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test that model configs can be loaded."""
    print("\n" + "=" * 80)
    print("Testing Config Loading")
    print("=" * 80)

    from fleet_optimizer.config.config_loader import load_config
    from fleet_optimizer.cli import create_optimization_model_from_config

    configs = [
        'configs/neural_network.yaml',
        'configs/dqn.yaml',
        'configs/ensemble.yaml'
    ]

    success = True

    for config_path in configs:
        try:
            # Load config
            config = load_config(config_path)

            # Try to create model (will warn about missing trained weights)
            model = create_optimization_model_from_config(config)

            print(f"✓ Config loaded: {config_path}")
            print(f"  - Model type: {config.get_model_params().get('type')}")
        except Exception as e:
            print(f"✗ Config failed: {config_path}")
            print(f"  - Error: {e}")
            success = False

    return success


def main():
    """Run all tests."""
    print("\n")
    print("█" * 80)
    print("  ML MODELS TEST SUITE")
    print("█" * 80)

    all_success = True

    # Run tests
    all_success &= test_imports()
    all_success &= test_model_instantiation()
    all_success &= test_feature_extraction()
    all_success &= test_config_loading()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if all_success:
        print("✓ All tests passed!")
        print("\nYou can now use the ML models with:")
        print("  - python -m fleet_optimizer simulate configs/neural_network.yaml")
        print("  - python -m fleet_optimizer simulate configs/dqn.yaml")
        print("  - python -m fleet_optimizer simulate configs/ensemble.yaml")
        print("\nNote: Models will use random weights until trained.")
        print("To train models, see docs/ML_MODELS.md")
        return 0
    else:
        print("✗ Some tests failed")
        print("\nPlease check the errors above and fix any issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
