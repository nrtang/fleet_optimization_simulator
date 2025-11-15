"""
Script to generate ML training data by running multiple simulations.

This script runs many simulation episodes with different configurations
and random seeds to create a diverse training dataset for machine learning models.
"""

import argparse
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_optimizer.cli import run_simulation
from fleet_optimizer.utils.config import Config


def generate_training_data(
    config_paths: list,
    episodes_per_config: int,
    output_dir: str,
    format: str = 'pickle',
    verbose: bool = False
):
    """
    Generate training data by running multiple simulations.

    Args:
        config_paths: List of paths to configuration files
        episodes_per_config: Number of episodes to run per configuration
        output_dir: Directory to save training data
        format: 'json' or 'pickle'
        verbose: Print detailed progress
    """
    os.makedirs(output_dir, exist_ok=True)

    total_episodes = len(config_paths) * episodes_per_config
    episode_counter = 0

    print(f"Generating training data...")
    print(f"  Configurations: {len(config_paths)}")
    print(f"  Episodes per config: {episodes_per_config}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Output directory: {output_dir}")
    print(f"  Format: {format}")
    print()

    # Progress bar
    pbar = tqdm(total=total_episodes, desc="Generating episodes")

    for config_path in config_paths:
        config_name = Path(config_path).stem

        for episode in range(episodes_per_config):
            # Create a modified config with different random seed
            config = Config.from_yaml(config_path)
            config_dict = config.to_dict()

            # Change random seed for each episode
            config_dict['simulation']['random_seed'] = 42 + episode_counter

            # Save modified config temporarily
            temp_config_path = f"/tmp/temp_config_{episode_counter}.yaml"
            import yaml
            with open(temp_config_path, 'w') as f:
                yaml.dump(config_dict, f)

            # Run simulation with data collection
            try:
                results, metrics = run_simulation(
                    temp_config_path,
                    verbose=verbose,
                    collect_training_data=True
                )

                # Save training data
                extension = 'pkl' if format == 'pickle' else 'json'
                output_file = os.path.join(
                    output_dir,
                    f"{config_name}_episode_{episode:04d}.{extension}"
                )

                if format == 'pickle':
                    import pickle
                    with open(output_file, 'wb') as f:
                        pickle.dump(results['training_data'], f)
                else:  # json
                    # Use simplified JSON format
                    with open(output_file, 'w') as f:
                        json.dump({
                            'metadata': {
                                'config_name': config_name,
                                'episode': episode,
                                'seed': 42 + episode_counter,
                                **results['training_data'].get('metadata', {})
                            },
                            'num_timesteps': results['training_data']['num_timesteps'],
                            'final_service_level': results['service_level'],
                            'final_revenue': metrics['primary']['total_revenue']
                        }, f, indent=2)

                # Clean up temp config
                os.remove(temp_config_path)

                pbar.set_postfix({
                    'config': config_name,
                    'episode': episode,
                    'service_level': f"{results['service_level']:.1%}"
                })
                pbar.update(1)

                episode_counter += 1

            except Exception as e:
                print(f"\nError in episode {episode_counter}: {e}")
                continue

    pbar.close()

    print(f"\n✓ Generated {episode_counter} episodes")
    print(f"  Saved to: {output_dir}")

    # Print summary statistics
    print("\nDataset Summary:")
    total_size = sum(os.path.getsize(os.path.join(output_dir, f))
                     for f in os.listdir(output_dir))
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Files: {len(os.listdir(output_dir))}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ML training data from fleet simulations"
    )

    parser.add_argument(
        '--configs',
        nargs='+',
        default=['configs/default.yaml', 'configs/rule_based.yaml', 'configs/event_driven.yaml'],
        help='Configuration files to use'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes per configuration'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='training_data',
        help='Output directory'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'pickle'],
        default='pickle',
        help='Output format'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    args = parser.parse_args()

    generate_training_data(
        config_paths=args.configs,
        episodes_per_config=args.episodes,
        output_dir=args.output,
        format=args.format,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
