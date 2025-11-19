"""
Training Script for DQN Model

This script trains a Deep Q-Network using reinforcement learning.
The agent learns by interacting with the simulation environment.

Usage:
    python scripts/train_dqn.py --episodes 1000 --config configs/default.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_optimizer.core.simulation_engine import SimulationEngine
from fleet_optimizer.models.dqn_model import DQNModel
from fleet_optimizer.config.config_loader import load_config


class DQNTrainer:
    """
    Trainer for DQN model using the simulation environment.
    """

    def __init__(self, config_path: str, model_config: dict):
        self.config = load_config(config_path)
        self.model_config = model_config
        self.model_config['training_mode'] = True

        # Initialize DQN model
        self.model = DQNModel(self.model_config)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

    def train_episode(self, episode_num: int) -> dict:
        """
        Train for one episode.

        Returns:
            Dictionary with episode statistics
        """
        # Create simulation engine with training data collection
        engine = SimulationEngine(
            config=self.config,
            model=self.model,
            collect_training_data=True
        )

        # Run simulation
        results = engine.run_simulation()

        # Extract episode statistics
        trajectory = results['training_data']['trajectory']
        total_reward = sum(step['reward'] for step in trajectory)
        episode_length = len(trajectory)

        # Train on collected experience
        if len(self.model.replay_buffer) >= self.model.batch_size:
            # Perform multiple training steps
            num_train_steps = min(episode_length, 100)
            episode_losses = []

            for _ in range(num_train_steps):
                loss = self.model.train_step()
                if loss is not None:
                    episode_losses.append(loss)

            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        else:
            avg_loss = 0.0

        # Store statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        if avg_loss > 0:
            self.losses.append(avg_loss)

        # Compute statistics
        stats = {
            'episode': episode_num,
            'reward': total_reward,
            'length': episode_length,
            'loss': avg_loss,
            'epsilon': self.model.epsilon,
            'buffer_size': len(self.model.replay_buffer),
            'avg_reward_100': np.mean(self.episode_rewards[-100:]),
            'revenue': results['metrics']['total_revenue'],
            'trips': results['metrics']['total_trips']
        }

        return stats

    def train(self, num_episodes: int, save_freq: int = 100):
        """
        Train the DQN for multiple episodes.

        Args:
            num_episodes: Number of episodes to train
            save_freq: Save checkpoint every N episodes
        """
        print(f"Starting DQN training for {num_episodes} episodes")
        print(f"Model config: {self.model_config}")
        print(f"Device: {self.model.device}")
        print("-" * 80)

        for episode in range(1, num_episodes + 1):
            # Train one episode
            stats = self.train_episode(episode)

            # Print progress
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Reward: {stats['reward']:.2f}")
            print(f"  Avg Reward (100): {stats['avg_reward_100']:.2f}")
            print(f"  Loss: {stats['loss']:.4f}")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Buffer Size: {stats['buffer_size']}")
            print(f"  Revenue: ${stats['revenue']:.2f}")
            print(f"  Trips: {stats['trips']}")
            print("-" * 80)

            # Save checkpoint periodically
            if episode % save_freq == 0:
                checkpoint_path = self.model_config['model_path'].replace(
                    '.pt', f'_episode_{episode}.pt'
                )
                self.model.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # Save final model
        self.model.save_checkpoint()
        print(f"\nTraining complete!")
        print(f"Final model saved to {self.model_config['model_path']}")

        # Print summary statistics
        self._print_summary()

    def _print_summary(self):
        """Print training summary statistics."""
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)

        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Best Reward: {np.max(self.episode_rewards):.2f}")
        print(f"Final Epsilon: {self.model.epsilon:.4f}")

        if self.losses:
            print(f"Average Loss: {np.mean(self.losses):.4f}")
            print(f"Final Loss: {self.losses[-1]:.4f}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train DQN model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to simulation config file')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--output', type=str, default='trained_models/dqn_model.pt',
                        help='Output path for trained model')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Save checkpoint every N episodes')

    # DQN hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Exploration decay rate')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--buffer-capacity', type=int, default=10000,
                        help='Replay buffer capacity')
    parser.add_argument('--target-update-freq', type=int, default=100,
                        help='Target network update frequency')

    # Decision parameters
    parser.add_argument('--low-battery-threshold', type=float, default=0.25,
                        help='Low battery threshold')
    parser.add_argument('--max-pickup-distance', type=float, default=15.0,
                        help='Maximum pickup distance (km)')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model config
    model_config = {
        'model_path': args.output,
        'training_mode': True,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'buffer_capacity': args.buffer_capacity,
        'target_update_freq': args.target_update_freq,
        'low_battery_threshold': args.low_battery_threshold,
        'max_pickup_distance_km': args.max_pickup_distance
    }

    # Create trainer
    trainer = DQNTrainer(args.config, model_config)

    # Train
    trainer.train(args.episodes, args.save_freq)


if __name__ == '__main__':
    main()
