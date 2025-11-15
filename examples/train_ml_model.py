"""
Example: Train an ML model using collected simulation data.

This demonstrates how to load training data and use it for:
1. Imitation learning (behavioral cloning)
2. Reinforcement learning (offline RL)
"""

import pickle
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_optimizer.utils.feature_extraction import StateFeatureExtractor


def load_training_data(data_dir: str):
    """
    Load all training data files from a directory.

    Args:
        data_dir: Directory containing .pkl files

    Returns:
        List of episode dictionaries
    """
    episodes = []

    for pkl_file in Path(data_dir).glob("*.pkl"):
        with open(pkl_file, 'rb') as f:
            episode_data = pickle.load(f)
            episodes.append(episode_data)

    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


def extract_features_and_labels(episodes):
    """
    Extract features and labels for supervised learning.

    Args:
        episodes: List of episode dictionaries

    Returns:
        X: Feature vectors (states)
        y: Action labels
        rewards: Rewards for each transition
    """
    feature_extractor = StateFeatureExtractor()

    all_states = []
    all_actions = []
    all_rewards = []

    for episode in episodes:
        trajectory = episode['trajectory']

        for step in trajectory:
            # Extract state features
            state_features = feature_extractor.extract_state_features(step['state'])
            all_states.append(state_features)

            # Extract action labels (simplified - just count action types)
            actions = step['actions']
            action_counts = np.zeros(4)  # [IDLE, PICKUP, CHARGING, REPOSITION]

            for action in actions:
                action_type = action.get('action', 'IDLE')
                if action_type == 'IDLE':
                    action_counts[0] += 1
                elif action_type == 'PICKUP_PASSENGER':
                    action_counts[1] += 1
                elif action_type == 'CHARGING':
                    action_counts[2] += 1
                elif action_type == 'REPOSITION':
                    action_counts[3] += 1

            all_actions.append(action_counts)
            all_rewards.append(step['reward'])

    X = np.array(all_states)
    y = np.array(all_actions)
    rewards = np.array(all_rewards)

    print(f"\nDataset shape:")
    print(f"  States (X): {X.shape}")
    print(f"  Actions (y): {y.shape}")
    print(f"  Rewards: {rewards.shape}")

    return X, y, rewards


def train_imitation_learning_model(X, y):
    """
    Train a simple imitation learning model.

    Uses sklearn RandomForestRegressor to predict action distributions.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    print("\n=== Training Imitation Learning Model ===")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\nTest MSE: {mse:.4f}")

    # Feature importances
    feature_extractor = StateFeatureExtractor()
    feature_names = feature_extractor.get_feature_names()
    importances = model.feature_importances_

    print("\nTop 10 Most Important Features:")
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    return model


def create_rl_dataset(episodes):
    """
    Create dataset for offline reinforcement learning.

    Returns state-action-reward-next_state tuples.
    """
    feature_extractor = StateFeatureExtractor()

    transitions = []

    for episode in episodes:
        trajectory = episode['trajectory']

        for i in range(len(trajectory) - 1):
            current_step = trajectory[i]
            next_step = trajectory[i + 1]

            # Extract features
            state = feature_extractor.extract_state_features(current_step['state'])
            next_state = feature_extractor.extract_state_features(next_step['state'])

            # Simplified action representation
            actions = current_step['actions']
            action_vector = np.zeros(4)
            for action in actions:
                action_type = action.get('action', 'IDLE')
                if action_type == 'IDLE':
                    action_vector[0] += 1
                elif action_type == 'PICKUP_PASSENGER':
                    action_vector[1] += 1
                elif action_type == 'CHARGING':
                    action_vector[2] += 1
                elif action_type == 'REPOSITION':
                    action_vector[3] += 1

            reward = current_step['reward']
            done = (i == len(trajectory) - 2)

            transitions.append({
                'state': state,
                'action': action_vector,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

    print(f"\nCreated {len(transitions)} transitions for RL")
    return transitions


def analyze_dataset(episodes):
    """Analyze the collected dataset"""
    print("\n=== Dataset Analysis ===")

    total_steps = sum(len(ep['trajectory']) for ep in episodes)
    total_reward = sum(sum(step['reward'] for step in ep['trajectory'])
                      for ep in episodes)

    avg_steps_per_episode = total_steps / len(episodes)
    avg_reward_per_episode = total_reward / len(episodes)

    print(f"Total episodes: {len(episodes)}")
    print(f"Total timesteps: {total_steps}")
    print(f"Avg steps per episode: {avg_steps_per_episode:.1f}")
    print(f"Avg reward per episode: {avg_reward_per_episode:.2f}")

    # Reward distribution
    all_rewards = [step['reward'] for ep in episodes for step in ep['trajectory']]
    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(all_rewards):.2f}")
    print(f"  Std: {np.std(all_rewards):.2f}")
    print(f"  Min: {np.min(all_rewards):.2f}")
    print(f"  Max: {np.max(all_rewards):.2f}")


def main():
    """Main example workflow"""
    print("=" * 60)
    print("ML Model Training Example")
    print("=" * 60)

    # Check if training data exists
    data_dir = "training_data"
    if not Path(data_dir).exists():
        print(f"\nError: Training data directory '{data_dir}' not found!")
        print("\nFirst generate training data by running:")
        print("  python scripts/generate_training_data.py --episodes 10")
        return

    # Load data
    print("\n1. Loading training data...")
    episodes = load_training_data(data_dir)

    if not episodes:
        print("No training data found!")
        return

    # Analyze dataset
    print("\n2. Analyzing dataset...")
    analyze_dataset(episodes)

    # Extract features for supervised learning
    print("\n3. Extracting features...")
    X, y, rewards = extract_features_and_labels(episodes)

    # Train imitation learning model
    print("\n4. Training model...")
    model = train_imitation_learning_model(X, y)

    # Create RL dataset
    print("\n5. Creating RL dataset...")
    rl_transitions = create_rl_dataset(episodes)

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Use 'model' to make predictions on new states")
    print("  - Use 'rl_transitions' to train an RL agent")
    print("  - Integrate trained model into fleet_optimizer.models")


if __name__ == '__main__':
    main()
