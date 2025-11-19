"""
Training Script for Neural Network Model

This script trains a supervised learning model using imitation learning
from expert demonstrations (e.g., from the greedy model).

Usage:
    python scripts/train_neural_network.py --data training_data --epochs 50
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_optimizer.models.neural_network_model import FleetPolicyNetwork
from fleet_optimizer.utils.feature_extraction import StateFeatureExtractor


class FleetDataset(Dataset):
    """
    PyTorch dataset for fleet optimization training data.

    Converts trajectories from expert demonstrations into
    (state, vehicle, action) tuples.
    """

    # Action mapping
    ACTION_MAP = {
        'IDLE': 0,
        'PICKUP_PASSENGER': 1,
        'CHARGING': 2,
        'REPOSITION': 3
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.feature_extractor = StateFeatureExtractor()
        self.samples = []

        print(f"Loading training data from {data_dir}...")
        self._load_data()
        print(f"Loaded {len(self.samples)} training samples")

    def _load_data(self):
        """Load all trajectory files and convert to training samples."""
        trajectory_files = list(self.data_dir.glob('trajectory_*.pkl'))

        if not trajectory_files:
            raise ValueError(f"No trajectory files found in {self.data_dir}")

        for traj_file in trajectory_files:
            with open(traj_file, 'rb') as f:
                data = pickle.load(f)

            self._process_trajectory(data)

    def _process_trajectory(self, trajectory_data: dict):
        """Convert trajectory to training samples."""
        trajectory = trajectory_data['trajectory']

        for step in trajectory:
            state = step['state']
            actions = step['actions']

            # Extract state features
            state_features = self.feature_extractor.extract_state_features(state)

            # Create sample for each action
            for action in actions:
                vehicle_id = action['vehicle_id']

                # Find the vehicle
                vehicle = state['fleet'].vehicles.get(vehicle_id)
                if not vehicle or not vehicle.available:
                    continue

                # Extract vehicle features
                vehicle_features = self._extract_vehicle_features(vehicle, state)

                # Get action label
                action_type = action['action']
                action_label = self.ACTION_MAP.get(action_type, 0)

                # Store sample
                self.samples.append({
                    'state_features': state_features,
                    'vehicle_features': vehicle_features,
                    'action_label': action_label
                })

    def _extract_vehicle_features(self, vehicle, state: dict) -> np.ndarray:
        """Extract vehicle-specific features (same as in model)."""
        features = np.zeros(7, dtype=np.float32)

        features[0] = vehicle.battery_level
        features[1] = 1.0 if vehicle.available else 0.0
        features[2] = 1.0 if vehicle.is_charging else 0.0

        # Distance to nearest depot
        min_depot_dist = float('inf')
        depots = state['depots']
        distance_func = state['distance_func']

        for depot in depots.values():
            dist = distance_func(
                vehicle.current_lat, vehicle.current_lon,
                depot.latitude, depot.longitude
            )
            min_depot_dist = min(min_depot_dist, dist)

        features[3] = min(min_depot_dist / 50.0, 1.0)

        # Distance to nearest request
        min_request_dist = float('inf')
        requests = state['active_requests']

        if requests:
            for request in requests.values():
                dist = distance_func(
                    vehicle.current_lat, vehicle.current_lon,
                    request.pickup_lat, request.pickup_lon
                )
                min_request_dist = min(min_request_dist, dist)
            features[4] = min(min_request_dist / 50.0, 1.0)
        else:
            features[4] = 1.0

        # Normalized position
        features[5] = (vehicle.current_lat - 37.0) / 1.0
        features[6] = (vehicle.current_lon + 122.0) / 1.0

        return features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Combine state and vehicle features
        state_features = torch.FloatTensor(sample['state_features'])
        vehicle_features = torch.FloatTensor(sample['vehicle_features'])
        features = torch.cat([state_features, vehicle_features])

        action = torch.LongTensor([sample['action_label']])

        return features, action


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, actions in dataloader:
        features = features.to(device)
        actions = actions.to(device).squeeze()

        # Forward pass
        action_logits, values = model(features)

        # Compute loss
        loss = criterion(action_logits, actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(action_logits, 1)
        total += actions.size(0)
        correct += (predicted == actions).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, actions in dataloader:
            features = features.to(device)
            actions = actions.to(device).squeeze()

            # Forward pass
            action_logits, values = model(features)

            # Compute loss
            loss = criterion(action_logits, actions)

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(action_logits, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Neural Network model')
    parser.add_argument('--data', type=str, default='training_data',
                        help='Directory containing training data')
    parser.add_argument('--output', type=str, default='trained_models/neural_network_model.pt',
                        help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    dataset = FleetDataset(args.data)

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    print("\nInitializing model...")
    model = FleetPolicyNetwork()
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            torch.save(checkpoint, args.output)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
