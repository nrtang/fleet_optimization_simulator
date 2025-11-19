"""
Deep Q-Network (DQN) Model for Fleet Optimization

This model uses reinforcement learning to learn optimal policies
through trial and error, optimizing for long-term cumulative rewards.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import random
from .base_model import BaseOptimizationModel
from ..utils.feature_extraction import StateFeatureExtractor


class DQNetwork(nn.Module):
    """
    Deep Q-Network for estimating Q-values of state-action pairs.

    Architecture:
    - Dueling DQN: Separate value and advantage streams
    - Input: State features (26) + Vehicle features (7) = 33 features
    - Output: Q-values for each action
    """

    def __init__(self, state_dim: int = 26, vehicle_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        input_dim = state_dim + vehicle_dim

        # Shared feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        # Dueling architecture: Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Dueling architecture: Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 actions
        )

    def forward(self, x):
        """
        Forward pass with dueling architecture.

        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

        Args:
            x: State-vehicle feature tensor

        Returns:
            Q-values for each action
        """
        features = self.feature_layers(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages (dueling architecture)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores (state, action, reward, next_state, done) transitions.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNModel(BaseOptimizationModel):
    """
    Deep Q-Network model for fleet optimization.

    Uses reinforcement learning with experience replay and target networks.
    Learns to maximize long-term cumulative rewards.
    """

    # Action indices
    ACTION_IDLE = 0
    ACTION_PICKUP = 1
    ACTION_CHARGING = 2
    ACTION_REPOSITION = 3

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.feature_extractor = StateFeatureExtractor()

        # Model parameters
        self.model_path = config.get('model_path', 'trained_models/dqn_model.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.epsilon = config.get('epsilon', 0.1)  # Exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 100)

        # Decision-making parameters
        self.low_battery_threshold = config.get('low_battery_threshold', 0.25)
        self.max_pickup_distance_km = config.get('max_pickup_distance_km', 15.0)

        # Training mode
        self.training_mode = config.get('training_mode', False)
        self.replay_buffer = ReplayBuffer(capacity=config.get('buffer_capacity', 10000))

        # Networks
        self.policy_network = DQNetwork()
        self.target_network = DQNetwork()
        self.policy_network.to(self.device)
        self.target_network.to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        # Training state
        self.steps_done = 0
        self.last_state = None
        self.last_action = None

        # Load trained weights if available
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                self.policy_network.load_state_dict(checkpoint['policy_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                if 'optimizer' in checkpoint and self.training_mode:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                if 'steps_done' in checkpoint:
                    self.steps_done = checkpoint['steps_done']
                print(f"Loaded trained DQN model from {self.model_path}")
            else:
                self.policy_network.load_state_dict(checkpoint)
                self.target_network.load_state_dict(checkpoint)
                print(f"Loaded DQN weights from {self.model_path}")
        except FileNotFoundError:
            print(f"Warning: No trained model found at {self.model_path}")
            print("Using randomly initialized weights. Train the model first for better performance.")
        except Exception as e:
            print(f"Warning: Error loading model: {e}")
            print("Using randomly initialized weights.")

        if not self.training_mode:
            self.policy_network.eval()
            self.target_network.eval()
        else:
            self.policy_network.train()

    def _extract_vehicle_features(self, vehicle, state: dict) -> np.ndarray:
        """
        Extract features for a specific vehicle.

        Returns 7-dimensional feature vector.
        """
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

    def make_decisions(self, state: dict) -> List[dict]:
        """
        Make decisions using DQN with epsilon-greedy exploration.

        Args:
            state: Current simulation state

        Returns:
            List of decision dictionaries
        """
        decisions = []
        fleet = state['fleet']

        # Extract global state features once
        state_features = self.feature_extractor.extract_state_features(state)
        state_tensor = torch.FloatTensor(state_features).to(self.device)

        for vehicle in fleet.vehicles.values():
            if not vehicle.available:
                continue

            # Extract vehicle-specific features
            vehicle_features = self._extract_vehicle_features(vehicle, state)
            vehicle_tensor = torch.FloatTensor(vehicle_features).to(self.device)

            # Combine state and vehicle features
            combined_features = torch.cat([state_tensor, vehicle_tensor]).unsqueeze(0)

            # Epsilon-greedy action selection
            if self.training_mode and random.random() < self.epsilon:
                # Explore: random action
                action_idx = random.randint(0, 3)
            else:
                # Exploit: best action according to policy network
                with torch.no_grad():
                    q_values = self.policy_network(combined_features)
                    action_idx = torch.argmax(q_values, dim=-1).item()

            # Convert action index to decision
            decision = self._action_to_decision(vehicle, action_idx, state)

            if decision:
                decisions.append(decision)

        self.steps_done += 1

        # Decay epsilon
        if self.training_mode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return decisions

    def _action_to_decision(
        self,
        vehicle,
        action_idx: int,
        state: dict
    ) -> Optional[dict]:
        """
        Convert action index to simulator decision.

        Args:
            vehicle: Vehicle object
            action_idx: Action index from DQN
            state: Current state

        Returns:
            Decision dictionary or None
        """
        # Safety check: always charge if battery is critically low
        if vehicle.battery_level < self.low_battery_threshold:
            return self._select_charging_depot(vehicle, state)

        if action_idx == self.ACTION_IDLE:
            return {
                'vehicle_id': vehicle.vehicle_id,
                'action': 'IDLE'
            }

        elif action_idx == self.ACTION_PICKUP:
            return self._select_pickup_request(vehicle, state)

        elif action_idx == self.ACTION_CHARGING:
            return self._select_charging_depot(vehicle, state)

        elif action_idx == self.ACTION_REPOSITION:
            return self._select_reposition_location(vehicle, state)

        return None

    def _select_pickup_request(self, vehicle, state: dict) -> Optional[dict]:
        """Select nearest available request."""
        requests = state['active_requests']
        distance_func = state['distance_func']

        if not requests:
            return {'vehicle_id': vehicle.vehicle_id, 'action': 'IDLE'}

        best_request = None
        min_distance = float('inf')

        for request in requests.values():
            dist = distance_func(
                vehicle.current_lat, vehicle.current_lon,
                request.pickup_lat, request.pickup_lon
            )

            if dist < min_distance and dist <= self.max_pickup_distance_km:
                min_distance = dist
                best_request = request

        if best_request:
            return {
                'vehicle_id': vehicle.vehicle_id,
                'action': 'PICKUP_PASSENGER',
                'request_id': best_request.request_id
            }

        return {'vehicle_id': vehicle.vehicle_id, 'action': 'IDLE'}

    def _select_charging_depot(self, vehicle, state: dict) -> dict:
        """Select nearest available charging depot."""
        depots = state['depots']
        distance_func = state['distance_func']

        best_depot = None
        min_distance = float('inf')

        for depot in depots.values():
            if depot.has_available_chargers():
                dist = distance_func(
                    vehicle.current_lat, vehicle.current_lon,
                    depot.latitude, depot.longitude
                )
                if dist < min_distance:
                    min_distance = dist
                    best_depot = depot

        if best_depot:
            return {
                'vehicle_id': vehicle.vehicle_id,
                'action': 'CHARGING',
                'depot_id': best_depot.depot_id
            }

        return {'vehicle_id': vehicle.vehicle_id, 'action': 'IDLE'}

    def _select_reposition_location(self, vehicle, state: dict) -> dict:
        """Select reposition location towards demand center."""
        requests = state['active_requests']

        if not requests:
            return {'vehicle_id': vehicle.vehicle_id, 'action': 'IDLE'}

        # Move towards center of mass of requests
        avg_lat = np.mean([r.pickup_lat for r in requests.values()])
        avg_lon = np.mean([r.pickup_lon for r in requests.values()])

        return {
            'vehicle_id': vehicle.vehicle_id,
            'action': 'REPOSITION',
            'target_lat': avg_lat,
            'target_lon': avg_lon
        }

    def train_step(self):
        """
        Perform one training step using experience replay.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Use policy network to select actions
            next_actions = self.policy_network(next_states).argmax(1, keepdim=True)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Compute loss
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return loss.item()

    def save_checkpoint(self, path: str = None):
        """Save model checkpoint."""
        if path is None:
            path = self.model_path

        checkpoint = {
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }

        torch.save(checkpoint, path)
        print(f"Saved DQN checkpoint to {path}")
