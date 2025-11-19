"""
Ensemble Model for Fleet Optimization

This model combines predictions from multiple models (rule-based, greedy, neural network, DQN)
to make robust decisions through voting or weighted averaging.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
from .base_model import BaseOptimizationModel
from .rule_based_model import RuleBasedModel
from .greedy_model import GreedyModel


class EnsembleModel(BaseOptimizationModel):
    """
    Ensemble model that combines multiple optimization models.

    Supports multiple aggregation strategies:
    - voting: Majority vote among models
    - weighted: Weighted voting based on model performance
    - cascade: Use models in priority order, fallback if no decision
    """

    def __init__(self, config: dict = None):
        super().__init__(config)

        self.aggregation_strategy = config.get('aggregation_strategy', 'voting')
        self.weights = config.get('weights', None)  # For weighted strategy

        # Initialize sub-models
        self.models = []
        self.model_names = []

        # Always include rule-based as fallback
        rule_based_config = config.get('rule_based_config', {})
        self.models.append(RuleBasedModel(rule_based_config))
        self.model_names.append('rule_based')

        # Add greedy model
        if config.get('use_greedy', True):
            greedy_config = config.get('greedy_config', {})
            self.models.append(GreedyModel(greedy_config))
            self.model_names.append('greedy')

        # Add neural network model if specified
        if config.get('use_neural_network', False):
            try:
                from .neural_network_model import NeuralNetworkModel
                nn_config = config.get('neural_network_config', {})
                self.models.append(NeuralNetworkModel(nn_config))
                self.model_names.append('neural_network')
                print("Added Neural Network model to ensemble")
            except Exception as e:
                print(f"Warning: Could not load Neural Network model: {e}")

        # Add DQN model if specified
        if config.get('use_dqn', False):
            try:
                from .dqn_model import DQNModel
                dqn_config = config.get('dqn_config', {})
                dqn_config['training_mode'] = False  # Inference only
                self.models.append(DQNModel(dqn_config))
                self.model_names.append('dqn')
                print("Added DQN model to ensemble")
            except Exception as e:
                print(f"Warning: Could not load DQN model: {e}")

        # Validate weights
        if self.weights:
            if len(self.weights) != len(self.models):
                print(f"Warning: Number of weights ({len(self.weights)}) doesn't match "
                      f"number of models ({len(self.models)}). Using equal weights.")
                self.weights = [1.0 / len(self.models)] * len(self.models)
            else:
                # Normalize weights
                total = sum(self.weights)
                self.weights = [w / total for w in self.weights]
        else:
            # Equal weights by default
            self.weights = [1.0 / len(self.models)] * len(self.models)

        print(f"Ensemble model initialized with {len(self.models)} models: {self.model_names}")
        print(f"Aggregation strategy: {self.aggregation_strategy}")
        print(f"Weights: {dict(zip(self.model_names, self.weights))}")

    def make_decisions(self, state: dict) -> List[dict]:
        """
        Make decisions by aggregating predictions from multiple models.

        Args:
            state: Current simulation state

        Returns:
            List of decision dictionaries
        """
        # Get decisions from all models
        all_decisions = []
        for model in self.models:
            try:
                decisions = model.make_decisions(state)
                all_decisions.append(decisions)
            except Exception as e:
                print(f"Warning: Model {model.__class__.__name__} failed: {e}")
                all_decisions.append([])

        # Aggregate decisions based on strategy
        if self.aggregation_strategy == 'voting':
            return self._voting_aggregation(all_decisions, state)
        elif self.aggregation_strategy == 'weighted':
            return self._weighted_aggregation(all_decisions, state)
        elif self.aggregation_strategy == 'cascade':
            return self._cascade_aggregation(all_decisions, state)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")

    def _voting_aggregation(self, all_decisions: List[List[dict]], state: dict) -> List[dict]:
        """
        Majority voting aggregation.

        For each vehicle, count votes for each action and select the most popular.
        """
        fleet = state['fleet']
        final_decisions = []

        # Group decisions by vehicle
        for vehicle in fleet.vehicles.values():
            if not vehicle.available:
                continue

            # Collect votes for this vehicle
            votes = []
            for decisions in all_decisions:
                for decision in decisions:
                    if decision['vehicle_id'] == vehicle.vehicle_id:
                        votes.append(decision)
                        break

            if not votes:
                # No decisions, default to IDLE
                final_decisions.append({
                    'vehicle_id': vehicle.vehicle_id,
                    'action': 'IDLE'
                })
                continue

            # Count action votes
            action_votes = Counter([v['action'] for v in votes])

            # Get most voted action
            most_voted_action = action_votes.most_common(1)[0][0]

            # Find a decision with this action
            for vote in votes:
                if vote['action'] == most_voted_action:
                    final_decisions.append(vote)
                    break

        return final_decisions

    def _weighted_aggregation(self, all_decisions: List[List[dict]], state: dict) -> List[dict]:
        """
        Weighted voting aggregation.

        Each model's vote is weighted by its performance weight.
        """
        fleet = state['fleet']
        final_decisions = []

        # Group decisions by vehicle
        for vehicle in fleet.vehicles.values():
            if not vehicle.available:
                continue

            # Collect weighted votes for this vehicle
            action_scores = defaultdict(float)
            action_decisions = {}

            for model_idx, decisions in enumerate(all_decisions):
                weight = self.weights[model_idx]

                for decision in decisions:
                    if decision['vehicle_id'] == vehicle.vehicle_id:
                        action = decision['action']
                        action_scores[action] += weight

                        # Store the first decision for each action type
                        if action not in action_decisions:
                            action_decisions[action] = decision
                        break

            if not action_scores:
                # No decisions, default to IDLE
                final_decisions.append({
                    'vehicle_id': vehicle.vehicle_id,
                    'action': 'IDLE'
                })
                continue

            # Select action with highest weighted score
            best_action = max(action_scores.items(), key=lambda x: x[1])[0]
            final_decisions.append(action_decisions[best_action])

        return final_decisions

    def _cascade_aggregation(self, all_decisions: List[List[dict]], state: dict) -> List[dict]:
        """
        Cascade aggregation (priority-based).

        Use decisions from the first model that provides them for each vehicle.
        Models are tried in order of their weights (highest first).
        """
        fleet = state['fleet']
        final_decisions = []

        # Sort models by weight (highest first)
        sorted_indices = sorted(
            range(len(self.models)),
            key=lambda i: self.weights[i],
            reverse=True
        )

        # Group decisions by vehicle
        for vehicle in fleet.vehicles.values():
            if not vehicle.available:
                continue

            decision_found = False

            # Try models in priority order
            for model_idx in sorted_indices:
                decisions = all_decisions[model_idx]

                for decision in decisions:
                    if decision['vehicle_id'] == vehicle.vehicle_id:
                        # Skip IDLE if not the last model
                        if decision['action'] == 'IDLE' and model_idx != sorted_indices[-1]:
                            continue

                        final_decisions.append(decision)
                        decision_found = True
                        break

                if decision_found:
                    break

            if not decision_found:
                # Fallback to IDLE
                final_decisions.append({
                    'vehicle_id': vehicle.vehicle_id,
                    'action': 'IDLE'
                })

        return final_decisions

    def update_weights(self, new_weights: List[float]):
        """
        Update model weights (for adaptive ensembles).

        Args:
            new_weights: List of new weights for each model
        """
        if len(new_weights) != len(self.models):
            raise ValueError(
                f"Number of weights ({len(new_weights)}) must match "
                f"number of models ({len(self.models)})"
            )

        # Normalize weights
        total = sum(new_weights)
        self.weights = [w / total for w in new_weights]

        print(f"Updated ensemble weights: {dict(zip(self.model_names, self.weights))}")

    def get_model_statistics(self) -> Dict:
        """
        Get statistics about the ensemble configuration.

        Returns:
            Dictionary with ensemble metadata
        """
        return {
            'num_models': len(self.models),
            'model_names': self.model_names,
            'weights': dict(zip(self.model_names, self.weights)),
            'aggregation_strategy': self.aggregation_strategy
        }


class AdaptiveEnsembleModel(EnsembleModel):
    """
    Adaptive ensemble that adjusts model weights based on recent performance.

    Tracks performance of each sub-model and updates weights using
    exponential moving average of rewards.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)

        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.window_size = config.get('window_size', 100)

        # Performance tracking
        self.model_rewards = [[] for _ in self.models]
        self.decisions_count = 0

    def make_decisions(self, state: dict) -> List[dict]:
        """
        Make decisions and track which model's advice was followed.

        Args:
            state: Current simulation state

        Returns:
            List of decision dictionaries
        """
        # Get decisions from parent
        decisions = super().make_decisions(state)

        self.decisions_count += 1

        # Adapt weights periodically
        if self.decisions_count % self.window_size == 0:
            self._adapt_weights()

        return decisions

    def record_rewards(self, model_idx: int, reward: float):
        """
        Record reward for a specific model.

        Args:
            model_idx: Index of the model
            reward: Reward received
        """
        self.model_rewards[model_idx].append(reward)

        # Keep only recent rewards
        if len(self.model_rewards[model_idx]) > self.window_size:
            self.model_rewards[model_idx].pop(0)

    def _adapt_weights(self):
        """
        Adapt model weights based on recent performance.

        Uses softmax of average rewards to compute new weights.
        """
        avg_rewards = []
        for rewards in self.model_rewards:
            if rewards:
                avg_rewards.append(np.mean(rewards))
            else:
                avg_rewards.append(0.0)

        # Compute softmax weights
        exp_rewards = np.exp(np.array(avg_rewards) - np.max(avg_rewards))
        new_weights = exp_rewards / exp_rewards.sum()

        # Exponential moving average with old weights
        self.weights = [
            (1 - self.adaptation_rate) * old_w + self.adaptation_rate * new_w
            for old_w, new_w in zip(self.weights, new_weights)
        ]

        # Normalize
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        print(f"Adapted ensemble weights: {dict(zip(self.model_names, self.weights))}")
