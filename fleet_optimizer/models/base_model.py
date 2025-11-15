"""
Base optimization model.

Defines the interface that all optimization models must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseOptimizationModel(ABC):
    """
    Abstract base class for optimization models.

    All optimization models must inherit from this class and implement
    the make_decisions method.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the optimization model.

        Args:
            config: Configuration dictionary with model-specific parameters
        """
        self.config = config or {}

    @abstractmethod
    def make_decisions(self, state: dict) -> List[dict]:
        """
        Make decisions for all vehicles based on current state.

        Args:
            state: Dictionary containing:
                - current_time: Current simulation time (minutes)
                - fleet: Fleet object
                - depots: Dict of Depot objects
                - active_requests: Dict of active RideRequest objects
                - distance_func: Function to calculate distance

        Returns:
            List of decision dictionaries, each containing:
                - vehicle_id: ID of the vehicle
                - action: One of ['PICKUP_PASSENGER', 'CHARGING', 'REPOSITION', 'IDLE']
                - Additional action-specific fields:
                    - For PICKUP_PASSENGER: request_id
                    - For CHARGING: depot_id
                    - For REPOSITION: target_location
        """
        pass

    def get_config(self) -> dict:
        """Get model configuration"""
        return self.config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
