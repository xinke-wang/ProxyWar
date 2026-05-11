"""
Base agent classes for ProxyWar framework.

This module defines the base agent interface and common agent implementations.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all game agents in ProxyWar framework.
    
    Agents are simple decision-making entities that receive game state
    and return actions. All complexity should be handled by the framework.
    """
    
    def __init__(self, name: str):
        """
        Initialize the base agent.
        
        Args:
            name: The name identifier for this agent
        """
        self.name = name
    
    @abstractmethod
    def select_action(self, observation: Any, action_mask: List[bool]) -> Optional[int]:
        """
        Select an action based on the current observation and available actions.
        
        Args:
            observation: The current game state observation
            action_mask: Boolean mask indicating which actions are legal
            
        Returns:
            The selected action index, or None if no action is possible
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class RandomAgent(BaseAgent):
    """
    A simple random agent that selects actions uniformly at random 
    from the set of legal actions.
    """
    
    def __init__(self, name: str, seed: Optional[int] = None):
        """
        Initialize the random agent.
        
        Args:
            name: The name identifier for this agent
            seed: Optional random seed for reproducibility
        """
        super().__init__(name)
        self.random_state = random.Random(seed)
    
    def select_action(self, observation: Any, action_mask: List[bool]) -> Optional[int]:
        """
        Select a random action from the available legal actions.
        
        Args:
            observation: The current game state observation (unused for random agent)
            action_mask: Boolean mask indicating which actions are legal
            
        Returns:
            A randomly selected legal action index, or None if no actions are available
        """
        legal_actions = [i for i, legal in enumerate(action_mask) if legal]
        if legal_actions:
            return self.random_state.choice(legal_actions)
        return None 