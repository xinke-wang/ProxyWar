"""
Base prompt classes for ProxyWar framework.

This module defines the interface for prompt generation for LLM code generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..games.base import BaseGame


class BasePrompt(ABC):
    """
    Abstract base class for all prompt generators in ProxyWar framework.
    
    Prompts are responsible for creating formatted instructions that guide
    LLMs to generate appropriate agent code for specific games.
    """
    
    def __init__(self, prompt_name: str):
        """
        Initialize the base prompt.
        
        Args:
            prompt_name: Name identifier for this prompt style
        """
        self.prompt_name = prompt_name
    
    @abstractmethod
    def generate_prompt(self, 
                       game: BaseGame,
                       additional_context: Optional[str] = None,
                       code_requirements: Optional[Dict[str, Any]] = None,
                       agent_name: Optional[str] = None) -> str:
        """
        Generate a complete prompt for agent code generation.
        
        Args:
            game: The game instance containing rules and state information
            additional_context: Optional additional context or constraints
            code_requirements: Optional specific code requirements or format constraints
            
        Returns:
            Complete formatted prompt string ready for LLM
        """
        pass
    
    def get_default_code_requirements(self) -> Dict[str, Any]:
        """
        Get default code requirements for agent generation.
        
        Returns:
            Dictionary containing default code structure requirements
        """
        return {
            'base_class': 'BaseAgent',
            'method_name': 'select_action',
            'input_params': ['observation', 'action_mask'],
            'return_type': 'Optional[int]',
            'language': 'python',
            'style': 'clean and readable',
            'comments': 'brief explanatory comments'
        }
    
    def __str__(self) -> str:
        """String representation of the prompt."""
        return f"{self.__class__.__name__}(name='{self.prompt_name}')" 