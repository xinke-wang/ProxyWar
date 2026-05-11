"""
Base coder classes for ProxyWar framework.

This module defines the interface for code generation from different LLMs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseCoder(ABC):
    """
    Abstract base class for all code generators in ProxyWar framework.
    
    Coders are responsible for generating agent code based on game rules,
    current game state, and model-specific prompts.
    """
    
    # Model categories
    GENERAL = "general"
    REASONING = "reasoning" 
    CODE = "code"
    
    def __init__(self, model_name: str, model_category: str = GENERAL, **kwargs):
        """
        Initialize the base coder.
        
        Args:
            model_name: Name/identifier of the underlying model
            model_category: Category of the model (general, reasoning, code)
            **kwargs: Additional model-specific configuration
        """
        self.model_name = model_name
        self.model_category = model_category
        self.config = kwargs
        self.conversation_history: List[Dict[str, str]] = []
    
    @abstractmethod
    def generate_agent_code(self, prompt: str) -> str:
        """
        Generate agent code based on the provided prompt.
        
        Args:
            prompt: The complete prompt string to send to the model
            
        Returns:
            Generated Python code as a string that implements an agent
        """
        pass
    
    def revise_agent_code(self, original_prompt: str, previous_code: str, 
                         test_errors: str, revision_prompt: str) -> str:
        """
        Revise agent code based on test errors and feedback.
        
        This method implements multi-round conversation to fix code issues.
        
        Args:
            original_prompt: The original prompt used to generate the code
            previous_code: The previously generated code that failed tests
            test_errors: Detailed test error information
            revision_prompt: Additional instructions for revision
            
        Returns:
            Revised Python code as a string
        """
        if not self.conversation_history:
            self.conversation_history = [
                {"role": "user", "content": original_prompt},
                {"role": "assistant", "content": previous_code}
            ]

        revision_message = f"""
The previous code failed testing with the following errors:

{test_errors}

{revision_prompt}

Please provide a corrected version of the complete agent code that addresses these issues.
"""

        self.conversation_history.append({"role": "user", "content": revision_message})

        revised_code = self._generate_with_history(self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": revised_code})

        return revised_code
    
    @abstractmethod
    def _generate_with_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate code using conversation history for multi-round chat.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Generated code as string
        """
        pass
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history.clear()
    
    def __str__(self) -> str:
        """String representation of the coder."""
        return f"{self.__class__.__name__}(model='{self.model_name}', category='{self.model_category}')" 