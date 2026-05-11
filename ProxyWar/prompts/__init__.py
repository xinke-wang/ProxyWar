"""
ProxyWar Prompts Module

This module contains prompt generation classes for different LLM interactions.
Prompts are responsible for creating formatted instructions for code generation.
"""

from .base import BasePrompt
from .plain import PlainPrompt

__all__ = ['BasePrompt', 'PlainPrompt'] 