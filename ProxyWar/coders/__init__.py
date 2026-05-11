"""
ProxyWar Coders Module

This module contains code generation interfaces for different LLMs.
Coders are responsible for generating agent code based on game rules and prompts.
"""

from .base import BaseCoder
from .open_router_coders import OpenRouterCoder, MiniMaxM1Coder, Phi4Coder
from .utils import clean_generated_code, load_agent_from_file

__all__ = ['BaseCoder', 'OpenRouterCoder', 'MiniMaxM1Coder', 'Phi4Coder', 'clean_generated_code', 'load_agent_from_file'] 