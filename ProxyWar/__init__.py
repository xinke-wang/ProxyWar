"""
ProxyWar: A Competitive Framework for Evaluating Code Generation Quality

This package implements a tournament-style framework for evaluating 
code generation models through game-based competitions.
"""

from .agents import BaseAgent, RandomAgent
from .coders import BaseCoder
from .games import BaseGame, TicTacToeGame, SnakeGame
from .prompts import BasePrompt, PlainPrompt
from .testers import BaseTester, TicTacToeTester, SnakeTester
from .registry import AGENT_REGISTRY, CODER_REGISTRY, GAME_REGISTRY, PROMPT_REGISTRY, list_registered_coders, print_registered_coders

__version__ = "0.1.0"
__all__ = ['BaseAgent', 'RandomAgent', 'BaseCoder', 'BaseGame', 'TicTacToeGame', 'SnakeGame',
           'BasePrompt', 'PlainPrompt', 'BaseTester', 'TicTacToeTester', 'SnakeTester',
           'AGENT_REGISTRY', 'CODER_REGISTRY', 'GAME_REGISTRY', 'PROMPT_REGISTRY',
           'list_registered_coders', 'print_registered_coders'] 