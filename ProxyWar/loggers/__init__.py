"""
ProxyWar Logging System

This package provides unified logging functionality for the ProxyWar framework.
"""

from .base_logger import BaseLogger
from .tournament_logger import TournamentLogger

__all__ = ['BaseLogger', 'TournamentLogger'] 