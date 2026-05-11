"""
ProxyWar Evaluations Module

This module handles evaluation management, including experiment setup,
agent code generation, match running, and result collection.
"""

from .tournament_manager import TournamentManager
from .stages.tournament_stage import run_tournament_from_config
from .config_loader import ConfigLoader, create_components_from_config
from .elo_system import TrueSkillSystem, SuccessRateTracker, HybridRatingSystem
from .data_models import CoderResult, MatchResult, SinglePlayerResult, RoundResult, MultiRoundStats

__all__ = [
    'TournamentManager', 'run_tournament_from_config',
    'ConfigLoader', 'create_components_from_config',
    'TrueSkillSystem', 'SuccessRateTracker', 'HybridRatingSystem',
    'CoderResult', 'MatchResult', 'SinglePlayerResult', 'RoundResult', 'MultiRoundStats'
]
