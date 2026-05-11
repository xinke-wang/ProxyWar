"""
Configuration loader for ProxyWar experiments.

This module provides functionality to load experiment configurations
using pjtools configurator system.
"""

import os
from typing import Dict, Any, Optional, List
from pjtools.configurator import AutoConfigurator


class ConfigLoader:
    """
    Loads and manages ProxyWar experiment configurations.
    """
    
    def __init__(self, config_path: str = "configs/minimal.py"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = None
    
    def load_config(self):
        """
        Load configuration from file using pjtools configurator.
        
        Returns:
            PyConfigurator object containing loaded configuration
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Use pjtools AutoConfigurator to load config
        configurator = AutoConfigurator()
        self.config = configurator.fromfile(self.config_path)
        return self.config
    
    def get_game_config(self):
        """Get game configuration as a list."""
        if not self.config:
            self.load_config()
        return self.config.games
    
    def get_coders_config(self):
        """Get coders configuration."""
        if not self.config:
            self.load_config()
        return self.config.coders
    
    def get_prompt_config(self):
        """Get prompt configuration."""
        if not self.config:
            self.load_config()
        return self.config.prompt
    
    def get_api_config(self):
        """Get API configuration."""
        if not self.config:
            self.load_config()
        return self.config.api
    
    def get_experiment_name(self):
        """Get experiment name."""
        if not self.config:
            self.load_config()
        return self.config.experiment_name
    
    def get_save_path(self):
        """Get save path for experiments."""
        if not self.config:
            self.load_config()
        return self.config.save_path
    
    def is_single_player_mode(self):
        """Check if configuration is for single-player mode."""
        if not self.config:
            self.load_config()
        return bool(getattr(self.config, 'single_player_mode', None))

    def use_success_rate_scoring(self):
        """Check if should use success rate instead of ELO scoring."""
        if not self.config:
            self.load_config()
        return bool(getattr(self.config, 'use_success_rate_scoring', None))

    def get_evaluation_criteria(self):
        """Get evaluation criteria for single-player games."""
        if not self.config:
            self.load_config()
        return getattr(self.config, 'evaluation_criteria', None) or {}


def create_components_from_config(config_loader: ConfigLoader):
    """
    Create ProxyWar components from configuration using registry system.
    
    Args:
        config_loader: Loaded configuration
        
    Returns:
        Tuple of (games, agents_info, prompt_generator) where:
        - games is a list of game instances
        - agents_info is a list of dicts containing agent name and coder instance
    """
    from .. import GAME_REGISTRY, CODER_REGISTRY, PROMPT_REGISTRY
    
    # Get configurations
    game_types = config_loader.get_game_config()  # Always a list now
    agents_coders = config_loader.get_coders_config()  # Now this is a list of coder type strings
    prompt_type = config_loader.get_prompt_config()
    
    # Validate game configuration
    if not isinstance(game_types, list):
        raise ValueError(f"Game configuration must be a list, got: {type(game_types)}")
    
    if not game_types:
        raise ValueError("At least one game must be specified in games list")
    
    # Get timeout configurations from config.
    # pjtools' configurator returns None for unset attributes, so the
    # getattr default is never used — coerce None to the intended default.
    movement_timeout = getattr(config_loader.config, 'movement_timeout', None) or 45.0
    single_player_timeout = getattr(config_loader.config, 'single_player_timeout', None) or 60.0
    
    # Create games
    games = []
    for game_type in game_types:
        if not game_type:
            raise ValueError("Empty game type found in games list")
        
        game_class = GAME_REGISTRY.get(game_type)
        if game_class is None:
            available_games = list(GAME_REGISTRY._modules.keys()) if hasattr(GAME_REGISTRY, '_modules') else []
            raise ValueError(f"Unknown game type: {game_type}. Available types: {available_games}")
        
        # Create game with appropriate timeout parameter
        # Check if this is a single-player game by checking if it's registered as such
        is_single_player_game = game_type in ['hanoi']  # Add other single-player games here as needed
        
        if is_single_player_game:
            # For single-player games, pass both movement_timeout and total_game_timeout
            game = game_class(movement_timeout=movement_timeout, total_game_timeout=single_player_timeout)
        else:
            # For multi-player games, only pass movement_timeout
            game = game_class(movement_timeout=movement_timeout)
        
        games.append(game)
    
    # Create prompt generator
    if not prompt_type:
        raise ValueError("Prompt type not specified in configuration")
    
    prompt_class = PROMPT_REGISTRY.get(prompt_type)
    if prompt_class is None:
        available_prompts = list(PROMPT_REGISTRY._modules.keys()) if hasattr(PROMPT_REGISTRY, '_modules') else []
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available types: {available_prompts}")
    
    prompt_generator = prompt_class()
    
    # Create agents info with their respective coders
    agents_info = []
    for coder_type in agents_coders:
        if not coder_type:
            raise ValueError("Empty coder type found in agents list")
        
        coder_class = CODER_REGISTRY.get(coder_type)
        if coder_class is None:
            available_coders = list(CODER_REGISTRY._modules.keys()) if hasattr(CODER_REGISTRY, '_modules') else []
            raise ValueError(f"Unknown coder type: {coder_type}. Available types: {available_coders}")
        
        coder = coder_class()
        
        # Auto-generate agent name based on coder type
        # Convert "minimax_m1" -> "MiniMaxM1Agent", "phi4" -> "Phi4Agent"
        agent_name = _generate_agent_name_from_coder_type(coder_type)
        
        agents_info.append({
            'name': agent_name,
            'coder': coder
        })
    
    # Always return list of games
    return games, agents_info, prompt_generator


def _generate_agent_name_from_coder_type(coder_type: str) -> str:
    """
    Generate a human-readable agent name from coder type.
    
    Args:
        coder_type: The coder type string (e.g., "minimax_m1", "phi4")
        
    Returns:
        Generated agent name (e.g., "MiniMaxM1Agent", "Phi4Agent")
    """
    # Split by underscore and capitalize each part
    parts = coder_type.split('_')
    capitalized_parts = []
    
    for part in parts:
        # Handle special cases
        if part.lower() == 'm1':
            capitalized_parts.append('M1')
        elif part.lower() == 'phi4':
            capitalized_parts.append('Phi4')
        elif part.lower() == 'minimax':
            capitalized_parts.append('MiniMax')
        else:
            # Default: capitalize first letter
            capitalized_parts.append(part.capitalize())
    
    return ''.join(capitalized_parts) + 'Agent' 