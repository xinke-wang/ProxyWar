"""
Connect Four game implementation for ProxyWar framework.

This module implements Connect Four using PettingZoo environment.
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
from pettingzoo.classic import connect_four_v3

from .base import TwoPlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


@GAME_REGISTRY.register('connectfour')
class ConnectFourGame(TwoPlayerGame):
    """
    Connect Four game implementation using PettingZoo.
    
    This class manages Connect Four matches between two agents.
    """
    
    def __init__(self, movement_timeout: float = 45.0):
        """Initialize Connect Four game."""
        super().__init__("ConnectFour", movement_timeout)
        self.env = None
        self.agent_mapping = {}
        
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """
        Setup a Connect Four match between two agents.
        
        Args:
            agent1: First agent (will play as Player 0)
            agent2: Second agent (will play as Player 1)
        """
        # Call parent setup_match to set agent1, agent2 and reset
        super().setup_match(agent1, agent2)
        
        # Create fresh environment for each match
        self.env = connect_four_v3.env(render_mode="rgb_array")
        self.env.reset()
        
        # Map PettingZoo agent names to our agents
        self.agent_mapping = {
            "player_0": agent1,  # First player
            "player_1": agent2   # Second player
        }
        
        print(f"Match setup: {agent1.name} (Player 0) vs {agent2.name} (Player 1)")
    
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete Connect Four match.
        
        Returns:
            Dictionary containing match results
        """
        if not self.env or not self.agent_mapping:
            raise ValueError("Match not properly setup. Call setup_match() first.")
        
        move_history = []
        move_history_with_timing = []
        game_step = 0
        final_rewards = {}
        
        print("Game started...")
        
        # Game loop
        for pz_agent_name in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            
            # Store reward for current agent
            current_agent = self.agent_mapping[pz_agent_name]
            final_rewards[current_agent.name] = reward
            
            if termination or truncation:
                print(f"Game ended: termination={termination}, truncation={truncation}")
                print(f"Final rewards: {final_rewards}")
                break
            
            # Check if observation is None
            if observation is None:
                break
                
            # Get action mask (legal actions)
            action_mask = observation["action_mask"]
            
            # Convert PettingZoo 3D observation to flat format for agent
            obs_3d = observation["observation"]  # Shape: (6, 7, 2)
            flat_observation = []
            for i in range(6):
                for j in range(7):
                    if obs_3d[i][j][0] == 1:  # Player 1
                        flat_observation.append(1)
                    elif obs_3d[i][j][1] == 1:  # Player 2
                        flat_observation.append(2)
                    else:
                        flat_observation.append(0)  # Empty
            
            # Agent selects action with timeout detection (simplified)
            action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
                current_agent, flat_observation, action_mask.tolist(),
                move_history, move_history_with_timing, game_step
            )
            
            # If timeout occurred, return the timeout result immediately
            if timeout_result is not None:
                return timeout_result
            
            if action is not None:
                # Record move with timing
                move_history.append((current_agent.name, action))
                move_history_with_timing.append((current_agent.name, action, decision_time))
                
                game_step += 1
                print(f"Turn {game_step}: {current_agent.name} drops piece in column {action} in {decision_time:.3f}s")
                
                # Execute action
                self.env.step(action)
            else:
                # If no legal action, pass None
                self.env.step(None)
        
        # Determine winner
        if final_rewards and len(final_rewards) >= 2:
            agent_names = list(final_rewards.keys())
            agent1_name, agent2_name = agent_names[0], agent_names[1]
            
            if final_rewards[agent1_name] > final_rewards[agent2_name]:
                winner = agent1_name
            elif final_rewards[agent2_name] > final_rewards[agent1_name]:
                winner = agent2_name
            else:
                winner = "draw"
        else:
            # Fallback if not enough agents recorded
            winner = "draw"
        
        self.current_match_history = move_history
        
        result = {
            'winner': winner,
            'scores': final_rewards,
            'moves': len(move_history),
            'match_history': move_history,
            'move_history_with_timing': move_history_with_timing
        }
        
        print(f"Game Over! Result: {winner}")
        print(f"Final Scores: {final_rewards}")
        
        return result
    
    def get_game_rules(self) -> str:
        """
        Get Connect Four game rules description.
        
        Returns:
            String description of Connect Four rules
        """
        return """
Connect Four Game Rules:

1. The game is played on a 6x7 grid (6 rows, 7 columns)
2. Two players take turns dropping colored discs into columns
3. Player 0 uses one color, Player 1 uses another color
4. Discs fall down due to gravity and occupy the lowest available space in the column
5. Players cannot place a disc in a full column
6. The first player to connect four of their discs in a row (horizontally, vertically, or diagonally) wins
7. If all 42 spaces are filled and no player has connected four, the game is a draw

Agent Interface:
- select_action(observation, action_mask) should return the column index (0-6) to drop the disc
- action_mask indicates which columns are legal (True) or full (False)
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for Connect Four."""
        # Create temporary environment to get real observation format
        temp_env = connect_four_v3.env(render_mode="rgb_array")
        temp_env.reset()
        
        # Get actual observation from the environment
        for agent_name in temp_env.agent_iter():
            observation, reward, termination, truncation, info = temp_env.last()
            
            if observation is None:
                break
                
            # Convert PettingZoo 3D observation to flat format that agents expect
            obs_3d = observation["observation"]  # Shape: (6, 7, 2)
            action_mask = observation["action_mask"]  # Shape: (7,)
            
            # Convert to flat board representation (0=empty, 1=player0, 2=player1)
            flat_board = []
            for i in range(6):  # 6 rows
                for j in range(7):  # 7 columns
                    if obs_3d[i][j][0] == 1:  # Player 0
                        flat_board.append(1)
                    elif obs_3d[i][j][1] == 1:  # Player 1
                        flat_board.append(2)
                    else:
                        flat_board.append(0)  # Empty
            
            temp_env.close()
            
            return {
                'description': '''
The observation is a list of 42 integers representing the 6x7 Connect Four board in row-major order.
- Position mapping: [0,  1,  2,  3,  4,  5,  6 ]    Row 0
                    [7,  8,  9,  10, 11, 12, 13]    Row 1
                    [14, 15, 16, 17, 18, 19, 20]    Row 2
                    [21, 22, 23, 24, 25, 26, 27]    Row 3
                    [28, 29, 30, 31, 32, 33, 34]    Row 4
                    [35, 36, 37, 38, 39, 40, 41]    Row 5
- Values: 0 = Empty, 1 = Player 0, 2 = Player 1
- action_mask is a list of 7 booleans indicating legal columns (True = can drop disc, False = column full)
''',
                'sample_observation': flat_board,
                'action_space_size': 7,
                'sample_action_mask': action_mask.tolist(),
                'position_mapping': 'Actions 0-6 correspond to columns 0-6. Discs drop to the lowest available row in the selected column.'
            }
        
        temp_env.close()
        return {}
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save game visualization using PettingZoo's rendering.
        
        Args:
            save_path: Path to save the visualization image
            
        Returns:
            True if successful, False otherwise
        """
        if not self.env:
            return False
        
        try:
            # Use PettingZoo's rendering functionality
            rendered_image = self.env.render()
            
            if rendered_image is not None and isinstance(rendered_image, np.ndarray):
                image = Image.fromarray(rendered_image)
                image.save(save_path)
                
                print(f"Game visualization saved to: {save_path}")
                print("\nGame Statistics:")
                print(f"Total Moves: {len(self.current_match_history)}")
                print("Move History:")
                for i, (player, action) in enumerate(self.current_match_history):
                    print(f"  Move {i+1}: {player} drops disc in column {action}")
                
                return True
            else:
                print("Unable to render game screen")
                return False
                
        except Exception as e:
            print(f"Error occurred while saving visualization: {e}")
            return False
    
    def reset(self) -> None:
        """Reset game state for a new match."""
        super().reset()
        if self.env:
            self.env.close()
        self.env = None
        self.agent_mapping = {}
        self.agents = {}
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.env:
            self.env.close() 