"""
TicTacToe game implementation for ProxyWar framework.

This module implements TicTacToe using PettingZoo environment.
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
from pettingzoo.classic import tictactoe_v3

from .base import TwoPlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


@GAME_REGISTRY.register('tictactoe')
class TicTacToeGame(TwoPlayerGame):
    """
    TicTacToe game implementation using PettingZoo.
    
    This class manages TicTacToe matches between two agents.
    """
    
    def __init__(self, movement_timeout: float = 45.0):
        """Initialize TicTacToe game."""
        super().__init__("TicTacToe", movement_timeout)
        self.env = None
        self.agent_mapping = {}
        
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """
        Setup a TicTacToe match between two agents.
        
        Args:
            agent1: First agent (will play as X)
            agent2: Second agent (will play as O)
        """
        # Call parent setup_match to set agent1, agent2 and reset
        super().setup_match(agent1, agent2)
        
        # Create fresh environment for each match
        self.env = tictactoe_v3.env(render_mode="rgb_array")
        self.env.reset()
        
        # Map PettingZoo agent names to our agents
        self.agent_mapping = {
            "player_1": agent1,  # X player
            "player_2": agent2   # O player
        }
        
        print(f"Match setup: {agent1.name} (X) vs {agent2.name} (O)")
    
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete TicTacToe match.
        
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
            obs_3d = observation["observation"]  # Shape: (3, 3, 2)
            flat_observation = []
            for i in range(3):
                for j in range(3):
                    if obs_3d[i][j][0] == 1:  # Player 1 (X)
                        flat_observation.append(1)
                    elif obs_3d[i][j][1] == 1:  # Player 2 (O)
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
                pos_row, pos_col = action // 3, action % 3
                print(f"Turn {game_step}: {current_agent.name} selects position ({pos_row}, {pos_col}) in {decision_time:.2e}s")
                
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
        Get TicTacToe game rules description.
        
        Returns:
            String description of TicTacToe rules
        """
        return """
TicTacToe Game Rules:

1. The game is played on a 3x3 grid
2. Two players take turns placing their marks (X and O)
3. Player 1 uses X, Player 2 uses O
4. The first player to get 3 of their marks in a row (horizontally, vertically, or diagonally) wins
5. If all 9 squares are filled and no player has 3 in a row, the game is a draw

Agent Interface:
- select_action(observation, action_mask) should return the position index (0-8) to place the mark
- action_mask indicates which positions are legal (True) or occupied (False)
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for TicTacToe."""
        # Create temporary environment to get real observation format
        temp_env = tictactoe_v3.env(render_mode="rgb_array")
        temp_env.reset()
        
        # Get actual observation from the environment
        for agent_name in temp_env.agent_iter():
            observation, reward, termination, truncation, info = temp_env.last()
            
            if observation is None:
                break
                
            # Convert PettingZoo 3D observation to flat format that agents expect
            obs_3d = observation["observation"]  # Shape: (3, 3, 2)
            action_mask = observation["action_mask"]  # Shape: (9,)
            
            # Convert to flat board representation (0=empty, 1=player1, 2=player2)
            flat_board = []
            for i in range(3):
                for j in range(3):
                    if obs_3d[i][j][0] == 1:  # Player 1 (X)
                        flat_board.append(1)
                    elif obs_3d[i][j][1] == 1:  # Player 2 (O) 
                        flat_board.append(2)
                    else:
                        flat_board.append(0)  # Empty
            
            temp_env.close()
            
            return {
                'description': '''
The observation is a list of 9 integers representing the 3x3 TicTacToe board in row-major order.
- Position mapping: [0, 1, 2]
                    [3, 4, 5] 
                    [6, 7, 8]
- Values: 0 = Empty, 1 = Player 1 (X), 2 = Player 2 (O)
- action_mask is a list of 9 booleans indicating legal moves (True = legal, False = occupied)
''',
                'sample_observation': flat_board,
                'action_space_size': 9,
                'sample_action_mask': action_mask.tolist(),
                'position_mapping': 'Positions 0-8 map to board as: 0=top-left, 1=top-center, 2=top-right, 3=middle-left, 4=center, 5=middle-right, 6=bottom-left, 7=bottom-center, 8=bottom-right'
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
                    if action is not None:
                        pos_row, pos_col = action // 3, action % 3
                        print(f"  Move {i+1}: {player} places at ({pos_row}, {pos_col})")
                    else:
                        print(f"  Move {i+1}: {player} made invalid move (None)")
                
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