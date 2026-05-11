"""
2048 game implementation for ProxyWar framework.

This module implements the classic 2048 puzzle game
where agents need to merge tiles to reach 2048.
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from .base import SinglePlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


class TwentyFortyEightState:
    """Represents the state of a 2048 game."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize 2048 state.
        
        Args:
            seed: Random seed for reproducible games
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize 4x4 board
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.start_time = time.time()
        self.move_count = 0
        self.invalid_moves = 0
        
        # Add two initial tiles
        self._add_random_tile()
        self._add_random_tile()
        
        # Track initial state for visualization
        self.initial_board = self.get_board_copy()
    
    def _add_random_tile(self):
        """Add a random tile (2 or 4) to an empty position."""
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_cells.append((i, j))
        
        if empty_cells:
            row, col = random.choice(empty_cells)
            # 90% chance of 2, 10% chance of 4
            self.board[row][col] = 2 if random.random() < 0.9 else 4
    
    def get_board_copy(self) -> List[List[int]]:
        """Get a copy of the current board state."""
        return [[self.board[row][col] for col in range(4)] for row in range(4)]
    
    def get_observation(self) -> List[int]:
        """
        Get flattened observation of the current game state.
        
        Returns:
            Flattened 16-element list representing the 4x4 grid
        """
        observation = []
        for row in range(4):
            for col in range(4):
                observation.append(self.board[row][col])
        return observation
    
    def get_action_mask(self) -> List[bool]:
        """
        Get action mask for valid moves.
        
        Returns:
            4-element boolean list for [UP, DOWN, LEFT, RIGHT]
        """
        return [
            self._can_move_up(),
            self._can_move_down(),
            self._can_move_left(),
            self._can_move_right()
        ]
    
    def _can_move_up(self) -> bool:
        """Check if up move is valid."""
        for col in range(4):
            for row in range(1, 4):
                if self.board[row][col] != 0:
                    if self.board[row-1][col] == 0 or self.board[row-1][col] == self.board[row][col]:
                        return True
        return False
    
    def _can_move_down(self) -> bool:
        """Check if down move is valid."""
        for col in range(4):
            for row in range(3):
                if self.board[row][col] != 0:
                    if self.board[row+1][col] == 0 or self.board[row+1][col] == self.board[row][col]:
                        return True
        return False
    
    def _can_move_left(self) -> bool:
        """Check if left move is valid."""
        for row in range(4):
            for col in range(1, 4):
                if self.board[row][col] != 0:
                    if self.board[row][col-1] == 0 or self.board[row][col-1] == self.board[row][col]:
                        return True
        return False
    
    def _can_move_right(self) -> bool:
        """Check if right move is valid."""
        for row in range(4):
            for col in range(3):
                if self.board[row][col] != 0:
                    if self.board[row][col+1] == 0 or self.board[row][col+1] == self.board[row][col]:
                        return True
        return False
    
    def _merge_line(self, line: List[int], reverse: bool = False) -> Tuple[List[int], int]:
        """
        Merge a single line according to 2048 rules.
        
        Args:
            line: A line of 4 values
            reverse: Whether to process in reverse order
            
        Returns:
            Tuple of (merged line, points scored)
        """
        if reverse:
            line = line[::-1]
        
        # Remove zeros and slide
        non_zero = [x for x in line if x != 0]
        
        # Merge adjacent equal values
        points = 0
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                points += non_zero[i] * 2
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros
        while len(merged) < 4:
            merged.append(0)
        
        if reverse:
            merged = merged[::-1]
        
        return merged, points
    
    def make_move(self, action: int) -> bool:
        """
        Make a move in the game.
        
        Args:
            action: Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            True if move was valid, False otherwise
        """
        if action < 0 or action > 3:
            return False
        
        # Save state before move
        board_before = self.get_board_copy()
        points_scored = 0
        
        if action == 0:  # UP
            for col in range(4):
                column = [self.board[row][col] for row in range(4)]
                merged_column, points = self._merge_line(column)
                points_scored += points
                for row in range(4):
                    self.board[row][col] = merged_column[row]
        
        elif action == 1:  # DOWN
            for col in range(4):
                column = [self.board[row][col] for row in range(4)]
                merged_column, points = self._merge_line(column, reverse=True)
                points_scored += points
                for row in range(4):
                    self.board[row][col] = merged_column[row]
        
        elif action == 2:  # LEFT
            for row in range(4):
                merged_row, points = self._merge_line(self.board[row])
                points_scored += points
                self.board[row] = merged_row
        
        elif action == 3:  # RIGHT
            for row in range(4):
                merged_row, points = self._merge_line(self.board[row], reverse=True)
                points_scored += points
                self.board[row] = merged_row
        
        # Check if move was valid (board changed)
        board_after = self.get_board_copy()
        if board_before != board_after:
            self.score += points_scored
            self.move_count += 1
            self._add_random_tile()
            return True
        else:
            self.invalid_moves += 1
            return False
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Check if any empty cells
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    return False
        
        # Check if any moves possible
        for row in range(4):
            for col in range(4):
                current = self.board[row][col]
                # Check right
                if col < 3 and self.board[row][col + 1] == current:
                    return False
                # Check down
                if row < 3 and self.board[row + 1][col] == current:
                    return False
        
        return True
    
    def get_score(self) -> int:
        """Get the current score."""
        return self.score
    
    def get_highest_tile(self) -> int:
        """Get the value of the highest tile on the board."""
        max_tile = 0
        for row in range(4):
            for col in range(4):
                max_tile = max(max_tile, self.board[row][col])
        return max_tile
    
    def has_won(self) -> bool:
        """Check if the player has reached 2048."""
        return self.get_highest_tile() >= 2048
    
    def __str__(self) -> str:
        """String representation of the game state."""
        result = "2048 Game State:\n"
        result += f"Score: {self.get_score()}\n"
        result += f"Moves: {self.move_count}\n"
        result += "Board:\n"
        for row in range(4):
            row_str = ""
            for col in range(4):
                value = self.board[row][col]
                if value == 0:
                    row_str += "   . "
                else:
                    row_str += f"{value:4d} "
            result += row_str + "\n"
        return result


@GAME_REGISTRY.register('2048')
class TwentyFortyEightGame(SinglePlayerGame):
    """
    2048 puzzle game implementation.
    
    This single-player game challenges agents to reach the 2048 tile
    by merging tiles with the same value.
    """
    
    def __init__(self, movement_timeout: float = 5.0, total_game_timeout: float = 180.0):
        """Initialize 2048 game with configurations."""
        super().__init__("2048", movement_timeout, total_game_timeout)
        
        # Store game states for visualization
        self.agent1_final_state: Optional[TwentyFortyEightState] = None
        self.agent2_final_state: Optional[TwentyFortyEightState] = None
        self.shared_seed: Optional[int] = None
        self.match_results: Optional[Dict[str, Any]] = None
    
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """Setup a 2048 match between two agents."""
        super().setup_match(agent1, agent2)
        print(f"2048 match: {agent1.name} vs {agent2.name}")
        print(f"Both agents will play the same initial game state")
        
        # Generate a shared random seed for both agents
        self.shared_seed = random.randint(0, 1000000)
    
    def run_match(self) -> Dict[str, Any]:
        """
        Run a 2048 match between two agents.
        
        Each agent plays the same initial game state independently.
        The agent with the higher score wins.
        
        Returns:
            Dictionary containing match results
        """
        if not self.agents or not self.agent1 or not self.agent2:
            raise ValueError("Match not properly setup. Call setup_match() first.")
        
        print(f"Starting 2048 match with shared seed: {self.shared_seed}")
        print("=" * 80)
        
        # Run game for both agents with the same seed
        if self.shared_seed is None:
            raise ValueError("Shared seed not initialized")
        agent1_result = self._run_agent_game(self.agent1, self.shared_seed, "Agent 1")
        agent2_result = self._run_agent_game(self.agent2, self.shared_seed, "Agent 2")
        
        # Store final states for visualization
        self.agent1_final_state = agent1_result['final_state']
        self.agent2_final_state = agent2_result['final_state']
        
        # Determine winner based on scores
        winner, scores = self._compare_agent_results(agent1_result, agent2_result)
        
        print(f"\nFINAL RESULTS:")
        print(f"  {self.agent1.name}: Score={agent1_result['score']}, Highest={agent1_result['highest_tile']}")
        print(f"  {self.agent2.name}: Score={agent2_result['score']}, Highest={agent2_result['highest_tile']}")
        print(f"  Winner: {winner}")
        
        # Prepare match history
        match_history = [
            (self.agent1.name, f"Game completed: Score={agent1_result['score']}"),
            (self.agent2.name, f"Game completed: Score={agent2_result['score']}")
        ]

        # Per-move history with timing — concatenated from both agents so the
        # tournament manager can populate coder_result.decision_times.
        move_history_with_timing = list(agent1_result['move_history']) + list(agent2_result['move_history'])

        self.current_match_history = match_history

        result = {
            'winner': winner,
            'scores': scores,
            'moves': agent1_result['moves'] + agent2_result['moves'],
            'match_history': match_history,
            'move_history_with_timing': move_history_with_timing,
            'agent1_result': agent1_result,
            'agent2_result': agent2_result
        }
        
        # Store match results for visualization
        self.match_results = result
        
        return result
    
    def _run_agent_game(self, agent: BaseAgent, seed: int, agent_label: str) -> Dict[str, Any]:
        """Run a single agent on a 2048 game."""
        print(f"  {agent_label}: {agent.name}")
        
        # Start agent session for cumulative timeout tracking
        self.start_agent_session(agent)
        
        # Create game state with the shared seed
        game_state = TwentyFortyEightState(seed=seed)
        move_history = []
        
        # Track errors and timeouts
        error_info = None
        timeout_info = None
        
        # Game loop
        while not game_state.is_game_over():
            # Check cumulative timeout
            remaining_time = self.get_remaining_time()
            if remaining_time <= 0:
                print(f"    {agent.name} exceeded cumulative timeout ({self.total_game_timeout:.1f}s)")
                timeout_info = {
                    'timed_out_agent': agent.name,
                    'timeout_at_move': game_state.move_count,
                    'total_time_used': self.total_game_timeout
                }
                break
            
            # Get current observation and action mask
            observation = game_state.get_observation()
            action_mask = game_state.get_action_mask()
            
            # Check if any moves are available
            if not any(action_mask):
                print(f"    No valid moves available - Game Over!")
                break
            
            # Agent selects action
            action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
                agent, observation, action_mask, move_history, [], game_state.move_count
            )
            
            # Handle timeout and errors
            if timeout_result is not None:
                if 'timeout' in timeout_result:
                    print(f"    {agent.name} timed out")
                    timeout_info = {
                        'timed_out_agent': agent.name,
                        'timeout_at_move': game_state.move_count,
                        'decision_time': decision_time
                    }
                elif 'error' in timeout_result:
                    print(f"    {agent.name} encountered error: {timeout_result['error']}")
                    error_info = {
                        'error_agent': agent.name,
                        'error_type': 'runtime_error',
                        'error_message': timeout_result['error']
                    }
                break
            
            # Validate action
            if action is None:
                print(f"    {agent.name} returned None action")
                error_info = {
                    'error_agent': agent.name,
                    'error_type': 'null_action',
                    'error_message': 'Agent returned None action'
                }
                break
            
            if not isinstance(action, int) or action < 0 or action > 3:
                print(f"    Invalid action from {agent.name}: {action}")
                error_info = {
                    'error_agent': agent.name,
                    'error_type': 'invalid_action',
                    'error_message': f'Invalid action: {action}'
                }
                break
            
            # Make the move
            if game_state.make_move(action):
                # Record (player, action, decision_time) so downstream telemetry
                # (coder_result.decision_times, *_report.json) sees real timings.
                move_history.append((agent.name, action, decision_time))

                # Print progress occasionally
                if game_state.move_count % 100 == 0:
                    print(f"    Move {game_state.move_count}: Score={game_state.get_score()}, "
                          f"Highest={game_state.get_highest_tile()}, Time left={remaining_time:.1f}s")
            else:
                # Invalid move attempted
                if game_state.invalid_moves > 10:
                    print(f"    Too many invalid moves ({game_state.invalid_moves})")
                    error_info = {
                        'error_agent': agent.name,
                        'error_type': 'too_many_invalid_moves',
                        'error_message': f'Too many invalid moves: {game_state.invalid_moves}'
                    }
                    break
        
        # Game ended
        final_score = game_state.get_score()
        highest_tile = game_state.get_highest_tile()
        moves = game_state.move_count
        
        print(f"    Game ended: Score={final_score}, Highest tile={highest_tile}, Moves={moves}")
        
        result = {
            'success': game_state.has_won(),
            'score': final_score,
            'highest_tile': highest_tile,
            'moves': moves,
            'move_history': move_history,
            'final_state': game_state,
            'timeout': remaining_time <= 0
        }
        
        # Add error and timeout info if they occurred
        if error_info:
            result['error_info'] = error_info
        if timeout_info:
            result['timeout_info'] = timeout_info
            
        return result
    
    def _compare_agent_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Compare two agent results and determine winner.
        
        Returns:
            Tuple of (winner, scores_dict)
        """
        if not self.agent1 or not self.agent2:
            raise ValueError("Agents not properly initialized")
        agent1_name = self.agent1.name
        agent2_name = self.agent2.name
        
        # Initialize scores
        scores = {agent1_name: 0.0, agent2_name: 0.0}
        
        # Compare by game score (primary criterion)
        if result1['score'] > result2['score']:
            scores[agent1_name] = 1.0
            scores[agent2_name] = 0.0
            winner = agent1_name
        elif result2['score'] > result1['score']:
            scores[agent1_name] = 0.0
            scores[agent2_name] = 1.0
            winner = agent2_name
        else:
            # Same score - compare by highest tile
            if result1['highest_tile'] > result2['highest_tile']:
                scores[agent1_name] = 1.0
                scores[agent2_name] = 0.0
                winner = agent1_name
            elif result2['highest_tile'] > result1['highest_tile']:
                scores[agent1_name] = 0.0
                scores[agent2_name] = 1.0
                winner = agent2_name
            else:
                # Same score and highest tile - compare by moves (fewer is better)
                if result1['moves'] < result2['moves']:
                    scores[agent1_name] = 1.0
                    scores[agent2_name] = 0.0
                    winner = agent1_name
                elif result2['moves'] < result1['moves']:
                    scores[agent1_name] = 0.0
                    scores[agent2_name] = 1.0
                    winner = agent2_name
                else:
                    # Complete tie
                    scores[agent1_name] = 0.5
                    scores[agent2_name] = 0.5
                    winner = 'draw'
        
        return winner, scores
    
    def get_game_rules(self) -> str:
        """Get 2048 game rules description."""
        return """
2048 Game Rules:

1. The game is played on a 4x4 grid
2. Each turn, a new tile (2 or 4) appears randomly on an empty spot
3. You can move tiles in four directions: UP (0), DOWN (1), LEFT (2), RIGHT (3)
4. When two tiles with the same value collide, they merge into one tile with double the value
5. The goal is to create a tile with the value 2048 (or higher)
6. The game ends when no valid moves are available
7. Your score increases by the value of merged tiles

Agent Interface:
- select_action(observation, action_mask) should return an action (0-3)
- observation: 16-element list representing the 4x4 grid (row by row)
- action_mask: 4-element boolean list showing which moves are valid
- Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for 2048."""
        sample_state = TwentyFortyEightState()
        
        return {
            'description': 'Flattened 4x4 grid of tile values',
            'observation_size': 16,
            'sample_observation': sample_state.get_observation(),
            'action_space_size': 4,
            'sample_action_mask': sample_state.get_action_mask(),
            'encoding': '0 for empty tiles, powers of 2 for tile values (2, 4, 8, 16, ...)',
            'action_encoding': '0=UP, 1=DOWN, 2=LEFT, 3=RIGHT',
            'indexing': 'Row-major order: index = row * 4 + col (0-based)',
            'position_mapping': 'Grid positions 0-15 map to cells as: 0-3=row1, 4-7=row2, 8-11=row3, 12-15=row4'
        }
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save a visualization of the 2048 match results.
        
        Creates a side-by-side comparison showing both agents' final boards.
        
        Args:
            save_path: Path where to save the visualization
            
        Returns:
            True if visualization was saved successfully, False otherwise
        """
        if not self.agent1_final_state or not self.agent2_final_state or not self.match_results:
            return False
        
        if not self.agent1 or not self.agent2:
            print("Error: Agents not properly initialized for visualization")
            return False
        
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import matplotlib.colors as mcolors
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Color mapping for tiles
            tile_colors = {
                0: '#CDC1B4',      # Empty
                2: '#EEE4DA',      
                4: '#EDE0C8',
                8: '#F2B179',
                16: '#F59563',
                32: '#F67C5F',
                64: '#F65E3B',
                128: '#EDCF72',
                256: '#EDCC61',
                512: '#EDC850',
                1024: '#EDC53F',
                2048: '#EDC22E',
                4096: '#ED9E1B',
                8192: '#ED8C0A'
            }
            
            def draw_board(ax, state, agent_name):
                """Draw a 2048 board."""
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 4)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                
                # Draw grid
                for i in range(5):
                    ax.axhline(i, color='#BBADA0', linewidth=2)
                    ax.axvline(i, color='#BBADA0', linewidth=2)
                
                # Fill background
                background = patches.Rectangle((0, 0), 4, 4, facecolor='#BBADA0')
                ax.add_patch(background)
                
                # Draw tiles
                for row in range(4):
                    for col in range(4):
                        value = state.board[row][col]
                        
                        # Get color
                        if value in tile_colors:
                            color = tile_colors[value]
                        else:
                            color = '#3C3A32'  # Very high values
                        
                        # Draw tile
                        tile = patches.Rectangle((col + 0.05, row + 0.05), 0.9, 0.9,
                                               facecolor=color, edgecolor='none')
                        ax.add_patch(tile)
                        
                        # Draw text
                        if value > 0:
                            # Text color
                            text_color = '#776E65' if value <= 4 else '#F9F6F2'
                            # Font size based on number of digits
                            if value < 100:
                                fontsize = 24
                            elif value < 1000:
                                fontsize = 20
                            else:
                                fontsize = 16
                            
                            ax.text(col + 0.5, row + 0.5, str(value),
                                   ha='center', va='center', fontsize=fontsize,
                                   weight='bold', color=text_color)
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add title with game info
                score = state.get_score()
                highest = state.get_highest_tile()
                moves = state.move_count
                
                title = f'{agent_name}\nScore: {score} | Highest: {highest} | Moves: {moves}'
                ax.set_title(title, fontsize=14, weight='bold', pad=10)
            
            # Draw both boards
            draw_board(ax1, self.agent1_final_state, self.agent1.name)
            draw_board(ax2, self.agent2_final_state, self.agent2.name)
            
            # Add overall match title
            winner = self.match_results['winner']
            if winner == 'draw':
                match_title = '2048 Match - Draw'
            else:
                match_title = f'2048 Match - Winner: {winner}'
            fig.suptitle(match_title, fontsize=18, weight='bold')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error saving 2048 visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def handle_agent_move_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any, 
                                       move_history: List, move_history_with_timing: List, 
                                       game_step: int) -> Tuple[Any, float, Optional[Dict]]:
        """
        Handle agent move with timeout detection for 2048.
        
        Returns:
            Tuple of (action, decision_time, timeout_result)
        """
        # Check cumulative timeout before calling agent
        remaining_time = self.get_remaining_time()
        if remaining_time <= 0:
            print(f"{agent.name} exceeded cumulative timeout ({self.total_game_timeout:.1f}s)")
            return None, 0.0, {'timeout': True}
        
        # Call agent with timeout
        action, decision_time, timed_out, error_message = self.call_agent_with_cumulative_timeout(agent, observation, action_mask)
        
        if timed_out:
            return None, decision_time, {'timeout': True}
        
        if error_message:
            return None, decision_time, {'error': error_message}
        
        return action, decision_time, None
    
    def reset(self) -> None:
        """Reset game state for a new match."""
        super().reset()
        self.agent1_final_state = None
        self.agent2_final_state = None
        self.shared_seed = None
        self.match_results = None 