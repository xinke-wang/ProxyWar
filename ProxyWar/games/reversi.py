"""
Reversi (Othello) game implementation for ProxyWar framework.

This module implements the classic Reversi/Othello game where two players
take turns placing pieces to capture opponent pieces by surrounding them.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

from .base import TwoPlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


class ReversiState:
    """Represents the state of a Reversi game."""
    
    def __init__(self, board_size: int = 8):
        """
        Initialize Reversi game state.
        
        Args:
            board_size: Size of the square board (default 8x8)
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        
        # Initialize starting position
        # Convention: 0=empty, 1=black, 2=white
        center = board_size // 2
        self.board[center-1, center-1] = 2  # White
        self.board[center-1, center] = 1    # Black
        self.board[center, center-1] = 1    # Black
        self.board[center, center] = 2      # White
        
        self.current_player = 1  # Black plays first
        self.turn_count = 0
        self.game_over = False
        self.winner = None
        
        # Track consecutive passes
        self.consecutive_passes = 0
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        self.directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
    
    def get_board_copy(self) -> np.ndarray:
        """Get a copy of the current board state."""
        return self.board.copy()
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board boundaries."""
        return 0 <= row < self.board_size and 0 <= col < self.board_size
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if position is empty."""
        return self.board[row, col] == 0
    
    def get_opponent(self, player: int) -> int:
        """Get opponent player number."""
        return 2 if player == 1 else 1
    
    def find_captures_in_direction(self, row: int, col: int, direction: Tuple[int, int], 
                                   player: int) -> List[Tuple[int, int]]:
        """
        Find pieces that would be captured in a specific direction.
        
        Args:
            row, col: Starting position
            direction: Direction tuple (dr, dc)
            player: Current player (1 or 2)
            
        Returns:
            List of positions that would be captured
        """
        dr, dc = direction
        captures = []
        r, c = row + dr, col + dc
        
        # Look for opponent pieces
        while self.is_valid_position(r, c) and self.board[r, c] == self.get_opponent(player):
            captures.append((r, c))
            r += dr
            c += dc
        
        # Check if we end with our own piece (valid capture)
        if (self.is_valid_position(r, c) and 
            self.board[r, c] == player and 
            len(captures) > 0):
            return captures
        
        return []
    
    def get_all_captures(self, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """Get all pieces that would be captured by placing a piece at (row, col)."""
        if not self.is_valid_position(row, col) or not self.is_empty(row, col):
            return []
        
        all_captures = []
        for direction in self.directions:
            captures = self.find_captures_in_direction(row, col, direction, player)
            all_captures.extend(captures)
        
        return all_captures
    
    def is_legal_move(self, row: int, col: int, player: int) -> bool:
        """Check if a move is legal for the given player."""
        if not self.is_valid_position(row, col) or not self.is_empty(row, col):
            return False
        
        # A move is legal if it captures at least one opponent piece
        return len(self.get_all_captures(row, col, player)) > 0
    
    def get_legal_moves(self, player: int) -> List[Tuple[int, int]]:
        """Get all legal moves for the given player."""
        legal_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_legal_move(row, col, player):
                    legal_moves.append((row, col))
        return legal_moves
    
    def make_move(self, row: int, col: int, player: int) -> bool:
        """
        Make a move on the board.
        
        Args:
            row, col: Position to place piece
            player: Player making the move (1 or 2)
            
        Returns:
            True if move was successful, False otherwise
        """
        if not self.is_legal_move(row, col, player):
            return False
        
        # Place the piece
        self.board[row, col] = player
        
        # Capture opponent pieces
        captures = self.get_all_captures(row, col, player)
        for capture_row, capture_col in captures:
            self.board[capture_row, capture_col] = player
        
        # Update game state
        self.turn_count += 1
        self.current_player = self.get_opponent(player)
        self.consecutive_passes = 0
        
        # Check if game is over
        self._check_game_over()
        
        return True
    
    def pass_turn(self) -> bool:
        """Pass the turn (when no legal moves available)."""
        self.consecutive_passes += 1
        self.current_player = self.get_opponent(self.current_player)
        
        # Game ends if both players pass consecutively
        if self.consecutive_passes >= 2:
            self._end_game()
            return True
        
        return False
    
    def _check_game_over(self) -> None:
        """Check if the game is over and determine winner."""
        # Check if board is full
        if np.sum(self.board == 0) == 0:
            self._end_game()
            return
        
        # Check if current player has legal moves
        if len(self.get_legal_moves(self.current_player)) == 0:
            # If no legal moves, pass turn
            self.pass_turn()
            # After passing, check if the new current player has moves
            # This prevents the need for recursive calls and handles double-pass scenario
            if not self.game_over and len(self.get_legal_moves(self.current_player)) == 0:
                # Both players have no moves, game should end
                self.pass_turn()  # This will trigger game end due to consecutive passes
    
    def _end_game(self) -> None:
        """End the game and determine winner."""
        self.game_over = True
        
        # Count pieces
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == 2)
        
        if black_count > white_count:
            self.winner = 1  # Black wins
        elif white_count > black_count:
            self.winner = 2  # White wins
        else:
            self.winner = 0  # Draw
    
    def get_scores(self) -> Dict[str, int]:
        """Get current scores (piece counts)."""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == 2)
        return {
            'black': black_count,
            'white': white_count,
            'empty': self.board_size * self.board_size - black_count - white_count
        }
    
    def get_observation(self) -> List[int]:
        """Get flattened board observation."""
        return self.board.flatten().tolist()
    
    def get_action_mask(self, player: int) -> List[bool]:
        """Get action mask indicating legal moves."""
        mask = [False] * (self.board_size * self.board_size)
        legal_moves = self.get_legal_moves(player)
        
        for row, col in legal_moves:
            idx = row * self.board_size + col
            mask[idx] = True
        
        return mask
    
    def action_to_position(self, action: int) -> Tuple[int, int]:
        """Convert action index to board position."""
        row = action // self.board_size
        col = action % self.board_size
        return row, col
    
    def position_to_action(self, row: int, col: int) -> int:
        """Convert board position to action index."""
        return row * self.board_size + col
    
    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(self.board_size)))
        for i in range(self.board_size):
            row_str = f"{i} "
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    row_str += ". "
                elif self.board[i, j] == 1:
                    row_str += "● "  # Black
                else:
                    row_str += "○ "  # White
            lines.append(row_str)
        return "\n".join(lines)


@GAME_REGISTRY.register('reversi')
class ReversiGame(TwoPlayerGame):
    """
    Reversi (Othello) game implementation.
    
    This class manages Reversi matches between two agents on an 8x8 board.
    """
    
    def __init__(self, movement_timeout: float = 30.0):
        """Initialize Reversi game."""
        super().__init__("Reversi", movement_timeout)
        self.game_state = None
        self.match_history = []
        self.final_state = None
        
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """
        Setup a Reversi match between two agents.
        
        Args:
            agent1: First agent (plays as Black)
            agent2: Second agent (plays as White)
        """
        super().setup_match(agent1, agent2)
        
        # Create new game state
        self.game_state = ReversiState(board_size=8)
        self.match_history = []
        self.final_state = None
        
        print(f"Reversi match setup: {agent1.name} (Black) vs {agent2.name} (White)")
        print("Board: 8x8, Black plays first")
        print("Actions: 0-63 (row*8 + col)")
    
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete Reversi match.
        
        Returns:
            Dictionary containing match results
        """
        if not self.game_state:
            raise ValueError("Match not properly setup. Call setup_match() first.")
        
        move_history = []
        move_history_with_timing = []
        game_step = 0
        
        print("Reversi game started...")
        print(f"Initial board:\n{self.game_state}")
        
        while not self.game_state.game_over and game_step < 100:  # Max 100 moves
            current_player = self.game_state.current_player
            current_agent = self.agent1 if current_player == 1 else self.agent2
            
            print(f"\nTurn {game_step + 1}: {current_agent.name} ({'Black' if current_player == 1 else 'White'})")
            
            # Get observation and action mask
            observation = self.game_state.get_observation()
            action_mask = self.game_state.get_action_mask(current_player)
            
            # Check if player has legal moves
            if not any(action_mask):
                print(f"No legal moves for {current_agent.name}, passing turn")
                self.game_state.pass_turn()
                move_history.append((current_agent.name, "pass"))
                move_history_with_timing.append((current_agent.name, "pass", 0.0))
                continue
            
            # Get action from agent
            action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
                current_agent, observation, action_mask, move_history, move_history_with_timing, game_step
            )
            
            if timeout_result is not None:
                return timeout_result
            
            # Validate and execute move
            if action is None or not isinstance(action, int):
                print(f"Invalid action from {current_agent.name}: {action}")
                # Force pass
                self.game_state.pass_turn()
                move_history.append((current_agent.name, "invalid_pass"))
                move_history_with_timing.append((current_agent.name, "invalid_pass", decision_time))
                continue
            
            # Convert action to position
            row, col = self.game_state.action_to_position(action)
            
            # Check if move is legal
            if not self.game_state.is_legal_move(row, col, current_player):
                print(f"Illegal move from {current_agent.name}: ({row}, {col})")
                # Force pass
                self.game_state.pass_turn()
                move_history.append((current_agent.name, f"illegal_pass_{action}"))
                move_history_with_timing.append((current_agent.name, f"illegal_pass_{action}", decision_time))
                continue
            
            # Make the move
            success = self.game_state.make_move(row, col, current_player)
            if success:
                print(f"{current_agent.name} plays at ({row}, {col}) in {decision_time:.2f}s")
                scores = self.game_state.get_scores()
                print(f"Current scores: Black={scores['black']}, White={scores['white']}")
                
                move_history.append((current_agent.name, action))
                move_history_with_timing.append((current_agent.name, action, decision_time))
                
                game_step += 1
                
                # Show board state
                print(f"Board after move:\n{self.game_state}")
            else:
                print(f"Failed to make move for {current_agent.name}")
                break
        
        # Store final state
        self.final_state = self.game_state.get_board_copy()
        
        # Determine winner
        scores = self.game_state.get_scores()
        if self.game_state.winner == 1:
            winner = self.agent1.name
            final_scores = {self.agent1.name: 1.0, self.agent2.name: 0.0}
        elif self.game_state.winner == 2:
            winner = self.agent2.name
            final_scores = {self.agent1.name: 0.0, self.agent2.name: 1.0}
        else:
            winner = "draw"
            final_scores = {self.agent1.name: 0.5, self.agent2.name: 0.5}
        
        print(f"\nGAME OVER!")
        print(f"Final scores: Black={scores['black']}, White={scores['white']}")
        print(f"Winner: {winner}")
        
        self.current_match_history = move_history
        
        result = {
            'winner': winner,
            'scores': final_scores,
            'moves': len(move_history),
            'match_history': move_history,
            'move_history_with_timing': move_history_with_timing,
            'final_board': self.final_state,
            'piece_counts': scores
        }
        
        return result
    
    def get_game_rules(self) -> str:
        """Get Reversi game rules description."""
        return """
Reversi (Othello) Game Rules:

1. The game is played on an 8x8 board
2. Two players (Black and White) take turns placing pieces
3. Black plays first
4. A move must capture at least one opponent piece by surrounding it
5. Captured pieces are flipped to the current player's color
6. If a player cannot make a legal move, they pass their turn
7. The game ends when:
   - The board is full, or
   - Both players pass consecutively (no legal moves for either)
8. The player with the most pieces wins

Agent Interface:
- select_action(observation, action_mask) should return position index (0-63)
- observation: 64-element flattened list of the 8x8 board
- Board encoding: 0=empty, 1=black, 2=white
- Actions: 0-63 where action = row*8 + col
- action_mask indicates legal moves (True) or illegal (False)
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for Reversi."""
        sample_state = ReversiState(board_size=8)
        
        return {
            'description': 'Flattened 8x8 Reversi board',
            'observation_size': 64,
            'sample_observation': sample_state.get_observation(),
            'action_space_size': 64,
            'sample_action_mask': sample_state.get_action_mask(1),
            'encoding': '0=empty, 1=black, 2=white',
            'action_format': 'Integer 0-63 where action = row*8 + col',
            'indexing': 'Row-major order: [0,1,2,3,4,5,6,7] = row 0, [8,9,10,11,12,13,14,15] = row 1, etc.',
            'board_size': '8x8',
            'initial_setup': 'Center 4 positions: (3,3)=White, (3,4)=Black, (4,3)=Black, (4,4)=White',
            'position_mapping': 'Actions 0-63 map to board positions as: action = row*8 + col. E.g., action 0=pos(0,0), action 7=pos(0,7), action 8=pos(1,0), action 63=pos(7,7). Must capture opponent pieces to make legal move.'
        }
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save a visualization of the Reversi match results.
        
        Args:
            save_path: Path where to save the visualization
            
        Returns:
            True if visualization was saved successfully, False otherwise
        """
        if self.final_state is None:
            return False
        
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Draw the board
            board_size = 8
            board = self.final_state
            
            # Set up the plot with green background (traditional board color)
            ax.set_xlim(0, board_size)
            ax.set_ylim(0, board_size)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left
            ax.set_facecolor('#228B22')  # Forest green background
            
            # Draw grid
            for i in range(board_size + 1):
                ax.axhline(i, color='black', linewidth=2)
                ax.axvline(i, color='black', linewidth=2)
            
            # Draw pieces
            for row in range(board_size):
                for col in range(board_size):
                    if board[row, col] == 1:  # Black
                        circle = plt.Circle((col + 0.5, row + 0.5), 0.4, color='black', 
                                          edgecolor='white', linewidth=1)
                        ax.add_patch(circle)
                    elif board[row, col] == 2:  # White
                        circle = plt.Circle((col + 0.5, row + 0.5), 0.4, color='white', 
                                          edgecolor='black', linewidth=2)
                        ax.add_patch(circle)
            
            # Add labels
            ax.set_xticks(range(board_size))
            ax.set_yticks(range(board_size))
            ax.set_xticklabels([str(i) for i in range(board_size)])
            ax.set_yticklabels([str(i) for i in range(board_size)])
            
            # Add title with game result
            if hasattr(self, 'game_state') and self.game_state:
                scores = self.game_state.get_scores()
                title = f"Reversi Match Result\n"
                title += f"{self.agent1.name} (Black): {scores['black']} pieces\n"
                title += f"{self.agent2.name} (White): {scores['white']} pieces"
                
                if self.game_state.winner == 1:
                    title += f"\nWinner: {self.agent1.name} (Black)"
                elif self.game_state.winner == 2:
                    title += f"\nWinner: {self.agent2.name} (White)"
                else:
                    title += "\nResult: Draw"
                
                ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error saving Reversi visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset(self) -> None:
        """Reset game state for a new match."""
        super().reset()
        self.game_state = None
        self.match_history = []
        self.final_state = None 