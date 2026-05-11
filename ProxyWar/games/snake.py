"""
Snake game implementation for ProxyWar framework.

This module implements a two-player Snake game where both players
control their snakes simultaneously on a 10x10 grid.
"""

import numpy as np
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .base import TwoPlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


@dataclass
class Position:
    """Represents a position on the game board."""
    x: int
    y: int
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class Snake:
    """Represents a snake in the game."""
    
    def __init__(self, start_pos: Position, player_id: int):
        self.body = [start_pos]
        self.player_id = player_id
        self.direction = 0  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.alive = True
        self.score = 0
        self.length = 1
        self.last_direction = 0
        
    def get_head(self) -> Position:
        """Get the head position of the snake."""
        return self.body[0]
    
    def get_next_position(self, direction: int) -> Position:
        """Get the next position if moving in the given direction."""
        head = self.get_head()
        
        if direction == 0:  # UP
            return Position(head.x, head.y - 1)
        elif direction == 1:  # DOWN
            return Position(head.x, head.y + 1)
        elif direction == 2:  # LEFT
            return Position(head.x - 1, head.y)
        elif direction == 3:  # RIGHT
            return Position(head.x + 1, head.y)
        else:
            return head
    
    def is_valid_direction(self, direction: int) -> bool:
        """Check if the direction is valid (not opposite to current direction)."""
        if len(self.body) == 1:
            return True
        
        # Can't move in opposite direction
        opposite_dirs = {0: 1, 1: 0, 2: 3, 3: 2}
        return direction != opposite_dirs.get(self.direction, -1)
    
    def move(self, direction: int, grow: bool = False) -> bool:
        """Move the snake in the given direction."""
        if not self.alive:
            return False
        
        # Validate direction
        if not self.is_valid_direction(direction):
            direction = self.direction  # Keep current direction
        
        next_pos = self.get_next_position(direction)
        
        # Add new head
        self.body.insert(0, next_pos)
        
        # Remove tail if not growing
        if not grow:
            self.body.pop()
        else:
            self.length += 1
            self.score += 10
        
        self.direction = direction
        self.last_direction = direction
        return True


class SnakeGameState:
    """Represents the state of a Snake game."""
    
    def __init__(self, width: int = 10, height: int = 10, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=int)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize snakes
        self.snake1 = Snake(Position(2, 5), 1)  # Player 1
        self.snake2 = Snake(Position(7, 5), 2)  # Player 2
        
        # Game state
        self.food_positions = []
        self.max_food_count = 5  # Maximum number of food items on the board
        self.turn = 0
        self.max_turns = 1000  # Maximum turns before declaring draw
        
        # Generate initial food
        self.generate_food()
        self.update_board()
    
    def generate_food(self) -> None:
        """Generate food items to maintain the maximum count."""
        while len(self.food_positions) < self.max_food_count:
            empty_positions = []
            
            for y in range(self.height):
                for x in range(self.width):
                    pos = Position(x, y)
                    if not self.is_position_occupied(pos) and pos not in self.food_positions:
                        empty_positions.append(pos)
            
            if empty_positions:
                new_food = random.choice(empty_positions)
                self.food_positions.append(new_food)
            else:
                break  # No more empty positions available
    
    def is_position_occupied(self, pos: Position) -> bool:
        """Check if a position is occupied by any snake body."""
        for body_pos in self.snake1.body:
            if pos == body_pos:
                return True
        for body_pos in self.snake2.body:
            if pos == body_pos:
                return True
        return False
    
    def is_position_in_bounds(self, pos: Position) -> bool:
        """Check if a position is within the game boundaries."""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def check_collision(self, snake: Snake, new_pos: Position) -> bool:
        """Check if the snake collides with walls or other snakes."""
        # Check bounds
        if not self.is_position_in_bounds(new_pos):
            return True
        
        # Check collision with own body (excluding head)
        for body_pos in snake.body[1:]:
            if new_pos == body_pos:
                return True
        
        # Check collision with other snake
        other_snake = self.snake2 if snake.player_id == 1 else self.snake1
        if other_snake.alive:
            for body_pos in other_snake.body:
                if new_pos == body_pos:
                    return True
        
        return False
    
    def update_board(self) -> None:
        """Update the board representation."""
        self.board.fill(0)
        
        # Place snake bodies
        if self.snake1.alive:
            for pos in self.snake1.body:
                if self.is_position_in_bounds(pos):
                    self.board[pos.y, pos.x] = 1
        
        if self.snake2.alive:
            for pos in self.snake2.body:
                if self.is_position_in_bounds(pos):
                    self.board[pos.y, pos.x] = 2
        
        # Place food
        for food_pos in self.food_positions:
            if self.is_position_in_bounds(food_pos):
                self.board[food_pos.y, food_pos.x] = 3
    
    def step(self, action1: int, action2: int) -> Dict[str, Any]:
        """Execute one step of the game."""
        self.turn += 1
        
        # Determine the actual directions to use (considering invalid directions)
        actual_direction1 = action1 if self.snake1.is_valid_direction(action1) else self.snake1.direction
        actual_direction2 = action2 if self.snake2.is_valid_direction(action2) else self.snake2.direction
        
        # Store original positions for collision detection using actual directions
        snake1_next = self.snake1.get_next_position(actual_direction1)
        snake2_next = self.snake2.get_next_position(actual_direction2)
        
        # Check for head-to-head collision
        if self.snake1.alive and self.snake2.alive and snake1_next == snake2_next:
            self.snake1.alive = False
            self.snake2.alive = False
            # Set directions to the actual directions used for collision
            self.snake1.direction = actual_direction1
            self.snake2.direction = actual_direction2
            return {
                'game_over': True,
                'winner': 'draw',
                'reason': 'Head-to-head collision'
            }
        
        # Check collisions for both snakes simultaneously (fair judgment)
        snake1_collision = self.snake1.alive and self.check_collision(self.snake1, snake1_next)
        snake2_collision = self.snake2.alive and self.check_collision(self.snake2, snake2_next)
        
        # Handle collisions
        rewards = {'player1': 0, 'player2': 0}
        
        # Process snake 1
        if self.snake1.alive:
            if snake1_collision:
                self.snake1.alive = False
                # Update direction to show what caused the collision
                if self.snake1.is_valid_direction(action1):
                    self.snake1.direction = action1
                else:
                    self.snake1.direction = actual_direction1
            else:
                # Check if eating food
                eat_food = snake1_next in self.food_positions
                self.snake1.move(actual_direction1, grow=eat_food)
                
                if eat_food:
                    rewards['player1'] = 10
                    self.food_positions.remove(snake1_next)
                    self.generate_food()
        
        # Process snake 2
        if self.snake2.alive:
            if snake2_collision:
                self.snake2.alive = False
                # Update direction to show what caused the collision
                if self.snake2.is_valid_direction(action2):
                    self.snake2.direction = action2
                else:
                    self.snake2.direction = actual_direction2
            else:
                # Check if eating food
                eat_food = snake2_next in self.food_positions
                self.snake2.move(actual_direction2, grow=eat_food)
                
                if eat_food:
                    rewards['player2'] = 10
                    self.food_positions.remove(snake2_next)
                    self.generate_food()
        
        # Update board
        self.update_board()
        
        # Check game over conditions
        if not self.snake1.alive and not self.snake2.alive:
            # Compare scores when both snakes die
            if self.snake1.score > self.snake2.score:
                winner = 'player1'
                reason = f'Both snakes died - Player 1 wins by score ({self.snake1.score} vs {self.snake2.score})'
            elif self.snake2.score > self.snake1.score:
                winner = 'player2'
                reason = f'Both snakes died - Player 2 wins by score ({self.snake2.score} vs {self.snake1.score})'
            else:
                winner = 'draw'
                reason = f'Both snakes died with equal scores ({self.snake1.score})'
            
            return {
                'game_over': True,
                'winner': winner,
                'reason': reason
            }
        elif not self.snake1.alive:
            return {
                'game_over': True,
                'winner': 'player2',
                'reason': 'Player 1 snake died'
            }
        elif not self.snake2.alive:
            return {
                'game_over': True,
                'winner': 'player1',
                'reason': 'Player 2 snake died'
            }
        elif self.turn >= self.max_turns:
            # Determine winner by score
            if self.snake1.score > self.snake2.score:
                winner = 'player1'
            elif self.snake2.score > self.snake1.score:
                winner = 'player2'
            else:
                winner = 'draw'
            
            return {
                'game_over': True,
                'winner': winner,
                'reason': f'Maximum turns reached ({self.max_turns})'
            }
        
        return {
            'game_over': False,
            'rewards': rewards
        }
    
    def get_observation(self) -> List[int]:
        """Get the current game state as a flattened observation."""
        return self.board.flatten().tolist()
    
    def get_action_mask(self, player_id: int) -> List[bool]:
        """Get the action mask for a specific player."""
        snake = self.snake1 if player_id == 1 else self.snake2
        
        if not snake.alive:
            return [False, False, False, False]
        
        # All directions are potentially valid, invalid directions will be ignored
        return [True, True, True, True]
    
    def get_scores(self) -> Dict[str, int]:
        """Get current scores for both players."""
        return {
            'player1': self.snake1.score,
            'player2': self.snake2.score
        }
    
    def get_game_info(self) -> Dict[str, Any]:
        """Get comprehensive game information."""
        return {
            'turn': self.turn,
            'max_turns': self.max_turns,
            'snake1_length': len(self.snake1.body),
            'snake2_length': len(self.snake2.body),
            'snake1_alive': self.snake1.alive,
            'snake2_alive': self.snake2.alive,
            'scores': self.get_scores(),
            'food_positions': [(pos.x, pos.y) for pos in self.food_positions],
            'food_count': len(self.food_positions)
        }


@GAME_REGISTRY.register('snake')
class SnakeGame(TwoPlayerGame):
    """
    Two-player Snake game implementation.
    
    This class manages Snake matches between two agents on a 10x10 grid.
    """
    
    def __init__(self, movement_timeout: float = 5.0):
        """Initialize Snake game."""
        super().__init__("Snake", movement_timeout)
        self.game_state = None
        self.match_history = []
        self.final_state = None
        
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """
        Setup a Snake match between two agents.
        
        Args:
            agent1: First agent (controls snake 1)
            agent2: Second agent (controls snake 2)
        """
        super().setup_match(agent1, agent2)
        
        # Create new game state
        self.game_state = SnakeGameState(width=10, height=10)
        self.match_history = []
        self.final_state = None
        
        print(f"Snake match setup: {agent1.name} vs {agent2.name}")
        print("Game board: 10x10")
        print("Controls: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT")
    
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete Snake match.
        
        Returns:
            Dictionary containing match results
        """
        if not self.game_state:
            raise ValueError("Match not properly setup. Call setup_match() first.")
        
        move_history = []
        move_history_with_timing = []
        game_step = 0
        
        print("Snake game started...")
        
        while True:
            # Get observations for both players
            observation = self.game_state.get_observation()
            action_mask1 = self.game_state.get_action_mask(1)
            action_mask2 = self.game_state.get_action_mask(2)
            
            # Get actions from both agents
            action1, time1, timeout1 = self.handle_agent_move_with_timeout(
                self.agent1, observation, action_mask1, move_history, move_history_with_timing, game_step
            )
            
            if timeout1 is not None:
                return timeout1
            
            action2, time2, timeout2 = self.handle_agent_move_with_timeout(
                self.agent2, observation, action_mask2, move_history, move_history_with_timing, game_step
            )
            
            if timeout2 is not None:
                return timeout2
            
            # Use default action if agent returned None
            if action1 is None:
                action1 = self.game_state.snake1.direction
            if action2 is None:
                action2 = self.game_state.snake2.direction
            
            # Execute game step
            result = self.game_state.step(action1, action2)
            
            # Record moves
            move_history.append((self.agent1.name, action1))
            move_history.append((self.agent2.name, action2))
            move_history_with_timing.append((self.agent1.name, action1, time1))
            move_history_with_timing.append((self.agent2.name, action2, time2))
            
            game_step += 1
            
            # Print progress occasionally
            if game_step % 100 == 0:
                game_info = self.game_state.get_game_info()
                print(f"Turn {game_step}: Snake1 len={game_info['snake1_length']}, "
                      f"Snake2 len={game_info['snake2_length']}, "
                      f"Scores: {game_info['scores']}")
            
            # Check if game is over
            if result['game_over']:
                print(f"Game over after {game_step} turns: {result['reason']}")
                break
        
        # Store final state
        self.final_state = self.game_state
        
        # Determine final results
        final_scores = self.game_state.get_scores()
        winner_key = result['winner']
        
        if winner_key == 'player1':
            winner = self.agent1.name
            scores = {self.agent1.name: 1.0, self.agent2.name: 0.0}
        elif winner_key == 'player2':
            winner = self.agent2.name
            scores = {self.agent1.name: 0.0, self.agent2.name: 1.0}
        else:
            winner = 'draw'
            scores = {self.agent1.name: 0.5, self.agent2.name: 0.5}
        
        self.current_match_history = move_history
        
        match_result = {
            'winner': winner,
            'scores': scores,
            'moves': len(move_history),
            'match_history': move_history,
            'move_history_with_timing': move_history_with_timing,
            'final_scores': final_scores,
            'turns': game_step,
            'reason': result['reason']
        }
        
        print(f"Final result: {winner}")
        print(f"Final scores: {final_scores}")
        
        return match_result
    
    def get_game_rules(self) -> str:
        """Get Snake game rules description."""
        return """
Snake Game Rules:

1. The game is played on a 10x10 grid
2. Each player controls a snake that moves automatically
3. Players can change their snake's direction using actions, but invalid direction changes (opposite direction to current direction) are ignored
4. Snakes grow longer when they eat food (represented by 3 on the board)
5. The game ends when:
   - A snake hits a wall (boundary)
   - A snake hits its own body
   - A snake hits the other snake's body
   - Both snakes collide head-to-head (draw)
   - Maximum turns (1000) are reached
6. Winner is determined by:
   - Last snake alive wins
   - If both die simultaneously, winner is determined by score (food eaten)
   - If both die with equal scores, it's a draw
   - If max turns reached, higher score wins

Agent Interface:
- select_action(observation, action_mask) should return direction (0-3)
- observation: 100-element flattened list of the 10x10 board
- Board encoding: 0=empty, 1=player1_snake, 2=player2_snake, 3=food
- Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

Note: Snakes move automatically each turn. You only control direction changes.
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for Snake."""
        # Create a sample game state
        sample_state = SnakeGameState(width=10, height=10, seed=42)
        sample_obs = sample_state.get_observation()
        
        return {
            'description': 'Flattened 10x10 Snake game board',
            'observation_size': 100,
            'sample_observation': sample_obs[:10] + [-1] + sample_obs[-10:],  # Use -1 as separator instead of '...'
            'action_space_size': 4,
            'sample_action_mask': [True, True, True, True],
            'encoding': '0=empty, 1=player1_snake, 2=player2_snake, 3=food',
            'actions': '0=UP, 1=DOWN, 2=LEFT, 3=RIGHT',
            'board_size': '10x10 (100 cells total)',
            'indexing': 'Row-major order: index = row * 10 + col',
            'position_mapping': 'Grid positions 0-99 map to cells as: 0-9=row0, 10-19=row1, ..., 90-99=row9. Actions 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT control snake direction.',
            'note': 'Snakes move automatically. You only control direction changes.'
        }
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save a visualization of the Snake game.
        
        Args:
            save_path: Path where to save the visualization
            
        Returns:
            True if visualization was saved successfully, False otherwise
        """
        if not self.final_state:
            return False
        
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            
            # Get the final board state
            board = self.final_state.board
            
            # Create a colored representation
            # 0=white (empty), 1=red (player1), 2=blue (player2), 3=green (food)
            color_map = np.zeros((board.shape[0], board.shape[1], 3), dtype=float)
            
            # Empty cells - white
            color_map[board == 0] = [1.0, 1.0, 1.0]
            
            # Food - green (from board)
            color_map[board == 3] = [0.0, 1.0, 0.0]
            
            # Draw snakes directly from their body positions (regardless of alive status)
            # Player 1 snake - red
            if self.final_state.snake1.body:
                for pos in self.final_state.snake1.body:
                    if 0 <= pos.y < board.shape[0] and 0 <= pos.x < board.shape[1]:
                        if self.final_state.snake1.alive:
                            color_map[pos.y, pos.x] = [1.0, 0.0, 0.0]  # Bright red for alive
                        else:
                            color_map[pos.y, pos.x] = [0.7, 0.0, 0.0]  # Darker red for dead
            
            # Player 2 snake - blue
            if self.final_state.snake2.body:
                for pos in self.final_state.snake2.body:
                    if 0 <= pos.y < board.shape[0] and 0 <= pos.x < board.shape[1]:
                        if self.final_state.snake2.alive:
                            color_map[pos.y, pos.x] = [0.0, 0.0, 1.0]  # Bright blue for alive
                        else:
                            color_map[pos.y, pos.x] = [0.0, 0.0, 0.7]  # Darker blue for dead
            
            # Display the board
            ax.imshow(color_map, interpolation='nearest')
            
            # Add grid lines
            for i in range(board.shape[0] + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.5)
            for j in range(board.shape[1] + 1):
                ax.axvline(j - 0.5, color='gray', linewidth=0.5)
            
            # Mark snake heads with circles and direction triangles
            def create_direction_triangle(x, y, direction, size=0.2):
                """Create a triangle pointing in the given direction."""
                # Direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
                # Note: In matplotlib, y=0 is at top, y increases downward
                if direction == 0:  # UP (y decreases in game logic)
                    triangle = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]])
                elif direction == 1:  # DOWN (y increases in game logic)  
                    triangle = np.array([[x, y + size], [x - size, y - size], [x + size, y - size]])
                elif direction == 2:  # LEFT (x decreases)
                    triangle = np.array([[x - size, y], [x + size, y - size], [x + size, y + size]])
                elif direction == 3:  # RIGHT (x increases)
                    triangle = np.array([[x + size, y], [x - size, y - size], [x - size, y + size]])
                else:
                    triangle = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]])  # Default UP
                return triangle
            
            if self.final_state.snake1.body:
                head1 = self.final_state.snake1.body[0]
                if self.final_state.snake1.alive:
                    circle1 = patches.Circle((head1.x, head1.y), 0.3, color='darkred', fill=True)
                    triangle_color = 'yellow'
                else:
                    circle1 = patches.Circle((head1.x, head1.y), 0.3, color='maroon', fill=True)  # Darker for dead
                    triangle_color = 'orange'
                ax.add_patch(circle1)
                
                # Add direction triangle
                triangle1_points = create_direction_triangle(head1.x, head1.y, self.final_state.snake1.direction)
                triangle1 = patches.Polygon(triangle1_points, closed=True, fill=True, color=triangle_color)
                ax.add_patch(triangle1)
            
            if self.final_state.snake2.body:
                head2 = self.final_state.snake2.body[0]
                if self.final_state.snake2.alive:
                    circle2 = patches.Circle((head2.x, head2.y), 0.3, color='darkblue', fill=True)
                    triangle_color = 'yellow'
                else:
                    circle2 = patches.Circle((head2.x, head2.y), 0.3, color='navy', fill=True)  # Darker for dead
                    triangle_color = 'orange'
                ax.add_patch(circle2)
                
                # Add direction triangle
                triangle2_points = create_direction_triangle(head2.x, head2.y, self.final_state.snake2.direction)
                triangle2 = patches.Polygon(triangle2_points, closed=True, fill=True, color=triangle_color)
                ax.add_patch(triangle2)
            
            # Set title and labels
            game_info = self.final_state.get_game_info()
            title = f"Snake Game - Turn {game_info['turn']}\n"
            title += f"{self.agent1.name} (Red): Score {game_info['scores']['player1']}, Length {game_info['snake1_length']}\n"
            title += f"{self.agent2.name} (Blue): Score {game_info['scores']['player2']}, Length {game_info['snake2_length']}"
            
            ax.set_title(title, fontsize=12, pad=20)
            ax.set_xlim(-0.5, board.shape[1] - 0.5)
            ax.set_ylim(-0.5, board.shape[0] - 0.5)
            ax.set_aspect('equal')
            ax.set_xticks(range(board.shape[1]))
            ax.set_yticks(range(board.shape[0]))
            
            # Add legend
            legend_elements = [
                patches.Patch(color='red', label=f'{self.agent1.name} Snake'),
                patches.Patch(color='blue', label=f'{self.agent2.name} Snake'),
                patches.Patch(color='green', label='Food'),
                patches.Patch(color='white', label='Empty', edgecolor='black')
            ]
            
            # Add status information to legend
            if not self.final_state.snake1.alive:
                legend_elements[0] = patches.Patch(color='darkred', label=f'{self.agent1.name} Snake (Dead)')
            if not self.final_state.snake2.alive:
                legend_elements[1] = patches.Patch(color='darkblue', label=f'{self.agent2.name} Snake (Dead)')
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error saving Snake visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset(self) -> None:
        """Reset game state for a new match."""
        super().reset()
        self.game_state = None
        self.match_history = []
        self.final_state = None 