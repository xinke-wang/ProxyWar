"""
Maze game implementation for ProxyWar framework.

This module implements a maze solving game using mazelib
where agents need to find the shortest path from start to end.
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.ShortestPath import ShortestPath

from .base import SinglePlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


class MazeState:
    """Represents the state of a maze game."""
    
    def __init__(self, size: int = 10, seed: Optional[int] = None):
        """
        Initialize Maze state.
        
        Args:
            size: Size of the maze (will create size x size logical maze)
            seed: Random seed for reproducible maze generation
        """
        self.size = size
        self.seed = seed
        
        # Set random seed for reproducible maze generation
        if seed is not None:
            np.random.seed(seed)
        
        # Generate maze using mazelib
        self.maze = Maze()
        self.maze.generator = Prims(size, size)  # type: ignore
        self.maze.generate()
        
        # Store the maze grid (1 = wall, 0 = path)
        self.grid = self.maze.grid
        assert self.grid is not None, "Maze generation failed"
        self.height, self.width = self.grid.shape
        
        # Find start and end positions
        self.start_pos = self._find_start_position()
        self.end_pos = self._find_end_position()
        
        # Current position of the agent
        self.current_pos = self.start_pos
        
        # Store the optimal solution for comparison
        self.optimal_path = self._find_optimal_path()
        self.optimal_moves = len(self.optimal_path) - 1 if self.optimal_path else float('inf')
        
        # Track game state
        self.moves_made = 0
        self.path_taken = [self.start_pos]
        self.visited_positions = {self.start_pos}
        self.start_time = time.time()
        self.max_moves = size * size * 2  # Reasonable upper bound
    
    def _find_start_position(self) -> Tuple[int, int]:
        """Find a good start position (top-left area)."""
        # Look for first open position from top-left
        assert self.grid is not None
        for row in range(1, self.height):
            for col in range(1, self.width):
                if self.grid[row, col] == 0:
                    return (row, col)
        return (1, 1)  # Fallback
    
    def _find_end_position(self) -> Tuple[int, int]:
        """Find a good end position (bottom-right area)."""
        # Look for first open position from bottom-right
        assert self.grid is not None
        for row in range(self.height - 2, 0, -1):
            for col in range(self.width - 2, 0, -1):
                if self.grid[row, col] == 0:
                    return (row, col)
        return (self.height - 2, self.width - 2)  # Fallback
    
    def _find_optimal_path(self) -> Optional[List[Tuple[int, int]]]:
        """Find the optimal path using a simple BFS algorithm."""
        try:
            assert self.grid is not None
            from collections import deque
            
            # Use BFS to find shortest path
            queue = deque([(self.start_pos, [self.start_pos])])
            visited = {self.start_pos}
            
            while queue:
                (row, col), path = queue.popleft()
                
                if (row, col) == self.end_pos:
                    return path
                
                # Check all 4 directions
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row, new_col = row + dr, col + dc
                    
                    if (0 <= new_row < self.height and 
                        0 <= new_col < self.width and 
                        self.grid[new_row, new_col] == 0 and 
                        (new_row, new_col) not in visited):
                        
                        visited.add((new_row, new_col))
                        queue.append(((new_row, new_col), path + [(new_row, new_col)]))
            
            return None  # No path found
        except Exception as e:
            # BFS over numpy grid should not raise; surface unexpected
            # failures without breaking maze setup.
            import sys
            print(f"[maze] _find_optimal_path failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None
    
    def get_observation(self) -> List[int]:
        """
        Get flattened observation of the current maze state.
        
        Returns:
            Flattened representation including:
            - Maze grid (flattened)
            - Current position
            - End position
        """
        observation = []
        
        # Add flattened maze grid
        assert self.grid is not None
        observation.extend(self.grid.flatten().tolist())
        
        # Add current position
        observation.extend([self.current_pos[0], self.current_pos[1]])
        
        # Add end position
        observation.extend([self.end_pos[0], self.end_pos[1]])
        
        return observation
    
    def get_action_mask(self) -> List[bool]:
        """
        Get action mask for valid moves.
        
        Returns:
            Boolean list for 4 actions: [UP, DOWN, LEFT, RIGHT]
        """
        row, col = self.current_pos
        
        # Check each direction
        up_valid = (row - 1 >= 0 and self.grid[row - 1, col] == 0)
        down_valid = (row + 1 < self.height and self.grid[row + 1, col] == 0)
        left_valid = (col - 1 >= 0 and self.grid[row, col - 1] == 0)
        right_valid = (col + 1 < self.width and self.grid[row, col + 1] == 0)
        
        return [up_valid, down_valid, left_valid, right_valid]
    
    def make_move(self, action: int) -> bool:
        """
        Make a move in the maze.
        
        Args:
            action: Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            True if move was valid, False otherwise
        """
        if action < 0 or action >= 4:
            return False
        
        row, col = self.current_pos
        
        # Calculate new position
        if action == 0:  # UP
            new_pos = (row - 1, col)
        elif action == 1:  # DOWN
            new_pos = (row + 1, col)
        elif action == 2:  # LEFT
            new_pos = (row, col - 1)
        elif action == 3:  # RIGHT
            new_pos = (row, col + 1)
        
        # Check if move is valid
        new_row, new_col = new_pos
        if (new_row < 0 or new_row >= self.height or 
            new_col < 0 or new_col >= self.width or 
            self.grid[new_row, new_col] == 1):
            return False
        
        # Make the move
        self.current_pos = new_pos
        self.moves_made += 1
        self.path_taken.append(new_pos)
        self.visited_positions.add(new_pos)
        
        return True
    
    def is_solved(self) -> bool:
        """Check if the maze is solved (reached end position)."""
        return self.current_pos == self.end_pos
    
    def is_game_over(self) -> bool:
        """Check if the game is over (solved or max moves reached)."""
        return self.is_solved() or self.moves_made >= self.max_moves
    
    def get_solution_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality of the current solution.
        
        Returns:
            Dictionary with quality metrics
        """
        quality = {
            'solved': self.is_solved(),
            'moves_made': self.moves_made,
            'optimal_moves': self.optimal_moves,
            'path_efficiency': 0.0,
            'exploration_ratio': 0.0,
            'time_taken': time.time() - self.start_time
        }
        
        if self.optimal_moves < float('inf') and self.moves_made > 0:
            quality['path_efficiency'] = min(1.0, self.optimal_moves / self.moves_made)
        
        # Calculate exploration ratio (how much of the maze was explored)
        total_open_cells = np.sum(self.grid == 0)
        if total_open_cells > 0:
            quality['exploration_ratio'] = len(self.visited_positions) / total_open_cells
        
        return quality
    
    def reset_to_start(self) -> None:
        """Reset agent position to start."""
        self.current_pos = self.start_pos
        self.moves_made = 0
        self.path_taken = [self.start_pos]
        self.visited_positions = {self.start_pos}
        self.start_time = time.time()
    
    def __str__(self) -> str:
        """String representation of the maze state."""
        result = f"Maze State ({self.width}x{self.height}):\n"
        result += f"Current Position: {self.current_pos}\n"
        result += f"End Position: {self.end_pos}\n"
        result += f"Moves Made: {self.moves_made}\n"
        result += f"Solved: {self.is_solved()}\n"
        result += f"Optimal Moves: {self.optimal_moves}\n"
        
        # Show maze with current position
        maze_str = ""
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) == self.current_pos:
                    maze_str += "A"  # Agent
                elif (row, col) == self.end_pos:
                    maze_str += "E"  # End
                elif (row, col) == self.start_pos:
                    maze_str += "S"  # Start
                elif self.grid[row, col] == 1:
                    maze_str += "█"  # Wall
                else:
                    maze_str += " "  # Path
            maze_str += "\n"
        
        result += maze_str
        return result


@GAME_REGISTRY.register('maze')
class MazeGame(SinglePlayerGame):
    """
    Maze solving game implementation.
    
    This single-player game challenges agents to find the shortest path
    from start to end in a randomly generated maze.
    """
    
    def __init__(self, movement_timeout: float = 5.0, total_game_timeout: float = 120.0):
        """Initialize Maze game with configurations."""
        super().__init__("Maze", movement_timeout, total_game_timeout)
        self.maze_size = 10  # Fixed size
        
        # Store results for visualization
        self.agent1_final_state = None
        self.agent2_final_state = None
        self.shared_seed = None
        self.match_results = None
        
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """Setup a Maze match between two agents."""
        super().setup_match(agent1, agent2)
        print(f"Maze match: {agent1.name} vs {agent2.name}")
        # maze_size counts logical cells; the rendered grid (printed below as
        # "Size: WxH") is wall-padded to (2*maze_size + 1) on each axis.
        print(f"Both agents will solve the same {self.maze_size}x{self.maze_size}-cell maze")
        
        # Generate a shared random seed for both agents
        import random
        self.shared_seed = random.randint(0, 1000000)
        
    def run_match(self) -> Dict[str, Any]:
        """
        Run a Maze match between two agents.
        
        Each agent attempts to solve the same maze.
        Winner is determined by: 1) Completion, 2) Fewer moves, 3) Faster time
        
        Returns:
            Dictionary containing match results
        """
        if not self.agents or not self.agent1 or not self.agent2:
            raise ValueError("Match not properly setup. Call setup_match() first.")
        
        print(f"Starting Maze match with shared seed: {self.shared_seed}")
        print("=" * 80)
        
        # Create a single shared maze that both agents will use
        if self.shared_seed is None:
            raise ValueError("Shared seed not initialized")
        
        shared_maze_state = MazeState(size=self.maze_size, seed=self.shared_seed)
        print(f"Generated shared maze:")
        print(f"  Size: {shared_maze_state.width}x{shared_maze_state.height}")
        print(f"  Start: {shared_maze_state.start_pos}, End: {shared_maze_state.end_pos}")
        print(f"  Optimal path: {shared_maze_state.optimal_moves} moves")
        
        # Run challenge for both agents on the same maze
        agent1_result = self._run_agent_on_shared_maze(self.agent1, shared_maze_state, "Agent 1")
        agent2_result = self._run_agent_on_shared_maze(self.agent2, shared_maze_state, "Agent 2")
        
        # Store results for visualization
        self.agent1_final_state = agent1_result['final_state']
        self.agent2_final_state = agent2_result['final_state']
        
        # Calculate scores
        winner, scores = self._compare_agent_results(agent1_result, agent2_result)
        
        print(f"\nFINAL RESULTS:")
        print(f"  {self.agent1.name}: {scores[self.agent1.name]:.3f}")
        print(f"  {self.agent2.name}: {scores[self.agent2.name]:.3f}")
        print(f"  Winner: {winner}")
        
        # Prepare match history
        match_history = [
            (self.agent1.name, f"Solved: {agent1_result['solved']}, Moves: {agent1_result['moves']}"),
            (self.agent2.name, f"Solved: {agent2_result['solved']}, Moves: {agent2_result['moves']}")
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
    
    def _run_agent_on_shared_maze(self, agent: BaseAgent, shared_maze_state: MazeState, agent_label: str) -> Dict[str, Any]:
        """Run a single agent on a shared maze state."""
        print(f"  {agent_label}: {agent.name} - {shared_maze_state.width}x{shared_maze_state.height} maze")
        
        # Create a copy of the shared maze state for this agent
        import copy
        maze_state = copy.deepcopy(shared_maze_state)
        maze_state.reset_to_start()  # Ensure agent starts from beginning
        
        print(f"    Start: {maze_state.start_pos}, End: {maze_state.end_pos}")
        print(f"    Optimal path: {maze_state.optimal_moves} moves")
        
        # Start agent session for cumulative timeout tracking
        self.start_agent_session(agent)
        
        move_history = []
        
        # Game loop
        while not maze_state.is_game_over():
            # Check cumulative timeout
            remaining_time = self.get_remaining_time()
            if remaining_time <= 0:
                print(f"    {agent.name} timed out")
                break
            
            # Get current state
            observation = maze_state.get_observation()
            action_mask = maze_state.get_action_mask()
            
            # Agent makes a move
            action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
                agent, observation, action_mask, move_history, [], maze_state.moves_made
            )
            
            # Handle timeout
            if timeout_result is not None:
                print(f"    {agent.name} timed out during move")
                break
            
            # Validate and make move
            if action is None or not isinstance(action, int):
                print(f"    {agent.name} returned invalid action: {action}")
                break
            
            if not maze_state.make_move(action):
                print(f"    {agent.name} made invalid move: {action}")
                break

            # Record (player, action, decision_time) so downstream telemetry
            # (coder_result.decision_times, *_report.json) sees real timings.
            move_history.append((agent.name, action, decision_time))

            # Print progress occasionally
            if maze_state.moves_made % 20 == 0:
                print(f"    Move {maze_state.moves_made}: pos={maze_state.current_pos}, "
                      f"remaining_time={remaining_time:.1f}s")

        # Game ended
        quality = maze_state.get_solution_quality()

        if quality['solved']:
            print(f"    {agent.name} solved the maze!")
            print(f"       Moves: {quality['moves_made']}, Optimal: {quality['optimal_moves']}")
            print(f"       Efficiency: {quality['path_efficiency']:.1%}")
        else:
            print(f"    {agent.name} failed to solve the maze")
            print(f"       Moves: {quality['moves_made']}, Explored: {quality['exploration_ratio']:.1%}")

        return {
            'solved': quality['solved'],
            'moves': quality['moves_made'],
            'optimal_moves': quality['optimal_moves'],
            'path_efficiency': quality['path_efficiency'],
            'exploration_ratio': quality['exploration_ratio'],
            'time_taken': quality['time_taken'],
            'move_history': move_history,
            'final_state': maze_state,
            'timeout': remaining_time <= 0
        }

    def _run_agent_on_maze(self, agent: BaseAgent, seed: int, agent_label: str) -> Dict[str, Any]:
        """Run a single agent on a maze."""
        print(f"  {agent_label}: {agent.name} - {self.maze_size}x{self.maze_size} maze")
        
        # Start agent session for cumulative timeout tracking
        self.start_agent_session(agent)
        
        # Create maze state with the shared seed
        maze_state = MazeState(size=self.maze_size, seed=seed)
        
        print(f"    Start: {maze_state.start_pos}, End: {maze_state.end_pos}")
        print(f"    Optimal path: {maze_state.optimal_moves} moves")
        
        move_history = []
        
        # Game loop
        while not maze_state.is_game_over():
            # Check cumulative timeout
            remaining_time = self.get_remaining_time()
            if remaining_time <= 0:
                print(f"    {agent.name} timed out")
                break
            
            # Get current state
            observation = maze_state.get_observation()
            action_mask = maze_state.get_action_mask()
            
            # Agent makes a move
            action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
                agent, observation, action_mask, move_history, [], maze_state.moves_made
            )
            
            # Handle timeout
            if timeout_result is not None:
                print(f"    {agent.name} timed out during move")
                break
            
            # Validate and make move
            if action is None or not isinstance(action, int):
                print(f"    {agent.name} returned invalid action: {action}")
                break
            
            if not maze_state.make_move(action):
                print(f"    {agent.name} made invalid move: {action}")
                break

            move_history.append((agent.name, action, decision_time))

            # Print progress occasionally
            if maze_state.moves_made % 20 == 0:
                print(f"    Move {maze_state.moves_made}: pos={maze_state.current_pos}, "
                      f"remaining_time={remaining_time:.1f}s")

        # Game ended
        quality = maze_state.get_solution_quality()

        if quality['solved']:
            print(f"    {agent.name} solved the maze!")
            print(f"       Moves: {quality['moves_made']}, Optimal: {quality['optimal_moves']}")
            print(f"       Efficiency: {quality['path_efficiency']:.1%}")
        else:
            print(f"    {agent.name} failed to solve the maze")
            print(f"       Moves: {quality['moves_made']}, Explored: {quality['exploration_ratio']:.1%}")
        
        return {
            'solved': quality['solved'],
            'moves': quality['moves_made'],
            'optimal_moves': quality['optimal_moves'],
            'path_efficiency': quality['path_efficiency'],
            'exploration_ratio': quality['exploration_ratio'],
            'time_taken': quality['time_taken'],
            'move_history': move_history,
            'final_state': maze_state,
            'timeout': remaining_time <= 0
        }
    
    def _compare_agent_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Compare two agent results and determine winner.
        
        Priority:
        1. Completion (solved vs not solved)
        2. Fewer moves (if both solved)
        3. Better exploration (if neither solved)
        
        Returns:
            Tuple of (winner, scores_dict)
        """
        if not self.agent1 or not self.agent2:
            raise ValueError("Agents not properly initialized")
            
        agent1_name = self.agent1.name
        agent2_name = self.agent2.name
        
        # Initialize scores
        scores = {agent1_name: 0.0, agent2_name: 0.0}
        
        # Check completion
        agent1_solved = result1['solved']
        agent2_solved = result2['solved']
        
        if agent1_solved and agent2_solved:
            # Both solved - compare by efficiency
            if result1['path_efficiency'] > result2['path_efficiency']:
                scores[agent1_name] = 1.0
                scores[agent2_name] = 0.0
                winner = agent1_name
                print(f"  Both solved, {agent1_name} wins by efficiency")
            elif result2['path_efficiency'] > result1['path_efficiency']:
                scores[agent1_name] = 0.0
                scores[agent2_name] = 1.0
                winner = agent2_name
                print(f"  Both solved, {agent2_name} wins by efficiency")
            else:
                # Same efficiency - compare by time
                if result1['time_taken'] < result2['time_taken']:
                    scores[agent1_name] = 1.0
                    scores[agent2_name] = 0.0
                    winner = agent1_name
                    print(f"  Both solved with same efficiency, {agent1_name} wins by time")
                elif result2['time_taken'] < result1['time_taken']:
                    scores[agent1_name] = 0.0
                    scores[agent2_name] = 1.0
                    winner = agent2_name
                    print(f"  Both solved with same efficiency, {agent2_name} wins by time")
                else:
                    scores[agent1_name] = 0.5
                    scores[agent2_name] = 0.5
                    winner = 'draw'
                    print(f"  Perfect tie - both solved with same efficiency and time")
        elif agent1_solved:
            # Only agent1 solved
            scores[agent1_name] = 1.0
            scores[agent2_name] = 0.0
            winner = agent1_name
            print(f"  {agent1_name} solved, {agent2_name} did not")
        elif agent2_solved:
            # Only agent2 solved
            scores[agent1_name] = 0.0
            scores[agent2_name] = 1.0
            winner = agent2_name
            print(f"  {agent2_name} solved, {agent1_name} did not")
        else:
            # Neither solved - compare by exploration
            if result1['exploration_ratio'] > result2['exploration_ratio']:
                scores[agent1_name] = 0.5
                scores[agent2_name] = 0.0
                winner = agent1_name
                print(f"  Neither solved, {agent1_name} wins by exploration")
            elif result2['exploration_ratio'] > result1['exploration_ratio']:
                scores[agent1_name] = 0.0
                scores[agent2_name] = 0.5
                winner = agent2_name
                print(f"  Neither solved, {agent2_name} wins by exploration")
            else:
                scores[agent1_name] = 0.25
                scores[agent2_name] = 0.25
                winner = 'draw'
                print(f"  Neither solved, same exploration - draw")
        
        return winner, scores
    
    def handle_agent_move_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any, 
                                       move_history: List, move_history_with_timing: List, 
                                       game_step: int) -> Tuple[Any, float, Optional[Dict]]:
        """
        Handle agent move with timeout detection for Maze.
        
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
    
    def get_game_rules(self) -> str:
        """Get Maze game rules description."""
        return """
Maze Game Rules:

1. You are an agent in a maze trying to reach the end position
2. The maze is a 2D grid where 1 represents walls and 0 represents open paths
3. You start at the start position and need to reach the end position
4. You can move in 4 directions: UP (0), DOWN (1), LEFT (2), RIGHT (3)
5. You cannot move through walls or outside the maze boundaries
6. The goal is to reach the end position in the minimum number of moves
7. The game ends when you reach the end or exceed the maximum number of moves

Agent Interface:
- select_action(observation, action_mask) should return an action (0-3)
- observation: flattened maze grid + current position + end position
- action_mask: 4-element boolean list showing which moves are valid
- Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

Note: Write a general maze solver that can handle any maze layout.
The same code will be tested on different randomly generated mazes.
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for Maze."""
        # Create sample maze for demonstration
        sample_state = MazeState(size=5, seed=42)
        
        return {
            'description': 'Flattened maze grid + current position + end position',
            'observation_size': sample_state.height * sample_state.width + 4,
            'sample_observation': sample_state.get_observation(),
            'action_space_size': 4,
            'sample_action_mask': sample_state.get_action_mask(),
            'encoding': 'Grid: 0=path, 1=wall; Positions: (row, col) coordinates',
            'action_encoding': '0=UP, 1=DOWN, 2=LEFT, 3=RIGHT',
            'observation_structure': f'[maze_grid_flattened({sample_state.height}x{sample_state.width}), current_row, current_col, end_row, end_col]',
            'position_mapping': 'Grid index = row * width + col (0-based)',
            'note': 'Agent must navigate from start to end position avoiding walls'
        }
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save a visualization of the Maze match results.
        
        Creates a side-by-side comparison showing both agents' paths.
        
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
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Helper function to draw a maze
            def draw_maze(ax, title, maze_state, agent_result):
                grid = maze_state.grid
                height, width = grid.shape
                
                # Create the maze visualization
                ax.imshow(grid, cmap='gray', origin='upper')
                
                # Mark start and end positions
                start_pos = maze_state.start_pos
                end_pos = maze_state.end_pos
                
                ax.scatter(start_pos[1], start_pos[0], c='green', s=100, marker='s', label='Start')
                ax.scatter(end_pos[1], end_pos[0], c='red', s=100, marker='s', label='End')
                
                # Draw the path taken
                if maze_state.path_taken:
                    path_y = [pos[0] for pos in maze_state.path_taken]
                    path_x = [pos[1] for pos in maze_state.path_taken]
                    ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Path')
                
                # Draw current position
                current_pos = maze_state.current_pos
                ax.scatter(current_pos[1], current_pos[0], c='blue', s=150, marker='o', label='Current')
                
                # Draw optimal path if available
                if maze_state.optimal_path:
                    opt_y = [pos[0] for pos in maze_state.optimal_path]
                    opt_x = [pos[1] for pos in maze_state.optimal_path]
                    ax.plot(opt_x, opt_y, 'r--', linewidth=1, alpha=0.5, label='Optimal')
                
                ax.set_xlim(-0.5, width - 0.5)
                ax.set_ylim(-0.5, height - 0.5)
                ax.set_aspect('equal')
                ax.legend()
                
                # Add title with results
                solved_text = "Solved" if agent_result['solved'] else "Failed"
                moves_text = f"Moves: {agent_result['moves']}"
                if agent_result['solved']:
                    efficiency_text = f"Efficiency: {agent_result['path_efficiency']:.1%}"
                    title_text = f"{title}\n{solved_text} | {moves_text} | {efficiency_text}"
                else:
                    exploration_text = f"Explored: {agent_result['exploration_ratio']:.1%}"
                    title_text = f"{title}\n{solved_text} | {moves_text} | {exploration_text}"
                
                ax.set_title(title_text, fontsize=12, pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Draw both agents' results
            draw_maze(ax1, self.agent1.name, self.agent1_final_state, self.match_results['agent1_result'])
            draw_maze(ax2, self.agent2.name, self.agent2_final_state, self.match_results['agent2_result'])
            
            # Add overall match title
            winner = self.match_results['winner']
            if winner == 'draw':
                match_title = 'Maze Match - Draw'
            else:
                match_title = f'Maze Match - Winner: {winner}'
            fig.suptitle(match_title, fontsize=16, weight='bold')
            
            # Save the figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error saving Maze visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset(self) -> None:
        """Reset game state for a new match."""
        super().reset()
        self.agent1_final_state = None
        self.agent2_final_state = None
        self.shared_seed = None
        self.match_results = None 