"""
Sudoku game implementation for ProxyWar framework.

This module implements the classic Sudoku puzzle game
where agents need to solve a 9x9 grid puzzle.
"""

import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sudoku import Sudoku

from .base import SinglePlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


class SudokuState:
    """Represents the state of a Sudoku game."""
    
    def __init__(self, difficulty: float = 0.5):
        """
        Initialize Sudoku state.
        
        Args:
            difficulty: Difficulty level between 0.0 and 1.0 (default 0.5)
        """
        self.difficulty = difficulty
        
        # Generate a Sudoku puzzle using py-sudoku with specified difficulty
        self.puzzle_generator = Sudoku(3, difficulty=difficulty)  # 3x3 sub-grids = 9x9 full grid
        
        # Get the puzzle with removed values
        self.puzzle = self.puzzle_generator.board
        self.original_puzzle = [[cell for cell in row] for row in self.puzzle]  # Deep copy
        
        # Solve it to get the solution
        solver = Sudoku(3, board=[[cell for cell in row] for row in self.puzzle])
        
        # Capture the solve output to parse the solution
        import io
        import sys
        from contextlib import redirect_stdout
        import re
        
        # Capture solve output
        f = io.StringIO()
        with redirect_stdout(f):
            solver.solve()
        
        solve_output = f.getvalue()
        
        # Parse the solution from the output
        self.solution = self._parse_solution_from_output(solve_output)
        
        # If parsing failed, use a fallback method
        if not self.solution:  # Check if empty list
            self.solution = self._solve_using_backtracking()
        
        # Track solving attempts
        self.attempts = 0
        self.start_time = time.time()
    
    def _parse_solution_from_output(self, output: str) -> List[List[int]]:
        """
        Parse the solution from py-sudoku solve output.
        
        Args:
            output: The string output from solver.solve()
            
        Returns:
            9x9 solution matrix or empty list if parsing failed
        """
        import re
        
        lines = output.split('\n')
        solution_lines = []
        
        for line in lines:
            # Look for lines that contain the solution (format: | 1 5 7 | 8 6 9 | 2 3 4 |)
            if '|' in line and not line.startswith('+') and not line.startswith('-'):
                # Extract all numbers from the line
                numbers = re.findall(r'\d+', line)
                if len(numbers) == 9:
                    solution_lines.append([int(n) for n in numbers])
        
        if len(solution_lines) == 9:
            return solution_lines
        return []  # Return empty list instead of None
    
    def _solve_using_backtracking(self) -> List[List[int]]:
        """
        Fallback method to solve sudoku using backtracking.
        
        Returns:
            9x9 solution matrix
        """
        # Create a copy of the puzzle for solving
        board = [[cell if cell is not None else 0 for cell in row] for row in self.puzzle]
        
        def is_valid(board, row, col, num):
            # Check row
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # Check column
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # Check 3x3 box
            box_row = (row // 3) * 3
            box_col = (col // 3) * 3
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def solve(board):
            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        for num in range(1, 10):
                            if is_valid(board, i, j, num):
                                board[i][j] = num
                                if solve(board):
                                    return True
                                board[i][j] = 0
                        return False
            return True
        
        if solve(board):
            return board
        else:
            # If backtracking fails, return a dummy solution
            return [[1 for _ in range(9)] for _ in range(9)]
    
    def get_observation(self) -> List[int]:
        """
        Get flattened observation of the current puzzle state.
        
        Returns:
            Flattened 81-element list where 0 represents empty cells
        """
        observation = []
        for row in self.puzzle:
            for cell in row:
                observation.append(cell if cell is not None else 0)
        return observation
    
    def get_action_mask(self) -> List[bool]:
        """
        Get action mask for Sudoku (not used in this implementation).
        
        For Sudoku, we expect a complete solution, not individual moves,
        so this returns a dummy mask.
        """
        return [True] * 81  # All positions can potentially be filled
    
    def is_valid_solution(self, solution: List[int]) -> bool:
        """
        Check if a proposed solution is valid.
        
        Args:
            solution: 81-element list representing the filled grid
            
        Returns:
            True if the solution is valid, False otherwise
        """
        if len(solution) != 81:
            return False
        
        # Convert flat solution to 2D grid
        grid = []
        for i in range(9):
            row = solution[i*9:(i+1)*9]
            grid.append(row)
        
        # Check if all cells are filled (1-9)
        for row in grid:
            for cell in row:
                if not isinstance(cell, int) or cell < 1 or cell > 9:
                    return False
        
        # Check rows
        for row in grid:
            if len(set(row)) != 9 or set(row) != set(range(1, 10)):
                return False
        
        # Check columns
        for col in range(9):
            column = [grid[row][col] for row in range(9)]
            if len(set(column)) != 9 or set(column) != set(range(1, 10)):
                return False
        
        # Check 3x3 sub-grids
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = []
                for i in range(3):
                    for j in range(3):
                        box.append(grid[box_row + i][box_col + j])
                if len(set(box)) != 9 or set(box) != set(range(1, 10)):
                    return False
        
        # Check if solution matches the original puzzle constraints
        for i in range(9):
            for j in range(9):
                if self.original_puzzle[i][j] is not None:
                    if grid[i][j] != self.original_puzzle[i][j]:
                        return False
        
        return True
    
    def is_correct_solution(self, solution: List[int]) -> bool:
        """
        Check if the solution matches the expected solution.
        
        Args:
            solution: 81-element list representing the filled grid
            
        Returns:
            True if the solution is correct, False otherwise
        """
        if not self.is_valid_solution(solution):
            return False
        
        # Convert solution to 2D grid and compare with expected solution
        grid = []
        for i in range(9):
            row = solution[i*9:(i+1)*9]
            grid.append(row)
        
        for i in range(9):
            for j in range(9):
                if grid[i][j] != self.solution[i][j]:
                    return False
        
        return True
    
    def get_solution_quality(self, solution: List[int]) -> Dict[str, Any]:
        """
        Analyze the quality of a proposed solution.
        
        Returns:
            Dictionary with quality metrics
        """
        quality = {
            'valid': False,
            'correct': False,
            'filled_cells': 0,
            'correct_cells': 0,
            'error_cells': 0,
            'completion_rate': 0.0,
            'accuracy_rate': 0.0,
            'time_taken': time.time() - self.start_time
        }
        
        if len(solution) != 81:
            return quality
        
        # Convert to 2D grid
        grid = []
        for i in range(9):
            row = solution[i*9:(i+1)*9]
            grid.append(row)
        
        # Check if all cells are filled with valid values (1-9)
        invalid_cells = 0
        for i in range(9):
            for j in range(9):
                if not isinstance(grid[i][j], int) or grid[i][j] < 1 or grid[i][j] > 9:
                    invalid_cells += 1
        
        if invalid_cells > 0:
            # Solution contains invalid values (0, None, or values outside 1-9)
            quality['filled_cells'] = 81 - invalid_cells
            quality['completion_rate'] = quality['filled_cells'] / 81.0
            # Only count valid cells for correctness
            for i in range(9):
                for j in range(9):
                    if isinstance(grid[i][j], int) and 1 <= grid[i][j] <= 9:
                        if grid[i][j] == self.solution[i][j]:
                            quality['correct_cells'] += 1
                        else:
                            quality['error_cells'] += 1
            if quality['filled_cells'] > 0:
                quality['accuracy_rate'] = quality['correct_cells'] / quality['filled_cells']
            return quality
        
        # All cells are filled with valid values (1-9)
        quality['filled_cells'] = 81
        quality['completion_rate'] = 1.0
        
        # Count correct and incorrect cells
        for i in range(9):
            for j in range(9):
                if grid[i][j] == self.solution[i][j]:
                    quality['correct_cells'] += 1
                else:
                    quality['error_cells'] += 1
        
        quality['accuracy_rate'] = quality['correct_cells'] / 81.0
        quality['valid'] = self.is_valid_solution(solution)
        quality['correct'] = self.is_correct_solution(solution)
        
        return quality
    
    def __str__(self) -> str:
        """String representation of the puzzle."""
        result = "Sudoku Puzzle:\n"
        for i, row in enumerate(self.puzzle):
            if i % 3 == 0 and i != 0:
                result += "------+-------+------\n"
            row_str = ""
            for j, cell in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += " | "
                row_str += str(cell) if cell is not None else "."
                if j < 8:
                    row_str += " "
            result += row_str + "\n"
        return result


@GAME_REGISTRY.register('sudoku')
class SudokuGame(SinglePlayerGame):
    """
    Sudoku puzzle game implementation.
    
    This single-player game challenges agents to solve Sudoku puzzles.
    Unlike other games, agents must provide a complete solution at once.
    """
    
    def __init__(self, movement_timeout: float = 45.0):
        """Initialize Sudoku game with configurations."""
        # For Sudoku, we only use one timeout for the entire solution
        super().__init__("Sudoku", movement_timeout, movement_timeout)
        self.difficulty = 0.5  # Fixed difficulty
        self.state = None
        
        # Store results for visualization
        self.agent1_result = None
        self.agent2_result = None
        self.shared_puzzle_state = None
        self.match_results = None
        
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """Setup a Sudoku robin-round challenge for two agents."""
        super().setup_match(agent1, agent2)
        print(f"Sudoku match: {agent1.name} vs {agent2.name}")
        print(f"Testing fixed difficulty Sudoku (0.5)")
        
    def run_match(self) -> Dict[str, Any]:
        """
        Run a Sudoku challenge between two agents.
        
        Each agent attempts to solve the same puzzle with fixed 0.5 difficulty.
        
        Returns:
            Dictionary containing match results
        """
        if not self.agents or not self.agent1 or not self.agent2:
            raise ValueError("Match not properly setup. Call setup_match() first.")
        
        print(f"Starting Sudoku match with 0.5 difficulty")
        print("=" * 80)
        
        # Generate a shared puzzle for both agents
        puzzle_state = SudokuState(difficulty=0.5)
        
        # Run challenge for both agents on the same puzzle
        agent1_result = self._run_agent_on_puzzle(self.agent1, puzzle_state, "Agent 1")
        agent2_result = self._run_agent_on_puzzle(self.agent2, puzzle_state, "Agent 2")
        
        # Store results for visualization
        self.agent1_result = agent1_result
        self.agent2_result = agent2_result
        self.shared_puzzle_state = puzzle_state
        
        # Calculate scores
        winner, scores = self._compare_agent_results(agent1_result, agent2_result)
        
        print(f"\nFINAL RESULTS:")
        print(f"  {self.agent1.name}: {scores[self.agent1.name]:.3f}")
        print(f"  {self.agent2.name}: {scores[self.agent2.name]:.3f}")
        print(f"  Winner: {winner}")
        
        # Prepare match history
        match_history = [
            (self.agent1.name, agent1_result['solution']),
            (self.agent2.name, agent2_result['solution'])
        ]
        match_history_with_timing = [
            (self.agent1.name, agent1_result['solution'], agent1_result['time_taken']),
            (self.agent2.name, agent2_result['solution'], agent2_result['time_taken'])
        ]
        
        self.current_match_history = match_history
        
        result = {
            'winner': winner,
            'scores': scores,
            'moves': 2,  # Two solution attempts
            'match_history': match_history,
            'move_history_with_timing': match_history_with_timing,
            'agent1_result': agent1_result,
            'agent2_result': agent2_result,
            'puzzle_state': puzzle_state
        }
        
        # Store match results for visualization
        self.match_results = result
        
        return result
    
    def _run_agent_on_puzzle(self, agent: BaseAgent, puzzle_state: SudokuState, agent_label: str) -> Dict[str, Any]:
        """Run a single agent on a Sudoku puzzle."""
        print(f"  {agent_label}: {agent.name} - difficulty {puzzle_state.difficulty:.0%}")
        
        # Start agent session for cumulative timeout tracking
        self.start_agent_session(agent)
        
        # Get puzzle observation
        observation = puzzle_state.get_observation()
        action_mask = puzzle_state.get_action_mask()
        
        # Agent provides complete solution
        start_time = time.time()
        action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
            agent, observation, action_mask, [], [], 0
        )
        
        # Handle timeout
        if timeout_result is not None:
            print(f"    {agent.name} timed out")
            return {
                'success': False,
                'solution': None,
                'time_taken': decision_time,
                'timeout': True,
                'quality': {
                    'valid': False,
                    'correct': False,
                    'filled_cells': 0,
                    'correct_cells': 0,
                    'error_cells': 0,
                    'completion_rate': 0.0,
                    'accuracy_rate': 0.0,
                    'time_taken': decision_time
                }
            }
        
        # Validate solution
        if action is None:
            print(f"    {agent.name} returned None solution")
            quality = puzzle_state.get_solution_quality([])
        else:
            if not isinstance(action, (list, tuple)) or len(action) != 81:
                print(f"    {agent.name} returned invalid solution format")
                quality = puzzle_state.get_solution_quality([])
            else:
                quality = puzzle_state.get_solution_quality(list(action))
                if quality['correct']:
                    print(f"    {agent.name} solved correctly! Time: {decision_time:.2f}s")
                elif quality['valid']:
                    print(f"     {agent.name} provided valid but incorrect solution")
                else:
                    print(f"    {agent.name} provided invalid solution")
                    print(f"       Filled: {quality['filled_cells']}/81, Correct: {quality['correct_cells']}, Errors: {quality['error_cells']}")
        
        return {
            'success': quality['correct'] if action is not None else False,
            'solution': action,
            'time_taken': decision_time,
            'timeout': False,
            'quality': quality
        }
    
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
        
        # Check if solutions are correct
        agent1_correct = result1['quality']['correct']
        agent2_correct = result2['quality']['correct']
        
        # Scoring logic
        if agent1_correct and agent2_correct:
            # Both correct - faster wins
            if result1['time_taken'] < result2['time_taken']:
                scores[agent1_name] = 1.0
                scores[agent2_name] = 0.0
                winner = agent1_name
                print(f"  Both correct, {agent1_name} wins by speed ({result1['time_taken']:.4f}s vs {result2['time_taken']:.4f}s)")
            elif result2['time_taken'] < result1['time_taken']:
                scores[agent1_name] = 0.0
                scores[agent2_name] = 1.0
                winner = agent2_name
                print(f"  Both correct, {agent2_name} wins by speed ({result2['time_taken']:.4f}s vs {result1['time_taken']:.4f}s)")
            else:
                scores[agent1_name] = 0.5
                scores[agent2_name] = 0.5
                winner = 'draw'
                print(f"  Both correct with same time - draw ({result1['time_taken']:.4f}s)")
        elif agent1_correct:
            # Only agent1 correct
            scores[agent1_name] = 1.0
            scores[agent2_name] = 0.0
            winner = agent1_name
            print(f"  {agent1_name} correct, {agent2_name} incorrect")
        elif agent2_correct:
            # Only agent2 correct
            scores[agent1_name] = 0.0
            scores[agent2_name] = 1.0
            winner = agent2_name
            print(f"  {agent2_name} correct, {agent1_name} incorrect")
        else:
            # Neither correct - compare accuracy
            agent1_accuracy = result1['quality']['accuracy_rate']
            agent2_accuracy = result2['quality']['accuracy_rate']
            
            if agent1_accuracy > agent2_accuracy:
                scores[agent1_name] = 0.5
                scores[agent2_name] = 0.0
                winner = agent1_name
                print(f"  Neither correct, {agent1_name} wins by accuracy ({agent1_accuracy:.1%} vs {agent2_accuracy:.1%})")
            elif agent2_accuracy > agent1_accuracy:
                scores[agent1_name] = 0.0
                scores[agent2_name] = 0.5
                winner = agent2_name
                print(f"  Neither correct, {agent2_name} wins by accuracy ({agent2_accuracy:.1%} vs {agent1_accuracy:.1%})")
            else:
                scores[agent1_name] = 0.25
                scores[agent2_name] = 0.25
                winner = 'draw'
                print(f"  Neither correct, same accuracy - draw")
        
        return winner, scores
    
    def get_game_rules(self) -> str:
        """Get Sudoku game rules description."""
        return """
Sudoku Game Rules:

1. The puzzle is a 9x9 grid divided into nine 3x3 sub-grids
2. Some cells are pre-filled with numbers 1-9
3. Fill the empty cells so that each row, column, and 3x3 sub-grid contains all digits 1-9
4. Each digit must appear exactly once in each row, column, and sub-grid
5. There is only one correct solution for each puzzle

Agent Interface:
- select_action(observation, action_mask) should return a complete solution as a list of 81 integers
- observation: 81-element list where 0 represents empty cells, 1-9 are given numbers
- The returned solution should be a list of 81 integers (1-9) representing the filled grid
- The solution is provided all at once, not step by step
- Return order: row by row, left to right (index 0-8 is row 1, 9-17 is row 2, etc.)

Note: Your agent should implement a general Sudoku solver that works for any valid puzzle.
Different difficulty levels will be tested with varying numbers of given cells.
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for Sudoku."""
        sample_state = SudokuState(difficulty=0.5)
        
        return {
            'description': 'Flattened 9x9 Sudoku grid',
            'observation_size': 81,
            'sample_observation': sample_state.get_observation()[:20] + ['...'] + sample_state.get_observation()[-20:],
            'action_space_size': 81,  # Complete solution required
            'sample_action_mask': [True] * 81,  # All positions can be filled
            'encoding': '0 for empty cells, 1-9 for filled cells',
            'solution_format': 'List of 81 integers (1-9) representing the complete solution',
            'indexing': 'Row-major order: index = row * 9 + col (0-based)',
            'position_mapping': 'Grid positions 0-80 map to cells as: 0-8=row1, 9-17=row2, ..., 72-80=row9. Each row is left to right. Return complete solution all at once.',
            'note': 'Agent must return complete solution, not individual moves'
        }
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save a visualization of the Sudoku match results.
        
        Creates a side-by-side comparison showing both agents' solutions.
        
        Args:
            save_path: Path where to save the visualization
            
        Returns:
            True if visualization was saved successfully, False otherwise
        """
        if not self.agent1_result or not self.agent2_result or not self.shared_puzzle_state:
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
            
            # Create figure with three subplots (agent1 solution, puzzle, agent2 solution)
            fig, (ax1, ax_puzzle, ax2) = plt.subplots(1, 3, figsize=(20, 7))
            
            # Helper function to draw a Sudoku grid
            def draw_sudoku(ax, title, puzzle_state, solution=None, is_original=False):
                ax.set_xlim(0, 9)
                ax.set_ylim(0, 9)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                
                # Draw grid lines
                for i in range(10):
                    lw = 2 if i % 3 == 0 else 1
                    ax.axhline(i, color='black', linewidth=lw)
                    ax.axvline(i, color='black', linewidth=lw)
                
                # Draw numbers
                for i in range(9):
                    for j in range(9):
                        # Get the value to display
                        if is_original:
                            # Original puzzle with solution
                            original_val = puzzle_state.original_puzzle[i][j]
                            solution_val = puzzle_state.solution[i][j]
                            if original_val is not None:
                                # Given number (black)
                                ax.text(j + 0.5, i + 0.5, str(original_val), 
                                       ha='center', va='center', fontsize=16, 
                                       weight='bold', color='black')
                            else:
                                # Solution number (blue)
                                ax.text(j + 0.5, i + 0.5, str(solution_val), 
                                       ha='center', va='center', fontsize=14, 
                                       weight='normal', color='blue', alpha=0.7)
                        else:
                            # Solution attempt
                            if solution and len(solution) == 81:
                                idx = i * 9 + j
                                val = solution[idx]
                                
                                # Determine color based on correctness
                                original_val = puzzle_state.original_puzzle[i][j]
                                correct_val = puzzle_state.solution[i][j]
                                
                                if original_val is not None:
                                    # Given number
                                    color = 'black'
                                    weight = 'bold'
                                elif val == correct_val:
                                    # Correct fill
                                    color = 'blue'
                                    weight = 'normal'
                                else:
                                    # Incorrect fill
                                    color = 'red'
                                    weight = 'normal'
                                
                                if isinstance(val, int) and 1 <= val <= 9:
                                    ax.text(j + 0.5, i + 0.5, str(val), 
                                           ha='center', va='center', fontsize=14,
                                           weight=weight, color=color)
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=16, weight='bold', pad=10)
            
            # Draw agent solutions and original puzzle
            agent1_quality = self.agent1_result['quality']
            if agent1_quality['correct']:
                agent1_title = f"{self.agent1.name}\n({self.agent1_result['time_taken']:.1f}s)"
            else:
                agent1_title = f"{self.agent1.name}\n({agent1_quality['correct_cells']}/81)"
            
            agent2_quality = self.agent2_result['quality']
            if agent2_quality['correct']:
                agent2_title = f"{self.agent2.name}\n({self.agent2_result['time_taken']:.1f}s)"
            else:
                agent2_title = f"{self.agent2.name}\n({agent2_quality['correct_cells']}/81)"
            
            draw_sudoku(ax1, agent1_title, self.shared_puzzle_state, self.agent1_result['solution'])
            draw_sudoku(ax_puzzle, "Puzzle & Solution", self.shared_puzzle_state, is_original=True)
            draw_sudoku(ax2, agent2_title, self.shared_puzzle_state, self.agent2_result['solution'])
            
            # Add legend
            legend_text = "Black=Given  Blue=Solution/Correct  Red=Incorrect"
            fig.text(0.5, 0.02, legend_text, ha='center', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            
            # Add overall match title
            if self.match_results:
                winner = self.match_results['winner']
                if winner == 'draw':
                    match_title = 'Sudoku Match - Draw'
                else:
                    match_title = f'Sudoku Match - Winner: {winner}'
                fig.suptitle(match_title, fontsize=16, weight='bold', y=0.96)
            
            # Save the figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.08)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error saving Sudoku visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def handle_agent_move_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any, 
                                       move_history: List, move_history_with_timing: List, 
                                       game_step: int) -> Tuple[Any, float, Optional[Dict]]:
        """
        Handle agent move with timeout detection for Sudoku.
        
        For Sudoku, the agent provides a complete solution in one call.
        
        Returns:
            Tuple of (solution, decision_time, timeout_result)
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
        self.agent1_result = None
        self.agent2_result = None
        self.shared_puzzle_state = None
        self.match_results = None 