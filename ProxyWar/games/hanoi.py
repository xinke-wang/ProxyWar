"""Hanoi Tower single-player puzzle (fixed at 12 disks)."""

import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .base import SinglePlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


class HanoiState:
    """Represents the state of a Hanoi Tower game."""
    
    def __init__(self, num_disks: int):
        """
        Initialize Hanoi state.
        
        Args:
            num_disks: Number of disks in the puzzle
        """
        self.num_disks = num_disks
        self.num_rods = 3
        
        # Initialize towers: tower[i] contains list of disks (largest at bottom)
        # Disk sizes are represented as integers 1 to num_disks
        self.towers = [
            list(range(num_disks, 0, -1)),  # Rod 0: all disks, largest to smallest
            [],  # Rod 1: empty
            []   # Rod 2: empty
        ]
        
        self.move_count = 0
        self.optimal_moves = (2 ** num_disks) - 1  # Minimum moves to solve
    
    def is_valid_move(self, from_rod: int, to_rod: int) -> bool:
        """Check if a move is valid."""
        if from_rod < 0 or from_rod >= self.num_rods or to_rod < 0 or to_rod >= self.num_rods:
            return False
        
        if from_rod == to_rod:
            return False
        
        if not self.towers[from_rod]:  # Source rod is empty
            return False
        
        if not self.towers[to_rod]:  # Destination rod is empty - always valid
            return True
        
        # Check if top disk of source is smaller than top disk of destination
        return self.towers[from_rod][-1] < self.towers[to_rod][-1]
    
    def make_move(self, from_rod: int, to_rod: int) -> bool:
        """
        Make a move if valid.
        
        Returns:
            True if move was made, False if invalid
        """
        if not self.is_valid_move(from_rod, to_rod):
            return False
        
        disk = self.towers[from_rod].pop()
        self.towers[to_rod].append(disk)
        self.move_count += 1
        return True
    
    def is_solved(self) -> bool:
        """Check if the puzzle is solved (all disks on the last rod)."""
        return len(self.towers[2]) == self.num_disks
    
    def get_legal_actions(self) -> List[int]:
        """
        Get list of legal action indices.
        
        Action encoding: action = from_rod * 3 + to_rod
        Total action space: 9 actions (3x3 combinations, some invalid)
        """
        legal_actions = []
        for from_rod in range(3):
            for to_rod in range(3):
                if self.is_valid_move(from_rod, to_rod):
                    action = from_rod * 3 + to_rod
                    legal_actions.append(action)
        return legal_actions
    
    def get_action_mask(self) -> List[bool]:
        """Get action mask for all possible actions."""
        mask = [False] * 9  # 3x3 = 9 possible actions
        legal_actions = self.get_legal_actions()
        for action in legal_actions:
            mask[action] = True
        return mask
    
    def action_to_move(self, action: int) -> Tuple[int, int]:
        """Convert action index to (from_rod, to_rod) tuple."""
        from_rod = action // 3
        to_rod = action % 3
        return from_rod, to_rod
    
    def get_observation(self) -> List[int]:
        """
        Get flattened observation of the current state.
        
        Returns:
            Flattened representation where each position indicates the disk size
            at that position, or 0 if empty. Format:
            [rod0_bottom, rod0_next, ..., rod1_bottom, rod1_next, ..., rod2_bottom, rod2_next, ...]
        """
        observation = []
        for rod in self.towers:
            # Pad rod to max possible height with zeros
            rod_obs = rod + [0] * (self.num_disks - len(rod))
            observation.extend(rod_obs)
        return observation
    
    def copy(self) -> 'HanoiState':
        """Create a copy of the current state."""
        new_state = HanoiState(self.num_disks)
        new_state.towers = [tower.copy() for tower in self.towers]
        new_state.move_count = self.move_count
        return new_state
    
    def __str__(self) -> str:
        """String representation of the state."""
        result = f"Hanoi Tower ({self.num_disks} disks, {self.move_count} moves):\n"
        for i, tower in enumerate(self.towers):
            result += f"Rod {i}: {tower}\n"
        return result


@GAME_REGISTRY.register('hanoi')
class HanoiTowerGame(SinglePlayerGame):
    """Tower of Hanoi single-player puzzle, fixed at 12 disks."""
    
    NUM_DISKS = 12
    MAX_MOVES_MULTIPLIER = 5

    def __init__(self, movement_timeout: float = 60.0, total_game_timeout: float = 60.0):
        """Initialize Hanoi Tower game (fixed at 12 disks)."""
        super().__init__("HanoiTower", movement_timeout, total_game_timeout)
        self.num_disks = self.NUM_DISKS
        self.state = None

        self.agent1_final_state = None
        self.agent2_final_state = None
        self.match_results = None

        optimal_moves = (2 ** self.num_disks) - 1
        self.max_moves = optimal_moves * self.MAX_MOVES_MULTIPLIER

    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """Setup a Hanoi Tower 12-disk challenge for two agents."""
        super().setup_match(agent1, agent2)
        print(f"Hanoi Tower {self.num_disks}-disk challenge: {agent1.name} vs {agent2.name}")
        print(f"Minimum {(2 ** self.num_disks) - 1} moves to solve")
    
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete 12-disk Hanoi Tower challenge.
        
        Each agent attempts the 12-disk challenge independently, then results are compared.
        
        Returns:
            Dictionary containing match results
        """
        if not self.agents or not self.agent1 or not self.agent2:
            raise ValueError("Match not properly setup. Call setup_match() first.")

        print(f"Starting {self.num_disks}-disk Hanoi Tower challenge")
        print("=" * 80)

        self.agent1_final_state = None
        self.agent2_final_state = None

        agent1_result = self._run_agent_on_puzzle(self.agent1, self.num_disks, "Agent 1")
        agent2_result = self._run_agent_on_puzzle(self.agent2, self.num_disks, "Agent 2")

        self.agent1_final_state = agent1_result['final_state']
        self.agent2_final_state = agent2_result['final_state']

        winner, final_scores = self._compare_agent_results(agent1_result, agent2_result)

        combined_history = []
        combined_history_with_timing = []
        for move in agent1_result['move_history']:
            combined_history.append((self.agent1.name, move[1]))
        for move in agent1_result['move_history_with_timing']:
            combined_history_with_timing.append((self.agent1.name, move[1], move[2]))
        for move in agent2_result['move_history']:
            combined_history.append((self.agent2.name, move[1]))
        for move in agent2_result['move_history_with_timing']:
            combined_history_with_timing.append((self.agent2.name, move[1], move[2]))

        total_moves = agent1_result['moves'] + agent2_result['moves']

        print(f"\nFINAL RESULTS:")
        print(f"  {self.agent1.name}: {final_scores[self.agent1.name]:.3f}")
        print(f"  {self.agent2.name}: {final_scores[self.agent2.name]:.3f}")
        if winner == 'draw':
            print(f"  Overall Draw")
        else:
            print(f"  Overall Winner: {winner}")

        self.current_match_history = combined_history

        result = {
            'winner': winner,
            'scores': final_scores,
            'moves': total_moves,
            'match_history': combined_history,
            'move_history_with_timing': combined_history_with_timing,
            'agent1_result': agent1_result,
            'agent2_result': agent2_result,
        }
        
        # Store match results for visualization
        self.match_results = result
        
        return result
    
    def _run_agent_on_puzzle(self, agent: BaseAgent, num_disks: int, agent_label: str) -> Dict[str, Any]:
        """Run a single agent on a Hanoi puzzle with specified number of disks."""
        optimal_moves = (2 ** num_disks) - 1
        max_moves = optimal_moves * self.MAX_MOVES_MULTIPLIER
        
        print(f"  {agent_label}: {agent.name} - {num_disks} disks (optimal: {optimal_moves}, max: {max_moves})")
        print(f"    Cumulative timeout: {self.total_game_timeout:.1f}s")
        
        # Start agent session for cumulative timeout tracking
        self.start_agent_session(agent)
        
        # Create fresh state for this puzzle
        agent_state = HanoiState(num_disks)
        move_history = []
        move_history_with_timing = []
        
        # Game loop
        while not agent_state.is_solved() and agent_state.move_count < max_moves:
            # Check cumulative timeout before each move
            remaining_time = self.get_remaining_time()
            if remaining_time <= 0:
                print(f"    {agent.name} exceeded cumulative timeout ({self.total_game_timeout:.1f}s)")
                return {
                    'success': False,
                    'moves': agent_state.move_count,
                    'move_history': move_history,
                    'move_history_with_timing': move_history_with_timing,
                    'timeout': True,
                    'efficiency': 0.0,
                    'final_state': agent_state,
                    'num_disks': num_disks,
                    'optimal_moves': optimal_moves
                }
            
            # Get current observation and action mask
            observation = agent_state.get_observation()
            action_mask = agent_state.get_action_mask()
            
            # Agent selects action with timeout detection
            action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
                agent, observation, action_mask, move_history, move_history_with_timing, agent_state.move_count
            )
            
            # Handle timeout
            if timeout_result is not None:
                print(f"    {agent.name} timed out on {num_disks}-disk variant")
                return {
                    'success': False,
                    'moves': agent_state.move_count,
                    'move_history': move_history,
                    'move_history_with_timing': move_history_with_timing,
                    'timeout': True,
                    'efficiency': 0.0,
                    'final_state': agent_state,
                    'num_disks': num_disks,
                    'optimal_moves': optimal_moves
                }
            
            if action is None or action < 0 or action >= 9:
                print(f"    Invalid action from {agent.name}: {action}")
                break
            
            # Convert action to move
            from_rod, to_rod = agent_state.action_to_move(action)
            
            # Attempt to make the move
            if agent_state.make_move(from_rod, to_rod):
                move_history.append((agent.name, action))
                move_history_with_timing.append((agent.name, action, decision_time))
                
                # Print move occasionally for larger puzzles
                if agent_state.move_count <= 10 or agent_state.move_count % max(5, optimal_moves // 5) == 0:
                    remaining = self.get_remaining_time()
                    print(f"    Move {agent_state.move_count}: rod {from_rod} -> rod {to_rod} ({decision_time:.2e}s, {remaining:.1f}s left)")
            else:
                print(f"    Illegal move attempted by {agent.name}: rod {from_rod} -> rod {to_rod}")
                break
        
        # Determine result
        success = agent_state.is_solved()
        efficiency = optimal_moves / max(agent_state.move_count, 1) if success else 0.0
        
        if success:
            print(f"    {agent.name} solved {num_disks}-disk in {agent_state.move_count} moves! (Efficiency: {efficiency:.2%})")
        else:
            if agent_state.move_count >= max_moves:
                print(f"    {agent.name} exceeded maximum moves limit ({max_moves}) on {num_disks}-disk")
            else:
                print(f"    {agent.name} failed to solve {num_disks}-disk puzzle")
        
        return {
            'success': success,
            'moves': agent_state.move_count,
            'move_history': move_history,
            'move_history_with_timing': move_history_with_timing,
            'timeout': False,
            'efficiency': efficiency,
            'final_state': agent_state,
            'num_disks': num_disks,
            'optimal_moves': optimal_moves
        }
    
    def _compare_agent_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """Compare two agent results and determine winner/scores."""
        assert self.agent1 is not None and self.agent2 is not None
        agent1_name = self.agent1.name
        agent2_name = self.agent2.name

        print(f"\nRESULTS COMPARISON:")
        indent = "  "

        print(f"{indent}{agent1_name}: {'Solved' if result1['success'] else 'Failed'} in {result1['moves']} moves")
        print(f"{indent}{agent2_name}: {'Solved' if result2['success'] else 'Failed'} in {result2['moves']} moves")
        
        # Determine winner based on success and efficiency
        if result1['success'] and result2['success']:
            # Both solved - compare moves (fewer is better)
            if result1['moves'] < result2['moves']:
                winner = agent1_name
                scores = {agent1_name: 1.0, agent2_name: 0.0}
                print(f"{indent}Winner: {agent1_name} (fewer moves: {result1['moves']} vs {result2['moves']})")
            elif result2['moves'] < result1['moves']:
                winner = agent2_name
                scores = {agent1_name: 0.0, agent2_name: 1.0}
                print(f"{indent}Winner: {agent2_name} (fewer moves: {result2['moves']} vs {result1['moves']})")
            else:
                winner = 'draw'
                scores = {agent1_name: 0.5, agent2_name: 0.5}
                print(f"{indent}Draw: Both solved in {result1['moves']} moves")
        elif result1['success'] and not result2['success']:
            winner = agent1_name
            scores = {agent1_name: 1.0, agent2_name: 0.0}
            print(f"{indent}Winner: {agent1_name} (only successful solver)")
        elif result2['success'] and not result1['success']:
            winner = agent2_name
            scores = {agent1_name: 0.0, agent2_name: 1.0}
            print(f"{indent}Winner: {agent2_name} (only successful solver)")
        else:
            # Neither solved - compare progress
            if result1['moves'] > result2['moves']:
                winner = agent1_name
                scores = {agent1_name: 1.0, agent2_name: 0.0}
                print(f"{indent}Winner: {agent1_name} (more progress: {result1['moves']} vs {result2['moves']} moves)")
            elif result2['moves'] > result1['moves']:
                winner = agent2_name
                scores = {agent1_name: 0.0, agent2_name: 1.0}
                print(f"{indent}Winner: {agent2_name} (more progress: {result2['moves']} vs {result1['moves']} moves)")
            else:
                winner = 'draw'
                scores = {agent1_name: 0.5, agent2_name: 0.5}
                print(f"{indent}Draw: Both failed after {result1['moves']} moves")
        
        return winner, scores
    
    def get_game_rules(self) -> str:
        """Get generic Hanoi Tower game rules description."""
        return """
Hanoi Tower Game Rules:

1. The puzzle consists of three rods and a variable number of disks of different sizes
2. Initially, all disks are stacked on the first rod in decreasing order of size (largest at bottom)
3. The goal is to move all disks to the third rod (rod 2)
4. Only one disk can be moved at a time
5. A disk can only be placed on top of a larger disk or on an empty rod
6. You win by moving all disks to the third rod in the minimum number of moves
7. You must solve the puzzle within the move limit or you will lose by timeout

Action Space:
- 9 possible actions representing moves from rod i to rod j
- Action encoding: action = from_rod * 3 + to_rod
- Valid actions: 1, 2, 3, 5, 6, 7 (rod i->j where i≠j)  
- Invalid actions: 0, 4, 8 (rod i->i, same rod moves)

Agent Interface:
- select_action(observation, action_mask) should return action index (0-8)
- observation: flattened state showing disk positions on all rods
- action_mask: boolean list indicating which actions are currently legal

Note: Write a GENERAL solution that works for any number of disks. The same code
will be tested on multiple variants with different disk counts.
        """
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get detailed observation format information for Hanoi Tower."""
        # Use current num_disks for example, but make it clear it's generic
        sample_state = HanoiState(self.num_disks)
        
        return {
            'description': f'Flattened representation of Hanoi Tower state (example with {self.num_disks} disks)',
            'observation_size': 'num_disks * 3 (3 rods, each can hold up to num_disks disks)',
            'sample_observation': sample_state.get_observation(),
            'action_space_size': 9,  # 3 rods x 3 rods = 9 possible moves
            'sample_action_mask': sample_state.get_action_mask(),
            'action_encoding': 'action = from_rod * 3 + to_rod',
            'disk_encoding': 'disk sizes 1 to num_disks, 0 for empty positions',
            'position_mapping': 'Actions 0-8 map to moves as: 0=rod0->rod0(invalid), 1=rod0->rod1, 2=rod0->rod2, 3=rod1->rod0, 4=rod1->rod1(invalid), 5=rod1->rod2, 6=rod2->rod0, 7=rod2->rod1, 8=rod2->rod2(invalid). Valid actions are 1,2,3,5,6,7. Observation positions 0-(n-1)=rod0, n-(2n-1)=rod1, 2n-(3n-1)=rod2.',
            'note': 'The observation size is fixed at 36 positions for the 12-disk challenge'
        }
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save a visualization of the multi-variant Hanoi Tower robin-round match results.
        
        Creates a side-by-side comparison of both agents' final states from their best performance.
        
        Args:
            save_path: Path where to save the visualization
            
        Returns:
            True if visualization was saved successfully, False otherwise
        """
        if not self.agent1_final_state or not self.agent2_final_state or not self.match_results:
            return False
        
        try:
            # Use non-interactive backend for WSL compatibility
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
            
            # Get the number of disks from the final states
            agent1_num_disks = self.agent1_final_state.num_disks
            agent2_num_disks = self.agent2_final_state.num_disks
            max_disks = max(agent1_num_disks, agent2_num_disks)
            
            # Setup both axes
            for ax in [ax1, ax2]:
                ax.set_xlim(0, 12)
                ax.set_ylim(0, max_disks + 4)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
            
            # Colors for disks
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'gray', 'brown', 'cyan', 'magenta', 'lime']
            rod_positions = [3, 6, 9]
            rod_names = ['Start', 'Auxiliary', 'Goal']
            
            # Helper function to draw a single Hanoi state
            def draw_hanoi_state(ax, state, agent_name):
                # Draw the base
                base = patches.Rectangle((1, 0), 10, 0.3, linewidth=1, edgecolor='black', facecolor='brown')
                ax.add_patch(base)
                
                # Draw rods and disks
                for i, x_pos in enumerate(rod_positions):
                    # Rod
                    rod = patches.Rectangle((x_pos - 0.1, 0.3), 0.2, max_disks + 1.5, 
                                          linewidth=1, edgecolor='black', facecolor='gray')
                    ax.add_patch(rod)
                    
                    # Rod label
                    ax.text(x_pos, -0.5, f'Rod {i}\n({rod_names[i]})', ha='center', va='top', fontsize=10, weight='bold')
                    
                    # Draw disks on this rod
                    tower = state.towers[i]
                    for j, disk_size in enumerate(tower):
                        disk_width = 0.3 + (disk_size - 1) * 0.3  # Scale disk width
                        disk_height = 0.3
                        disk_x = x_pos - disk_width / 2
                        disk_y = 0.3 + j * disk_height
                        
                        # Choose color based on disk size
                        color = colors[(disk_size - 1) % len(colors)]
                        
                        disk = patches.Rectangle((disk_x, disk_y), disk_width, disk_height,
                                               linewidth=2, edgecolor='black', facecolor=color, alpha=0.8)
                        ax.add_patch(disk)
                        
                        # Add disk number
                        ax.text(x_pos, disk_y + disk_height/2, str(disk_size), 
                               ha='center', va='center', fontsize=12, weight='bold', color='white')
                
                # Add title
                status = "SOLVED! " if state.is_solved() else "Failed "
                title = f'{agent_name}\n{status} ({state.num_disks} disks)'
                ax.set_title(title, fontsize=16, weight='bold', pad=20)
                
                # Add statistics
                efficiency = state.optimal_moves / max(state.move_count, 1) if state.move_count > 0 else 1.0
                stats_text = f'Moves: {state.move_count} | Efficiency: {efficiency:.1%}'
                if state.is_solved():
                    stats_color = 'lightgreen'
                else:
                    stats_color = 'lightcoral'
                    
                ax.text(6, max_disks + 2.5, stats_text, ha='center', va='center', fontsize=11, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=stats_color, alpha=0.7))
            
            # Check for valid agents
            if not self.agent1 or not self.agent2:
                print("Error: Agents not properly initialized for visualization")
                return False
            
            # Draw both agent states
            draw_hanoi_state(ax1, self.agent1_final_state, self.agent1.name)
            draw_hanoi_state(ax2, self.agent2_final_state, self.agent2.name)
            
            # Add overall match information
            winner = self.match_results['winner']
            if winner == 'draw':
                match_title = f'Hanoi Tower: {self.agent1.name} vs {self.agent2.name} - DRAW'
                title_color = 'orange'
            else:
                match_title = f'Hanoi Tower: {self.agent1.name} vs {self.agent2.name} - Winner: {winner}'
                title_color = 'green'

            fig.suptitle(match_title, fontsize=18, weight='bold', y=0.95, color=title_color)

            scores = self.match_results['scores']
            agent1_score = scores[self.agent1.name]
            agent2_score = scores[self.agent2.name]

            challenge_info = f'{self.num_disks}-Disk Hanoi Tower Challenge\n'
            challenge_info += f'Final Scores: {self.agent1.name}: {agent1_score:.3f} | {self.agent2.name}: {agent2_score:.3f}'
            
            fig.text(0.5, 0.02, challenge_info, ha='center', va='bottom', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            
            # Save the figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.1)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error saving Hanoi Tower visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset(self) -> None:
        """Reset game state for a new multi-variant match."""
        super().reset()
        
        # Reset visualization states
        self.agent1_final_state = None
        self.agent2_final_state = None
        self.match_results = None 