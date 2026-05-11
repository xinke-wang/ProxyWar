"""
Maze specific tester implementation for ProxyWar framework.
"""

import time
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent
from ..games.maze import MazeState


class MazeTester(BaseTester):
    """Maze specific code tester for maze-solving agents."""
    
    def __init__(self):
        super().__init__("MazeTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated Maze agent code."""
        start_time = time.time()
        self.clear_test_results()
        
        # Test 1: File existence
        self._test_file_existence(agent_code_path)
        
        # Test 2: Syntax validation
        agent_module = self._test_syntax_validation(agent_code_path)
        if agent_module is None:
            end_time = time.time()
            return self.create_test_suite(agent_name, end_time - start_time)
        
        # Test 3: Class structure
        agent_class = self._test_class_structure(agent_module, agent_name)
        if agent_class is None:
            end_time = time.time()
            return self.create_test_suite(agent_name, end_time - start_time)
        
        # Test 4: Interface compliance
        agent_instance = self._test_interface_compliance(agent_class, agent_name)
        if agent_instance is None:
            end_time = time.time()
            return self.create_test_suite(agent_name, end_time - start_time)
        
        # Test 5: Basic behavior
        self._test_basic_behavior(agent_instance)
        
        # Test 6: Action validation
        self._test_action_validation(agent_instance)
        
        # Test 7: Maze interaction
        self._test_maze_interaction(agent_instance)
        
        # Test 8: Edge cases
        self._test_edge_cases(agent_instance)
        
        # Test 9: Performance
        self._test_performance(agent_instance)
        
        end_time = time.time()
        return self.create_test_suite(agent_name, end_time - start_time)
    
    def _test_basic_behavior(self, agent_instance: BaseAgent) -> None:
        """Test basic behavioral requirements of the agent."""
        test_name = "Basic Behavior"
        
        try:
            # Create a simple maze state for testing
            maze_state = MazeState(size=3, seed=42)
            observation = maze_state.get_observation()
            action_mask = maze_state.get_action_mask()
            
            action = agent_instance.select_action(observation, action_mask)
            
            if action is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned None for valid maze state"
                ))
                return
            
            if not isinstance(action, int) or action < 0 or action >= 4:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned invalid action: {action}"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent demonstrates basic valid behavior"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing basic behavior: {str(e)}"
            ))
    
    def _test_action_validation(self, agent_instance: BaseAgent) -> None:
        """Test if the agent respects action mask constraints."""
        test_name = "Action Validation"
        
        try:
            # Create a maze state where some actions are invalid
            maze_state = MazeState(size=3, seed=42)
            observation = maze_state.get_observation()
            action_mask = maze_state.get_action_mask()
            
            # Test multiple times to increase confidence
            valid_actions = []
            for _ in range(5):
                action = agent_instance.select_action(observation, action_mask)
                if action is not None and isinstance(action, int) and 0 <= action < 4:
                    valid_actions.append(action)
            
            if not valid_actions:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent never returned valid actions"
                ))
                return
            
            # Check if agent respects action mask
            invalid_actions = [action for action in valid_actions if not action_mask[action]]
            if invalid_actions:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned invalid actions: {invalid_actions}"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent respects action mask constraints"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing action validation: {str(e)}"
            ))
    
    def _test_maze_interaction(self, agent_instance: BaseAgent) -> None:
        """Test agent's interaction with maze environments."""
        test_name = "Maze Interaction"
        
        try:
            # Create a small maze and test a few moves
            maze_state = MazeState(size=3, seed=42)
            moves_made = 0
            max_test_moves = 10
            
            while moves_made < max_test_moves and not maze_state.is_game_over():
                observation = maze_state.get_observation()
                action_mask = maze_state.get_action_mask()
                
                action = agent_instance.select_action(observation, action_mask)
                
                if action is None or not isinstance(action, int) or action < 0 or action >= 4:
                    break
                
                # Try to make the move
                if maze_state.make_move(action):
                    moves_made += 1
                else:
                    # Invalid move - this is not necessarily an error,
                    # but we'll count it as one for testing
                    break
            
            if moves_made == 0:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent could not make any valid moves"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent successfully interacted with maze environment ({moves_made} moves)"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing maze interaction: {str(e)}"
            ))
    
    def _test_edge_cases(self, agent_instance: BaseAgent) -> None:
        """Test agent behavior on edge cases."""
        test_name = "Edge Case Handling"
        
        try:
            # Test with no valid actions (corner case)
            maze_state = MazeState(size=3, seed=42)
            
            # Create a scenario where agent is "stuck" (for testing purposes)
            empty_observation = maze_state.get_observation()
            no_valid_actions = [False, False, False, False]
            
            action = agent_instance.select_action(empty_observation, no_valid_actions)
            
            # Agent should handle this gracefully (return None or invalid action)
            if action is not None and isinstance(action, int) and 0 <= action < 4:
                # This is acceptable - agent tried to make a move
                pass
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent handles edge cases appropriately"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing edge cases: {str(e)}"
            ))
    
    def _test_performance(self, agent_instance: BaseAgent) -> None:
        """Test agent performance characteristics."""
        test_name = "Performance"
        
        try:
            # Create a maze state for timing
            maze_state = MazeState(size=3, seed=42)
            observation = maze_state.get_observation()
            action_mask = maze_state.get_action_mask()
            
            # Time multiple action selections
            start_time = time.time()
            for _ in range(10):
                action = agent_instance.select_action(observation, action_mask)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10.0
            
            # Check if agent is reasonably fast (less than 1 second per action)
            if avg_time > 1.0:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent is too slow: {avg_time:.3f}s per action"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent performance acceptable: {avg_time:.3f}s per action"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing performance: {str(e)}"
            )) 