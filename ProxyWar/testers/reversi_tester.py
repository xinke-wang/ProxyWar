"""
Reversi specific tester implementation for ProxyWar framework.
"""

import time
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent
from ..games.reversi import ReversiState


class ReversiTester(BaseTester):
    """Reversi specific code tester for two-player strategy games."""
    
    def __init__(self):
        super().__init__("ReversiTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated Reversi agent code."""
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
        
        # Test 7: Game interaction
        self._test_game_interaction(agent_instance)
        
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
            # Test with initial Reversi state
            state = ReversiState(board_size=8)
            observation = state.get_observation()
            action_mask = state.get_action_mask(1)  # Black player
            
            action = agent_instance.select_action(observation, action_mask)
            
            if action is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned None for valid game state"
                ))
                return
            
            if not isinstance(action, int):
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent must return an integer action, got {type(action).__name__}"
                ))
                return
            
            if action < 0 or action >= 64:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned action outside valid range (0-63): {action}"
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
        """Test if agent respects action mask and returns valid actions."""
        test_name = "Action Validation"
        
        try:
            # Test with initial Reversi state
            state = ReversiState(board_size=8)
            observation = state.get_observation()
            action_mask = state.get_action_mask(1)  # Black player
            
            # Test multiple times to ensure consistency
            for _ in range(5):
                action = agent_instance.select_action(observation, action_mask)
                
                if action is None:
                    continue  # None is acceptable if no legal moves
                
                if not isinstance(action, int):
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Action must be integer, got {type(action).__name__}"
                    ))
                    return
                
                if action < 0 or action >= 64:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Action {action} is out of valid range (0-63)"
                    ))
                    return
                
                # Check if action respects the action mask
                if not action_mask[action]:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Agent returned illegal action {action} (action_mask[{action}] is False)"
                    ))
                    return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent respects action mask and returns valid actions"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing action validation: {str(e)}"
            ))
    
    def _test_game_interaction(self, agent_instance: BaseAgent) -> None:
        """Test basic game interaction without requiring good play."""
        test_name = "Game Interaction"
        
        try:
            # Test that agent can play a few moves without crashing
            state = ReversiState(board_size=8)
            moves_played = 0
            max_test_moves = 10
            
            for _ in range(max_test_moves):
                current_player = state.current_player
                observation = state.get_observation()
                action_mask = state.get_action_mask(current_player)
                
                # Check if game is over or no legal moves
                if state.game_over or not any(action_mask):
                    break
                
                # Get agent action (only test with current player)
                if current_player == 1:  # Test with black player
                    action = agent_instance.select_action(observation, action_mask)
                    
                    if action is None:
                        # Agent can return None if no legal moves
                        state.pass_turn()
                        continue
                    
                    if not isinstance(action, int) or action < 0 or action >= 64:
                        self.add_test_result(TestResult(
                            test_name=test_name,
                            status=TestStatus.FAILED,
                            message=f"Agent returned invalid action {action} at move {moves_played + 1}"
                        ))
                        return
                    
                    # Convert action to position
                    row, col = state.action_to_position(action)
                    
                    # Make the move
                    if state.is_legal_move(row, col, current_player):
                        state.make_move(row, col, current_player)
                        moves_played += 1
                    else:
                        self.add_test_result(TestResult(
                            test_name=test_name,
                            status=TestStatus.FAILED,
                            message=f"Agent made illegal move at ({row}, {col}) on move {moves_played + 1}"
                        ))
                        return
                else:
                    # For white player, make a random legal move
                    legal_moves = state.get_legal_moves(current_player)
                    if legal_moves:
                        import random
                        row, col = random.choice(legal_moves)
                        state.make_move(row, col, current_player)
                    else:
                        state.pass_turn()
                
                if state.game_over:
                    break
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent successfully played {moves_played} moves without crashing"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing game interaction: {str(e)}"
            ))
    
    def _test_edge_cases(self, agent_instance: BaseAgent) -> None:
        """Test edge case handling."""
        test_name = "Edge Case Handling"
        
        try:
            # Test 1: No legal moves scenario
            state = ReversiState(board_size=8)
            # Create a scenario with no legal moves (simplified test)
            observation = [0] * 64  # Empty board
            action_mask = [False] * 64  # No legal moves
            
            action = agent_instance.select_action(observation, action_mask)
            if action is not None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent should return None when no legal moves, but returned: {action}"
                ))
                return
            
            # Test 2: Nearly full board
            state = ReversiState(board_size=8)
            # Fill most of the board
            for i in range(8):
                for j in range(8):
                    if i < 6 and j < 6:  # Leave some empty spaces
                        state.board[i, j] = 1 if (i + j) % 2 == 0 else 2
            
            observation = state.get_observation()
            action_mask = state.get_action_mask(1)
            
            action = agent_instance.select_action(observation, action_mask)
            if action is not None:
                if not isinstance(action, int) or action < 0 or action >= 64:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Invalid action for nearly full board: {action}"
                    ))
                    return
                
                if any(action_mask) and not action_mask[action]:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Agent returned illegal action {action} on nearly full board"
                    ))
                    return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent handles edge cases correctly"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing edge cases: {str(e)}"
            ))
    
    def _test_performance(self, agent_instance: BaseAgent) -> None:
        """Test agent performance (response time)."""
        test_name = "Performance"
        
        try:
            # Test with standard initial state
            state = ReversiState(board_size=8)
            observation = state.get_observation()
            action_mask = state.get_action_mask(1)
            
            # Test response time
            start_time = time.time()
            action = agent_instance.select_action(observation, action_mask)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time > 10.0:  # More than 10 seconds is too slow
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent response time too slow: {response_time:.2f}s"
                ))
                return
            
            # Test multiple calls to check consistency
            total_time = 0
            num_calls = 5
            
            for _ in range(num_calls):
                start_time = time.time()
                action = agent_instance.select_action(observation, action_mask)
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_time = total_time / num_calls
            
            if avg_time > 5.0:  # Average more than 5 seconds is concerning
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent average response time too slow: {avg_time:.2f}s"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent response time acceptable: avg {avg_time:.2e}s"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing performance: {str(e)}"
            )) 