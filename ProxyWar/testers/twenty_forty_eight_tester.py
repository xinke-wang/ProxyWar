"""
2048 specific tester implementation for ProxyWar framework.
"""

import time
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent
from ..games.twenty_forty_eight import TwentyFortyEightState


class TwentyFortyEightTester(BaseTester):
    """2048 specific code tester for game-playing agents."""
    
    def __init__(self):
        super().__init__("TwentyFortyEightTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated 2048 agent code."""
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
            # Test with a simple 2048 state
            state = TwentyFortyEightState(seed=42)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
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
        """Test if agent returns valid actions."""
        test_name = "Action Validation"
        
        try:
            state = TwentyFortyEightState(seed=42)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
            # Test multiple times to check consistency
            valid_actions = 0
            for _ in range(5):
                action = agent_instance.select_action(observation, action_mask)
                
                if action is None:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message="Agent returned None action"
                    ))
                    return
                
                if not isinstance(action, int):
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Action must be an integer, got {type(action).__name__}"
                    ))
                    return
                
                if action < 0 or action > 3:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Action must be in range 0-3, got {action}"
                    ))
                    return
                
                valid_actions += 1
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent returns valid actions (tested {valid_actions} times)"
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
            state = TwentyFortyEightState(seed=123)
            moves_played = 0
            max_test_moves = 20
            
            for _ in range(max_test_moves):
                observation = state.get_observation()
                action_mask = state.get_action_mask()
                
                # Check if game is over
                if not any(action_mask):
                    break
                
                # Get agent action
                action = agent_instance.select_action(observation, action_mask)
                
                if action is None:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Agent returned None action at move {moves_played + 1}"
                    ))
                    return
                
                if not isinstance(action, int) or action < 0 or action > 3:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Agent returned invalid action {action} at move {moves_played + 1}"
                    ))
                    return
                
                # Make the move
                state.make_move(action)
                moves_played += 1
                
                if state.is_game_over():
                    break
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent successfully played {moves_played} moves without errors"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing game interaction: {str(e)}"
            ))
    
    def _test_edge_cases(self, agent_instance: BaseAgent) -> None:
        """Test basic edge case handling."""
        test_name = "Edge Case Handling"
        
        try:
            # Test 1: Empty action mask (no valid moves)
            # Create a state and simulate no valid moves
            state = TwentyFortyEightState(seed=42)
            observation = state.get_observation()
            empty_action_mask = [False, False, False, False]
            
            # Agent should still return an action without crashing.
            # Broad except is intentional: agent code is untrusted LLM
            # output, any exception counts as a fail.
            try:
                action = agent_instance.select_action(observation, empty_action_mask)
                # We don't care what action is returned, just that it doesn't crash
            except Exception as e:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent crashed when no valid moves available: {type(e).__name__}: {e}"
                ))
                return
            
            # Test 2: Different game states
            seeds = [100, 200, 300]
            for seed in seeds:
                state = TwentyFortyEightState(seed=seed)
                observation = state.get_observation()
                action_mask = state.get_action_mask()
                
                try:
                    action = agent_instance.select_action(observation, action_mask)
                    if action is not None and (not isinstance(action, int) or action < 0 or action > 3):
                        self.add_test_result(TestResult(
                            test_name=test_name,
                            status=TestStatus.FAILED,
                            message=f"Agent returned invalid action for seed {seed}"
                        ))
                        return
                except Exception as e:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Agent crashed on game with seed {seed}: {type(e).__name__}: {e}"
                    ))
                    return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent handles edge cases without crashing"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing edge cases: {str(e)}"
            ))
    
    def _test_performance(self, agent_instance: BaseAgent) -> None:
        """Test agent response time performance."""
        test_name = "Performance"
        
        try:
            state = TwentyFortyEightState(seed=42)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
            # Test response time
            start_time = time.time()
            action = agent_instance.select_action(observation, action_mask)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time > 5.0:  # 5 seconds timeout per move
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent response time too slow: {response_time:.2f}s"
                ))
                return
            
            # Test multiple moves to get average
            total_time = response_time
            moves_tested = 1
            
            for _ in range(9):  # Test 10 moves total
                if state.is_game_over():
                    break
                
                observation = state.get_observation()
                action_mask = state.get_action_mask()
                
                if not any(action_mask):
                    break
                
                start_time = time.time()
                action = agent_instance.select_action(observation, action_mask)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                moves_tested += 1
                
                if action is not None and isinstance(action, int) and 0 <= action <= 3:
                    state.make_move(action)
            
            avg_time = total_time / moves_tested
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent performance acceptable: avg {avg_time:.3f}s per move ({moves_tested} moves tested)"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing performance: {str(e)}"
            )) 