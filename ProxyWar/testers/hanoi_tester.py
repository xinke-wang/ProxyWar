"""
Hanoi Tower specific tester implementation for ProxyWar framework.
"""

import time
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent
from ..games.hanoi import HanoiState


class HanoiTowerTester(BaseTester):
    """Hanoi Tower specific code tester for single-player puzzle solving."""
    
    def __init__(self):
        super().__init__("HanoiTowerTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated Hanoi Tower agent code."""
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
        
        # Test 6: Simple puzzle solving
        self._test_simple_puzzle_solving(agent_instance)
        
        # Test 7: Edge cases
        self._test_edge_cases(agent_instance)
        
        # Test 8: Performance and timeout resistance
        self._test_performance(agent_instance)
        
        # Test 9: Different puzzle sizes
        self._test_scalability(agent_instance)
        
        end_time = time.time()
        return self.create_test_suite(agent_name, end_time - start_time)
    
    def _test_basic_behavior(self, agent_instance: BaseAgent) -> None:
        """Test basic behavioral requirements of the agent."""
        test_name = "Basic Behavior"
        
        try:
            # Test with 3-disk initial state
            state = HanoiState(3)
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
            
            if not isinstance(action, int) or action < 0 or action >= 9:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned invalid action: {action}"
                ))
                return
            
            # Check if the action is legal
            if not action_mask[action]:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned illegal action: {action}"
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
    
    def _test_simple_puzzle_solving(self, agent_instance: BaseAgent) -> None:
        """Test if agent returns valid actions for different game states."""
        test_name = "Simple Puzzle Solving"
        
        try:
            # Test on different states to ensure agent can handle various scenarios
            test_states = [
                (3, "3-disk initial state"),
                (4, "4-disk initial state"),
                (2, "2-disk initial state")
            ]
            
            for num_disks, description in test_states:
                state = HanoiState(num_disks)
                observation = state.get_observation()
                action_mask = state.get_action_mask()
                
                # Test that agent returns valid action
                action = agent_instance.select_action(observation, action_mask)
                
                if action is None:
                    # None is acceptable if no legal moves (shouldn't happen in initial state)
                    if not any(action_mask):
                        continue
                    else:
                        self.add_test_result(TestResult(
                            test_name=test_name,
                            status=TestStatus.FAILED,
                            message=f"Agent returned None when legal moves available in {description}"
                        ))
                        return
                
                if not isinstance(action, int):
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Agent returned non-integer action: {action} for {description}"
                    ))
                    return
                
                if action < 0 or action >= 9:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Action out of range: {action} for {description}"
                    ))
                    return
                
                if not action_mask[action]:
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message=f"Agent chose illegal action: {action} for {description}"
                    ))
                    return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent returns valid actions for different game states"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing simple puzzle solving: {str(e)}"
            ))
    
    def _test_edge_cases(self, agent_instance: BaseAgent) -> None:
        """Test edge case handling."""
        test_name = "Edge Case Handling"
        
        try:
            # Test 1: Different puzzle sizes - just test interface
            for num_disks in [2, 3, 4]:
                state = HanoiState(num_disks)
                observation = state.get_observation()
                action_mask = state.get_action_mask()
                
                # Test that agent can handle different observation sizes
                action = agent_instance.select_action(observation, action_mask)
                
                if action is not None:
                    if not isinstance(action, int) or action < 0 or action >= 9:
                        self.add_test_result(TestResult(
                            test_name=test_name,
                            status=TestStatus.FAILED,
                            message=f"Invalid action type/range for {num_disks}-disk: {action}"
                        ))
                        return
                    
                    if not action_mask[action]:
                        self.add_test_result(TestResult(
                            test_name=test_name,
                            status=TestStatus.FAILED,
                            message=f"Illegal action for {num_disks}-disk: {action}"
                        ))
                        return
            
            # Test 2: No legal moves
            state = HanoiState(2)
            observation = state.get_observation()
            fake_action_mask = [False] * 9
            action = agent_instance.select_action(observation, fake_action_mask)
            if action is not None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent should return None when no legal moves, but returned: {action}"
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
        """Test agent performance (response time only)."""
        test_name = "Performance"
        
        try:
            state = HanoiState(3)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
            # Test response time - single call
            start_time = time.time()
            action = agent_instance.select_action(observation, action_mask)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time > 5.0:  # Reasonable timeout for interface test
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent response time too slow: {response_time:.2f}s"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent performance acceptable: {response_time:.2e}s"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing performance: {str(e)}"
            ))
    
    def _test_scalability(self, agent_instance: BaseAgent) -> None:
        """Test agent interface compliance with different puzzle sizes."""
        test_name = "Scalability"
        
        try:
            puzzle_sizes = [2, 4, 5]  # Test different difficulty levels
            results = []
            
            for num_disks in puzzle_sizes:
                try:
                    state = HanoiState(num_disks)
                    observation = state.get_observation()
                    action_mask = state.get_action_mask()
                    
                    start_time = time.time()
                    action = agent_instance.select_action(observation, action_mask)
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    
                    # Only check interface compliance, not puzzle solving
                    if action is None:
                        if any(action_mask):  # Should not be None if legal moves exist
                            results.append(f"{num_disks}-disk: INVALID_NONE")
                        else:
                            results.append(f"{num_disks}-disk: OK(no_moves)")
                    elif not isinstance(action, int) or action < 0 or action >= 9:
                        results.append(f"{num_disks}-disk: INVALID_ACTION({action})")
                    elif not action_mask[action]:
                        results.append(f"{num_disks}-disk: ILLEGAL_ACTION({action})")
                    elif response_time > 10.0:  # More lenient timeout
                        results.append(f"{num_disks}-disk: TOO_SLOW({response_time:.1f}s)")
                    else:
                        results.append(f"{num_disks}-disk: OK({response_time:.2e}s)")
                        
                except Exception as e:
                    error_msg = str(e)[:30]  # Truncate long error messages
                    results.append(f"{num_disks}-disk: ERROR({error_msg})")
            
            # Check if agent handles all sizes without errors
            error_count = sum(1 for r in results if 'ERROR' in r or 'INVALID' in r or 'ILLEGAL' in r)
            if error_count > 0:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent failed on {error_count}/{len(puzzle_sizes)} puzzle sizes: {'; '.join(results)}"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent interface compliant across puzzle sizes: {'; '.join(results)}"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing scalability: {str(e)}"
            )) 