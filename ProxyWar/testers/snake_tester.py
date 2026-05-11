"""
Snake specific tester implementation for ProxyWar framework.
"""

import time
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent


class SnakeTester(BaseTester):
    """Snake game specific code tester."""
    
    def __init__(self):
        super().__init__("SnakeTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated Snake agent code."""
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
        
        # Test 6: Edge cases
        self._test_edge_cases(agent_instance)
        
        # Test 7: Performance
        self._test_performance(agent_instance)
        
        end_time = time.time()
        return self.create_test_suite(agent_name, end_time - start_time)
    
    def _test_basic_behavior(self, agent_instance: BaseAgent) -> None:
        """Test basic behavior of the Snake agent."""
        test_name = "Basic Behavior"
        
        try:
            # Test with basic observation
            test_observation = [0] * 100  # 10x10 board
            
            # Place player 1 snake at position (2, 5) - index 5*10 + 2 = 52
            test_observation[52] = 1  # Player 1 snake head
            test_observation[51] = 1  # Player 1 snake body
            
            # Place player 2 snake at position (7, 5) - index 5*10 + 7 = 57
            test_observation[57] = 2  # Player 2 snake head
            test_observation[56] = 2  # Player 2 snake body
            
            # Place food at multiple positions
            test_observation[25] = 3  # Food at (5, 2)
            test_observation[45] = 3  # Food at (5, 4)
            test_observation[75] = 3  # Food at (5, 7)
            
            action_mask = [True, True, True, True]  # All directions available
            
            action = agent_instance.select_action(test_observation, action_mask)
            
            # Check if action is valid
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
                    message=f"Action should be int, got {type(action)}"
                ))
                return
            
            if action not in [0, 1, 2, 3]:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Action should be in [0,1,2,3], got {action}"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent returned valid action for realistic game scenario"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error in basic behavior test: {str(e)}"
            ))
    
    def _test_edge_cases(self, agent_instance: BaseAgent) -> None:
        """Test edge cases for the Snake agent."""
        test_name = "Edge Case Handling"
        
        try:
            # Test 1: Crowded scenario with multiple snakes and food
            test_observation = [0] * 100  # 10x10 board
            
            # Place player 1 snake (longer snake)
            test_observation[52] = 1  # Player 1 snake head at (2, 5)
            test_observation[51] = 1  # Player 1 snake body
            test_observation[50] = 1  # Player 1 snake body
            test_observation[49] = 1  # Player 1 snake tail
            
            # Place player 2 snake (adjacent positions)
            test_observation[57] = 2  # Player 2 snake head at (7, 5)
            test_observation[56] = 2  # Player 2 snake body
            test_observation[55] = 2  # Player 2 snake body
            
            # Place multiple foods
            test_observation[23] = 3  # Food at (3, 2)
            test_observation[35] = 3  # Food at (5, 3)
            test_observation[67] = 3  # Food at (7, 6)
            test_observation[83] = 3  # Food at (3, 8)
            test_observation[91] = 3  # Food at (1, 9)
            
            action_mask = [True, True, True, True]
            action = agent_instance.select_action(test_observation, action_mask)
            
            if action is None or not isinstance(action, int) or action not in [0, 1, 2, 3]:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent failed to handle crowded game scenario"
                ))
                return
            
            # Test 2: Action mask with some directions blocked
            restricted_action_mask = [True, False, True, False]  # Only UP and LEFT allowed
            action = agent_instance.select_action(test_observation, restricted_action_mask)
            
            if action is None or not isinstance(action, int) or action not in [0, 2]:  # Should be UP(0) or LEFT(2)
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent failed to respect action mask restrictions"
                ))
                return
            
            # Test 3: Near-border scenario
            border_observation = [0] * 100
            # Place snake near top border
            border_observation[5] = 1   # Player 1 snake head at (5, 0) - top edge
            border_observation[15] = 1  # Player 1 snake body at (5, 1)
            border_observation[94] = 2  # Player 2 snake head at (4, 9) - near bottom
            border_observation[84] = 2  # Player 2 snake body at (4, 8)
            border_observation[43] = 3  # Food at (3, 4)
            
            action = agent_instance.select_action(border_observation, action_mask)
            
            if action is None or not isinstance(action, int) or action not in [0, 1, 2, 3]:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent failed to handle border scenario"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent handled all edge cases appropriately"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error in edge case test: {str(e)}"
            ))
    
    def _test_performance(self, agent_instance: BaseAgent) -> None:
        """Test performance of the Snake agent."""
        test_name = "Performance"
        
        try:
            # Create a complex game scenario for performance testing
            test_observation = [0] * 100  # 10x10 board
            
            # Place longer snakes with multiple body segments
            test_observation[52] = 1  # Player 1 snake head at (2, 5)
            test_observation[51] = 1  # Player 1 snake body
            test_observation[50] = 1  # Player 1 snake body
            test_observation[49] = 1  # Player 1 snake body
            test_observation[48] = 1  # Player 1 snake tail
            
            test_observation[57] = 2  # Player 2 snake head at (7, 5)
            test_observation[56] = 2  # Player 2 snake body
            test_observation[55] = 2  # Player 2 snake body
            test_observation[54] = 2  # Player 2 snake body
            
            # Place multiple foods across the board
            test_observation[12] = 3  # Food at (2, 1)
            test_observation[33] = 3  # Food at (3, 3)
            test_observation[67] = 3  # Food at (7, 6)
            test_observation[85] = 3  # Food at (5, 8)
            test_observation[91] = 3  # Food at (1, 9)
            
            action_mask = [True, True, True, True]
            
            # Test response time
            start_time = time.time()
            action = agent_instance.select_action(test_observation, action_mask)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time > 1.0:  # More than 1 second is too slow
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent response time too slow: {response_time:.3f}s"
                ))
                return
            
            # Validate the action is reasonable
            if action is None or not isinstance(action, int) or action not in [0, 1, 2, 3]:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned invalid action in complex scenario"
                ))
                return
            
            # Test consistency - same input should give same output (if agent is deterministic)
            action2 = agent_instance.select_action(test_observation, action_mask)
            
            if action != action2:
                # Non-deterministic behavior is acceptable, just note it
                details = {"deterministic": False, "response_time": response_time}
            else:
                details = {"deterministic": True, "response_time": response_time}
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent performance acceptable (response time: {response_time:.3f}s)",
                details=details
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error in performance test: {str(e)}"
            )) 