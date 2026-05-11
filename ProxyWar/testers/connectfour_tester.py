"""
Connect Four specific tester implementation for ProxyWar framework.
"""

import time
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent


class ConnectFourTester(BaseTester):
    """Connect Four specific code tester."""
    
    def __init__(self):
        super().__init__("ConnectFourTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated Connect Four agent code."""
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
        """Test basic behavioral requirements of the agent."""
        test_name = "Basic Behavior"
        
        try:
            # Empty Connect Four board (6x7 = 42 positions)
            observation = [0] * 42
            action_mask = [True] * 7  # All columns available
            
            action = agent_instance.select_action(observation, action_mask)
            
            if action is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned None for valid game state"
                ))
                return
            
            if not isinstance(action, int) or action < 0 or action >= 7:
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
    
    def _test_edge_cases(self, agent_instance: BaseAgent) -> None:
        """Test edge case handling."""
        test_name = "Edge Case Handling"
        
        try:
            # Test: No legal moves (all columns full)
            # Create a board where all top row is filled
            observation = [1, 2, 1, 2, 1, 2, 1] + [0] * 35  # Top row full, rest empty
            action_mask = [False] * 7  # No columns available
            
            action = agent_instance.select_action(observation, action_mask)
            if action is not None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent should return None when no legal moves, but returned: {action}"
                ))
                return
            
            # Test: Only one legal move
            observation = [1, 2, 1, 2, 1, 2, 0] + [0] * 35  # Only column 6 available
            action_mask = [False, False, False, False, False, False, True]
            
            action = agent_instance.select_action(observation, action_mask)
            if action is None or action != 6:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent should select the only legal move (6), but returned: {action}"
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
            observation = [0] * 42  # Empty board
            action_mask = [True] * 7  # All columns available
            
            start_time = time.time()
            action = agent_instance.select_action(observation, action_mask)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time > 5.0:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent response time too slow: {response_time:.2f}s"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent response time acceptable: {response_time:.2e}s"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing performance: {str(e)}"
            )) 