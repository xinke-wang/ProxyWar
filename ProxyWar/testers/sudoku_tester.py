"""
Sudoku specific tester implementation for ProxyWar framework.
"""

import time
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent
from ..games.sudoku import SudokuState


class SudokuTester(BaseTester):
    """Sudoku specific code tester for puzzle solving agents."""
    
    def __init__(self):
        super().__init__("SudokuTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated Sudoku agent code."""
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
        
        # Test 6: Solution format validation
        self._test_solution_format(agent_instance)
        
        # Test 7: Simple puzzle solving
        self._test_simple_puzzle_solving(agent_instance)
        
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
            # Test with a simple Sudoku state
            state = SudokuState(difficulty=0.3)  # Easy puzzle
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
            
            if not isinstance(action, (list, tuple)):
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent must return a list or tuple, got {type(action).__name__}"
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
    
    def _test_solution_format(self, agent_instance: BaseAgent) -> None:
        """Test if agent returns solutions in the correct format."""
        test_name = "Solution Format"
        
        try:
            state = SudokuState(difficulty=0.3)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
            solution = agent_instance.select_action(observation, action_mask)
            
            if solution is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned None solution"
                ))
                return
            
            if not isinstance(solution, (list, tuple)):
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Solution must be a list or tuple, got {type(solution).__name__}"
                ))
                return
            
            if len(solution) != 81:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Solution must have exactly 81 elements, got {len(solution)}"
                ))
                return
            
            # Check if all elements are integers between 1-9
            valid_values = True
            for i, val in enumerate(solution):
                if not isinstance(val, int) or val < 1 or val > 9:
                    valid_values = False
                    break
            
            if not valid_values:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Solution must contain only integers 1-9"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Solution format is correct"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing solution format: {str(e)}"
            ))
    
    def _test_simple_puzzle_solving(self, agent_instance: BaseAgent) -> None:
        """Test basic puzzle interaction without requiring actual solving."""
        test_name = "Basic Puzzle Interaction"
        
        try:
            # Test basic puzzle interaction
            state = SudokuState(difficulty=0.5)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
            # Just test that agent can respond without crashing
            solution = agent_instance.select_action(observation, action_mask)
            
            # Check if agent returns a response
            if solution is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned None solution"
                ))
                return
            
            # Check if solution format is correct (81 elements)
            if not isinstance(solution, (list, tuple)) or len(solution) != 81:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned invalid solution format: expected 81 elements, got {len(solution) if isinstance(solution, (list, tuple)) else 'non-list'}"
                ))
                return
            
            # Check if all elements are valid integers (1-9)
            valid_elements = True
            for i, val in enumerate(solution):
                if not isinstance(val, int) or val < 1 or val > 9:
                    valid_elements = False
                    break
            
            if not valid_elements:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned solution with invalid elements (must be integers 1-9)"
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent provides valid solution format and responds properly"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing basic puzzle interaction: {str(e)}"
            ))
    
    def _test_edge_cases(self, agent_instance: BaseAgent) -> None:
        """Test basic edge case handling."""
        test_name = "Edge Case Handling"
        
        try:
            # Test with standard difficulty
            state = SudokuState(difficulty=0.5)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
            # Test that agent can handle the call without crashing.
            # We catch broadly because the agent code under test is
            # untrusted LLM output: any exception type counts as a fail.
            try:
                solution = agent_instance.select_action(observation, action_mask)
                if solution is not None and not isinstance(solution, (list, tuple)):
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message="Agent returned invalid solution type"
                    ))
                    return
            except Exception as e:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent crashed during execution: {type(e).__name__}: {e}"
                ))
                return
            
            # Test with different observation (simulate different puzzle)
            state2 = SudokuState(difficulty=0.5)
            observation2 = state2.get_observation()
            action_mask2 = state2.get_action_mask()
            
            try:
                solution2 = agent_instance.select_action(observation2, action_mask2)
                if solution2 is not None and not isinstance(solution2, (list, tuple)):
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message="Agent returned invalid solution type for different puzzle"
                    ))
                    return
            except Exception as e:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent crashed on different puzzle: {type(e).__name__}: {e}"
                ))
                return
            
            # Basic consistency check - agent should return similar format
            if solution is not None and solution2 is not None:
                if len(solution) != len(solution2):
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message="Agent returned inconsistent solution lengths"
                    ))
                    return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent handles basic edge cases without crashing"
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
            state = SudokuState(difficulty=0.5)
            observation = state.get_observation()
            action_mask = state.get_action_mask()
            
            # Test response time
            start_time = time.time()
            solution = agent_instance.select_action(observation, action_mask)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time > 30.0:  # 30 seconds timeout for Sudoku solving
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent response time too slow: {response_time:.2f}s"
                ))
                return
            
            # Check if solution format is valid (not necessarily correct)
            format_msg = ""
            if solution is not None and isinstance(solution, (list, tuple)) and len(solution) == 81:
                # Check if all elements are valid integers (1-9)
                valid_elements = all(isinstance(val, int) and 1 <= val <= 9 for val in solution)
                if valid_elements:
                    format_msg = " (valid format)"
                else:
                    format_msg = " (invalid format)"
            elif solution is None:
                format_msg = " (no solution)"
            else:
                format_msg = " (invalid format)"
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent performance acceptable: {response_time:.2f}s{format_msg}"
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing performance: {str(e)}"
            )) 