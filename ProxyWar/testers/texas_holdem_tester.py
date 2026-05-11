"""
Texas Hold'em specific tester implementation for ProxyWar framework.
"""

import time
import traceback
from typing import Any, Optional

from .base import BaseTester, TestResult, TestSuite, TestStatus
from ..agents.base import BaseAgent


class TexasHoldemTester(BaseTester):
    """Texas Hold'em specific code tester."""
    
    def __init__(self):
        super().__init__("TexasHoldemTester")
    
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """Test the generated Texas Hold'em agent code."""
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
    
    def _test_basic_behavior(self, agent: BaseAgent) -> None:
        """Test basic Texas Hold'em behavior."""
        test_name = "Basic Behavior Test"
        try:
            start_time = time.time()
            
            # Test with a typical pre-flop scenario
            observation = {
                'hole_cards': ['AS', 'KH'],
                'community_cards': [],
                'current_bet': 2,
                'pot_size': 3,
                'your_chips': 100,
                'betting_round': 'pre_flop',
                'position': 'middle',
                'players_in_hand': 4,
                'legal_actions': ['Call', 'Raise', 'Fold'],
                'action_mask': [True, True, True, False]
            }
            
            action_mask = [True, True, True, False]  # Can Call, Raise, Fold, but not Check
            
            action = agent.select_action(observation, action_mask)
            
            # Validate action
            if action is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent returned None action for valid scenario",
                    execution_time=time.time() - start_time
                ))
                return
            
            if not isinstance(action, int):
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned non-integer action: {type(action)}",
                    execution_time=time.time() - start_time
                ))
                return
            
            if action not in [0, 1, 2]:  # Valid actions based on mask
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent returned invalid action: {action}",
                    execution_time=time.time() - start_time
                ))
                return
            
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Agent successfully returned valid action: {action}",
                execution_time=time.time() - start_time
            ))
            
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Exception during basic behavior test: {str(e)}",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                error_type=type(e).__name__,
                full_traceback=self._get_traceback()
            ))
    
    def _test_edge_cases(self, agent: BaseAgent) -> None:
        """Test edge cases for Texas Hold'em."""
        
        # Test 1: All-in scenario
        test_name = "All-in Scenario Test"
        try:
            start_time = time.time()
            
            observation = {
                'hole_cards': ['AS', 'AH'],
                'community_cards': ['AD', 'AC', '2S'],
                'current_bet': 50,
                'pot_size': 100,
                'your_chips': 50,  # Exactly enough to call
                'betting_round': 'flop',
                'position': 'late',
                'players_in_hand': 2,
                'legal_actions': ['Call', 'Fold'],
                'action_mask': [True, False, True, False]
            }
            
            action_mask = [True, False, True, False]  # Can Call or Fold only
            action = agent.select_action(observation, action_mask)
            
            if action is not None and action in [0, 2]:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.PASSED,
                    message=f"Agent handled all-in scenario correctly: {action}",
                    execution_time=time.time() - start_time
                ))
            else:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent failed all-in scenario: {action}",
                    execution_time=time.time() - start_time
                ))
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Exception during all-in test: {str(e)}",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                error_type=type(e).__name__,
                full_traceback=self._get_traceback()
            ))
        
        # Test 2: No legal actions (should return None)
        test_name = "No Legal Actions Test"
        try:
            start_time = time.time()
            
            observation = {
                'hole_cards': ['2S', '3H'],
                'community_cards': ['AD', 'KC', 'QS', 'JD', 'TH'],
                'current_bet': 0,
                'pot_size': 20,
                'your_chips': 0,
                'betting_round': 'river',
                'position': 'button',
                'players_in_hand': 1,
                'legal_actions': [],
                'action_mask': [False, False, False, False]
            }
            
            action_mask = [False, False, False, False]  # No legal actions
            action = agent.select_action(observation, action_mask)
            
            if action is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.PASSED,
                    message="Agent correctly returned None for no legal actions",
                    execution_time=time.time() - start_time
                ))
            else:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent should return None when no legal actions: {action}",
                    execution_time=time.time() - start_time
                ))
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Exception during no legal actions test: {str(e)}",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                error_type=type(e).__name__,
                full_traceback=self._get_traceback()
            ))
        
        # Test 3: Check scenario
        test_name = "Check Scenario Test"
        try:
            start_time = time.time()
            
            observation = {
                'hole_cards': ['7S', '8H'],
                'community_cards': ['9D', 'TC', 'JS'],
                'current_bet': 0,
                'pot_size': 10,
                'your_chips': 50,
                'betting_round': 'flop',
                'position': 'early',
                'players_in_hand': 3,
                'legal_actions': ['Raise', 'Fold', 'Check'],
                'action_mask': [False, True, True, True]
            }
            
            action_mask = [False, True, True, True]  # Can Raise, Fold, or Check
            action = agent.select_action(observation, action_mask)
            
            if action is not None and action in [1, 2, 3]:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.PASSED,
                    message=f"Agent handled check scenario correctly: {action}",
                    execution_time=time.time() - start_time
                ))
            else:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent failed check scenario: {action}",
                    execution_time=time.time() - start_time
                ))
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Exception during check scenario test: {str(e)}",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                error_type=type(e).__name__,
                full_traceback=self._get_traceback()
            ))
    
    def _test_performance(self, agent: BaseAgent) -> None:
        """Test performance requirements."""
        test_name = "Performance Test"
        try:
            start_time = time.time()
            
            observation = {
                'hole_cards': ['QS', 'JH'],
                'community_cards': ['QD', 'JC', 'TS', '9H'],
                'current_bet': 10,
                'pot_size': 50,
                'your_chips': 100,
                'betting_round': 'turn',
                'position': 'middle',
                'players_in_hand': 2,
                'legal_actions': ['Call', 'Raise', 'Fold'],
                'action_mask': [True, True, True, False]
            }
            
            action_mask = [True, True, True, False]
            
            # Test multiple decisions for performance
            decision_times = []
            for _ in range(5):
                decision_start = time.time()
                action = agent.select_action(observation, action_mask)
                decision_time = time.time() - decision_start
                decision_times.append(decision_time)
            
            avg_decision_time = sum(decision_times) / len(decision_times)
            max_decision_time = max(decision_times)
            
            # Performance requirements: average < 2s, max < 5s
            if avg_decision_time < 2.0 and max_decision_time < 5.0:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.PASSED,
                    message=f"Performance acceptable: avg={avg_decision_time:.3f}s, max={max_decision_time:.3f}s",
                    execution_time=time.time() - start_time
                ))
            else:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Performance too slow: avg={avg_decision_time:.3f}s, max={max_decision_time:.3f}s",
                    execution_time=time.time() - start_time
                ))
                
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Exception during performance test: {str(e)}",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                error_type=type(e).__name__,
                full_traceback=self._get_traceback()
            ))
    
    def _get_traceback(self) -> str:
        """Get the current traceback as a string."""
        return traceback.format_exc()