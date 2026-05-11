"""
Base tester classes for ProxyWar framework.

This module defines the interface for testing generated agent code
before they participate in actual battles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import importlib.util
import math
import os
import time
import traceback
import sys
import re
import signal
import threading
from functools import wraps

from ..agents.base import BaseAgent


class TimeoutError(Exception):
    """Exception raised when test execution times out."""
    pass


def timeout(seconds):
    """Decorator to add timeout functionality to test methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use signal-based timeout only on POSIX main thread
            use_signal = (
                threading.current_thread() is threading.main_thread()
                and hasattr(signal, 'SIGALRM')
            )
            if use_signal:
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Test execution timed out after {seconds} seconds")

                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(max(1, math.ceil(seconds)))

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Use threading-based timeout in worker threads
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=seconds)
                
                if thread.is_alive():
                    # Thread is still running, timeout occurred
                    raise TimeoutError(f"Test execution timed out after {seconds} seconds")
                
                if exception[0]:
                    raise exception[0]
                
                return result[0]
        
        return wrapper
    return decorator


class TestStatus(Enum):
    """Test status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """
    Represents the result of a single test.
    """
    test_name: str
    status: TestStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error_type: Optional[str] = None
    line_number: Optional[int] = None
    full_traceback: Optional[str] = None


@dataclass
class TestSuite:
    """
    Represents the results of a complete test suite.
    """
    agent_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    test_results: List[TestResult]
    total_execution_time: float
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of tests."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def is_passing(self) -> bool:
        """Check if all tests are passing."""
        return self.failed_tests == 0 and self.error_tests == 0


class BaseTester(ABC):
    """
    Abstract base class for all agent code testers in ProxyWar framework.
    
    Testers are responsible for validating generated agent code before
    they participate in actual battles.
    """
    
    def __init__(self, tester_name: str):
        """
        Initialize the base tester.
        
        Args:
            tester_name: Name identifier for this tester
        """
        self.tester_name = tester_name
        self.test_results: List[TestResult] = []
    
    def test_agent_code(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """
        Test the generated agent code with timeout protection.
        
        Args:
            agent_code_path: Path to the generated agent code file
            agent_name: Name of the agent being tested
            
        Returns:
            TestSuite containing all test results
        """
        try:
            return self._test_agent_code_with_timeout(agent_code_path, agent_name)
        except TimeoutError as e:
            # Create a timeout test result
            timeout_result = TestResult(
                test_name="Timeout Protection",
                status=TestStatus.FAILED,
                message=str(e),
                error_type="timeout"
            )
            
            # Clear any existing results and add timeout result
            self.clear_test_results()
            self.add_test_result(timeout_result)
            
            return TestSuite(
                agent_name=agent_name,
                total_tests=1,
                passed_tests=0,
                failed_tests=1,
                error_tests=0,
                skipped_tests=0,
                test_results=[timeout_result],
                total_execution_time=30.0
            )
    
    @timeout(30)  # 30-second timeout
    def _test_agent_code_with_timeout(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """
        Internal method with timeout protection that calls the actual test implementation.
        
        Args:
            agent_code_path: Path to the generated agent code file
            agent_name: Name of the agent being tested
            
        Returns:
            TestSuite containing all test results
        """
        return self._test_agent_code_impl(agent_code_path, agent_name)
    
    @abstractmethod
    def _test_agent_code_impl(self, agent_code_path: str, agent_name: str) -> TestSuite:
        """
        Test the generated agent code implementation.
        
        Args:
            agent_code_path: Path to the generated agent code file
            agent_name: Name of the agent being tested
            
        Returns:
            TestSuite containing all test results
        """
        pass
    
    def add_test_result(self, result: TestResult) -> None:
        """Add a test result to the current test session."""
        self.test_results.append(result)
    
    def clear_test_results(self) -> None:
        """Clear all test results."""
        self.test_results.clear()
    
    def create_test_suite(self, agent_name: str, total_execution_time: float) -> TestSuite:
        """
        Create a test suite from current test results.
        
        Args:
            agent_name: Name of the agent being tested
            total_execution_time: Total time taken for all tests
            
        Returns:
            TestSuite object containing all results
        """
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        error = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
        
        return TestSuite(
            agent_name=agent_name,
            total_tests=len(self.test_results),
            passed_tests=passed,
            failed_tests=failed,
            error_tests=error,
            skipped_tests=skipped,
            test_results=self.test_results.copy(),
            total_execution_time=total_execution_time
        )
    
    def classify_error(self, error_message: str, error_type: Optional[type] = None) -> str:
        """
        Classify error into categories for detailed analysis.
        
        Args:
            error_message: The error message string
            error_type: The exception type (if available)
            
        Returns:
            String classification of the error
        """
        error_msg_lower = error_message.lower()
        
        # Direct type-based classification
        if error_type:
            if error_type.__name__ in ['SyntaxError', 'IndentationError', 'TabError']:
                return "syntax_error"
            elif error_type.__name__ in ['NameError', 'AttributeError', 'ImportError', 'ModuleNotFoundError']:
                return "import_error"
            elif error_type.__name__ in ['IndexError', 'KeyError', 'ValueError', 'TypeError']:
                return "runtime_error"
            elif error_type.__name__ in ['TimeoutError']:
                return "timeout"
            elif error_type.__name__ in ['AssertionError']:
                return "logic_error"
        
        # Message-based classification
        if any(keyword in error_msg_lower for keyword in ['syntaxerror', 'invalid syntax', 'indentationerror']):
            return "syntax_error"
        elif any(keyword in error_msg_lower for keyword in ['nameerror', 'attributeerror', 'importerror', 'modulenotfounderror']):
            return "import_error"
        elif any(keyword in error_msg_lower for keyword in ['indexerror', 'keyerror', 'valueerror', 'typeerror']):
            return "runtime_error"
        elif any(keyword in error_msg_lower for keyword in ['timeout', 'time limit']):
            return "timeout"
        elif any(keyword in error_msg_lower for keyword in ['assertion', 'failed test', 'wrong action']):
            return "logic_error"
        else:
            return "unknown_error"
    
    def extract_line_number(self, traceback_str: str) -> Optional[int]:
        """
        Extract line number from traceback string.
        
        Args:
            traceback_str: Full traceback string
            
        Returns:
            Line number if found, None otherwise
        """
        try:
            # Look for pattern like "line 123" in traceback
            pattern = r'line (\d+)'
            matches = re.findall(pattern, traceback_str, re.IGNORECASE)
            if matches:
                return int(matches[-1])  # Return the last line number found
        except (ValueError, IndexError):
            pass
        return None
    
    def create_detailed_test_result(self, test_name: str, status: TestStatus, message: str, 
                                    exception: Optional[Exception] = None, execution_time: Optional[float] = None,
                                    details: Optional[Dict[str, Any]] = None) -> TestResult:
        """
        Create a detailed test result with error classification.
        
        Args:
            test_name: Name of the test
            status: Test status
            message: Test message
            exception: Exception object (if test failed)
            execution_time: Time taken for the test
            details: Additional details
            
        Returns:
            Detailed TestResult object
        """
        error_type = None
        line_number = None
        full_traceback = None
        
        if exception:
            error_type = self.classify_error(str(exception), type(exception))
            full_traceback = traceback.format_exc()
            line_number = self.extract_line_number(full_traceback)
        
        return TestResult(
            test_name=test_name,
            status=status,
            message=message,
            details=details,
            execution_time=execution_time,
            error_type=error_type,
            line_number=line_number,
            full_traceback=full_traceback
        )

    def _test_file_existence(self, agent_code_path: str) -> None:
        """Test if the agent code file exists, is readable, and is non-empty."""
        test_name = "File Existence"

        try:
            if not os.path.exists(agent_code_path):
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Agent code file does not exist: {agent_code_path}"
                ))
                return

            with open(agent_code_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    self.add_test_result(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        message="Agent code file is empty"
                    ))
                    return

            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="File exists and is readable"
            ))

        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error reading file: {str(e)}"
            ))

    def _test_syntax_validation(self, agent_code_path: str) -> Optional[Any]:
        """Load the agent code as a module to validate its Python syntax."""
        test_name = "Syntax Validation"

        try:
            spec = importlib.util.spec_from_file_location("agent_module", agent_code_path)
            if spec is None or spec.loader is None:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Could not create module spec"
                ))
                return None

            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)

            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Code has valid Python syntax"
            ))

            return agent_module

        except SyntaxError as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                message=f"Syntax error: {str(e)}"
            ))
            return None
        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error loading module: {str(e)}"
            ))
            return None

    def _test_class_structure(self, agent_module: Any, agent_name: str) -> Optional[Any]:
        """Reflect on the loaded module to find a BaseAgent subclass with select_action."""
        test_name = "Class Structure"

        try:
            agent_classes = []
            for attr_name in dir(agent_module):
                attr = getattr(agent_module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseAgent) and
                    attr != BaseAgent):
                    agent_classes.append(attr)

            if not agent_classes:
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="No agent class found that inherits from BaseAgent"
                ))
                return None

            agent_class = agent_classes[0]

            if not hasattr(agent_class, 'select_action'):
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Missing required method: select_action"
                ))
                return None

            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message=f"Found valid agent class: {agent_class.__name__}"
            ))

            return agent_class

        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error analyzing class structure: {str(e)}"
            ))
            return None

    def _test_interface_compliance(self, agent_class: Any, agent_name: str) -> Optional[BaseAgent]:
        """Instantiate the agent class and verify it returns a BaseAgent."""
        test_name = "Interface Compliance"

        try:
            agent_instance = agent_class(agent_name)

            if not isinstance(agent_instance, BaseAgent):
                self.add_test_result(TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message="Agent instance is not a BaseAgent"
                ))
                return None

            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                message="Agent complies with required interface"
            ))

            return agent_instance

        except Exception as e:
            self.add_test_result(TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Error testing interface compliance: {str(e)}"
            ))
            return None

    def __str__(self) -> str:
        """String representation of the tester."""
        return f"{self.__class__.__name__}(name='{self.tester_name}')"