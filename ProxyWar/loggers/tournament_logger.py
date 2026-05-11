"""
Tournament Logger for ProxyWar framework
"""

import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from .base_logger import BaseLogger, LogLevel


@dataclass
class TestFailureRecord:
    """Record of test failure details"""
    test_name: str
    error_type: str  # "syntax_error", "runtime_error", "logic_error", "timeout"
    error_message: str
    line_number: Optional[int] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class CoderTestingRecord:
    """Record of coder testing phase"""
    coder_name: str
    game: str
    revision_count: int = 0
    total_revision_time: float = 0.0
    test_passed: bool = False
    test_failures: List[TestFailureRecord] = field(default_factory=list)
    final_code: str = ""


@dataclass
class MatchRecord:
    """Record of a single match"""
    match_id: str
    game: str
    player1: str
    player2: str
    player1_role: str  # "first" or "second"
    player2_role: str  # "first" or "second" 
    winner: str
    final_scores: Dict[str, float]
    move_history: List[tuple]  # [(player_name, action, decision_time), ...]
    match_duration: float
    total_moves: int
    final_board_state: List[int]
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class TournamentLogger(BaseLogger):
    """
    Specialized logger for tournament operations.
    
    Extends BaseLogger with tournament-specific functionality including
    performance tracking, match recording, and detailed statistics.
    """
    
    def __init__(self, name: str = "Tournament", log_level: LogLevel = LogLevel.INFO):
        """Initialize tournament logger."""
        super().__init__(name, log_level)
        
        # Data storage
        self.coder_testing_records: Dict[str, CoderTestingRecord] = {}
        self.match_records: List[MatchRecord] = []
        
    def experiment_start(self):
        """Log experiment start."""
        self.section_header("PROXYWAR TOURNAMENT SYSTEM")
        self.info("Tournament logging system initialized")
    
    def log_coder_testing_start(self, coder_name: str, game: str):
        """Start recording coder testing phase."""
        record_key = f"{coder_name}_{game}"
        self.coder_testing_records[record_key] = CoderTestingRecord(
            coder_name=coder_name,
            game=game
        )
    
    def log_test_failure(self, coder_name: str, game: str, test_name: str, 
                        error_type: str, error_message: str, line_number: Optional[int] = None):
        """Log a test failure for a coder."""
        record_key = f"{coder_name}_{game}"
        if record_key in self.coder_testing_records:
            failure_record = TestFailureRecord(
                test_name=test_name,
                error_type=error_type,
                error_message=error_message,
                line_number=line_number
            )
            self.coder_testing_records[record_key].test_failures.append(failure_record)
    
    def log_revision_attempt(self, coder_name: str, game: str, revision_num: int, success: bool):
        """Log a code revision attempt."""
        record_key = f"{coder_name}_{game}"
        if record_key in self.coder_testing_records:
            self.coder_testing_records[record_key].revision_count = revision_num
            if success:
                self.coder_testing_records[record_key].test_passed = True
    
    def log_coder_testing_complete(self, coder_name: str, game: str, 
                                  revision_time: float, final_code: str, success: bool):
        """Complete coder testing record."""
        record_key = f"{coder_name}_{game}"
        if record_key in self.coder_testing_records:
            record = self.coder_testing_records[record_key]
            record.total_revision_time = revision_time
            record.final_code = final_code
            record.test_passed = success
    
    def log_match(self, match_record: MatchRecord):
        """Log a tournament match."""
        self.match_records.append(match_record)
    
    def classify_error(self, error_message: str) -> str:
        """Classify error message into error type."""
        error_msg_lower = error_message.lower()
        
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
    
    def save_detailed_logs(self, base_dir: str):
        """Save all detailed logs to files."""
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Import the conversion function
        from ..evaluations.tournament_manager import _convert_to_serializable
        
        # Save testing records
        testing_data = {
            'timestamp': datetime.now().isoformat(),
            'coder_testing_records': {
                key: _convert_to_serializable(record) for key, record in self.coder_testing_records.items()
            }
        }
        
        testing_file = os.path.join(results_dir, "coder_testing_details.json")
        with open(testing_file, 'w', encoding='utf-8') as f:
            json.dump(testing_data, f, indent=2, ensure_ascii=False)
        
        # Save match records
        match_data = {
            'timestamp': datetime.now().isoformat(),
            'match_records': [_convert_to_serializable(record) for record in self.match_records]
        }
        
        match_file = os.path.join(results_dir, "detailed_match_history.json")
        with open(match_file, 'w', encoding='utf-8') as f:
            json.dump(match_data, f, indent=2, ensure_ascii=False)
    
    def get_coder_statistics(self, coder_name: str, game: str) -> Dict[str, Any]:
        """Get statistics for a specific coder."""
        record_key = f"{coder_name}_{game}"
        if record_key not in self.coder_testing_records:
            return {}
        
        record = self.coder_testing_records[record_key]
        
        # Count error types
        error_counts = {}
        for failure in record.test_failures:
            error_type = failure.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'revision_count': record.revision_count,
            'total_revision_time': record.total_revision_time,
            'test_passed': record.test_passed,
            'total_test_failures': len(record.test_failures),
            'error_type_counts': error_counts,
            'failure_details': [
                {
                    'test_name': f.test_name,
                    'error_type': f.error_type,
                    'error_message': f.error_message[:200]  # Truncate long messages
                }
                for f in record.test_failures
            ]
        } 