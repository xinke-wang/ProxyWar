"""
Data models for ProxyWar evaluation framework.

This module contains all data classes used across the evaluation system,
including results, statistics, and tournament data structures.
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import trueskill as ts

from ..agents import BaseAgent
from ..coders.base import BaseCoder


@dataclass
class CoderResult:
    """Results of a coder's testing phase."""
    name: str
    coder: BaseCoder
    code_file: Optional[str] = None
    passed_tests: bool = False
    agent_instance: Optional[BaseAgent] = None
    trueskill_rating: ts.Rating = field(default_factory=lambda: ts.Rating())
    conservative_rating: float = 0.0
    skill_estimate: float = 25.0
    uncertainty: float = 8.333
    wins: int = 0
    losses: int = 0
    draws: int = 0
    revision_count: int = 0
    total_revision_time: float = 0.0
    average_decision_time: float = 0.0
    
    # Extended evaluation metrics
    in_game_errors: List[Dict[str, str]] = field(default_factory=list)
    in_game_error_count: int = 0
    test_failures: List[Dict[str, str]] = field(default_factory=list)
    decision_times: List[float] = field(default_factory=list)
    code_generation_time: float = 0.0
    total_testing_time: float = 0.0
    
    # Timeout tracking
    timeout_count: int = 0
    timeout_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Single-player game metrics
    single_player_challenges: int = 0
    single_player_successes: int = 0
    single_player_failures: int = 0
    challenge_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Test statistics from the initial (first-version) test attempt.
    # Note: `passed_tests` (bool) above is a separate field meaning "did the
    # coder pass overall after revisions"; do not conflate with passed_test_count.
    total_tests: int = 0
    passed_test_count: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    
    @property
    def total_game_failures(self) -> int:
        """Total number of game failures (losses + in-game errors)"""
        return self.losses + self.in_game_error_count
    
    @property
    def success_rate(self) -> float:
        """Success rate in games (wins / total_games for multiplayer, successes / attempts for single-player)"""
        if self.single_player_challenges > 0:
            # Single-player game success rate
            return self.single_player_successes / self.single_player_challenges
        else:
            # Multiplayer game success rate
            total_games = self.wins + self.losses + self.draws + self.in_game_error_count
            return self.wins / total_games if total_games > 0 else 0.0
    
    @property
    def single_player_success_rate(self) -> float:
        """Success rate specifically for single-player challenges"""
        return self.single_player_successes / self.single_player_challenges if self.single_player_challenges > 0 else 0.0
    
    @property
    def avg_decision_time(self) -> float:
        """Average decision time across all moves"""
        return sum(self.decision_times) / len(self.decision_times) if self.decision_times else 0.0
    
    @property
    def max_decision_time(self) -> float:
        """Maximum decision time in a single move"""
        return max(self.decision_times) if self.decision_times else 0.0


@dataclass
class MatchResult:
    """Result of a single match between agents (supports both 2-player and multi-player)."""
    winner: str  # agent name or "draw"
    scores: Dict[str, float]  # agent name -> score
    moves: int
    match_duration: float
    match_history: List[Any]  # Game-specific move history
    player_decision_times: Dict[str, List[float]]
    avg_decision_times: Dict[str, float]
    match_number: int = 0
    
    # Legacy 2-player fields (for backward compatibility)
    agent1_name: str = ""
    agent2_name: str = ""
    agent1_score: float = 0.0
    agent2_score: float = 0.0
    agent1_role: str = ""  # "first" or "second" 
    agent2_role: str = ""  # "first" or "second"
    move_history: List[Tuple[str, int, float]] = field(default_factory=list)  # (player_name, action, decision_time)
    final_board_state: List[int] = field(default_factory=list)  # Final state of the game board
    
    # Multi-player specific fields
    multi_player_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize legacy fields for backward compatibility."""
        if not self.agent1_name and not self.agent2_name:
            # Try to extract from scores for 2-player games
            if len(self.scores) == 2:
                agents = list(self.scores.keys())
                self.agent1_name = agents[0]
                self.agent2_name = agents[1]
                self.agent1_score = self.scores[agents[0]]
                self.agent2_score = self.scores[agents[1]]
        # Backfill role labels when the producing code didn't set them
        # (multi-player matches use _handle_multi_player_match_results, which
        # has no "first"/"second" semantics). "seat_N" is a stable label so
        # downstream JSON consumers don't see empty strings.
        if not self.agent1_role and self.agent1_name:
            self.agent1_role = "seat_1"
        if not self.agent2_role and self.agent2_name:
            self.agent2_role = "seat_2"


@dataclass
class SinglePlayerResult:
    """Result of a single-player challenge."""
    agent_name: str
    challenge_name: str
    game_variant: str  # e.g., "HanoiTower-5"
    success: bool
    score: float  # 0.0-1.0
    moves: int
    optimal_moves: Optional[int] = None
    efficiency: Optional[float] = None  # optimal_moves / actual_moves
    match_duration: float = 0.0
    move_history: List[Tuple[str, int, float]] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    timeout_occurred: bool = False
    
    @property
    def move_efficiency(self) -> float:
        """Calculate move efficiency if optimal moves is known."""
        if self.optimal_moves and self.moves > 0:
            return min(1.0, self.optimal_moves / self.moves)
        return 0.0


@dataclass
class RoundResult:
    """Results from a single tournament round."""
    round_number: int
    coder_results: List[CoderResult]
    match_results: List[MatchResult]
    final_rankings: List[Dict[str, Any]]
    experiment_folder: str
    results_file: str


@dataclass
class MultiRoundStats:
    """Statistics from multiple tournament rounds for a single coder."""
    name: str
    trueskill_ratings: List[ts.Rating] = field(default_factory=list)  # TrueSkill rating from each round
    conservative_ratings: List[float] = field(default_factory=list)  # Conservative ratings from each round
    skill_estimates: List[float] = field(default_factory=list)  # Skill estimates from each round
    uncertainties: List[float] = field(default_factory=list)  # Uncertainties from each round
    wins: List[int] = field(default_factory=list)  # Wins from each round
    losses: List[int] = field(default_factory=list)  # Losses from each round
    draws: List[int] = field(default_factory=list)  # Draws from each round
    win_rates: List[float] = field(default_factory=list)  # Win rate from each round
    passed_tests_count: int = 0  # Number of rounds where tests were passed
    total_rounds: int = 0  # Total number of rounds
    
    # Extended metrics
    revision_counts: List[int] = field(default_factory=list)  # Revision attempts per round
    code_generation_times: List[float] = field(default_factory=list)  # Code generation time per round
    total_testing_times: List[float] = field(default_factory=list)  # Testing time per round
    in_game_error_counts: List[int] = field(default_factory=list)  # In-game errors per round
    avg_decision_times: List[float] = field(default_factory=list)  # Average decision time per round
    max_decision_times: List[float] = field(default_factory=list)  # Max decision time per round
    
    # Error analysis
    all_in_game_errors: List[Dict[str, Any]] = field(default_factory=list)  # All in-game errors across rounds
    all_test_failures: List[Dict[str, Any]] = field(default_factory=list)  # All test failures across rounds
    
    # Test count tracking per round
    total_tests_per_round: List[int] = field(default_factory=list)  # Total tests run per round
    passed_tests_per_round: List[int] = field(default_factory=list)  # Passed tests per round
    failed_tests_per_round: List[int] = field(default_factory=list)  # Failed tests per round
    
    # Timeout tracking
    timeout_counts: List[int] = field(default_factory=list)  # Timeout counts per round
    all_timeout_details: List[Dict[str, Any]] = field(default_factory=list)  # All timeout details across rounds
    
    # Per-round match participation count (number of matches this coder played in each round)
    matches_played_per_round: List[int] = field(default_factory=list)

    # Single-player challenge tracking
    single_player_challenges_per_round: List[int] = field(default_factory=list)  # Challenges per round
    single_player_successes_per_round: List[int] = field(default_factory=list)  # Successes per round
    single_player_results: List[SinglePlayerResult] = field(default_factory=list)  # All single-player results
    challenge_success_rates: Dict[str, float] = field(default_factory=dict)  # Success rate by challenge type
    
    # Round-by-round details
    round_details: List[Dict[str, Any]] = field(default_factory=list)  # Detailed stats for each round
    
    # Computed statistics
    @property
    def conservative_rating_max(self) -> float:
        return max(self.conservative_ratings) if self.conservative_ratings else 0.0
    
    @property
    def conservative_rating_min(self) -> float:
        return min(self.conservative_ratings) if self.conservative_ratings else 0.0
    
    @property
    def conservative_rating_avg(self) -> float:
        return statistics.mean(self.conservative_ratings) if self.conservative_ratings else 0.0
    
    @property
    def conservative_rating_std(self) -> float:
        return statistics.stdev(self.conservative_ratings) if len(self.conservative_ratings) > 1 else 0.0
    
    @property
    def skill_estimate_avg(self) -> float:
        return statistics.mean(self.skill_estimates) if self.skill_estimates else 25.0
    
    @property
    def skill_estimate_std(self) -> float:
        return statistics.stdev(self.skill_estimates) if len(self.skill_estimates) > 1 else 0.0
    
    @property
    def avg_uncertainty(self) -> float:
        return statistics.mean(self.uncertainties) if self.uncertainties else 8.333
    
    @property
    def rating_robustness_score(self) -> float:
        """Calculate robustness score: 1/(1+std(conservative_rating))"""
        return 1.0 / (1.0 + self.conservative_rating_std)
    
    @property
    def win_rate_avg(self) -> float:
        return statistics.mean(self.win_rates) if self.win_rates else 0.0
    
    @property
    def success_rate(self) -> float:
        """Percentage of rounds where coder passed tests"""
        return self.passed_tests_count / self.total_rounds if self.total_rounds > 0 else 0.0
    
    @property
    def avg_revision_count(self) -> float:
        """Average number of revisions per round"""
        return statistics.mean(self.revision_counts) if self.revision_counts else 0.0
    
    @property
    def avg_code_generation_time(self) -> float:
        """Average code generation time per round"""
        return statistics.mean(self.code_generation_times) if self.code_generation_times else 0.0
    
    @property
    def avg_testing_time(self) -> float:
        """Average testing time per round"""
        return statistics.mean(self.total_testing_times) if self.total_testing_times else 0.0
    
    @property
    def total_in_game_errors(self) -> int:
        """Total number of in-game errors across all rounds"""
        return sum(self.in_game_error_counts)
    
    @property
    def avg_in_game_errors_per_round(self) -> float:
        """Average number of in-game errors per round"""
        return statistics.mean(self.in_game_error_counts) if self.in_game_error_counts else 0.0
    
    @property
    def overall_avg_decision_time(self) -> float:
        """Overall average decision time across all rounds"""
        return statistics.mean(self.avg_decision_times) if self.avg_decision_times else 0.0
    
    @property
    def overall_max_decision_time(self) -> float:
        """Overall maximum decision time across all rounds"""
        return max(self.max_decision_times) if self.max_decision_times else 0.0
    
    @property
    def total_timeouts(self) -> int:
        """Total number of timeouts across all rounds"""
        return sum(self.timeout_counts)
    
    @property
    def avg_timeouts_per_round(self) -> float:
        """Average number of timeouts per round"""
        return statistics.mean(self.timeout_counts) if self.timeout_counts else 0.0
    
    @property
    def total_games(self) -> int:
        """Total number of games played (wins + losses + draws)"""
        return sum(self.wins) + sum(self.losses) + sum(self.draws)

    @property
    def total_matches_played(self) -> int:
        """Total number of matches the coder participated in across all rounds.

        Counted directly from per-round match-participation tallies (which include
        matches that ended in in-game errors). Falls back to wins+losses+draws if
        per-round counts were not recorded.
        """
        if self.matches_played_per_round:
            return sum(self.matches_played_per_round)
        return self.total_games
    
    @property
    def overall_single_player_success_rate(self) -> float:
        """Overall success rate across all single-player challenges"""
        total_challenges = sum(self.single_player_challenges_per_round)
        total_successes = sum(self.single_player_successes_per_round)
        return total_successes / total_challenges if total_challenges > 0 else 0.0
    
    @property
    def avg_single_player_success_rate(self) -> float:
        """Average success rate per round for single-player challenges"""
        if not self.single_player_challenges_per_round:
            return 0.0
        
        round_success_rates = []
        for i, challenges in enumerate(self.single_player_challenges_per_round):
            if challenges > 0:
                successes = self.single_player_successes_per_round[i] if i < len(self.single_player_successes_per_round) else 0
                round_success_rates.append(successes / challenges)
        
        return statistics.mean(round_success_rates) if round_success_rates else 0.0
    
    @property
    def best_challenge_success_rate(self) -> float:
        """Best success rate among all challenge types"""
        return max(self.challenge_success_rates.values()) if self.challenge_success_rates else 0.0
    
    @property
    def avg_challenge_success_rate(self) -> float:
        """Average success rate across all challenge types"""
        return statistics.mean(list(self.challenge_success_rates.values())) if self.challenge_success_rates else 0.0 