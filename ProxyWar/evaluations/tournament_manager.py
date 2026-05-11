"""
Tournament management functionality for ProxyWar framework.

This module contains the TournamentManager class for running complete tournaments
with multiple AI-generated agents, including testing, round-robin battles, and TrueSkill ratings.
"""

import os
import json
import itertools
import time
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import concurrent.futures
from multiprocessing import cpu_count

from ..agents import BaseAgent
from ..games.base import BaseGame, MultiPlayerGame, SinglePlayerGame, TwoPlayerGame
from ..games.tictactoe import TicTacToeGame
from ..games.connectfour import ConnectFourGame
from ..coders.base import BaseCoder
from ..coders import load_agent_from_file
from ..prompts.base import BasePrompt
from ..loggers.tournament_logger import TournamentLogger, MatchRecord
from ..coders.utils import clean_generated_code
from .data_models import CoderResult, MatchResult, RoundResult, MultiRoundStats
from .elo_system import TrueSkillSystem
from .stages.report_stage import _convert_to_serializable


class TournamentManager:
    """
    Run a tournament across a list of coders for a single game.

    Inputs: a game instance, the list of coders to evaluate, and the prompt
    generator used to build code-generation prompts.

    Pipeline: generate-and-test each coder's agent (with up to 3 repair
    rounds), then run round-robin matches between agents that passed
    testing, updating TrueSkill ratings.

    Side effects: writes generated agent files, per-round histories, and
    Markdown / JSON reports under the experiment folder.
    """

    def __init__(self, game: BaseGame, coders_info: List[Dict], prompt_generator: BasePrompt):
        self.game = game
        self.coders_info = coders_info
        self.prompt_generator = prompt_generator
        self.experiment_folder: Optional[str] = None
        self.logger: Optional[TournamentLogger] = None
        self.tester = self._initialize_tester()

        self.detailed_logger = None
        self.trueskill_system = TrueSkillSystem()
        self.coder_results: List[CoderResult] = []
        self.match_results: List[MatchResult] = []
        self.save_visualizations = False

    def _initialize_tester(self):
        """Initialize appropriate tester based on game type."""
        if isinstance(self.game, TicTacToeGame):
            from ..testers.tictactoe_tester import TicTacToeTester
            return TicTacToeTester()
        elif isinstance(self.game, ConnectFourGame):
            from ..testers.connectfour_tester import ConnectFourTester
            return ConnectFourTester()
        else:
            # Imports kept local to avoid pulling every game module at top-level.
            from ..games.hanoi import HanoiTowerGame
            from ..games.sudoku import SudokuGame
            from ..games.snake import SnakeGame
            from ..games.reversi import ReversiGame
            from ..games.texas_holdem import TexasHoldemGame
            from ..games.twenty_forty_eight import TwentyFortyEightGame
            from ..games.maze import MazeGame
            if isinstance(self.game, MazeGame):
                from ..testers.maze_tester import MazeTester
                return MazeTester()
            elif isinstance(self.game, HanoiTowerGame):
                from ..testers.hanoi_tester import HanoiTowerTester
                return HanoiTowerTester()
            elif isinstance(self.game, SudokuGame):
                from ..testers.sudoku_tester import SudokuTester
                return SudokuTester()
            elif isinstance(self.game, SnakeGame):
                from ..testers.snake_tester import SnakeTester
                return SnakeTester()
            elif isinstance(self.game, ReversiGame):
                from ..testers.reversi_tester import ReversiTester
                return ReversiTester()
            elif isinstance(self.game, TexasHoldemGame):
                from ..testers.texas_holdem_tester import TexasHoldemTester
                return TexasHoldemTester()
            elif isinstance(self.game, TwentyFortyEightGame):
                from ..testers.twenty_forty_eight_tester import TwentyFortyEightTester
                return TwentyFortyEightTester()
            return None

    def create_experiment_folder(self) -> str:
        """Create timestamped experiment folder structure and initialize logger."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_folder = f"experiments/tournament_{timestamp}"
        os.makedirs(exp_folder, exist_ok=True)
        os.makedirs(f"{exp_folder}/agents", exist_ok=True)
        os.makedirs(f"{exp_folder}/results", exist_ok=True)

        self.experiment_folder = exp_folder
        self.logger = TournamentLogger(name="Tournament")
        self.detailed_logger = None
        return exp_folder


    def generate_and_test_all_coders(self, max_revisions: int = 3) -> List[CoderResult]:
        """
        Generate and test code for all coders in parallel.
        
        Args:
            max_revisions: Maximum number of revision attempts per coder
            
        Returns:
            List of CoderResult objects
        """
        if not self.experiment_folder:
            raise ValueError("Experiment folder not created. Call create_experiment_folder() first.")
        
        if self.logger:
            self.logger.info("Generating and testing code for all coders in parallel...")
            self.logger.subsection_header("")
        
        self.coder_results = []

        coder_tasks = []
        for coder_info in self.coders_info:
            coder_name = coder_info['name']
            coder = coder_info['coder']
            result = CoderResult(name=coder_name, coder=coder)
            coder_tasks.append((coder_name, coder, result))

        max_workers = min(len(self.coders_info), cpu_count())
        if self.logger:
            self.logger.info(f"Running {len(coder_tasks)} coders in parallel with {max_workers} workers")
            self.logger.info("")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_coder = {}
            for coder_name, coder, result in coder_tasks:
                if self.logger:
                    print(f"Starting {coder_name}...")
                future = executor.submit(self._process_single_coder_parallel, coder_name, coder, max_revisions, result)
                future_to_coder[future] = (coder_name, result)

            for future in concurrent.futures.as_completed(future_to_coder):
                coder_name, result = future_to_coder[future]
                try:
                    completed_result = future.result()
                    self.coder_results.append(completed_result)

                    if completed_result.passed_tests:
                        print(f"{coder_name} completed successfully")
                    else:
                        print(f"{coder_name} failed to generate working code")

                except Exception as exc:
                    print(f"{coder_name} generated an exception: {exc}")
                    result.passed_tests = False
                    result.test_failures.append({
                        'test_name': 'parallel_execution_error',
                        'error_type': 'execution_error',
                        'error_message': str(exc),
                        'details': f"Failed during parallel execution"
                    })
                    self.coder_results.append(result)
        
        # Seed TrueSkill with every coder (passing and failing) so failed coders
        # keep the normal prior rating instead of being silently treated as zero.
        for coder_result in self.coder_results:
            self.trueskill_system.add_player(coder_result.name)
            coder_result.trueskill_rating = self.trueskill_system.get_rating(coder_result.name)
            coder_result.conservative_rating = self.trueskill_system.get_conservative_rating(coder_result.name)
            coder_result.skill_estimate = self.trueskill_system.get_skill_estimate(coder_result.name)
            coder_result.uncertainty = self.trueskill_system.get_uncertainty(coder_result.name)
        
        passing_coders = [c for c in self.coder_results if c.passed_tests]
        
        if self.logger:
            self.logger.info("")
            self.logger.info(f"Parallel code generation completed: {len(passing_coders)}/{len(self.coder_results)} coders passed tests")
        
        return self.coder_results

    def _process_single_coder_parallel(self, coder_name: str, coder: BaseCoder, max_revisions: int, result: CoderResult) -> CoderResult:
        """
        Process a single coder for parallel execution - generate, test and load agent.
        
        Args:
            coder_name: Name of the coder
            coder: Coder instance
            max_revisions: Maximum number of revision attempts
            result: CoderResult object to populate
            
        Returns:
            Completed CoderResult object
        """
        generation_start_time = time.time()

        agent_code_path = self._generate_and_test_single_coder_enhanced(coder_name, coder, max_revisions, result)

        result.total_testing_time = time.time() - generation_start_time

        if agent_code_path:
            try:
                agent_instance = load_agent_from_file(agent_code_path, coder_name)
                result.agent_instance = agent_instance
                result.passed_tests = True
                result.code_file = agent_code_path

            except Exception as e:
                result.passed_tests = False

                result.test_failures.append({
                    'test_name': 'agent_loading',
                    'error_type': 'loading_error',
                    'error_message': str(e),
                    'details': f"Failed to load agent from {agent_code_path}"
                })
        else:
            result.passed_tests = False
        
        return result

    def _generate_and_test_single_coder_enhanced(self, coder_name: str, coder: BaseCoder, max_revisions: int, result: CoderResult) -> Optional[str]:
        """Enhanced version of single coder generation with detailed tracking."""
        # Open the per-coder testing record before any log_test_failure /
        # log_revision_attempt / log_coder_testing_complete call below — those
        # helpers silently no-op when the record key is missing.
        if self.logger:
            self.logger.log_coder_testing_start(coder_name, self.game.game_name)
        try:
            # Track code generation time
            code_gen_start = time.time()
            
            # Generate agent code using the prompt
            original_prompt = self.prompt_generator.generate_prompt(
                game=self.game, 
                additional_context=f"Create a strategic {coder_name} agent that plays optimally.",
                agent_name=coder_name
            )
            
            agent_code = coder.generate_agent_code(original_prompt)
            cleaned_code = clean_generated_code(agent_code)
            
            result.code_generation_time = time.time() - code_gen_start
            
            # Save the code to file. Force UTF-8 because the default codec on
            # Windows is gbk, which can't encode non-ASCII characters that
            # frontier models routinely emit in comments / strings.
            code_file = f"{self.experiment_folder}/agents/{coder_name.lower()}.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            # Test the code and collect detailed failure information
            test_passed, test_failure_details, test_stats = self._test_agent_code_with_details(code_file, coder_name)
            
            # Store test statistics from initial attempt
            result.total_tests = test_stats['total_tests']
            result.passed_test_count = test_stats['passed_tests']
            result.failed_tests = test_stats['failed_tests']
            result.error_tests = test_stats['error_tests']
            result.skipped_tests = test_stats['skipped_tests']
            
            # Add initial test failures to result
            if test_failure_details:
                result.test_failures.extend(test_failure_details)
            
            if test_passed:
                if self.logger:
                    self.logger.success(f"  {coder_name} passed tests on first attempt")
                    self.logger.log_coder_testing_complete(
                        coder_name, self.game.game_name, 
                        result.total_testing_time, cleaned_code, True
                    )
                return code_file
            else:
                # Log detailed test failure information
                if self.logger:
                    self.logger.info(f"   Initial code failed tests:")
                    if test_failure_details:
                        for failure in test_failure_details:
                            self.logger.info(f"     - {failure['test_name']}: {failure['error_message'][:200]}...")
                            self.logger.log_test_failure(
                                coder_name, self.game.game_name,
                                failure['test_name'], failure['error_type'], failure['error_message']
                            )
                    else:
                        self.logger.info(f"     - No detailed error information available")
                    self.logger.info(f"   Attempting revisions...")
            
            # Attempt revisions with detailed tracking
            first_revision = True
            for revision_num in range(1, max_revisions + 1):
                result.revision_count += 1
                revision_start = time.time()
                
                if self.logger:
                    self.logger.info(f"  Revision attempt {revision_num}/{max_revisions}...")
                    self.logger.log_revision_attempt(coder_name, self.game.game_name, result.revision_count, False)
                
                if self.tester is None:
                    if self.logger:
                        self.logger.failure(f"  No tester available, cannot provide detailed errors")
                    break
                
                # Get detailed test errors and display them
                test_suite = self.tester.test_agent_code(code_file, coder_name)
                error_details = self._format_test_errors(test_suite)
                
                if self.logger and error_details:
                    self.logger.info(f"    Current test failures:")
                    for line in error_details.split('\n'):
                        if line.strip():
                            self.logger.info(f"      {line.strip()}")
                    self.logger.info(f"    Generating revised code...")
                
                # Generate revision prompt
                revision_prompt = f"""
Please analyze the test failures and fix the code accordingly. The code should:
1. Have valid Python syntax
2. Import BaseAgent from 'from ProxyWar.agents.base import BaseAgent'
3. Class name should be exactly '{coder_name}' (not TicTacToeAgent or generic names)
4. Inherit from BaseAgent class (do NOT redefine BaseAgent)
5. Implement the select_action method correctly
6. Handle edge cases properly
7. Return valid actions (integers 0-8 for TicTacToe, or None if no legal moves)
8. Output ONLY the Python code without explanatory text

Test Errors:
{error_details}
"""
                
                # Generate revised code
                try:
                    revised_code = coder.revise_agent_code(
                        original_prompt=original_prompt,
                        previous_code=cleaned_code,
                        test_errors=error_details,
                        revision_prompt=revision_prompt
                    )
                    
                    cleaned_revised_code = clean_generated_code(revised_code)
                    
                    # Save version to history before overwriting
                    if first_revision:
                        # For the first revision, save the original version as v1
                        self._save_code_version_to_history(code_file, coder_name, 0)  # Save original as v1
                        first_revision = False
                    else:
                        # For subsequent revisions, save the current version before overwriting
                        self._save_code_version_to_history(code_file, coder_name, revision_num - 1)
                    
                    # Save revised code (UTF-8 for the same reason as the
                    # initial write above).
                    with open(code_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_revised_code)
                    
                    # Test revised code
                    test_passed, test_failure_details, _ = self._test_agent_code_with_details(code_file, coder_name)
                    
                    # Add revision test failures to result
                    if test_failure_details:
                        for failure in test_failure_details:
                            failure['revision_number'] = str(revision_num)
                        result.test_failures.extend(test_failure_details)
                    
                    result.total_revision_time += time.time() - revision_start
                    
                    if test_passed:
                        if self.logger:
                            self.logger.success(f"  {coder_name} passed tests after {revision_num} revision(s)")
                            self.logger.log_revision_attempt(coder_name, self.game.game_name, result.revision_count, True)
                            self.logger.log_coder_testing_complete(
                                coder_name, self.game.game_name, 
                                result.total_testing_time, cleaned_revised_code, True
                            )
                        return code_file
                    else:
                        # Log detailed revision failure information
                        if self.logger:
                            self.logger.info(f"    Revision {revision_num} still has test failures:")
                            if test_failure_details:
                                for failure in test_failure_details:
                                    self.logger.info(f"      - {failure['test_name']}: {failure['error_message'][:200]}...")
                                    self.logger.log_test_failure(
                                        coder_name, self.game.game_name,
                                        failure['test_name'], failure['error_type'], failure['error_message']
                                    )
                            else:
                                self.logger.info(f"      - No detailed error information available")
                    
                    # Update code for next iteration
                    cleaned_code = cleaned_revised_code
                    
                except Exception as e:
                    result.total_revision_time += time.time() - revision_start
                    
                    # Add revision generation error to test failures
                    result.test_failures.append({
                        'test_name': f'revision_{revision_num}',
                        'error_type': 'generation_error',
                        'error_message': str(e),
                        'revision_number': str(revision_num),
                        'details': f"Failed to generate revision {revision_num}"
                    })
                    
                    if self.logger:
                        self.logger.failure(f"    Revision {revision_num} generation failed: {str(e)[:200]}...")
                        self.logger.log_test_failure(
                            coder_name, self.game.game_name,
                            f"revision_{revision_num}", "generation_error", str(e)
                        )
                    continue
            
            # All attempts failed
            if self.logger:
                self.logger.failure(f"  {coder_name} failed after {max_revisions} revision attempts")
                if result.test_failures:
                    self.logger.info(f"    Final error summary:")
                    # Show the most recent errors (last few failures)
                    recent_failures = result.test_failures[-3:] if len(result.test_failures) > 3 else result.test_failures
                    for failure in recent_failures:
                        self.logger.info(f"      - {failure['test_name']}: {failure['error_message'][:150]}...")
                self.logger.log_coder_testing_complete(
                    coder_name, self.game.game_name, 
                    result.total_testing_time, cleaned_code, False
                )
            return None
            
        except Exception as e:
            # Add top-level generation error
            result.test_failures.append({
                'test_name': 'code_generation',
                'error_type': 'generation_error',
                'error_message': str(e),
                'details': "Failed during initial code generation"
            })
            
            if self.logger:
                self.logger.failure(f"Failed to generate code for {coder_name}: {e}")
                self.logger.log_test_failure(
                    coder_name, self.game.game_name,
                    "code_generation", "generation_error", str(e)
                )
                self.logger.log_coder_testing_complete(
                    coder_name, self.game.game_name, 
                    result.total_testing_time, "", False
                )
            return None
    
    def _test_agent_code(self, agent_code_path: str, coder_name: str) -> bool:
        """Test generated agent code."""
        if self.tester is None:
            if self.logger:
                self.logger.info(f"No tester available for {self.game.game_name}, skipping tests")
            return True
        
        test_suite = self.tester.test_agent_code(agent_code_path, coder_name)
        return test_suite.is_passing
    
    def _test_agent_code_with_details(self, agent_code_path: str, coder_name: str) -> tuple[bool, List[Dict[str, str]], Dict[str, int]]:
        """Test generated agent code and return detailed failure information and test statistics."""
        if self.tester is None:
            if self.logger:
                self.logger.info(f"No tester available for {self.game.game_name}, skipping tests")
            return True, [], {'total_tests': 0, 'passed_tests': 0, 'failed_tests': 0, 'error_tests': 0, 'skipped_tests': 0}
        
        test_suite = self.tester.test_agent_code(agent_code_path, coder_name)
        
        # Extract test statistics
        test_stats = {
            'total_tests': test_suite.total_tests,
            'passed_tests': test_suite.passed_tests,
            'failed_tests': test_suite.failed_tests,
            'error_tests': test_suite.error_tests,
            'skipped_tests': test_suite.skipped_tests
        }
        
        failure_details = []
        if not test_suite.is_passing:
            failed_tests = [test for test in test_suite.test_results 
                           if test.status.value in ['failed', 'error']]
            
            for test in failed_tests:
                error_type = "unknown_error"
                if self.logger:
                    error_type = self.logger.classify_error(test.message)
                
                failure_details.append({
                    'test_name': test.test_name,
                    'error_type': error_type,
                    'error_message': test.message,
                    'details': test.details if hasattr(test, 'details') else ""
                })
        
        return test_suite.is_passing, failure_details, test_stats
    
    def _format_test_errors(self, test_suite) -> str:
        """Format test errors for LLM feedback."""
        error_lines = []
        error_lines.append(f"Test Results Summary:")
        error_lines.append(f"- Total tests: {test_suite.total_tests}")
        error_lines.append(f"- Passed: {test_suite.passed_tests}")
        error_lines.append(f"- Failed: {test_suite.failed_tests}")
        error_lines.append(f"- Errors: {test_suite.error_tests}")
        error_lines.append("")
        
        failed_tests = [test for test in test_suite.test_results 
                       if test.status.value in ['failed', 'error']]
        
        if failed_tests:
            error_lines.append("Failed Tests Details:")
            for test in failed_tests:
                error_lines.append(f"- {test.test_name}: {test.message}")
                if test.details:
                    error_lines.append(f"  Details: {test.details}")
            error_lines.append("")
        
        return "\n".join(error_lines)
    
    def run_round_robin_tournament(self) -> List[MatchResult]:
        """
        Run round-robin tournament between all passing coders.
        Each pair of agents plays twice (once as first player, once as second player).
        Failed coders are ranked last in the current round.
        
        Returns:
            List of match results
        """
        # Get passing coders
        passing_coders = [r for r in self.coder_results if r.passed_tests and r.agent_instance]
        failed_coders = [r for r in self.coder_results if not r.passed_tests]
        
        if len(passing_coders) < 2:
            if self.logger:
                self.logger.error(f"Not enough agents for tournament. Need at least 2, got {len(passing_coders)}")
            return []
        
        if self.logger:
            self.logger.info("")
            self.logger.info(f"Starting Round Robin Tournament")
            self.logger.info(f"Participants: {[c.name for c in passing_coders]}")
            if failed_coders:
                self.logger.info(f"Failed coders (will be ranked last): {[c.name for c in failed_coders]}")
            self.logger.subsection_header("")
        
        self.match_results = []
        
        # Generate all possible matchups
        matchups = []
        is_single_player = isinstance(self.game, SinglePlayerGame)
        for coder1, coder2 in itertools.combinations(passing_coders, 2):
            if is_single_player:
                # SinglePlayerGame: no first/second-player advantage, one match per pair.
                matchups.append((coder1, coder2, "agent1", "agent2"))
            else:
                # TwoPlayerGame: play both directions for first/second player fairness.
                matchups.append((coder1, coder2, "first", "second"))
                matchups.append((coder2, coder1, "first", "second"))

        total_matches = len(matchups)

        if self.logger:
            if is_single_player:
                self.logger.info(f"Total matches to play: {total_matches} (single-player game: one match per pair)")
            else:
                self.logger.info(f"Total matches to play: {total_matches} (each pair plays twice for first/second balance)")
            self.logger.info("")
        
        # Play all matches
        for match_num, (coder1, coder2, role1, role2) in enumerate(matchups, 1):
            if self.logger:
                self.logger.info(f"Match {match_num}/{total_matches}: {coder1.name} ({role1}) vs {coder2.name} ({role2})")
            
            # Run the match (agents are guaranteed to be non-None due to filtering above)
            assert coder1.agent_instance is not None and coder2.agent_instance is not None
            match_result = self._run_single_match(coder1.agent_instance, coder2.agent_instance, role1, role2)
            self.match_results.append(match_result)
            
            # Update TrueSkill ratings and statistics
            self._update_trueskill_and_stats(match_result, coder1, coder2)
            
            if self.logger:
                current_rating1 = self.trueskill_system.get_conservative_rating(coder1.name)
                current_rating2 = self.trueskill_system.get_conservative_rating(coder2.name)
                self.logger.info(f"  Winner: {match_result.winner}")
                self.logger.info(f"  Move count: {match_result.moves}")
                self.logger.info(f"  New ratings: {coder1.name}: {current_rating1:.1f}, {coder2.name}: {current_rating2:.1f}")
                self.logger.info("")
        
        
        return self.match_results
    
    def run_multi_player_tournament(self, matches_per_agent: int = 100) -> List[MatchResult]:
        """
        Run multi-player tournament with fixed number of matches per agent.
        
        Args:
            matches_per_agent: Number of matches each agent should participate in
            
        Returns:
            List of match results
        """
        # Get passing coders
        passing_coders = [r for r in self.coder_results if r.passed_tests and r.agent_instance]
        failed_coders = [r for r in self.coder_results if not r.passed_tests]
        
        if len(passing_coders) < 2:
            if self.logger:
                self.logger.error(f"Not enough agents for multi-player tournament. Need at least 2, got {len(passing_coders)}")
            return []
        
        if self.logger:
            self.logger.info("")
            self.logger.info(f"Starting Multi-Player Tournament")
            self.logger.info(f"Participants: {[c.name for c in passing_coders]}")
            self.logger.info(f"Matches per agent: {matches_per_agent}")
            # Check if game is a MultiPlayerGame with player limits
            min_players = getattr(self.game, 'min_players', 2)
            max_players = getattr(self.game, 'max_players', len(passing_coders))
            self.logger.info(f"Players per match: {min_players}-{max_players}")
            if failed_coders:
                self.logger.info(f"Failed coders (will be ranked last): {[c.name for c in failed_coders]}")
            self.logger.subsection_header("")
        
        self.match_results = []
        
        # Generate matchups ensuring each agent plays the specified number of matches
        matchups = self._generate_multi_player_matchups(passing_coders, matches_per_agent)
        total_matches = len(matchups)
        
        if self.logger:
            self.logger.info(f"Total matches to play: {total_matches}")
            self.logger.info("")
        
        # Run all matches
        for match_idx, (agents_in_match, match_number) in enumerate(matchups):
            if self.logger:
                agent_names = [agent.name for agent in agents_in_match]
                self.logger.info(f"Match {match_idx + 1}/{total_matches} (#{match_number}): {', '.join(agent_names)}")
            
            # Run the multi-player match
            match_result = self._run_multi_player_match(agents_in_match, match_number)
            self.match_results.append(match_result)
            
            # Update TrueSkill ratings
            self._update_trueskill_multi_player(match_result, agents_in_match)
            
            if self.logger:
                # Show updated ratings
                rating_updates = []
                for agent in agents_in_match:
                    current_rating = self.trueskill_system.get_conservative_rating(agent.name)
                    rating_updates.append(f"{agent.name}: {current_rating:.1f}")
                self.logger.info(f"  Updated ratings: {', '.join(rating_updates)}")
                
                # Show detailed match information for every match
                self._log_detailed_match_info(match_result, agents_in_match, match_idx + 1)
                
                self.logger.info("")
        
        
        return self.match_results
    
    def _generate_multi_player_matchups(self, passing_coders: List[CoderResult], 
                                       matches_per_agent: int) -> List[Tuple[List[BaseAgent], int]]:
        """
        Generate matchups ensuring each agent plays the specified number of matches.
        
        Args:
            passing_coders: List of coders that passed tests
            matches_per_agent: Number of matches each agent should participate in
            
        Returns:
            List of (agents_in_match, match_number) tuples
        """
        import random
        
        agents = [coder.agent_instance for coder in passing_coders]
        # Filter out None agents for safety
        agents = [agent for agent in agents if agent is not None]
        num_agents = len(agents)
        
        # Track how many matches each agent has played
        agent_match_counts = {agent.name: 0 for agent in agents}
        matchups = []
        match_number = 0
        
        # Determine match size based on number of agents
        max_players = getattr(self.game, 'max_players', num_agents)
        if num_agents <= max_players:
            # If we have fewer agents than max players, use all of them
            match_size = num_agents
        else:
            # Use the maximum number of players
            match_size = max_players
        
        # Generate matches until all agents have played enough
        max_iterations = matches_per_agent * num_agents * 2  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Check if all agents have played enough matches
            agents_needing_matches = [agent for agent in agents 
                                    if agent_match_counts[agent.name] < matches_per_agent]
            
            if not agents_needing_matches:
                break
                
            # Select agents for this match
            if len(agents_needing_matches) >= match_size:
                # Randomly select from agents who need matches
                selected_agents = random.sample(agents_needing_matches, match_size)
            else:
                # Not enough agents needing matches, fill with any agents
                selected_agents = agents_needing_matches[:]
                remaining_spots = match_size - len(selected_agents)
                
                # Fill remaining spots with agents who have played the least
                available_agents = [agent for agent in agents if agent not in selected_agents]
                available_agents.sort(key=lambda a: agent_match_counts[a.name])
                
                selected_agents.extend(available_agents[:remaining_spots])
            
            # Add this match
            matchups.append((selected_agents, match_number))
            match_number += 1
            
            # Update match counts
            for agent in selected_agents:
                agent_match_counts[agent.name] += 1
        
        if self.logger:
            self.logger.info(f"Generated {len(matchups)} matches")
            for agent in agents:
                count = agent_match_counts[agent.name]
                self.logger.info(f"  {agent.name}: {count} matches")
        
        return matchups
    
    def _run_multi_player_match(self, agents: List[BaseAgent], match_number: int) -> MatchResult:
        """
        Run a single multi-player match.
        
        Args:
            agents: List of agents to participate in the match
            match_number: Match number for tracking
            
        Returns:
            MatchResult object
        """
        match_start_time = time.time()
        
        # Setup and run the match (check if game has setup_match method)
        setup_match_func = getattr(self.game, 'setup_match', None)
        if setup_match_func:
            setup_match_func(agents)
        
        try:
            result = self.game.run_match()
            match_duration = time.time() - match_start_time
        except Exception as e:
            # Handle in-game errors
            match_duration = time.time() - match_start_time
            error_message = str(e)
            print(f"In-game error occurred: {error_message}")
            
            # Create a result for the failed match
            result = {
                'results': {agent.name: 0.0 for agent in agents},
                'moves': 0,
                'match_history': [],
                'final_rankings': [agent.name for agent in agents],
                'rewards': {agent.name: 0.0 for agent in agents},
                'timeout_info': [],
                'total_time': match_duration,
                'game_type': 'multi_player',
                'error_info': {
                    'error_message': error_message,
                    'error_type': 'in_game_error'
                }
            }
        
        # Convert multi-player result to MatchResult format
        return self._convert_multi_player_result(result, agents, match_number)
    
    def _convert_multi_player_result(self, result: Dict[str, Any], agents: List[BaseAgent], 
                                   match_number: int) -> MatchResult:
        """
        Convert multi-player game result to MatchResult format.
        
        Args:
            result: Result dictionary from multi-player game
            agents: List of agents that participated
            match_number: Match number for tracking
            
        Returns:
            MatchResult object
        """
        # Extract move history with timing
        move_history = result.get('match_history', [])
        move_history_with_timing = result.get('move_history_with_timing', [])
        
        # If no detailed timing records, create basic format
        if not move_history_with_timing and move_history:
            move_history_with_timing = [(entry.get('agent', ''), entry.get('action', 0), 
                                       entry.get('decision_time', 0.0)) for entry in move_history]
        
        # Collect decision time statistics for each player
        player_decision_times = {agent.name: [] for agent in agents}
        for entry in move_history_with_timing:
            if isinstance(entry, dict):
                player = entry.get('agent', '')
                decision_time = entry.get('decision_time', 0.0)
            else:
                player, action, decision_time = entry

            if player in player_decision_times:
                player_decision_times[player].append(decision_time)

        # Mirror the two-player path: write per-move timings back into each
        # CoderResult so downstream MultiRoundStats / per-game reports see
        # them. (coder_performance.json reads match_results directly and was
        # fine without this, but the report-stage path goes through
        # coder_result.decision_times.)
        for coder_result in self.coder_results:
            if coder_result.name in player_decision_times:
                coder_result.decision_times.extend(player_decision_times[coder_result.name])

        # Calculate average decision times
        avg_decision_times = {}
        for agent_name, times in player_decision_times.items():
            avg_decision_times[agent_name] = sum(times) / len(times) if times else 0.0
        
        # Extract final rankings (always available, regardless of reward system)
        final_rankings = result.get('final_rankings', [])
        
        # Use actual rewards from the game (PettingZoo provides real chip changes)
        rewards = result.get('rewards', {})
        scores = {}
        
        # If we have real rewards, use them
        if rewards:
            scores = {agent_name: float(reward) for agent_name, reward in rewards.items()}
        else:
            # Fallback to ranking-based scoring
            num_agents = len(agents)
            for i, agent_name in enumerate(final_rankings):
                if num_agents == 1:
                    scores[agent_name] = 1.0
                else:
                    scores[agent_name] = (num_agents - i - 1) / (num_agents - 1)
        
        # Ensure all agents have scores
        for agent in agents:
            if agent.name not in scores:
                scores[agent.name] = 0.0
        
        # Determine winner based on actual rewards
        if rewards:
            max_reward = max(rewards.values())
            if max_reward > 0:
                winners = [agent_name for agent_name, reward in rewards.items() if reward == max_reward]
                winner = winners[0] if len(winners) == 1 else 'draw'
            else:
                winner = 'draw'
        else:
            # Fallback to ranking-based winner
            winner = final_rankings[0] if final_rankings else 'draw'
        
        return MatchResult(
            winner=winner,
            scores=scores,
            moves=result.get('moves', 0),
            match_duration=result.get('total_time', 0.0),
            match_history=move_history,
            player_decision_times=player_decision_times,
            avg_decision_times=avg_decision_times,
            match_number=match_number,
            multi_player_info={
                'final_rankings': final_rankings,
                'rewards': result.get('rewards', {}),
                'timeout_info': result.get('timeout_info', [])
            }
        )
    
    def _update_trueskill_multi_player(self, match_result: MatchResult, agents: List[BaseAgent]):
        """
        Update TrueSkill ratings for multi-player match.
        
        Args:
            match_result: Result of the multi-player match
            agents: List of agents that participated
        """
        # Extract final rankings from match result
        final_rankings = match_result.multi_player_info.get('final_rankings', [])
        
        # If no rankings available, treat as all tied
        if not final_rankings:
            final_rankings = [agent.name for agent in agents]
        
        # Create ranking groups for TrueSkill
        # Each group contains agents that tied for that position
        ranking_groups = []
        
        # Group agents by their final position
        position_groups = {}
        for i, agent_name in enumerate(final_rankings):
            if i not in position_groups:
                position_groups[i] = []
            position_groups[i].append(agent_name)
        
        # Convert to TrueSkill ranking format
        for position in sorted(position_groups.keys()):
            ranking_groups.append(position_groups[position])
        
        # TrueSkill requires at least 2 groups. If we only have 1 group (all tied),
        # we need to handle it differently
        if len(ranking_groups) < 2:
            # All players tied - use draw update for all pairs
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    self.trueskill_system.update_ratings_two_player(
                        agents[i].name, agents[j].name, winner='draw'
                    )
        else:
            # Update TrueSkill ratings with multiple groups
            self.trueskill_system.update_ratings_multiplayer(ranking_groups)
        
        # Update individual agent statistics
        for agent in agents:
            # Update TrueSkill ratings
            for coder_result in self.coder_results:
                if coder_result.name == agent.name:
                    coder_result.trueskill_rating = self.trueskill_system.get_rating(agent.name)
                    coder_result.conservative_rating = self.trueskill_system.get_conservative_rating(agent.name)
                    
                    # Update win/loss/draw stats based on final ranking
                    agent_position = final_rankings.index(agent.name) if agent.name in final_rankings else len(final_rankings)
                    
                    if agent_position == 0:
                        coder_result.wins += 1
                    elif agent_position == len(final_rankings) - 1:
                        coder_result.losses += 1
                    else:
                        coder_result.draws += 1
                    
                    # Note: win_rate is calculated as a property in CoderResult, not stored as an attribute
                    break
    
    
    def _run_single_match(self, agent1: BaseAgent, agent2: BaseAgent, role1: str, role2: str) -> MatchResult:
        """Run a single match between two agents."""
        match_start_time = time.time()
        
        # Setup and run the match with error handling
        setup_match_func = getattr(self.game, 'setup_match', None)
        if setup_match_func:
            setup_match_func(agent1, agent2)
        
        try:
            result = self.game.run_match()
            match_duration = time.time() - match_start_time
        except Exception as e:
            # Handle in-game errors
            match_duration = time.time() - match_start_time
            error_message = str(e)
            print(f"In-game error occurred: {error_message}")
            
            # Determine which agent caused the error based on current player
            current_agent = getattr(self.game, 'current_agent', None)
            error_agent_name = None
            
            if current_agent:
                error_agent_name = current_agent.name
            else:
                # Fallback: assume last active agent caused error
                # This is a heuristic but better than nothing
                error_agent_name = agent1.name  # Could be improved with better game state tracking
            
            # Update in-game error statistics
            for coder_result in self.coder_results:
                if coder_result.name == error_agent_name:
                    coder_result.in_game_error_count += 1
                    coder_result.in_game_errors.append({
                        'match': f"{agent1.name}_vs_{agent2.name}",
                        'error_type': 'runtime_error',
                        'error_message': error_message,
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"Updated in-game error stats for {error_agent_name}: {coder_result.in_game_error_count} errors total")
                    break
            
            # Create a result for the failed match
            # The agent that didn't cause the error wins by default
            winner = agent2.name if error_agent_name == agent1.name else agent1.name
            result = {
                'winner': winner,
                'moves': 0,
                'scores': {agent1.name: -1 if error_agent_name == agent1.name else 1, 
                          agent2.name: -1 if error_agent_name == agent2.name else 1},
                'match_history': [],
                'move_history_with_timing': [],
                'error_info': {
                    'error_agent': error_agent_name,
                    'error_message': error_message,
                    'error_type': 'in_game_error'
                }
            }
        
        # Handle timeout cases and update coder statistics
        if 'timeout_info' in result:
            timeout_info = result['timeout_info']
            timed_out_agent_name = timeout_info['timed_out_agent']
            
            # Find the coder that timed out and update their statistics
            for coder_result in self.coder_results:
                if coder_result.name == timed_out_agent_name:
                    coder_result.timeout_count += 1
                    coder_result.timeout_details.append({
                        'match': f"{agent1.name}_vs_{agent2.name}",
                        'timeout_at_move': timeout_info['timeout_at_move'],
                        'decision_time': timeout_info.get('decision_time', 0.0),
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"Updated timeout stats for {timed_out_agent_name}: {coder_result.timeout_count} timeouts total")
                    break
        
        # Handle SinglePlayerGame errors that are passed in agent results
        if 'agent1_result' in result and 'error_info' in result['agent1_result']:
            agent1_error = result['agent1_result']['error_info']
            for coder_result in self.coder_results:
                if coder_result.name == agent1_error['error_agent']:
                    coder_result.in_game_error_count += 1
                    coder_result.in_game_errors.append({
                        'match': f"{agent1.name}_vs_{agent2.name}",
                        'error_type': agent1_error['error_type'],
                        'error_message': agent1_error['error_message'],
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"Updated in-game error stats for {agent1_error['error_agent']}: {coder_result.in_game_error_count} errors total")
                    break
        
        if 'agent2_result' in result and 'error_info' in result['agent2_result']:
            agent2_error = result['agent2_result']['error_info']
            for coder_result in self.coder_results:
                if coder_result.name == agent2_error['error_agent']:
                    coder_result.in_game_error_count += 1
                    coder_result.in_game_errors.append({
                        'match': f"{agent1.name}_vs_{agent2.name}",
                        'error_type': agent2_error['error_type'],
                        'error_message': agent2_error['error_message'],
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"Updated in-game error stats for {agent2_error['error_agent']}: {coder_result.in_game_error_count} errors total")
                    break
        
        # Handle SinglePlayerGame timeout info that is passed in agent results
        if 'agent1_result' in result and 'timeout_info' in result['agent1_result']:
            agent1_timeout = result['agent1_result']['timeout_info']
            for coder_result in self.coder_results:
                if coder_result.name == agent1_timeout['timed_out_agent']:
                    coder_result.timeout_count += 1
                    coder_result.timeout_details.append({
                        'match': f"{agent1.name}_vs_{agent2.name}",
                        'timeout_at_move': agent1_timeout['timeout_at_move'],
                        'decision_time': agent1_timeout.get('decision_time', 0.0),
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"Updated timeout stats for {agent1_timeout['timed_out_agent']}: {coder_result.timeout_count} timeouts total")
                    break
        
        if 'agent2_result' in result and 'timeout_info' in result['agent2_result']:
            agent2_timeout = result['agent2_result']['timeout_info']
            for coder_result in self.coder_results:
                if coder_result.name == agent2_timeout['timed_out_agent']:
                    coder_result.timeout_count += 1
                    coder_result.timeout_details.append({
                        'match': f"{agent1.name}_vs_{agent2.name}",
                        'timeout_at_move': agent2_timeout['timeout_at_move'],
                        'decision_time': agent2_timeout.get('decision_time', 0.0),
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"Updated timeout stats for {agent2_timeout['timed_out_agent']}: {coder_result.timeout_count} timeouts total")
                    break
        
        # Save visualization if enabled
        if hasattr(self, 'save_visualizations') and self.save_visualizations:
            self._save_match_visualization(agent1, agent2, result['winner'])
        
        # Extract detailed move history with timing
        move_history = result.get('match_history', [])
        move_history_with_timing = result.get('move_history_with_timing', [])
        
        # If no detailed timing records, create basic format
        if not move_history_with_timing and move_history:
            move_history_with_timing = [(player, action, 0.0) for player, action in move_history]
        
        # Collect decision time statistics for each player
        player_decision_times = {agent1.name: [], agent2.name: []}
        for player, action, decision_time in move_history_with_timing:
            if player in player_decision_times:
                player_decision_times[player].append(decision_time)
        
        # Update decision times in coder results
        for coder_result in self.coder_results:
            if coder_result.name in player_decision_times:
                coder_result.decision_times.extend(player_decision_times[coder_result.name])
        
        # Get final board state (reconstruct from move history based on game type)
        final_board_state = self._reconstruct_final_board_state([(p, a) for p, a, _ in move_history_with_timing])
        
        # Log detailed match information using new logger
        if self.logger:
            match_id = f"{agent1.name}_vs_{agent2.name}_{int(time.time())}"
            match_record = MatchRecord(
                match_id=match_id,
                game=self.game.__class__.__name__.lower().replace('game', ''),
                player1=agent1.name,
                player2=agent2.name,
                player1_role=role1,
                player2_role=role2,
                winner=result['winner'],
                final_scores=result['scores'],
                move_history=move_history_with_timing,
                match_duration=match_duration,
                total_moves=result['moves'],
                final_board_state=final_board_state
            )
            self.logger.log_match(match_record)
        
        # Convert result to MatchResult
        # Calculate average decision times
        avg_decision_times = {}
        for player, times in player_decision_times.items():
            avg_decision_times[player] = sum(times) / len(times) if times else 0.0
        
        return MatchResult(
            winner=result['winner'],
            scores=result['scores'],
            moves=result['moves'],
            match_duration=match_duration,
            match_history=move_history,
            player_decision_times=player_decision_times,
            avg_decision_times=avg_decision_times,
            # Legacy fields for compatibility
            agent1_name=agent1.name,
            agent2_name=agent2.name,
            agent1_score=result['scores'].get(agent1.name, 0),
            agent2_score=result['scores'].get(agent2.name, 0),
            agent1_role=role1,
            agent2_role=role2,
            move_history=move_history_with_timing,
            final_board_state=final_board_state
        )
    
    def _save_match_visualization(self, agent1: BaseAgent, agent2: BaseAgent, winner: str) -> None:
        """Save a visualization of the final game state."""
        if not self.experiment_folder:
            return
        
        try:
            # Create match ID
            timestamp = datetime.now().strftime("%H%M%S")
            match_id = f"{agent1.name}_vs_{agent2.name}_{timestamp}"
            
            # Create visualizations directory
            vis_dir = f"{self.experiment_folder}/visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save visualization
            vis_path = f"{vis_dir}/{match_id}_{self.game.game_name.lower()}.png"
            success = self.game.save_visualization(vis_path)
            
            if success:
                print(f"Game visualization saved: {vis_path}")
            else:
                print(f" Failed to save visualization for match: {match_id}")
                
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def _reconstruct_final_board_state(self, move_history: List[Tuple[str, int]]) -> List[int]:
        """Reconstruct the final board state from move history based on game type."""
        # Import here to avoid circular imports
        from ..games.sudoku import SudokuGame
        
        # For Sudoku, return the solution directly
        if isinstance(self.game, SudokuGame):
            # For Sudoku, the action is the complete solution
            if move_history and len(move_history) > 0:
                last_solution = move_history[-1][1]  # Get the last solution
                if isinstance(last_solution, (list, tuple)) and len(last_solution) == 81:
                    return list(last_solution)
            return [0] * 81  # Empty board if no valid solution
        
        # Initialize board based on game type
        if isinstance(self.game, TicTacToeGame):
            board = [0] * 9
            board_size = 9
        elif isinstance(self.game, ConnectFourGame):
            board = [0] * 42  # 6x7 = 42
            board_size = 42
        else:
            # Default fallback
            board = [0] * 9
            board_size = 9
        
        # Apply moves in sequence
        for i, (player_name, action) in enumerate(move_history):
            if action is not None and isinstance(action, int) and 0 <= action < board_size:
                # For Connect Four, we need different logic
                if isinstance(self.game, ConnectFourGame):
                    # Find the lowest available row in the selected column
                    col = action
                    for row in range(5, -1, -1):  # Start from bottom row (5) to top (0)
                        pos = row * 7 + col
                        if board[pos] == 0:
                            board[pos] = 1 if i % 2 == 0 else 2
                            break
                else:
                    # For TicTacToe and other games
                    player_symbol = 1 if i % 2 == 0 else 2
                    board[action] = player_symbol
        
        return board
    
    def _update_trueskill_and_stats(self, match_result: MatchResult, coder1: CoderResult, coder2: CoderResult):
        """Update TrueSkill ratings and win/loss statistics."""
        # Update win/loss statistics
        if match_result.winner == coder1.name:
            coder1.wins += 1
            coder2.losses += 1
            winner = coder1.name
        elif match_result.winner == coder2.name:
            coder1.losses += 1
            coder2.wins += 1
            winner = coder2.name
        else:  # draw
            coder1.draws += 1
            coder2.draws += 1
            winner = None
        
        # Update TrueSkill ratings
        self.trueskill_system.update_ratings_two_player(
            coder1.name, coder2.name, 
            winner=winner, 
            is_draw=(match_result.winner == "draw")
        )
        
        # Update coder objects with new ratings
        coder1.trueskill_rating = self.trueskill_system.get_rating(coder1.name)
        coder1.conservative_rating = self.trueskill_system.get_conservative_rating(coder1.name)
        coder1.skill_estimate = self.trueskill_system.get_skill_estimate(coder1.name)
        coder1.uncertainty = self.trueskill_system.get_uncertainty(coder1.name)
        
        coder2.trueskill_rating = self.trueskill_system.get_rating(coder2.name)
        coder2.conservative_rating = self.trueskill_system.get_conservative_rating(coder2.name)
        coder2.skill_estimate = self.trueskill_system.get_skill_estimate(coder2.name)
        coder2.uncertainty = self.trueskill_system.get_uncertainty(coder2.name)
    
    def generate_final_rankings(self) -> List[Dict[str, Any]]:
        """Generate final tournament rankings."""
        # Separate passing and failed coders
        passing_rankings = []
        failed_rankings = []
        
        for coder_result in self.coder_results:
            if not coder_result.passed_tests:
                # Failed coders get their initial TrueSkill rating
                current_conservative_rating = self.trueskill_system.get_conservative_rating(coder_result.name)
                ranking_data = {
                    'rank': 0,  # Will be set to last position(s)
                    'name': coder_result.name,
                    'conservative_rating': current_conservative_rating,
                    'skill_estimate': self.trueskill_system.get_skill_estimate(coder_result.name),
                    'uncertainty': self.trueskill_system.get_uncertainty(coder_result.name),
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                    'total_games': 0,
                    'win_rate': 0.0,
                    'status': 'Failed Testing'
                }
                failed_rankings.append(ranking_data)
            else:
                total_games = coder_result.wins + coder_result.losses + coder_result.draws
                win_rate = coder_result.wins / total_games if total_games > 0 else 0.0
                
                ranking_data = {
                    'rank': 0,  # Will be set based on TrueSkill ranking
                    'name': coder_result.name,
                    'conservative_rating': coder_result.conservative_rating,
                    'skill_estimate': coder_result.skill_estimate,
                    'uncertainty': coder_result.uncertainty,
                    'wins': coder_result.wins,
                    'losses': coder_result.losses,
                    'draws': coder_result.draws,
                    'total_games': total_games,
                    'win_rate': win_rate,
                    'status': 'Tournament Participant'
                }
                passing_rankings.append(ranking_data)
        
        # Sort passing coders by conservative TrueSkill rating (descending)
        passing_rankings.sort(key=lambda x: x['conservative_rating'], reverse=True)
        
        # Assign ranks to passing coders
        for i, ranking in enumerate(passing_rankings, 1):
            ranking['rank'] = i
        
        # All failed coders get the same rank (last position)
        last_rank = len(passing_rankings) + 1
        for ranking in failed_rankings:
            ranking['rank'] = last_rank
        
        # Combine rankings: passing coders first, then failed coders
        rankings = passing_rankings + failed_rankings
        
        return rankings
    
    def save_tournament_results(self) -> str:
        """Save complete tournament results to files with simplified structure."""
        if not self.experiment_folder:
            raise ValueError("Experiment folder not created.")
        
        # Ensure results directory exists
        results_dir = f"{self.experiment_folder}/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate final rankings
        final_rankings = self.generate_final_rankings()
        
        # 1. Save simplified tournament results (basic info only, no detailed movements)
        tournament_data = {
            'experiment_folder': self.experiment_folder,
            'timestamp': datetime.now().isoformat(),
            'game_name': self.game.game_name,
            'total_coders': len(self.coder_results),
            'passing_coders': len([c for c in self.coder_results if c.passed_tests]),
            'total_matches': len(self.match_results),
            'final_rankings': final_rankings,
            'match_results': [
                {
                    'agent1': match.agent1_name,
                    'agent2': match.agent2_name,
                    'winner': match.winner,
                    'roles': {match.agent1_name: match.agent1_role, match.agent2_name: match.agent2_role},
                    'scores': {match.agent1_name: match.agent1_score, match.agent2_name: match.agent2_score},
                    'total_moves': match.moves,
                    'match_duration': match.match_duration
                }
                for match in self.match_results
            ]
        }
        
        # 2. Save detailed game history (including all movement information)
        game_history_data = {
            'game_name': self.game.game_name,
            'timestamp': datetime.now().isoformat(),
            'matches': [
                {
                    'match_id': f"{match.agent1_name}_vs_{match.agent2_name}_{i}",
                    'players': {
                        'first': match.agent1_name if match.agent1_role == "first" else match.agent2_name,
                        'second': match.agent2_name if match.agent2_role == "second" else match.agent1_name
                    },
                    'winner': match.winner,
                    'match_duration': match.match_duration,
                    'move_history': [
                        {
                            'player': move[0],
                            'action': move[1],
                            'decision_time': move[2] if len(move) > 2 else 0.0,
                            'move_number': j + 1
                        }
                        for j, move in enumerate(match.move_history)
                    ],
                    'final_board_state': match.final_board_state,
                    'player_decision_stats': {
                        player: {
                            'total_decisions': len(times),
                            'average_decision_time': sum(times) / len(times) if times else 0.0,
                            'total_decision_time': sum(times)
                        }
                        for player, times in match.player_decision_times.items()
                    }
                }
                for i, match in enumerate(self.match_results)
            ]
        }
        
        # 3. Save coder performance statistics
        coder_performance_data = {
            'timestamp': datetime.now().isoformat(),
            'game_name': self.game.game_name,
            'coders': []
        }
        
        for coder in self.coder_results:
            # Compute average decision time
            all_decision_times = []
            for match in self.match_results:
                if coder.name in match.player_decision_times:
                    all_decision_times.extend(match.player_decision_times[coder.name])
            
            avg_decision_time = sum(all_decision_times) / len(all_decision_times) if all_decision_times else 0.0
            
            # Pull detailed stats from logger
            detailed_stats = {}
            if self.logger:
                detailed_stats = self.logger.get_coder_statistics(coder.name, self.game.game_name)
            
            coder_stats = {
                'name': coder.name,
                'passed_tests': coder.passed_tests,
                'revision_count': detailed_stats.get('revision_count', 0),
                'total_revision_time': detailed_stats.get('total_revision_time', 0.0),
                'average_decision_time': avg_decision_time,
                'total_decisions': len(all_decision_times),
                'conservative_rating': coder.conservative_rating,
                'wins': coder.wins,
                'losses': coder.losses,
                'draws': coder.draws,
                'total_games': coder.wins + coder.losses + coder.draws,
                'win_rate': coder.wins / (coder.wins + coder.losses + coder.draws) if (coder.wins + coder.losses + coder.draws) > 0 else 0.0,
                'test_failure_details': detailed_stats.get('failure_details', []),
                'error_type_counts': detailed_stats.get('error_type_counts', {}),
                'total_test_failures': detailed_stats.get('total_test_failures', 0)
            }
            coder_performance_data['coders'].append(coder_stats)
        
        # Save three files to results directory
        tournament_file = f"{results_dir}/tournament_results.json"
        with open(tournament_file, 'w', encoding='utf-8') as f:
            json.dump(_convert_to_serializable(tournament_data), f, indent=2, ensure_ascii=False)
        
        game_history_file = f"{results_dir}/game_history.json"
        with open(game_history_file, 'w', encoding='utf-8') as f:
            json.dump(_convert_to_serializable(game_history_data), f, indent=2, ensure_ascii=False)
        
        coder_performance_file = f"{results_dir}/coder_performance.json"
        with open(coder_performance_file, 'w', encoding='utf-8') as f:
            json.dump(_convert_to_serializable(coder_performance_data), f, indent=2, ensure_ascii=False)
        
        return tournament_file
    
    def run_full_tournament(self, require_api_key: bool = True, max_revisions: int = 3, save_visualizations: bool = False) -> Dict[str, Any]:
        """
        Run a complete tournament from start to finish.
        
        Args:
            require_api_key: Whether to require API key for LLM generation
            max_revisions: Maximum revision attempts per coder
            save_visualizations: Whether to save game visualizations
            
        Returns:
            Complete tournament results
        """
        # Store visualization setting
        self.save_visualizations = save_visualizations
        
        # Create experiment folder
        exp_folder = self.create_experiment_folder()
        
        # Initialize logger
        if self.logger:
            self.logger.experiment_start()
        
        # Check API key if required
        if require_api_key and not os.getenv("OPENROUTER_API_KEY"):
            if self.logger:
                self.logger.error("No API key available - cannot generate agents")
            raise RuntimeError("OpenRouter API key required for agent generation")
        
        if self.logger:
            self.logger.success("OpenRouter API key found")
            self.logger.info(f"Tournament folder created: {exp_folder}")
            self.logger.info(f"Game: {self.game.game_name}")
            self.logger.info(f"Coders: {[info['name'] for info in self.coders_info]}")
            self.logger.info("")
        
        # Phase 1: Generate and test all coders
        self.generate_and_test_all_coders(max_revisions)
        
        # Phase 2: Run appropriate tournament type
        passing_coders_count = len([c for c in self.coder_results if c.passed_tests and c.agent_instance])
        if passing_coders_count >= 2:
            # Detect tournament type based on game
            if isinstance(self.game, MultiPlayerGame):
                if self.logger:
                    self.logger.info(f"Detected multi-player game: {self.game.game_name}")
                self.run_multi_player_tournament(matches_per_agent=100)
            else:
                if self.logger:
                    game_kind = "single-player" if isinstance(self.game, SinglePlayerGame) else "two-player"
                    self.logger.info(f"Detected {game_kind} game: {self.game.game_name}")
                self.run_round_robin_tournament()
        else:
            if self.logger:
                self.logger.error(f"Not enough coders passed testing for tournament (need 2, got {passing_coders_count})")
                self.logger.error("   Note: Coders must have both passed_tests=True AND valid agent_instance")
                if isinstance(self.game, MultiPlayerGame):
                    self.logger.error("   Multi-player games cannot run with insufficient agents - skipping tournament")
        
        # Phase 3: Generate and save results
        results_file = self.save_tournament_results()
        final_rankings = self.generate_final_rankings()

        # Save detailed logs
        if self.logger:
            self.logger.save_detailed_logs(exp_folder)
            self.logger.info(f"Detailed performance and game history saved to: {exp_folder}")

        # Log final results
        self._log_final_results(final_rankings)

        # Build a single-round RoundResult and reuse the multi-round stats path
        # so downstream reporting (which is keyed on `multi_round_stats`) works
        # uniformly for both single-round and multi-round tournaments.
        round_result = RoundResult(
            round_number=1,
            coder_results=self.coder_results,
            match_results=self.match_results,
            final_rankings=final_rankings,
            experiment_folder=exp_folder,
            results_file=results_file,
        )
        multi_round_stats = self._calculate_enhanced_multi_round_stats([round_result])

        return {
            'experiment_folder': exp_folder,
            'results_file': results_file,
            'final_rankings': final_rankings,
            'match_results': self.match_results,
            'coder_results': self.coder_results,
            'multi_round_stats': multi_round_stats,
        }
    
    def _log_final_results(self, final_rankings: List[Dict[str, Any]]):
        """Log final tournament results."""
        if self.logger:
            self.logger.info("")
            self.logger.section_header("TOURNAMENT FINAL RANKINGS")
            
            self.logger.info("")
            for ranking in final_rankings:
                if ranking['status'] == 'Failed Testing':
                    self.logger.info(f"#{ranking['rank']} {ranking['name']} - {ranking['status']} (TrueSkill: {ranking['conservative_rating']:.0f})")
                else:
                    win_rate_pct = ranking['win_rate'] * 100
                    self.logger.info(f"#{ranking['rank']} {ranking['name']} - TrueSkill: {ranking['conservative_rating']:.1f} "
                                   f"({ranking['wins']}-{ranking['losses']}-{ranking['draws']}, {win_rate_pct:.1f}% win rate)")
            
            self.logger.info("")
            self.logger.section_header("", 60)

    def run_multi_round_tournament(self, num_rounds: int = 5, require_api_key: bool = True, max_revisions: int = 3, save_visualizations: bool = False) -> Dict[str, Any]:
        """
        Run multiple tournament rounds for robustness testing.
        
        Args:
            num_rounds: Number of tournament rounds to run
            require_api_key: Whether to require API key for LLM generation
            max_revisions: Maximum revision attempts per coder
            save_visualizations: Whether to save game visualizations
            
        Returns:
            Multi-round tournament results with robustness statistics
        """
        print(f"Starting multi-round tournament ({num_rounds} rounds)")
        print(f"Game: {self.game.game_name}")
        print(f"Coders: {[info['name'] for info in self.coders_info]}")
        print("=" * 60)
        
        # Use existing experiment folder if already set, otherwise create new one
        if hasattr(self, 'experiment_folder') and self.experiment_folder:
            main_exp_folder = self.experiment_folder
            print(f"Using existing experiment folder: {main_exp_folder}")
        else:
            # Create main experiment folder for all rounds (fallback for standalone usage)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            main_exp_folder = f"experiments/multi_round_{self.game.game_name.lower()}_{timestamp}"
            os.makedirs(main_exp_folder, exist_ok=True)
            print(f"Main experiment folder: {main_exp_folder}")
        
        # Create necessary subfolders
        os.makedirs(f"{main_exp_folder}/rounds", exist_ok=True)
        os.makedirs(f"{main_exp_folder}/coder_evaluations", exist_ok=True)
        
        round_results: List[RoundResult] = []
        
        # Run each round
        for round_num in range(1, num_rounds + 1):
            print(f"\nRound {round_num}/{num_rounds}")
            print("-" * 40)
            
            # Create subfolder for this round
            round_folder = f"{main_exp_folder}/rounds/round_{round_num:02d}"
            os.makedirs(round_folder, exist_ok=True)
            os.makedirs(f"{round_folder}/agents", exist_ok=True)  # Create agents subfolder
            os.makedirs(f"{round_folder}/results", exist_ok=True)  # Create results subfolder
            
            # Create a fresh tournament manager for this round
            round_manager = TournamentManager(self.game, self.coders_info, self.prompt_generator)
            
            # Override the experiment folder to use our round folder
            round_manager.experiment_folder = round_folder
            round_manager.logger = TournamentLogger(name=f"Round_{round_num}")
            
            # Run single tournament round with enhanced error tracking
            try:
                single_round_results = self._run_single_round_with_error_tracking(
                    round_manager, round_num, require_api_key, max_revisions, save_visualizations
                )
                
                # Create round result
                round_result = RoundResult(
                    round_number=round_num,
                    coder_results=single_round_results['coder_results'],
                    match_results=single_round_results['match_results'],
                    final_rankings=single_round_results['final_rankings'],
                    experiment_folder=round_folder,
                    results_file=single_round_results['results_file']
                )
                
                round_results.append(round_result)
                
                print(f"Round {round_num} completed")
                
                # Log round summary
                passed_coders = [c for c in round_result.coder_results if c.passed_tests]
                print(f"   {len(passed_coders)}/{len(round_result.coder_results)} coders passed tests")
                if passed_coders:
                    top_coder = max(passed_coders, key=lambda x: x.conservative_rating)
                    print(f"   Round winner: {top_coder.name} (TrueSkill: {top_coder.conservative_rating:.1f})")
                
            except Exception as e:
                print(f"Round {round_num} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nAll {len(round_results)} rounds completed!")
        
        # Calculate enhanced multi-round statistics
        multi_round_stats = self._calculate_enhanced_multi_round_stats(round_results)
        
        # Save comprehensive results
        combined_results = self._save_comprehensive_multi_round_results(
            main_exp_folder, round_results, multi_round_stats
        )
        
        # Save individual coder evaluations
        self._save_individual_coder_evaluations(main_exp_folder, multi_round_stats)
        
        # Log final multi-round results
        self._log_enhanced_multi_round_results(multi_round_stats)
        
        return {
            'main_experiment_folder': main_exp_folder,
            'num_rounds': len(round_results),
            'round_results': round_results,
            'multi_round_stats': multi_round_stats,
            'combined_results_file': combined_results,
            'game_name': self.game.game_name
        }

    def _calculate_enhanced_multi_round_stats(self, round_results: List[RoundResult]) -> Dict[str, MultiRoundStats]:
        """Calculate enhanced statistics across multiple rounds for each coder."""
        coder_stats = {}
        
        # Initialize stats for each coder
        for coder_info in self.coders_info:
            coder_name = coder_info['name']
            coder_stats[coder_name] = MultiRoundStats(
                name=coder_name,
                conservative_ratings=[],
                wins=[],
                losses=[],
                draws=[],
                win_rates=[],
                passed_tests_count=0,
                total_rounds=len(round_results)
            )
        
        # Collect data from each round
        for round_num, round_result in enumerate(round_results, 1):
            # Create round detail for each coder
            round_detail = {
                'round_number': round_num,
                'experiment_folder': round_result.experiment_folder,
                'coder_performances': {}
            }

            # Count actual matches each coder participated in this round.
            # A coder participated in a match if its name appears in the score dict.
            matches_played_this_round: Dict[str, int] = {info['name']: 0 for info in self.coders_info}
            for match in round_result.match_results:
                for participant_name in match.scores.keys():
                    if participant_name in matches_played_this_round:
                        matches_played_this_round[participant_name] += 1

            for ranking in round_result.final_rankings:
                coder_name = ranking['name']
                stats = coder_stats[coder_name]
                stats.matches_played_per_round.append(matches_played_this_round.get(coder_name, 0))
                
                # Find detailed coder result for this round
                coder_result = None
                for cr in round_result.coder_results:
                    if cr.name == coder_name:
                        coder_result = cr
                        break
                
                if coder_result:
                    if ranking['status'] == 'Failed Testing':
                        # Failed testing - record initial TrueSkill rating, no matches played
                        stats.conservative_ratings.append(ranking['conservative_rating'])
                        stats.wins.append(0)
                        stats.losses.append(0)  # No losses since no matches played
                        stats.draws.append(0)
                        stats.win_rates.append(0.0)
                        stats.revision_counts.append(coder_result.revision_count)
                        stats.code_generation_times.append(coder_result.code_generation_time)
                        stats.total_testing_times.append(coder_result.total_testing_time)
                        stats.in_game_error_counts.append(coder_result.in_game_error_count)
                        stats.avg_decision_times.append(0.0)
                        stats.max_decision_times.append(0.0)
                        
                        # Add timeout tracking
                        stats.timeout_counts.append(coder_result.timeout_count)
                        stats.all_timeout_details.extend([
                            {**timeout, 'round': round_num} for timeout in coder_result.timeout_details
                        ])
                        
                        # Add test statistics
                        stats.total_tests_per_round.append(coder_result.total_tests)
                        stats.passed_tests_per_round.append(coder_result.passed_test_count)
                        stats.failed_tests_per_round.append(coder_result.failed_tests)
                        
                        # Add error details
                        stats.all_test_failures.extend([
                            {**error, 'round': round_num} for error in coder_result.test_failures
                        ])
                    else:
                        # Passed testing - record actual performance
                        stats.conservative_ratings.append(ranking['conservative_rating'])
                        stats.wins.append(ranking['wins'])
                        stats.losses.append(ranking['losses'])
                        stats.draws.append(ranking['draws'])
                        stats.win_rates.append(ranking['win_rate'])
                        stats.passed_tests_count += 1
                        stats.revision_counts.append(coder_result.revision_count)
                        stats.code_generation_times.append(coder_result.code_generation_time)
                        stats.total_testing_times.append(coder_result.total_testing_time)
                        stats.in_game_error_counts.append(coder_result.in_game_error_count)
                        stats.avg_decision_times.append(coder_result.avg_decision_time)
                        stats.max_decision_times.append(coder_result.max_decision_time)
                        
                        # Add timeout tracking
                        stats.timeout_counts.append(coder_result.timeout_count)
                        stats.all_timeout_details.extend([
                            {**timeout, 'round': round_num} for timeout in coder_result.timeout_details
                        ])
                        
                        # Add test statistics
                        stats.total_tests_per_round.append(coder_result.total_tests)
                        stats.passed_tests_per_round.append(coder_result.passed_test_count)
                        stats.failed_tests_per_round.append(coder_result.failed_tests)
                        
                        # Add error details with round information
                        stats.all_in_game_errors.extend([
                            {**error, 'round': round_num} for error in coder_result.in_game_errors
                        ])
                        stats.all_test_failures.extend([
                            {**error, 'round': round_num} for error in coder_result.test_failures
                        ])
                    
                    # Add to round detail
                    round_detail['coder_performances'][coder_name] = {
                        'status': ranking['status'],
                        'conservative_rating': ranking.get('conservative_rating', 0.0),
                        'wins': ranking.get('wins', 0),
                        'losses': ranking.get('losses', 0),
                        'draws': ranking.get('draws', 0),
                        'win_rate': ranking.get('win_rate', 0.0),
                        'revision_count': coder_result.revision_count,
                        'code_generation_time': coder_result.code_generation_time,
                        'total_testing_time': coder_result.total_testing_time,
                        'in_game_error_count': coder_result.in_game_error_count,
                        'avg_decision_time': coder_result.avg_decision_time,
                        'max_decision_time': coder_result.max_decision_time
                    }
                
                # Add round detail to all coder stats
                for stats in coder_stats.values():
                    stats.round_details.append(round_detail)
        
        return coder_stats

    def _save_comprehensive_multi_round_results(self, main_folder: str, round_results: List[RoundResult], multi_round_stats: Dict[str, MultiRoundStats]) -> str:
        """Save comprehensive multi-round tournament results."""
        # Create summary data
        summary_data = {
            'tournament_info': {
                'type': 'multi_round_enhanced',
                'game': self.game.game_name,
                'total_rounds': len(round_results),
                'coders': [info['name'] for info in self.coders_info],
                'timestamp': datetime.now().isoformat()
            },
            'round_summaries': [],
            'comprehensive_coder_statistics': {},
            'robustness_rankings': [],
            'efficiency_rankings': [],
            'error_analysis': {}
        }
        
        # Add round summaries
        for round_result in round_results:
            round_summary = {
                'round_number': round_result.round_number,
                'experiment_folder': round_result.experiment_folder,
                'results_file': round_result.results_file,
                'rankings': round_result.final_rankings,
                'total_matches': len(round_result.match_results),
                'passing_coders': len([r for r in round_result.final_rankings if r['status'] != 'Failed Testing'])
            }
            summary_data['round_summaries'].append(round_summary)
        
        # Add comprehensive coder statistics
        for coder_name, stats in multi_round_stats.items():
            summary_data['comprehensive_coder_statistics'][coder_name] = {
                'name': stats.name,
                'tournament_summary': {
                    'total_rounds': stats.total_rounds,
                    'passed_tests_count': stats.passed_tests_count,
                    'success_rate': stats.success_rate,
                },
                'performance_statistics': {
                    'conservative_ratings': stats.conservative_ratings,
                    'conservative_rating_max': stats.conservative_rating_max,
                    'conservative_rating_min': stats.conservative_rating_min,
                    'conservative_rating_avg': stats.conservative_rating_avg,
                    'conservative_rating_std': stats.conservative_rating_std,
                    'rating_robustness_score': stats.rating_robustness_score,
                    'wins': stats.wins,
                    'losses': stats.losses,
                    'draws': stats.draws,
                    'win_rates': stats.win_rates,
                    'avg_win_rate': stats.win_rate_avg
                },
                'efficiency_metrics': {
                    'revision_counts': stats.revision_counts,
                    'avg_revision_count': stats.avg_revision_count,
                    'code_generation_times': stats.code_generation_times,
                    'avg_code_generation_time': stats.avg_code_generation_time,
                    'total_testing_times': stats.total_testing_times,
                    'avg_testing_time': stats.avg_testing_time,
                    'avg_decision_times': stats.avg_decision_times,
                    'overall_avg_decision_time': stats.overall_avg_decision_time,
                    'max_decision_times': stats.max_decision_times,
                    'overall_max_decision_time': stats.overall_max_decision_time
                },
                'error_analysis': {
                    'in_game_error_counts': stats.in_game_error_counts,
                    'total_in_game_errors': stats.total_in_game_errors,
                    'avg_in_game_errors_per_round': stats.avg_in_game_errors_per_round,
                    'all_in_game_errors': stats.all_in_game_errors,
                    'all_test_failures': stats.all_test_failures
                },
                'round_by_round_details': stats.round_details
            }
        
        # Create robustness rankings (sorted by robustness score)
        valid_coders = [(name, stats) for name, stats in multi_round_stats.items() if stats.conservative_ratings]
        robustness_ranking = sorted(valid_coders, key=lambda x: x[1].rating_robustness_score, reverse=True)
        
        for rank, (coder_name, stats) in enumerate(robustness_ranking, 1):
            summary_data['robustness_rankings'].append({
                'rank': rank,
                'name': coder_name,
                'robustness_score': stats.rating_robustness_score,
                'conservative_rating_avg': stats.conservative_rating_avg,
                'conservative_rating_std': stats.conservative_rating_std,
                'success_rate': stats.success_rate,
                'total_in_game_errors': stats.total_in_game_errors
            })
        
        # Create efficiency rankings (sorted by avg revision count, then by avg generation time)
        efficiency_ranking = sorted(valid_coders, key=lambda x: (x[1].avg_revision_count, x[1].avg_code_generation_time))
        
        for rank, (coder_name, stats) in enumerate(efficiency_ranking, 1):
            summary_data['efficiency_rankings'].append({
                'rank': rank,
                'name': coder_name,
                'avg_revision_count': stats.avg_revision_count,
                'avg_code_generation_time': stats.avg_code_generation_time,
                'avg_testing_time': stats.avg_testing_time,
                'overall_avg_decision_time': stats.overall_avg_decision_time
            })
        
        # Create error analysis summary
        all_error_types = {}
        for coder_name, stats in multi_round_stats.items():
            coder_errors = {}
            
            # Analyze in-game errors
            for error in stats.all_in_game_errors:
                error_type = error.get('error_type', 'Unknown')
                if error_type not in coder_errors:
                    coder_errors[error_type] = 0
                coder_errors[error_type] += 1
            
            # Analyze test failures
            for failure in stats.all_test_failures:
                failure_type = f"TestFailure_{failure.get('test_name', 'Unknown')}"
                if failure_type not in coder_errors:
                    coder_errors[failure_type] = 0
                coder_errors[failure_type] += 1
            
            summary_data['error_analysis'][coder_name] = coder_errors
            
            # Aggregate error types
            for error_type, count in coder_errors.items():
                if error_type not in all_error_types:
                    all_error_types[error_type] = 0
                all_error_types[error_type] += count
        
        summary_data['error_analysis']['overall_error_distribution'] = all_error_types
        
        # Save to file
        results_file = os.path.join(main_folder, "comprehensive_multi_round_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive results saved to: {results_file}")
        return results_file

    def _save_individual_coder_evaluations(self, main_folder: str, multi_round_stats: Dict[str, MultiRoundStats]):
        """Save individual coder evaluations to separate files (high-level summaries only)."""
        eval_folder = os.path.join(main_folder, "coder_evaluations")
        
        for coder_name, stats in multi_round_stats.items():
            # Create simplified individual coder evaluation (high-level summaries only)
            coder_evaluation = {
                'coder_info': {
                    'name': stats.name,
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'total_rounds': stats.total_rounds
                },
                'overall_performance_summary': {
                    'passed_tests_count': stats.passed_tests_count,
                    'success_rate': stats.success_rate,
                    'rating_robustness_score': stats.rating_robustness_score,
                    'avg_win_rate': stats.win_rate_avg,
                    'total_in_game_errors': stats.total_in_game_errors,
                    'total_timeouts': stats.total_timeouts
                },
                'aggregated_metrics': {
                    'trueskill_performance': {
                        'max': stats.conservative_rating_max,
                        'min': stats.conservative_rating_min,
                        'average': stats.conservative_rating_avg,
                        'standard_deviation': stats.conservative_rating_std,
                        'robustness_score': stats.rating_robustness_score
                    },
                    'efficiency_summary': {
                        'avg_revision_count': stats.avg_revision_count,
                        'avg_testing_time': stats.avg_testing_time,
                        'overall_avg_decision_time': stats.overall_avg_decision_time,
                        'overall_max_decision_time': stats.overall_max_decision_time
                    },
                    'error_summary': {
                        'total_in_game_errors': stats.total_in_game_errors,
                        'avg_in_game_errors_per_round': stats.avg_in_game_errors_per_round,
                        'total_test_failures': len(stats.all_test_failures),
                        'avg_timeouts_per_round': stats.avg_timeouts_per_round
                    }
                },
                'note': "Detailed round-by-round data is available in the game_history folder"
            }
            
            # Save coder evaluation
            safe_coder_name = coder_name.replace(' ', '_').replace('/', '_')
            eval_file = os.path.join(eval_folder, f"{safe_coder_name}_evaluation.json")
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(coder_evaluation, f, indent=2, ensure_ascii=False)
            
            print(f"{coder_name} evaluation saved to: {eval_file}")

    def _log_enhanced_multi_round_results(self, multi_round_stats: Dict[str, MultiRoundStats]):
        """Log enhanced multi-round tournament results."""
        print("\n" + "=" * 80)
        print("ENHANCED MULTI-ROUND TOURNAMENT FINAL RESULTS")
        print("=" * 80)
        
        # Sort coders by robustness score
        valid_coders = [(name, stats) for name, stats in multi_round_stats.items() if stats.conservative_ratings]
        robustness_ranking = sorted(valid_coders, key=lambda x: x[1].rating_robustness_score, reverse=True)
        
        print(f"\nROBUSTNESS RANKINGS (based on TrueSkill consistency):")
        print("-" * 60)
        
        for rank, (coder_name, stats) in enumerate(robustness_ranking, 1):
            print(f"#{rank} {coder_name}")
            print(f"   Robustness Score: {stats.rating_robustness_score:.3f}")
            print(f"   TrueSkill: {stats.conservative_rating_avg:.1f} ± {stats.conservative_rating_std:.1f} (max: {stats.conservative_rating_max:.1f}, min: {stats.conservative_rating_min:.1f})")
            print(f"   Success Rate: {stats.success_rate:.1%} ({stats.passed_tests_count}/{stats.total_rounds})")
            print(f"   Avg Win Rate: {stats.win_rate_avg:.1%}")
            print(f"   Avg Revisions: {stats.avg_revision_count:.1f}")
            print(f"   Avg Code Gen Time: {stats.avg_code_generation_time:.2f}s")
            print(f"   Avg Decision Time: {stats.overall_avg_decision_time:.2e}s")
            print(f"   Total In-Game Errors: {stats.total_in_game_errors}")
            print(f"   Total Timeouts: {stats.total_timeouts}")
            print()
        
        # Show efficiency rankings
        print(f"\nEFFICIENCY RANKINGS (based on speed and revision count):")
        print("-" * 60)
        
        efficiency_ranking = sorted(valid_coders, key=lambda x: (x[1].avg_revision_count, x[1].avg_code_generation_time))
        
        for rank, (coder_name, stats) in enumerate(efficiency_ranking, 1):
            print(f"#{rank} {coder_name}")
            print(f"   Avg Revisions: {stats.avg_revision_count:.1f}")
            print(f"   Avg Code Gen: {stats.avg_code_generation_time:.2f}s")
            print(f"   Avg Testing: {stats.avg_testing_time:.2f}s")
            print(f"   Avg Decision: {stats.overall_avg_decision_time:.2e}s")
            print()
        
        # Show error analysis
        print(f"\nERROR ANALYSIS:")
        print("-" * 40)
        
        for coder_name, stats in multi_round_stats.items():
            if stats.total_in_game_errors > 0 or stats.all_test_failures:
                print(f"{coder_name}:")
                
                if stats.total_in_game_errors > 0:
                    print(f"   In-Game Errors: {stats.total_in_game_errors}")
                    error_types = {}
                    for error in stats.all_in_game_errors:
                        error_type = error.get('error_type', 'Unknown')
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    for error_type, count in error_types.items():
                        print(f"      • {error_type}: {count}")
                
                if stats.all_test_failures:
                    print(f"   Test Failures: {len(stats.all_test_failures)}")
                    failure_types = {}
                    for failure in stats.all_test_failures:
                        test_name = failure.get('test_name', 'Unknown')
                        failure_types[test_name] = failure_types.get(test_name, 0) + 1
                    
                    for test_name, count in failure_types.items():
                        print(f"      • {test_name}: {count}")
                
                print()
        
        # Show failed coders
        failed_coders = [(name, stats) for name, stats in multi_round_stats.items() if not stats.conservative_ratings]
        if failed_coders:
            print("CODERS THAT FAILED ALL ROUNDS:")
            print("-" * 40)
            for coder_name, stats in failed_coders:
                print(f"   {coder_name} - Failed all {stats.total_rounds} rounds")
                if stats.all_test_failures:
                    print(f"      Common test failures: {len(stats.all_test_failures)}")
            print()
        
        print("=" * 80)

    def _run_single_round_with_error_tracking(self, round_manager, round_num: int, require_api_key: bool, max_revisions: int, save_visualizations: bool) -> Dict[str, Any]:
        """
        Run a single tournament round with enhanced error tracking.
        
        Args:
            round_manager: Tournament manager instance for this round
            round_num: Round number
            require_api_key: Whether to require API key
            max_revisions: Maximum revisions per coder
            save_visualizations: Whether to save visualizations
            
        Returns:
            Single round results with enhanced error information
        """
        # Set visualization setting
        round_manager.save_visualizations = save_visualizations
        
        # Initialize logger
        if round_manager.logger:
            round_manager.logger.info(f"Starting Round {round_num}")
            round_manager.logger.experiment_start()
        
        # Check API key if required
        if require_api_key and not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OpenRouter API key required for agent generation")
        
        if round_manager.logger:
            round_manager.logger.success("OpenRouter API key found")
            round_manager.logger.info(f"Game: {round_manager.game.game_name}")
            round_manager.logger.info(f"Coders: {[info['name'] for info in round_manager.coders_info]}")
        
        # Phase 1: Generate and test all coders with enhanced tracking
        round_manager.generate_and_test_all_coders(max_revisions)
        
        # Phase 2: Run appropriate tournament type with error handling
        passing_coders_count = len([c for c in round_manager.coder_results if c.passed_tests and c.agent_instance])
        if passing_coders_count >= 2:
            try:
                # Detect tournament type based on game
                if isinstance(round_manager.game, MultiPlayerGame):
                    if round_manager.logger:
                        round_manager.logger.info(f"Detected multi-player game: {round_manager.game.game_name}")
                    round_manager.run_multi_player_tournament(matches_per_agent=100)
                else:
                    if round_manager.logger:
                        game_kind = "single-player" if isinstance(round_manager.game, SinglePlayerGame) else "two-player"
                        round_manager.logger.info(f"Detected {game_kind} game: {round_manager.game.game_name}")
                    round_manager.run_round_robin_tournament()
            except Exception as e:
                # Log the error but continue with partial results
                print(f"Tournament encountered errors but continued: {str(e)}")
                if round_manager.logger:
                    round_manager.logger.error(f"Tournament errors occurred: {str(e)}")
                # The match results and coder statistics should still be partially collected
                # The error handling should have captured individual game errors
        else:
            if round_manager.logger:
                round_manager.logger.error(f"Not enough coders passed testing for tournament (need 2, got {passing_coders_count})")
                round_manager.logger.error("   Note: Coders must have both passed_tests=True AND valid agent_instance")
                if isinstance(round_manager.game, MultiPlayerGame):
                    round_manager.logger.error("   Multi-player games cannot run with insufficient agents - skipping tournament")
        
        # Phase 3: Generate and save results
        results_file = round_manager.save_tournament_results()
        final_rankings = round_manager.generate_final_rankings()
        
        # Save detailed logs
        if round_manager.logger:
            round_manager.logger.save_detailed_logs(round_manager.experiment_folder)
        
        return {
            'experiment_folder': round_manager.experiment_folder,
            'results_file': results_file,
            'final_rankings': final_rankings,
            'match_results': round_manager.match_results,
            'coder_results': round_manager.coder_results
        }

    def _save_code_version_to_history(self, code_file: str, coder_name: str, version_num: int) -> str:
        """
        Save the current version of code to history folder before overwriting.
        
        Args:
            code_file: Path to current code file
            coder_name: Name of the coder
            version_num: Version number (0 for original, 1+ for revisions)
            
        Returns:
            Path to the saved history version file
        """
        # Create history_version folder if it doesn't exist
        history_folder = f"{self.experiment_folder}/agents/history_version"
        os.makedirs(history_folder, exist_ok=True)
        
        # Generate version filename (start from v1)
        version_filename = f"{coder_name.lower()}-v{version_num + 1}.py"
        
        history_file_path = f"{history_folder}/{version_filename}"
        
        # Copy current code to history
        if os.path.exists(code_file):
            shutil.copy2(code_file, history_file_path)
            
            if self.logger:
                self.logger.info(f"    Saved version to: {version_filename}")
        
        return history_file_path

    def _log_detailed_match_info(self, match_result: MatchResult, agents: List[BaseAgent], match_number: int):
        """
        Log simplified match information showing player actions, winner, and chip changes.
        
        Args:
            match_result: The match result object
            agents: List of agents that participated
            match_number: Current match number
        """
        if not self.logger:
            return
            
        # Get action names from the game if available; fall back to the
        # hard-coded Texas Hold'em mapping below if the optional hook
        # raises (signature drift between game implementations).
        action_names = {}
        if hasattr(self.game, '_get_action_meanings'):
            try:
                action_names = self.game._get_action_meanings()
            except Exception:
                action_names = {}
        
        # Fallback action mapping for Texas Hold'em  
        if not action_names:
            action_names = {
                0: "Call",
                1: "Raise", 
                2: "Fold",
                3: "Check"
            }
        
        # Show player actions from match history
        if match_result.match_history:
            self.logger.info("  Actions:")
            
            for i, move in enumerate(match_result.match_history):
                if isinstance(move, dict):
                    # Format: {'agent': 'AgentName', 'action': 1, 'decision_time': 0.5}
                    agent_name = move.get('agent', 'Unknown')
                    action = move.get('action', -1)
                elif isinstance(move, (list, tuple)) and len(move) >= 2:
                    # Format: ('AgentName', 1, 0.5) or ['AgentName', 1]
                    agent_name = move[0]
                    action = move[1]
                else:
                    continue
                
                action_name = action_names.get(action, f"Unknown({action})")
                
                # Show move details
                self.logger.info(f"     {agent_name} -> {action_name}")
        else:
            self.logger.info("  Actions: No actions recorded")
        
        # Show match outcome
        self.logger.info(f"  Winner: {match_result.winner}")
        
        # Show chip changes (scores)
        if match_result.scores:
            self.logger.info("  Chip Changes:")
            for agent_name, score in match_result.scores.items():
                if score > 0:
                    self.logger.info(f"     {agent_name}: +{score:.1f}")
                elif score < 0:
                    self.logger.info(f"     {agent_name}: {score:.1f}")
                else:
                    self.logger.info(f"     {agent_name}: 0.0")

