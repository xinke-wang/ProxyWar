"""
Test Analysis Module for ProxyWar Framework.

This module provides functionality to analyze test results across different games and coders,
calculating first-attempt pass rates and categorized test pass rates.
"""

import statistics
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .data_models import MultiRoundStats


@dataclass
class TestCategoryResult:
    """Results for a specific test category."""
    category_name: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate for this category."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0


@dataclass
class CoderTestAnalysis:
    """Comprehensive test analysis for a single coder."""
    coder_name: str
    total_tests: int = 0
    total_passed: int = 0
    total_failed: int = 0
    overall_pass_rate: float = 0.0
    
    # Category-wise results
    category_results: Dict[str, TestCategoryResult] = field(default_factory=dict)
    
    # Round-specific data
    rounds_analyzed: int = 0
    first_attempt_only: bool = True  # Whether this analysis includes only first attempts


@dataclass
class GameTestAnalysis:
    """Test analysis results for a specific game."""
    game_name: str
    coder_analyses: Dict[str, CoderTestAnalysis] = field(default_factory=dict)
    
    @property
    def average_pass_rate(self) -> float:
        """Calculate average pass rate across all coders."""
        if not self.coder_analyses:
            return 0.0
        return statistics.mean([analysis.overall_pass_rate for analysis in self.coder_analyses.values()])


class TestAnalyzer:
    """Analyzer for test results across games and coders."""
    
    # Standard test categories that appear across most games
    STANDARD_CATEGORIES = {
        "File Existence",
        "Syntax Validation", 
        "Syntax Validity",  # Alternative name used in some testers
        "Class Structure",
        "Interface Compliance",
        "Basic Behavior",
        "Edge Case Handling",
        "Performance"
    }
    
    # Game-specific categories
    GAME_SPECIFIC_CATEGORIES = {
        "Action Validation",
        "Game Interaction", 
        "Solution Format",
        "Simple Puzzle Solving",
        "Maze Interaction",
        "Scalability"
    }
    
    def __init__(self):
        self.all_categories = self.STANDARD_CATEGORIES.union(self.GAME_SPECIFIC_CATEGORIES)
    
    def analyze_game_tests(self, game_name: str, multi_round_stats: Dict[str, MultiRoundStats], 
                          first_attempt_only: bool = True) -> GameTestAnalysis:
        """
        Analyze test results for a specific game.
        
        Args:
            game_name: Name of the game
            multi_round_stats: Multi-round statistics for all coders
            first_attempt_only: If True, only analyze first attempts (no revisions)
            
        Returns:
            GameTestAnalysis containing detailed test analysis
        """
        game_analysis = GameTestAnalysis(game_name=game_name)
        
        for coder_name, stats in multi_round_stats.items():
            coder_analysis = self._analyze_coder_tests(
                coder_name, stats, first_attempt_only
            )
            game_analysis.coder_analyses[coder_name] = coder_analysis
        
        return game_analysis
    
    def _analyze_coder_tests(self, coder_name: str, stats: MultiRoundStats, 
                           first_attempt_only: bool = True) -> CoderTestAnalysis:
        """
        Analyze test results for a specific coder.
        
        Args:
            coder_name: Name of the coder
            stats: Multi-round statistics
            first_attempt_only: If True, only analyze first attempts
            
        Returns:
            CoderTestAnalysis with detailed results
        """
        analysis = CoderTestAnalysis(
            coder_name=coder_name,
            first_attempt_only=first_attempt_only
        )
        
        # Initialize category results
        for category in self.all_categories:
            analysis.category_results[category] = TestCategoryResult(category_name=category)
        
        # Analyze test failures from each round
        for round_idx, round_detail in enumerate(stats.round_details):
            # If first_attempt_only, we need to identify which tests were first attempts
            round_test_failures = []
            
            # Extract test failures for this specific round and coder
            coder_performance = round_detail.get('coder_performances', {}).get(coder_name, {})
            
            # Get test failures from stats.all_test_failures for this round
            round_failures = [
                failure for failure in stats.all_test_failures 
                if failure.get('round') == round_idx + 1
            ]
            
            if first_attempt_only:
                # Filter to only include failures from initial attempts (not revisions)
                round_failures = [
                    failure for failure in round_failures
                    if 'revision_number' not in failure  # No revision means first attempt
                ]
            
            # Count tests by category
            self._count_tests_by_category(analysis, round_failures, round_idx + 1)
        
        # Calculate overall statistics
        analysis.total_tests = sum(cat.total_tests for cat in analysis.category_results.values())
        analysis.total_passed = sum(cat.passed_tests for cat in analysis.category_results.values())
        analysis.total_failed = sum(cat.failed_tests for cat in analysis.category_results.values())
        analysis.overall_pass_rate = analysis.total_passed / analysis.total_tests if analysis.total_tests > 0 else 0.0
        analysis.rounds_analyzed = len(stats.round_details)
        
        return analysis
    
    def _count_tests_by_category(self, analysis: CoderTestAnalysis, 
                               test_failures: List[Dict[str, Any]], round_num: int) -> None:
        """
        Count tests by category and update analysis.
        
        Args:
            analysis: CoderTestAnalysis to update
            test_failures: List of test failure records
            round_num: Round number being analyzed
        """
        # For each round, we assume there's one test per standard category
        # (This is based on the tester structure we observed)
        
        # Count failed tests by category
        failed_by_category = defaultdict(int)
        for failure in test_failures:
            test_name = failure.get('test_name', 'Unknown')
            category = self._map_test_name_to_category(test_name)
            failed_by_category[category] += 1
        
        # For each category, assume 1 test per round and update counts
        for category_name, category_result in analysis.category_results.items():
            # Standard categories exist in every round
            if category_name in self.STANDARD_CATEGORIES:
                category_result.total_tests += 1
                if category_name in failed_by_category:
                    category_result.failed_tests += 1
                else:
                    category_result.passed_tests += 1
            
            # Game-specific categories might not exist in all games
            elif category_name in failed_by_category:
                category_result.total_tests += 1
                category_result.failed_tests += 1
            # Note: We can't count passed tests for game-specific categories 
            # without knowing if they were actually run
    
    def _map_test_name_to_category(self, test_name: str) -> str:
        """
        Map a test name to its category.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Category name
        """
        test_name_lower = test_name.lower()
        
        # Direct mappings
        if test_name in self.all_categories:
            return test_name
        
        # Fuzzy mappings based on keywords
        if "file" in test_name_lower and "exist" in test_name_lower:
            return "File Existence"
        elif "syntax" in test_name_lower:
            if "validation" in test_name_lower:
                return "Syntax Validation"
            else:
                return "Syntax Validity"
        elif "class" in test_name_lower and "struct" in test_name_lower:
            return "Class Structure"
        elif "interface" in test_name_lower:
            return "Interface Compliance"
        elif "basic" in test_name_lower and "behavior" in test_name_lower:
            return "Basic Behavior"
        elif "action" in test_name_lower and "validation" in test_name_lower:
            return "Action Validation"
        elif "game" in test_name_lower and "interaction" in test_name_lower:
            return "Game Interaction"
        elif "edge" in test_name_lower:
            return "Edge Case Handling"
        elif "performance" in test_name_lower:
            return "Performance"
        elif "solution" in test_name_lower and "format" in test_name_lower:
            return "Solution Format"
        elif "puzzle" in test_name_lower and "solving" in test_name_lower:
            return "Simple Puzzle Solving"
        elif "maze" in test_name_lower:
            return "Maze Interaction"
        elif "scalability" in test_name_lower:
            return "Scalability"
        else:
            return "Other"
    
    def generate_test_analysis_table(self, game_analysis: GameTestAnalysis) -> str:
        """
        Generate a markdown table showing test pass rates.
        
        Args:
            game_analysis: Game test analysis results
            
        Returns:
            Markdown table as string
        """
        if not game_analysis.coder_analyses:
            return "No test data available for analysis.\n"
        
        # Get all categories that have data
        all_categories_with_data = set()
        for analysis in game_analysis.coder_analyses.values():
            for cat_name, cat_result in analysis.category_results.items():
                if cat_result.total_tests > 0:
                    all_categories_with_data.add(cat_name)
        
        # Sort categories: standard categories first, then game-specific
        standard_cats = sorted([cat for cat in all_categories_with_data if cat in self.STANDARD_CATEGORIES])
        game_specific_cats = sorted([cat for cat in all_categories_with_data if cat in self.GAME_SPECIFIC_CATEGORIES])
        other_cats = sorted([cat for cat in all_categories_with_data if cat not in self.all_categories])
        
        ordered_categories = standard_cats + game_specific_cats + other_cats
        
        if not ordered_categories:
            return "No test categories found with data.\n"
        
        # Build table
        lines = []
        lines.append("## Test Pass Rate Analysis (First Attempt Only)")
        lines.append("")
        lines.append("*This table shows the pass rate for initial test attempts, excluding revisions.*")
        lines.append("")
        
        # Table header
        header = "| Coder | Overall Pass Rate |"
        separator = "|-------|-------------------|"
        
        for category in ordered_categories:
            header += f" {category} |"
            separator += "----------|"
        
        lines.append(header)
        lines.append(separator)
        
        # Sort coders by overall pass rate (descending)
        sorted_coders = sorted(
            game_analysis.coder_analyses.items(),
            key=lambda x: x[1].overall_pass_rate,
            reverse=True
        )
        
        # Table rows
        for coder_name, analysis in sorted_coders:
            row = f"| **{coder_name}** | {analysis.overall_pass_rate:.1%} |"
            
            for category in ordered_categories:
                cat_result = analysis.category_results.get(category)
                if cat_result and cat_result.total_tests > 0:
                    row += f" {cat_result.pass_rate:.1%} |"
                else:
                    row += " - |"
            
            lines.append(row)
        
        lines.append("")
        lines.append(f"**Game Average**: {game_analysis.average_pass_rate:.1%}")
        lines.append("")
        
        # Add legend
        lines.append("### Category Definitions")
        lines.append("")
        lines.append("**Standard Categories** (common across all games):")
        for cat in standard_cats:
            lines.append(f"- **{cat}**: Core functionality test")
        
        if game_specific_cats:
            lines.append("")
            lines.append("**Game-Specific Categories**:")
            for cat in game_specific_cats:
                lines.append(f"- **{cat}**: Game-specific functionality test")
        
        if other_cats:
            lines.append("")
            lines.append("**Other Categories**:")
            for cat in other_cats:
                lines.append(f"- **{cat}**: Additional test category")
        
        return "\n".join(lines)


def analyze_cross_game_tests(all_game_results: Dict[str, Any], 
                           first_attempt_only: bool = True) -> Dict[str, GameTestAnalysis]:
    """
    Analyze test results across multiple games.
    
    Args:
        all_game_results: Results from all games
        first_attempt_only: If True, only analyze first attempts
        
    Returns:
        Dictionary mapping game names to their test analyses
    """
    analyzer = TestAnalyzer()
    analyses = {}
    
    for game_name, game_results in all_game_results.items():
        if 'multi_round_stats' in game_results:
            analysis = analyzer.analyze_game_tests(
                game_name, 
                game_results['multi_round_stats'],
                first_attempt_only
            )
            analyses[game_name] = analysis
    
    return analyses


def generate_test_analysis_section(game_analyses: Dict[str, GameTestAnalysis]) -> str:
    """
    Generate a complete test analysis section for multiple games.
    
    Args:
        game_analyses: Test analyses for all games
        
    Returns:
        Complete markdown section
    """
    if not game_analyses:
        return ""
    
    analyzer = TestAnalyzer()
    lines = []
    
    for game_name, game_analysis in game_analyses.items():
        lines.append(f"### {game_name.replace('_', ' ').title()}")
        lines.append("")
        table = analyzer.generate_test_analysis_table(game_analysis)
        lines.append(table)
        lines.append("")
    
    return "\n".join(lines) 