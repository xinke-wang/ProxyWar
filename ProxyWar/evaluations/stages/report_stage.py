"""Report generation stage.

Renders the Markdown / JSON artifacts produced after a tournament run:
per-game reports (Table 4 of the paper), cross-game summary, and the
individual coder analysis files.

All functions in this module are pure with respect to the inputs they are
given (dicts of MultiRoundStats / cross-game stats); they do filesystem
I/O but do not mutate shared tournament state.
"""

import os
import json
import statistics
from datetime import datetime
from typing import Any, Dict

from ..test_analysis import analyze_cross_game_tests, generate_test_analysis_section


def _fmt_rate_cell(passed: int, total: int) -> str:
    """Format a "rate (passed/total)" report cell.

    Returns "N/A" when the category had no tests in this run, instead of the
    misleading "0.0% (0/0)" — readers were interpreting that as "100% failed"
    rather than "no data".
    """
    if total <= 0:
        return "N/A"
    return f"{passed / total:.1%} ({passed}/{total})"


def _classify_test_name(test_name: str) -> str:
    """
    Classify test names into four balanced categories based on actual test analysis.

    Args:
        test_name: Name of the test

    Returns:
        Category name
    """
    test_name_lower = test_name.lower().replace(" ", "_")

    code_structure_tests = ['file_existence', 'syntax_validation', 'class_structure', 'interface_compliance']
    basic_functionality_tests = ['basic_behavior', 'basic_behavior_test', 'action_validation', 'solution_format', 'basic_puzzle_interaction']
    game_interaction_tests = ['game_interaction', 'simple_puzzle_solving', 'maze_interaction', 'all_in_scenario_test',
                             'no_legal_actions_test', 'check_scenario_test', 'all-in_scenario_test']
    robustness_performance_tests = ['edge_case_handling', 'edge_cases', 'performance', 'performance_test', 'scalability']

    for test_keyword in code_structure_tests:
        if test_keyword in test_name_lower:
            return "Code Structure & Validation"

    for test_keyword in basic_functionality_tests:
        if test_keyword in test_name_lower:
            return "Basic Functionality"

    for test_keyword in game_interaction_tests:
        if test_keyword in test_name_lower:
            return "Game Interaction & Logic"

    for test_keyword in robustness_performance_tests:
        if test_keyword in test_name_lower:
            return "Robustness & Performance"

    return "Basic Functionality"


def _analyze_test_failures_by_category(multi_round_stats: Dict) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Analyze test failures by category for each coder.
    Use consistent test case counts based on final version test results.
    """
    coder_category_stats = {}

    all_test_names_by_category = {
        "Code Structure & Validation": set(),
        "Basic Functionality": set(),
        "Game Interaction & Logic": set(),
        "Robustness & Performance": set()
    }

    for coder_name, stats in multi_round_stats.items():
        if hasattr(stats, 'all_test_failures') and stats.all_test_failures:
            for failure in stats.all_test_failures:
                test_name = failure.get('test_name', 'unknown')
                category = _classify_test_name(test_name)
                all_test_names_by_category[category].add(test_name)

    tests_per_category_per_round = {}
    for category, test_names in all_test_names_by_category.items():
        tests_per_category_per_round[category] = len(test_names)

    if sum(tests_per_category_per_round.values()) == 0:
        # When all tests pass we cannot derive counts from failures; report 0 rather than guess.
        tests_per_category_per_round = {
            "Code Structure & Validation": 0,
            "Basic Functionality": 0,
            "Game Interaction & Logic": 0,
            "Robustness & Performance": 0
        }

    for category in all_test_names_by_category.keys():
        if category not in tests_per_category_per_round:
            tests_per_category_per_round[category] = 0

    for coder_name, stats in multi_round_stats.items():
        coder_category_stats[coder_name] = {
            "Code Structure & Validation": {"total": 0, "failed": 0, "passed": 0},
            "Basic Functionality": {"total": 0, "failed": 0, "passed": 0},
            "Game Interaction & Logic": {"total": 0, "failed": 0, "passed": 0},
            "Robustness & Performance": {"total": 0, "failed": 0, "passed": 0}
        }

        total_rounds = stats.total_rounds
        for category, tests_per_round in tests_per_category_per_round.items():
            total_tests_this_category = total_rounds * tests_per_round
            coder_category_stats[coder_name][category]["total"] = total_tests_this_category

        if hasattr(stats, 'all_test_failures') and stats.all_test_failures:
            # Dedupe per (round, test_name) so a test failing multiple revisions counts once.
            round_test_failures = {}
            for failure in stats.all_test_failures:
                test_name = failure.get('test_name', 'unknown')
                round_num = failure.get('round', 1)
                key = (round_num, test_name)
                if key not in round_test_failures:
                    round_test_failures[key] = failure

            for (round_num, test_name), failure in round_test_failures.items():
                category = _classify_test_name(test_name)
                coder_category_stats[coder_name][category]["failed"] += 1

        for category in coder_category_stats[coder_name].keys():
            failed_count = coder_category_stats[coder_name][category]["failed"]
            total_count = coder_category_stats[coder_name][category]["total"]
            passed_count = max(0, total_count - failed_count)
            coder_category_stats[coder_name][category]["passed"] = passed_count

    return coder_category_stats


def _calculate_overall_test_stats(coder_category_stats: Dict[str, Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, Any]]:
    """Calculate overall test statistics for each coder."""
    overall_stats = {}

    for coder_name, categories in coder_category_stats.items():
        total_tests = sum(cat_data["total"] for cat_data in categories.values())
        total_failed = sum(cat_data["failed"] for cat_data in categories.values())
        total_passed = sum(cat_data["passed"] for cat_data in categories.values())

        test_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0

        overall_stats[coder_name] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "test_pass_rate": test_pass_rate,
            "test_pass_fraction": f"({total_passed}/{total_tests})"
        }

    return overall_stats


def _calculate_repair_rates(multi_round_stats: Dict) -> Dict[str, Dict[str, Any]]:
    """Calculate repair rates for each coder based on test failures in first vs final version."""
    repair_rates = {}

    for coder_name, stats in multi_round_stats.items():
        repair_rates[coder_name] = {
            "repair_rate": 0.0,
            "repair_fraction": "(0/0)",
            "first_version_errors": 0,
            "final_version_errors": 0,
            "errors_fixed": 0
        }

        if hasattr(stats, 'all_test_failures') and stats.all_test_failures:
            failures_by_round = {}
            for failure in stats.all_test_failures:
                round_num = failure.get('round', 1)
                test_name = failure.get('test_name', 'unknown')
                revision_num = failure.get('revision_number')

                if round_num not in failures_by_round:
                    failures_by_round[round_num] = {}
                if test_name not in failures_by_round[round_num]:
                    failures_by_round[round_num][test_name] = []

                failures_by_round[round_num][test_name].append({
                    'revision': revision_num,
                    'failure': failure
                })

            total_first_errors = 0
            total_errors_fixed = 0

            for round_num, round_tests in failures_by_round.items():
                for test_name, test_failures in round_tests.items():
                    # revision_number == None marks first-attempt failures.
                    first_version_failed = any(f['revision'] is None for f in test_failures)

                    if first_version_failed:
                        total_first_errors += 1

                        # If the round eventually passed (passed_tests_count >= round_num)
                        # the first-attempt failure was repaired.
                        if hasattr(stats, 'passed_tests_count') and stats.passed_tests_count >= round_num:
                            total_errors_fixed += 1

            if total_first_errors > 0:
                repair_rate = total_errors_fixed / total_first_errors

                repair_rates[coder_name] = {
                    "repair_rate": repair_rate,
                    "repair_fraction": f"({total_errors_fixed}/{total_first_errors})",
                    "first_version_errors": total_first_errors,
                    "final_version_errors": total_first_errors - total_errors_fixed,
                    "errors_fixed": total_errors_fixed
                }

    return repair_rates


def _analyze_first_version_test_failures(multi_round_stats: Dict) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Analyze test failures for first version (initial submission) only."""
    coder_category_stats = {}

    all_test_names_by_category = {
        "Code Structure & Validation": set(),
        "Basic Functionality": set(),
        "Game Interaction & Logic": set(),
        "Robustness & Performance": set()
    }

    for coder_name, stats in multi_round_stats.items():
        if hasattr(stats, 'all_test_failures') and stats.all_test_failures:
            for failure in stats.all_test_failures:
                test_name = failure.get('test_name', 'unknown')
                category = _classify_test_name(test_name)
                all_test_names_by_category[category].add(test_name)

    tests_per_category_per_round = {}
    for category, test_names in all_test_names_by_category.items():
        tests_per_category_per_round[category] = len(test_names)

    if sum(tests_per_category_per_round.values()) == 0:
        # No failures recorded — try to recover counts from per-round totals before defaulting to 0.
        actual_tests_per_category = {}

        for coder_name, stats in multi_round_stats.items():
            if hasattr(stats, 'total_tests_per_round') and stats.total_tests_per_round:
                if hasattr(stats, 'round_details') and stats.round_details:
                    for round_detail in stats.round_details:
                        total_tests = stats.total_tests_per_round[0] if stats.total_tests_per_round else 0
                        if total_tests > 0:
                            # Fallback split based on observed tester layouts; only used when no
                            # failure data is available to drive a precise classification.
                            actual_tests_per_category = {
                                "Code Structure & Validation": max(1, int(total_tests * 0.4)),
                                "Basic Functionality": max(1, int(total_tests * 0.3)),
                                "Game Interaction & Logic": max(1, int(total_tests * 0.2)),
                                "Robustness & Performance": max(1, int(total_tests * 0.1))
                            }
                            current_total = sum(actual_tests_per_category.values())
                            if current_total < total_tests:
                                actual_tests_per_category["Code Structure & Validation"] += (total_tests - current_total)
                            break
                break

        if actual_tests_per_category:
            tests_per_category_per_round = actual_tests_per_category
        else:
            tests_per_category_per_round = {
                "Code Structure & Validation": 0,
                "Basic Functionality": 0,
                "Game Interaction & Logic": 0,
                "Robustness & Performance": 0
            }

    for category in all_test_names_by_category.keys():
        if category not in tests_per_category_per_round:
            tests_per_category_per_round[category] = 0

    for coder_name, stats in multi_round_stats.items():
        coder_category_stats[coder_name] = {
            "Code Structure & Validation": {"total": 0, "failed": 0, "passed": 0},
            "Basic Functionality": {"total": 0, "failed": 0, "passed": 0},
            "Game Interaction & Logic": {"total": 0, "failed": 0, "passed": 0},
            "Robustness & Performance": {"total": 0, "failed": 0, "passed": 0}
        }

        total_rounds = stats.total_rounds
        for category, tests_per_round in tests_per_category_per_round.items():
            total_tests_this_category = total_rounds * tests_per_round
            coder_category_stats[coder_name][category]["total"] = total_tests_this_category

        if hasattr(stats, 'all_test_failures') and stats.all_test_failures:
            round_test_failures = {}
            for failure in stats.all_test_failures:
                # First-version failures only: revision_number is None.
                if failure.get('revision_number') is None:
                    test_name = failure.get('test_name', 'unknown')
                    round_num = failure.get('round', 1)
                    key = (round_num, test_name)
                    if key not in round_test_failures:
                        round_test_failures[key] = failure

            for (round_num, test_name), failure in round_test_failures.items():
                category = _classify_test_name(test_name)
                coder_category_stats[coder_name][category]["failed"] += 1

        for category in coder_category_stats[coder_name].keys():
            failed_count = coder_category_stats[coder_name][category]["failed"]
            total_count = coder_category_stats[coder_name][category]["total"]
            passed_count = max(0, total_count - failed_count)
            coder_category_stats[coder_name][category]["passed"] = passed_count

    return coder_category_stats


def _calculate_average_rankings(multi_round_stats: Dict) -> Dict[str, float]:
    """
    Calculate average ranking for each coder across all rounds.
    Failed coders are ranked last in each round they fail.
    """
    average_rankings = {}

    all_coders = [(name, stats) for name, stats in multi_round_stats.items()]

    if not all_coders:
        return average_rankings

    num_rounds = max(len(stats.conservative_ratings) for _, stats in all_coders if stats.conservative_ratings)

    for coder_name, stats in all_coders:
        round_rankings = []

        for round_idx in range(num_rounds):
            if round_idx < len(stats.conservative_ratings):
                coder_rating = stats.conservative_ratings[round_idx]

                coder_passed = (round_idx < stats.passed_tests_count)

                if coder_passed:
                    round_ratings = []
                    for other_name, other_stats in all_coders:
                        if (round_idx < len(other_stats.conservative_ratings) and
                            round_idx < other_stats.passed_tests_count):
                            round_ratings.append((other_name, other_stats.conservative_ratings[round_idx]))

                    round_ratings.sort(key=lambda x: x[1], reverse=True)

                    for rank, (name, rating) in enumerate(round_ratings, 1):
                        if name == coder_name:
                            round_rankings.append(rank)
                            break
                else:
                    # Failed coder ranks last in this round.
                    total_coders_this_round = sum(1 for _, other_stats in all_coders
                                                if round_idx < len(other_stats.conservative_ratings))
                    round_rankings.append(total_coders_this_round)

        if round_rankings:
            average_rankings[coder_name] = sum(round_rankings) / len(round_rankings)
        else:
            average_rankings[coder_name] = len(all_coders)

    return average_rankings


def _convert_to_serializable(obj: Any) -> Any:
    """Convert dataclass objects to serializable dictionaries, excluding non-serializable objects."""
    import numpy as np

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'tolist') and hasattr(obj, 'dtype'):
        return obj.tolist()
    elif hasattr(obj, 'item') and hasattr(obj, 'dtype'):
        return obj.item()

    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)

            # `coder` and `agent_instance` reference live objects that are not JSON-serializable;
            # collapse them to their .name attribute.
            if field_name in ['coder', 'agent_instance']:
                if field_name == 'coder':
                    result[field_name] = getattr(field_value, 'name', str(field_value)) if field_value else None
                elif field_name == 'agent_instance':
                    result[field_name] = getattr(field_value, 'name', str(field_value)) if field_value else None
                continue

            try:
                result[field_name] = _convert_to_serializable(field_value)
            except (TypeError, AttributeError):
                result[field_name] = str(field_value) if field_value is not None else None

        return result
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            try:
                result[key] = _convert_to_serializable(value)
            except (TypeError, AttributeError):
                result[key] = str(value) if value is not None else None
        return result
    elif isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            try:
                result.append(_convert_to_serializable(item))
            except (TypeError, AttributeError):
                result.append(str(item) if item is not None else None)
        return result
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, AttributeError):
            return str(obj) if obj is not None else None


def _generate_individual_game_markdown(markdown_path: str, game_results: Dict, game_name: str) -> None:
    """Generate markdown report for a single game."""
    multi_round_stats = game_results['multi_round_stats']

    average_rankings = _calculate_average_rankings(multi_round_stats)

    # Sort by average ranking (lower is better), then by TrueSkill Rating as tiebreaker.
    game_coders = [(name, stats) for name, stats in multi_round_stats.items() if stats.conservative_ratings]
    game_ranking = sorted(game_coders, key=lambda x: (average_rankings.get(x[0], 999), -x[1].conservative_rating_avg))

    failed_coders = [(name, stats) for name, stats in multi_round_stats.items() if not stats.conservative_ratings]

    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(f"# {game_name.replace('_', ' ').title()} Tournament Report\n\n")
        f.write(f"**Generated Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Participants:** {len(multi_round_stats)}\n")
        f.write(f"- **Successful Participants:** {len(game_coders)}\n")
        f.write(f"- **Failed Participants:** {len(failed_coders)}\n\n")

        f.write("## TrueSkill Rankings\n\n")
        if game_coders:
            f.write("| Rank | Coder | TrueSkill Rating | Robustness Score | Average Ranking | Win Rate | Participate Rate |\n")
            f.write("|------|-------|------------------|------------------|----------------|----------|------------------|\n")

            for rank, (coder_name, stats) in enumerate(game_ranking, 1):
                total_wins = sum(stats.wins) if stats.wins else 0
                total_losses = sum(stats.losses) if stats.losses else 0
                total_draws = sum(stats.draws) if stats.draws else 0
                total_matches = total_wins + total_losses + total_draws

                win_rate_cell = _fmt_rate_cell(total_wins, total_matches)
                participate_cell = _fmt_rate_cell(stats.passed_tests_count, stats.total_rounds)

                avg_rank = average_rankings.get(coder_name, 0.0)

                f.write(f"| {rank} | **{coder_name}** | {stats.conservative_rating_avg:.1f} ± {stats.conservative_rating_std:.1f} | {stats.rating_robustness_score:.3f} | {avg_rank:.1f} | {win_rate_cell} | {participate_cell} |\n")
        else:
            f.write("*No participants successfully completed the tournament.*\n")

        f.write("\n## Pass@1 (First-Try Test Success Rate)\n\n")

        first_version_category_stats = _analyze_first_version_test_failures(multi_round_stats)
        first_version_overall_stats = _calculate_overall_test_stats(first_version_category_stats)
        repair_rates = _calculate_repair_rates(multi_round_stats)

        f.write("| Coder | Pass@1 | Avg Revisions | Failed Rounds | Total Timeouts | Repair Rate | Code Structure & Validation | Basic Functionality | Game Interaction & Logic | Robustness & Performance |\n")
        f.write("|-------|-------------------------------|---------------|---------------|----------------|-------------|-------------------|-------------------|----------------------|------------------------|\n")

        for coder_name, stats in multi_round_stats.items():
            failed_rounds = stats.total_rounds - stats.passed_tests_count

            # First-time test success rate is computed directly from per-round
            # totals, which capture every test the tester ran on the initial code
            # (including tests that always pass). Falling back to the failure-derived
            # estimate only if no per-round totals were recorded.
            total_first_version_tests = sum(stats.total_tests_per_round)
            total_first_version_passed = sum(stats.passed_tests_per_round)
            if total_first_version_tests > 0:
                first_version_overall_stats_for_coder = {
                    "test_pass_rate": total_first_version_passed / total_first_version_tests,
                    "test_pass_fraction": f"({total_first_version_passed}/{total_first_version_tests})",
                }
            else:
                first_version_overall_stats_for_coder = first_version_overall_stats.get(coder_name, {
                    "test_pass_rate": 0.0,
                    "test_pass_fraction": "(0/0)"
                })

            repair_stats = repair_rates.get(coder_name, {
                "repair_rate": 0.0,
                "repair_fraction": "(0/0)"
            })

            first_version_category_stats_for_coder = first_version_category_stats.get(coder_name, {})

            _empty_cat = {"total": 0, "failed": 0, "passed": 0}
            code_structure_stats = first_version_category_stats_for_coder.get("Code Structure & Validation", _empty_cat)
            basic_functionality_stats = first_version_category_stats_for_coder.get("Basic Functionality", _empty_cat)
            game_interaction_stats = first_version_category_stats_for_coder.get("Game Interaction & Logic", _empty_cat)
            robustness_performance_stats = first_version_category_stats_for_coder.get("Robustness & Performance", _empty_cat)

            code_structure_cell = _fmt_rate_cell(code_structure_stats["passed"], code_structure_stats["total"])
            basic_functionality_cell = _fmt_rate_cell(basic_functionality_stats["passed"], basic_functionality_stats["total"])
            game_interaction_cell = _fmt_rate_cell(game_interaction_stats["passed"], game_interaction_stats["total"])
            robustness_performance_cell = _fmt_rate_cell(robustness_performance_stats["passed"], robustness_performance_stats["total"])

            # repair_rates returns "(repaired/eligible)" formatted strings;
            # parse them back so we can route 0/0 through _fmt_rate_cell.
            _repair_fraction = repair_stats.get('repair_fraction', '(0/0)').strip('()')
            try:
                _repaired, _eligible = (int(p) for p in _repair_fraction.split('/'))
            except (ValueError, AttributeError):
                _repaired, _eligible = 0, 0
            repair_cell = _fmt_rate_cell(_repaired, _eligible)

            f.write(f"| {coder_name} | {first_version_overall_stats_for_coder['test_pass_rate']:.1%} {first_version_overall_stats_for_coder['test_pass_fraction']} | {stats.avg_revision_count:.1f} | {failed_rounds} | {stats.total_timeouts} | {repair_cell} | {code_structure_cell} | {basic_functionality_cell} | {game_interaction_cell} | {robustness_performance_cell} |\n")

        f.write("\n## In-Game Error Analysis\n\n")
        f.write("| Coder | Total Errors | Avg Errors/Round | Error Rate |\n")
        f.write("|-------|--------------|------------------|------------|\n")

        for coder_name, stats in multi_round_stats.items():
            denom = stats.total_matches_played
            error_rate_cell = _fmt_rate_cell(stats.total_in_game_errors, denom)

            f.write(f"| {coder_name} | {stats.total_in_game_errors} | {stats.avg_in_game_errors_per_round:.2f} | {error_rate_cell} |\n")

        if failed_coders:
            f.write("\n## Failed Participants\n\n")
            f.write("| Coder | Rounds Attempted | Passed Tests | Failure Rate | Avg Revisions |\n")
            f.write("|-------|------------------|--------------|--------------|---------------|\n")

            for coder_name, stats in failed_coders:
                failed_tests = stats.total_rounds - stats.passed_tests_count
                failure_rate = failed_tests / stats.total_rounds if stats.total_rounds > 0 else 0.0
                failure_rate_fraction = f"({failed_tests}/{stats.total_rounds})"

                f.write(f"| {coder_name} | {stats.total_rounds} | {stats.passed_tests_count} | {failure_rate:.1%} {failure_rate_fraction} | {stats.avg_revision_count:.1f} |\n")


def _generate_individual_game_json(json_path: str, game_results: Dict, game_name: str) -> None:
    """Generate JSON report for a single game with comprehensive data."""
    multi_round_stats = game_results['multi_round_stats']

    game_coders = [(name, stats) for name, stats in multi_round_stats.items() if stats.conservative_ratings]
    game_ranking = sorted(game_coders, key=lambda x: x[1].conservative_rating_avg, reverse=True)

    failed_coders = [(name, stats) for name, stats in multi_round_stats.items() if not stats.conservative_ratings]

    game_report = {
        'game_name': game_name,
        'timestamp': datetime.now().isoformat(),
        'experiment_folder': game_results['experiment_folder'],
        'total_participants': len(multi_round_stats),
        'successful_participants': len(game_coders),
        'failed_participants': len(failed_coders),
        'trueskill_rankings': [],
        'failed_coders': [],
        'test_results': {},
        'error_analysis': {},
        'summary_statistics': {}
    }

    for rank, (coder_name, stats) in enumerate(game_ranking, 1):
        game_report['trueskill_rankings'].append({
            'rank': rank,
            'coder_name': coder_name,
            'conservative_rating': stats.conservative_rating_avg,
            'rating_std_dev': stats.conservative_rating_std,
            'robustness_score': stats.rating_robustness_score,
            'win_rate': stats.win_rate_avg,
            'success_rate': stats.success_rate,
            'total_rounds': stats.total_rounds,
            'passed_tests': stats.passed_tests_count,
            'avg_revisions': stats.avg_revision_count,
            'avg_decision_time': stats.overall_avg_decision_time,
            'total_timeouts': stats.total_timeouts,
            'total_in_game_errors': stats.total_in_game_errors
        })

    for coder_name, stats in failed_coders:
        game_report['failed_coders'].append({
            'coder_name': coder_name,
            'total_rounds_attempted': stats.total_rounds,
            'passed_tests_count': stats.passed_tests_count,
            'failed_rounds': stats.total_rounds - stats.passed_tests_count,
            'failure_rate': (stats.total_rounds - stats.passed_tests_count) / stats.total_rounds if stats.total_rounds > 0 else 0.0,
            'avg_revision_count': stats.avg_revision_count,
            'test_failures': len(stats.all_test_failures) if hasattr(stats, 'all_test_failures') else 0
        })

    for coder_name, stats in multi_round_stats.items():
        game_report['test_results'][coder_name] = {
            'test_pass_rate': stats.success_rate,
            'total_rounds': stats.total_rounds,
            'passed_tests': stats.passed_tests_count,
            'failed_rounds': stats.total_rounds - stats.passed_tests_count,
            'avg_revision_count': stats.avg_revision_count,
            'revision_counts': stats.revision_counts,
            'all_test_failures': stats.all_test_failures[:10] if hasattr(stats, 'all_test_failures') else []
        }

    for coder_name, stats in multi_round_stats.items():
        game_report['error_analysis'][coder_name] = {
            'total_in_game_errors': stats.total_in_game_errors,
            'avg_errors_per_round': stats.avg_in_game_errors_per_round,
            'total_timeouts': stats.total_timeouts,
            'error_details': stats.all_in_game_errors[:10] if hasattr(stats, 'all_in_game_errors') else []
        }

    if game_coders:
        conservative_ratings = [stats.conservative_rating_avg for _, stats in game_coders]
        robustness_scores = [stats.rating_robustness_score for _, stats in game_coders]
        win_rates = [stats.win_rate_avg for _, stats in game_coders]
        success_rates = [stats.success_rate for _, stats in game_coders]

        game_report['summary_statistics'] = {
            'avg_conservative_rating': statistics.mean(conservative_ratings),
            'conservative_rating_range': [min(conservative_ratings), max(conservative_ratings)],
            'avg_robustness_score': statistics.mean(robustness_scores),
            'robustness_score_range': [min(robustness_scores), max(robustness_scores)],
            'avg_win_rate': statistics.mean(win_rates),
            'avg_success_rate': statistics.mean(success_rates),
            'total_test_failures': sum(len(stats.all_test_failures) if hasattr(stats, 'all_test_failures') else 0 for _, stats in multi_round_stats.items()),
            'total_in_game_errors': sum(stats.total_in_game_errors for _, stats in multi_round_stats.items()),
            'total_timeouts': sum(stats.total_timeouts for _, stats in multi_round_stats.items())
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(game_report, f, indent=2, ensure_ascii=False, default=_convert_to_serializable)


def _generate_individual_game_reports(game_folder: str, game_results: Dict, game_name: str) -> None:
    """Generate individual markdown and JSON reports for a single game in its own folder."""
    markdown_report_path = os.path.join(game_folder, f"{game_name}_report.md")
    _generate_individual_game_markdown(markdown_report_path, game_results, game_name)

    json_report_path = os.path.join(game_folder, f"{game_name}_report.json")
    _generate_individual_game_json(json_report_path, game_results, game_name)

    print(f"Individual reports for {game_name} saved to:")
    print(f"   Markdown: {markdown_report_path}")
    print(f"   JSON: {json_report_path}")


def _generate_game_specific_reports(main_folder: str, all_game_results: Dict) -> None:
    """Generate detailed JSON reports for each game."""
    game_reports_dir = os.path.join(main_folder, "game_reports")
    os.makedirs(game_reports_dir, exist_ok=True)

    for game_name, game_results in all_game_results.items():
        multi_round_stats = game_results['multi_round_stats']

        game_coders = [(name, stats) for name, stats in multi_round_stats.items() if stats.conservative_ratings]
        game_ranking = sorted(game_coders, key=lambda x: x[1].rating_robustness_score, reverse=True)

        failed_coders = [(name, stats) for name, stats in multi_round_stats.items() if not stats.conservative_ratings]

        game_report = {
            'game_name': game_name,
            'timestamp': datetime.now().isoformat(),
            'total_participants': len(multi_round_stats),
            'successful_participants': len(game_coders),
            'failed_participants': len(failed_coders),
            'performance_rankings': [],
            'failed_coders': [],
            'error_analysis': {},
            'summary_statistics': {}
        }

        for rank, (coder_name, stats) in enumerate(game_ranking, 1):
            game_report['performance_rankings'].append({
                'rank': rank,
                'coder_name': coder_name,
                'avg_conservative_rating': stats.conservative_rating_avg,
                'conservative_rating_std_dev': stats.conservative_rating_std,
                'robustness_score': stats.rating_robustness_score,
                'avg_win_rate': stats.win_rate_avg,
                'success_rate': stats.success_rate,
                'avg_revisions': stats.avg_revision_count,
                'avg_decision_time': stats.overall_avg_decision_time,
                'total_timeouts': stats.total_timeouts,
                'total_in_game_errors': stats.total_in_game_errors,
                'avg_errors_per_round': stats.avg_in_game_errors_per_round,
                'test_failures': len(stats.all_test_failures),
                'failed_rounds': stats.total_rounds - stats.passed_tests_count,
                'round_failure_rate': (stats.total_rounds - stats.passed_tests_count) / stats.total_rounds if stats.total_rounds > 0 else 0.0
            })

        for coder_name, stats in failed_coders:
            game_report['failed_coders'].append({
                'coder_name': coder_name,
                'total_rounds_attempted': stats.total_rounds,
                'passed_tests_count': stats.passed_tests_count,
                'failed_rounds': stats.total_rounds - stats.passed_tests_count,
                'round_failure_rate': (stats.total_rounds - stats.passed_tests_count) / stats.total_rounds if stats.total_rounds > 0 else 0.0,
                'avg_revision_count': stats.avg_revision_count if hasattr(stats, 'avg_revision_count') else 0,
                'test_failures': len(stats.all_test_failures) if hasattr(stats, 'all_test_failures') else 0
            })

        for coder_name, stats in multi_round_stats.items():
            game_report['error_analysis'][coder_name] = {
                'total_in_game_errors': stats.total_in_game_errors,
                'avg_errors_per_round': stats.avg_in_game_errors_per_round,
                'failed_rounds': stats.total_rounds - stats.passed_tests_count,
                'round_failure_rate': (stats.total_rounds - stats.passed_tests_count) / stats.total_rounds if stats.total_rounds > 0 else 0.0,
                'test_failures': len(stats.all_test_failures),
                'total_timeouts': stats.total_timeouts,
                'error_details': stats.all_in_game_errors[:10] if hasattr(stats, 'all_in_game_errors') else [],
                'test_failure_details': stats.all_test_failures[:10] if hasattr(stats, 'all_test_failures') else []
            }

        if game_coders:
            conservative_ratings = [stats.conservative_rating_avg for _, stats in game_coders]
            robustness_scores = [stats.rating_robustness_score for _, stats in game_coders]

            game_report['summary_statistics'] = {
                'avg_conservative_rating': statistics.mean(conservative_ratings),
                'conservative_rating_range': [min(conservative_ratings), max(conservative_ratings)],
                'avg_robustness_score': statistics.mean(robustness_scores),
                'robustness_score_range': [min(robustness_scores), max(robustness_scores)]
            }

        game_report_file = os.path.join(game_reports_dir, f"{game_name}_detailed_report.json")
        with open(game_report_file, 'w', encoding='utf-8') as f:
            json.dump(game_report, f, indent=2, ensure_ascii=False, default=_convert_to_serializable)

        print(f"{game_name} detailed report saved to: {game_report_file}")


def _generate_markdown_summary(main_folder: str, all_game_results: Dict, cross_game_stats: Dict) -> None:
    """Generate markdown summary table with key metrics."""
    md_file = os.path.join(main_folder, "tournament_summary.md")

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Multi-Round Multi-Game Tournament Results Summary\n\n")
        f.write(f"**Generated Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        games = list(all_game_results.keys())
        f.write(f"**Games Played:** {', '.join(games)}\n")
        f.write(f"**Number of Games:** {len(games)}\n\n")

        f.write("## Per-Game Performance Rankings\n\n")

        all_coders = set()
        for game_results in all_game_results.values():
            all_coders.update(game_results['multi_round_stats'].keys())

        if all_coders:
            header = "| Model |"
            separator = "|-------|"

            for game in games:
                header += f" {game.replace('_', ' ').title()} |"
                separator += "------|"

            header += " Average Revision times | Test Fail Rate | Running Timeout | Running Error |"
            separator += "----------------------|----------------|-----------------|---------------|"

            f.write(header + "\n")
            f.write(separator + "\n")

            # Sort coders by overall robustness score so the per-game table matches the detailed report ordering.
            coder_robustness_ranking = []
            for coder_name in all_coders:
                if coder_name in cross_game_stats and cross_game_stats[coder_name]['overall_conservative_ratings']:
                    robustness_score = cross_game_stats[coder_name]['overall_robustness_score']
                    coder_robustness_ranking.append((coder_name, robustness_score))
                else:
                    coder_robustness_ranking.append((coder_name, 0.0))

            coder_robustness_ranking.sort(key=lambda x: x[1], reverse=True)

            for coder_name, _ in coder_robustness_ranking:
                row = f"| {coder_name} |"

                for game in games:
                    if game in all_game_results and coder_name in all_game_results[game]['multi_round_stats']:
                        game_stats = all_game_results[game]['multi_round_stats'][coder_name]

                        if game_stats.conservative_ratings:
                            game_participants = [(name, stats) for name, stats in all_game_results[game]['multi_round_stats'].items()
                                               if stats.conservative_ratings]

                            total_rank = 0
                            rank_count = 0

                            for round_idx in range(len(game_stats.conservative_ratings)):
                                round_ratings = []
                                for participant_name, participant_stats in game_participants:
                                    if round_idx < len(participant_stats.conservative_ratings):
                                        round_ratings.append((participant_name, participant_stats.conservative_ratings[round_idx]))

                                round_ratings.sort(key=lambda x: x[1], reverse=True)
                                for rank, (name, _) in enumerate(round_ratings, 1):
                                    if name == coder_name:
                                        total_rank += rank
                                        rank_count += 1
                                        break

                            avg_rank = total_rank / rank_count if rank_count > 0 else 0
                            row += f" {avg_rank:.1f} |"
                        else:
                            row += " - |"
                    else:
                        row += " - |"

                if coder_name in cross_game_stats:
                    avg_revision_counts = []
                    test_fail_rates = []
                    timeout_rates = []
                    error_rates = []

                    for game_name, game_results in all_game_results.items():
                        if coder_name in game_results['multi_round_stats']:
                            game_stats = game_results['multi_round_stats'][coder_name]

                            if game_stats.revision_counts:
                                avg_revision_counts.append(game_stats.avg_revision_count)

                            if game_stats.total_rounds > 0:
                                # A round is counted as a test failure only if the coder couldn't pass after max revisions.
                                failed_rounds = game_stats.total_rounds - game_stats.passed_tests_count
                                fail_rate = failed_rounds / game_stats.total_rounds
                                test_fail_rates.append(fail_rate)

                            if game_stats.total_rounds > 0:
                                timeout_rate = game_stats.total_timeouts / game_stats.total_rounds
                                timeout_rates.append(timeout_rate)
                            else:
                                timeout_rates.append(0.0)

                            # In-game error rate uses matches actually played, not configured rounds.
                            denom = game_stats.total_matches_played
                            if denom > 0:
                                error_rates.append(game_stats.total_in_game_errors / denom)
                            else:
                                error_rates.append(0.0)

                    avg_revision = statistics.mean(avg_revision_counts) if avg_revision_counts else 0.0
                    avg_test_fail_rate = statistics.mean(test_fail_rates) if test_fail_rates else 0.0
                    avg_timeout_rate = statistics.mean(timeout_rates) if timeout_rates else 0.0
                    avg_error_rate = statistics.mean(error_rates) if error_rates else 0.0

                    row += f" {avg_revision:.1f} | {avg_test_fail_rate:.1%} | {avg_timeout_rate:.1%} | {avg_error_rate:.1%} |"
                else:
                    row += " - | - | - | - |"

                f.write(row + "\n")

        # Per-game detail JSONs are emitted alongside the markdown rather than embedded.
        _generate_game_specific_reports(main_folder, all_game_results)

        f.write("## Performance Analysis\n\n")

        valid_coders = [(name, stats) for name, stats in cross_game_stats.items() if stats['overall_conservative_ratings']]
        cross_game_ranking = sorted(valid_coders, key=lambda x: x[1]['overall_robustness_score'], reverse=True)

        if cross_game_ranking:
            f.write("### Efficiency Rankings (by revision count and decision speed)\n\n")

            efficiency_data = []
            for coder_name, stats in cross_game_stats.items():
                if stats['overall_conservative_ratings']:
                    avg_revision_counts = []
                    avg_decision_times = []

                    for game_name, game_results in all_game_results.items():
                        if coder_name in game_results['multi_round_stats']:
                            game_stats = game_results['multi_round_stats'][coder_name]
                            if game_stats.revision_counts:
                                avg_revision_counts.append(game_stats.avg_revision_count)
                            if game_stats.avg_decision_times:
                                avg_decision_times.append(game_stats.overall_avg_decision_time)

                    efficiency_data.append({
                        'name': coder_name,
                        'avg_revision_count': statistics.mean(avg_revision_counts) if avg_revision_counts else 0.0,
                        'avg_decision_time': statistics.mean(avg_decision_times) if avg_decision_times else 0.0,
                        'conservative_rating': stats['overall_conservative_rating_avg']
                    })

            efficiency_ranking = sorted(efficiency_data, key=lambda x: (x['avg_revision_count'], x['avg_decision_time']))

            f.write("| Rank | Coder | Avg Revisions | Avg Decision Time (s) | TrueSkill Rating |\n")
            f.write("|------|-------|---------------|----------------------|-----------------|\n")

            for rank, data in enumerate(efficiency_ranking, 1):
                f.write(f"| {rank} | **{data['name']}** | {data['avg_revision_count']:.1f} | {data['avg_decision_time']:.2e} | {data['conservative_rating']:.1f} |\n")

        try:
            f.write("\n")
            test_analyses = analyze_cross_game_tests(all_game_results, first_attempt_only=True)
            test_analysis_section = generate_test_analysis_section(test_analyses)
            if test_analysis_section:
                f.write(test_analysis_section)
                f.write("\n")
        except Exception as e:
            print(f" Warning: Could not generate test analysis section: {e}")

        f.write("\n---\n")
        f.write("*Generated by ProxyWar Multi-Round Multi-Game Tournament System*\n")

    print(f"Markdown summary generated: {md_file}")


def _save_cross_game_coder_analysis(coder_analysis_folder: str, all_game_results: Dict, cross_game_stats: Dict) -> None:
    """Save individual coder analysis files in the coder_analysis folder."""

    for coder_name, coder_stats in cross_game_stats.items():
        if coder_stats['overall_conservative_ratings']:
            coder_file = os.path.join(coder_analysis_folder, f"{coder_name}_analysis.json")

            coder_analysis = {
                'coder_name': coder_name,
                'overall_stats': {
                    'overall_robustness_score': coder_stats['overall_robustness_score'],
                    'cross_game_robustness_avg': coder_stats['cross_game_robustness_avg'],
                    'overall_conservative_rating_avg': coder_stats['overall_conservative_rating_avg'],
                    'overall_conservative_rating_std': coder_stats['overall_conservative_rating_std'],
                    'overall_success_rate_avg': coder_stats['overall_success_rate_avg'],
                    'games_participated': coder_stats['games_participated']
                },
                'game_specific_performance': {},
                'detailed_round_data': {}
            }

            for game_name, game_results in all_game_results.items():
                if coder_name in game_results['multi_round_stats']:
                    game_stats = game_results['multi_round_stats'][coder_name]

                    serializable_stats = _convert_to_serializable(game_stats)

                    coder_analysis['game_specific_performance'][game_name] = serializable_stats

                    if isinstance(serializable_stats, dict):
                        coder_analysis['detailed_round_data'][game_name] = {
                            'round_details': serializable_stats.get('round_details', []),
                            'all_in_game_errors': serializable_stats.get('all_in_game_errors', []),
                            'all_test_failures': serializable_stats.get('all_test_failures', [])
                        }
                    else:
                        coder_analysis['detailed_round_data'][game_name] = {
                            'round_details': [],
                            'all_in_game_errors': [],
                            'all_test_failures': []
                        }

            with open(coder_file, 'w', encoding='utf-8') as f:
                json.dump(coder_analysis, f, indent=2, ensure_ascii=False)

    print(f"Individual coder analysis files saved to: {coder_analysis_folder}")


def _save_cross_game_results(main_folder: str, all_game_results: Dict, cross_game_stats: Dict) -> str:
    """Save combined cross-game multi-round results."""
    serializable_game_results = _convert_to_serializable(all_game_results)
    serializable_cross_game_stats = _convert_to_serializable(cross_game_stats)

    combined_data = {
        'tournament_info': {
            'type': 'multi_round_multi_game',
            'games': list(all_game_results.keys()),
            'timestamp': datetime.now().isoformat()
        },
        'game_results': serializable_game_results,
        'cross_game_statistics': serializable_cross_game_stats,
        'cross_game_rankings': []
    }

    valid_coders = [(name, stats) for name, stats in cross_game_stats.items() if stats['overall_conservative_ratings']]
    cross_game_ranking = sorted(valid_coders, key=lambda x: x[1]['overall_robustness_score'], reverse=True)

    for rank, (coder_name, stats) in enumerate(cross_game_ranking, 1):
        combined_data['cross_game_rankings'].append({
            'rank': rank,
            'name': coder_name,
            'overall_robustness_score': stats['overall_robustness_score'],
            'cross_game_robustness_avg': stats['cross_game_robustness_avg'],
            'overall_conservative_rating_avg': stats['overall_conservative_rating_avg'],
            'overall_conservative_rating_std': stats['overall_conservative_rating_std'],
            'overall_success_rate_avg': stats['overall_success_rate_avg'],
            'games_participated': stats['games_participated']
        })

    detailed_results_dir = os.path.join(main_folder, "detailed_results")
    os.makedirs(detailed_results_dir, exist_ok=True)

    results_file = os.path.join(detailed_results_dir, "cross_game_multi_round_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"Cross-game combined results saved to: {results_file}")

    _generate_markdown_summary(main_folder, all_game_results, cross_game_stats)

    return results_file


def _log_cross_game_results(cross_game_stats: Dict) -> None:
    """Log final cross-game multi-round results."""
    print("\n" + "=" * 70)
    print("CROSS-GAME MULTI-ROUND TOURNAMENT FINAL RESULTS")
    print("=" * 70)

    valid_coders = [(name, stats) for name, stats in cross_game_stats.items() if stats['overall_conservative_ratings']]
    cross_game_ranking = sorted(valid_coders, key=lambda x: x[1]['overall_robustness_score'], reverse=True)

    print(f"\nCROSS-GAME TRUESKILL RANKINGS:")
    print("-" * 50)

    for rank, (coder_name, stats) in enumerate(cross_game_ranking, 1):
        print(f"#{rank} {coder_name}")
        print(f"   TrueSkill Rating (μ-3σ): {stats['overall_conservative_rating_avg']:.1f}")
        print(f"   Mean Skill (μ): {stats['overall_skill_estimate_avg']:.1f}")
        print(f"   Uncertainty (σ): {stats['overall_uncertainty_avg']:.1f}")
        print(f"   Games Participated: {stats['games_participated']}")
        print()

    failed_coders = [(name, stats) for name, stats in cross_game_stats.items() if not stats['overall_conservative_ratings']]
    if failed_coders:
        print("CODERS THAT FAILED ALL GAMES:")
        print("-" * 30)
        for coder_name, stats in failed_coders:
            print(f"   {coder_name} - Failed all games")
        print()

    print("=" * 70)
