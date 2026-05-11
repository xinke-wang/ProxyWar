"""Tournament orchestration stage.

Top-level entry points that turn a config file into one or more tournament
runs. Each entry point composes per-game `TournamentManager` runs and the
report-stage helpers; the manager itself is responsible for code generation,
testing, and match execution within a single game.
"""

import os
import statistics
from datetime import datetime
from typing import Any, Dict, List, Tuple

from ..tournament_manager import TournamentManager
from ...games.base import BaseGame
from ...prompts.base import BasePrompt
from .report_stage import (
    _generate_individual_game_reports,
    _log_cross_game_results,
    _save_cross_game_coder_analysis,
    _save_cross_game_results,
)


def run_tournament_from_config(config_path: str = "configs/minimal.py", save_visualizations: bool = False) -> Dict[str, Any]:
    """
    Run a tournament using configuration file. Supports multiple games and multiple rounds.

    Args:
        config_path: Path to the configuration file
        save_visualizations: Whether to save game state visualizations

    Returns:
        Tournament results dictionary
    """
    from ..config_loader import ConfigLoader, create_components_from_config

    config_loader = ConfigLoader(config_path)
    config_loader.load_config()

    games, coders_info, prompt_generator = create_components_from_config(config_loader)

    api_config = config_loader.get_api_config()
    require_api_key = api_config.require_key if hasattr(api_config, 'require_key') else True

    # pjtools' configurator returns None for unset attributes, so the
    # getattr default is never used — coerce None to the intended default.
    tournament_rounds = getattr(config_loader.config, 'tournament_rounds', None) or 1
    parallel_execution = getattr(config_loader.config, 'parallel_execution', None)
    if parallel_execution is None:
        parallel_execution = True

    if tournament_rounds > 1:
        return run_multi_round_multi_game_tournament(games, coders_info, prompt_generator, tournament_rounds, require_api_key, save_visualizations, parallel_execution)
    else:
        return run_multi_game_tournament(games, coders_info, prompt_generator, require_api_key, save_visualizations, parallel_execution)


def _run_single_game_tournament(game_info: Tuple[BaseGame, str, str, List[Dict], BasePrompt, int, bool, bool]) -> Tuple[str, Dict[str, Any]]:
    """Helper function to run a single game multi-round tournament."""
    game, game_name, game_folder, coders_info, prompt_generator, num_rounds, require_api_key, save_visualizations = game_info

    print(f"Starting {game.game_name} tournament...")

    tournament_manager = TournamentManager(game, coders_info, prompt_generator)

    # Direct the manager at the pre-created per-game folder so its rounds nest under it.
    tournament_manager.experiment_folder = game_folder

    game_results = tournament_manager.run_multi_round_tournament(
        num_rounds=num_rounds,
        require_api_key=require_api_key,
        max_revisions=3,
        save_visualizations=save_visualizations
    )

    game_results['experiment_folder'] = game_folder

    _generate_individual_game_reports(game_folder, game_results, game_name)

    print(f"{game.game_name} tournament completed")

    return game_name, game_results


def _run_single_game_single_round_tournament(game_info: Tuple[BaseGame, str, List[Dict], BasePrompt, bool, bool]) -> Tuple[str, Dict[str, Any]]:
    """Helper function to run a single game single-round tournament."""
    game, game_name, coders_info, prompt_generator, require_api_key, save_visualizations = game_info

    print(f"Starting {game.game_name} tournament...")

    tournament_manager = TournamentManager(game, coders_info, prompt_generator)

    game_results = tournament_manager.run_full_tournament(
        require_api_key=require_api_key,
        max_revisions=3,
        save_visualizations=save_visualizations
    )

    _generate_individual_game_reports(game_results['experiment_folder'], game_results, game_name)

    print(f"{game.game_name} tournament completed")

    return game_name, game_results


def run_multi_round_multi_game_tournament(games: List[BaseGame], coders_info: List[Dict], prompt_generator: BasePrompt, num_rounds: int = 5, require_api_key: bool = True, save_visualizations: bool = False, parallel: bool = False) -> Dict[str, Any]:
    """
    Run multi-round tournaments for multiple games and combine results with robustness analysis.
    """
    execution_mode = "sequential (games run one by one with parallel coder generation)"
    print(f"Running multi-round multi-game tournament ({execution_mode}):")
    print(f"   Games: {len(games)} ({', '.join([game.game_name for game in games])})")
    print(f"   Rounds per game: {num_rounds}")
    print(f"   Coders: {len(coders_info)} ({', '.join([info['name'] for info in coders_info])})")
    if save_visualizations:
        print("   Game visualizations will be saved")
    print(f"   Parallel coder generation within each game")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_folder = f"experiments/ProxyWar_Experiment_{timestamp}"
    os.makedirs(main_folder, exist_ok=True)

    game_history_folder = os.path.join(main_folder, "game_history")
    coder_analysis_folder = os.path.join(main_folder, "coder_analysis")
    os.makedirs(game_history_folder, exist_ok=True)
    os.makedirs(coder_analysis_folder, exist_ok=True)

    print(f"Experiment folder created: {main_folder}")
    print(f"   Game history: {game_history_folder}")
    print(f"   Coder analysis: {coder_analysis_folder}")
    print()

    all_game_results = {}

    game_infos = []
    for game in games:
        game_name = game.game_name.lower().replace(" ", "_")
        game_specific_folder = os.path.join(game_history_folder, game_name)
        os.makedirs(game_specific_folder, exist_ok=True)

        game_infos.append((game, game_name, game_specific_folder, coders_info, prompt_generator, num_rounds, require_api_key, save_visualizations))

    # Games run sequentially; per-game coder generation is parallelized inside TournamentManager.
    for i, game_info in enumerate(game_infos):
        game, game_name, game_specific_folder, coders_info, prompt_generator, num_rounds, require_api_key, save_visualizations = game_info
        print(f"Game {i+1}/{len(games)}: {game.game_name}")
        print("=" * 60)

        try:
            game_name, game_results = _run_single_game_tournament(game_info)
            all_game_results[game_name] = game_results
        except Exception as exc:
            print(f"{game.game_name} generated an exception: {exc}")
            import traceback
            traceback.print_exc()
            continue

        print()

    print("Calculating cross-game robustness statistics...")
    cross_game_stats = _calculate_cross_game_robustness(all_game_results, coders_info)

    _save_cross_game_coder_analysis(coder_analysis_folder, all_game_results, cross_game_stats)

    combined_file = _save_cross_game_results(main_folder, all_game_results, cross_game_stats)

    _log_cross_game_results(cross_game_stats)

    return {
        'main_experiment_folder': main_folder,
        'game_history_folder': game_history_folder,
        'coder_analysis_folder': coder_analysis_folder,
        'num_games': len(games),
        'num_rounds_per_game': num_rounds,
        'all_game_results': all_game_results,
        'cross_game_stats': cross_game_stats,
        'combined_results_file': combined_file
    }


def _calculate_cross_game_robustness(all_game_results: Dict[str, Dict], coders_info: List[Dict]) -> Dict[str, Dict]:
    """Calculate robustness statistics across multiple games and rounds."""
    cross_game_stats = {}

    for coder_info in coders_info:
        coder_name = coder_info['name']
        cross_game_stats[coder_name] = {
            'name': coder_name,
            'game_stats': {},
            'overall_robustness_scores': [],
            'overall_conservative_ratings': [],
            'overall_skill_estimates': [],
            'overall_uncertainties': [],
            'overall_success_rates': [],
            'games_participated': 0
        }

    for game_name, game_results in all_game_results.items():
        multi_round_stats = game_results['multi_round_stats']

        for coder_name, stats in multi_round_stats.items():
            coder_cross_stats = cross_game_stats[coder_name]

            coder_cross_stats['game_stats'][game_name] = {
                'robustness_score': stats.rating_robustness_score,
                'conservative_rating_avg': stats.conservative_rating_avg,
                'skill_estimate_avg': stats.skill_estimate_avg,
                'avg_uncertainty': stats.avg_uncertainty,
                'success_rate': stats.success_rate,
                'conservative_rating_std': stats.conservative_rating_std
            }

            if stats.conservative_ratings:
                coder_cross_stats['overall_robustness_scores'].append(stats.rating_robustness_score)
                coder_cross_stats['overall_conservative_ratings'].extend(stats.conservative_ratings)
                coder_cross_stats['overall_skill_estimates'].extend(stats.skill_estimates)
                coder_cross_stats['overall_uncertainties'].extend(stats.uncertainties)
                coder_cross_stats['overall_success_rates'].append(stats.success_rate)
                coder_cross_stats['games_participated'] += 1

    for coder_name, coder_stats in cross_game_stats.items():
        if coder_stats['overall_conservative_ratings']:
            coder_stats['overall_conservative_rating_avg'] = statistics.mean(coder_stats['overall_conservative_ratings'])
            coder_stats['overall_conservative_rating_std'] = statistics.stdev(coder_stats['overall_conservative_ratings']) if len(coder_stats['overall_conservative_ratings']) > 1 else 0.0

            coder_stats['overall_skill_estimate_avg'] = statistics.mean(coder_stats['overall_skill_estimates']) if coder_stats['overall_skill_estimates'] else 25.0
            coder_stats['overall_uncertainty_avg'] = statistics.mean(coder_stats['overall_uncertainties']) if coder_stats['overall_uncertainties'] else 8.333

            coder_stats['overall_robustness_score'] = 1.0 / (1.0 + coder_stats['overall_conservative_rating_std'])
            coder_stats['overall_success_rate_avg'] = statistics.mean(coder_stats['overall_success_rates'])
            coder_stats['cross_game_robustness_avg'] = statistics.mean(coder_stats['overall_robustness_scores'])
        else:
            coder_stats['overall_conservative_rating_avg'] = 0.0
            coder_stats['overall_conservative_rating_std'] = 0.0
            coder_stats['overall_skill_estimate_avg'] = 25.0
            coder_stats['overall_uncertainty_avg'] = 8.333
            coder_stats['overall_robustness_score'] = 0.0
            coder_stats['overall_success_rate_avg'] = 0.0
            coder_stats['cross_game_robustness_avg'] = 0.0

    return cross_game_stats


def run_multi_game_tournament(games: List[BaseGame], coders_info: List[Dict], prompt_generator: BasePrompt, require_api_key: bool = True, save_visualizations: bool = False, parallel: bool = False) -> Dict[str, Any]:
    """Run tournaments for multiple games and combine results."""
    execution_mode = "sequential (games run one by one with parallel coder generation)"
    print(f"Running multi-game tournament ({execution_mode}) with {len(games)} games:")
    for game in games:
        print(f"  - {game.game_name}")
    if save_visualizations:
        print("Game visualizations will be saved")
    print(f"Parallel coder generation within each game")
    print()

    all_results = {}
    combined_final_rankings = []
    total_experiment_folder = None

    game_infos = []
    for game in games:
        game_name = game.game_name.lower().replace(" ", "_")
        game_infos.append((game, game_name, coders_info, prompt_generator, require_api_key, save_visualizations))

    # Games run sequentially; coder generation is parallelized inside each game's TournamentManager.
    for i, game_info in enumerate(game_infos):
        game, game_name, coders_info, prompt_generator, require_api_key, save_visualizations = game_info
        print(f"Running tournament {i+1}/{len(games)}: {game.game_name}")
        print("=" * 60)

        try:
            game_name, game_results = _run_single_game_single_round_tournament(game_info)
            all_results[game_name] = game_results

            if total_experiment_folder is None:
                total_experiment_folder = game_results['experiment_folder']

        except Exception as exc:
            print(f"{game.game_name} generated an exception: {exc}")
            import traceback
            traceback.print_exc()
            continue

        print()

    print("Combining results from all games...")

    coder_combined_stats = {}
    for coder_info in coders_info:
        coder_name = coder_info['name']
        total_conservative_rating = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0
        games_participated = 0

        for game_name, game_results in all_results.items():
            for ranking in game_results['final_rankings']:
                if ranking['name'] == coder_name:
                    if ranking['status'] != 'Failed Testing':
                        total_conservative_rating += ranking['conservative_rating']
                        total_wins += ranking['wins']
                        total_losses += ranking['losses']
                        total_draws += ranking['draws']
                        games_participated += 1
                    break

        avg_conservative_rating = total_conservative_rating / games_participated if games_participated > 0 else 0
        total_games = total_wins + total_losses + total_draws
        win_rate = total_wins / total_games if total_games > 0 else 0.0

        coder_combined_stats[coder_name] = {
            'name': coder_name,
            'avg_conservative_rating': avg_conservative_rating,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'total_draws': total_draws,
            'total_games': total_games,
            'win_rate': win_rate,
            'games_participated': games_participated,
            'status': 'Tournament Participant' if games_participated > 0 else 'Failed All Tests'
        }

    combined_rankings = list(coder_combined_stats.values())
    combined_rankings.sort(key=lambda x: x['avg_conservative_rating'], reverse=True)

    for i, ranking in enumerate(combined_rankings, 1):
        ranking['rank'] = i

    combined_results = {
        'experiment_folder': total_experiment_folder,
        'tournament_type': 'multi_game',
        'games_played': [game.game_name for game in games],
        'total_games': len(games),
        'final_rankings': combined_rankings,
        'individual_game_results': all_results
    }

    print("COMBINED FINAL RANKINGS (across all games):")
    print("=" * 60)
    for ranking in combined_rankings:
        if ranking['status'] == 'Failed All Tests':
            print(f"#{ranking['rank']} {ranking['name']} - Failed All Tests")
        else:
            win_rate_pct = ranking['win_rate'] * 100
            print(f"#{ranking['rank']} {ranking['name']} - Avg TrueSkill: {ranking['avg_conservative_rating']:.1f} "
                  f"({ranking['total_wins']}-{ranking['total_losses']}-{ranking['total_draws']}, "
                  f"{win_rate_pct:.1f}% win rate, {ranking['games_participated']}/{len(games)} games)")

    return combined_results
