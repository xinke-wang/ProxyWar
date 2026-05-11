#!/usr/bin/env python3
"""
Proxy War - Tournament Runner

This script runs tournament experiments using configuration files, enabling
multi-coder round-robin tournaments with ELO ratings.
"""

import argparse
from ProxyWar.evaluations import run_tournament_from_config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run ProxyWar tournaments with multiple coders and ELO ratings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                              # Use default config (configs/minimal.py)
  python run.py --config configs/example.py  # Full multi-game tournament
  python run.py -c configs/custom.py         # Use short form
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/minimal.py",
        help="Path to configuration file (default: configs/minimal.py)"
    )
    
    parser.add_argument(
        "--vis", "--visualization",
        action="store_true",
        help="Save game visualizations (screenshots) after each match"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Running tournament with config: {args.config}")
        if args.vis:
            print("Game visualization enabled - screenshots will be saved")
        
        # Run tournament using configuration file
        results = run_tournament_from_config(args.config, save_visualizations=args.vis)
        
        print("Tournament completed successfully!")
        
        # Handle different result formats
        if 'main_experiment_folder' in results:
            # Multi-round tournament format
            print(f"Results saved to: {results['main_experiment_folder']}")
            
            if 'cross_game_stats' in results:
                # Multi-round multi-game tournament
                print(f"Completed {results['num_games']} games with {results['num_rounds_per_game']} rounds each")
            else:
                # Single-game multi-round tournament
                print(f"Completed {results['num_rounds']} rounds")
                print("\nMulti-Round Robustness Rankings:")
                
                multi_round_stats = results.get('multi_round_stats', {})
                # Sort by robustness score
                valid_coders = [(name, stats) for name, stats in multi_round_stats.items() if stats.conservative_ratings]
                robustness_ranking = sorted(valid_coders, key=lambda x: x[1].rating_robustness_score, reverse=True)
                
                for rank, (coder_name, stats) in enumerate(robustness_ranking, 1):
                    print(f"#{rank} {coder_name}")
                    print(f"   Robustness Score: {stats.rating_robustness_score:.3f}")
                    print(f"   TrueSkill: {stats.conservative_rating_avg:.1f} ± {stats.conservative_rating_std:.1f} (max: {stats.conservative_rating_max:.1f}, min: {stats.conservative_rating_min:.1f})")
                    print(f"   Success Rate: {stats.success_rate:.1%} ({stats.passed_tests_count}/{stats.total_rounds})")
                    print(f"   Avg Win Rate: {stats.win_rate_avg:.1%}")
        else:
            # Single-round tournament format (original)
            print(f"Results saved to: {results['experiment_folder']}")
            
            # Display final rankings
            print("\nFinal Rankings:")
            final_rankings = results.get('final_rankings', [])
            for ranking in final_rankings:
                if ranking['status'] == 'Failed Testing' or ranking['status'] == 'Failed All Tests':
                    print(f"#{ranking['rank']} {ranking['name']} - {ranking['status']}")
                else:
                    win_rate_pct = ranking['win_rate'] * 100
                    # Handle both single-game and multi-game result formats
                    trueskill_score = ranking.get('conservative_rating', ranking.get('avg_conservative_rating', 0))
                    wins = ranking.get('wins', ranking.get('total_wins', 0))
                    losses = ranking.get('losses', ranking.get('total_losses', 0))
                    draws = ranking.get('draws', ranking.get('total_draws', 0))

                    if 'avg_conservative_rating' in ranking:
                        # Multi-game format
                        games_played = ranking.get('games_participated', 0)
                        total_games = results.get('total_games', 1)
                        print(f"#{ranking['rank']} {ranking['name']} - Avg TrueSkill: {trueskill_score:.1f} "
                              f"({wins}-{losses}-{draws}, {win_rate_pct:.1f}% win rate, {games_played}/{total_games} games)")
                    else:
                        # Single-game format
                        print(f"#{ranking['rank']} {ranking['name']} - TrueSkill: {trueskill_score:.1f} "
                              f"({wins}-{losses}-{draws}, {win_rate_pct:.1f}% win rate)")
        
    except Exception as e:
        print(f"Tournament failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 