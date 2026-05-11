"""
TrueSkill Rating System for ProxyWar evaluation framework.

This module provides TrueSkill rating system implementation for tournament
management and player ranking, with support for both multiplayer TrueSkill
ratings and single-player success rate tracking.
"""

from typing import Dict, Optional, List, Tuple, Any
import trueskill as ts


class TrueSkillSystem:
    """TrueSkill rating system implementation for competitive games."""
    
    def __init__(self, mu: float = 25.0, sigma: float = 8.333, beta: float = 4.166, 
                 tau: float = 0.083, draw_probability: float = 0.10):
        """
        Initialize TrueSkill system.
        
        Args:
            mu: Initial mean skill (default 25.0)
            sigma: Initial skill uncertainty (default 25/3)
            beta: Skill class width (default mu/6)
            tau: Dynamics factor (default sigma/100)
            draw_probability: Probability of draws (default 0.10)
        """
        # Configure TrueSkill environment
        self.environment = ts.TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability
        )
        ts.setup(env=self.environment)
        
        self.ratings: Dict[str, ts.Rating] = {}
        self.initial_mu = mu
        self.initial_sigma = sigma
    
    def add_player(self, player_name: str, rating: Optional[ts.Rating] = None):
        """Add a player to the rating system."""
        if rating is not None:
            self.ratings[player_name] = rating
        else:
            self.ratings[player_name] = ts.Rating(mu=self.initial_mu, sigma=self.initial_sigma)
    
    def get_rating(self, player_name: str) -> ts.Rating:
        """Get a player's current rating."""
        if player_name not in self.ratings:
            self.add_player(player_name)
        return self.ratings[player_name]
    
    def get_conservative_rating(self, player_name: str) -> float:
        """Get conservative rating estimate (mu - 3*sigma)."""
        rating = self.get_rating(player_name)
        return rating.mu - 3 * rating.sigma
    
    def get_skill_estimate(self, player_name: str) -> float:
        """Get skill estimate (mu)."""
        rating = self.get_rating(player_name)
        return rating.mu
    
    def get_uncertainty(self, player_name: str) -> float:
        """Get uncertainty estimate (sigma)."""
        rating = self.get_rating(player_name)
        return rating.sigma
    
    def update_ratings_two_player(self, player1: str, player2: str, 
                                  winner: Optional[str] = None, is_draw: bool = False):
        """
        Update ratings after a two-player match.
        
        Args:
            player1: Name of first player
            player2: Name of second player
            winner: Name of winner, or None for draw
            is_draw: Whether the match was a draw
        """
        rating1 = self.get_rating(player1)
        rating2 = self.get_rating(player2)
        
        if is_draw:
            # Draw case
            new_rating1, new_rating2 = ts.rate_1vs1(rating1, rating2, drawn=True)
        elif winner == player1:
            # Player 1 wins
            new_rating1, new_rating2 = ts.rate_1vs1(rating1, rating2, drawn=False)
        elif winner == player2:
            # Player 2 wins
            new_rating2, new_rating1 = ts.rate_1vs1(rating2, rating1, drawn=False)
        else:
            raise ValueError("Either winner must be specified or is_draw must be True")
        
        self.ratings[player1] = new_rating1
        self.ratings[player2] = new_rating2
    
    def update_ratings_multiplayer(self, ranking: List[List[str]]):
        """
        Update ratings after a multiplayer match.
        
        Args:
            ranking: List of player groups in order of performance (best to worst).
                    Players in the same group are considered tied.
                    Example: [['player1'], ['player2', 'player3'], ['player4']]
                    means player1 won, player2 and player3 tied for second, player4 lost.
        """
        # Convert to rating groups
        rating_groups = []
        for group in ranking:
            rating_group = [self.get_rating(player) for player in group]
            rating_groups.append(rating_group)
        
        # Calculate new ratings
        new_rating_groups = ts.rate(rating_groups)
        
        # Update ratings
        for i, group in enumerate(ranking):
            for j, player in enumerate(group):
                self.ratings[player] = new_rating_groups[i][j]
    
    def get_all_ratings(self) -> Dict[str, ts.Rating]:
        """Get all current ratings."""
        return self.ratings.copy()
    
    def get_all_conservative_ratings(self) -> Dict[str, float]:
        """Get all conservative ratings."""
        return {name: self.get_conservative_rating(name) for name in self.ratings}
    
    def get_all_skill_estimates(self) -> Dict[str, float]:
        """Get all skill estimates."""
        return {name: self.get_skill_estimate(name) for name in self.ratings}
    
    def reset_ratings(self):
        """Reset all ratings to initial values."""
        for player in self.ratings:
            self.ratings[player] = ts.Rating(mu=self.initial_mu, sigma=self.initial_sigma)
    
    def remove_player(self, player_name: str):
        """Remove a player from the rating system."""
        if player_name in self.ratings:
            del self.ratings[player_name]


class SuccessRateTracker:
    """
    Success rate tracking system for single-player games.
    
    This class tracks success rates and performance metrics for
    single-player challenges where ELO ratings don't apply.
    """
    
    def __init__(self):
        """Initialize success rate tracker."""
        self.player_stats: Dict[str, Dict] = {}
    
    def add_player(self, player_name: str):
        """Add a player to the success rate tracker."""
        if player_name not in self.player_stats:
            self.player_stats[player_name] = {
                'total_attempts': 0,
                'successes': 0,
                'failures': 0,
                'success_rate': 0.0,
                'challenge_results': [],  # List of (challenge_name, success, score, moves, efficiency)
                'challenge_stats': {}  # Per-challenge statistics
            }
    
    def record_result(self, player_name: str, challenge_name: str, success: bool, 
                     score: Optional[float] = None, moves: Optional[int] = None, 
                     efficiency: Optional[float] = None, 
                     additional_info: Optional[Dict] = None):
        """
        Record a single-player game result.
        
        Args:
            player_name: Name of the player
            challenge_name: Name of the challenge/variant
            success: Whether the challenge was completed successfully
            score: Numeric score (0.0-1.0 for binary success/failure)
            moves: Number of moves taken
            efficiency: Efficiency score (optimal_moves / actual_moves)
            additional_info: Additional challenge-specific information
        """
        self.add_player(player_name)
        
        stats = self.player_stats[player_name]
        
        # Update overall stats
        stats['total_attempts'] += 1
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        # Update success rate
        stats['success_rate'] = stats['successes'] / stats['total_attempts']
        
        # Record detailed result
        result_record = {
            'challenge_name': challenge_name,
            'success': success,
            'score': score if score is not None else (1.0 if success else 0.0),
            'moves': moves,
            'efficiency': efficiency,
            'additional_info': additional_info or {}
        }
        stats['challenge_results'].append(result_record)
        
        # Update per-challenge statistics
        if challenge_name not in stats['challenge_stats']:
            stats['challenge_stats'][challenge_name] = {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0,
                'best_efficiency': 0.0,
                'avg_moves': 0.0,
                'results': []
            }
        
        challenge_stats = stats['challenge_stats'][challenge_name]
        challenge_stats['attempts'] += 1
        if success:
            challenge_stats['successes'] += 1
        
        challenge_stats['success_rate'] = challenge_stats['successes'] / challenge_stats['attempts']
        challenge_stats['results'].append(result_record)
        
        # Update best efficiency and average moves
        if efficiency is not None and efficiency > challenge_stats['best_efficiency']:
            challenge_stats['best_efficiency'] = efficiency
        
        if moves is not None:
            all_moves = [r['moves'] for r in challenge_stats['results'] if r['moves'] is not None]
            challenge_stats['avg_moves'] = sum(all_moves) / len(all_moves) if all_moves else 0.0
    
    def get_success_rate(self, player_name: str) -> float:
        """Get overall success rate for a player."""
        if player_name not in self.player_stats:
            return 0.0
        return self.player_stats[player_name]['success_rate']
    
    def get_challenge_success_rate(self, player_name: str, challenge_name: str) -> float:
        """Get success rate for a specific challenge."""
        if player_name not in self.player_stats:
            return 0.0
        
        challenge_stats = self.player_stats[player_name]['challenge_stats'].get(challenge_name)
        if not challenge_stats:
            return 0.0
        
        return challenge_stats['success_rate']
    
    def get_player_stats(self, player_name: str) -> Dict:
        """Get comprehensive statistics for a player."""
        if player_name not in self.player_stats:
            return {}
        return self.player_stats[player_name].copy()
    
    def get_all_players(self) -> List[str]:
        """Get list of all tracked players."""
        return list(self.player_stats.keys())
    
    def get_leaderboard(self, sort_by: str = 'success_rate') -> List[Tuple[str, Dict]]:
        """
        Get leaderboard sorted by specified metric.
        
        Args:
            sort_by: Metric to sort by ('success_rate', 'total_attempts', 'successes')
        
        Returns:
            List of (player_name, stats) tuples sorted by the specified metric
        """
        if not self.player_stats:
            return []
        
        def sort_key(item):
            player_name, stats = item
            if sort_by == 'success_rate':
                return stats['success_rate']
            elif sort_by == 'total_attempts':
                return stats['total_attempts']
            elif sort_by == 'successes':
                return stats['successes']
            else:
                return 0
        
        return sorted(self.player_stats.items(), key=sort_key, reverse=True)
    
    def reset_player(self, player_name: str):
        """Reset statistics for a specific player."""
        if player_name in self.player_stats:
            del self.player_stats[player_name]
    
    def reset_all(self):
        """Reset all player statistics."""
        self.player_stats = {}


class HybridRatingSystem:
    """
    Hybrid rating system that supports both TrueSkill ratings for multiplayer
    games and success rates for single-player games.
    """
    
    def __init__(self, mu: float = 25.0, sigma: float = 8.333, beta: float = 4.166, 
                 tau: float = 0.083, draw_probability: float = 0.10):
        """Initialize hybrid rating system."""
        self.trueskill_system = TrueSkillSystem(mu, sigma, beta, tau, draw_probability)
        self.success_tracker = SuccessRateTracker()
    
    def add_player(self, player_name: str, initial_rating: Optional[ts.Rating] = None):
        """Add a player to both rating systems."""
        self.trueskill_system.add_player(player_name, initial_rating)
        self.success_tracker.add_player(player_name)
    
    def update_multiplayer_result(self, ranking: List[List[str]]):
        """Update TrueSkill ratings for a multiplayer game result."""
        self.trueskill_system.update_ratings_multiplayer(ranking)
    
    def update_two_player_result(self, player1: str, player2: str, 
                                winner: Optional[str] = None, is_draw: bool = False):
        """Update TrueSkill ratings for a two-player game result."""
        self.trueskill_system.update_ratings_two_player(player1, player2, winner, is_draw)
    
    def update_singleplayer_result(self, player_name: str, challenge_name: str, success: bool,
                                  score: Optional[float] = None, moves: Optional[int] = None, 
                                  efficiency: Optional[float] = None,
                                  additional_info: Optional[Dict] = None):
        """Update success rate for a single-player game result."""
        self.success_tracker.record_result(player_name, challenge_name, success, 
                                         score, moves, efficiency, additional_info)
    
    def get_trueskill_rating(self, player_name: str) -> ts.Rating:
        """Get TrueSkill rating for a player."""
        return self.trueskill_system.get_rating(player_name)
    
    def get_conservative_rating(self, player_name: str) -> float:
        """Get conservative TrueSkill rating for a player."""
        return self.trueskill_system.get_conservative_rating(player_name)
    
    def get_skill_estimate(self, player_name: str) -> float:
        """Get skill estimate for a player."""
        return self.trueskill_system.get_skill_estimate(player_name)
    
    def get_uncertainty(self, player_name: str) -> float:
        """Get uncertainty estimate for a player."""
        return self.trueskill_system.get_uncertainty(player_name)
    
    def get_success_rate(self, player_name: str) -> float:
        """Get overall success rate for a player."""
        return self.success_tracker.get_success_rate(player_name)
    
    def get_combined_score(self, player_name: str, trueskill_weight: float = 0.5, 
                          success_weight: float = 0.5) -> float:
        """
        Get a combined score based on both TrueSkill rating and success rate.
        
        Args:
            player_name: Name of the player
            trueskill_weight: Weight for TrueSkill component (0.0-1.0)
            success_weight: Weight for success rate component (0.0-1.0)
        
        Returns:
            Combined score normalized to 0.0-1.0 range
        """
        # Normalize TrueSkill conservative rating to 0-1 range (assuming 0-50 range)
        conservative_rating = self.get_conservative_rating(player_name)
        normalized_trueskill = max(0.0, min(1.0, conservative_rating / 50.0))
        
        # Success rate is already 0-1
        success_rate = self.get_success_rate(player_name)
        
        # Ensure weights sum to 1
        total_weight = trueskill_weight + success_weight
        if total_weight > 0:
            trueskill_weight /= total_weight
            success_weight /= total_weight
        
        return trueskill_weight * normalized_trueskill + success_weight * success_rate 