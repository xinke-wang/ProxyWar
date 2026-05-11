"""
Texas Hold'em Poker game implementation using PettingZoo.

This module implements a multi-player Texas Hold'em poker game that supports
3-6 players and integrates with the ProxyWar evaluation framework.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pettingzoo.classic import texas_holdem_v4

from .base import MultiPlayerGame
from ..agents.base import BaseAgent
from ..registry import GAME_REGISTRY


@GAME_REGISTRY.register('texas_holdem')
class TexasHoldemGame(MultiPlayerGame):
    """
    Texas Hold'em poker game implementation.
    
    Supports 2-6 players using PettingZoo's texas_holdem_v4 environment.
    Converts complex PettingZoo observations into simplified, human-readable format.
    """
    
    def __init__(self, movement_timeout: float = 45.0, num_players: int = 6):
        """
        Initialize Texas Hold'em game.
        
        Args:
            movement_timeout: Maximum time in seconds for each agent decision
            num_players: Number of players (2-6)
        """
        super().__init__("texas_holdem", movement_timeout, max_players=6, min_players=2)
        self.num_players = min(max(num_players, 2), 6)
        self.env = None
        self.agent_mapping = {}
        
        # Card representation helpers
        self.suits = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']  # T = 10
        self.action_names = ['Call', 'Raise', 'Fold', 'Check']
        
        # Initialize position management
        self.button_position = 0  # Button starts at player 0
        self.current_hand = 0
    
    def setup_match(self, agents: List[BaseAgent]) -> None:
        """Setup a Texas Hold'em match."""
        super().setup_match(agents)
        
        # Create environment with the correct number of players
        self.env = texas_holdem_v4.env(num_players=len(agents))
        self.env.reset()
        
        # Map PettingZoo agent names to our BaseAgent instances
        self.agent_mapping = {}
        for i, agent in enumerate(agents):
            pz_agent_name = f"player_{i}"
            self.agent_mapping[pz_agent_name] = agent
        
        # Update player count for this match
        self.num_players = len(agents)
    
    def get_game_rules(self) -> str:
        """Get Texas Hold'em game rules."""
        return """
Texas Hold'em Poker Rules:

OBJECTIVE:
Win chips by having the best 5-card hand or by making all other players fold.

GAME STRUCTURE:
- Each player receives 2 private hole cards
- 5 community cards are dealt in stages: flop (3), turn (1), river (1)
- Players make the best 5-card hand from their 2 hole cards + 5 community cards

BETTING ROUNDS:
1. Pre-flop: After receiving hole cards
2. Flop: After first 3 community cards
3. Turn: After 4th community card
4. River: After 5th community card

ACTIONS:
- Call: Match the current bet
- Raise: Increase the bet amount
- Fold: Discard hand and exit the round
- Check: Pass action (only when no bet to call)

HAND RANKINGS (high to low):
1. Royal Flush: A-K-Q-J-10 of same suit
2. Straight Flush: 5 consecutive cards of same suit
3. Four of a Kind: 4 cards of same rank
4. Full House: 3 of a kind + pair
5. Flush: 5 cards of same suit
6. Straight: 5 consecutive cards
7. Three of a Kind: 3 cards of same rank
8. Two Pair: 2 pairs of different ranks
9. One Pair: 2 cards of same rank
10. High Card: Highest card when no other hand is made

WINNING:
- Player with best hand wins the pot
- If all other players fold, remaining player wins
- Multiple players can split the pot if they have identical hands
"""
    
    def get_observation_format(self) -> Dict[str, Any]:
        """Get observation format for Texas Hold'em."""
        # Generate a sample observation
        sample_env = texas_holdem_v4.env(num_players=self.num_players)
        sample_env.reset()
        
        # Get first observation
        for agent in sample_env.agent_iter():
            obs, reward, term, trunc, info = sample_env.last()
            if not term and not trunc and obs is not None:
                sample_simplified = self._simplify_observation(obs, agent)
                break
        
        return {
            'description': """
Texas Hold'em Observation Format:

The observation is a dictionary with the following keys:
- 'hole_cards': List of 2 strings representing your private cards (e.g., ['AS', 'KH'])
- 'community_cards': List of 0-5 strings representing visible community cards
- 'current_bet': Integer representing the current bet amount to call
- 'pot_size': Integer representing the current pot size
- 'your_chips': Integer representing your remaining chips
- 'betting_round': String indicating current round ('pre_flop', 'flop', 'turn', 'river')
- 'position': String indicating your position ('button', 'small_blind', 'big_blind', 'early', 'middle', 'late')
- 'players_in_hand': Integer representing number of players still in the hand
- 'legal_actions': List of strings representing valid actions you can take

Card Format:
- Ranks: A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K (T = 10)
- Suits: S (Spades), H (Hearts), D (Diamonds), C (Clubs)
- Examples: AS = Ace of Spades, TH = Ten of Hearts, KC = King of Clubs

The action space consists of 4 possible actions:
- 0: Call (match the current bet)
- 1: Raise (increase the bet)
- 2: Fold (discard your hand)
- 3: Check (pass action when no bet to call)
""",
            'sample_observation': sample_simplified,
            'action_space_size': 4,
            'sample_action_mask': [True, True, True, False],  # Example: can't check when there's a bet
            'position_mapping': 'Position is automatically managed and provided in the observation',
            'action_meanings': {
                0: 'Call',
                1: 'Raise', 
                2: 'Fold',
                3: 'Check'
            }
        }
    
    def _card_index_to_string(self, card_index: int) -> str:
        """Convert card index to readable string."""
        suit_idx = card_index // 13
        rank_idx = card_index % 13
        return f"{self.ranks[rank_idx]}{self.suits[suit_idx]}"
    
    def _get_player_position(self, pz_agent_name: str) -> str:
        """
        Get the position of a player relative to the button.
        
        Args:
            pz_agent_name: PettingZoo agent name (e.g., "player_0")
            
        Returns:
            Position string: "button", "small_blind", "big_blind", "early", "middle", "late"
        """
        # Extract player number from agent name
        player_num = int(pz_agent_name.split('_')[1])
        
        # Calculate position relative to button
        position_from_button = (player_num - self.button_position) % self.num_players
        
        if self.num_players == 2:
            # Heads-up: button is small blind, other is big blind
            if position_from_button == 0:
                return "button"  # Also small blind
            else:
                return "big_blind"
        
        elif self.num_players == 3:
            # Three-handed
            if position_from_button == 0:
                return "button"
            elif position_from_button == 1:
                return "small_blind"
            else:
                return "big_blind"
        
        else:
            # 4+ players
            if position_from_button == 0:
                return "button"
            elif position_from_button == 1:
                return "small_blind"
            elif position_from_button == 2:
                return "big_blind"
            elif position_from_button == 3:
                return "early"
            elif position_from_button < self.num_players - 1:
                return "middle"
            else:
                return "late"
    
    def _simplify_observation(self, pz_observation: Dict[str, Any], current_player: str = None) -> Dict[str, Any]:
        """
        Convert PettingZoo observation to simplified format.
        
        Args:
            pz_observation: Raw PettingZoo observation
            current_player: Current player name (e.g., "player_0")
            
        Returns:
            Simplified observation dictionary
        """
        obs_vec = pz_observation['observation']
        action_mask = pz_observation['action_mask']
        
        # Extract card information (indices 0-51)
        cards = obs_vec[:52]
        non_zero_indices = np.where(cards != 0)[0]
        
        # Convert to readable card strings
        card_strings = [self._card_index_to_string(idx) for idx in non_zero_indices]
        
        # Split into hole cards and community cards
        # In Texas Hold'em, first 2 cards are hole cards, rest are community
        hole_cards = card_strings[:2] if len(card_strings) >= 2 else card_strings
        community_cards = card_strings[2:] if len(card_strings) > 2 else []
        
        # Extract betting information (indices 52-71)
        betting_info = obs_vec[52:72]
        
        # Determine current betting round based on community cards
        if len(community_cards) == 0:
            betting_round = 'pre_flop'
        elif len(community_cards) == 3:
            betting_round = 'flop'
        elif len(community_cards) == 4:
            betting_round = 'turn'
        elif len(community_cards) == 5:
            betting_round = 'river'
        else:
            betting_round = 'unknown'
        
        # Extract betting round information
        # Each betting round has 5 indices (0-4 representing chip amounts)
        current_bet = 0
        for round_start in [52, 57, 62, 67]:  # Start indices for each round
            round_info = betting_info[round_start-52:round_start-52+5]
            if np.any(round_info):
                current_bet = int(np.where(round_info)[0][-1])  # Last non-zero position
        
        # Get legal actions
        legal_actions = [self.action_names[i] for i in range(len(action_mask)) if action_mask[i]]
        
        # Get player position
        position = self._get_player_position(current_player) if current_player else 'unknown'
        
        # Estimate pot size from betting information
        pot_size = sum(betting_info) * 2  # Rough estimate
        
        # Count active players (players with cards)
        players_in_hand = len(non_zero_indices) // 2 if len(non_zero_indices) >= 2 else 1
        
        simplified_obs = {
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'current_bet': current_bet,
            'pot_size': pot_size,
            'your_chips': 100,  # Default chips (not available in PettingZoo obs)
            'betting_round': betting_round,
            'position': position,
            'players_in_hand': players_in_hand,
            'legal_actions': legal_actions,
            'action_mask': action_mask.tolist()
        }
        
        return simplified_obs
    
    def get_default_action(self, action_mask: Any) -> Any:
        """Get default action (fold) for timed out agents."""
        # Always fold when timing out or erroring
        if hasattr(action_mask, '__len__') and len(action_mask) > 2:
            if action_mask[2]:  # Fold is available
                return 2
        return 2  # Fold
    
    def run_match(self) -> Dict[str, Any]:
        """Run a complete Texas Hold'em match."""
        if not self.current_agents:
            raise RuntimeError("No agents setup for match")
        
        move_history = []
        move_history_with_timing = []
        timeout_info = []
        final_rewards = {}  # Initialize rewards collection
        
        start_time = time.time()
        step_count = 0
        
        # Run the game
        for pz_agent_name in self.env.agent_iter():
            step_count += 1
            observation, reward, termination, truncation, info = self.env.last()
            
            # Get our agent corresponding to this PettingZoo agent
            agent = self.agent_mapping.get(pz_agent_name)
            if agent is None:
                print(f"Warning: No agent found for {pz_agent_name}")
                self.env.step(None)
                continue
            
            # Collect rewards for each agent (including 0 rewards)
            if agent.name not in final_rewards:
                final_rewards[agent.name] = 0.0
            final_rewards[agent.name] += float(reward) if reward is not None else 0.0
            
            # Check if game is over
            if termination or truncation:
                self.env.step(None)
                continue
            
            # Simplify observation with current player info
            simplified_obs = self._simplify_observation(observation, pz_agent_name)
            action_mask = simplified_obs['action_mask']  # Use the converted list instead of numpy array
            
            # Get agent action with timeout handling
            action, decision_time, timeout_result = self.handle_agent_move_with_timeout(
                agent, simplified_obs, action_mask, move_history, move_history_with_timing, step_count
            )
            
            # Record move
            move_data = {
                'step': step_count,
                'agent': agent.name,
                'pz_agent': pz_agent_name,
                'observation': simplified_obs,
                'action': action,
                'action_name': self.action_names[action] if action is not None else 'None',
                'decision_time': decision_time
            }
            
            move_history.append(move_data)
            move_history_with_timing.append({
                'agent': agent.name,
                'action': action,
                'decision_time': decision_time
            })
            
            if timeout_result:
                timeout_info.append(timeout_result)
            
            # Execute action in environment
            self.env.step(action)
            
            # Safety limit
            if step_count > 1000:
                print("Warning: Game exceeded 1000 steps, ending early")
                break
        
        # Final rewards were collected during the game loop above
        
        # Calculate final rankings based on rewards
        sorted_agents = sorted(final_rewards.items(), key=lambda x: x[1], reverse=True)
        final_rankings = [agent_name for agent_name, _ in sorted_agents]
        
        # Update button position for next hand
        self.current_hand += 1
        self.button_position = (self.button_position + 1) % self.num_players
        
        # Create standardized result
        match_result = {
            'results': final_rewards,
            'moves': len(move_history),
            'match_history': move_history,
            'move_history_with_timing': move_history_with_timing,
            'final_rankings': final_rankings,
            'rewards': final_rewards,
            'timeout_info': timeout_info,
            'total_time': time.time() - start_time,
            'game_type': 'multi_player'
        }
        
        return match_result
    
    def save_visualization(self, save_path: str) -> bool:
        """Save game visualization (not implemented for Texas Hold'em)."""
        # Texas Hold'em visualization is complex and not implemented yet
        return False
    
    def reset(self) -> None:
        """Reset game state."""
        super().reset()
        self.env = None
        self.agent_mapping = {}
        self.current_match_results = {}
        # Note: Don't reset button_position and current_hand here to maintain continuity between hands