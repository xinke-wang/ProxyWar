"""
Base game classes for ProxyWar framework.

This module defines the abstract game interfaces and concrete implementations
for different game types: single-player, two-player, and multi-player games.
"""

import math
import time
import signal
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from ..agents.base import BaseAgent


class TimeoutError(Exception):
    """Exception raised when agent decision times out."""
    pass


class TimeoutHandler:
    """Context manager for handling agent decision timeouts."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
        self._use_signal = (
            threading.current_thread() is threading.main_thread()
            and hasattr(signal, 'SIGALRM')
        )

    def timeout_handler(self, signum, frame):
        self.timed_out = True
        raise TimeoutError(f"Agent decision timed out after {self.timeout_seconds} seconds")

    def __enter__(self):
        if self._use_signal:
            # Use signal-based timeout in main thread (POSIX only)
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(max(1, math.ceil(self.timeout_seconds)))
        # For non-main threads or Windows, timeout is handled in call_agent_with_timeout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._use_signal:
            # Cancel the timeout
            signal.alarm(0)
        return False


class BaseGame(ABC):
    """
    Ultra-abstract base class for all games in ProxyWar framework.
    
    This class defines only the most essential interfaces that all games
    must implement, regardless of player count or game type.
    """
    
    def __init__(self, game_name: str, movement_timeout: float = 45.0):
        """
        Initialize the base game.
        
        Args:
            game_name: Name identifier for this game
            movement_timeout: Maximum time in seconds for each agent decision
        """
        self.game_name = game_name
        self.movement_timeout = movement_timeout
        self.current_match_history = []
    
    @abstractmethod
    def get_game_rules(self) -> str:
        """
        Get a string description of the game rules.
        
        Returns:
            Human-readable description of game rules and objectives
        """
        pass
    
    @abstractmethod
    def get_observation_format(self) -> Dict[str, Any]:
        """
        Get detailed information about the observation format for this game.
        
        Returns:
            Dictionary containing:
            - 'description': Human-readable description of observation format
            - 'sample_observation': Example observation data
            - 'action_space_size': Number of possible actions
            - 'sample_action_mask': Example action mask
        """
        pass
    
    @abstractmethod
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete match.
        
        Returns:
            Dictionary containing match results with standardized keys
        """
        pass
    
    def save_visualization(self, save_path: str) -> bool:
        """
        Save a visualization of the current/last game state.
        
        Args:
            save_path: Path where to save the visualization
            
        Returns:
            True if visualization was saved successfully, False otherwise
        """
        # Default implementation - games can override this
        return False
    
    def call_agent_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any) -> Tuple[Any, float, bool]:
        """
        Call agent's select_action with timeout detection.
        
        Args:
            agent: The agent to call
            observation: The observation to pass to the agent
            action_mask: The action mask to pass to the agent
            
        Returns:
            Tuple of (action, decision_time, timed_out)
        """
        start_time = time.time()
        timed_out = False
        action = None
        
        # Check if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            # Use signal-based timeout in main thread
            try:
                with TimeoutHandler(self.movement_timeout):
                    action = agent.select_action(observation, action_mask)
            except TimeoutError:
                timed_out = True
                print(f"Agent {agent.name} timed out after {self.movement_timeout} seconds!")
            except Exception as e:
                print(f"Agent {agent.name} encountered error: {e}")
                timed_out = False  # This is an error, not a timeout
        else:
            # Use threading-based timeout in worker threads
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = agent.select_action(observation, action_mask)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.movement_timeout)
            
            if thread.is_alive():
                # Thread is still running, timeout occurred
                timed_out = True
                print(f"Agent {agent.name} timed out after {self.movement_timeout} seconds!")
            elif exception[0]:
                print(f"Agent {agent.name} encountered error: {exception[0]}")
                timed_out = False  # This is an error, not a timeout
            else:
                action = result[0]
        
        decision_time = time.time() - start_time
        return action, decision_time, timed_out
    
    def reset(self) -> None:
        """Reset game state for a new match."""
        self.current_match_history = []
    
    def __str__(self) -> str:
        """String representation of the game."""
        return f"{self.__class__.__name__}(name='{self.game_name}')"


class CompetitiveGame(BaseGame):
    """
    Abstract base class for competitive games between two agents.
    
    Handles common logic for agent management, setup, and result formatting
    for both direct head-to-head matches and robin-round challenges.
    This eliminates code duplication between TwoPlayerGame and SinglePlayerGame.
    """
    
    def __init__(self, game_name: str, movement_timeout: float = 45.0):
        """Initialize competitive game."""
        super().__init__(game_name, movement_timeout)
        self.agent1 = None
        self.agent2 = None
        self.agents = {}
    
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """
        Setup a match between two agents.
        
        Args:
            agent1: First agent
            agent2: Second agent
        """
        self.reset()
        self.agent1 = agent1
        self.agent2 = agent2
        self.agents = {agent1.name: agent1, agent2.name: agent2}
        # Subclasses should call super() then add their specific setup logic
    
    def _get_other_agent(self, current_agent: BaseAgent) -> BaseAgent:
        """Get the other agent in this match."""
        if self.agent1 is None or self.agent2 is None:
            raise RuntimeError("Match not properly setup - agents are None")
        
        if current_agent == self.agent1:
            return self.agent2
        elif current_agent == self.agent2:
            return self.agent1
        else:
            raise RuntimeError(f"Agent {current_agent.name} is not part of this match")
    
    @abstractmethod
    def handle_agent_move_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any, 
                                       move_history: List, move_history_with_timing: List, 
                                       game_step: int) -> Tuple[Any, float, Optional[Dict]]:
        """
        Handle agent move with timeout detection.
        
        Implementation varies by game type (head-to-head vs robin-round).
        
        Returns:
            Tuple of (action, decision_time, timeout_result)
        """
        pass


class TwoPlayerGame(CompetitiveGame):
    """
    Abstract base class for two-player competitive games.
    
    Handles direct head-to-head matches between two agents where
    they interact with each other (e.g., TicTacToe, ConnectFour).
    """
    
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """Setup a two-player head-to-head match."""
        super().setup_match(agent1, agent2)
        print(f"Two-player match setup: {agent1.name} vs {agent2.name}")
    
    def handle_agent_move_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any, 
                                       move_history: List, move_history_with_timing: List, 
                                       game_step: int) -> Tuple[Any, float, Optional[Dict]]:
        """
        Handle agent move with timeout detection for two-player games.
        
        Returns:
            Tuple of (action, decision_time, timeout_result)
        """
        action, decision_time, timed_out = self.call_agent_with_timeout(agent, observation, action_mask)
        
        if timed_out:
            print(f"{agent.name} timed out! Auto-forfeit.")
            
            # Find the other agent (winner by forfeit)
            other_agent = self._get_other_agent(agent)
            
            timeout_result = {
                'winner': other_agent.name,
                'scores': {agent.name: 0.0, other_agent.name: 1.0},
                'moves': len(move_history),
                'match_history': move_history,
                'move_history_with_timing': move_history_with_timing,
                'timeout_info': {
                    'timed_out_agent': agent.name,
                    'timeout_at_move': game_step + 1,
                    'decision_time': decision_time
                }
            }
            return None, decision_time, timeout_result
        
        return action, decision_time, None
    
    @abstractmethod
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete two-player match.
        
        Returns:
            Dictionary containing match results with keys:
            - 'winner': winner agent name or 'draw'
            - 'scores': dict mapping agent names to scores
            - 'moves': number of moves/turns taken
            - 'match_history': list of moves/actions taken
        """
        pass


class SinglePlayerGame(CompetitiveGame):
    """
    Abstract base class for single-player challenge games.
    
    Handles robin-round competition where two agents each complete
    the same single-player challenge independently, then their results 
    are compared to determine a winner for ELO scoring.
    """
    
    def __init__(self, game_name: str, movement_timeout: float = 45.0, total_game_timeout: float = None):
        """
        Initialize single-player game with cumulative timeout.
        
        Args:
            game_name: Name of the game
            movement_timeout: Per-move timeout (used for emergency cases)
            total_game_timeout: Total cumulative timeout for entire game session
        """
        super().__init__(game_name, movement_timeout)
        # For single-player games, use cumulative timeout instead of per-move timeout
        # Default: 5 minutes for simple games, can be overridden by specific games
        self.total_game_timeout = total_game_timeout if total_game_timeout is not None else 300.0
        self.current_session_start_time = None
        self.current_session_elapsed_time = 0.0
    
    def setup_match(self, agent1: BaseAgent, agent2: BaseAgent) -> None:
        """Setup a robin-round challenge between two agents."""
        super().setup_match(agent1, agent2)
        print(f"Single-player robin-round setup: {agent1.name} vs {agent2.name} on {self.game_name}")
        print(f"  Using cumulative timeout: {self.total_game_timeout:.1f}s per agent session")
    
    def start_agent_session(self, agent: BaseAgent) -> None:
        """Start a new agent session with cumulative timeout tracking."""
        self.current_session_start_time = time.time()
        self.current_session_elapsed_time = 0.0
        print(f"  Starting {agent.name} session with {self.total_game_timeout:.1f}s cumulative timeout")
    
    def get_remaining_time(self) -> float:
        """Get remaining time for current agent session."""
        if self.current_session_start_time is None:
            return float(self.total_game_timeout)
        
        elapsed = float(time.time() - self.current_session_start_time)
        remaining = float(self.total_game_timeout) - elapsed
        return max(0.0, remaining)
    
    def call_agent_with_cumulative_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any) -> Tuple[Any, float, bool, Optional[str]]:
        """
        Call agent's select_action with cumulative timeout detection.
        
        Args:
            agent: The agent to call
            observation: The observation to pass to the agent
            action_mask: The action mask to pass to the agent
            
        Returns:
            Tuple of (action, decision_time, timed_out, error_message)
        """
        start_time = time.time()
        timed_out = False
        action = None
        error_message = None
        
        # Check if we've already exceeded cumulative timeout
        remaining_time = self.get_remaining_time()
        if remaining_time <= 0:
            print(f"{agent.name} exceeded cumulative timeout ({self.total_game_timeout:.1f}s)")
            return None, 0.0, True, None
        
        # Use the smaller of remaining time and per-move timeout for safety
        effective_timeout = min(remaining_time, self.movement_timeout)
        
        # Check if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            # Use signal-based timeout in main thread
            try:
                with TimeoutHandler(float(effective_timeout)):
                    action = agent.select_action(observation, action_mask)
            except TimeoutError:
                timed_out = True
                remaining = self.get_remaining_time()
                if remaining <= 0:
                    print(f"{agent.name} exceeded cumulative timeout ({self.total_game_timeout:.1f}s total)")
                else:
                    print(f"{agent.name} timed out on single move ({effective_timeout:.1f}s), {remaining:.1f}s remaining")
            except Exception as e:
                error_message = str(e)
                print(f"{agent.name} encountered error: {e}")
                timed_out = False  # This is an error, not a timeout
        else:
            # Use threading-based timeout in worker threads
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = agent.select_action(observation, action_mask)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=effective_timeout)
            
            if thread.is_alive():
                # Thread is still running, timeout occurred
                timed_out = True
                remaining = self.get_remaining_time()
                if remaining <= 0:
                    print(f"{agent.name} exceeded cumulative timeout ({self.total_game_timeout:.1f}s total)")
                else:
                    print(f"{agent.name} timed out on single move ({effective_timeout:.1f}s), {remaining:.1f}s remaining")
            elif exception[0]:
                error_message = str(exception[0])
                print(f"{agent.name} encountered error: {exception[0]}")
                timed_out = False  # This is an error, not a timeout
            else:
                action = result[0]
        
        decision_time = time.time() - start_time
        return action, decision_time, timed_out, error_message
    
    def handle_agent_move_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any, 
                                       move_history: List, move_history_with_timing: List, 
                                       game_step: int) -> Tuple[Any, float, Optional[Dict]]:
        """
        Handle agent move with timeout detection for single-player games.
        
        For single-player games, uses cumulative timeout instead of per-move timeout.
        
        Returns:
            Tuple of (action, decision_time, timeout_result)
        """
        # Use cumulative timeout for single-player games
        action, decision_time, timed_out, error_message = self.call_agent_with_cumulative_timeout(agent, observation, action_mask)
        
        if timed_out:
            print(f"{agent.name} timed out! Challenge attempt failed.")
            
            timeout_result = {
                'success': False,
                'moves': len(move_history),
                'move_history': move_history,
                'move_history_with_timing': move_history_with_timing,
                'timeout': True,
                'efficiency': 0.0,
                'timeout_info': {
                    'timed_out_agent': agent.name,
                    'timeout_at_move': game_step + 1,
                    'decision_time': decision_time,
                    'total_time_used': self.total_game_timeout - self.get_remaining_time()
                }
            }
            return None, decision_time, timeout_result
        
        if error_message:
            print(f"{agent.name} encountered error! Challenge attempt failed.")
            
            error_result = {
                'success': False,
                'moves': len(move_history),
                'move_history': move_history,
                'move_history_with_timing': move_history_with_timing,
                'timeout': False,
                'efficiency': 0.0,
                'error_info': {
                    'error_agent': agent.name,
                    'error_type': 'runtime_error',
                    'error_message': error_message,
                    'error_at_move': game_step + 1
                }
            }
            return None, decision_time, error_result
        
        return action, decision_time, None
    
    @abstractmethod
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete robin-round challenge between two agents.
        
        Each agent will complete the same challenge independently,
        then their results will be compared to determine winner/loser/draw.
        
        Returns:
            Dictionary containing match results with keys:
            - 'winner': agent name of the winner, or 'draw' for tie
            - 'scores': dict mapping agent name to score (1.0=win, 0.5=draw, 0.0=loss)
            - 'moves': total number of moves across both attempts
            - 'match_history': list of moves/actions from both agents
            - 'challenge_results': dict with detailed results for each agent
        """
        pass


class MultiPlayerGame(BaseGame):
    """
    Abstract base class for multi-player competitive games.
    
    Handles multi-player matches where multiple agents interact simultaneously
    (e.g., Texas Hold'em poker with 3-6 players). Uses a different tournament
    structure than robin-round - each agent participates in multiple matches
    with different combinations of opponents.
    """
    
    def __init__(self, game_name: str, movement_timeout: float = 45.0, 
                 max_players: int = 6, min_players: int = 2):
        """
        Initialize multi-player game.
        
        Args:
            game_name: Name of the game
            movement_timeout: Maximum time in seconds for each agent decision
            max_players: Maximum number of players per match
            min_players: Minimum number of players per match
        """
        super().__init__(game_name, movement_timeout)
        self.max_players = max_players
        self.min_players = min_players
        self.current_agents = []
        self.agents_dict = {}
        self.current_match_results = {}
    
    def setup_match(self, agents: List[BaseAgent]) -> None:
        """
        Setup a multi-player match.
        
        Args:
            agents: List of agents to participate in this match
        """
        if len(agents) < self.min_players:
            raise ValueError(f"Not enough agents: {len(agents)} < {self.min_players}")
        if len(agents) > self.max_players:
            raise ValueError(f"Too many agents: {len(agents)} > {self.max_players}")
        
        self.reset()
        self.current_agents = agents
        self.agents_dict = {agent.name: agent for agent in agents}
        self.current_match_results = {}
        
        agent_names = [agent.name for agent in agents]
    
    def handle_agent_move_with_timeout(self, agent: BaseAgent, observation: Any, action_mask: Any, 
                                       move_history: List, move_history_with_timing: List, 
                                       game_step: int) -> Tuple[Any, float, Optional[Dict]]:
        """
        Handle agent move with timeout detection for multi-player games.
        
        For multi-player games, if an agent times out or errors, they are treated as folding
        or taking a default action to avoid disrupting the game for other players.
        
        Returns:
            Tuple of (action, decision_time, timeout_result)
        """
        action, decision_time, timed_out = self.call_agent_with_timeout(agent, observation, action_mask)
        
        if timed_out or action is None:
            # For multi-player games, handle timeout gracefully
            default_action = self.get_default_action(action_mask)
            print(f"{agent.name} timed out/errored! Using default action: {default_action}")
            
            timeout_result = {
                'timed_out_agent': agent.name,
                'timeout_at_step': game_step,
                'decision_time': decision_time,
                'default_action_used': default_action
            }
            return default_action, decision_time, timeout_result
        
        return action, decision_time, None
    
    @abstractmethod
    def get_default_action(self, action_mask: Any) -> Any:
        """
        Get a default action for when an agent times out or errors.
        
        Args:
            action_mask: Valid actions mask
            
        Returns:
            Default action (usually fold or pass)
        """
        pass
    
    @abstractmethod
    def run_match(self) -> Dict[str, Any]:
        """
        Run a complete multi-player match.
        
        Returns:
            Dictionary containing match results with keys:
            - 'results': dict mapping agent names to their final scores/rankings
            - 'moves': total number of moves/actions taken
            - 'match_history': list of moves/actions from all agents
            - 'final_rankings': list of agent names in order of finish
            - 'rewards': dict mapping agent names to their final rewards
        """
        pass


