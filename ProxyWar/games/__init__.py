"""
ProxyWar Games Module

This module contains game implementations and base game interface.
"""

from .base import BaseGame, TwoPlayerGame, SinglePlayerGame, MultiPlayerGame
from .tictactoe import TicTacToeGame
from .connectfour import ConnectFourGame
from .hanoi import HanoiTowerGame
from .sudoku import SudokuGame
from .twenty_forty_eight import TwentyFortyEightGame
from .maze import MazeGame
from .snake import SnakeGame
from .reversi import ReversiGame
from .texas_holdem import TexasHoldemGame

__all__ = [
    'BaseGame', 'TwoPlayerGame', 'SinglePlayerGame', 'MultiPlayerGame',
    'TicTacToeGame', 'ConnectFourGame', 'HanoiTowerGame', 'SudokuGame',
    'TwentyFortyEightGame', 'MazeGame', 'SnakeGame', 'ReversiGame', 'TexasHoldemGame'
]
