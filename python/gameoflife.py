"""
Run with:  python3 python/gameoflife.py

Creates a ramdom board and steps in time, 
"""

import numpy as np
import argparse
from gol import GameOfLife


def create_grid(height: int, width: int) -> np.ndarray:
    state =  np.array(np.random.rand(height, width) > 0.5, dtype=np.uint8)
    return state


def render(board: np.ndarray) -> str:
    BG_ALIVE = "\x1b[38;2;255;228;181;208m\x1b[48;2;5;5;20;208m\u25A0"
    BG_DEAD = "\x1b[38;2;10;9;59;208m\x1b[48;2;5;5;20;208m\u25A0"
    RESET = "\x1b[0m"

    return "\n".join(
        "".join((BG_ALIVE if c else BG_DEAD) + " " for c in row) + RESET
        for row in board
    ) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conway's Game of Life"
    )

    parser.add_argument(
        "--rows", "-r",
        type=int,
        default=20,
        help="Number of rows",
    )

    parser.add_argument(
        "--cols", "-c",
        type=int,
        default=20,
        help="Number of columns",
    )

    parser.add_argument(
        "--back", "-b",
        type=str,
        default='gpu',
        help="Backend to run the calculations",
    )

    args = parser.parse_args()

    rows = args.rows
    cols = args.cols

    game = GameOfLife(create_grid(rows, cols), backend=args.back)

    for gen in range(1000):
        print(render(game.state()), end="")
        print(f"\x1b[{rows}F", end="")
        game.step(1)
