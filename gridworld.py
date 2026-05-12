"""
GridWorld environment compatible with DRL-in-Action Chapter 4.
Supports three initialisation modes: static, player, random.

Grid layout (4×4, row 0 at top):
  Coordinates are (row, col), both 0-indexed.

Objects
-------
  Player : the agent
  Goal   : +10 reward, terminates episode
  Pit    : -10 reward, terminates episode
  Wall   : impassable; collision → -1 reward, agent stays

Actions: 0=up, 1=down, 2=left, 3=right

State representations
---------------------
  enhanced=False (default): 64-dim flat one-hot (4 channels × 4×4).
  enhanced=True           : 64-dim one-hot + 8-dim normalised relative
                            features = 72-dim total.

The 8 relative features are:
  player_row/3, player_col/3,
  (goal-player)_row/3, (goal-player)_col/3,
  (pit-player)_row/3,  (pit-player)_col/3,
  (wall-player)_row/3, (wall-player)_col/3

These directly encode the spatial relationships that matter for
navigation and allow the network to generalise across random
object configurations that share the same relative structure.
"""
import random
import numpy as np


class GridWorld:
    def __init__(self, size: int = 4, mode: str = "static",
                 enhanced: bool = False):
        assert mode in ("static", "player", "random")
        self.size     = size
        self.mode     = mode
        self.enhanced = enhanced
        # state_dim exposed so callers can build models correctly
        self.state_dim = 64 + (8 if enhanced else 0)
        self.reset()

    # ------------------------------------------------------------------
    def _random_pos(self, exclude: list[tuple]) -> tuple[int, int]:
        while True:
            pos = (random.randint(0, self.size - 1),
                   random.randint(0, self.size - 1))
            if pos not in exclude:
                return pos

    def reset(self) -> np.ndarray:
        if self.mode == "static":
            self.player = (0, 3)
            self.goal   = (0, 0)
            self.pit    = (0, 1)
            self.wall   = (1, 1)
        elif self.mode == "player":
            self.goal   = (0, 0)
            self.pit    = (0, 1)
            self.wall   = (1, 1)
            self.player = self._random_pos([self.goal, self.pit, self.wall])
        else:  # random
            taken = []
            self.goal   = self._random_pos(taken); taken.append(self.goal)
            self.pit    = self._random_pos(taken); taken.append(self.pit)
            self.wall   = self._random_pos(taken); taken.append(self.wall)
            self.player = self._random_pos(taken)

        self.done = False
        return self._get_state()

    # ------------------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        # 64-dim one-hot: 4 objects × 4×4 grid
        grid = np.zeros((4, self.size, self.size), dtype=np.float32)
        grid[0][self.player] = 1.0
        grid[1][self.goal]   = 1.0
        grid[2][self.pit]    = 1.0
        grid[3][self.wall]   = 1.0
        base = grid.flatten()

        if not self.enhanced:
            return base

        # 8 relative/absolute navigation features, normalised to [-1, 1]
        n = float(self.size - 1)
        rel = np.array([
            self.player[0] / n,
            self.player[1] / n,
            (self.goal[0]  - self.player[0]) / n,
            (self.goal[1]  - self.player[1]) / n,
            (self.pit[0]   - self.player[0]) / n,
            (self.pit[1]   - self.player[1]) / n,
            (self.wall[0]  - self.player[0]) / n,
            (self.wall[1]  - self.player[1]) / n,
        ], dtype=np.float32)
        return np.concatenate([base, rel])

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """action: 0=up,1=down,2=left,3=right"""
        if self.done:
            raise RuntimeError("Call reset() before step() after episode end.")

        r, c = self.player
        if   action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1

        new_pos = (r, c)
        if (r < 0 or r >= self.size or c < 0 or c >= self.size or
                new_pos == self.wall):
            return self._get_state(), -1.0, False

        self.player = new_pos

        if self.player == self.goal:
            reward, self.done = 10.0, True
        elif self.player == self.pit:
            reward, self.done = -10.0, True
        else:
            reward = -1.0

        return self._get_state(), reward, self.done

    # ------------------------------------------------------------------
    def manhattan_to_goal(self) -> int:
        return abs(self.player[0] - self.goal[0]) + \
               abs(self.player[1] - self.goal[1])

    def render(self) -> str:
        symbols = {self.player: "P", self.goal: "G",
                   self.pit: "X", self.wall: "#"}
        rows = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                row += symbols.get((r, c), ".")
            rows.append(row)
        return "\n".join(rows)
