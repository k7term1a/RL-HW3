"""
Shared utilities: plotting, GIF rendering, metric helpers, early stopping.
"""
import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ---------------------------------------------------------------------------
class EarlyStopping:
    """
    Stop training when the MA-window reward:
      (a) exceeds `target` for `patience` consecutive checks, OR
      (b) hasn't improved by `min_delta` over the last `patience` checks.

    Call .step(rewards_list) every episode; returns True when training
    should stop.

    Parameters
    ----------
    target      : reward threshold considered "solved"
    patience    : how many consecutive checks must pass the target (a),
                  or how many checks without improvement trigger stop (b)
    window      : moving-average window size
    min_delta   : minimum improvement required to reset the no-improvement counter
    check_every : evaluate only every N episodes (reduces overhead)
    """

    def __init__(
        self,
        target:      float = 7.0,
        patience:    int   = 5,
        window:      int   = 50,
        min_delta:   float = 0.2,
        check_every: int   = 10,
    ):
        self.target      = target
        self.patience    = patience
        self.window      = window
        self.min_delta   = min_delta
        self.check_every = check_every

        self._consec_hit  = 0   # consecutive checks above target
        self._no_imp      = 0   # consecutive checks without improvement
        self._best_ma     = -float("inf")

    def step(self, rewards: list[float]) -> bool:
        if len(rewards) % self.check_every != 0:
            return False
        if len(rewards) < self.window:
            return False

        ma = float(np.mean(rewards[-self.window:]))

        # (a) target reached
        if ma >= self.target:
            self._consec_hit += 1
            if self._consec_hit >= self.patience:
                print(f"[EarlyStopping] Solved: MA-{self.window}={ma:.2f} >= {self.target} "
                      f"for {self.patience} checks.")
                return True
        else:
            self._consec_hit = 0

        # (b) no improvement
        if ma > self._best_ma + self.min_delta:
            self._best_ma = ma
            self._no_imp  = 0
        else:
            self._no_imp += 1
            if self._no_imp >= self.patience * 10:
                print(f"[EarlyStopping] No improvement: MA-{self.window}={ma:.2f}, "
                      f"best={self._best_ma:.2f}. Stopping.")
                return True

        return False


# ---------------------------------------------------------------------------
def moving_average(values: list, window: int = 50) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ---------------------------------------------------------------------------
def plot_metrics(rewards: list, losses: list, title: str, save_path: str):
    """Two-panel figure: reward curve + loss curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    ax1.plot(rewards, alpha=0.3, color="steelblue", label="reward")
    if len(rewards) >= 50:
        ma = moving_average(rewards, 50)
        ax1.plot(range(49, 49 + len(ma)), ma, color="steelblue", label="MA-50")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(losses, alpha=0.3, color="tomato", label="loss")
    if len(losses) >= 50:
        ma = moving_average(losses, 50)
        ax2.plot(range(49, 49 + len(ma)), ma, color="tomato", label="MA-50")
    ax2.set_xlabel("Update Step"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_comparison(results: dict, save_path: str):
    """Overlay MA-50 reward curves for multiple agents."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]
    for i, (name, rewards) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax.plot(rewards, alpha=0.15, color=color)
        if len(rewards) >= 50:
            ma = moving_average(rewards, 50)
            ax.plot(range(49, 49 + len(ma)), ma, color=color, label=name, linewidth=2)
        else:
            ax.plot(rewards, color=color, label=name, linewidth=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward (MA-50)")
    ax.set_title("DQN Variant Comparison")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _render_grid_frame(env, episode: int, step: int, total_reward: float) -> Image.Image:
    """Render a single GridWorld frame as a PIL Image."""
    size = env.size
    cell = 80
    fig_size = cell * size / 100

    fig, ax = plt.subplots(figsize=(fig_size, fig_size + 0.6))
    ax.set_xlim(0, size); ax.set_ylim(0, size)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("#f8f8f8")

    colors = {"P": "#4a90d9", "G": "#2ecc71", "X": "#e74c3c", "#": "#7f8c8d"}
    labels = {"P": "Agent", "G": "Goal", "X": "Pit", "#": "Wall"}

    objs = {env.player: "P", env.goal: "G", env.pit: "X", env.wall: "#"}
    for (r, c), sym in objs.items():
        x, y = c, size - 1 - r
        rect = patches.FancyBboxPatch(
            (x + 0.05, y + 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.05",
            facecolor=colors[sym], edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.5, labels[sym], ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    # grid lines
    for i in range(size + 1):
        ax.axhline(i, color="#cccccc", linewidth=0.5)
        ax.axvline(i, color="#cccccc", linewidth=0.5)

    ax.set_title(f"Ep {episode}  Step {step}  R={total_reward:.1f}",
                 fontsize=9, pad=4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def record_episode(env, policy_fn, episode: int) -> list[Image.Image]:
    """
    Run one episode using policy_fn(state) -> action.
    Returns list of PIL frames.
    """
    state = env.reset()
    frames, total_reward, step = [], 0.0, 0
    frames.append(_render_grid_frame(env, episode, step, total_reward))

    for _ in range(50):
        action = policy_fn(state)
        state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        frames.append(_render_grid_frame(env, episode, step, total_reward))
        if done:
            break
    return frames


def save_gif(frames: list[Image.Image], path: str, fps: int = 4):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
