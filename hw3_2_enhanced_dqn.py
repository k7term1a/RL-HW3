"""
HW3-2 – Double DQN vs Dueling DQN on GridWorld (player mode).

Run:
    uv run python hw3_2_enhanced_dqn.py

Produces side-by-side comparison plot and per-agent GIFs.
"""
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gridworld    import GridWorld
from models       import NaiveDQN, DuelingDQN
from replay_buffer import ReplayBuffer
from utils        import plot_metrics, plot_comparison, record_episode, save_gif, EarlyStopping

# ── Hyper-parameters ────────────────────────────────────────────────────────
SEED        = 42
EPISODES    = 3000          # upper bound; early stopping usually triggers earlier
MAX_STEPS   = 50
GAMMA       = 0.9
LR          = 1e-3
BATCH_SIZE  = 64
BUFFER_SIZE = 10_000
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 0.997
TARGET_SYNC = 20
GIF_EVERY   = 50

OUT_DIR = "outputs/hw3_2"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Generic training loop ─────────────────────────────────────────────────────
def train(model_cls, name: str, double: bool) -> tuple[list, list]:
    """
    Train one agent.
    double=True  → Double DQN target.
    double=False → Vanilla DQN target.
    """
    env     = GridWorld(mode="player")
    online  = model_cls().to(device)
    target  = copy.deepcopy(online).to(device); target.eval()
    opt     = optim.Adam(online.parameters(), lr=LR)
    buf     = ReplayBuffer(BUFFER_SIZE)
    loss_fn = nn.MSELoss()
    epsilon = EPS_START

    # Player mode: target MA-50 >= 5, patience=5 checks
    es = EarlyStopping(target=5.0, patience=5, window=50, min_delta=0.2, check_every=10)

    agent_out = os.path.join(OUT_DIR, name)
    os.makedirs(agent_out, exist_ok=True)

    rewards, losses = [], []

    def select_action(s, eps):
        if random.random() < eps:
            return random.randint(0, 3)
        with torch.no_grad():
            t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            return int(online(t).argmax(1).item())

    def _greedy(s, _net=online):
        with torch.no_grad():
            t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            return int(_net(t).argmax(1).item())

    def update_step():
        if len(buf) < BATCH_SIZE:
            return None
        s, a, r, ns, d = buf.sample(BATCH_SIZE)
        s  = torch.tensor(s,  device=device)
        a  = torch.tensor(a,  device=device)
        r  = torch.tensor(r,  device=device)
        ns = torch.tensor(ns, device=device)
        d  = torch.tensor(d,  device=device)

        q_cur = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if double:
                best_a = online(ns).argmax(1, keepdim=True)
                q_next = target(ns).gather(1, best_a).squeeze(1)
            else:
                q_next = target(ns).max(1).values

        q_tgt = r + GAMMA * q_next * (1 - d)
        loss  = loss_fn(q_cur, q_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    for ep in range(1, EPISODES + 1):
        state     = env.reset()
        ep_reward = 0.0

        if ep % GIF_EVERY == 0 or ep == 1:
            frames = record_episode(GridWorld(mode="player"), _greedy, ep)
            save_gif(frames, os.path.join(agent_out, f"ep{ep:04d}.gif"))

        for _ in range(MAX_STEPS):
            action           = select_action(state, epsilon)
            ns, reward, done = env.step(action)
            buf.push(state, action, reward, ns, done)
            state      = ns
            ep_reward += reward
            loss = update_step()
            if loss is not None:
                losses.append(loss)
            if done:
                break

        rewards.append(ep_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % TARGET_SYNC == 0:
            target.load_state_dict(online.state_dict())

        if ep % 200 == 0:
            avg = np.mean(rewards[-100:])
            print(f"[{name}] Ep {ep:4d} | avg_r={avg:6.2f} | eps={epsilon:.3f}")

        if es.step(rewards):
            if ep % GIF_EVERY != 0:
                frames = record_episode(GridWorld(mode="player"), _greedy, ep)
                save_gif(frames, os.path.join(agent_out, f"ep{ep:04d}.gif"))
            print(f"[{name}] Early stop at episode {ep}.")
            break

    torch.save(online.state_dict(), os.path.join(agent_out, f"{name}.pt"))
    plot_metrics(rewards, losses, f"{name} – Player Mode",
                 os.path.join(agent_out, "metrics.png"))
    print(f"[{name}] Total episodes: {len(rewards)}")
    return rewards, losses


# ── Run experiments ───────────────────────────────────────────────────────────
print("=" * 50)
print("Training Naive DQN (no double) …")
rewards_naive, _ = train(NaiveDQN, "naive_dqn", double=False)

print("=" * 50)
print("Training Double DQN …")
rewards_double, _ = train(NaiveDQN, "double_dqn", double=True)

print("=" * 50)
print("Training Dueling DQN …")
rewards_dueling, _ = train(DuelingDQN, "dueling_dqn", double=False)

print("=" * 50)
print("Training Dueling Double DQN …")
rewards_dd, _ = train(DuelingDQN, "dueling_double_dqn", double=True)

# ── Comparison plot ───────────────────────────────────────────────────────────
plot_comparison(
    {
        "Naive DQN":          rewards_naive,
        "Double DQN":         rewards_double,
        "Dueling DQN":        rewards_dueling,
        "Dueling+Double DQN": rewards_dd,
    },
    os.path.join(OUT_DIR, "comparison.png"),
)
print(f"\nAll done. Outputs in {OUT_DIR}/")
