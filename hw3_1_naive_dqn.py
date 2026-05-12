"""
HW3-1 – Naive DQN with Experience Replay on GridWorld (static mode).

Run:
    uv run python hw3_1_naive_dqn.py
"""
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gridworld    import GridWorld
from models       import NaiveDQN
from replay_buffer import ReplayBuffer
from utils        import plot_metrics, record_episode, save_gif, EarlyStopping

# ── Hyper-parameters ────────────────────────────────────────────────────────
SEED        = 42
EPISODES    = 2000          # upper bound; early stopping will usually trigger earlier
MAX_STEPS   = 50
GAMMA       = 0.9
LR          = 1e-3
BATCH_SIZE  = 64
BUFFER_SIZE = 10_000
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 0.995
TARGET_SYNC = 20
GIF_EVERY   = 50

OUT_DIR = "outputs/hw3_1"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Components ───────────────────────────────────────────────────────────────
env      = GridWorld(mode="static")
online   = NaiveDQN().to(device)
target   = copy.deepcopy(online).to(device)
target.eval()

optimizer = optim.Adam(online.parameters(), lr=LR)
buffer    = ReplayBuffer(BUFFER_SIZE)
loss_fn   = nn.MSELoss()

# Static mode is easiest: target MA-50 >= 7, patience=5 checks of 10 ep each
early_stop = EarlyStopping(target=7.0, patience=5, window=50,
                           min_delta=0.2, check_every=10)

# ── Training ─────────────────────────────────────────────────────────────────
epsilon     = EPS_START
all_rewards = []
all_losses  = []


def select_action(state: np.ndarray, eps: float) -> int:
    if random.random() < eps:
        return random.randint(0, 3)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(online(s).argmax(dim=1).item())


def update():
    if len(buffer) < BATCH_SIZE:
        return None
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    s  = torch.tensor(states,      device=device)
    a  = torch.tensor(actions,     device=device)
    r  = torch.tensor(rewards,     device=device)
    ns = torch.tensor(next_states, device=device)
    d  = torch.tensor(dones,       device=device)

    q_cur    = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target(ns).max(1).values
    q_target = r + GAMMA * q_next * (1 - d)

    loss = loss_fn(q_cur, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


for ep in range(1, EPISODES + 1):
    state     = env.reset()
    ep_reward = 0.0

    if ep % GIF_EVERY == 0 or ep == 1:
        def _greedy(s):
            with torch.no_grad():
                t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                return int(online(t).argmax(dim=1).item())
        frames = record_episode(GridWorld(mode="static"), _greedy, ep)
        save_gif(frames, os.path.join(OUT_DIR, f"ep{ep:04d}.gif"))

    for _ in range(MAX_STEPS):
        action              = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state      = next_state
        ep_reward += reward
        loss = update()
        if loss is not None:
            all_losses.append(loss)
        if done:
            break

    all_rewards.append(ep_reward)
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if ep % TARGET_SYNC == 0:
        target.load_state_dict(online.state_dict())

    if ep % 100 == 0:
        avg = np.mean(all_rewards[-100:])
        print(f"Ep {ep:4d} | avg_r={avg:6.2f} | eps={epsilon:.3f} | buf={len(buffer)}")

    if early_stop.step(all_rewards):
        # record one final snapshot at the stopping episode
        if ep % GIF_EVERY != 0:
            frames = record_episode(GridWorld(mode="static"), _greedy, ep)
            save_gif(frames, os.path.join(OUT_DIR, f"ep{ep:04d}.gif"))
        print(f"Early stop at episode {ep}.")
        break

# ── Save artefacts ─────────────────────────────────────────────────────────
torch.save(online.state_dict(), os.path.join(OUT_DIR, "naive_dqn.pt"))
plot_metrics(all_rewards, all_losses, "Naive DQN – Static Mode",
             os.path.join(OUT_DIR, "metrics.png"))
print(f"Total episodes: {len(all_rewards)}")
print(f"Outputs saved to {OUT_DIR}/")
