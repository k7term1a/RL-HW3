"""
HW3-4 (Bonus) – Rainbow DQN for GridWorld (random mode).

Rainbow components implemented:
  1. Double DQN          – online net selects action, target net evaluates
  2. Dueling Networks    – value + advantage streams
  3. Noisy Nets          – replaces ε-greedy exploration
  4. C51 (Distributional)– categorical distribution over returns
  5. Prioritised Replay  – proportional PER with IS weights
  6. Multi-step returns  – n-step TD target (n=3)

Improvements for random mode:
  • Enhanced state (64-dim one-hot + 8-dim relative features = 72-dim)
  • Potential-based reward shaping toward Goal
  • _greedy GIF recording uses try/finally to guarantee online.train()

Run:
    uv run python hw3_4_rainbow_dqn.py
"""
import os
import copy
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gridworld     import GridWorld
from models        import RainbowDQN
from replay_buffer import PrioritisedReplayBuffer
from utils         import plot_metrics, record_episode, save_gif, EarlyStopping

# ── Hyper-parameters ────────────────────────────────────────────────────────
SEED           = 42
EPISODES       = 5000           # upper bound; early stopping usually fires earlier
MAX_STEPS      = 50
GAMMA          = 0.9
N_STEP         = 3
LR             = 5e-4
BATCH_SIZE     = 64
BUFFER_SIZE    = 20_000
WARMUP         = 1000
TAU            = 0.005
GRAD_CLIP      = 10.0
PER_ALPHA      = 0.6
PER_BETA0      = 0.4
PER_BETA_STEPS = EPISODES * MAX_STEPS
N_ATOMS        = 51
V_MIN          = -15.0          # -10 was too narrow: 3-step worst-case ≈ -11.1
V_MAX          = 12.0           # +10 was too narrow: goal+shaping ≈ +10.4
GIF_EVERY      = 50
SHAPING_SCALE  = 0.4            # potential-based shaping weight

STATE_DIM = 72                  # 64 one-hot + 8 relative features

OUT_DIR = "outputs/hw3_4"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Models ────────────────────────────────────────────────────────────────────
online = RainbowDQN(state_dim=STATE_DIM,
                    n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX).to(device)
target = copy.deepcopy(online).to(device); target.eval()
for p in target.parameters():
    p.requires_grad = False

opt     = optim.Adam(online.parameters(), lr=LR)
buf     = PrioritisedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
support = online.support

delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)


# ── N-step buffer ─────────────────────────────────────────────────────────────
class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self.deque: deque = deque()

    def push(self, *transition):
        self.deque.append(transition)

    def ready(self):
        return len(self.deque) >= self.n

    def get(self):
        """Return n-step (s0, a0, G_n, s_n, done_n)."""
        s0, a0, _, _, _ = self.deque[0]
        G, done_n = 0.0, False
        for i, (_, _, r, _, d) in enumerate(list(self.deque)[:self.n]):
            G += (self.gamma ** i) * r
            if d:
                done_n = True
                break
        _, _, _, sn, _ = self.deque[min(self.n - 1, len(self.deque) - 1)]
        self.deque.popleft()
        return s0, a0, G, sn, float(done_n)

    def flush(self):
        results = []
        while self.deque:
            s0, a0, _, _, _ = self.deque[0]
            G, done_n = 0.0, False
            for i, (_, _, r, _, d) in enumerate(self.deque):
                G += (self.gamma ** i) * r
                if d:
                    done_n = True
                    break
            _, _, _, sn, _ = self.deque[-1]
            self.deque.popleft()
            results.append((s0, a0, G, sn, float(done_n)))
        return results


# ── C51 distributional loss ───────────────────────────────────────────────────
def distributional_loss(
    states:      torch.Tensor,
    actions:     torch.Tensor,
    rewards:     torch.Tensor,
    next_states: torch.Tensor,
    dones:       torch.Tensor,
    weights:     torch.Tensor,
    n:           int,
) -> tuple[torch.Tensor, np.ndarray]:
    B = states.size(0)

    log_p   = online(states)
    log_p_a = log_p[range(B), actions]         # (B, Z)

    with torch.no_grad():
        # Double DQN: online selects, target evaluates
        q_next   = online.q_values(next_states)
        best_a   = q_next.argmax(1)
        p_next   = target(next_states).exp()
        p_next_a = p_next[range(B), best_a]    # (B, Z)

        gamma_n = GAMMA ** n
        Tz = (rewards.unsqueeze(1)
              + (1 - dones.unsqueeze(1)) * gamma_n * support.unsqueeze(0))
        Tz = Tz.clamp(V_MIN, V_MAX)

        b  = (Tz - V_MIN) / delta_z
        lo = b.floor().long().clamp(0, N_ATOMS - 1)
        hi = b.ceil().long().clamp(0, N_ATOMS - 1)

        proj   = torch.zeros(B, N_ATOMS, device=device)
        offset = torch.arange(B, device=device).unsqueeze(1) * N_ATOMS
        proj.view(-1).scatter_add_(
            0, (lo + offset).view(-1),
            (p_next_a * (hi.float() - b + 1e-8)).view(-1))
        proj.view(-1).scatter_add_(
            0, (hi + offset).view(-1),
            (p_next_a * (b - lo.float() + 1e-8)).view(-1))

    loss_per  = -(proj * log_p_a).sum(1)
    loss      = (weights * loss_per).mean()
    td_errors = loss_per.detach().abs().cpu().numpy()
    return loss, td_errors


# ── Pre-fill buffer ───────────────────────────────────────────────────────────
env   = GridWorld(mode="random", enhanced=True)
n_buf = NStepBuffer(N_STEP, GAMMA)
state = env.reset()
for _ in range(WARMUP):
    a           = random.randint(0, 3)
    old_dist    = env.manhattan_to_goal()
    ns, r, done = env.step(a)
    new_dist    = env.manhattan_to_goal()
    shaped_r    = r + SHAPING_SCALE * (old_dist - new_dist)

    n_buf.push(state, a, shaped_r, ns, done)
    if n_buf.ready():
        buf.push(*n_buf.get())
    if done:
        for t in n_buf.flush():
            buf.push(*t)
        n_buf = NStepBuffer(N_STEP, GAMMA)
        state = env.reset()
    else:
        state = ns

print(f"Buffer pre-filled: {len(buf)} transitions")

# ── Training loop ─────────────────────────────────────────────────────────────
all_rewards: list[float] = []
all_losses:  list[float] = []
beta_step   = 0
early_stop  = EarlyStopping(target=3.0, patience=15, window=50,
                             min_delta=0.05, check_every=10)


def get_beta() -> float:
    frac = min(1.0, beta_step / PER_BETA_STEPS)
    return PER_BETA0 + frac * (1.0 - PER_BETA0)


def polyak_update():
    with torch.no_grad():
        for po, pt in zip(online.parameters(), target.parameters()):
            pt.data.copy_(TAU * po.data + (1 - TAU) * pt.data)


def collect_action(s: np.ndarray) -> int:
    """Action selection during training: stays in train mode so NoisyLinear provides exploration."""
    with torch.no_grad():
        t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        return int(online.q_values(t).argmax(1).item())


def greedy_action(s: np.ndarray) -> int:
    """Action selection for GIF/evaluation: eval mode uses mean weights (no noise)."""
    online.eval()
    try:
        with torch.no_grad():
            t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            return int(online.q_values(t).argmax(1).item())
    finally:
        online.train()


for ep in range(1, EPISODES + 1):
    state     = env.reset()
    n_buf     = NStepBuffer(N_STEP, GAMMA)
    ep_reward = 0.0
    online.train()
    online.sample_noise()

    if ep % GIF_EVERY == 0 or ep == 1:
        gif_env = GridWorld(mode="random", enhanced=True)
        frames  = record_episode(gif_env, greedy_action, ep)
        save_gif(frames, os.path.join(OUT_DIR, f"ep{ep:04d}.gif"))
        online.train()   # ensure training mode after GIF recording

    for _ in range(MAX_STEPS):
        old_dist = env.manhattan_to_goal()
        action   = collect_action(state)  # train mode → NoisyLinear explores

        ns, reward, done = env.step(action)
        new_dist  = env.manhattan_to_goal()
        shaped_r  = reward + SHAPING_SCALE * (old_dist - new_dist)
        ep_reward += reward               # log unshaped

        n_buf.push(state, action, shaped_r, ns, done)
        if n_buf.ready():
            buf.push(*n_buf.get())
        if done:
            for t in n_buf.flush():
                buf.push(*t)

        # Gradient update
        if len(buf) >= BATCH_SIZE:
            beta = get_beta(); beta_step += 1
            s_, a_, r_, ns_, d_, idxs, w_ = buf.sample(BATCH_SIZE, beta=beta)

            online.sample_noise()
            loss, td_errors = distributional_loss(
                torch.tensor(s_,  device=device),
                torch.tensor(a_,  device=device),
                torch.tensor(r_,  device=device),
                torch.tensor(ns_, device=device),
                torch.tensor(d_,  device=device),
                torch.tensor(w_,  device=device),
                N_STEP,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
            opt.step()

            buf.update_priorities(idxs, td_errors)
            all_losses.append(loss.item())
            polyak_update()

        state = env.reset() if done else ns
        if done:
            break

    all_rewards.append(ep_reward)

    if ep % 200 == 0:
        avg = np.mean(all_rewards[-100:])
        print(f"Ep {ep:4d} | avg_r={avg:6.2f} | buf={len(buf)}")

    if early_stop.step(all_rewards):
        if ep % GIF_EVERY != 0:
            gif_env = GridWorld(mode="random", enhanced=True)
            frames  = record_episode(gif_env, greedy_action, ep)
            save_gif(frames, os.path.join(OUT_DIR, f"ep{ep:04d}.gif"))
        print(f"Early stop at episode {ep}.")
        break

# ── Save artefacts ────────────────────────────────────────────────────────────
torch.save(online.state_dict(), os.path.join(OUT_DIR, "rainbow_dqn.pt"))
plot_metrics(all_rewards, all_losses, "Rainbow DQN – Random Mode",
             os.path.join(OUT_DIR, "metrics.png"))
print(f"Total episodes: {len(all_rewards)}")
print(f"Outputs in {OUT_DIR}/")
