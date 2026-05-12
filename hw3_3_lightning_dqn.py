"""
HW3-3 – Dueling Double DQN in PyTorch Lightning for GridWorld (random mode).

Training tips applied:
  • Enhanced state (64-dim one-hot + 8-dim relative features = 72-dim)
  • Potential-based reward shaping toward Goal
  • Gradient clipping (max_norm=10)
  • Cosine annealing learning-rate schedule
  • Target-network soft update (Polyak averaging τ=0.005)
  • 4 gradient updates per episode
  • Warm-up period before training starts

Run:
    uv run python hw3_3_lightning_dqn.py
"""
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

from gridworld    import GridWorld
from models       import DuelingDQN
from replay_buffer import ReplayBuffer
from utils        import plot_metrics, record_episode, save_gif, EarlyStopping

# ── Hyper-parameters ────────────────────────────────────────────────────────
SEED            = 42
EPISODES        = 2000          # upper bound; early stopping usually fires earlier
MAX_STEPS       = 50
GAMMA           = 0.9
LR              = 5e-4
BATCH_SIZE      = 128
BUFFER_SIZE     = 20_000
WARMUP_STEPS    = 1000          # more warm-up so buffer has diverse samples
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.997
TAU             = 0.005         # Polyak soft-update coefficient
GRAD_CLIP       = 10.0
GIF_EVERY       = 50
GRAD_STEPS_EP   = 4             # gradient updates per episode
SHAPING_SCALE   = 0.4           # potential-based reward shaping weight

STATE_DIM = 72                  # 64 one-hot + 8 relative features

OUT_DIR = "outputs/hw3_3"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Iterable Dataset – yields GRAD_STEPS_EP independent batches per epoch ────
class RLDataset(IterableDataset):
    """
    Each __iter__ call samples n_batches independent mini-batches from
    the replay buffer, yielding all items individually so that DataLoader
    can re-batch them.  This gives n_batches gradient updates per episode.
    """

    def __init__(self, buffer: ReplayBuffer, batch_size: int, n_batches: int):
        self.buffer     = buffer
        self.batch_size = batch_size
        self.n_batches  = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            s, a, r, ns, d = self.buffer.sample(self.batch_size)
            for i in range(self.batch_size):
                yield s[i], a[i], r[i], ns[i], d[i]


# ── Lightning Module ──────────────────────────────────────────────────────────
class DQNLightning(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.env    = GridWorld(mode="random", enhanced=True)
        self.buffer = ReplayBuffer(hparams["buffer_size"])

        self.online = DuelingDQN(state_dim=STATE_DIM)
        self.target = copy.deepcopy(self.online); self.target.eval()
        self._freeze_target()

        self.loss_fn  = nn.SmoothL1Loss()
        self.epsilon  = hparams["eps_start"]
        self.episode  = 0

        self.all_rewards:  list[float] = []
        self.all_losses:   list[float] = []
        self._should_stop: bool        = False
        self._early_stop = EarlyStopping(target=3.0, patience=5, window=50,
                                         min_delta=0.2, check_every=10)

        self._populate_buffer(hparams["warmup_steps"])

    # ------------------------------------------------------------------
    def _freeze_target(self):
        for p in self.target.parameters():
            p.requires_grad = False

    def _polyak_update(self):
        tau = self.hparams["tau"]
        with torch.no_grad():
            for po, pt in zip(self.online.parameters(),
                               self.target.parameters()):
                pt.data.copy_(tau * po.data + (1 - tau) * pt.data)

    def _populate_buffer(self, n: int):
        state = self.env.reset()
        for _ in range(n):
            action           = random.randint(0, 3)
            ns, reward, done = self.env.step(action)
            self.buffer.push(state, action, reward, ns, done)
            state = self.env.reset() if done else ns

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(t).argmax(1).item())

    # ------------------------------------------------------------------
    def play_episode(self):
        state     = self.env.reset()
        ep_reward = 0.0

        for _ in range(self.hparams["max_steps"]):
            # Potential-based reward shaping: F = γ·Φ(s') - Φ(s)
            # Φ(s) = -dist_to_goal  →  F ≈ scale·(old_dist - new_dist)
            old_dist = self.env.manhattan_to_goal()

            action           = self.select_action(state)
            ns, reward, done = self.env.step(action)

            new_dist      = self.env.manhattan_to_goal()
            shaped_reward = reward + self.hparams["shaping_scale"] * (old_dist - new_dist)

            self.buffer.push(state, action, shaped_reward, ns, done)
            state      = ns
            ep_reward += reward   # log original (unshaped) reward

            if done:
                break

        self.episode += 1
        self.epsilon  = max(self.hparams["eps_end"],
                            self.epsilon * self.hparams["eps_decay"])
        self.all_rewards.append(ep_reward)
        self._should_stop = self._early_stop.step(self.all_rewards)

        if (self.episode % self.hparams["gif_every"] == 0
                or self.episode == 1
                or self._should_stop):
            dev = self.device
            def _greedy(s, _net=self.online, _dev=dev):
                _net.eval()
                with torch.no_grad():
                    t = torch.tensor(s, dtype=torch.float32, device=_dev).unsqueeze(0)
                    action = int(_net(t).argmax(1).item())
                _net.train()
                return action
            frames = record_episode(
                GridWorld(mode="random", enhanced=True), _greedy, self.episode)
            save_gif(frames, os.path.join(self.hparams["out_dir"],
                                          f"ep{self.episode:04d}.gif"))

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        s, a, r, ns, d = batch
        s  = s.float(); ns = ns.float()
        a  = a.long();  r  = r.float(); d = d.float()

        q_cur = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.online(ns).argmax(1, keepdim=True)
            q_next = self.target(ns).gather(1, best_a).squeeze(1)
        q_tgt = r + self.hparams["gamma"] * q_next * (1 - d)

        loss = self.loss_fn(q_cur, q_tgt)
        self.log("train/loss", loss, prog_bar=True)
        self.all_losses.append(loss.item())
        self._polyak_update()
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.online.parameters(), lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams["episodes"], eta_min=1e-5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                     gradient_clip_algorithm=None):
        self.clip_gradients(optimizer,
                            gradient_clip_val=self.hparams["grad_clip"],
                            gradient_clip_algorithm="norm")

    def train_dataloader(self):
        self.play_episode()
        dataset = RLDataset(self.buffer,
                            self.hparams["batch_size"],
                            self.hparams["grad_steps_ep"])
        return DataLoader(dataset, batch_size=self.hparams["batch_size"])

    def on_train_epoch_end(self):
        if self.all_rewards and len(self.all_rewards) % 200 == 0:
            avg = np.mean(self.all_rewards[-100:])
            print(f"Ep {self.episode:4d} | avg_r={avg:6.2f} | "
                  f"eps={self.epsilon:.3f}")
        if self._should_stop:
            print(f"Early stop at episode {self.episode}.")
            self.trainer.should_stop = True


# ── Trainer ───────────────────────────────────────────────────────────────────
hparams = dict(
    episodes       = EPISODES,
    max_steps      = MAX_STEPS,
    gamma          = GAMMA,
    lr             = LR,
    batch_size     = BATCH_SIZE,
    buffer_size    = BUFFER_SIZE,
    warmup_steps   = WARMUP_STEPS,
    eps_start      = EPS_START,
    eps_end        = EPS_END,
    eps_decay      = EPS_DECAY,
    tau            = TAU,
    grad_clip      = GRAD_CLIP,
    gif_every      = GIF_EVERY,
    grad_steps_ep  = GRAD_STEPS_EP,
    shaping_scale  = SHAPING_SCALE,
    out_dir        = OUT_DIR,
)

model = DQNLightning(hparams)

trainer = pl.Trainer(
    max_epochs                        = EPISODES,
    accelerator                       = "gpu" if torch.cuda.is_available() else "cpu",
    devices                           = 1,
    enable_checkpointing              = False,
    logger                            = False,
    enable_progress_bar               = True,
    log_every_n_steps                 = 1,
    reload_dataloaders_every_n_epochs = 1,   # call train_dataloader() every epoch so play_episode() runs each episode
)
trainer.fit(model)

torch.save(model.online.state_dict(), os.path.join(OUT_DIR, "lightning_dqn.pt"))
plot_metrics(model.all_rewards, model.all_losses,
             "Dueling Double DQN (Lightning) – Random Mode",
             os.path.join(OUT_DIR, "metrics.png"))
print(f"Total episodes: {len(model.all_rewards)}")
print(f"Outputs in {OUT_DIR}/")
