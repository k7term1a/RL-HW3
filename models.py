"""
Neural-network models for all DQN variants.

NaiveDQN        – plain MLP (HW3-1)
DuelingDQN      – dueling streams (HW3-2 / HW3-3)
NoisyLinear     – factorised noisy linear layer (Rainbow)
RainbowDQN      – Dueling + Noisy + C51 distributional heads (HW3-4)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class NaiveDQN(nn.Module):
    def __init__(self, state_dim: int = 64, n_actions: int = 4, hidden: int = 150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
class DuelingDQN(nn.Module):
    """Dueling architecture: shared body → value stream + advantage stream."""

    def __init__(self, state_dim: int = 64, n_actions: int = 4, hidden: int = 150):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        feat = self.body(x)
        val  = self.value_stream(feat)          # (B, 1)
        adv  = self.adv_stream(feat)            # (B, A)
        # Q = V + (A - mean(A))
        return val + adv - adv.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """Factorised Gaussian Noisy Linear (Fortunato et al., 2017)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self._reset_parameters()
        self.sample_noise()

    def _reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu,   -bound, bound)
        nn.init.uniform_(self.bias_mu,     -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma,   self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_i = self._f(torch.randn(self.in_features))
        eps_j = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_j.outer(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
class RainbowDQN(nn.Module):
    """
    Rainbow = Dueling + Noisy Nets + C51 (distributional).
    n_atoms atoms over [v_min, v_max].
    """

    def __init__(
        self,
        state_dim:  int   = 64,
        n_actions:  int   = 4,
        hidden:     int   = 256,
        n_atoms:    int   = 51,
        v_min:      float = -10.0,
        v_max:      float = 10.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),   # deterministic body per Rainbow paper; only output heads are noisy
            nn.ReLU(),
        )
        self.value_hidden = NoisyLinear(hidden, hidden)
        self.value_out    = NoisyLinear(hidden, n_atoms)

        self.adv_hidden = NoisyLinear(hidden, hidden)
        self.adv_out    = NoisyLinear(hidden, n_actions * n_atoms)

    def forward(self, x) -> torch.Tensor:
        """Returns log-probabilities, shape (B, A, n_atoms)."""
        feat = self.body(x)

        val = F.relu(self.value_hidden(feat))
        val = self.value_out(val).view(-1, 1, self.n_atoms)       # (B,1,Z)

        adv = F.relu(self.adv_hidden(feat))
        adv = self.adv_out(adv).view(-1, self.n_actions, self.n_atoms)  # (B,A,Z)

        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        return F.log_softmax(q_atoms, dim=2)

    def q_values(self, x) -> torch.Tensor:
        """Expected Q-values for greedy action selection."""
        log_p = self.forward(x)
        return (log_p.exp() * self.support.unsqueeze(0).unsqueeze(0)).sum(2)

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()
