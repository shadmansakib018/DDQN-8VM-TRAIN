import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import threading
import os
import subprocess


class ActorCritic(nn.Module):
    """
    Single network with two heads:
      - Actor: outputs action logits (replaces Q-value output)
      - Critic: outputs scalar state value V(s) (new — DDQN had no critic)
    Shared backbone keeps parameter count reasonable.
    """
    def __init__(self, input_dim=21, output_dim=8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)       # logits, not Q-values
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)                # scalar V(s)
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)


class PPOAgent:
    def __init__(
        self,
        path,
        state_dim=21,
        action_dim=8,
        gamma=0.99,
        lr=3e-4,
        clip_eps=0.2,          # PPO clipping range
        epochs_per_train=10,   # PPO reuses each batch multiple times
        gae_lambda=0.95,       # GAE smoothing factor
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs_per_train = epochs_per_train
        self.gae_lambda = gae_lambda

        self.lock = threading.Lock()
        self.train_lock = threading.Lock()

        # ── Trajectory buffer (replaces replay buffer) ──────────────────────
        # PPO is ON-POLICY: we collect a fresh batch each episode,
        # train on it, then throw it away. No random sampling from history.
        self.states    = []
        self.actions   = []
        self.log_probs = []   # NEW: needed for importance ratio r_t = π_new/π_old
        self.rewards   = []
        self.values    = []   # NEW: critic estimates, needed for GAE

        self.total_simulations = 0
        self.reward_summation  = 0
        self.checkpoint_interval = 100
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Single network; no target network needed (PPO doesn't use one)
        self.ac_net   = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        self.load_checkpoint(path)

    # ── Experience storage ───────────────────────────────────────────────────
    def remember(self, state, action, reward, next_state, log_prob, value, source_id):
        """
        Two new required fields vs DDQN:
          log_prob – log π(a|s) at decision time, for computing ratio r_t
          value    – V(s) from critic at decision time, for GAE
        next_state is kept for API compatibility but PPO doesn't need it.
        """
        with self.lock:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.values.append(value)
        self.reward_summation += reward

    # ── GAE computation ──────────────────────────────────────────────────────
    def _compute_gae(self, rewards, values):
        """
        Generalized Advantage Estimation — smoother than plain TD or MC.
        λ=0 → pure TD(0); λ=1 → pure Monte Carlo.
        """
        advantages = []
        gae = 0
        # Bootstrap last value as 0 (episode ended)
        extended_values = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * extended_values[t + 1] - extended_values[t]
            gae   = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    # ── Training ─────────────────────────────────────────────────────────────
    def train(self, source_id):
        with self.train_lock:
            # Drain trajectory buffer
            with self.lock:
                if not self.states:
                    return
                states    = list(self.states);    self.states.clear()
                actions   = list(self.actions);   self.actions.clear()
                log_probs = list(self.log_probs); self.log_probs.clear()
                rewards   = list(self.rewards);   self.rewards.clear()
                values    = list(self.values);    self.values.clear()

            advantages, returns = self._compute_gae(rewards, values)

            # Move everything to tensors once
            s_t    = torch.FloatTensor(states).to(self.device)
            a_t    = torch.LongTensor(actions).to(self.device)
            olp_t  = torch.FloatTensor(log_probs).to(self.device)
            adv_t  = torch.FloatTensor(advantages).to(self.device)
            ret_t  = torch.FloatTensor(returns).to(self.device)

            # Normalize advantages (stabilizes training)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            epoch_losses = 0.0
            for _ in range(self.epochs_per_train):
                logits, values_pred = self.ac_net(s_t)
                dist           = Categorical(logits=logits)
                new_log_probs  = dist.log_prob(a_t)
                entropy        = dist.entropy().mean()

                # ── PPO clipped objective ────────────────────────────────
                ratio    = (new_log_probs - olp_t).exp()          # π_new / π_old
                clipped  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss  = -torch.min(ratio * adv_t, clipped * adv_t).mean()

                # Critic loss (same MSE you used before)
                critic_loss = nn.MSELoss()(values_pred.squeeze(), ret_t)

                # Entropy bonus encourages exploration (replaces ε-greedy)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                epoch_losses += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.total_simulations += 1
            print("SIMULATION NUMBER:", self.total_simulations)

            if self.total_simulations >= 300 and self.total_simulations % self.checkpoint_interval == 0:
                self.save_checkpoint()

            with open("loss_log.txt", "a") as f:
                f.write(f"{epoch_losses:.4f},{self.reward_summation:.2f}\n")
            self.reward_summation = 0

    # ── Action selection ─────────────────────────────────────────────────────
    def act(self, state):
        """
        Returns (action, log_prob, value) — not just action like DDQN.
        log_prob and value must be sent back by Java when storing experience.
        No epsilon-greedy: stochasticity comes from the distribution itself.
        """
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.ac_net(s_t)
            dist      = Categorical(logits=logits)
            action    = dist.sample()
            log_prob  = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.total_simulations}.pth")
        torch.save(self.ac_net.state_dict(), path)
        print(f"[Checkpoint] Saved to {path}")
        try:
            subprocess.Popen(["python", "./validation/validate.py", path])
        except Exception as e:
            print(f"[Validation Error] {e}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        model_state = self.ac_net.state_dict()
        # Only transfer layers whose name AND shape match (handles 10→8 VM action dim change)
        compatible = {k: v for k, v in ckpt.items()
                      if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in ckpt if k not in compatible]
        self.ac_net.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"[Checkpoint] Skipped (shape mismatch, re-initialised): {skipped}")
        print(f"[Checkpoint] Loaded {len(compatible)}/{len(ckpt)} layers from {path}")