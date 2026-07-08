import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class RunningMeanStd:
    def __init__(self, shape: int, device: torch.device = torch.device('cpu')):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total_count) / total_count
        self.mean = new_mean
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)

    def state_dict(self) -> dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_init: float = 0.5):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 5,
        batch_size: int = 32,
        entropy_coeff_start: float = 0.05,
        entropy_coeff_end: float = 0.001,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        log_std_init: float = 0.5,
        total_episodes: int = 200000,
        device: torch.device = torch.device('cpu'),
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_end = entropy_coeff_end
        self.entropy_coeff = entropy_coeff_start
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.total_episodes = total_episodes
        self._episode_count = 0

        self.state_normalizer = RunningMeanStd(state_dim, device=device)

        self.policy = PolicyNetwork(state_dim, action_dim,
                                    log_std_init=log_std_init).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_episodes, eta_min=lr * 0.01
        )

        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[torch.Tensor] = []
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

        self._last_done: bool = False

        self.last_actor_loss: float = 0.0
        self.last_critic_loss: float = 0.0
        self.last_entropy: float = 0.0

    def _update_entropy_coeff(self) -> None:
        progress = self._episode_count / max(self.total_episodes, 1)
        if progress <= 0.8:
            self.entropy_coeff = (
                self.entropy_coeff_start
                + (self.entropy_coeff_end - self.entropy_coeff_start) * (progress / 0.8)
            )
        else:
            self.entropy_coeff = self.entropy_coeff_end

    def normalize_state(self, state: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self.state_normalizer.update(state)
        return self.state_normalizer.normalize(state)

    def reset_episode(self) -> None:
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.states.clear()
        self.actions.clear()
        self._last_done = False

    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        if not torch.is_grad_enabled():
            eval_mode = True
        state = state.to(self.device).unsqueeze(0)

        norm_state = self.normalize_state(state, update=not eval_mode)

        mean, std = self.policy(norm_state)
        if eval_mode:
            action = torch.clamp(mean, -1.0, 1.0)
            return action.squeeze(0).cpu()

        value = self.value_net(norm_state)
        self.values.append(value.detach().squeeze())
        self.states.append(norm_state.detach().squeeze(0))

        dist = Normal(mean, std)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        self.log_probs.append(log_prob.detach())

        self.actions.append(action.detach().squeeze(0))

        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).cpu()

    def set_done(self, done: bool) -> None:
        self._last_done = done

    def _compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        T = len(self.rewards)

        if self._last_done:
            next_value = torch.tensor(0.0, device=self.device)
        else:
            with torch.no_grad():
                last_state = self.states[-1].unsqueeze(0)
                next_value = self.value_net(last_state).squeeze()

        advantages = torch.zeros(T, device=self.device)
        gae = torch.tensor(0.0, device=self.device)

        values_tensor = torch.stack(self.values)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values_tensor[t + 1]

            delta = rewards_tensor[t] + self.gamma * next_val - values_tensor[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        returns = advantages + values_tensor
        return advantages, returns

    def finish_episode(self) -> float:
        if len(self.rewards) == 0 or len(self.states) == 0:
            self.reset_episode()
            return 0.0

        advantages, returns = self._compute_gae()

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.stack(self.values)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = len(states)
        effective_batch = min(self.batch_size, n_samples)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _epoch in range(self.ppo_epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, effective_batch):
                batch_idx = indices[start:start + effective_batch]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_values = old_values[batch_idx]

                mean, std = self.policy(batch_states)
                dist = Normal(mean, std)

                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                new_values = self.value_net(batch_states).squeeze(-1)
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (new_values - batch_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy = dist.entropy().sum(dim=-1).mean()

                total_loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.last_actor_loss = total_actor_loss / max(n_updates, 1)
        self.last_critic_loss = total_critic_loss / max(n_updates, 1)
        self.last_entropy = total_entropy / max(n_updates, 1)

        self.scheduler.step()
        self._episode_count += 1
        self._update_entropy_coeff()

        self.reset_episode()

        return self.last_actor_loss + self.last_critic_loss

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                'policy': self.policy.state_dict(),
                'value_net': self.value_net.state_dict(),
                'state_normalizer': self.state_normalizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy'])
            if 'value_net' in checkpoint and checkpoint['value_net'] is not None:
                self.value_net.load_state_dict(checkpoint['value_net'])
            if 'state_normalizer' in checkpoint:
                self.state_normalizer.load_state_dict(checkpoint['state_normalizer'])
        else:
            self.policy.load_state_dict(checkpoint)

    def inference(self, state: torch.Tensor) -> torch.Tensor:
        return self.select_action(state, eval_mode=True)