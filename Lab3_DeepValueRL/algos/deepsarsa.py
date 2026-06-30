import os

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algos.utils import compute_value_grid, log_map, plot_discrete_policy, plot_value_heatmap
from env.gridworld_c1 import GridWorldEnv_c1


class QNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=8, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepSARSATrainer:
    def __init__(self, args):
        self.args = args
        self.set_seeds()

    def set_seeds(self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def initialize(self):
        os.makedirs(self.args.logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.args.logdir, self.args.algo, self.args.save_name))
        config_path = os.path.join('configs', f'{self.args.map}.yaml')
        self.env = GridWorldEnv_c1(config_path, step_size_m=self.args.step_size)
        self.agent = DeepSARSAAgent(self.env)

    def train(self):
        map_img, height, width = log_map(self.writer, self.env)

        for ep in range(1, self.args.episodes + 1):
            state = self.env.reset()
            self.agent.reset_episode()
            total_reward = 0.0

            self.writer.add_scalar('Epsilon', self.agent.epsilon, ep)

            action = self.agent.select_action(state)
            for t in range(self.args.max_steps):
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_action = self.agent.select_action(next_state) if not done else 0

                loss = self.agent.learn(state, action, reward, next_state, next_action, done)
                self.writer.add_scalar('Loss', loss, ep)

                state, action = next_state, next_action
                if self.args.render:
                    self.env.render(tick=5000)
                if done:
                    break

            self.writer.add_scalar('Reward', total_reward, ep)

            if ep % self.args.heatmap_interval == 0:
                values, xs, ys, qvals = compute_value_grid(
                    self.env,
                    self.agent.qnet,
                    self.args.resolution,
                    device=self.agent.device,
                )
                plot_value_heatmap(self.writer, values, xs, ys, ep)
                plot_discrete_policy(self.writer, self.env, qvals, xs, ys, map_img, height, width, ep)

            if ep % 100 == 0:
                print(f'[Deep SARSA] Episode: {ep}, Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}')

    def save(self):
        os.makedirs('checkpoints', exist_ok=True)
        self.agent.save(f'checkpoints/{self.args.algo}_{self.args.save_name}.pth')
        self.writer.close()


class DeepSARSAAgent:
    """
    On-policy deep SARSA agent for the discrete-action continuous GridWorld.
    """
    def __init__(
        self,
        env: GridWorldEnv_c1,
        lr: float = 1e-3,
        gamma: float = 0.995,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.9999,
        max_grad_norm: float = 1.0,
        device: torch.device = None,
    ):
        self.env = env
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.state_scale = np.array([env.height, env.width], dtype=np.float32)
        self.reward_scale = 100.0

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

        self.qnet = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

    def _normalize_state(self, state):
        return np.array(state, dtype=np.float32) / self.state_scale

    def _valid_actions(self, state):
        old_pos = np.array(state, dtype=float)
        valid_actions = []

        for action, delta in enumerate(self.env.deltas):
            lo, hi = 0.0, 1.0
            for _ in range(8):
                mid = (lo + hi) / 2
                test_pos = old_pos + delta * mid
                px = test_pos[1] * self.env.cell_size_px
                py = test_pos[0] * self.env.cell_size_px
                agent_rect = pygame.Rect(
                    px - self.env.agent_radius,
                    py - self.env.agent_radius,
                    self.env.agent_radius * 2,
                    self.env.agent_radius * 2,
                )
                wall_hit = any(agent_rect.colliderect(wall) for wall in self.env.wall_rects)
                out_of_bounds = (
                    agent_rect.left < 0
                    or agent_rect.top < 0
                    or agent_rect.right > self.env.screen_w
                    or agent_rect.bottom > self.env.screen_h
                )

                if wall_hit or out_of_bounds:
                    hi = mid
                else:
                    lo = mid

            if np.linalg.norm(delta * lo) > 1e-6:
                valid_actions.append(action)

        if not valid_actions:
            return np.arange(8, dtype=np.int64)
        return np.array(valid_actions, dtype=np.int64)

    def select_action(self, state, eval=False):
        state_v = torch.tensor(self._normalize_state(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        valid_actions = self._valid_actions(state)

        if not eval and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            q_values = self.qnet(state_v)[0]
            valid_actions_v = torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)
            best_idx = q_values[valid_actions_v].argmax().item()
            return int(valid_actions_v[best_idx].item())

    def learn(self, state, action, reward, next_state, next_action, done):
        state_v = torch.tensor(self._normalize_state(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_v = torch.tensor(self._normalize_state(next_state), dtype=torch.float32, device=self.device).unsqueeze(0)
        scaled_reward = reward / self.reward_scale

        q_value = self.qnet(state_v)[0, action]

        with torch.no_grad():
            if done:
                target = torch.tensor(scaled_reward, dtype=torch.float32, device=self.device)
            else:
                q_next = self.qnet(next_state_v)[0, next_action]
                target = scaled_reward + self.gamma * q_next

        loss = nn.functional.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.total_steps += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()

    def reset_episode(self):
        pass

    def finish_episode(self, episode_idx=None):
        pass

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.qnet.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def inference(self, state):
        return self.select_action(state, eval=True)
