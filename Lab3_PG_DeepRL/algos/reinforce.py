import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from env.gridworld_c2 import GridWorldEnv_c2
from algos.utils import *

class PolicyNetwork(nn.Module):
	def __init__(self, state_dim=2, action_dim=2, hidden_dim=256):
		super(PolicyNetwork, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU()
		)
		self.mean_layer = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Parameter(torch.zeros(action_dim))
		self._initialize_weights()

	def forward(self, state):
		x = self.net(state)
		mean = self.mean_layer(x)
		std = torch.exp(self.log_std) + 1e-6
		return mean, std

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		
		nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
		nn.init.constant_(self.mean_layer.bias, 0)
		nn.init.constant_(self.log_std, -0.5)


class REINFORCEAgent:
	def __init__(self,
				 env: GridWorldEnv_c2,
				 lr=1e-3, 
				 gamma=0.995,
				 entropy_coeff=0.001, 
				 max_grad_norm=0.1): 
		self.env = env
		self.device = torch.device('cpu')
		self.policy = PolicyNetwork().to(self.device)
		self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
		self.gamma = gamma
		self.entropy_coeff = entropy_coeff
		self.max_grad_norm = max_grad_norm

		self.log_probs = []
		self.rewards = []
		self.entropies = []

	def reset_episode(self):
		self.log_probs = []
		self.rewards = []
		self.entropies = []

	def select_action(self, state, eval=False):
		state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		mean, std = self.policy(state_v)
		
		if eval:
			action = torch.clamp(mean, -1.0, 1.0)
			return action.cpu().detach().numpy()[0]

		dist = Normal(mean, std)
		action = dist.rsample()
		action = torch.clamp(action, -1.0, 1.0)

		log_prob = dist.log_prob(action).sum(dim=-1)
		entropy = dist.entropy().sum(dim=-1)

		self.log_probs.append(log_prob)
		self.entropies.append(entropy)
		return action.cpu().detach().numpy()[0]

	def finish_episode(self):
		# compute discounted returns
		returns = []
		G = 0
		for r in reversed(self.rewards):
			G = r + self.gamma * G
			returns.insert(0, G)
		returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
		
		if len(returns) > 1 and returns.std() > 1e-3:
			returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-9)

		# policy loss
		policy_losses = []
		for log_prob, Gt in zip(self.log_probs, returns):
			policy_losses.append(-log_prob * Gt)
		
		# entropy bonus
		entropy_bonus = torch.stack(self.entropies).sum()
		entropy_loss = -self.entropy_coeff * entropy_bonus

		loss = torch.stack(policy_losses).sum() + entropy_loss

		# update policy with gradient clipping
		self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
		self.optimizer.step()

		return loss.item()

	def save(self, path):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torch.save({
			'policy_state_dict': self.policy.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
		}, path)

	def load(self, path):
		checkpoint = torch.load(path, map_location=self.device)
		self.policy.load_state_dict(checkpoint['policy_state_dict'])

	def inference(self, state):
		return self.select_action(state, eval=True)


class REINFORCETrainer:
	def __init__(self, args):
		self.args = args
		self.save_best_model = False
		self.set_seeds()


	def initialize(self):
		if not os.path.exists(self.args.logdir):
			os.makedirs(self.args.logdir, exist_ok=True)

		self.writer = SummaryWriter(log_dir=os.path.join(self.args.logdir, self.args.algo, self.args.save_name))
		config_path = os.path.join('configs', f'{self.args.map}.yaml')
		self.env = GridWorldEnv_c2(config_path)
		self.agent = REINFORCEAgent(self.env)


	def set_seeds(self):
		seed = self.args.seed
		torch.manual_seed(seed)
		np.random.seed(seed)


	def train(self):
		map_img, H, W = log_map(self.writer, self.env)
		best_reward = float('-inf')

		for ep in range(1, self.args.episodes+1):
			state = self.env.reset()
			self.agent.reset_episode()
			total_R = 0.0

			if hasattr(self.agent,'epsilon'):
				self.writer.add_scalar('Epsilon', self.agent.epsilon, ep)

			# episode rollout
			action = self.agent.select_action(state)
			for t in range(self.args.max_steps):
				next_s, r, done, _ = self.env.step(action)
				total_R += r
				self.agent.rewards.append(r)
				next_a = self.agent.select_action(next_s)
				state, action = next_s, next_a
				if self.args.render: self.env.render(tick=5000)
				if done: break

			# finish episode and update policy
			loss = self.agent.finish_episode()
			self.writer.add_scalar('Loss', loss, ep)
			self.writer.add_scalar('Reward', total_R, ep)

			# save best model (optional)
			if self.save_best_model and total_R > best_reward:
				best_reward = total_R
				self.agent.save(f'checkpoints/{self.args.algo}_{self.args.save_name}_best.pth')

			# periodic visualization
			if ep % self.args.heatmap_interval==0:
				plot_continuous_policy(self.writer, self.env, self.agent,
									map_img, H, W,
									self.args.resolution, ep)

			if ep % 100 == 0:
				disp = f", Epsilon: {self.agent.epsilon:.3f}" if hasattr(self.agent,'epsilon') else ''
				print(f"[REINFORCE] Episode: {ep}, Reward: {total_R:.2f}{disp}")

	def save(self):
		os.makedirs('checkpoints', exist_ok=True)
		self.agent.save(f'checkpoints/{self.args.algo}_{self.args.save_name}.pth')
		self.writer.close()
