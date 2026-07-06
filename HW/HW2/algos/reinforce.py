import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """
    정책 신경망 (Actor): 주어진 상태(state)를 입력받아
    행동(action)의 평균(mean)과 표준편차(std)를 출력합니다.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        # 은닉층 구성: state_dim -> hidden_dim -> hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 출력층: hidden_dim -> action_dim (평균)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # 행동 분포의 로그 표준편차를 학습 가능한 파라미터로 선언
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수:
        state shape: [batch_size, state_dim]
        반환: mean shape [batch_size, action_dim], std shape [action_dim]
        """
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)  # 양수 표준편차
        return mean, std


class ValueNetwork(nn.Module):
    """
    가치 신경망 (Critic): 주어진 상태(state)를 입력받아
    스칼라 상태 가치(V(s))를 출력합니다.
    """
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
        """
        순전파 함수:
        state shape: [batch_size, state_dim]
        반환: value shape [batch_size, 1]
        """
        return self.net(state)


class REINFORCEAgent:
    """
    Actor-Critic (GAE) 에이전트:
    - Actor (PolicyNetwork): 정책 학습
    - Critic (ValueNetwork): 상태 가치 추정 (GAE 기반 advantage 계산)
    - 에피소드별 전이(transition) 버퍼를 유지
    - 에피소드 종료 시 GAE로 advantage를 계산하고 Actor/Critic 동시 업데이트

    기존 REINFORCE 인터페이스와 완전 호환:
    - select_action(state_tensor) -> action_tensor
    - finish_episode() -> loss (float)
    - save(path) / load(path)
    - inference(state_tensor) -> action_tensor
    - agent.rewards 리스트에 보상 추가 방식 유지
    - agent.policy 속성으로 PolicyNetwork 접근 가능 (시각화 호환)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        total_episodes: int = 200000,
        device: torch.device = torch.device('cpu')
    ):
        """
        초기화 파라미터:
        state_dim: 상태 차원 수
        action_dim: 행동 차원 수
        lr: 학습률 (Actor/Critic 공통)
        gamma: 할인율
        gae_lambda: GAE λ 파라미터 (0=TD(0), 1=Monte-Carlo)
        entropy_coeff: 엔트로피 정규화 계수 (탐색 장려)
        value_loss_coeff: Critic 손실 가중치
        max_grad_norm: Gradient clipping 최대 norm
        total_episodes: 총 에피소드 수 (LR 스케줄러 주기)
        device: 연산 디바이스 ("cpu" 또는 "cuda")
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm

        # Actor (정책 네트워크) 생성 및 디바이스 할당
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        # Critic (가치 네트워크) 생성 및 디바이스 할당
        self.value_net = ValueNetwork(state_dim).to(self.device)

        # Actor, Critic 파라미터를 하나의 옵티마이저로 관리
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )

        # 학습률 스케줄러: Cosine Annealing (lr → eta_min으로 점진 감소)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_episodes, eta_min=lr * 0.01
        )

        # 에피소드별 버퍼 초기화
        self.log_probs: list[torch.Tensor] = []   # 행동 선택 시 log_prob 저장
        self.rewards: list[float] = []             # 에피소드 보상 저장
        self.values: list[torch.Tensor] = []       # Critic이 추정한 V(s_t)
        self.entropies: list[torch.Tensor] = []    # 정책 분포의 엔트로피
        self.states: list[torch.Tensor] = []       # 상태 텐서 저장 (GAE 계산용)

        # 마지막 상태에 대한 done 플래그 (GAE bootstrap 용)
        self._last_done: bool = False

        # 디버깅/로깅용: 마지막 에피소드의 Actor/Critic loss
        self.last_actor_loss: float = 0.0
        self.last_critic_loss: float = 0.0
        self.last_entropy: float = 0.0

    def reset_episode(self) -> None:
        """새 에피소드 시작 시 버퍼 비우기"""
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.entropies.clear()
        self.states.clear()
        self._last_done = False

    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """
        상태(state)를 받아 행동(action)을 샘플링하거나 선택합니다.
        eval_mode=True일 경우 평균 행동(mean)을 반환합니다.

        Args:
            state: torch.Tensor, shape [state_dim]
            eval_mode: bool, 평가 모드 여부
        Returns:
            action: torch.Tensor, shape [action_dim]
        """
        state = state.to(self.device).unsqueeze(0)  # 배치 차원 추가
        mean, std = self.policy(state)
        if eval_mode:
            # 평가 시에는 평균 행동 사용 (클램프로 범위 제한)
            action = torch.clamp(mean, -1.0, 1.0)
            return action.squeeze(0).cpu()

        # Critic으로 현재 상태 가치 추정
        value = self.value_net(state)
        self.values.append(value.squeeze())
        self.states.append(state.squeeze(0))

        # 확률 분포 생성 및 score-function estimator용 샘플링
        dist = Normal(mean, std)
        action = dist.sample()

        # log_prob 저장
        log_prob = dist.log_prob(action).sum(dim=-1)
        self.log_probs.append(log_prob)

        # 엔트로피 저장 (탐색 장려용)
        entropy = dist.entropy().sum(dim=-1)
        self.entropies.append(entropy)

        # 실제 환경에 보낼 행동값은 클램프 처리
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).cpu()

    def set_done(self, done: bool) -> None:
        """에피소드 종료 여부 설정 (GAE bootstrap 결정에 사용)"""
        self._last_done = done

    def _compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GAE (Generalized Advantage Estimation) 계산:

        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_l

        Returns:
            advantages: shape [T], GAE advantage 값
            returns: shape [T], advantage + value (Critic 학습 타깃)
        """
        T = len(self.rewards)

        # 마지막 상태의 부트스트랩 가치 결정
        if self._last_done:
            next_value = torch.tensor(0.0, device=self.device)
        else:
            # 에피소드가 truncation으로 끝난 경우 마지막 상태의 V 추정
            with torch.no_grad():
                last_state = self.states[-1].unsqueeze(0)
                next_value = self.value_net(last_state).squeeze()

        # GAE 역방향 계산
        advantages = torch.zeros(T, device=self.device)
        gae = torch.tensor(0.0, device=self.device)

        values_tensor = torch.stack([v.detach() for v in self.values])
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values_tensor[t + 1]

            delta = rewards_tensor[t] + self.gamma * next_val - values_tensor[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        # Critic 학습 타깃: returns = advantages + V(s)
        returns = advantages + values_tensor
        return advantages, returns

    def finish_episode(self) -> float:
        """
        에피소드가 종료된 후 호출하여:
        1) GAE로 advantage 및 returns 계산
        2) Advantage 정규화
        3) Actor 손실: -log_prob * advantage - entropy_coeff * entropy
        4) Critic 손실: MSE(V(s), returns)
        5) 결합 손실로 역전파 및 gradient clipping
        6) 버퍼 초기화

        Returns:
            total_loss.item(): 업데이트 후 총 손실 값 (float)
        """
        if len(self.rewards) == 0:
            self.reset_episode()
            return 0.0

        # 1) GAE 계산
        advantages, returns = self._compute_gae()

        # 2) Advantage 정규화
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3) Actor 손실: policy gradient with GAE advantage
        log_probs_tensor = torch.stack(self.log_probs).view(-1)
        entropies_tensor = torch.stack(self.entropies).view(-1)

        actor_loss = -(log_probs_tensor * advantages.detach()).mean()
        entropy_bonus = -self.entropy_coeff * entropies_tensor.mean()

        # 4) Critic 손실: MSE loss
        values_tensor = torch.stack(self.values).view(-1)
        critic_loss = nn.functional.mse_loss(values_tensor, returns.detach())

        # 5) 총 손실
        total_loss = actor_loss + entropy_bonus + self.value_loss_coeff * critic_loss

        # 6) 역전파 및 gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        # 디버깅/로깅용 기록
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_entropy = entropies_tensor.mean().item()

        # 7) 학습률 스케줄러 스텝
        self.scheduler.step()

        # 8) 다음 에피소드 준비를 위한 버퍼 초기화
        self.reset_episode()

        return total_loss.item()

    def get_lr(self) -> float:
        """현재 학습률 반환 (로깅용)"""
        return self.optimizer.param_groups[0]['lr']

    def save(self, path: str) -> None:
        """Actor 및 Critic 네트워크 파라미터를 파일에 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """
        저장된 파라미터를 불러와 네트워크에 로드.
        기존 REINFORCE 형식 (policy state_dict만 저장)과도 호환됩니다.
        """
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            # 새 형식: {'policy': ..., 'value_net': ...}
            self.policy.load_state_dict(checkpoint['policy'])
            if 'value_net' in checkpoint:
                self.value_net.load_state_dict(checkpoint['value_net'])
        else:
            # 기존 형식: policy state_dict만 저장된 경우
            self.policy.load_state_dict(checkpoint)

    def inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        평가 모드에서 행동(action)을 반환합니다.
        내부적으로 select_action(eval_mode=True) 호출
        """
        return self.select_action(state, eval_mode=True)
