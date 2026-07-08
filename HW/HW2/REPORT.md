# [2026_DS_RL] HW2 과제 보고서 — PPO 구현

## 1. REINFORCE 알고리즘에서 수정한 내용과 그 이유

기본 제공 코드는 연속 행동 공간에서 REINFORCE를 사용하는 구조이다. REINFORCE는 구현이 단순하지만 episode가 끝난 뒤 Monte-Carlo return만으로 policy를 업데이트하기 때문에 분산이 크고, sparse reward 환경에서는 학습이 불안정할 수 있다. 본 과제에서는 Policy Gradient 계열 알고리즘인 **PPO(Proximal Policy Optimization)**를 PyTorch로 직접 구현하여 적용하였다.

주요 수정은 아래와 같다.

### 1) Actor-Critic 구조 및 GAE (Generalized Advantage Estimation)
- **이유**: 순수 REINFORCE의 Monte-Carlo return은 분산이 커서 학습이 불안정하다.
- **수정**: Value Network(Critic)를 추가하여 Actor-Critic 구조로 변경하고, GAE(γ=0.98, λ=0.95)로 Advantage를 계산하였다. Critic이 제공하는 baseline은 Advantage의 분산을 줄여 policy gradient의 안정성을 높인다.

### 2) PPO Clipped Surrogate Objective
- **이유**: Policy gradient는 step size에 민감하여, 큰 update로 인해 policy가 붕괴(policy collapse)될 수 있다.
- **수정**: PPO의 clipped surrogate objective를 적용하여, ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 가 $[1-\epsilon, 1+\epsilon]$ 범위를 벗어나지 않도록 clipping(ε=0.2) 하였다. 이를 통해 기존 policy에서 너무 멀리 이동하는 update를 제한한다.

### 3) Reward Scaling
- **이유**: 환경 reward는 -1(이동), -100(trap), +100(goal)로 스케일 차이가 크다. 이 상태로 value loss를 계산하면 numerical instability가 발생할 수 있다.
- **수정**: 에이전트 내부에서 reward에 scale factor 0.01을 곱하여 GAE advantage 계산과 value target의 numerical scale을 안정화하였다. 환경 reward 자체는 수정하지 않았다.

### 4) Observation Normalization (RunningMeanStd)
- **이유**: 상태 벡터에는 로봇 위치(0~7)와 ray 센서 값(0~1)이 섞여 있어 스케일 차이가 발생한다.
- **수정**: Welford 알고리즘 기반 RunningMeanStd로 각 상태 차원을 online으로 정규화하여 신경망 입력 스케일을 통일하였다.

### 5) Entropy Regularization 및 Annealing
- **이유**: 연속 행동 공간에서 조기 수렴(local optima)을 방지하고 충분한 탐험을 유지해야 한다.
- **수정**: Policy 분포의 entropy에 비례하는 보너스를 loss에 추가하였다. 학습 초기에는 entropy 계수를 0.1로 높여 탐험을 장려하고, 학습 후반에는 0.01로 선형 감소시켜 exploitation으로 전환하였다.

### 6) 그 외 안정화 기법
- **Gradient Clipping** (max_norm=0.5): 큰 gradient로 인한 불안정을 방지
- **Cosine Annealing LR** (T_max=200000, eta_min=lr×0.01): 학습률을 점진적으로 감소
- **단일 Optimizer**: Actor와 Critic 파라미터를 하나의 Adam optimizer로 공동 학습

---

## 2. 학습 전략

### Curriculum Learning (Map 1, Map 2)
- **Phase 1 (0~40%)**: BFS로 계산한 goal까지의 거리 기반으로, 가까운 셀부터 먼 셀까지 점진적으로 시작 위치를 확장. 함정 인접 셀은 제외.
- **Phase 2 (40~100%)**: 50%는 공식 시작점(0,0)에서 deterministic evaluation, 50%는 BFS reachable 셀에서 training. Policy generalization을 통해 전체 경로 학습.

### Transfer Learning (Map 3)
- Map 3는 goal 주변이 함정으로 둘러싸여 있어 curriculum learning이 적용되기 어려웠다.
- Map 2의 best checkpoint를 pretrained weight로 불러와 goal-seeking 행동을 transfer하였다.
- `--no-curriculum` 으로 처음부터 (0,0)에서만 학습하며, lr=1e-4, γ=0.99로 보수적으로 학습.

### Early Stopping
- 최근 100회의 evaluation episode 이동평균 success rate가 95% 이상을 50회 연속 달성하면 조기 종료.

---

## 3. 실험 결과 (100회 평가)

| 맵 | 난이도 | 평가 시도 | 성공 | 실패 | 성공률 | 학습 소요 (episodes) |
|---|---|---|---|---|---|---|
| hw_map1 | 하 | 100 | 100 | 0 | **100%** | ~3,100 (early stop) |
| hw_map2 | 중 | 100 | 100 | 0 | **100%** | ~10,000 (best SR 96%) |
| hw_map3 | 상 | 100 | 100 | 0 | **100%** | ~7,900 (early stop, transfer) |

모든 맵에서 결정론적 평가(deterministic evaluation, eval_mode=True)로 100회 시도 결과 **단 한 번의 실패도 없이 100% 성공률**을 달성하였다.

- **hw_map1**: 직선 경로의 단순한 장애물 회피만 필요하여 약 3,100 episode에서 early stop.
- **hw_map2**: 벽 우회가 필요하지만, curriculum phase에서 goal-seeking skill을 습득한 후 mixed phase에서 generalization이 이루어져 최종 100% 성공.
- **hw_map3**: 가장 난이도가 높은 맵으로, transfer learning + no curriculum + 저learning rate 전략으로 약 7,900 episode에서 early stop.

---

## 4. 주요 하이퍼파라미터

```python
# algos/reinforce.py — REINFORCEAgent (PPO 구현)
lr = 3e-4                # map1, map2; map3는 1e-4
gamma = 0.98             # map1, map2; map3는 0.99
gae_lambda = 0.95
clip_epsilon = 0.2       # PPO clipping 범위 (map3는 0.15)
reward_scale = 0.01      # 내부 reward scaling
entropy_coeff_start = 0.1
entropy_coeff_end = 0.01
value_loss_coeff = 0.5
max_grad_norm = 0.5
log_std_init = 0.5       # 초기 std ≈ exp(0.5) ≈ 1.65
total_episodes = 200000  # scheduler 기준 (실제 episode보다 크게 설정)
```

---

## 5. 알고리즘의 한계점

- **On-policy 특성**: PPO는 on-policy 알고리즘이므로, 매 episode의 데이터를 한 번만 사용한다. Off-policy 알고리즘 대비 샘플 효율이 낮다.
- **연속 행동 공간의 미세 제어**: 좁은 통로나 함정이 밀집된 영역에서는 continuous action의 미세한 편차도 실패로 이어질 수 있다. 결정론적 평가(eval_mode=True)에서만 안정적인 성능을 보인다.
- **Map 3의 사전학습 의존성**: 가장 어려운 map3는 map2의 사전학습 가중치 없이는 수렴이 어려웠다. 완전히 scratch부터 학습하려면 더 많은 episode와 탐험 전략이 필요할 것으로 보인다.

---

## 6. 실행 방법

```bash
# 학습 (rl conda 환경 사용)
python train_r.py --map hw_map1.yaml --episodes 5000 --max-steps 150
python train_r.py --map hw_map2.yaml --episodes 10000 --max-steps 200

# map3 (transfer learning)
python train_r.py --map hw_map3.yaml --episodes 10000 --max-steps 250 \
    --lr 1e-4 --gamma 0.99 --no-curriculum \
    --pretrained checkpoints/reinforce_hw_map2_best.pth

# 평가
python test_r.py --map hw_map1.yaml --attempts 100 --headless
python test_r.py --map hw_map2.yaml --attempts 100 --headless
python test_r.py --map hw_map3.yaml --attempts 100 --headless
```

---

## 7. 파일 구조

```
HW2/
├── algos/
│   └── reinforce.py       # PPO 구현 (수정)
├── configs/
│   ├── hw_map1.yaml
│   ├── hw_map2.yaml
│   └── hw_map3.yaml
├── env/
│   └── gridworld_c2.py    # 환경 (수정 불가)
├── train_r.py             # 학습 루프 (수정)
├── test_r.py              # 평가 스크립트 (수정 없음)
└── checkpoints/
    ├── reinforce_hw_map1.pth / _best.pth
    ├── reinforce_hw_map2.pth / _best.pth
    └── reinforce_hw_map3.pth / _best.pth
```
