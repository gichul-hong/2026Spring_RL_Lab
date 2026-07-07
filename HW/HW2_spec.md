# [2026 DS-RL] HW2 — 구현 요청 스펙 문서

> **목적**: 이 문서는 Opus/Fable 등 고성능 LLM에 구현을 요청할 때 사용하는 정제된 컨텍스트입니다.  
> **제출 마감**: 2026년 7월 17일 (금) 23:59

---

## 1. 과제 목표

`REINFORCE` 기반 Policy Gradient 알고리즘을 수정·개선하여,  
`hw_map1 / hw_map2 / hw_map3` 세 개의 Continuous-Action GridWorld 맵에서 **높은 성공률**을 달성한다.

### 채점 기준

| 맵 | 난이도 | 배점 | 5회↑ 성공 (50%) | 8회↑ 성공 (100%) |
|----|--------|------|-----------------|-----------------|
| hw_map1 | 하 | 15점 | 7.5점 | 15점 |
| hw_map2 | 중 | 25점 | 12.5점 | 25점 |
| hw_map3 | 상 | 30점 | 15점 | 30점 |

- 조교 컴퓨터에서 각 맵 **10회 실행**으로 평가
- 5회 미만: 코드 내용 기반 정성 평가

---

## 2. 실험 제약 조건

### ✅ 변경 가능
- 하이퍼파라미터 (learning rate, discount factor 등)
- 네트워크 크기 및 레이어 수 (과도한 크기 비권장)
- **Policy Gradient 기반 알고리즘** 구현 및 적용 가능
  - 예: Actor-Critic, N-step returns, TD(λ), GAE, PPO 등

### ❌ 변경 불가
- 외부 RL 라이브러리 사용 금지 (e.g., stable-baselines3)
- 환경 요소 수정 금지 (step size, reward 등)

### 실행 제약
```
max_episodes: 200,000 (상한, 이전에 수렴해도 됨)
max_steps:
  hw_map1: 150
  hw_map2: 200
  hw_map3: 250
```

---

## 3. 환경 명세 (`env/gridworld_c2.py`)

### 상태 공간 (state_dim = 10)
```
[row_m, col_m, ray_0, ray_1, ..., ray_7]
 └─ 연속 위치(m)    └─ 8방향 ray-sensor 거리 (최대 1.0m)
```

### 행동 공간 (action_dim = 2)
```
action ∈ [-1, 1]^2  (연속, 내부적으로 clamp)
```

### 보상 구조
| 이벤트 | 보상 | 종료 여부 |
|--------|------|-----------|
| 이동 | -1 | 계속 |
| 트랩 도달 | -100 | ✅ 종료 |
| 목표 도달 | +100 | ✅ 종료 |

### 맵 셀 타입
- `0`: Normal, `1`: Wall, `2`: Trap, `3`: Goal

---

## 4. 기존 코드베이스 분석

### 파일 구조
```
HW/HW2/
├── algos/reinforce.py     ← 핵심 수정 대상
├── env/gridworld_c2.py    ← 수정 불가 (환경)
├── configs/
│   ├── hw_map1.yaml
│   ├── hw_map2.yaml
│   └── hw_map3.yaml
├── train_r.py             ← 학습 루프 (수정 가능)
└── test_r.py              ← 평가 스크립트
```

### 현재 구현 상태 (`algos/reinforce.py`)

**PolicyNetwork** (이미 구현됨)
```python
# state_dim → 256 → 256 → action_dim
# mean_layer: Linear(256, 2)
# log_std: nn.Parameter(zeros(2))  # 학습 가능한 log std
# forward() → (mean, std)
```

**REINFORCEAgent** (이미 구현됨)
```python
# select_action(): Normal(mean, std) 에서 샘플링, log_prob 저장
# finish_episode():
#   1) 할인 누적 보상 G_t 계산 (gamma=0.99)
#   2) 보상 정규화 (mean/std)
#   3) policy_loss = -sum(log_prob * G_t)
#   4) Adam optimizer (lr=1e-3)
# save() / load() / inference()
```

**train_r.py** (이미 구현됨)
- TensorBoard 로깅 (Reward, Loss, PolicyArrows)
- 체크포인트 저장: `checkpoints/reinforce_{map_name}.pth`

---

## 5. 기존 구현의 한계 및 개선 포인트

기본 REINFORCE의 알려진 한계:
1. **높은 분산**: Monte-Carlo return은 분산이 커서 학습이 불안정
2. **Baseline 없음**: Value function baseline 미적용 → advantage 추정 불안정
3. **On-policy + 데이터 비효율**: 매 에피소드 폐기
4. **log_std 고정 초기화**: 탐색 스케줄 없음

권장 개선 방향 (수업 내용 기반):
- **Baseline with Value Network** (Actor-Critic)
- **N-step returns** 또는 **GAE (Generalized Advantage Estimation)**
- **Entropy regularization** (탐색 장려)
- **학습률 스케줄링**
- **Gradient clipping**

---

## 6. 구현 요청사항

아래 내용을 `algos/reinforce.py` (및 필요 시 `train_r.py`)에 구현해주세요.

### 요청 사항
1. **Actor-Critic 구조**로 확장
   - `PolicyNetwork` (Actor): 기존 유지 또는 개선
   - `ValueNetwork` (Critic): 상태 → 스칼라 가치 추정
   - Advantage = G_t - V(s_t) (또는 GAE)

2. **호환성 유지**
   - `train_r.py`의 학습 루프 인터페이스 최대한 유지
   - `agent.select_action(state_tensor)` 인터페이스 유지
   - `agent.finish_episode()` → loss 반환 유지
   - `agent.save(path)` / `agent.load(path)` 유지

3. **TensorBoard 로깅 확장** (train_r.py)
   - Actor loss, Critic loss 별도 기록

4. **체크포인트 형식**
   - 기존: `torch.save(policy.state_dict(), path)`
   - 확장 시 actor/critic 모두 저장

### 실행 커맨드 (변경 없음)
```bash
# 학습
python train_r.py --map hw_map1.yaml --episodes 200000 --max-steps 150

# 평가
python test_r.py --map hw_map1.yaml --attempts 10 --headless
```

---

## 7. 참고: 토큰 절약 팁

이 문서만으로 구현 요청 시 충분합니다.  
전체 코드를 다시 붙여넣을 필요 없이 아래 파일 경로를 참조하도록 안내하세요:
- `HW/HW2/algos/reinforce.py` (155줄)
- `HW/HW2/train_r.py` (180줄)
- `HW/HW2/env/gridworld_c2.py` (239줄, 수정 불가)
