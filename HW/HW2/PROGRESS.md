# HW2 Actor-Critic (GAE) 구현 진행 상황

## 최종 목표
`HW/HW2_spec.md` 기반으로 `HW/HW2/algos/reinforce.py`와 `HW/HW2/train_r.py`를 Actor-Critic + GAE로 개선.
- 각 맵에서 `python test_r.py --map hw_mapX.yaml --attempts 10 --headless`로 평가 시 **최소 98% 성공률** 필요.
- 200,000 episode 상한 내에서 조기 종료 허용.
- **rl conda env 사용**: `& "C:\Users\삼성\.conda\envs\rl\python.exe" ...`

## 현재 상태 요약

| 맵      | 학습 완료 | 평가 성공률 | 체크포인트                                                     |
|---------|-----------|-------------|----------------------------------------------------------------|
| hw_map1 | ✅        | **100%** (10/10) | `checkpoints/reinforce_hw_map1.pth`, `_best.pth`         |
| hw_map2 | ✅        | **100%** (10/10) | `checkpoints/reinforce_hw_map2.pth`, `_best.pth`         |
| hw_map3 | ❌ (미해결)| 미평가       | `checkpoints/reinforce_hw_map3_best.pth` (수렴 실패)     |

**map1, map2는 완전히 해결되었고, map3만 남은 상태다.**

## 파일 구조 및 변경 사항

### 현재 코드 상태
- **`HW/HW2/algos/reinforce.py`**: Actor-Critic + GAE 구현 완료 (State normalization, entropy annealing, cosine LR schedule)
- **`HW/HW2/train_r.py`**: Curriculum learning + evaluation-based early stopping
- **`HW/HW2/test_r.py`**: 원본 그대로 유지 (수정 X). `REINFORCEAgent(state_dim, action_dim, device=device)` + `agent.load()` + `agent.select_action()`만 호출.

### `algos/reinforce.py` 핵심 구성
1. **`RunningMeanStd`**: Welford 알고리즘 기반 상태 정규화 (position [0-7]과 ray [0-1] 스케일 차이 보정 → **가장 중요한 요소**)
2. **`PolicyNetwork`** (Actor): `state_dim → 256 → 256 → action_dim`, `log_std_init=0.5` (초기 std ≈ 1.65)
3. **`ValueNetwork`** (Critic): `state_dim → 256 → 256 → 1`
4. **`REINFORCEAgent`**:
   - **단일 optimizer**로 actor+critic 함께 학습
   - **CosineAnnealingLR** (`T_max=total_episodes`, `eta_min=lr*0.01`)
   - **Entropy 어닐링**: `entropy_coeff_start=0.1 → entropy_coeff_end=0.01` (0-80% linear)
   - **GAE** (γ=0.98, λ=0.95), truncation 시 마지막 상태의 V로 bootstrap
   - `set_done(done)` 인터페이스로 마지막 done 전달
   - **호환성 유지**: `REINFORCEAgent`, `select_action(state, eval_mode)`, `finish_episode()`, `save/load`, `agent.rewards.append(reward)` 인터페이스
   - Save/load format: `{'policy', 'value_net', 'state_normalizer'}` (구 형식 backward 호환)

### `train_r.py` 핵심 구성
1. **Curriculum learning (BFS 기반)**:
   - `progress_ratio ≤ 0.4` (Curriculum Phase): `max_dist = 3.0 → max_possible_dist` 선형 증가, 그 안의 셀에서 균일 샘플링
   - `progress_ratio > 0.4` (Mixed Phase): 50% 확률로 공식 시작점(0,0), 50% 랜덤 셀
   - `trap_adjacent` 셀 계산: 함정에 8방향 인접한 셀은 curriculum phase 시작점에서 **제외** (즉시 함정 빠짐 방지)
2. **Evaluation-based early stopping**:
   - `is_eval_episode` (Mixed Phase의 공식 시작점 에피소드)에서 성공률 계산
   - `success_window=100`, `early_stop_threshold=0.95`, `early_stop_patience=50`
   - 100번의 평가 에피소드 이동평균 성공률이 95% 이상을 50번 연속 달성하면 종료
3. **핵심 트릭 (LR/entropy 감쇠 방지)**:
   ```python
   agent = REINFORCEAgent(
       state_dim=state_dim, action_dim=action_dim,
       total_episodes=max(args.episodes, 200000),  # ← 이게 중요!
   )
   ```
   실제 `args.episodes`가 짧아도 `total_episodes=200000`으로 스케줄러 기준을 크게 잡아서 초반 학습 동안 LR과 entropy가 거의 감소하지 않게 함. **이걸 안 하면 짧은 episode에서 LR과 entropy가 너무 빨리 감쇠해서 학습 실패.**
4. **Best model save**: 새 best success rate 달성 시 `_best.pth`로 저장

## 학습/평가 명령어

```powershell
# rl conda env Python
$py = "C:\Users\삼성\.conda\envs\rl\python.exe"

# 학습 (from HW/HW2/)
& $py -u train_r.py --map hw_map1.yaml --episodes 2000 --max-steps 150
& $py -u train_r.py --map hw_map2.yaml --episodes 3000 --max-steps 200
& $py -u train_r.py --map hw_map3.yaml --episodes 10000 --max-steps 250

# 평가
& $py -u test_r.py --map hw_map1.yaml --attempts 10 --headless
& $py -u test_r.py --map hw_map2.yaml --attempts 10 --headless
& $py -u test_r.py --map hw_map3.yaml --attempts 10 --headless
```

## 학습 결과 (실제 실행)

### map1 (2000 episodes)
- Episode 900-1100에서 SuccessRate 100% 달성 → early stop
- **평가 10/10 = 100%** ✅

### map2 (3000 episodes)
- Curriculum Phase (0-1200): 성공/실패 반복하며 학습
- Mixed Phase (1300-): SuccessRate 70% → 92% → 97%
- Episode 3000에서 SuccessRate 97%로 종료 (early stop 조건 미달)
- **평가 10/10 = 100%** ✅

### map3 (10000 episodes 시도 → 실패)
- **문제**: Curriculum Phase에서 성공(100)과 실패(-250)가 반복되고 절대 안정화되지 않음
- 여러 하이퍼파라미터 조정 시도했지만 모두 실패
- Best model은 저장되었지만 신뢰 X (평가 실행 안 함)

## map3에서 시도한 것들 (모두 실패)

1. **`total_episodes=200000` 고정**: LR/entropy 감쇠 방지 → map2는 이걸로 해결됐지만 map3에는 부족
2. **Curriculum 시작점을 `frontier`가 아닌 균일 샘플링**으로 변경 → 개선 없음
3. **`trap_adjacent` 셀 제외**: goal 근처의 함정 인접 셀에서 시작하지 않도록 → 유효 셀이 너무 줄어듦
4. **`current_max_dist` 초기값 `2.0 → 3.0`**: 더 넓은 초기 범위 → 개선 없음
5. **Mixed Phase에서 trap_adjacent 셀 포함**: 학습 후반에 어려운 셀도 노출 → 개선 없음
6. **`entropy_coeff_start=0.05 → 0.1`, `end=0.001 → 0.01`**: 탐색 강화 → 개선 없음

## map3가 어려운 이유 (구조 분석)

Map3 layout (7x7):
```
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 1 1 1 0 0
0 0 1 3 3 0 2    ← goal 옆에 함정
0 0 1 3 0 0 2    ← goal 옆에 함정
0 0 0 0 0 0 2    ← 오른쪽 열 전부 함정
0 0 0 2 2 2 2    ← 아래쪽 열 함정
```
- **Goal (3,3), (3,4), (4,3)**은 벽으로 둘러싸여 있어 접근 경로가 제한적
- **Goal 바로 옆이 함정**: (3,5)/(4,5)는 안전하지만 (3,6)/(4,6)/(5,6)는 함정
- **아래쪽 전체가 함정**: (6,3)-(6,6)
- (0,0)에서 goal까지 BFS 거리 9. 정상 경로: 오른쪽으로 (0,5)/(1,5) 갔다가 아래로 (2,5)/(3,5)/(4,5) 지나 왼쪽으로 (4,4) → goal
- 함정 인접 셀이 매우 많아서 curriculum이 유용한 셀이 부족

## 다음에 시도해볼 방안

### 우선순위 높음
1. **Transfer learning**: map1 또는 map2에서 학습된 정책을 `initial_state_dict`로 사용 (goal-seeking 행동 이미 학습됨)
   - `train_r.py`에 `--pretrained` 옵션 추가하고 `agent.load(pretrained_path)` 후 학습 계속
2. **Curriculum 없이 (0,0) fixed로 학습**: 처음부터 실제 문제만 풀기. Episode 20000-50000 정도로 길게.
3. **더 큰 log_std_init (1.0 정도)**: 탐색을 훨씬 크게. 특히 초기에 goal 우회 경로 발견 기회 늘림
4. **Gradient normalization / PPO clipping 추가**: 정책 붕괴 방지. 백업 코드에는 없지만 안정성 향상

### 우선순위 중간
5. **`gamma=0.99`로 상향**: map3는 길이가 길어서 discount 완화가 도움될 수 있음
6. **Value network warm-up**: 처음 N episode는 critic만 학습하고 그 후 actor 학습
7. **Reward shaping (금지되어 있음)**: 스펙상 환경 수정 불가하므로 안됨

### 참고 정보
- `HW_BACKUP/reinforce.py`, `HW_BACKUP/train_ac.py`: 원본 백업 (map2까지는 성공, map3는 사용자 말로 "잘 수렴하지 않았음")
- `HW_BACKUP/reinforce_hw_map1_best.pth`, `reinforce_hw_map2_best.pth`: backup 학습된 모델 (evaluation으로 확인: map1 100%, map2 100%)
- Backup에서 map3 학습 실패 이력이 있음 → **map3는 원래 어려운 문제**. 새 접근이 필요.

## 인터페이스 호환성 체크리스트

`test_r.py`는 수정 안 함. 다음 인터페이스 유지 필수:
- ✅ `REINFORCEAgent(state_dim=..., action_dim=..., device=device)`
- ✅ `agent.load(model_path)` (구/신 checkpoint format 모두 지원)
- ✅ `agent.reset_episode()`
- ✅ `agent.select_action(state_tensor)` → action tensor (eval mode)
- ✅ `agent.rewards.append(reward)` (train_r.py용)
- ✅ `agent.policy(states)` (train_r.py 시각화용, normalized state 필요)
- ✅ `agent.normalize_state(state, update=False)` (train_r.py 시각화용)

## 주요 하이퍼파라미터 (현재 값)

```python
# algos/reinforce.py REINFORCEAgent.__init__
lr = 3e-4
gamma = 0.98
gae_lambda = 0.95
entropy_coeff_start = 0.1     # 최근에 0.05 → 0.1로 상향 (map3용)
entropy_coeff_end = 0.01      # 최근에 0.001 → 0.01로 상향 (map3용)
value_loss_coeff = 0.5
max_grad_norm = 0.5
log_std_init = 0.5

# train_r.py
early_stop_threshold = 0.95
early_stop_patience = 50
success_window = 100
agent.total_episodes = max(args.episodes, 200000)  # ← 핵심 트릭
```

**주의**: map1/map2는 `entropy_coeff_start=0.05, end=0.001`로 학습되었다. 지금 코드는 `0.1/0.01`로 바뀌어 있어서 map1/map2를 재학습하면 결과가 달라질 수 있다. 다만 map1/map2 체크포인트는 이미 완성됐으므로 재학습 불필요.
