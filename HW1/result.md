# HW1 과제 결과 보고서

## 1. 최종 평가 결과

`eval.py` 기준 10회 독립 실행 결과, **4가지 조합 모두 100% 성공률**을 달성했습니다.

| 알고리즘 | 대상 맵 | 성공률 | 통과 여부 |
| :--- | :--- | :---: | :---: |
| **SARSA** (max_steps=500) | `HW1_1.json` | **100% (10/10)** | ✅ 통과 |
| | `HW1_2.json` | **100% (10/10)** | ✅ 통과 |
| **Q-Learning** (max_steps=100) | `HW1_1.json` | **100% (10/10)** | ✅ 통과 |
| | `HW1_2.json` | **100% (10/10)** | ✅ 통과 |

---

## 2. 변경 사항 요약

SARSA (`algos/sarsa.py`)와 Q-Learning (`algos/q_learning.py`)에 공통으로 아래 3가지를 수정했습니다.

### 2.1 Q-value Optimistic Initialization

```python
# Before (baseline)
Q = defaultdict(lambda: {a: 0.0 for a in Action})

# After
Q = defaultdict(lambda: {a: 0.3 for a in Action})
```

- **목적**: 15x15 sparse reward 환경에서 Q값이 모두 0.0이면, agent가 방문하지 않은 상태와 방문한 상태를 구분하지 못해 goal 탐색이 어렵습니다.
- **효과**: 초기 Q값을 0.3으로 설정하여 미방문 state-action의 Q값을 높이면, agent가 자연스럽게 전 영역을 탐색하게 되어 goal을 빨리 발견할 확률이 크게 증가합니다.

### 2.2 Epsilon-greedy Tie-breaking

```python
# Before (baseline)
action = max(Q[state], key=Q[state].get)

# After
q_values = Q[state]
max_q = max(q_values.values())
best_actions = [a for a, q in q_values.items() if q == max_q]
action = random.choice(best_actions)
```

- **목적**: 여러 action이 동일한 최대 Q값을 가질 때, `max()`는 항상 첫 번째 action만 선택하여 편향이 발생합니다.
- **효과**: 동일한 Q값을 가진 action들 사이에서 무작위 선택을 통해 편향을 제거하고, 초기 탐험의 다양성을 보장합니다. SARSA의 next_action 선택에도 동일하게 적용했습니다.

### 2.3 Epsilon Decay 파라미터 조정

```python
# Before (baseline)
min_epsilon = 0.05
decay_rate = 0.99

# After
min_epsilon = 0.01
decay_rate = 0.99
```

- **min_epsilon 0.05 -> 0.01**: 학습 후반부에 탐험을 최소화하여 학습된 정책으로 goal 도달률을 극대화합니다.
- **decay_rate**: 0.99를 유지하며, epsilon이 천천히 감소하여 충분한 탐험 시간을 확보합니다.

---

## 3. 학습 실행 설정

평가 실행 시 다음 커맨드라인 인자를 사용합니다.

```bash
python eval.py --algo sarsa --map HW1_1.json --episodes 500 --alpha 0.3
python eval.py --algo sarsa --map HW1_2.json --episodes 500 --alpha 0.3
python eval.py --algo q_learning --map HW1_1.json --episodes 500 --alpha 0.3
python eval.py --algo q_learning --map HW1_2.json --episodes 500 --alpha 0.3
```

- **alpha 0.1 -> 0.3**: 학습률을 3배 높여 500 에피소드 내 Q-value 수렴 속도를 향상시킵니다.
- **episodes 500**: 충분한 학습 기간을 보장합니다.

---

## 4. 파라미터 변경 비교표

| 파라미터 | Baseline | 최종 | 변경 이유 |
| :--- | :---: | :---: | :--- |
| Q 초기값 | 0.0 | **0.3** | Optimistic initialization으로 탐험 유도 |
| min_epsilon | 0.05 | **0.01** | 후반 탐험 최소화, exploitation 강화 |
| decay_rate | 0.99 | **0.99** | 충분한 탐험 시간 확보 (유지) |
| alpha | 0.1 | **0.3** | 학습 속도 3배 향상 |
| tie-breaking | 없음 | **추가** | 동일 Q값 action 간 편향 제거 |

---

## 5. 수정된 파일 목록

- `Lab2_ModelFree/algos/sarsa.py`
- `Lab2_ModelFree/algos/q_learning.py`
