# RL HW1 과제 (홍기철)

## 1. SARSA 알고리즘에서 수정한 내용과 그 이유

* **Optimistic Initial Values 적용**
  * **수정 내용**: Q-value 테이블의 초기값을 기존 `0.0`에서 `0.3`으로 변경
  * **이유**: Sparse reward(희소 보상) 격자 환경에서의 탐험 유도 목적. 미방문 상태-행동 쌍의 가치를 높게 평가하여 에이전트가 맵 전체를 적극적으로 탐색하도록 유도함.
* **학습률(Learning Rate, $\alpha$) 상향 조정**
  * **수정 내용**: 학습률 인자 `alpha`를 기존 `0.1`에서 `0.3`으로 변경
  * **이유**: 500 에피소드라는 제한된 학습 횟수 내에서 Q-value의 빠른 수렴 도모.
* **최소 탐험률(Minimum Epsilon, $\epsilon_{min}$) 조정**
  * **수정 내용**: 최소 탐험률 `min_epsilon`을 `0.05`에서 `0.01`로 축소
  * **이유**: 학습 후반부 불필요한 무작위 움직임을 최소화하고, 이미 학습된 최적 정책에 따른 목표 지점 도달(Exploitation) 확률을 극대화함.

---

## 2. Q-Learning 알고리즘에서 수정한 내용과 그 이유

* **Q-value 낙관적 초기화 및 Epsilon-greedy 조정**
  * **수정 내용**: SARSA와 동일하게 초기 Q-value를 `0.3`으로 상향하고, 최소 탐험률을 `0.01`로 조정
  * **이유**: 환경 전반에 대한 신속한 탐색을 촉진하고 최적 경로 학습 후 정책 안정성 강화.
* **학습률(Learning Rate, $\alpha$) 상향 조정**
  * **수정 내용**: `alpha` 파라미터를 `0.3`으로 변경
  * **이유**: 학습 가속화 및 조기 수렴 유도.

---

## 3. 자체 실험 결과 분석 및 설명

### 3.1 하이퍼파라미터 변경 사항 비교
| 파라미터 | Baseline | 최종 설정 | 변경 이유 및 효과 |
| :--- | :---: | :---: | :--- |
| **Q 초기값** | `0.0` | **`0.3`** | Optimistic initialization 기법 적용을 통한 능동적 탐험 유도 |
| **최소 탐험률 (`min_epsilon`)** | `0.05` | **`0.01`** | 학습 후반 탐험 감소 및 최적 정책 수행(Exploitation) 강화 |
| **탐험 감소율 (`decay_rate`)** | `0.99` | **`0.99`** | 탐험 시간을 점진적으로 유지하여 안정적인 경로 습득 (유지) |
| **학습률 (`alpha`)** | `0.1` | **`0.3`** | 수렴 속도의 약 3배 향상 |


### 3.2 평가 결과
`eval.py`를 기준으로 알고리즘당 10회 독립 실행 평가를 수행한 결과, 두 알고리즘 및 제공된 테스트 맵에서 **모두 100% 성공률** 달성.

| 알고리즘 | 대상 맵 | 성공률 (성공 횟수) |
| :--- | :--- | :---: |
| **SARSA** (max_steps=500) | `HW1_1.json` | **100% (10/10)** |
| | `HW1_2.json` | **100% (10/10)** |
| **Q-Learning** (max_steps=100) | `HW1_1.json` | **100% (10/10)** |
| | `HW1_2.json` | **100% (10/10)** |

### 3.3 실험 결과 분석
* **희소 보상 문제 극복**: Goal 지점에 도달했을 때만 큰 양의 보상(+100)을 받는 환경 특성상, 낙관적 초기화(`0.3`)가 상태 공간 전반의 가치를 끌어올려 에이전트의 경로 발견 효율 개선
* **알고리즘별 특성**: SARSA는 On-policy 방식으로 수렴 과정에서 비교적 보수적인 경로를 탐색하는 경향을 보임. 반면 Off-policy 방식인 Q-Learning은 탐험 중에도 최선의 행동만을 타겟으로 삼아 더욱 직접적이고 최단에 가까운 정책을 효율적으로 학습함. 특히 Q-Learning에서 최대 스텝 수 제한을 100으로 줄였음에도 성공률 100%를 달성

---

## 4. 실행 방법에 대한 설명

### 4.1 train & render
* **SARSA 실행 명령어 (맵 1 & 맵 2)**
  ```bash
  python train.py --algo sarsa --map HW1_1.json
  python render.py --policy policy_sarsa_None.pkl --map HW1_1.json
  ```
  ```bash
  python train.py --algo sarsa --map HW1_2.json
  python render.py --policy policy_sarsa_None.pkl --map HW1_2.json
  ```
* **Q-Learning 실행 명령어 (맵 1 & 맵 2)**
  ```bash
  python train.py --algo q_learning --map HW1_1.json
  python render.py --policy policy_q_learning_None.pkl --map HW1_1.json
  ```
  ```bash
  python train.py --algo q_learning --map HW1_2.json
  python render.py --policy policy_q_learning_None.pkl --map HW1_2.json
  ```

### 4.2 evaluation
* eval.py에서 episodes의 default value는 500이라서 별도로 argument로 넘기진 않음, alpha는 default value가 0.1 이지만, 각각의 알고리즘 function 내부에서 0.3으로 하드코딩 해서 argument로 넘길 필요 없으나 보험 성격으로 추가
* **SARSA 실행 명령어 (맵 1 & 맵 2)**
  ```bash
  python eval.py --algo sarsa --map HW1_1.json --alpha 0.3
  ```
  ```bash
  python eval.py --algo sarsa --map HW1_2.json --alpha 0.3
  ```
* **Q-Learning 실행 명령어 (맵 1 & 맵 2)**
  ```bash
  python eval.py --algo q_learning --map HW1_1.json --alpha 0.3
  ```
  ```bash
  python eval.py --algo q_learning --map HW1_2.json --alpha 0.3
  ```