# Actor-Critic (GAE) 구현 요약

## 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `algos/reinforce.py` | REINFORCE → **Actor-Critic with GAE** 전면 확장 |
| `train_r.py` | GAE bootstrap + Best model 저장 + Early stopping + TensorBoard 확장 |

---

## 핵심 개선 사항

### 알고리즘 (`reinforce.py`)

| 기존 (Vanilla REINFORCE) | 개선 (Actor-Critic + GAE) |
|--------------------------|--------------------------|
| Monte-Carlo return (높은 분산) | **GAE** (λ=0.95) advantage 추정 |
| Baseline 없음 | **ValueNetwork (Critic)** baseline |
| 엔트로피 미고려 | **Entropy regularization** (coeff=0.01) |
| Gradient clipping 없음 | **Grad norm clipping** (max=0.5) |
| lr=1e-3 (고정) | **lr=3e-4 + CosineAnnealingLR** (점진 감소) |

### 학습 안정성 (`train_r.py`)

| 기능 | 설명 |
|------|------|
| **Best model 저장** | 성공률이 역대 최고일 때 `_best.pth`로 별도 저장 |
| **Early stopping** | 성공률 ≥ 98%가 10번(=1,000 에피소드) 연속 시 자동 종료 |
| **LR 스케줄링** | CosineAnnealingLR로 lr → lr×0.01까지 점진 감소, 수렴 후 붕괴 방지 |

---

## 호환성 유지

- ✅ `agent.select_action(state_tensor)` — 인터페이스 동일
- ✅ `agent.finish_episode()` → `float` 반환
- ✅ `agent.rewards.append(reward)` — 외부 보상 추가 방식 동일
- ✅ `agent.policy` 속성 유지 (시각화 호환)
- ✅ `agent.save()/load()` — 새 형식 + 기존 `.pth` 하위 호환
- ✅ `test_r.py` 수정 불필요

---

## 실행 방법 (변경 없음)

```bash
# 학습
python train_r.py --map hw_map1.yaml --episodes 200000
python train_r.py --map hw_map2.yaml --episodes 200000
python train_r.py --map hw_map3.yaml --episodes 200000

# 평가 (best model 사용 권장)
python test_r.py --map hw_map1.yaml --model reinforce_hw_map1_best.pth --attempts 10 --headless
```

---

## 저장되는 체크포인트

| 파일 | 설명 |
|------|------|
| `checkpoints/reinforce_hw_mapX.pth` | 학습 종료 시점의 최종 모델 |
| `checkpoints/reinforce_hw_mapX_best.pth` | 성공률이 가장 높았던 시점의 모델 |

**평가 시에는 `_best.pth`를 사용하세요** — 최종 모델은 정책 붕괴 가능성이 있습니다.

---

## TensorBoard 메트릭

| 메트릭 | 설명 |
|--------|------|
| `Loss` | 총 손실 (Actor + Entropy + Critic) |
| `Loss/Actor` | Actor(정책) 손실 |
| `Loss/Critic` | Critic(가치) 손실 |
| `Entropy` | 정책 분포 엔트로피 (탐색-활용 균형) |
| `Reward` | 에피소드 누적 보상 |
| `SuccessRate` | 최근 100 에피소드 성공률 |
| `LearningRate` | 현재 학습률 (CosineAnnealing 추이) |

```bash
tensorboard --logdir runs/
```

---

## 정책 붕괴 문제 해결

기존 문제: 학습이 수렴한 후에도 계속 업데이트 → 정책 붕괴 (100% → 27%)

해결:
1. **LR 스케줄링**: 학습 후반부 lr이 0에 가까워져 수렴 상태 유지
2. **Best model 저장**: 붕괴 전 최고 성능 모델이 이미 저장됨
3. **Early stopping**: 충분히 학습되면 자동 종료하여 과도한 업데이트 방지
