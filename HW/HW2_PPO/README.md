# HW2

This repository contains a basic REINFORCE implementation for the continuous-action GridWorld environment.

Implemented algorithm:

- **REINFORCE** (Monte-Carlo policy gradient for continuous actions)

---

## Training

To train the REINFORCE agent, run `train_r.py`.

```bash
python train_r.py --map {MAP_NAME}.yaml --episodes {EPISODES} --max-steps {MAX_STEPS}
```

Example commands:

```bash
python train_r.py --map map0.yaml --episodes 1000 --max-steps 100
python train_r.py --map map1.yaml --episodes 1000 --max-steps 100

python train_r.py --map hw_map1.yaml --episodes 200000 --max-steps 150
python train_r.py --map hw_map2.yaml --episodes 200000 --max-steps 200
python train_r.py --map hw_map3.yaml --episodes 200000 --max-steps 250
```

For `hw_map1.yaml`, `hw_map2.yaml`, and `hw_map3.yaml`, `--max-steps` can be omitted. The default values are 150, 200, and 250.

The trained policy is saved under `checkpoints/` using the map name.

```text
checkpoints/reinforce_map0.pth
checkpoints/reinforce_map1.pth
checkpoints/reinforce_hw_map1.pth
checkpoints/reinforce_hw_map2.pth
checkpoints/reinforce_hw_map3.pth
```

TensorBoard logs are saved under `runs/`.

```bash
tensorboard --logdir runs/
```

TensorBoard logs include:

- Reward curve
- Loss
- Policy arrows

---

## Testing / Rendering

To test a trained policy:

```bash
python test_r.py --map {MAP_NAME}.yaml --attempts 10 --max-steps {MAX_STEPS}
```

Example commands:

```bash
python test_r.py --map hw_map1.yaml --attempts 10 --max-steps 150
python test_r.py --map hw_map2.yaml --attempts 10 --max-steps 200
python test_r.py --map hw_map3.yaml --attempts 10 --max-steps 250
```

If `--model` is omitted, `test_r.py` loads the checkpoint that matches the map name. For example, `--map hw_map1.yaml` loads `checkpoints/reinforce_hw_map1.pth`.

Use `--headless` when rendering is not needed.

```bash
python test_r.py --map hw_map1.yaml --attempts 10 --headless
```

---

## Continuous-Action GridWorld

- **State:** continuous `(row, col)` position and 8 ray-sensor distances
- **Action:** 2D vector in `[-1, 1]^2`, clamped internally
- **Rewards:**
  - Move: `-1`
  - Trap: `-100`, episode ends
  - Goal: `+100`, episode ends

Each cell in the map can be one of:

- Normal (`0`)
- Wall (`1`)
- Trap (`2`)
- Goal (`3`)

---

## Folder Structure

```text
HW2/
├── algos/
│   └── reinforce.py
├── configs/
│   ├── map0.yaml
│   ├── map1.yaml
│   ├── hw_map1.yaml
│   ├── hw_map2.yaml
│   └── hw_map3.yaml
├── env/
│   └── gridworld_c2.py
├── train_r.py
├── test_r.py
└── README.md
```

Generated files such as checkpoints and TensorBoard logs are not included in the release.
