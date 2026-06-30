## Lab 4. Policy Gradient

This lab introduces policy-gradient control with a continuous-action GridWorld.
Implemented algorithm:

- **REINFORCE**

The previous value-based deep RL material is now separated into `Lab3_DeepValueRL`.

### Continuous-Action GridWorld

`gridworld_c2.py` represents the agent state as a continuous `(row, col)` position.
The action is a 2D vector in `[-1, 1]^2`, which is clamped internally by the
environment.

Rewards:

- Move: `-1`
- Trap: `-100`, episode ends
- Goal: `+100`, episode ends

Each map cell is one of:

- Normal `0`
- Wall `1`
- Trap `2`
- Goal `3`

### Training

Run `train.py` with REINFORCE.

```bash
python train.py --algo reinforce --map {MAP_NAME} --save_name {SAVE_NAME}
```

Arguments:

- `--algo`: `reinforce`
- `--map`: `map0`, `map1`, `map2`, or `map3`
- `--save_name`: checkpoint suffix. Default is `example`.
- `--episodes`: number of training episodes. Default is `1000`.
- `--max-steps`: maximum steps per episode. Default is `100`.
- `--render`: render the environment during training.
- `--logdir`: TensorBoard log directory. Default is `runs`.
- `--seed`: random seed. Default is `42`.

Example:

```bash
python train.py --algo reinforce --map map1 --save_name example --episodes 4000 --max-steps 50
```

### TensorBoard

The root `requirements.txt` already includes `tensorboard`. If it is missing in
your environment, install it with:

```bash
pip install tensorboard
```

After training, run:

```bash
tensorboard --logdir runs/
```

Logged items include:

- Reward curve
- Loss
- Continuous policy arrows

### Testing A Trained Policy

```bash
python test.py --algo reinforce --map {MAP_NAME} --save_name {SAVE_NAME}
```

Checkpoints are saved as:

```text
checkpoints/reinforce_{save_name}.pth
```
