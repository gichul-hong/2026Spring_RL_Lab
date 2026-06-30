## Lab 3. Deep Value-Based Reinforcement Learning
This provides an introduction to value-based deep reinforcement learning with function approximation.
Implemented algorithms include:

- **Deep SARSA**
- **DQN**

### Continuous GridWorld Environment

#### Discrete-Action Continuous GridWorld (`gridworld_c1.py`)
- **State:** continuous \((row, col)\) in meters
- **Actions:** 8 directions (N, NE, E, SE, S, SW, W, NW), fixed step length
- **Rewards:**
  - Move: **-1**
  - Trap: **-100**, episode ends
  - Goal: **+100**, episode ends

Each cell in the map can be one of:
- **Normal** (0): free to move, reward -1
- **Wall** (1): blocks movement, reward -1
- **Trap** (2): ends episode, reward -100
- **Goal** (3): ends episode, reward +100

### Training

To train an RL agent, run the `train.py` script with the desired algorithm and optional arguments.
```bash
python train.py --algo {ALGORITHM} [--save_name SAVE_NAME] [--map MAP_NAME] [--render]
```
**Arguments**
- --algo (str, required): Choose the learning algorithm (Deep SARSA, DQN)
	- Options: deepsarsa, dqn
- --map (str, optional): Select GridWorld Map (Choice: map0, map1, map2, map3). Default is map2.
- --save_name (str): Filename to save the policy and the plots.
- --render (action flag): If specified, render the agent's behavior during training.
- --logdir (str, optional): Directory to save tensorboard log files. Default is `runs`.
- --seed (int, optional): Seed for training. Default is 42.

**Parameter Arguments**
You can adjust the parameters for training with additional arguments
- --episodes (int) Number of episodes. Default is 1500.
- --max-steps (int) Number of maximum step per episode. Default is 100.
- --step-size (float) Movement length for each discrete action. Default is 1.0.

**To Run TensorBoard**

Run the following script to visualize TensorBoard logs.
```bash
tensorboard --logdir runs/
```

TensorBoard logs include:
- **Reward Curve**
- **Epsilon Decay**
- **Loss**
- **StateValueHeatmap** (figure visualization)
- **PolicyArrows** (figure visualization)

**Save**
The trained policy will be saved as `checkpoints/{args.algo}_{args.save_name}.pth`.

### Rendering a Trained Policy
You can visualize a learned policy using the `test.py` script:
```bash
python test.py --algo {ALGORITHM} [--save_name SAVE_NAME] [--iter ITER]
```
**Arguments**
- --algo (str, required): Choose the learning algorithm (Deep SARSA, DQN)
	- Options: deepsarsa, dqn
- --map (str, optional): Select GridWorld Map (Choice: map0, map1, map2, map3). Default is map2.
- --save_name (str): Filename to load the policy saved as `checkpoints/{args.algo}_{args.save_name}.pth`
- --step-size (float): Movement length used by the environment. Use the same value as training.
- --iter (int): Number of iterations to test. Default is 10.

This will render the agent's behavior following the trained policy in the GridWorld environment.

### Example Script
#### Deep SARSA
```bash
# train
python train.py --algo deepsarsa
# test
python test.py --algo deepsarsa
```
#### DQN
```bash
# train
python train.py --algo dqn
# test
python test.py --algo dqn
```
