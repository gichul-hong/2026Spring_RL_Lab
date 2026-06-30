## Lab 3. Policy Gradient and Deep RL
This provides an introduction to policy gradient and basics of deep reinforcement learning (deep RL).
Implemented algorithms include:

- **Policy Gradient: REINFORCE**
- **Deep RL: DQN**

### Continuous Space GridWorld Environment

There are two variants:

#### Discrete-Action GridWorld (`gridworld_c1.py`)
- **State:** continuous \((row, col)\) in meters  
- **Actions:** 8 directions (N, NE, E, SE, S, SW, W, NW), fixed step length  
- **Rewards:**  
  - Move: **–1**  
  - Trap: **–100**, episode ends  
  - Goal: **+100**, episode ends  

#### Continuous-Action GridWorld (`gridworld_c2.py`)
- **State:** continuous \((row, col)\) in meters  
- **Action:** 2D vector \(\in[-1,1]^2\) (clamped internally)  
- **Rewards:** same as discrete  

Each cell in the map can be one of:
- 🟩 **Normal** (0): free to move, reward –1  
- 🧱 **Wall** (1): blocks movement, reward –1  
- ☠️ **Trap** (2): ends episode, reward –100  
- 🎯 **Goal** (3): ends episode, reward +100 

### Training

To train an RL agent, run the `train.py` script with the desired algorithm and optional arguments.
```bash
python train.py --algo {ALGORITHM} [--save_name SAVE_NAME] [--map MAP_NAME]  [--render]
```
**Arguments**
- --algo (str, required): Choose the learning algorithm (REINFORCE, DQN)
	- Options: reinforce, dqn
- --map (str, optional): Select GridWorld Map (Choice: map0, map1, map2, map3). Default is map1.
- --save_name (str): Filename to save the policy and the plots.
- --render (action flag): If specified, render the agent's behavior during training.
- --logdir (str, optional): Directory to save tensorboard log files. Default is `/runs`.
- --seed (int, optional): Seed for training. Default is 42.

**Parameter Arguments**
You can adjust the parameters for training with additional arguments
- --episodes (int) Number of episodes. Default is 1000.
- --max-steps (int) Number of maximum step per episode. Default is 100.

**To Run TensorBoard**

Run the following script to visualize TensorBoard logs.
```bash
tensorboard --logdir runs/
```

TensorBoard logs include:
- **Reward Curve**
- **Epsilon Decay** (for DQN)  
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
- --algo (str, required): Choose the learning algorithm (REINFORCE, DQN)
	- Options: reinforce, dqn
- --map (str, optional): Select GridWorld Map (Choice: map0, map1, map2, map3). Default is map1.
- --save_name (str): Filename to load the policy saved as `checkpoints/{args.algo}_{args.save_name}.pth`
- --iter (int): Number of iterations to test. Default is 10.

This will render the agent's behavior following the trained policy in the GridWorld environment.

### Example Script
#### REINFORCE
```bash
# train
python train.py --algo reinforce --map map1  --save_name example  --episodes 4000 --max-steps 50 --render 
# test
python test.py --algo reinforce --map map1 --save_name example 
```
#### DQN
```bash
# train
python train.py --algo dqn --map map1 --save_name example --render
# test
python test.py --algo dqn --map map1 --save_name example 
```