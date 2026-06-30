import os
import random
from collections import defaultdict
from env.gridworld_env import Action
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from algos.utils import plot_value_and_policy

def q_learning(env, save_name, episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, render=False, log_interval=100):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})
    replay_buffer = []         # Replay buffer for model-free experience replay
    replay_batch_size = 50     # Batch size for experience replay
    
    reward_history = []
    success_rate_history = []
    success_count = 0
    max_steps = 100

    # epsilon decay settings
    initial_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.99

    for episode in tqdm(range(episodes), desc="Training Q-Learning"):
        # epsilon decay (initially, high epsilon to encourage exploration)
        epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))

        state = tuple(env.reset())
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            if render and episode % 500 == 0:
                env.render()
                time.sleep(0.05)

            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(list(Action))
            else:
                action = max(Q[state], key=Q[state].get)

            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)

            # Reward update: explicitly reinforce when goal is reached
            if done and reward == 100:
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                max_next = max(Q[next_state].values())
                Q[state][action] += alpha * (reward + gamma * max_next - Q[state][action])

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
 
            state = next_state
            episode_reward += reward
            if reward == 100:
                success_count += 1
 
            steps += 1
 
            # Experience Replay
            if len(replay_buffer) > 0:
                batch_size = min(len(replay_buffer), replay_batch_size)
                batch = random.sample(replay_buffer, batch_size)
                for s_b, a_b, r_b, s_n_b, d_b in batch:
                    if d_b and r_b == 100:
                        Q[s_b][a_b] += alpha * (r_b - Q[s_b][a_b])
                    else:
                        max_next_b = max(Q[s_n_b].values())
                        Q[s_b][a_b] += alpha * (r_b + gamma * max_next_b - Q[s_b][a_b])

        reward_history.append(episode_reward)
        output_folder = f"./outputs/q_learning_{save_name}"
        if (episode + 1) % log_interval == 0 or episode == episodes - 1:
            V = {s: max(Q[s].values()) for s in Q}
            policy = {s: max(Q[s], key=Q[s].get) for s in Q}
            plot_value_and_policy(V, policy, env.grid, episode, env.width, env.height, output_folder)

            avg_reward = sum(reward_history[-log_interval:]) / log_interval
            success_rate = success_count
            success_rate_history.append(success_rate)
            print(f"[Ep {episode+1}] Avg Reward: {avg_reward:.2f}, Successes: {success_rate}")
            success_count = 0

    # Return final policy (Full Policy Filling to prevent None action crashes in eval.py)
    policy = {}
    for y in range(env.height):
        for x in range(env.width):
            state = (y, x)
            if state in Q:
                policy[state] = max(Q[state], key=Q[state].get)
            else:
                policy[state] = random.choice([Action.DOWN, Action.RIGHT])

    plt.figure()
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning - Episode Rewards')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'q_learning_rewards.png'))
    plt.close()

    return Q, policy