import random
from collections import defaultdict
from env.gridworld_env import Action
import time
from tqdm import tqdm  
from algos.utils import plot_value_and_policy
import matplotlib.pyplot as plt


def sarsa(env, save_name, episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, render=False, log_interval=100):
    def get_expected_value(Q, state, epsilon):
        actions = Q[state]
        best_action = max(actions, key=actions.get)
        expected_val = 0.0
        num_actions = len(Action)
        for a in Action:
            prob = (1.0 - epsilon + epsilon / num_actions) if a == best_action else (epsilon / num_actions)
            expected_val += prob * actions[a]
        return expected_val

    # Initialize Q-value table
    Q = defaultdict(lambda: {a: 0.0 for a in Action})
    replay_buffer = []         # Replay buffer for model-free experience replay
    replay_batch_size = 50     # Batch size for experience replay

    all_rewards = []
    success_count = 0
    max_steps = 500

    # epsilon decay settings
    initial_epsilon = epsilon
    min_epsilon = 0.05
    decay_rate = 0.99

    for episode in tqdm(range(episodes), desc="Training SARSA"):
        # Decrease epsilon every episode
        epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))
        state = tuple(env.reset())

        # Select action using epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(list(Action))  # Exploration
        else:
            action = max(Q[state], key=Q[state].get)  # Exploitation

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            if render and episode % log_interval == 0:
                env.render()
                time.sleep(0.05)
            
            # Get next state and reward after taking action
            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)

            total_reward += reward

            # Select next action also using epsilon-greedy
            if random.random() < epsilon:
                next_action = random.choice(list(Action))
            else:
                next_action = max(Q[next_state], key=Q[next_state].get)

            # Expected SARSA update for actual step
            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                expected_next = get_expected_value(Q, next_state, epsilon)
                Q[state][action] += alpha * (reward + gamma * expected_next - Q[state][action])

            # Store experienced transition in buffer
            replay_buffer.append((state, action, reward, next_state, done))

            state, action = next_state, next_action
            steps += 1

            # Experience Replay using Expected SARSA update
            if len(replay_buffer) > 0:
                batch_size = min(len(replay_buffer), replay_batch_size)
                batch = random.sample(replay_buffer, batch_size)
                for s_b, a_b, r_b, s_n_b, d_b in batch:
                    if d_b:
                        Q[s_b][a_b] += alpha * (r_b - Q[s_b][a_b])
                    else:
                        expected_next_b = get_expected_value(Q, s_n_b, epsilon)
                        Q[s_b][a_b] += alpha * (r_b + gamma * expected_next_b - Q[s_b][a_b])

        all_rewards.append(total_reward)
        if reward == 100:  # When goal is reached
            success_count += 1

        output_folder = f"./outputs/sarsa_{save_name}"
        if (episode + 1) % log_interval == 0 or episode == episodes - 1:
            # Calculate V(s), π(s)
            V = {s: max(Q[s].values()) for s in Q}
            policy = {s: max(Q[s], key=Q[s].get) for s in Q}
            plot_value_and_policy(V, policy, env.grid, episode, env.width, env.height, output_folder=output_folder)

            # Print log
            avg_reward = sum(all_rewards[-log_interval:]) / log_interval
            success_rate = success_count / log_interval * 100
            print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")
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

    return Q, policy