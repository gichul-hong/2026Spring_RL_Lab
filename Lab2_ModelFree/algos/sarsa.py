import random
from collections import defaultdict
from env.gridworld_env import Action
import time
from tqdm import tqdm  
from algos.utils import plot_value_and_policy
import matplotlib.pyplot as plt


def sarsa(env, save_name, episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, render=False, log_interval=100):
    # Initialize Q-value table: set Q values to 0 for all state-action pairs
    Q = defaultdict(lambda: {a: 1.0 for a in Action})  # Optimistic initialization

    all_rewards = []
    success_count = 0
    max_steps = 500

    # epsilon decay settings
    initial_epsilon = epsilon
    min_epsilon = 0.01
    decay_rate = 0.994

    for episode in tqdm(range(episodes), desc="Training SARSA"):
        # Decrease epsilon every episode
        epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))
        state = tuple(env.reset())

        # Select action using epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(list(Action))  # Exploration
        else:
            q_values = Q[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)  # Exploitation with tie-breaking

        done = False
        total_reward = 0
        steps = 0

        # while not done and steps < max_steps:
        #     if render and episode % log_interval == 0:
        #         env.render()
        #         time.sleep(0.05)
            
        #     # Get next state and reward after taking action
        #     next_state, reward, done = env.step(action.value)
        #     next_state = tuple(next_state)

        #     total_reward += reward

        #     # Select next action also using epsilon-greedy
        #     if random.random() < epsilon:
        #         next_action = random.choice(list(Action))
        #     else:
        #         next_action = max(Q[next_state], key=Q[next_state].get)

        #     # SARSA update
        #     td_target = reward + gamma * Q[next_state][next_action]
        #     Q[state][action] += alpha * (td_target - Q[state][action])

        #     state, action = next_state, next_action
        #     steps += 1

        while not done and steps < max_steps:
            if render and episode % log_interval == 0:
                env.render()
                time.sleep(0.05)
            
            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)
            total_reward += reward

            # 1. 다음 행동 선택 시 타이브레이킹 적용
            if random.random() < epsilon:
                next_action = random.choice(list(Action))
            else:
                next_q_values = Q[next_state]
                max_next_q = max(next_q_values.values())
                best_next_actions = [a for a, q in next_q_values.items() if q == max_next_q]
                next_action = random.choice(best_next_actions)

            # 2. 터미널 상태 분기를 통한 SARSA 업데이트 안정화
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * Q[next_state][next_action]
                
            Q[state][action] += alpha * (td_target - Q[state][action])

            state, action = next_state, next_action
            steps += 1
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

    # Return final policy
    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}

    return Q, policy
