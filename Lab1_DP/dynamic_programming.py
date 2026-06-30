import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from env.gridworld_env import Action, TileType
from env.gridworld_mdp import GridWorldMDP
import random

ARROWS = {
    Action.UP: '↑',
    Action.DOWN: '↓',
    Action.LEFT: '←',
    Action.RIGHT: '→'
}


# plot value and policy per every iteration
def plot_value_and_policy(V, policy, grid, iteration, width, height, policy_name, prefix='vi'):
    # dir
    save_dir = os.path.join("outputs", f"{prefix}_{policy_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    # plotting
    value_grid = np.full((height, width), np.nan)
    policy_grid = np.full((height, width), '', dtype=object)

    for (y, x), v in V.items():
        value_grid[y][x] = v
        policy_grid[y][x] = ARROWS[policy[(y, x)]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im = axes[0].imshow(value_grid, cmap='coolwarm', interpolation='nearest')
    for y in range(height):
        for x in range(width):
            if not np.isnan(value_grid[y, x]):
                axes[0].text(x, y, f"{value_grid[y, x]:.1f}", ha='center', va='center', color='black')
    axes[0].set_title(f"Value Function - Iteration {iteration}")

    axes[1].imshow(np.ones_like(value_grid), cmap='gray', vmin=0, vmax=1)
    for y in range(height):
        for x in range(width):
            if policy_grid[y][x]:
                axes[1].text(x, y, policy_grid[y][x], ha='center', va='center', fontsize=16)
            if grid[y][x] == TileType.WALL:
                axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))
            elif grid[y][x] == TileType.TRAP:
                axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='red'))
            elif grid[y][x] == TileType.GOAL:
                axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='green'))
    axes[1].set_title(f"Policy - Iteration {iteration}")

    for ax in axes:
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/iteration_{iteration}.png")
    plt.close()


#-----Policy Iteration-----#
def policy_evaluation(policy, mdp, gamma=0.95, theta=1e-3):
    V = {s: 0 for s in mdp.states} # initial value as zero
    while True:
        delta = 0
        for s in mdp.states:
            a = policy[s] # action with current state and policy
            next_s, reward, done = mdp.get_transition(s, a)
            V_new = reward + gamma * V.get(next_s, 0) * (not done) # Bellman Equation
            delta = max(delta, abs(V[s] - V_new))
            V[s] = V_new

        if delta < theta:
            break
    return V

def policy_improvement(V, mdp, gamma=0.95):
    policy_stable = True
    policy = {}
    for s in mdp.states:
        old_action = policy.get(s) 
        action_values = {}

        # compute q(s,a) for all actions possible for the current state
        for a in mdp.actions: 
            next_s, reward, done = mdp.get_transition(s, a)
            action_values[a] = reward + gamma * V.get(next_s, 0) * (not done) # Bellman Equation

        best_action = max(action_values, key=action_values.get) # search max q 
        policy[s] = best_action # select action with highest q and update policy

        # if updated, not converged
        if old_action is not None and old_action != best_action:
            policy_stable = False

    return policy, policy_stable

def policy_iteration(mdp, policy_name, gamma=0.95, theta=1e-03, max_iterations=80):
    # # uncomment if you want to set initial policy as random movement
    # policy = {s: random.choice(mdp.actions) for s in mdp.states} 
    # set initial policy as moving right, for faster convergence.
    policy = {s: Action.RIGHT for s in mdp.states} 
    iteration = 0
    policy_changes = []

    while iteration < max_iterations:
        # value function on current policy
        V = policy_evaluation(policy, mdp, gamma, theta) 
                
        old_policy = policy.copy()
        # compute policy improvement based on current value function
        policy, _ = policy_improvement(V, mdp, gamma) 

        # measure how much policy has changed for each state
        changed = sum(old_policy[s] != policy[s] for s in mdp.states) 
        policy_changes.append(changed)

        plot_value_and_policy(V, policy, mdp.env.grid, iteration, mdp.width, mdp.height, policy_name, prefix='pi')
        print(f"[PI Iter {iteration}] policy changes: {changed}")

        # check convergence
        if changed == 0:
            break

        iteration += 1

    return V, policy




#-----Value Iteration-----#
def value_iteration(mdp, policy_name, gamma=0.95, theta=1e-3, max_iterations=80):
    V = {s: 0 for s in mdp.states} # initial zero
    policy = {s: Action.UP for s in mdp.states} # set moving up as initial policy
    iteration = 0
    deltas = []

    while iteration < max_iterations:
        delta = 0
        new_V = V.copy()

        for s in mdp.states:
            max_value = float('-inf')
            best_action = None

            # compute q(s,a) for all actions possible for the current state
            for a in mdp.actions:
                next_s, reward, done = mdp.get_transition(s, a)
                value = reward + gamma * (0 if done else V[next_s]) # Bellman equation

                # select action with highest q value
                if value > max_value:
                    max_value = value
                    best_action = a

            # update V[s] and policy
            new_V[s] = max_value
            policy[s] = best_action

            delta = max(delta, abs(V[s] - new_V[s])) # how much v decreased

        V = new_V
        deltas.append(delta)
        plot_value_and_policy(V, policy, mdp.env.grid, iteration, mdp.width, mdp.height, policy_name, prefix='vi')

        print(f"[VI Iter {iteration}] max Δ: {delta:.5f}")

        # if delta is smaller than theta, then converged
        if delta < theta:
            break

        iteration += 1

    return V, policy



