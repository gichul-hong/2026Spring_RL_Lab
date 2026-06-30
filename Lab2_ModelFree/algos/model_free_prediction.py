import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from env.gridworld_env import Action, TileType
from algos.utils import make_output_folder

def random_policy(state):
    return random.choice(list(Action))

# Monte Carlo method estimates V(s) by calculating total return (G) after episode ends
# first_visit = True, use_incremental_mean = False
# → First-Visit MC
#   - Updates only when a (state, action) pair appears for the first time in an episode
#   - Stores all return values G in returns[(s,a)] and computes Q(s,a) as their mean
#
# first_visit = False, use_incremental_mean = False
# → Every-Visit MC
#   - Updates every time a (state, action) pair appears in an episode
#   - Stores all G values in a list and computes Q(s,a) as their mean
#
# first_visit = True, use_incremental_mean = True
# → First-Visit MC + Incremental Mean
#   - Updates only the first occurrence of (state, action) in an episode
#   - Updates Q(s,a) incrementally without storing all returns:
#     Q(s,a) ← Q(s,a) + (1 / N(s,a)) * (G - Q(s,a))
#
# first_visit = False, use_incremental_mean = True
# → Every-Visit MC + Incremental Mean
#   - Updates Q(s,a) every time the (state, action) pair appears
#   - Applies incremental update immediately using each G value

def monte_carlo_prediction(env, policy, episodes=1000, gamma=0.99, 
                          first_visit=True, use_incremental_mean=False):
    V = defaultdict(float)
    returns = defaultdict(list)
    counts = defaultdict(int)  # For incremental mean. visit counter N(s)

    for _ in range(episodes):
        state = tuple(env.reset())
        episode = []
        done = False

        # Store (state, reward) tuples within one episode
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action.value)
            episode.append((state, reward))
            state = tuple(next_state)

        G = 0
        visited = set()  # for first-visit
        
        # Calculate episode rewards in reverse order (for efficient cumulative reward calculation)
        for t in reversed(range(len(episode))):
            s_t, r_t = episode[t]
            G = gamma * G + r_t  # Calculate total return G
            
            # Every-visit / or first time visit when first_visit is True
            if not first_visit or s_t not in visited:
                if use_incremental_mean:
                    # update V by adjusting the difference between existing V value and G with learning rate α
                    counts[s_t] += 1
                    alpha = 1 / counts[s_t]
                    V[s_t] += alpha * (G - V[s_t])
                else:  # store all G values, then update V by calculating the average
                    returns[s_t].append(G)
                    V[s_t] = sum(returns[s_t]) / len(returns[s_t])  # Average of returns per state
                
                visited.add(s_t)

    return V


# Incrementally updates V(s) with reward from environment and next state's value estimate at each step
def td0_prediction(env, policy, episodes=1000, alpha=0.1, gamma=0.99):
    V = defaultdict(float)
    
    for _ in range(episodes):
        state = tuple(env.reset())
        done = False

        while not done: # Incrementally update V(s) at each step
            action = policy(state)
            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)

            # TD(0) update
            td_target = reward + gamma * V[next_state]
            V[state] += alpha * (td_target - V[state])

            state = next_state

    return V

# Monte Carlo vs TD(0): Compare average value estimates and variance for each state through repeated training
# Run each algorithm runs times and analyze result statistics
def run_prediction_experiment(env, save_name, episodes=1000, runs=30, gamma=0.99, alpha=0.1, first_visit=True, use_incremental_mean=False):

    output_folder_name = f"outputs/mf_pred_{save_name}/"
    make_output_folder(output_folder_name)

    all_mc_values = []
    all_td_values = []

    # Execute runs times and collect multiple estimates from Monte Carlo and TD(0)
    for _ in range(runs):
        V_mc = monte_carlo_prediction(env, random_policy, episodes, gamma, first_visit, use_incremental_mean)
        V_td = td0_prediction(env, random_policy, episodes, alpha, gamma)
        all_mc_values.append(V_mc)
        all_td_values.append(V_td)

    all_states = set().union(*[v.keys() for v in all_mc_values + all_td_values])
    all_states = sorted(all_states)

    mc_means, td_means = [], []
    mc_vars, td_vars = [], []

    # Calculate mean and variance of V(s) estimated by Monte Carlo or TD for each state
    for s in all_states:
        mc_vals = [v.get(s, 0.0) for v in all_mc_values]
        td_vals = [v.get(s, 0.0) for v in all_td_values]

        mc_means.append(np.mean(mc_vals))
        td_means.append(np.mean(td_vals))
        mc_vars.append(np.var(mc_vals))
        td_vars.append(np.var(td_vals))

    # Plot graphs
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x_labels = [str(s) for s in all_states]

    axs[0].plot(x_labels, mc_means, label="Monte Carlo", marker='o')
    axs[0].plot(x_labels, td_means, label="TD(0)", marker='x')
    axs[0].set_ylabel("Mean V(s)")
    axs[0].set_title("Monte Carlo vs TD(0) Value Estimates (Mean)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(x_labels, mc_vars, label="Monte Carlo", marker='o')
    axs[1].plot(x_labels, td_vars, label="TD(0)", marker='x')
    axs[1].set_ylabel("Variance")
    axs[1].set_xlabel("State")
    axs[1].set_title("Monte Carlo vs TD(0) Value Estimates (Variance)")
    axs[1].legend()
    axs[1].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder_name, "bias_variance_comparison.png"))
    plt.close()