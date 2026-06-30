import argparse
import pickle
from env.gridworld_env import GridWorldEnv
from env.gridworld_mdp import GridWorldMDP

# 알고리즘 로드
from dynamic_programming import policy_iteration, value_iteration

def save_policy(pi, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pi, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        choices=["vi", "pi"],
                        help="Choose algorithm: vi, pi")
    parser.add_argument('--map_size', type=int, default=6)
    parser.add_argument('--save_name', type=str, default=None,
                        help="Filename of saving policy to pkl file")
    # parameters
    parser.add_argument('--gamma', type=float, default=0.95, help="Gamma value")
    parser.add_argument('--max_iterations', type=int, default=80, help="Max iterations")
    parser.add_argument('--theta', type=float, default=1e-3, help="Theta value for value iteration")

    args = parser.parse_args()

    map_name = f"map_{args.map_size}.json"
    env = GridWorldEnv(width=args.map_size, height=args.map_size, map_file=map_name)

    print(f"=== Running {args.algo.upper()} ===")
    
    mdp = GridWorldMDP(env)

    algo_func_dict = {
        'pi': policy_iteration,
        'vi': value_iteration,
    }

    algo_func = algo_func_dict[args.algo]
    
    _, pi = algo_func(
            mdp,
            policy_name=args.save_name, 
            gamma=args.gamma, 
            theta=args.theta, 
            max_iterations=args.max_iterations
        )
        
    save_path = f"checkpoints/policy_{args.algo}_{args.save_name}.pkl"
    save_policy(pi, save_path)
    print(f"Policy saved to {save_path}")

if __name__ == "__main__":
    main() 



