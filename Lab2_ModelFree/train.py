import argparse
import pickle
from env.gridworld_env import GridWorldEnv

# model free pred
from algos.model_free_prediction import run_prediction_experiment
# model free control
from algos.sarsa import sarsa
from algos.q_learning import q_learning

def save_policy(pi, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pi, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        choices=["mf_pred", "mc", "sarsa", "q_learning"],
                        help="Choose algorithm: mf_pred for model free prediction, or mc, sarsa, q_learning for control.")
    parser.add_argument('--map_size', type=int, default=6)
    parser.add_argument('--map', type=str, default=None, help="Map to run.")
    parser.add_argument('--save_name', type=str, default=None,
                        help="Filename of saving policy to pkl file")
    parser.add_argument('--render', action='store_true', help="Render environment during training")

    # parameters for mf_pred
    parser.add_argument('--episodes', type=int, default=1000, help="Num of episodes")
    parser.add_argument('--runs', type=int, default=30, help="Num to run experiments")
    parser.add_argument('--every_visit', action='store_true', help="Use every visit in MC")
    parser.add_argument('--use_incr_mean', action='store_true', help="Use incremental mean in MC")

    # parameters
    parser.add_argument('--gamma', type=float, default=0.99, help="Gamma value")
    parser.add_argument('--alpha', type=float, default=0.1, help="Alpha value")

    args = parser.parse_args()

    map_name = args.map if args.map else f"map_{args.map_size}.json"
    env = GridWorldEnv(width=args.map_size, height=args.map_size, map_file=map_name)

    print(f"=== Running {args.algo.upper()} ===")
    
    if args.algo == 'mf_pred':
        kwargs = {
            "save_name": args.save_name,
            "episodes": args.episodes,
            "runs": args.runs,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "first_visit": not args.every_visit,
            "use_incremental_mean": args.use_incr_mean
        }
        run_prediction_experiment(env, **kwargs)
        return
    
    else:
        mfc_algo_func_dict = {
            'sarsa': sarsa,
            'q_learning': q_learning,
        }
        if args.algo not in mfc_algo_func_dict:
            raise NotImplementedError(f"Algo should be either mf_pred, sarsa, and q_learning. Current one: {args.algo}")
        
        kwargs = {
            "save_name": args.save_name,
            "episodes": args.episodes,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "render": args.render,
        }
        algo_func = mfc_algo_func_dict[args.algo]
        _, pi = algo_func(env, **kwargs)

        save_path = f"checkpoints/policy_{args.algo}_{args.save_name}.pkl"
        save_policy(pi, save_path)
        print(f"Policy saved to {save_path}")

if __name__ == "__main__":
    main() 



