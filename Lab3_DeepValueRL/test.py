import argparse
import importlib
import os


AGENT_MAP = {
    'deepsarsa': 'DeepSARSAAgent',
    'dqn': 'DQNAgent',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(AGENT_MAP.keys()), required=True)
    parser.add_argument('--save_name', type=str, default='example', help='Filename of saved policy pth file')
    parser.add_argument('--map', type=str, default='map2')
    parser.add_argument('--step-size', type=float, default=1.0)
    parser.add_argument('--iter', type=int, default=10)
    args = parser.parse_args()

    from env.gridworld_c1 import GridWorldEnv_c1

    config_path = os.path.join('configs', f'{args.map}.yaml')

    mod = importlib.import_module(f'algos.{args.algo}')
    agent_class = getattr(mod, AGENT_MAP[args.algo])

    env = GridWorldEnv_c1(config_path, step_size_m=args.step_size)
    agent = agent_class(env)
    agent.load(os.path.join('checkpoints', f'{args.algo}_{args.save_name}.pth'))

    for _ in range(args.iter):
        state = env.reset()
        done = False
        while not done:
            action = agent.inference(state)
            state, reward, done, _ = env.step(action)
            env.render()
    env.close()


if __name__ == '__main__':
    main()
