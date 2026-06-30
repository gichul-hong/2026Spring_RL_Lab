import argparse
import importlib
import os
from env.gridworld_c2 import GridWorldEnv_c2

# algo name mapped with agent classes
AGENT_MAP = {
    'reinforce': 'REINFORCEAgent',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(AGENT_MAP.keys()), required=True)
    parser.add_argument('--save_name', type=str, default='example',
                        help="Filename of saving policy to pth file")
    parser.add_argument('--map', type=str, default='map1')
    parser.add_argument('--iter', type=int, default=10)
    args = parser.parse_args()

    config_path = os.path.join('configs', f'{args.map}.yaml')

    mod = importlib.import_module(f'algos.{args.algo}')
    AgentClass = getattr(mod, AGENT_MAP[args.algo])

    env = GridWorldEnv_c2(config_path)

    # load agent and checkpoint
    agent = AgentClass(env)
    os.makedirs('checkpoints', exist_ok=True)
    agent.load(os.path.join('checkpoints', f'{args.algo}_{args.save_name}.pth'))

    # run test
    for _ in range(args.iter):
        state = env.reset()
        done = False
        while not done:
            # eval=True (no exploration)
            action = agent.inference(state)
            state, reward, done, _ = env.step(action)
            env.render()
    env.close()

if __name__ == '__main__':
    main()
