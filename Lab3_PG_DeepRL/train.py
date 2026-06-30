import os
import argparse
import importlib
# algo name mapped with trainer classes
TRAINER_MAP = {
    'dqn':       'DQNTrainer',
    'reinforce': 'REINFORCETrainer',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(TRAINER_MAP.keys()), required=True, help="Choose algorithm to run. Choices: [reinforce, dqn]")
    parser.add_argument('--map', type=str, default='map1', help="Map to run. Choices: [map0, map1, map2, map3]")
    parser.add_argument('--save_name', type=str, default=None,
                        help="Filename of saving policy to pth file")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--logdir', type=str, default='runs')

    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=100)
    # for plotting
    parser.add_argument('--heatmap-interval', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.1)
    # training seed
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # create log dir if needed
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)

    # load trainer 
    mod = importlib.import_module(f'algos.{args.algo}')
    TrainerClass = getattr(mod, TRAINER_MAP[args.algo])
    trainer = TrainerClass(args)

    trainer.initialize()
    trainer.train()
    trainer.save()

if __name__=='__main__':
    main()
