import argparse
import importlib
import os


TRAINER_MAP = {
    'deepsarsa': 'DeepSARSATrainer',
    'dqn': 'DQNTrainer',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo',
        choices=list(TRAINER_MAP.keys()),
        required=True,
        help='Choose algorithm to run. Choices: [deepsarsa, dqn]',
    )
    parser.add_argument('--map', type=str, default='map2', help='Map to run. Choices: [map0, map1, map2, map3]')
    parser.add_argument('--save_name', type=str, default='example', help='Filename of saving policy to pth file')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--logdir', type=str, default='runs')
    parser.add_argument('--step-size', type=float, default=1.0, help='Movement length in meters for each discrete action')

    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--heatmap-interval', type=int, default=100)
    parser.add_argument('--resolution', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    mod = importlib.import_module(f'algos.{args.algo}')
    trainer_class = getattr(mod, TRAINER_MAP[args.algo])
    trainer = trainer_class(args)

    trainer.initialize()
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    main()
