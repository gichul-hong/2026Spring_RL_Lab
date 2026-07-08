import os
import argparse
import torch
from env.gridworld_c2 import GridWorldEnv_c2
from algos.reinforce import REINFORCEAgent

DEFAULT_MAX_STEPS = {
    'hw_map1.yaml': 150,
    'hw_map2.yaml': 200,
    'hw_map3.yaml': 250,
}


def resolve_max_steps(map_file: str, max_steps: int | None) -> int:
    if max_steps is not None:
        return max_steps
    return DEFAULT_MAX_STEPS.get(os.path.basename(map_file), 300)


def build_run_name(map_file: str) -> str:
    map_stem = os.path.splitext(os.path.basename(map_file))[0]
    return f'reinforce_{map_stem}'


def resolve_model_path(map_file: str, model_file: str | None) -> str:
    if model_file is None:
        model_file = f'{build_run_name(map_file)}.pth'
    if os.path.isabs(model_file) or os.path.dirname(model_file):
        return model_file
    return os.path.join('checkpoints', model_file)


def main():
    parser = argparse.ArgumentParser(description='Test trained REINFORCE policy')
    parser.add_argument('--map', type=str, default='map1.yaml', help='Map configuration file')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--attempts', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum steps per episode')
    parser.add_argument('--headless', action='store_true', help='Disable rendering window')
    args = parser.parse_args()
    max_steps = resolve_max_steps(args.map, args.max_steps)

    # 환경 초기화
    config_path = os.path.join('configs', args.map)
    env = GridWorldEnv_c2(config_path, headless=args.headless)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 에이전트 초기화 및 모델 로드
    device = torch.device('cpu')
    agent = REINFORCEAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    model_path = resolve_model_path(args.map, args.model)
    agent.load(model_path)

    map_name = os.path.basename(args.map)
    successes = 0
    failures = 0

    for ep in range(1, args.attempts + 1):
        state = env.reset()
        agent.reset_episode()
        done = False
        reward = 0
        reward_accum = 0.0

        for step in range(max_steps):
            # 학습된 Gaussian policy에서 행동 샘플링
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor = agent.select_action(state_tensor)
            action = action_tensor.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            reward_accum += reward
            state = next_state

            if not args.headless:
                env.render()

            if done:
                break

        # 목표 달성 여부 판단 (보상 100 받으면 성공)
        if reward == 100.0:
            successes += 1
        else:
            failures += 1

        print(f"Episode {ep}: Reward {reward_accum:.2f}, {'Success' if reward == 100.0 else 'Failure'}")

    env.close()
    success_rate = 100.0 * successes / args.attempts if args.attempts > 0 else 0.0

    # 최종 결과 출력
    print('--- Test Summary ---')
    print(f"Map: {map_name}")
    print(f"Attempts: {args.attempts}")
    print(f"Max Steps: {max_steps}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success Rate: {success_rate:.2f}%")


if __name__ == '__main__':
    main()
