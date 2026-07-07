import os
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
    return DEFAULT_MAX_STEPS.get(os.path.basename(map_file), 100)


def build_run_name(map_file: str) -> str:
    map_stem = os.path.splitext(os.path.basename(map_file))[0]
    return f'reinforce_{map_stem}'


def log_map(writer, env):
    grid = env.grid
    H, W = grid.shape
    color_map = {
        0: [220, 220, 220],
        1: [50, 50, 50],
        2: [200, 0, 0],
        3: [0, 200, 0],
    }
    map_img = np.zeros((H, W, 3), dtype=np.uint8)
    for v, c in color_map.items():
        map_img[grid == v] = c
    map_tensor = torch.tensor(map_img.transpose(2, 0, 1), dtype=torch.uint8)
    writer.add_image('Map', map_tensor, 0, dataformats='CHW')
    return map_img, H, W


def plot_continuous_policy(writer, env, agent, map_img, H, W, resolution, ep, step=3):
    xs = np.arange(resolution / 2, W, resolution)
    ys = np.arange(resolution / 2, H, resolution)
    grid_states = np.stack([[y, x] for y in ys for x in xs], axis=0)

    ray_len = env.ray_length_m
    ray_features = np.ones((grid_states.shape[0], env.ray_num), dtype=np.float32) * ray_len
    full_states = np.concatenate([grid_states, ray_features], axis=1)

    st = torch.tensor(full_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        norm_st = agent.normalize_state(st, update=False)
        means, _ = agent.policy(norm_st)
    means = means.cpu().numpy().reshape(len(ys), len(xs), 2)
    means = np.clip(means, -1.0, 1.0)

    norms = np.linalg.norm(means, axis=2)
    dirs = np.zeros_like(means)
    mask = norms > 1e-6
    dirs[mask] = means[mask] / norms[mask][..., None]

    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ds = dirs[::step, ::step]
    Ns = norms[::step, ::step]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_img, origin='upper', extent=[0, W, H, 0], alpha=0.6, zorder=0)
    for (i, j), norm in np.ndenumerate(Ns):
        x = Xs[i, j]
        y = Ys[i, j]
        dy, dx = Ds[i, j]
        length = norm * 0.7
        ax.arrow(
            x, y,
            dx * length, dy * length,
            head_width=length * 0.7,
            head_length=length * 0.5,
            width=0.007,
        )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.set_title('PolicyArrows (Continuous)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)


def train(args):
    run_name = build_run_name(args.map)
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))

    config_path = os.path.join('configs', args.map)
    env = GridWorldEnv_c2(config_path, headless=not args.render)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = REINFORCEAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        log_std_init=args.log_std_init,
        total_episodes=max(args.episodes, 200000),
    )
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        agent.load(args.pretrained)
    max_steps = resolve_max_steps(args.map, args.max_steps)

    map_img, H, W = log_map(writer, env)

    grid_dist = np.full(env.grid.shape, 999.0)
    goals = np.argwhere(env.grid == 3)
    queue = []
    for r, c in goals:
        grid_dist[r, c] = 0.0
        queue.append((r, c))
    while queue:
        r, c = queue.pop(0)
        d = grid_dist[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width:
                if env.grid[nr, nc] not in [1, 2]:
                    if grid_dist[nr, nc] > d + 1.0:
                        grid_dist[nr, nc] = d + 1.0
                        queue.append((nr, nc))

    # 함정에 인접한 셀은 초기 시작점에서 제외 (agent가 시작 즉시 함정에 빠지는 것 방지)
    trap_adjacent = np.zeros(env.grid.shape, dtype=bool)
    traps = np.argwhere(env.grid == 2)
    for tr, tc in traps:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    trap_adjacent[nr, nc] = True

    def get_start_cells_by_max_distance(max_d, exclude_trap_adj=True):
        cells = []
        for r in range(env.height):
            for c in range(env.width):
                if 0.0 <= grid_dist[r, c] <= max_d:
                    if exclude_trap_adj and trap_adjacent[r, c]:
                        continue
                    cells.append([r, c])
        return cells

    valid_dists = grid_dist[(env.grid != 1) & (env.grid != 2) & (grid_dist < 999.0)]
    max_possible_dist = np.max(valid_dists) if len(valid_dists) > 0 else 10.0
    print(f"Max reachable distance: {max_possible_dist}")

    success_window = 100
    recent_successes: list[bool] = []

    best_success_rate = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    best_path = f'checkpoints/{run_name}_best.pth'

    early_stop_threshold = 0.95
    early_stop_patience = 50
    early_stop_counter = 0

    for ep in range(1, args.episodes + 1):
        progress_ratio = ep / args.episodes

        is_eval_episode = False

        if args.no_curriculum:
            # Without curriculum, just start from (0,0) or do eval-like
            if np.random.rand() < 0.5:
                is_eval_episode = True
                state = env.reset()
            else:
                state = env.reset(start_pos=[0, 0])
        else:
            if progress_ratio <= 0.4:
                current_max_dist = 3.0 + (max_possible_dist - 3.0) * (progress_ratio / 0.4)
                candidates = get_start_cells_by_max_distance(current_max_dist,
                                                            exclude_trap_adj=True)

                if not candidates:
                    candidates = [[0, 0]]

                start_cell = candidates[np.random.choice(len(candidates))]
                state = env.reset(start_pos=start_cell)
            else:
                if np.random.rand() < 0.5:
                    is_eval_episode = True
                    state = env.reset()
                else:
                    candidates = get_start_cells_by_max_distance(max_possible_dist,
                                                                 exclude_trap_adj=False)
                    if not candidates:
                        candidates = [[0, 0]]
                    start_cell = candidates[np.random.choice(len(candidates))]
                    state = env.reset(start_pos=start_cell)

        agent.reset_episode()
        total_R = 0.0
        done = False

        for t in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)
            if is_eval_episode:
                with torch.no_grad():
                    action_tensor = agent.select_action(state_tensor, eval_mode=True)
            else:
                action_tensor = agent.select_action(state_tensor)
            action = action_tensor.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            total_R += reward

            agent.rewards.append(reward)

            state = next_state
            if args.render:
                env.render()
            if done:
                break

        agent.set_done(done)

        loss = agent.finish_episode()
        writer.add_scalar('Loss', loss, ep)
        writer.add_scalar('Loss/Actor', agent.last_actor_loss, ep)
        writer.add_scalar('Loss/Critic', agent.last_critic_loss, ep)
        writer.add_scalar('Entropy', agent.last_entropy, ep)
        writer.add_scalar('Reward', total_R, ep)
        writer.add_scalar('LearningRate', agent.get_lr(), ep)
        writer.add_scalar('EntropyCoeff', agent.entropy_coeff, ep)

        if is_eval_episode:
            is_success = (total_R > 0)
            recent_successes.append(is_success)
            if len(recent_successes) > success_window:
                recent_successes.pop(0)

            success_rate = sum(recent_successes) / len(recent_successes)
            writer.add_scalar('SuccessRate', success_rate, ep)

            if len(recent_successes) >= success_window:
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    agent.save(best_path)

                if success_rate >= early_stop_threshold:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f"\n[Early Stop] Eval SuccessRate >= {early_stop_threshold:.0%} "
                              f"for {early_stop_patience} consecutive. Training finished.")
                        break
                else:
                    early_stop_counter = 0

        if ep % 100 == 0:
            current_sr = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
            if args.no_curriculum:
                phase_str = "No Curriculum"
            else:
                phase_str = "Mixed Phase" if progress_ratio > 0.4 else "Curriculum Phase"
            print(f"[REINFORCEAgent] Episode: {ep}, Reward: {total_R:.2f}, "
                  f"Eval SuccessRate: {current_sr:.1%}, LR: {agent.get_lr():.2e} ({phase_str})")

        if ep % args.heatmap_interval == 0:
            plot_continuous_policy(writer, env, agent, map_img, H, W, args.resolution, ep)

    agent.save(f'checkpoints/{run_name}.pth')
    print(f"\n[Saved]")
    print(f"  Final model: checkpoints/{run_name}.pth")
    print(f"  Best model: {best_path} (SuccessRate: {best_success_rate:.1%})")
    writer.close()
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', type=str, default='map1.yaml')
    p.add_argument('--episodes', type=int, default=200000)
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--render', action='store_true')
    p.add_argument('--logdir', type=str, default='runs')
    p.add_argument('--heatmap-interval', type=int, default=500)
    p.add_argument('--resolution', type=float, default=0.1)
    p.add_argument('--pretrained', type=str, default=None, help='Path to pretrained checkpoint')
    p.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--log-std-init', type=float, default=0.5)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()