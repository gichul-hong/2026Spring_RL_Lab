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
    """
    환경의 grid 정보를 바탕으로 배경 맵 이미지를 TensorBoard에 기록.
    """
    grid = env.grid
    H, W = grid.shape
    color_map = {
        0: [220,220,220],  # normal
        1: [50,50,50],     # wall
        2: [200,0,0],      # trap
        3: [0,200,0],      # goal
    }
    map_img = np.zeros((H, W, 3), dtype=np.uint8)
    for v, c in color_map.items():
        map_img[grid == v] = c
    map_tensor = torch.tensor(map_img.transpose(2,0,1), dtype=torch.uint8)
    writer.add_image('Map', map_tensor, 0, dataformats='CHW')
    return map_img, H, W


def plot_continuous_policy(writer, env, agent, map_img, H, W, resolution, ep, step=3):
    """
    연속 정책 시각화: 배경 map 위에 subsample된 상태에서 평균 행동 방향으로 화살표 그리기
    """
    # 1) 그리드 지점 생성 (단위: 미터)
    xs = np.arange(resolution/2, W, resolution)
    ys = np.arange(resolution/2, H, resolution)
    grid_states = np.stack([[y, x] for y in ys for x in xs], axis=0)

    # 2) ray 정보를 최대 거리(감지 없음)로 패딩
    ray_len = env.ray_length_m
    ray_features = np.ones((grid_states.shape[0], env.ray_num), dtype=np.float32) * ray_len
    full_states = np.concatenate([grid_states, ray_features], axis=1)

    # 3) 정책 네트워크 호출 (정규화된 상태 사용)
    st = torch.tensor(full_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        norm_st = agent.normalize_state(st, update=False)
        means, _ = agent.policy(norm_st)
    means = means.cpu().numpy().reshape(len(ys), len(xs), 2)
    means = np.clip(means, -1.0, 1.0)

    # 4) 방향과 크기 계산
    norms = np.linalg.norm(means, axis=2)
    dirs = np.zeros_like(means)
    mask = norms > 1e-6
    dirs[mask] = means[mask] / norms[mask][..., None]

    # 5) subsampling
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ds = dirs[::step, ::step]
    Ns = norms[::step, ::step]

    # 6) 시각화
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
            width=0.007
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
    # 1) SummaryWriter, 환경, 에이전트 초기화
    run_name = build_run_name(args.map)
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))

    config_path = os.path.join('configs', args.map)
    env = GridWorldEnv_c2(config_path, headless=not args.render)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = REINFORCEAgent(
        state_dim=state_dim, action_dim=action_dim,
        total_episodes=args.episodes
    )
    max_steps = resolve_max_steps(args.map, args.max_steps)

    # 2) 맵 이미지 기록
    map_img, H, W = log_map(writer, env)

    # BFS로 모든 셀에서 골(3)까지의 최단 거리 계산 (커리큘럼 러닝 거리 측정용)
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

    # 거리가 d 이하인 빈 셀 목록 수집
    def get_start_cells_by_max_distance(max_d):
        cells = []
        for r in range(env.height):
            for c in range(env.width):
                if 0.0 <= grid_dist[r, c] <= max_d:
                    cells.append([r, c])
        return cells

    valid_dists = grid_dist[(env.grid != 1) & (env.grid != 2) & (grid_dist < 999.0)]
    max_possible_dist = np.max(valid_dists) if len(valid_dists) > 0 else 10.0
    print(f"맵의 최대 도달 가능 거리: {max_possible_dist}")

    # 성공률 이동 평균을 위한 버퍼
    success_window = 100
    recent_successes: list[bool] = []

    # Best model 추적
    best_success_rate = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    best_path = f'checkpoints/{run_name}_best.pth'

    # Early stopping: 평가 에피소드 기준 연속 달성 횟수
    early_stop_threshold = 0.95
    early_stop_patience = 50  # 50번의 평가 에피소드 동안 연속으로 임계치 이상
    early_stop_counter = 0

    # 3) 학습 루프
    for ep in range(1, args.episodes + 1):
        progress_ratio = ep / args.episodes
        
        is_eval_episode = False
        
        # 커리큘럼 페이즈 (0% ~ 40%): 최대 거리를 2.0에서 max_possible_dist까지 선형적으로 점진적 확장
        if progress_ratio <= 0.4:
            current_max_dist = 2.0 + (max_possible_dist - 2.0) * (progress_ratio / 0.4)
            candidates = get_start_cells_by_max_distance(current_max_dist)
            
            # Frontier 기반 샘플링: 현재 해금된 가장 먼 거리(어려운 곳) 위주로 집중 학습
            frontier = [c for c in candidates if grid_dist[c[0], c[1]] >= current_max_dist - 2.0]
            if not frontier:
                frontier = candidates if candidates else [[0, 0]]
                
            start_cell = frontier[np.random.choice(len(frontier))]
            state = env.reset(start_pos=start_cell)
        else:
            # 40% 이후 (혼합 페이즈): 50% 공식 시작점(평가), 50% 랜덤 시작점(망각 방지)
            if np.random.rand() < 0.5:
                is_eval_episode = True
                state = env.reset()  # 공식 시작점
            else:
                candidates = get_start_cells_by_max_distance(max_possible_dist)
                if not candidates:
                    candidates = [[0, 0]]
                start_cell = candidates[np.random.choice(len(candidates))]
                state = env.reset(start_pos=start_cell)

        agent.reset_episode()
        total_R = 0.0
        done = False

        for t in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)
            action_tensor = agent.select_action(state_tensor)
            action = action_tensor.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            total_R += reward

            # 순수 환경 보상만 저장 (환경 요소를 절대로 수정하지 않음)
            agent.rewards.append(reward)

            state = next_state
            if args.render:
                env.render()
            if done:
                break

        # GAE bootstrap 결정을 위해 done 플래그 전달
        agent.set_done(done)

        # 에피소드 종료 시 정책 업데이트 및 로깅
        loss = agent.finish_episode()
        writer.add_scalar('Loss', loss, ep)
        writer.add_scalar('Loss/Actor', agent.last_actor_loss, ep)
        writer.add_scalar('Loss/Critic', agent.last_critic_loss, ep)
        writer.add_scalar('Entropy', agent.last_entropy, ep)
        writer.add_scalar('Reward', total_R, ep)
        writer.add_scalar('LearningRate', agent.get_lr(), ep)
        writer.add_scalar('EntropyCoeff', agent.entropy_coeff, ep)

        # 성공률 이동 평균 계산 및 로깅
        # 평가 에피소드일 때만 성공률 이동 평균 업데이트 및 조기 종료 체크
        if is_eval_episode:
            is_success = (total_R > 0)  # 목표 도달 시 reward 합 > 0
            recent_successes.append(is_success)
            if len(recent_successes) > success_window:
                recent_successes.pop(0)
                
            success_rate = sum(recent_successes) / len(recent_successes)
            writer.add_scalar('SuccessRate', success_rate, ep)
            
            if len(recent_successes) >= success_window:
                # Best model 저장
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    agent.save(best_path)

                # Early stopping 체크
                if success_rate >= early_stop_threshold:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f"\n[Early Stop] 평가 에피소드 SuccessRate >= {early_stop_threshold:.0%} 연속 {early_stop_patience}회 달성. 학습 종료.")
                        break
                else:
                    early_stop_counter = 0

        if ep % 100 == 0:
            current_sr = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
            phase_str = "Mixed Phase" if progress_ratio > 0.4 else "Curriculum Phase"
            print(f"[REINFORCEAgent] Episode: {ep}, Reward: {total_R:.2f}, "
                  f"Eval SuccessRate: {current_sr:.1%}, LR: {agent.get_lr():.2e} ({phase_str})")

        # 주기적 정책 시각화
        if ep % args.heatmap_interval == 0:
            plot_continuous_policy(writer, env, agent, map_img, H, W, args.resolution, ep)

    # 4) 최종 모델 저장 및 종료
    agent.save(f'checkpoints/{run_name}.pth')
    print(f"\n[저장 완료]")
    print(f"  최종 모델: checkpoints/{run_name}.pth")
    print(f"  최고 모델: {best_path} (SuccessRate: {best_success_rate:.1%})")
    writer.close()
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', type=str, default='map1.yaml')
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--render', action='store_true')
    p.add_argument('--logdir', type=str, default='runs')
    p.add_argument('--heatmap-interval', type=int, default=100)
    p.add_argument('--resolution', type=float, default=0.1)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
