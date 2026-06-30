import matplotlib.pyplot as plt
import numpy as np
import torch


def log_map(writer, env):
    grid = env.grid
    height, width = grid.shape
    color_map = {
        0: [220, 220, 220],
        1: [50, 50, 50],
        2: [200, 0, 0],
        3: [0, 200, 0],
    }
    map_img = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in color_map.items():
        map_img[grid == value] = color

    map_tensor = torch.tensor(map_img.transpose(2, 0, 1), dtype=torch.uint8)
    writer.add_image('Map', map_tensor, 0, dataformats='CHW')
    return map_img, height, width


def plot_continuous_policy(writer, env, agent, map_img, height, width, resolution, ep, step=3):
    xs = np.arange(resolution / 2, width, resolution)
    ys = np.arange(resolution / 2, height, resolution)
    grid_states = np.stack([[y, x] for y in ys for x in xs], axis=0)

    state_tensor = torch.tensor(grid_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        means, _ = agent.policy(state_tensor)

    means = means.cpu().numpy().reshape(len(ys), len(xs), 2)
    means = np.clip(means, -1.0, 1.0)

    norms = np.linalg.norm(means, axis=2)
    dirs = np.zeros_like(means)
    mask = norms > 1e-6
    dirs[mask] = means[mask] / norms[mask][..., None]

    x_grid, y_grid = np.meshgrid(xs, ys)
    sampled_x = x_grid[::step, ::step]
    sampled_y = y_grid[::step, ::step]
    sampled_dirs = dirs[::step, ::step]
    sampled_norms = norms[::step, ::step]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_img, origin='upper', extent=[0, width, height, 0], alpha=0.6, zorder=0)

    for (i, j), norm in np.ndenumerate(sampled_norms):
        x = sampled_x[i, j]
        y = sampled_y[i, j]
        dy, dx = sampled_dirs[i, j]
        length = norm * 0.1
        ax.arrow(
            x,
            y,
            dx * length,
            dy * length,
            head_width=length * 0.7,
            head_length=length * 0.5,
            fc='blue',
            ec='blue',
            width=0.007,
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.set_title('PolicyArrows')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)
