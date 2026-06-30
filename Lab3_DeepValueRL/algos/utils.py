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


def compute_value_grid(env, qnet, resolution, device='cpu'):
    height, width = env.height, env.width

    xs = np.arange(resolution / 2, width, resolution)
    ys = np.arange(resolution / 2, height, resolution)
    states = np.stack([[y, x] for y in ys for x in xs], axis=0).astype(np.float32)

    state_scale = np.array([height, width], dtype=np.float32)
    states_norm = states / state_scale

    qnet.eval()
    with torch.no_grad():
        state_tensor = torch.tensor(states_norm, dtype=torch.float32, device=device)
        q_values = qnet(state_tensor).cpu().numpy()

    q_values = q_values.reshape(len(ys), len(xs), -1)
    values = q_values.max(axis=2)
    return values, xs, ys, q_values


def plot_value_heatmap(writer, values, xs, ys, ep):
    fig, ax = plt.subplots(figsize=(4, 4))
    extent = [
        xs[0] - (xs[1] - xs[0]) / 2,
        xs[-1] + (xs[1] - xs[0]) / 2,
        ys[-1] + (ys[1] - ys[0]) / 2,
        ys[0] - (ys[1] - ys[0]) / 2,
    ]
    image = ax.imshow(values, origin='upper', interpolation='bilinear', cmap='viridis', extent=extent)
    fig.colorbar(image, ax=ax, label='V(s)')
    ax.set_title('StateValueHeatmap')
    writer.add_figure('StateValueHeatmap', fig, global_step=ep)
    plt.close(fig)


def plot_discrete_policy(writer, env, qvals, xs, ys, map_img, height, width, ep, step=3):
    deltas = env.deltas
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    unit = deltas / norms * 0.2

    actions = qvals.argmax(axis=2)
    u = unit[actions][:, :, 1]
    v = -unit[actions][:, :, 0]

    x_grid, y_grid = np.meshgrid(xs, ys)
    sampled_x = x_grid[::step, ::step]
    sampled_y = y_grid[::step, ::step]
    sampled_u = u[::step, ::step]
    sampled_v = v[::step, ::step]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_img, origin='upper', extent=[0, width, height, 0], alpha=0.6, zorder=0)
    ax.quiver(
        sampled_x,
        sampled_y,
        sampled_u,
        sampled_v,
        color='blue',
        scale_units='xy',
        scale=1,
        width=0.005,
        zorder=1,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title('PolicyArrows')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)
