import matplotlib.pyplot as plt
import torch
import numpy as np

def log_map(writer, env):
    """
    Record background map image to TensorBoard based on environment's grid information.
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
    # CHW, uint8
    map_tensor = torch.tensor(map_img.transpose(2,0,1), dtype=torch.uint8)
    writer.add_image('Map', map_tensor, 0, dataformats='CHW')
    return map_img, H, W

def compute_value_grid(env, qnet, resolution, device='cpu'):
    """
    Compute V(s) at all state coordinates, return grid coordinates.
    """
    H, W = env.height, env.width

    # 1) Generate grid coordinates
    xs = np.arange(resolution/2, W, resolution)
    ys = np.arange(resolution/2, H, resolution)
    n_x, n_y = len(xs), len(ys)
    
    # 2) Generate all cell coordinates
    states = np.stack([[y, x] for y in ys for x in xs], axis=0).astype(np.float32)

    # 3) Apply same normalization as agent
    state_scale = np.array([H, W], dtype=np.float32)
    states_norm = states / state_scale  # shape (n_y*n_x, 2)

    qnet.eval()  # Set to evaluation mode in case of batchnorm/dropout
    with torch.no_grad():
        st = torch.tensor(states_norm, dtype=torch.float32, device=device)
        q = qnet(st).cpu().numpy()  # shape (n_y*n_x, A)

    # 4) Reshape back to grid form
    q = q.reshape(n_y, n_x, -1)
    values = q.max(axis=2)

    return values, xs, ys, q

def plot_value_heatmap(writer, values, xs, ys, ep):
    """
    Generate StateValueHeatmap Figure and record to TensorBoard.
    """
    fig, ax = plt.subplots(figsize=(4,4))
    extent = [
        xs[0] - (xs[1]-xs[0])/2,
        xs[-1]+ (xs[1]-xs[0])/2,
        ys[-1]+ (ys[1]-ys[0])/2,
        ys[0] - (ys[1]-ys[0])/2
    ]
    im = ax.imshow(values,
                   origin='upper',
                   interpolation='bilinear',
                   cmap='viridis',
                   extent=extent)
    fig.colorbar(im, ax=ax, label='V(s)')
    ax.set_title('StateValueHeatmap')
    writer.add_figure('StateValueHeatmap', fig, global_step=ep)
    plt.close(fig)

def plot_discrete_policy(writer, env, qvals, xs, ys, map_img, H, W, ep, step=3):
    """
    For DeepSARSA/DQN: display argmax Q action direction as arrows at each grid cell.
    """
    # Arrow directions (unit vectors over 8 actions)
    deltas = env.deltas  # (8,2) in meters
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    unit = deltas / norms * 0.2
    # Optimal action indices
    actions = qvals.argmax(axis=2)  # (n_y, n_x)
    # Vector components
    U = unit[actions][:,:,1]
    V = -unit[actions][:,:,0]
    # subsample
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]; Ys = Y[::step, ::step]
    Us = U[::step, ::step]; Vs = V[::step, ::step]
    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    # map background
    ax.imshow(map_img, origin='upper', extent=[0,W,H,0], alpha=0.6, zorder=0)
    ax.quiver(Xs, Ys, Us, Vs,
              color='blue', scale_units='xy', scale=1,
              width=0.005, zorder=1)
    ax.set_xlim(0,W); ax.set_ylim(H,0)
    ax.set_title('PolicyArrows (Discrete)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)

def plot_continuous_policy(writer, env, agent, map_img, H, W, resolution, ep, step=3):
    """
    Continuous policy visualization for REINFORCE:
    - On map background, at subsampled states
      * Draw ax.arrow in mean action direction
      * Arrow length ∝ action norm (scaled by 0.1)
      * Shaft width fixed at 0.01
    """
    # 1) Generate sampling grid
    xs = np.arange(resolution/2, W, resolution)
    ys = np.arange(resolution/2, H, resolution)
    grid_states = np.stack([[y, x] for y in ys for x in xs], axis=0)

    # 2) Extract mean values from policy network
    st = torch.tensor(grid_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        means, _ = agent.policy(st)               # (N, 2)
    means = means.cpu().numpy().reshape(len(ys), len(xs), 2)
    means = np.clip(means, -1.0, 1.0)

    # 3) Calculate norm and direction (unit)
    norms = np.linalg.norm(means, axis=2)        # (n_y, n_x)
    dirs = np.zeros_like(means)
    mask = norms > 1e-6
    dirs[mask] = means[mask] / norms[mask][..., None]

    # 4) subsample
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ds = dirs[::step, ::step]                    # (m, n, 2)
    Ns = norms[::step, ::step]                   # (m, n)

    # 5) Draw
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_img, origin='upper', extent=[0, W, H, 0], alpha=0.6, zorder=0)

    # For each arrow
    for (i, j), norm in np.ndenumerate(Ns):
        x = Xs[i, j]
        y = Ys[i, j]
        dy, dx = Ds[i, j]  # unit vector [dr, dc]
        length = norm * 0.1  # Scale 0~1 norm by 0.1
        # Arrow in dx, dy direction
        ax.arrow(
            x, y,
            dx * length, dy * length,   # Consider screen y-axis flip
            head_width=length * 0.7,
            head_length=length * 0.5,
            fc='blue', ec='blue',
            width=0.007                  # Fixed shaft width
        )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.set_title('PolicyArrows (Continuous)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)