import os
import numpy as np
import yaml
import pygame
from typing import Optional, Union, Sequence

class GridWorldEnv_c2:
    """
    GridWorld_c2 environment: continuous actions (Δrow, Δcol) scaled by step_size_m,
    precise collider using pygame.Rect, pixel/meter separation, robust map parsing,
    headless support, and optional gym.Env compatibility.
    """
    def __init__(
        self,
        config_path: str,
        cell_size_px: int = 100,
        cell_size_m: float = 1.0,
        step_size_m: float = 1.0,
        agent_radius_px: Optional[int] = 20,
        headless: bool = False
    ):
        # ---- Load and parse map config ----
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.width = cfg['width']
        self.height = cfg['height']
        raw_map = cfg['map']
        if isinstance(raw_map[0], str):
            self.grid = np.array([[int(c) for c in row] for row in raw_map], dtype=int)
        else:
            self.grid = np.array(raw_map, dtype=int)

        # ---- Determine start cell ----
        if cfg.get('start') is not None:
            self.start_cell = tuple(cfg['start'])
        else:
            zeros = np.argwhere(self.grid == 0)
            if zeros.size == 0:
                raise ValueError("No empty cell (0) in map to start.")
            self.start_cell = tuple(zeros[0])

        # ---- Physical parameters ----
        self.cell_size_m = cell_size_m
        self.step_size_m = step_size_m

        # ---- Pygame / rendering setup ----
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.cell_size_px = cell_size_px
        if agent_radius_px is None:
            self.agent_radius = int(cell_size_px * 0.4)
        else:
            self.agent_radius = agent_radius_px

        screen_w = self.width * cell_size_px
        screen_h = self.height * cell_size_px
        if not headless:
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("GridWorld_c2 (Continuous Action)")
        else:
            self.screen = None
        self.clock = pygame.time.Clock()

        # ---- Prepare rects for collisions (pixel coords) ----
        self.wall_rects = []
        self.trap_rects = []
        self.goal_rects = []
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c * cell_size_px, r * cell_size_px,
                    cell_size_px, cell_size_px
                )
                t = self.grid[r, c]
                if t == 1:
                    self.wall_rects.append(rect)
                elif t == 2:
                    self.trap_rects.append(rect)
                elif t == 3:
                    self.goal_rects.append(rect)

        # ---- Agent state ----
        self.agent_pos = None  # [row_m, col_m]
        self.done = False


    def reset(self, start_pos: Optional[Sequence[Union[int, float]]] = None) -> np.ndarray:
        # Place agent at center of start cell
        if start_pos is None:
            r, c = self.start_cell
        else:
            r, c = start_pos
        # convert cell coords to meters
        self.agent_pos = np.array([r + 0.5, c + 0.5], dtype=float) * self.cell_size_m
        self.done = False
        return self._get_obs()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.step_size_m
        old_pos = self.agent_pos.copy()
        delta = a

        # Binary search to find t ∈ [0,1] for maximum movement
        lo, hi = 0.0, 1.0
        screen_w = self.width * self.cell_size_px
        screen_h = self.height * self.cell_size_px
        for _ in range(8):
            mid = (lo + hi) / 2
            test_pos = old_pos + delta * mid
            px = test_pos[1] * self.cell_size_px
            py = test_pos[0] * self.cell_size_px
            rect = pygame.Rect(
                px - self.agent_radius, py - self.agent_radius,
                self.agent_radius * 2, self.agent_radius * 2
            )

            # 1) collision with wall
            wall_hit = any(rect.colliderect(w) for w in self.wall_rects)
            # 2) out of bounds
            out_of_bounds = (
                    rect.left < 0 or
                    rect.top < 0 or
                    rect.right > screen_w or
                    rect.bottom > screen_h
            )

            if wall_hit or out_of_bounds:
                hi = mid
            else:
                lo = mid

        # final agent pos
        self.agent_pos = old_pos + delta * lo

        # trap/goal check
        px = self.agent_pos[1] / self.cell_size_m * self.cell_size_px
        py = self.agent_pos[0] / self.cell_size_m * self.cell_size_px
        agent_rect = pygame.Rect(px - self.agent_radius, py - self.agent_radius,
                                 self.agent_radius*2, self.agent_radius*2)

        if any(agent_rect.colliderect(t) for t in self.trap_rects):
            self.done = True
            return self._get_obs(), -100.0, True, {}
        if any(agent_rect.colliderect(g) for g in self.goal_rects):
            self.done = True
            return self._get_obs(), 100.0, True, {}
        return self._get_obs(), -1.0, False, {}

    def _no_move(self):
        # Stay in place with -1 reward
        return self._get_obs(), -1.0, False, {}

    def _get_obs(self) -> np.ndarray:
        # Return continuous position (row, col) in meters
        return self.agent_pos.copy()

    def render(self, tick: int = 30) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # draw cells
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c*self.cell_size_px, r*self.cell_size_px,
                                    self.cell_size_px, self.cell_size_px)
                t = self.grid[r, c]
                if t == 1:
                    color = (50,50,50)
                elif t == 2:
                    color = (200,0,0)
                elif t == 3:
                    color = (0,200,0)
                else:
                    color = (220,220,220)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150,150,150), rect, 1)
        # draw agent
        px = self.agent_pos[1] / self.cell_size_m * self.cell_size_px
        py = self.agent_pos[0] / self.cell_size_m * self.cell_size_px
        pygame.draw.circle(self.screen, (0,0,255), (int(px), int(py)), self.agent_radius)
        pygame.display.flip()
        self.clock.tick(tick)

    def close(self) -> None:
        pygame.quit()
