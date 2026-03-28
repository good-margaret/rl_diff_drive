# """
# Среда дифференциального привода для задачи «прийти в начало координат».

# Кинематика (непрерывная):
#     v  = (v_r + v_l) / 2           — линейная скорость
#     ω  = (v_r - v_l) / L           — угловая скорость
#     ẋ  = v · cos(θ)
#     ẏ  = v · sin(θ)
#     θ̇  = ω
# """

# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces


# class DiffDriveEnv(gym.Env):
#     """
#     Observation (5-мерный вектор):
#         [x/10, y/10, cos(θ), sin(θ), dist/14.14]

#     Action (2-мерный непрерывный):
#         [v_left, v_right] ∈ [-1, 1]

#     Goal: x→0, y→0, θ→0
#     """

#     metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

#     # ── физика ──────────────────────────────────────────────────────────────
#     L         = 1.0    # расстояние между колёсами (м)
#     V_MAX     = 3.0    # макс. скорость колеса (м/с)
#     DT        = 0.05   # шаг интегрирования (с)
#     MAX_STEPS = 600    # max шагов на эпизод

#     # ── условие достижения цели ──────────────────────────────────────────────
#     GOAL_DIST  = 0.30   # м
#     GOAL_ANGLE = 0.15   # рад (~8.6°)

#     # ── границы поля ────────────────────────────────────────────────────────
#     FIELD = 10.0

#     def __init__(self):
#         super().__init__()

#         # непрерывные действия [v_l, v_r]
#         self.action_space = spaces.Box(
#             low=-1.0, high=1.0, shape=(2,), dtype=np.float32
#         )
#         # наблюдение
#         obs_low  = np.array([-1, -1, -1, -1, 0 ], dtype=np.float32)
#         obs_high = np.array([ 1,  1,  1,  1, 1 ], dtype=np.float32)
#         self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

#         self.x = self.y = self.theta = 0.0
#         self.steps = 0
#         self.trajectory: list[tuple] = []

#     # ────────────────────────────────────────────────────────────────────────
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         # равномерное случайное начальное положение
#         self.x     = self.np_random.uniform(-self.FIELD, self.FIELD)
#         self.y     = self.np_random.uniform(-self.FIELD, self.FIELD)
#         self.theta = self.np_random.uniform(-np.pi, np.pi)
#         self.steps = 0
#         self.trajectory = [(self.x, self.y, self.theta)]
#         return self._obs(), {}

#     # ────────────────────────────────────────────────────────────────────────
#     def step(self, action):
#         v_l, v_r = np.clip(action, -1.0, 1.0) * self.V_MAX

#         # Euler-интегрирование кинематики
#         v     = (v_r + v_l) / 2.0
#         omega = (v_r - v_l) / self.L

#         self.x     += v * np.cos(self.theta) * self.DT
#         self.y     += v * np.sin(self.theta) * self.DT
#         self.theta += omega * self.DT

#         # нормализация угла в (-π, π]
#         self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
#         # мягкий барьер поля (не жёсткий clip — робот может чуть выехать)
#         self.x = np.clip(self.x, -15, 15)
#         self.y = np.clip(self.y, -15, 15)

#         self.steps += 1
#         self.trajectory.append((self.x, self.y, self.theta))

#         dist  = np.hypot(self.x, self.y)
#         angle = abs(self.theta)

#         # ── награда ──────────────────────────────────────────────────────────
#         # 1) плотная: отрицательное расстояние + штраф за угол
#         reward = -0.10 * dist - 0.03 * angle
#         # 2) штраф за выезд за поле
#         if abs(self.x) > self.FIELD or abs(self.y) > self.FIELD:
#             reward -= 1.0
#         # 3) бонус за достижение цели
#         terminated = bool(dist < self.GOAL_DIST and angle < self.GOAL_ANGLE)
#         if terminated:
#             reward += 200.0

#         truncated = self.steps >= self.MAX_STEPS

#         info = {"dist": dist, "angle": angle, "success": terminated}
#         return self._obs(), reward, terminated, truncated, info

#     def step_manual(self, left_speed, right_speed):
#         """
#         Шаг с прямым управлением скоростями колес.
        
#         Args:
#             left_speed (float): скорость левого колеса [-V_MAX, V_MAX]
#             right_speed (float): скорость правого колеса [-V_MAX, V_MAX]
        
#         Returns:
#             state, reward, done, truncated, info
#         """
#         # Преобразуем скорости колес в линейную и угловую
#         v = (left_speed + right_speed) / 2
#         omega = (right_speed - left_speed) / self.L
        
#         x, y, theta = self.state
#         x += v * np.cos(theta) * self.dt
#         y += v * np.sin(theta) * self.dt
#         theta += omega * self.dt
#         theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
#         self.state = np.array([x, y, theta], dtype=np.float32)
#         self.trajectory.append((x, y, theta))
        
#         # Награда для ручного режима (опционально)
#         dist = np.hypot(x, y)
#         angle_error = abs(theta)
#         reward = -dist - 0.5 * angle_error
        
#         done = False
#         if dist < self.GOAL_DIST and angle_error < self.GOAL_ANGLE:
#             reward += 100.0
#             done = True
        
#         return self.state.copy(), reward, done, False, {}

#     # ────────────────────────────────────────────────────────────────────────
#     def _obs(self):
#         dist = np.hypot(self.x, self.y)
#         return np.array([
#             self.x  / self.FIELD,
#             self.y  / self.FIELD,
#             np.cos(self.theta),
#             np.sin(self.theta),
#             dist    / (self.FIELD * np.sqrt(2)),
#         ], dtype=np.float32)

#     # ────────────────────────────────────────────────────────────────────────
#     def get_state(self):
#         return self.x, self.y, self.theta

# """
# Среда дифференциального привода для задачи «прийти в начало координат».

# Кинематика (непрерывная):
#     v  = (v_r + v_l) / 2           — линейная скорость
#     ω  = (v_r - v_l) / L           — угловая скорость
#     ẋ  = v · cos(θ)
#     ẏ  = v · sin(θ)
#     θ̇  = ω
# """

# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces


# class DiffDriveEnv(gym.Env):
#     """
#     Observation (5-мерный вектор):
#         [x/10, y/10, cos(θ), sin(θ), dist/14.14]

#     Action (2-мерный непрерывный):
#         [v_left, v_right] ∈ [-1, 1]

#     Goal: x→0, y→0, θ→0
#     """

#     metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

#     # ── физика ──────────────────────────────────────────────────────────────
#     L         = 1.0    # расстояние между колёсами (м)
#     V_MAX     = 3.0    # макс. скорость колеса (м/с)
#     DT        = 0.05   # шаг интегрирования (с)
#     MAX_STEPS = 600    # max шагов на эпизод

#     # ── условие достижения цели ──────────────────────────────────────────────
#     GOAL_DIST  = 0.30   # м
#     GOAL_ANGLE = 0.15   # рад (~8.6°)

#     # ── границы поля ────────────────────────────────────────────────────────
#     FIELD = 10.0

#     def __init__(self):
#         super().__init__()

#         # непрерывные действия [v_l, v_r]
#         self.action_space = spaces.Box(
#             low=-1.0, high=1.0, shape=(2,), dtype=np.float32
#         )
#         # наблюдение
#         obs_low  = np.array([-1, -1, -1, -1, 0 ], dtype=np.float32)
#         obs_high = np.array([ 1,  1,  1,  1, 1 ], dtype=np.float32)
#         self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

#         self.x = self.y = self.theta = 0.0
#         self.steps = 0
#         self.trajectory: list[tuple] = []

#     # ────────────────────────────────────────────────────────────────────────
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         # равномерное случайное начальное положение
#         self.x     = self.np_random.uniform(-self.FIELD, self.FIELD)
#         self.y     = self.np_random.uniform(-self.FIELD, self.FIELD)
#         self.theta = self.np_random.uniform(-np.pi, np.pi)
#         self.steps = 0
#         self.trajectory = [(self.x, self.y, self.theta)]
#         return self._obs(), {}

#     # ────────────────────────────────────────────────────────────────────────
#     def step(self, action):
#         v_l, v_r = np.clip(action, -1.0, 1.0) * self.V_MAX

#         # Euler-интегрирование кинематики
#         v     = (v_r + v_l) / 2.0
#         omega = (v_r - v_l) / self.L

#         self.x     += v * np.cos(self.theta) * self.DT
#         self.y     += v * np.sin(self.theta) * self.DT
#         self.theta += omega * self.DT

#         # нормализация угла в (-π, π]
#         self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
#         # мягкий барьер поля (не жёсткий clip — робот может чуть выехать)
#         self.x = np.clip(self.x, -15, 15)
#         self.y = np.clip(self.y, -15, 15)

#         self.steps += 1
#         self.trajectory.append((self.x, self.y, self.theta))

#         dist  = np.hypot(self.x, self.y)
#         angle = abs(self.theta)

#         # ── награда ──────────────────────────────────────────────────────────
#         # 1) плотная: отрицательное расстояние + штраф за угол
#         reward = -0.10 * dist - 0.03 * angle
#         # 2) штраф за выезд за поле
#         if abs(self.x) > self.FIELD or abs(self.y) > self.FIELD:
#             reward -= 1.0
#         # 3) бонус за достижение цели
#         terminated = bool(dist < self.GOAL_DIST and angle < self.GOAL_ANGLE)
#         if terminated:
#             reward += 200.0

#         truncated = self.steps >= self.MAX_STEPS

#         info = {"dist": dist, "angle": angle, "success": terminated}
#         return self._obs(), reward, terminated, truncated, info

#     def step_manual(self, left_speed, right_speed):
#         """
#         Шаг с прямым управлением скоростями колес.
        
#         Args:
#             left_speed (float): скорость левого колеса [-V_MAX, V_MAX]
#             right_speed (float): скорость правого колеса [-V_MAX, V_MAX]
        
#         Returns:
#             observation, reward, terminated, truncated, info
#         """
#         # Ограничиваем скорости
#         left_speed = np.clip(left_speed, -self.V_MAX, self.V_MAX)
#         right_speed = np.clip(right_speed, -self.V_MAX, self.V_MAX)
        
#         # Преобразуем скорости колес в линейную и угловую
#         v = (left_speed + right_speed) / 2.0
#         omega = (right_speed - left_speed) / self.L
        
#         # Обновляем состояние (используем self.x, self.y, self.theta)
#         self.x += v * np.cos(self.theta) * self.DT
#         self.y += v * np.sin(self.theta) * self.DT
#         self.theta += omega * self.DT
        
#         # Нормализация угла в (-π, π]
#         self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
#         # Мягкий барьер поля
#         self.x = np.clip(self.x, -15, 15)
#         self.y = np.clip(self.y, -15, 15)
        
#         # Добавляем в траекторию
#         self.trajectory.append((self.x, self.y, self.theta))
#         self.steps += 1
        
#         # Вычисляем расстояние и ошибку угла
#         dist = np.hypot(self.x, self.y)
#         angle_error = abs(self.theta)
        
#         # Награда для ручного режима (опционально)
#         reward = -0.10 * dist - 0.03 * angle_error
        
#         # Штраф за выезд за поле
#         if abs(self.x) > self.FIELD or abs(self.y) > self.FIELD:
#             reward -= 1.0
        
#         # Проверка достижения цели
#         terminated = bool(dist < self.GOAL_DIST and angle_error < self.GOAL_ANGLE)
#         if terminated:
#             reward += 200.0
        
#         truncated = self.steps >= self.MAX_STEPS
        
#         info = {"dist": dist, "angle": angle_error, "success": terminated}
        
#         # Возвращаем observation в том же формате, что и step()
#         return self._obs(), reward, terminated, truncated, info

#     # ────────────────────────────────────────────────────────────────────────
#     def _obs(self):
#         dist = np.hypot(self.x, self.y)
#         return np.array([
#             self.x  / self.FIELD,
#             self.y  / self.FIELD,
#             np.cos(self.theta),
#             np.sin(self.theta),
#             dist    / (self.FIELD * np.sqrt(2)),
#         ], dtype=np.float32)

#     # ────────────────────────────────────────────────────────────────────────
#     def get_state(self):
#         """
#         Возвращает текущее состояние робота в формате (x, y, theta)
#         для использования в визуализации.
#         """
#         return self.x, self.y, self.theta
    
#     # ────────────────────────────────────────────────────────────────────────
#     def get_trajectory(self):
#         """
#         Возвращает траекторию движения робота.
#         """
#         return self.trajectory

"""
Differential drive environment for the task "reach the origin".

Kinematics (continuous):
    v  = (v_r + v_l) / 2           — linear velocity
    ω  = (v_r - v_l) / L           — angular velocity
    ẋ  = v · cos(θ)
    ẏ  = v · sin(θ)
    θ̇  = ω
"""
"""
environment.py — Differential drive environment for RL training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


class DiffDriveEnv(gym.Env):
    """
    Observation (5-dimensional vector):
        [x/FIELD, y/FIELD, cos(θ), sin(θ), dist/(FIELD·√2)]

    Action (2-dimensional continuous):
        [v_left, v_right] ∈ [-1, 1]

    Goal: x→0, y→0, θ→0
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    # Physics parameters
    L = 1.0                     # wheel base (m)
    V_MAX = 3.0                 # maximum wheel speed (m/s)
    DT = 0.05                   # integration time step (s)
    MAX_STEPS = 600             # maximum steps per episode

    # Goal conditions
    GOAL_DIST = 0.30            # m
    GOAL_ANGLE = 0.15           # rad (~8.6°)

    # Field boundaries
    FIELD = 10.0

    # Reward parameters (configurable)
    REWARD_DIST_COEF = 0.10
    REWARD_ANGLE_COEF = 0.03
    REWARD_BOUNDARY_PENALTY = 1.0
    REWARD_SUCCESS_BONUS = 200.0
    REWARD_PROGRESS_COEF = 5.0
    REWARD_STOP_PENALTY = 0.5
    REWARD_ALIVE_BONUS = 0.01

    def __init__(self, render_mode=None):
        super().__init__()

        # Continuous action space [v_left, v_right]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space
        obs_low = np.array([-1, -1, -1, -1, 0], dtype=np.float32)
        obs_high = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # State variables
        self.x = self.y = self.theta = 0.0
        self.steps = 0
        self.trajectory = []

        # Tracking variables for reward shaping
        self.prev_dist = None
        self.stuck_counter = 0
        self.last_action = None

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Visualization constants
        self.window_size = 800
        self.field_size = self.FIELD
        self.scale = self.window_size / (2 * self.field_size)

    # ═══════════════════════════════════════════════════════════════════════
    # Rendering methods
    # ═══════════════════════════════════════════════════════════════════════

    def render(self, mode = None):
        """Render the environment."""
        render_mode = mode if mode is not None else self.render_mode

        if render_mode == "rgb_array":
            return self._render_rgb_array()
        elif render_mode == "human":
            return self._render_human()
        return None

        

    def _render_rgb_array(self):
        """Render as RGB array for GIF recording."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.window_size, self.window_size))

        self.screen.fill((10, 10, 20))
        self._draw_grid()
        self._draw_goal()
        self._draw_robot()

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def _render_human(self):
        """Render for human viewing."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((10, 10, 20))
        self._draw_grid()
        self._draw_goal()
        self._draw_robot()

        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

    def _draw_grid(self):
        """Draw grid and axes."""
        center_x = self.window_size // 2
        center_y = self.window_size // 2

        # Grid lines
        for i in range(-10, 11, 2):
            x = int(center_x + i * self.scale)
            y = int(center_y - i * self.scale)
            pygame.draw.line(self.screen, (30, 35, 60), (x, 0), (x, self.window_size), 1)
            pygame.draw.line(self.screen, (30, 35, 60), (0, y), (self.window_size, y), 1)

        # Axes
        pygame.draw.line(self.screen, (255, 0, 0), (0, center_y), (self.window_size, center_y), 2)
        pygame.draw.line(self.screen, (0, 255, 0), (center_x, 0), (center_x, self.window_size), 2)

        # Field boundary
        left = int(center_x - self.field_size * self.scale)
        right = int(center_x + self.field_size * self.scale)
        top = int(center_y - self.field_size * self.scale)
        bottom = int(center_y + self.field_size * self.scale)
        pygame.draw.rect(self.screen, (80, 90, 180), (left, top, right - left, bottom - top), 2)

    def _draw_goal(self):
        """Draw goal marker."""
        center_x = self.window_size // 2
        center_y = self.window_size // 2
        pygame.draw.circle(self.screen, (50, 220, 120), (center_x, center_y), 10, 2)
        pygame.draw.line(self.screen, (50, 220, 120), (center_x - 15, center_y), (center_x + 15, center_y), 2)
        pygame.draw.line(self.screen, (50, 220, 120), (center_x, center_y - 15), (center_x, center_y + 15), 2)

    def _draw_robot(self):
        """Draw robot."""
        center_x = self.window_size // 2
        center_y = self.window_size // 2
        rx = int(center_x + self.x * self.scale)
        ry = int(center_y - self.y * self.scale)

        # Body
        pygame.draw.circle(self.screen, (255, 200, 50), (rx, ry), 12)

        # Direction arrow
        arrow_x = rx + int(20 * np.cos(self.theta))
        arrow_y = ry - int(20 * np.sin(self.theta))
        pygame.draw.line(self.screen, (255, 80, 80), (rx, ry), (arrow_x, arrow_y), 3)

    def close(self):
        """Close rendering windows."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # ═══════════════════════════════════════════════════════════════════════
    # Core environment methods
    # ═══════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial position within field
        self.x = self.np_random.uniform(-self.FIELD, self.FIELD)
        self.y = self.np_random.uniform(-self.FIELD, self.FIELD)
        self.theta = self.np_random.uniform(-np.pi, np.pi)

        self.steps = 0
        self.trajectory = [(self.x, self.y, self.theta)]

        # Reset tracking variables
        self.prev_dist = np.hypot(self.x, self.y)
        self.stuck_counter = 0
        self.last_action = None

        return self._obs(), {}

    def step(self, action):
        """Execute one step in the environment."""
        # Clip and scale actions to physical speeds
        v_l, v_r = np.clip(action, -1.0, 1.0) * self.V_MAX

        # Euler integration of kinematics
        v = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / self.L

        self.x += v * np.cos(self.theta) * self.DT
        self.y += v * np.sin(self.theta) * self.DT
        self.theta += omega * self.DT

        # Normalize angle to (-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Soft boundary
        self.x = np.clip(self.x, -15, 15)
        self.y = np.clip(self.y, -15, 15)

        self.steps += 1
        self.trajectory.append((self.x, self.y, self.theta))

        # Calculate metrics
        dist = np.hypot(self.x, self.y)
        angle_error = abs(self.theta)
        angle_to_goal = np.arctan2(-self.y, -self.x) - self.theta
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))

        # Calculate reward
        reward = self._compute_reward(dist, angle_error, angle_to_goal, action, v)

        # Check termination conditions
        terminated = bool(dist < self.GOAL_DIST and angle_error < self.GOAL_ANGLE)
        truncated = self.steps >= self.MAX_STEPS

        info = {
            "dist": dist,
            "angle": angle_error,
            "angle_to_goal": angle_to_goal,
            "success": terminated,
            "steps": self.steps,
        }

        return self._obs(), reward, terminated, truncated, info

    def _compute_reward(self, dist, angle_error, angle_to_goal, action, v):
        """Compute reward with improved shaping."""
        reward = 0.0

        # 1. Progress reward
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            if progress > 0:
                reward += progress * self.REWARD_PROGRESS_COEF
            elif progress < -0.1:
                reward += progress * (self.REWARD_PROGRESS_COEF * 0.3)
        self.prev_dist = dist

        # 2. Distance penalty
        reward -= self.REWARD_DIST_COEF * dist

        # 3. Angle penalty (more important when close)
        if dist < 3.0:
            reward -= self.REWARD_ANGLE_COEF * 2.0 * angle_error
            reward -= self.REWARD_ANGLE_COEF * 3.0 * abs(angle_to_goal)
        else:
            reward -= self.REWARD_ANGLE_COEF * angle_error

        # 4. Movement encouragement
        action_magnitude = np.abs(action).mean()
        if abs(angle_to_goal) < 1.0 and v > 0:
            reward += 0.1 * min(1.0, v / self.V_MAX)

        # 5. Stopping penalty
        if action_magnitude < 0.1:
            reward -= self.REWARD_STOP_PENALTY

        # 6. Oscillation penalty
        if self.last_action is not None:
            action_change = np.abs(action - self.last_action).mean()
            if action_change > 1.0:
                reward -= 0.2
        self.last_action = action.copy()

        # 7. Stuck detection
        if hasattr(self, '_last_progress'):
            if abs(progress) < 0.01 and action_magnitude > 0.3:
                self.stuck_counter += 1
                if self.stuck_counter > 30:
                    reward -= 1.0
                    self.stuck_counter = 0
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        # 8. Boundary penalty
        if abs(self.x) > self.FIELD or abs(self.y) > self.FIELD:
            reward -= self.REWARD_BOUNDARY_PENALTY

        # 9. Alive bonus
        reward += self.REWARD_ALIVE_BONUS

        # 10. Success bonus
        if dist < self.GOAL_DIST and angle_error < self.GOAL_ANGLE:
            reward += self.REWARD_SUCCESS_BONUS

        return reward

    def step_manual(self, left_speed, right_speed):
        """Step with direct wheel speed control for manual mode."""
        # Clip speeds
        left_speed = np.clip(left_speed, -self.V_MAX, self.V_MAX)
        right_speed = np.clip(right_speed, -self.V_MAX, self.V_MAX)

        # Convert to linear and angular velocities
        v = (left_speed + right_speed) / 2.0
        omega = (right_speed - left_speed) / self.L

        # Update state
        self.x += v * np.cos(self.theta) * self.DT
        self.y += v * np.sin(self.theta) * self.DT
        self.theta += omega * self.DT

        # Normalize angle
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Soft boundary
        self.x = np.clip(self.x, -15, 15)
        self.y = np.clip(self.y, -15, 15)

        # Update trajectory
        self.trajectory.append((self.x, self.y, self.theta))
        self.steps += 1

        # Calculate metrics
        dist = np.hypot(self.x, self.y)
        angle_error = abs(self.theta)

        # Progress tracking for manual mode
        if not hasattr(self, '_manual_prev_dist'):
            self._manual_prev_dist = dist
        progress = self._manual_prev_dist - dist
        self._manual_prev_dist = dist

        # Reward for manual mode
        reward = -0.10 * dist - 0.03 * angle_error
        if progress > 0:
            reward += progress * 3.0

        # Boundary penalty
        if abs(self.x) > self.FIELD or abs(self.y) > self.FIELD:
            reward -= 1.0

        # Success condition
        terminated = bool(dist < self.GOAL_DIST and angle_error < self.GOAL_ANGLE)
        if terminated:
            reward += 200.0

        truncated = self.steps >= self.MAX_STEPS

        info = {
            "dist": dist,
            "angle": angle_error,
            "success": terminated,
            "progress": progress,
        }

        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        """Construct observation vector."""
        dist = np.hypot(self.x, self.y)
        return np.array([
            self.x / self.FIELD,
            self.y / self.FIELD,
            np.cos(self.theta),
            np.sin(self.theta),
            dist / (self.FIELD * np.sqrt(2)),
        ], dtype=np.float32)

    # ═══════════════════════════════════════════════════════════════════════
    # Helper methods
    # ═══════════════════════════════════════════════════════════════════════

    def get_state(self):
        """Return current robot state as (x, y, theta)."""
        return self.x, self.y, self.theta

    def get_trajectory(self):
        """Return robot trajectory."""
        return self.trajectory

    def set_reward_parameters(self, **kwargs):
        """Dynamically adjust reward parameters."""
        for key, value in kwargs.items():
            attr_name = f"REWARD_{key.upper()}"
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)
                print(f"Reward parameter updated: {key.upper()} = {value}")
            else:
                print(f"Warning: Unknown reward parameter: {key}")


# ══════════════════════════════════════════════════════════════════════════════
# Factory function
# ══════════════════════════════════════════════════════════════════════════════

def make_env(config=None, render_mode=None):
    """Create environment with optional configuration."""
    env = DiffDriveEnv(render_mode=render_mode)

    if config is not None and hasattr(config, 'env'):
        # Apply environment parameters from config
        env.L = getattr(config.env, 'L', env.L)
        env.V_MAX = getattr(config.env, 'V_MAX', env.V_MAX)
        env.DT = getattr(config.env, 'DT', env.DT)
        env.MAX_STEPS = getattr(config.env, 'MAX_STEPS', env.MAX_STEPS)
        env.GOAL_DIST = getattr(config.env, 'GOAL_DIST', env.GOAL_DIST)
        env.GOAL_ANGLE = getattr(config.env, 'GOAL_ANGLE', env.GOAL_ANGLE)
        env.FIELD = getattr(config.env, 'FIELD', env.FIELD)

        # Apply reward parameters
        env.REWARD_DIST_COEF = getattr(config.env, 'REWARD_DIST_COEF', env.REWARD_DIST_COEF)
        env.REWARD_ANGLE_COEF = getattr(config.env, 'REWARD_ANGLE_COEF', env.REWARD_ANGLE_COEF)
        env.REWARD_BOUNDARY_PENALTY = getattr(config.env, 'REWARD_BOUNDARY_PENALTY', env.REWARD_BOUNDARY_PENALTY)
        env.REWARD_SUCCESS_BONUS = getattr(config.env, 'REWARD_SUCCESS_BONUS', env.REWARD_SUCCESS_BONUS)
        env.REWARD_PROGRESS_COEF = getattr(config.env, 'REWARD_PROGRESS_COEF', env.REWARD_PROGRESS_COEF)
        env.REWARD_STOP_PENALTY = getattr(config.env, 'REWARD_STOP_PENALTY', env.REWARD_STOP_PENALTY)
        env.REWARD_ALIVE_BONUS = getattr(config.env, 'REWARD_ALIVE_BONUS', env.REWARD_ALIVE_BONUS)

    return env