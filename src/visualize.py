
"""
visualize.py — красивая Pygame-анимация с поддержкой ручного управления и выбора модели.

Режимы:
    python visualize.py                          — автоматический режим (PPO модель по умолчанию)
    python visualize.py --model models/ppo_diff_drive.zip
    python visualize.py --model models/ppo_diff_drive --model-type PPO
    python visualize.py --manual                  — ручное управление с клавиатуры
    python visualize.py --demo                    — демо без модели (случайные действия)
    python visualize.py --list-models             — показать доступные модели

Управление в ручном режиме:
    ↑   — движение вперед
    ↓   — движение назад
    ←   — поворот налево
    →   — поворот направо
    Q/E — уменьшить/увеличить скорость
    M   — переключить режим (ручной/авто)
    SPACE — следующий эпизод
    R   — сброс в то же начальное положение
    ESC — выход
"""

# import sys
# import os
# import math
# import argparse
# import numpy as np
# import pygame
# from pygame import gfxdraw
# from pathlib import Path

# from environment import DiffDriveEnv

# # ══════════════════════════════════════════════════════════════════════════════
# # Константы визуализации
# W, H       = 900, 900
# FIELD      = 10.0          # физические метры (±10)
# SCALE      = (W * 0.80) / (2 * FIELD)   # пикселей на метр

# # Цветовая схема (тёмная + неон)
# BG         = (10,  10,  20)
# GRID_COLOR = (30,  35,  60)
# AXIS_COLOR = (60,  70, 120)
# GOAL_COLOR = (50, 220, 120, 180)
# TRAJ_COLOR_BASE = (80, 140, 255)
# ROBOT_BODY = (255, 200,  50)
# WHEEL_COL  = (200, 160,  30)
# DIR_COL    = (255,  80,  80)
# TEXT_COLOR = (210, 210, 240)
# SUCCESS_C  = (50, 220, 120)
# FAIL_C     = (255,  80,  80)
# MANUAL_MODE_COLOR = (100, 150, 255, 80)


# # ══════════════════════════════════════════════════════════════════════════════
# def find_available_models(model_dir="models"):
#     """Находит все доступные модели в директории."""
#     models = []
#     model_path = Path(model_dir)
    
#     if model_path.exists():
#         # Ищем .zip файлы (стабильные модели)
#         for file in model_path.glob("*.zip"):
#             models.append({
#                 'path': str(file),
#                 'name': file.stem,
#                 'type': 'PPO/A2C/DQN'
#             })
        
#         # Ищем .pkl файлы (Q-learning)
#         for file in model_path.glob("*.pkl"):
#             models.append({
#                 'path': str(file),
#                 'name': file.stem,
#                 'type': 'Q-Learning'
#             })
        
#         # Ищем .pt/.pth файлы (PyTorch)
#         for ext in ['*.pt', '*.pth']:
#             for file in model_path.glob(ext):
#                 models.append({
#                     'path': str(file),
#                     'name': file.stem,
#                     'type': 'PyTorch'
#                 })
    
#     return models


# def load_model(model_path, model_type=None):
#     """
#     Загружает модель указанного типа.
    
#     Args:
#         model_path: путь к файлу модели
#         model_type: тип модели ('PPO', 'A2C', 'DQN', 'Q-Learning', 'auto')
    
#     Returns:
#         model: загруженная модель или None
#     """
#     if not os.path.exists(model_path):
#         print(f" Model not found: {model_path}")
#         return None
    
#     try:
#         # Автоопределение типа по расширению
#         if model_type is None or model_type == 'auto':
#             if model_path.endswith('.pkl'):
#                 model_type = 'qlearning'
#             elif model_path.endswith('.zip'):
#                 model_type = 'sb3'  # Stable-Baselines3
#             elif model_path.endswith(('.pt', '.pth')):
#                 model_type = 'pytorch'
#             else:
#                 model_type = 'sb3'  # по умолчанию
        
#         # Загрузка модели в зависимости от типа
#         if model_type.lower() in ['ppo', 'a2c', 'dqn', 'sb3', 'stable-baselines']:
#             from stable_baselines3 import PPO, A2C, DQN
            
#             # Пробуем разные алгоритмы
#             try:
#                 model = PPO.load(model_path)
#                 print(f"PPO model loaded: {model_path}")
#                 return model
#             except:
#                 try:
#                     model = A2C.load(model_path)
#                     print(f"A2C model loaded: {model_path}")
#                     return model
#                 except:
#                     try:
#                         model = DQN.load(model_path)
#                         print(f"DQN model loaded: {model_path}")
#                         return model
#                     except:
#                         print(f"Failed to load SB3 model: {model_path}")
#                         return None
        
#         elif model_type.lower() in ['qlearning', 'q-learning', 'q_table', 'pkl']:
#             # Загрузка Q-Learning модели
#             import pickle
#             with open(model_path, 'rb') as f:
#                 q_table = pickle.load(f)
#             print(f"Q-Learning model loaded: {model_path}")
#             return q_table
        
#         elif model_type.lower() in ['pytorch', 'pt', 'pth']:
#             print(f"PyTorch models require additional setup")
#             print(f"   Model found: {model_path}, but loading is not implemented")
#             return None
        
#         else:
#             print(f"Unknown model type: {model_type}")
#             return None
            
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None


# class ModelWrapper:
#     """Обертка для унифицированного интерфейса разных типов моделей."""
    
#     def __init__(self, model, model_type='sb3'):
#         self.model = model
#         self.model_type = model_type
    
#     def predict(self, obs, deterministic=True):
#         """Предсказание действия."""
#         if self.model_type in ['sb3', 'ppo', 'a2c', 'dqn']:
#             # Stable-Baselines3 модели
#             action, _ = self.model.predict(obs, deterministic=deterministic)
#             return action
        
#         elif self.model_type == 'qlearning':
#             # Q-Learning модель (нужна функция дискретизации)
#             # Для простоты возвращаем случайное действие
#             # В реальном проекте нужно добавить метод discretize_state
#             print("Q-Learning models require further development")
#             return np.random.uniform(-1, 1, 2)
        
#         else:
#             # Fallback: случайное действие
#             return np.random.uniform(-1, 1, 2)


# # ══════════════════════════════════════════════════════════════════════════════
# def world_to_screen(x, y):
#     """Физические координаты → пиксели (центр экрана = (0,0))."""
#     px = int(W / 2 + x * SCALE)
#     py = int(H / 2 - y * SCALE)
#     return px, py


# def draw_grid(surf, font_sm):
#     """Сетка и оси."""
#     for i in range(-10, 11, 2):
#         # вертикальные линии
#         x1, y1 = world_to_screen(i, -12)
#         x2, y2 = world_to_screen(i,  12)
#         pygame.draw.line(surf, GRID_COLOR, (x1, y1), (x2, y2), 1)
#         # горизонтальные линии
#         x1, y1 = world_to_screen(-12, i)
#         x2, y2 = world_to_screen( 12, i)
#         pygame.draw.line(surf, GRID_COLOR, (x1, y1), (x2, y2), 1)
#         # числа
#         if i != 0:
#             lbl = font_sm.render(str(i), True, (70, 80, 110))
#             sx, sy = world_to_screen(i, 0)
#             surf.blit(lbl, (sx - 8, sy + 4))

#     # оси
#     ax0, ay0 = world_to_screen(-12, 0)
#     ax1, ay1 = world_to_screen( 12, 0)
#     pygame.draw.line(surf, AXIS_COLOR, (ax0, ay0), (ax1, ay1), 2)
#     bx0, by0 = world_to_screen(0, -12)
#     bx1, by1 = world_to_screen(0,  12)
#     pygame.draw.line(surf, AXIS_COLOR, (bx0, by0), (bx1, by1), 2)

#     # граница поля ±10
#     border_pts = [
#         world_to_screen(-FIELD, -FIELD), world_to_screen( FIELD, -FIELD),
#         world_to_screen( FIELD,  FIELD), world_to_screen(-FIELD,  FIELD),
#     ]
#     pygame.draw.polygon(surf, (80, 90, 180), border_pts, 2)


# def draw_goal(surf):
#     """Цель — пульсирующий круг в начале координат."""
#     gx, gy = world_to_screen(0, 0)
#     r_goal = int(DiffDriveEnv.GOAL_DIST * SCALE)

#     overlay = pygame.Surface((W, H), pygame.SRCALPHA)
#     pygame.draw.circle(overlay, GOAL_COLOR, (gx, gy), max(r_goal, 8))
#     surf.blit(overlay, (0, 0))

#     pygame.draw.line(surf, SUCCESS_C, (gx - 15, gy), (gx + 15, gy), 2)
#     pygame.draw.line(surf, SUCCESS_C, (gx, gy - 15), (gx, gy + 15), 2)


# def draw_trajectory(surf, traj, color_base):
#     """Траектория с градиентом яркости."""
#     if len(traj) < 2:
#         return
#     n = len(traj)
#     for i in range(1, n):
#         alpha = i / n
#         c = tuple(int(c * (0.2 + 0.8 * alpha)) for c in color_base)
#         p1 = world_to_screen(*traj[i - 1][:2])
#         p2 = world_to_screen(*traj[i][:2])
#         pygame.draw.line(surf, c, p1, p2, 2)

#     sx, sy = world_to_screen(*traj[0][:2])
#     pygame.draw.circle(surf, (255, 160, 50), (sx, sy), 5)


# def draw_robot(surf, x, y, theta, v_l=0, v_r=0):
#     """Отрисовка робота."""
#     cx, cy = world_to_screen(x, y)
#     R = 14
#     L_px = int(DiffDriveEnv.L * SCALE / 2)

#     pygame.draw.circle(surf, ROBOT_BODY, (cx, cy), R)
#     pygame.draw.circle(surf, (40, 40, 40), (cx, cy), R, 2)

#     ex = cx + int((R + 8) * math.cos(theta))
#     ey = cy - int((R + 8) * math.sin(theta))
#     pygame.draw.line(surf, DIR_COL, (cx, cy), (ex, ey), 3)
#     for da in [0.5, -0.5]:
#         tip_x = ex + int(7 * math.cos(theta + math.pi + da))
#         tip_y = ey - int(7 * math.sin(theta + math.pi + da))
#         pygame.draw.line(surf, DIR_COL, (ex, ey), (tip_x, tip_y), 2)

#     perp = theta + math.pi / 2
#     for side, v in [(1, v_l), (-1, v_r)]:
#         wx = cx + side * int(L_px * math.cos(perp))
#         wy = cy - side * int(L_px * math.sin(perp))
#         speed_norm = abs(v) / DiffDriveEnv.V_MAX
#         wc = (int(200 * speed_norm), int(200 * speed_norm), int(255 * speed_norm))
#         wc = (max(80, wc[0]), max(80, wc[1]), max(80, wc[2]))
#         wheel_rect = pygame.Rect(0, 0, int(L_px * 0.7), 8)
#         wheel_surf = pygame.Surface(wheel_rect.size, pygame.SRCALPHA)
#         wheel_surf.fill((*wc, 220))
#         angle_deg = math.degrees(theta)
#         rotated = pygame.transform.rotate(wheel_surf, angle_deg)
#         rr = rotated.get_rect(center=(wx, wy))
#         surf.blit(rotated, rr.topleft)


# def draw_hud(surf, font, font_sm, x, y, theta, dist, step, ep, success,
#              done, elapsed_ms, action, manual_mode=False, max_speed=1.0,
#              model_name=None):
#     """Информационная панель (HUD) справа."""
#     panel_x = int(W * 0.83)
#     panel_w = W - panel_x - 5

#     pygame.draw.rect(surf, (15, 17, 35), (panel_x, 0, panel_w, H))
#     pygame.draw.line(surf, (50, 60, 120), (panel_x, 0), (panel_x, H), 2)
    
#     if manual_mode:
#         overlay = pygame.Surface((panel_w, H), pygame.SRCALPHA)
#         overlay.fill(MANUAL_MODE_COLOR)
#         surf.blit(overlay, (panel_x, 0))

#     def row(label, value, y_off, color=TEXT_COLOR):
#         lbl = font_sm.render(label, True, (130, 140, 180))
#         val = font.render(value, True, color)
#         surf.blit(lbl, (panel_x + 8, y_off))
#         surf.blit(val, (panel_x + 8, y_off + 16))

#     y0 = 20
#     mode_text = "MANUAL CONTROL" if manual_mode else f"RL AGENT: {model_name or 'PPO'}"
#     title_color = MANUAL_MODE_COLOR[:3] if manual_mode else (180, 200, 255)
#     title = font.render(mode_text, True, title_color)
#     surf.blit(title, (panel_x + 8, y0))

#     y0 += 40
#     row("Position X", f"{x:+.2f} м", y0)
#     y0 += 42
#     row("Position Y", f"{y:+.2f} м", y0)
#     y0 += 42
#     row("Angle θ", f"{math.degrees(theta):+.1f}°", y0)
#     y0 += 42
#     row("Distance", f"{dist:.2f} м", y0, color=(
#         SUCCESS_C if dist < 1.0 else TEXT_COLOR))
#     y0 += 42
#     row("Step", f"{step}", y0)
#     y0 += 42
#     row("Episode", f"{ep}", y0)
    
#     if manual_mode:
#         y0 += 42
#         row("Max Speed", f"{max_speed:.2f}", y0)

#     if action is not None:
#         y0 += 50
#         lbl = font_sm.render("Wheel Speeds", True, (130, 140, 180))
#         surf.blit(lbl, (panel_x + 8, y0))
#         y0 += 16
#         _draw_wheel_bar(surf, panel_x + 8, y0, "LEFT", action[0])
#         _draw_wheel_bar(surf, panel_x + 8, y0 + 22, "RIGHT", action[1])
        
#         if manual_mode:
#             y0 += 50
#             controls = font_sm.render("↑ ↓ ← →  |  Q/E speed", True, (130, 140, 180))
#             surf.blit(controls, (panel_x + 8, y0))
#             y0 += 20
#             switch = font_sm.render("M - switch mode", True, (130, 140, 180))
#             surf.blit(switch, (panel_x + 8, y0))

#     y0 += 70
#     if done:
#         msg = "SUCCESS ✓" if success else "TIMEOUT ✗"
#         col = SUCCESS_C if success else FAIL_C
#         lbl = font.render(msg, True, col)
#         surf.blit(lbl, (panel_x + 8, y0))
#         y0 += 28
#         hint = font_sm.render("SPACE — next episode", True, (130, 140, 180))
#         surf.blit(hint, (panel_x + 8, y0))


# def _draw_wheel_bar(surf, x, y, label, value):
#     """Маленький бар скорости колеса."""
#     font_tiny = pygame.font.SysFont("monospace", 12)
#     lbl = font_tiny.render(label, True, (180, 180, 220))
#     surf.blit(lbl, (x, y))
#     bar_x = x + 40
#     bar_w = 70
#     bar_h = 12
#     pygame.draw.rect(surf, (40, 40, 70), (bar_x, y, bar_w, bar_h), 0, 4)
#     fill_w = int(bar_w * (value + 1) / 2)
#     fill_w = max(0, min(bar_w, fill_w))
#     col = (80, 200, 255) if value >= 0 else (255, 100, 80)
#     pygame.draw.rect(surf, col, (bar_x, y, fill_w, bar_h), 0, 4)
#     zero_x = bar_x + int(bar_w * 0.5)
#     pygame.draw.line(surf, (255, 255, 255), (zero_x, y - 2), (zero_x, y + bar_h + 2), 1)


# def get_manual_action(keys, max_speed=1.0):
#     """Получить действие из клавиатуры."""
#     left = 0.0
#     right = 0.0
    
#     if keys[pygame.K_UP]:
#         left += max_speed
#         right += max_speed
#     if keys[pygame.K_DOWN]:
#         left -= max_speed
#         right -= max_speed
#     if keys[pygame.K_LEFT]:
#         left -= max_speed * 0.8
#         right += max_speed * 0.8
#     if keys[pygame.K_RIGHT]:
#         left += max_speed * 0.8
#         right -= max_speed * 0.8
    
#     left = np.clip(left, -max_speed, max_speed)
#     right = np.clip(right, -max_speed, max_speed)
    
#     if not any([keys[pygame.K_UP], keys[pygame.K_DOWN], 
#                 keys[pygame.K_LEFT], keys[pygame.K_RIGHT]]):
#         left = right = 0.0
    
#     return np.array([left, right])


# # ══════════════════════════════════════════════════════════════════════════════
# def run(manual_mode=False, demo_mode=False, model_path=None, model_type=None):
#     pygame.init()
#     screen = pygame.display.set_mode((W, H))
#     pygame.display.set_caption("Differential Drive / RL Agent / Manual Control")
#     clock = pygame.time.Clock()

#     font    = pygame.font.SysFont("monospace", 15, bold=True)
#     font_sm = pygame.font.SysFont("monospace", 12)

#     # ── загрузка модели ───────────────────────────────────────────────────────
#     model = None
#     model_name = None
    
#     if not demo_mode and not manual_mode:
#         # Если указан путь к модели
#         if model_path:
#             loaded_model = load_model(model_path, model_type)
#             if loaded_model is not None:
#                 model = ModelWrapper(loaded_model, 'sb3')
#                 model_name = os.path.basename(model_path).replace('.zip', '')
#                 print(f" Model loaded: {model_path}")
#             else:
#                 print(f"Failed to load model, switching to demo mode")
#                 demo_mode = True
#         else:
#             # Пытаемся загрузить модель по умолчанию
#             default_path = "models/ppo_diff_drive.zip"
#             if os.path.exists(default_path):
#                 loaded_model = load_model(default_path)
#                 if loaded_model is not None:
#                     model = ModelWrapper(loaded_model, 'sb3')
#                     model_name = "ppo_diff_drive"
#                     print(f"Default model loaded: {default_path}")
#                 else:
#                     print(f"Default model not loaded, switching to demo mode")
#                     demo_mode = True
#             else:
#                 print(f"Default model not found: {default_path}")
#                 print(f"   Use --model to specify the path to the model")
#                 print(f"   Or use --list-models to view available models")
#                 demo_mode = True
    
#     if manual_mode:
#         print(" Manual mode")
#         print("   ↑ ↓   — movement forward/backward")
#         print("   ← →   — rotation")
#         print("   Q/E   — decrease/increase speed")
#         print("   M     — switch to automatic mode")
#         print("   SPACE — new episode")
#         print("   R     — reset to same position")

#     env = DiffDriveEnv()

#     episode = 0
#     obs, _ = env.reset()
#     done = False
#     success = False
#     action = np.zeros(2)
#     step = 0
#     manual_active = manual_mode
#     max_speed = 1.0
    
#     traj_colors = [
#         (80, 140, 255), (255, 140, 80), (80, 255, 140),
#         (255, 80, 200), (200, 255, 80),
#     ]

#     # ── главный цикл ──────────────────────────────────────────────────────────
#     while True:
#         keys = pygame.key.get_pressed()
        
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit(); sys.exit()
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     pygame.quit(); sys.exit()
                    
#                 if event.key == pygame.K_m and not demo_mode:
#                     manual_active = not manual_active
#                     mode_str = "MANUAL" if manual_active else "AUTOMATIC"
#                     print(f"🔄 Switched to {mode_str} mode")
                    
#                 if manual_active:
#                     if event.key == pygame.K_q:
#                         max_speed = max(0.2, max_speed - 0.1)
#                         print(f" Speed decreased: {max_speed:.1f}")
#                     if event.key == pygame.K_e:
#                         max_speed = min(1.5, max_speed + 0.1)
#                         print(f" Speed increased: {max_speed:.1f}")
                
#                 if event.key == pygame.K_SPACE and done:
#                     episode += 1
#                     obs, _ = env.reset()
#                     done = False
#                     success = False
#                     step = 0
#                     action = np.zeros(2)
                    
#                 if event.key == pygame.K_r:
#                     obs, _ = env.reset(seed=episode * 17)
#                     done = False
#                     success = False
#                     step = 0
#                     action = np.zeros(2)

#         # ── шаг агента ────────────────────────────────────────────────────────
#         if not done:
#             if manual_active:
#                 action = get_manual_action(keys, max_speed)
#                 obs, reward, terminated, truncated, info = env.step_manual(action[0], action[1])
#                 done = terminated or truncated
#                 success = terminated
#             else:
#                 if model is not None:
#                     action = model.predict(obs, deterministic=True)
#                 else:
#                     action = env.action_space.sample()
                
#                 obs, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated
#                 success = terminated
            
#             step += 1

#         # ── отрисовка ─────────────────────────────────────────────────────────
#         screen.fill(BG)
#         draw_grid(screen, font_sm)
#         draw_goal(screen)

#         traj = env.trajectory
#         col  = traj_colors[episode % len(traj_colors)]
#         draw_trajectory(screen, traj, col)

#         x, y, theta = env.get_state()
#         dist = np.hypot(x, y)

#         if manual_active:
#             v_l, v_r = action[0], action[1]
#         else:
#             v_l = action[0] * DiffDriveEnv.V_MAX
#             v_r = action[1] * DiffDriveEnv.V_MAX
        
#         draw_robot(screen, x, y, theta, v_l, v_r)

#         draw_hud(screen, font, font_sm,
#                  x, y, theta, dist, step, episode + 1,
#                  success, done, clock.get_time(), 
#                  action, manual_active, max_speed,
#                  model_name)

#         # Нижняя строка подсказок
#         if demo_mode:
#             mode_txt = "DEMO (random actions)"
#         elif manual_active:
#             mode_txt = "MANUAL CONTROL | ↑↓←→ | Q/E speed | M=auto | ESC"
#         else:
#             mode_txt = f"AUTO ({model_name or 'PPO'}) | M=manual | SPACE=next | ESC"
            
#         mtxt = font_sm.render(mode_txt, True, (90, 100, 160))
#         screen.blit(mtxt, (10, H - 22))

#         pygame.display.flip()
#         clock.tick(60)


# # ══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Visualization of a differential drive robot with support for manual control and model selection.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples of use:
#   python visualize.py                                    # automatic with default model
#   python visualize.py --model models/ppo_diff_drive.zip  # specify model
#   python visualize.py --manual                           # manual control
#   python visualize.py --demo                             # demo without model
#   python visualize.py --list-models                      # show available models
#         """
#     )
    
#     parser.add_argument("--manual", action="store_true", 
#                        help="ручной режим управления с клавиатуры")
#     parser.add_argument("--demo", action="store_true", 
#                        help="демо без модели (случайные действия)")
#     parser.add_argument("--model", "-m", type=str,
#                        help="путь к файлу модели (например, models/ppo_diff_drive.zip)")
#     parser.add_argument("--model-type", "-t", type=str, default="auto",
#                        choices=['auto', 'ppo', 'a2c', 'dqn', 'qlearning', 'pytorch'],
#                        help="тип модели (по умолчанию: auto)")
#     parser.add_argument("--list-models", "-l", action="store_true",
#                        help="показать доступные модели в папке models/")
    
#     args = parser.parse_args()
    
#     if args.demo and args.manual:
#         print("Cannot use --demo and --manual simultaneously")
#         sys.exit(1)
    
#     # Показать доступные модели
#     if args.list_models:
#         print("\n Available models:")
#         models = find_available_models()
#         if models:
#             for i, m in enumerate(models, 1):
#                 print(f"  {i}. {m['name']} ({m['type']})")
#                 print(f"     Path: {m['path']}")
#         else:
#             print("  No models available in the models/ folder")
#         print("\nTip: use --model <path> to load a model")
#         sys.exit(0)
    
#     run(manual_mode=args.manual, demo_mode=args.demo, 
#         model_path=args.model, model_type=args.model_type)

"""
visualize.py — Visualization of differential drive robot with RL agent and manual control.

Modes:
    python visualize.py                          — automatic mode (default model)
    python visualize.py --model models/ppo_diff_drive.zip
    python visualize.py --model models/ppo_diff_drive --model-type PPO
    python visualize.py --manual                  — manual control
    python visualize.py --demo                    — demo without model
    python visualize.py --list-models             — show available models

Manual control:
    ↑   — forward
    ↓   — backward
    ←   — turn left
    →   — turn right
    Q/E — decrease/increase speed
    M   — switch mode (manual/auto)
    SPACE — next episode
    R   — reset to same position
    ESC — exit
"""

import sys
import os
import math
import argparse
import numpy as np
import pygame
from pathlib import Path

from environment import DiffDriveEnv

# ══════════════════════════════════════════════════════════════════════════════
# Visualization constants
W, H       = 1100, 900      # Increased width to accommodate wider HUD
FIELD      = 10.0           # physical meters (±10)
PANEL_WIDTH = 280           # Wider panel for longer model names
VIEW_WIDTH = W - PANEL_WIDTH - 20  # Main view width
SCALE      = (VIEW_WIDTH * 0.85) / (2 * FIELD)  # pixels per meter

# Center the field in the available view
VIEW_OFFSET_X = 20

# Color scheme (dark + neon)
BG         = (10,  10,  20)
GRID_COLOR = (30,  35,  60)
AXIS_COLOR = (60,  70, 120)
GOAL_COLOR = (50, 220, 120, 180)
TRAJ_COLOR_BASE = (80, 140, 255)
ROBOT_BODY = (255, 200,  50)
DIR_COL    = (255,  80,  80)
TEXT_COLOR = (210, 210, 240)
SUCCESS_C  = (50, 220, 120)
FAIL_C     = (255,  80,  80)
MANUAL_MODE_COLOR = (100, 150, 255, 80)


# ══════════════════════════════════════════════════════════════════════════════
def find_available_models(model_dir="models"):
    """Find all available models in the directory."""
    models = []
    model_path = Path(model_dir)
    
    if model_path.exists():
        # Find .zip files (Stable-Baselines3 models)
        for file in model_path.glob("*.zip"):
            models.append({
                'path': str(file),
                'name': file.stem,
                'type': 'PPO/A2C/DQN'
            })
        
        # Find .pkl files (Q-learning)
        for file in model_path.glob("*.pkl"):
            models.append({
                'path': str(file),
                'name': file.stem,
                'type': 'Q-Learning'
            })
        
        # Find .pt/.pth files (PyTorch)
        for ext in ['*.pt', '*.pth']:
            for file in model_path.glob(ext):
                models.append({
                    'path': str(file),
                    'name': file.stem,
                    'type': 'PyTorch'
                })
    
    return models


def load_model(model_path, model_type=None):
    """
    Load model of specified type.
    
    Args:
        model_path: path to model file
        model_type: model type ('PPO', 'A2C', 'DQN', 'Q-Learning', 'auto')
    
    Returns:
        model: loaded model or None
    """
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    try:
        # Auto-detect type by extension
        if model_type is None or model_type == 'auto':
            if model_path.endswith('.pkl'):
                model_type = 'qlearning'
            elif model_path.endswith('.zip'):
                model_type = 'sb3'
            elif model_path.endswith(('.pt', '.pth')):
                model_type = 'pytorch'
            else:
                model_type = 'sb3'
        
        # Load model based on type
        if model_type.lower() in ['ppo', 'a2c', 'dqn', 'sb3', 'stable-baselines']:
            from stable_baselines3 import PPO, A2C, DQN
            
            try:
                model = PPO.load(model_path)
                print(f"PPO model loaded: {model_path}")
                return model
            except:
                try:
                    model = A2C.load(model_path)
                    print(f"A2C model loaded: {model_path}")
                    return model
                except:
                    try:
                        model = DQN.load(model_path)
                        print(f"DQN model loaded: {model_path}")
                        return model
                    except:
                        print(f"Failed to load SB3 model: {model_path}")
                        return None
        
        elif model_type.lower() in ['qlearning', 'q-learning', 'q_table', 'pkl']:
            import pickle
            with open(model_path, 'rb') as f:
                q_table = pickle.load(f)
            print(f"Q-Learning model loaded: {model_path}")
            return q_table
        
        elif model_type.lower() in ['pytorch', 'pt', 'pth']:
            print(f"PyTorch models require additional setup")
            print(f"Model found: {model_path}, but loading is not implemented")
            return None
        
        else:
            print(f"Unknown model type: {model_type}")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


class ModelWrapper:
    """Wrapper for unified interface of different model types."""
    
    def __init__(self, model, model_type='sb3'):
        self.model = model
        self.model_type = model_type
    
    def predict(self, obs, deterministic=True):
        """Predict action."""
        if self.model_type in ['sb3', 'ppo', 'a2c', 'dqn']:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return action
        
        elif self.model_type == 'qlearning':
            print("Q-Learning models require further development")
            return np.random.uniform(-1, 1, 2)
        
        else:
            return np.random.uniform(-1, 1, 2)


# ══════════════════════════════════════════════════════════════════════════════
def world_to_screen(x, y):
    """Physical coordinates → pixels (center of view = (0,0))."""
    center_x = VIEW_OFFSET_X + VIEW_WIDTH // 2
    center_y = H // 2
    px = int(center_x + x * SCALE)
    py = int(center_y - y * SCALE)
    return px, py


def draw_grid(surf, font_sm):
    """Draw grid and axes."""
    for i in range(-10, 11, 2):
        # Vertical lines
        x1, y1 = world_to_screen(i, -12)
        x2, y2 = world_to_screen(i, 12)
        pygame.draw.line(surf, GRID_COLOR, (x1, y1), (x2, y2), 1)
        # Horizontal lines
        x1, y1 = world_to_screen(-12, i)
        x2, y2 = world_to_screen(12, i)
        pygame.draw.line(surf, GRID_COLOR, (x1, y1), (x2, y2), 1)
        # Numbers
        if i != 0:
            lbl = font_sm.render(str(i), True, (70, 80, 110))
            sx, sy = world_to_screen(i, 0)
            surf.blit(lbl, (sx - 8, sy + 4))

    # Axes
    ax0, ay0 = world_to_screen(-12, 0)
    ax1, ay1 = world_to_screen(12, 0)
    pygame.draw.line(surf, AXIS_COLOR, (ax0, ay0), (ax1, ay1), 2)
    bx0, by0 = world_to_screen(0, -12)
    bx1, by1 = world_to_screen(0, 12)
    pygame.draw.line(surf, AXIS_COLOR, (bx0, by0), (bx1, by1), 2)

    # Field boundary ±10
    border_pts = [
        world_to_screen(-FIELD, -FIELD), world_to_screen(FIELD, -FIELD),
        world_to_screen(FIELD, FIELD), world_to_screen(-FIELD, FIELD),
    ]
    pygame.draw.polygon(surf, (80, 90, 180), border_pts, 2)


def draw_goal(surf):
    """Draw goal as a pulsing circle at origin."""
    gx, gy = world_to_screen(0, 0)
    r_goal = int(DiffDriveEnv.GOAL_DIST * SCALE)

    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.circle(overlay, GOAL_COLOR, (gx, gy), max(r_goal, 8))
    surf.blit(overlay, (0, 0))

    pygame.draw.line(surf, SUCCESS_C, (gx - 15, gy), (gx + 15, gy), 2)
    pygame.draw.line(surf, SUCCESS_C, (gx, gy - 15), (gx, gy + 15), 2)


def draw_trajectory(surf, traj, color_base):
    """Draw trajectory with gradient brightness."""
    if len(traj) < 2:
        return
    n = len(traj)
    for i in range(1, n):
        alpha = i / n
        c = tuple(int(c * (0.2 + 0.8 * alpha)) for c in color_base)
        p1 = world_to_screen(*traj[i - 1][:2])
        p2 = world_to_screen(*traj[i][:2])
        pygame.draw.line(surf, c, p1, p2, 2)

    sx, sy = world_to_screen(*traj[0][:2])
    pygame.draw.circle(surf, (255, 160, 50), (sx, sy), 5)


def draw_robot(surf, x, y, theta, v_l=0, v_r=0):
    """Draw robot: body, wheels, and direction arrow."""
    cx, cy = world_to_screen(x, y)
    R = 14
    L_px = int(DiffDriveEnv.L * SCALE / 2)

    pygame.draw.circle(surf, ROBOT_BODY, (cx, cy), R)
    pygame.draw.circle(surf, (40, 40, 40), (cx, cy), R, 2)

    # Direction arrow
    ex = cx + int((R + 8) * math.cos(theta))
    ey = cy - int((R + 8) * math.sin(theta))
    pygame.draw.line(surf, DIR_COL, (cx, cy), (ex, ey), 3)
    for da in [0.5, -0.5]:
        tip_x = ex + int(7 * math.cos(theta + math.pi + da))
        tip_y = ey - int(7 * math.sin(theta + math.pi + da))
        pygame.draw.line(surf, DIR_COL, (ex, ey), (tip_x, tip_y), 2)

    # Wheels
    perp = theta + math.pi / 2
    for side, v in [(1, v_l), (-1, v_r)]:
        wx = cx + side * int(L_px * math.cos(perp))
        wy = cy - side * int(L_px * math.sin(perp))
        speed_norm = abs(v) / DiffDriveEnv.V_MAX
        wc = (int(200 * speed_norm), int(200 * speed_norm), int(255 * speed_norm))
        wc = (max(80, wc[0]), max(80, wc[1]), max(80, wc[2]))
        wheel_rect = pygame.Rect(0, 0, int(L_px * 0.7), 8)
        wheel_surf = pygame.Surface(wheel_rect.size, pygame.SRCALPHA)
        wheel_surf.fill((*wc, 220))
        angle_deg = math.degrees(theta)
        rotated = pygame.transform.rotate(wheel_surf, angle_deg)
        rr = rotated.get_rect(center=(wx, wy))
        surf.blit(rotated, rr.topleft)


def draw_hud(surf, font, font_sm, x, y, theta, dist, step, ep, success,
             done, elapsed_ms, action, manual_mode=False, max_speed=1.0,
             model_name=None):
    """Information panel (HUD) on the right side."""
    panel_x = VIEW_OFFSET_X + VIEW_WIDTH + 10
    panel_w = PANEL_WIDTH

    pygame.draw.rect(surf, (15, 17, 35), (panel_x, 0, panel_w, H))
    pygame.draw.line(surf, (50, 60, 120), (panel_x, 0), (panel_x, H), 2)
    
    if manual_mode:
        overlay = pygame.Surface((panel_w, H), pygame.SRCALPHA)
        overlay.fill(MANUAL_MODE_COLOR)
        surf.blit(overlay, (panel_x, 0))

    def row(label, value, y_off, color=TEXT_COLOR):
        lbl = font_sm.render(label, True, (130, 140, 180))
        val = font.render(value, True, color)
        surf.blit(lbl, (panel_x + 8, y_off))
        surf.blit(val, (panel_x + 8, y_off + 16))

    y0 = 20
    mode_text = "MANUAL CONTROL" if manual_mode else f"RL AGENT: {model_name or 'PPO'}"
    # Wrap long model names
    if len(mode_text) > 25 and not manual_mode:
        mode_text = f"RL AGENT: {model_name[:22]}..." if model_name else "RL AGENT: PPO"
    
    title_color = MANUAL_MODE_COLOR[:3] if manual_mode else (180, 200, 255)
    title = font.render(mode_text, True, title_color)
    surf.blit(title, (panel_x + 8, y0))

    y0 += 45
    row("Position X", f"{x:+.2f} m", y0)
    y0 += 42
    row("Position Y", f"{y:+.2f} m", y0)
    y0 += 42
    row("Angle", f"{math.degrees(theta):+.1f} deg", y0)
    y0 += 42
    row("Distance", f"{dist:.2f} m", y0, color=(
        SUCCESS_C if dist < 1.0 else TEXT_COLOR))
    y0 += 42
    row("Step", f"{step}", y0)
    y0 += 42
    row("Episode", f"{ep}", y0)
    
    if manual_mode:
        y0 += 42
        row("Max Speed", f"{max_speed:.2f}", y0)

    if action is not None:
        y0 += 50
        lbl = font_sm.render("Wheel Speeds", True, (130, 140, 180))
        surf.blit(lbl, (panel_x + 8, y0))
        y0 += 20
        _draw_wheel_bar(surf, panel_x + 8, y0, "LEFT", action[0])
        _draw_wheel_bar(surf, panel_x + 8, y0 + 25, "RIGHT", action[1])
        
        if manual_mode:
            y0 += 65
            controls = font_sm.render("↑ ↓ ← →  |  Q/E speed", True, (130, 140, 180))
            surf.blit(controls, (panel_x + 8, y0))
            y0 += 20
            switch = font_sm.render("M - switch mode", True, (130, 140, 180))
            surf.blit(switch, (panel_x + 8, y0))

    y0 += 80
    if done:
        msg = "SUCCESS" if success else "TIMEOUT"
        col = SUCCESS_C if success else FAIL_C
        lbl = font.render(msg, True, col)
        surf.blit(lbl, (panel_x + 8, y0))
        y0 += 30
        hint = font_sm.render("SPACE — next episode", True, (130, 140, 180))
        surf.blit(hint, (panel_x + 8, y0))


def _draw_wheel_bar(surf, x, y, label, value):
    """Draw small wheel speed bar."""
    font_tiny = pygame.font.SysFont("monospace", 12)
    lbl = font_tiny.render(label, True, (180, 180, 220))
    surf.blit(lbl, (x, y))
    bar_x = x + 45
    bar_w = 90
    bar_h = 12
    pygame.draw.rect(surf, (40, 40, 70), (bar_x, y, bar_w, bar_h), 0, 4)
    fill_w = int(bar_w * (value + 1) / 2)
    fill_w = max(0, min(bar_w, fill_w))
    col = (80, 200, 255) if value >= 0 else (255, 100, 80)
    pygame.draw.rect(surf, col, (bar_x, y, fill_w, bar_h), 0, 4)
    zero_x = bar_x + int(bar_w * 0.5)
    pygame.draw.line(surf, (255, 255, 255), (zero_x, y - 2), (zero_x, y + bar_h + 2), 1)
    # Show numeric value
    val_text = font_tiny.render(f"{value:.2f}", True, (180, 180, 220))
    surf.blit(val_text, (bar_x + bar_w + 5, y))


def get_manual_action(keys, max_speed=1.0):
    """Get action from keyboard input."""
    left = 0.0
    right = 0.0
    
    if keys[pygame.K_UP]:
        left += max_speed
        right += max_speed
    if keys[pygame.K_DOWN]:
        left -= max_speed
        right -= max_speed
    if keys[pygame.K_LEFT]:
        left -= max_speed * 0.8
        right += max_speed * 0.8
    if keys[pygame.K_RIGHT]:
        left += max_speed * 0.8
        right -= max_speed * 0.8
    
    left = np.clip(left, -max_speed, max_speed)
    right = np.clip(right, -max_speed, max_speed)
    
    if not any([keys[pygame.K_UP], keys[pygame.K_DOWN], 
                keys[pygame.K_LEFT], keys[pygame.K_RIGHT]]):
        left = right = 0.0
    
    return np.array([left, right])


# ══════════════════════════════════════════════════════════════════════════════
def run(manual_mode=False, demo_mode=False, model_path=None, model_type=None):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Differential Drive Robot - RL Agent / Manual Control")
    clock = pygame.time.Clock()

    font    = pygame.font.SysFont("monospace", 15, bold=True)
    font_sm = pygame.font.SysFont("monospace", 12)

    # Load model
    model = None
    model_name = None
    
    if not demo_mode and not manual_mode:
        if model_path:
            loaded_model = load_model(model_path, model_type)
            if loaded_model is not None:
                model = ModelWrapper(loaded_model, 'sb3')
                model_name = os.path.basename(model_path).replace('.zip', '')
                print(f"Model loaded: {model_path}")
            else:
                print(f"Failed to load model, switching to demo mode")
                demo_mode = True
        else:
            default_path = "models/ppo_diff_drive.zip"
            if os.path.exists(default_path):
                loaded_model = load_model(default_path)
                if loaded_model is not None:
                    model = ModelWrapper(loaded_model, 'sb3')
                    model_name = "ppo_diff_drive"
                    print(f"Default model loaded: {default_path}")
                else:
                    print(f"Default model not loaded, switching to demo mode")
                    demo_mode = True
            else:
                print(f"Default model not found: {default_path}")
                print(f"Use --model to specify the path to the model")
                print(f"Or use --list-models to view available models")
                demo_mode = True
    
    if manual_mode:
        print("Manual mode")
        print("  ↑ ↓   — forward/backward")
        print("  ← →   — rotation")
        print("  Q/E   — decrease/increase speed")
        print("  M     — switch to automatic mode")
        print("  SPACE — new episode")
        print("  R     — reset to same position")

    env = DiffDriveEnv()

    episode = 0
    obs, _ = env.reset()
    done = False
    success = False
    action = np.zeros(2)
    step = 0
    manual_active = manual_mode
    max_speed = 1.0
    
    traj_colors = [
        (80, 140, 255), (255, 140, 80), (80, 255, 140),
        (255, 80, 200), (200, 255, 80),
    ]

    # Main loop
    while True:
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                    
                if event.key == pygame.K_m and not demo_mode:
                    manual_active = not manual_active
                    mode_str = "MANUAL" if manual_active else "AUTOMATIC"
                    print(f"Switched to {mode_str} mode")
                    
                if manual_active:
                    if event.key == pygame.K_q:
                        max_speed = max(0.2, max_speed - 0.1)
                        print(f"Speed decreased: {max_speed:.1f}")
                    if event.key == pygame.K_e:
                        max_speed = min(1.5, max_speed + 0.1)
                        print(f"Speed increased: {max_speed:.1f}")
                
                if event.key == pygame.K_SPACE and done:
                    episode += 1
                    obs, _ = env.reset()
                    done = False
                    success = False
                    step = 0
                    action = np.zeros(2)
                    
                if event.key == pygame.K_r:
                    obs, _ = env.reset(seed=episode * 17)
                    done = False
                    success = False
                    step = 0
                    action = np.zeros(2)

        # Agent step
        if not done:
            if manual_active:
                action = get_manual_action(keys, max_speed)
                obs, reward, terminated, truncated, info = env.step_manual(action[0], action[1])
                done = terminated or truncated
                success = terminated
            else:
                if model is not None:
                    action = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                success = terminated
            
            step += 1

        # Rendering
        screen.fill(BG)
        
        # Draw field boundary
        field_rect = pygame.Rect(VIEW_OFFSET_X, 0, VIEW_WIDTH, H)
        pygame.draw.rect(screen, (20, 20, 35), field_rect)
        
        draw_grid(screen, font_sm)
        draw_goal(screen)

        traj = env.trajectory
        col = traj_colors[episode % len(traj_colors)]
        draw_trajectory(screen, traj, col)

        x, y, theta = env.get_state()
        dist = np.hypot(x, y)

        if manual_active:
            v_l, v_r = action[0], action[1]
        else:
            v_l = action[0] * DiffDriveEnv.V_MAX
            v_r = action[1] * DiffDriveEnv.V_MAX
        
        draw_robot(screen, x, y, theta, v_l, v_r)

        draw_hud(screen, font, font_sm,
                 x, y, theta, dist, step, episode + 1,
                 success, done, clock.get_time(), 
                 action, manual_active, max_speed,
                 model_name)

        # Bottom status line
        if demo_mode:
            mode_txt = "DEMO (random actions)"
        elif manual_active:
            mode_txt = "MANUAL CONTROL | ↑↓←→ | Q/E speed | M=auto | ESC"
        else:
            mode_txt = f"AUTO ({model_name or 'PPO'}) | M=manual | SPACE=next | ESC"
            
        mtxt = font_sm.render(mode_txt, True, (90, 100, 160))
        screen.blit(mtxt, (10, H - 22))

        pygame.display.flip()
        clock.tick(60)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualization of a differential drive robot with support for manual control and model selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py                                    # automatic with default model
  python visualize.py --model models/ppo_diff_drive.zip  # specify model
  python visualize.py --manual                           # manual control
  python visualize.py --demo                             # demo without model
  python visualize.py --list-models                      # show available models
        """
    )
    
    parser.add_argument("--manual", action="store_true", 
                       help="manual control mode with keyboard")
    parser.add_argument("--demo", action="store_true", 
                       help="demo without model (random actions)")
    parser.add_argument("--model", "-m", type=str,
                       help="path to model file (e.g., models/ppo_diff_drive.zip)")
    parser.add_argument("--model-type", "-t", type=str, default="auto",
                       choices=['auto', 'ppo', 'a2c', 'dqn', 'qlearning', 'pytorch'],
                       help="model type (default: auto)")
    parser.add_argument("--list-models", "-l", action="store_true",
                       help="show available models in models/ folder")
    
    args = parser.parse_args()
    
    if args.demo and args.manual:
        print("Cannot use --demo and --manual simultaneously")
        sys.exit(1)
    
    if args.list_models:
        print("\nAvailable models:")
        models = find_available_models()
        if models:
            for i, m in enumerate(models, 1):
                print(f"  {i}. {m['name']} ({m['type']})")
                print(f"     Path: {m['path']}")
        else:
            print("  No models available in the models/ folder")
        print("\nTip: use --model <path> to load a model")
        sys.exit(0)
    
    run(manual_mode=args.manual, demo_mode=args.demo, 
        model_path=args.model, model_type=args.model_type)