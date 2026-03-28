"""
evaluate.py — оценка обученной модели на N эпизодах.

Запуск: python evaluate.py [--episodes 100]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from stable_baselines3 import PPO
from environment import DiffDriveEnv

import os
os.makedirs("plots", exist_ok=True)


def evaluate(n_episodes=100, render_episodes=5, seed=42):
    print(f"Оценка модели на {n_episodes} эпизодах...")
    model = PPO.load("models/ppo_diff_drive")
    env   = DiffDriveEnv()

    successes = 0
    final_dists  = []
    final_angles = []
    ep_lengths   = []

    # Сохраним несколько красивых траекторий для визуализации
    saved_trajs = []

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        seed_ep = int(rng.integers(0, 1_000_000))
        obs, _ = env.reset(seed=seed_ep)
        done = False
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_len += 1

        successes += int(info["success"])
        final_dists.append(info["dist"])
        final_angles.append(info["angle"])
        ep_lengths.append(ep_len)

        if len(saved_trajs) < render_episodes and info["success"]:
            saved_trajs.append(list(env.trajectory))

    # ── статистика ────────────────────────────────────────────────────────────
    sr = successes / n_episodes * 100
    print(f"\n{'='*50}")
    print(f"  Success rate       : {sr:.1f}%  ({successes}/{n_episodes})")
    print(f"  Финальная dist (m) : mean={np.mean(final_dists):.2f}  "
          f"median={np.median(final_dists):.2f}")
    print(f"  Финальный угол (°) : mean={np.degrees(np.mean(final_angles)):.1f}")
    print(f"  Длина эп. (шаги)   : mean={np.mean(ep_lengths):.0f}  "
          f"min={np.min(ep_lengths)}  max={np.max(ep_lengths)}")
    print(f"{'='*50}")

    # ── красивый рисунок траекторий ───────────────────────────────────────────
    _plot_trajectories(saved_trajs)
    _plot_eval_hist(final_dists, final_angles)

    return sr


def _plot_trajectories(trajs):
    if not trajs:
        print("Нет успешных траекторий для визуализации.")
        return

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a18")
    ax.set_facecolor("#0f0f22")
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect("equal")
    ax.set_title("Примеры успешных траекторий агента PPO",
                 color="white", fontsize=13, fontweight="bold")

    # сетка
    for i in range(-10, 11, 2):
        ax.axhline(i, color=(0.12, 0.14, 0.28), linewidth=0.5)
        ax.axvline(i, color=(0.12, 0.14, 0.28), linewidth=0.5)
    ax.axhline(0, color=(0.25, 0.3, 0.55), linewidth=1.0)
    ax.axvline(0, color=(0.25, 0.3, 0.55), linewidth=1.0)

    # граница поля
    rect = plt.Rectangle((-10, -10), 20, 20, fill=False,
                          edgecolor=(0.3, 0.35, 0.7), linewidth=1.5)
    ax.add_patch(rect)

    # цель
    goal_circle = plt.Circle((0, 0), DiffDriveEnv.GOAL_DIST,
                              color="#2ecc71", alpha=0.35, zorder=5)
    ax.add_patch(goal_circle)
    ax.plot(0, 0, "+", color="#2ecc71", ms=14, mew=2, zorder=6)

    colors = ["#4e9dff", "#ff8c42", "#a8ff78", "#ff5ebd", "#ffd166"]
    for i, traj in enumerate(trajs):
        traj = np.array(traj)
        xs, ys = traj[:, 0], traj[:, 1]
        c = colors[i % len(colors)]
        ax.plot(xs, ys, color=c, linewidth=1.8, alpha=0.85, zorder=3)
        # старт
        ax.scatter(xs[0], ys[0], color=c, s=80, zorder=7,
                   edgecolors="white", linewidths=0.8)
        # стрелка в начале
        dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
        norm = max(np.hypot(dx, dy), 1e-6)
        ax.annotate("", xy=(xs[0] + dx / norm * 0.8, ys[0] + dy / norm * 0.8),
                    xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))

    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#303060")
    ax.set_xlabel("X, м", color="gray")
    ax.set_ylabel("Y, м", color="gray")

    plt.tight_layout()
    path = "plots/trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Траектории сохранены → {path}")


def _plot_eval_hist(dists, angles):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0a0a18")

    for ax, data, title, xlabel, color in [
        (ax1, dists,               "Финальное расстояние до цели", "dist, м",  "#4e9dff"),
        (ax2, np.degrees(angles),  "Финальная угловая ошибка",     "angle, °", "#ff8c42"),
    ]:
        ax.set_facecolor("#0f0f22")
        ax.hist(data, bins=30, color=color, alpha=0.8, edgecolor="black")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel(xlabel, color="gray")
        ax.set_ylabel("Кол-во эпизодов", color="gray")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#303060")

    plt.tight_layout()
    path = "plots/eval_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Гистограммы сохранены → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    evaluate(n_episodes=args.episodes)