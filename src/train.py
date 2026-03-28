# """
# train.py — обучение агента PPO для задачи управления дифференциальным приводом.

# Запуск:
#     python train.py                              # стандартная конфигурация
#     python train.py --config fast                # быстрая конфигурация
#     python train.py --config precision           # высокая точность
#     python train.py --config robust              # устойчивое обучение
#     python train.py --config my_config.yaml      # пользовательский конфиг
#     python train.py --timesteps 1000000          # изменить количество шагов
#     python train.py --list-configs               # показать доступные конфиги
#     python train.py --create-template            # создать шаблон конфигурации
# """

# import os
# import sys
# import time
# import argparse
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
# from stable_baselines3.common.monitor import Monitor

# from environment import DiffDriveEnv
# from config import Config, load_config, list_available_configs, create_config_template

# # ══════════════════════════════════════════════════════════════════════════════
# class ProgressCallback(BaseCallback):
#     """Выводит прогресс каждые N шагов и собирает кривые."""

#     def __init__(self, config: Config):
#         super().__init__()
#         self.config = config
#         self.log_every = config.training.log_every
#         self.ep_rewards: list[float] = []
#         self.ep_lengths: list[int]   = []
#         self.success_flags: list[float] = []

#     def _on_step(self) -> bool:
#         for info in self.locals.get("infos", []):
#             if "episode" in info:
#                 self.ep_rewards.append(info["episode"]["r"])
#                 self.ep_lengths.append(info["episode"]["l"])
#             if info.get("success"):
#                 self.success_flags.append(1.0)
#             elif "episode" in info:
#                 self.success_flags.append(0.0)

#         if self.n_calls % self.log_every == 0:
#             n = len(self.ep_rewards)
#             if n > 0:
#                 recent = self.ep_rewards[-100:]
#                 sr = np.mean(self.success_flags[-200:]) * 100 if self.success_flags else 0
#                 print(
#                     f"  step={self.num_timesteps:>8,} | "
#                     f"ep={n:>5} | "
#                     f"reward(last100)={np.mean(recent):>8.1f} | "
#                     f"success%={sr:>5.1f}"
#                 )
#         return True


# # ══════════════════════════════════════════════════════════════════════════════
# class SaveCallback(BaseCallback):
#     """Периодическое сохранение модели."""
    
#     def __init__(self, config: Config, save_freq: int):
#         super().__init__()
#         self.config = config
#         self.save_freq = save_freq
        
#     def _on_step(self) -> bool:
#         if self.n_calls % self.save_freq == 0:
#             path = f"{self.config.training.model_dir}/{self.config.training.model_name}_step{self.n_calls}"
#             self.model.save(path)
#             print(f" Model saved to: {path}")
#         return True


# # ══════════════════════════════════════════════════════════════════════════════
# def make_env(config: Config):
#     """Создает среду с заданной конфигурацией."""
#     config.apply_to_env(DiffDriveEnv)
#     return Monitor(DiffDriveEnv())


# def train(config: Config):
#     print("=" * 60)
#     print("  Training PPO — Differential Drive → (0, 0, 0)")
#     print("=" * 60)
    
#     # Выводим конфигурацию
#     config.print_summary()
    
#     # Создаем векторную среду
#     vec_env = make_vec_env(
#         lambda: make_env(config), 
#         n_envs=config.training.n_envs
#     )
    
#     # Создаем модель PPO
#     model = PPO(
#         policy          = "MlpPolicy",
#         env             = vec_env,
#         learning_rate   = config.ppo.learning_rate,
#         n_steps         = config.ppo.n_steps,
#         batch_size      = config.ppo.batch_size,
#         n_epochs        = config.ppo.n_epochs,
#         gamma           = config.ppo.gamma,
#         gae_lambda      = config.ppo.gae_lambda,
#         clip_range      = config.ppo.clip_range,
#         ent_coef        = config.ppo.ent_coef,
#         vf_coef         = config.ppo.vf_coef,
#         max_grad_norm   = config.ppo.max_grad_norm,
#         policy_kwargs   = dict(net_arch=config.ppo.net_arch),
#         verbose         = 0,
#     )
    
#     # Колбэки
#     cb_progress = ProgressCallback(config)
#     cb_save = SaveCallback(config, config.training.save_freq)
    
#     callbacks = [cb_progress, cb_save]
    
#     # Eval callback
#     if config.training.eval_freq > 0:
#         eval_env = make_vec_env(lambda: make_env(config), n_envs=1)
#         eval_callback = EvalCallback(
#             eval_env,
#             best_model_save_path=config.training.model_dir,
#             log_path=config.training.log_dir,
#             eval_freq=config.training.eval_freq,
#             deterministic=True,
#             render=False,
#             n_eval_episodes=config.training.eval_episodes
#         )
#         callbacks.append(eval_callback)
    
#     print(f"\n Start ot the training ({config.training.total_timesteps:,} шагов)...")
#     print(f"   • {config.training.n_envs} parallel environments")
#     print(f"   • Saving every {config.training.save_freq:,} steps")
#     print()
    
#     t0 = time.time()
#     model.learn(
#         total_timesteps=config.training.total_timesteps, 
#         callback=callbacks, 
#         progress_bar=False
#     )
#     elapsed = time.time() - t0
#     print(f"\nTraining completed in {elapsed:.1f} seconds")
    
#     # Сохраняем финальную модель
#     model_path = f"{config.training.model_dir}/{config.training.model_name}"
#     model.save(model_path)
#     print(f"Model saved to → {model_path}.zip")
    
#     # Сохраняем логи
#     log_path = f"{config.training.model_dir}/{config.training.log_name}"
#     np.savez(
#         log_path,
#         ep_rewards    = np.array(cb_progress.ep_rewards),
#         ep_lengths    = np.array(cb_progress.ep_lengths),
#         success_flags = np.array(cb_progress.success_flags),
#     )
#     print(f"Logs saved to → {log_path}")
    
#     # Сохраняем конфигурацию
#     config.save(f"{config.training.model_dir}/config.yaml")
#     print(f"Config saved to → {config.training.model_dir}/config.yaml")
    
#     # Строим графики
#     _plot_training(cb_progress, config)
    
#     return model


# # ══════════════════════════════════════════════════════════════════════════════
# def _smooth(arr, w=50):
#     if len(arr) < w:
#         return arr
#     return np.convolve(arr, np.ones(w) / w, mode="valid")


# def _plot_training(cb: ProgressCallback, config: Config):
#     """Сохраняет красивый график кривых обучения."""
#     fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0f0f1a")
#     fig.suptitle("Training Curves — Differential Drive PPO",
#                  color="white", fontsize=14, fontweight="bold", y=0.98)

#     COLOR_RAW    = "#3a6fc4"
#     COLOR_SMOOTH = "#f0a500"
#     COLOR_SR     = "#2ecc71"

#     rewards = np.array(cb.ep_rewards)
#     success = np.array(cb.success_flags)

#     # ── (1) Награда за эпизод ─────────────────────────────────────────────────
#     ax1 = axes[0]
#     ax1.set_facecolor("#12122a")
#     ep_idx = np.arange(len(rewards))
#     ax1.plot(ep_idx, rewards, color=COLOR_RAW, alpha=0.25, linewidth=0.5)
#     if len(rewards) >= 50:
#         ax1.plot(
#             ep_idx[49:], _smooth(rewards),
#             color=COLOR_SMOOTH, linewidth=2, label="smoothed average (50 episodes)"
#         )
#     ax1.set_ylabel("Total Reward", color="white")
#     ax1.set_xlabel("Episode", color="white")
#     ax1.tick_params(colors="white")
#     ax1.legend(facecolor="#12122a", labelcolor="white")
#     ax1.spines[:].set_color("#444")
#     ax1.set_title("Total Reward", color="white", fontsize=11)
    
#     # Добавляем информацию о конфигурации
#     ax1.text(0.02, 0.98, 
#              f"V_max={config.env.V_MAX} | Goal_dist={config.env.GOAL_DIST}",
#              transform=ax1.transAxes, color='gray', fontsize=8,
#              verticalalignment='top')

#     # ── (2) Success rate ──────────────────────────────────────────────────────
#     ax2 = axes[1]
#     ax2.set_facecolor("#12122a")
#     if len(success) > 0:
#         sr = _smooth(success, w=100) * 100
#         ax2.plot(np.arange(len(sr)), sr, color=COLOR_SR, linewidth=2)
#         ax2.axhline(y=80, color="#ff5555", linestyle="--", alpha=0.5, label="80% цель")
#         ax2.axhline(y=95, color="#ffaa55", linestyle=":", alpha=0.5, label="95% цель")
#         ax2.set_ylim(0, 105)
#     ax2.set_ylabel("Success rate, %", color="white")
#     ax2.set_xlabel("Episode", color="white")
#     ax2.tick_params(colors="white")
#     ax2.legend(facecolor="#12122a", labelcolor="white")
#     ax2.spines[:].set_color("#444")
#     ax2.set_title("Success Rate (Rolling Average of 100 Episodes)", 
#                   color="white", fontsize=11)

#     plt.tight_layout()
#     path = f"{config.training.plots_dir}/training_curves.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"Graph saved to → {path}")


# # ══════════════════════════════════════════════════════════════════════════════
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Training PPO for Differential Drive",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples of use:
#   python train.py                              # default configuration
#   python train.py --config fast                # fast configuration
#   python train.py --config precision           # high precision configuration
#   python train.py --config my_config.yaml      # custom config
#   python train.py --timesteps 1000000          # change the number of steps
#   python train.py --list-configs               # show available configs
#   python train.py --create-template            # create a config template
#         """
#     )
    
#     parser.add_argument("--config", "-c", type=str, default="default",
#                        help="Configuration name (default/fast/precision/robust) or path to YAML file")
    
#     parser.add_argument("--timesteps", "-t", type=int,
#                        help="Override the number of training steps")
    
#     parser.add_argument("--n-envs", "-n", type=int,
#                        help="Override the number of parallel environments")
    
#     parser.add_argument("--lr", type=float,
#                        help="Override the learning rate")
    
#     parser.add_argument("--list-configs", "-l", action="store_true",
#                        help="Show available configurations")
    
#     parser.add_argument("--create-template", action="store_true",
#                        help="Create a config template in the configs/ folder")
    
#     return parser.parse_args()


# # ══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     args = parse_args()
    
#     # Показать доступные конфигурации
#     if args.list_configs:
#         print("\n Available configurations:")
#         configs = list_available_configs()
#         if configs:
#             for cfg in configs:
#                 print(f"  • {cfg}")
#         else:
#             print("  No configurations found in configs/ folder")
#         print("\nTip: create a file configs/my_config.yaml")
#         exit(0)
    
#     # Создать шаблон конфигурации
#     if args.create_template:
#         os.makedirs("configs", exist_ok=True)
#         create_config_template("configs/template.yaml")
#         print("\nTemplate created. Edit it to suit your needs:")
#         print("   configs/template.yaml")
#         exit(0)
    
#     # Загружаем конфигурацию
#     try:
#         # Проверяем, является ли аргумент путем к файлу
#         if args.config.endswith(('.yaml', '.yml')):
#             config = Config.from_yaml(args.config)
#             print(f"Loaded configuration from {args.config}")
#         else:
#             config = load_config(args.config)
#             print(f"Loaded configuration: {args.config}")
#     except FileNotFoundError as e:
#         print(f"❌ {e}")
#         print(f"\nAvailable configurations: {list_available_configs()}")
#         exit(1)
    
#     # Переопределяем параметры из командной строки
#     if args.timesteps:
#         config.training.total_timesteps = args.timesteps
#         print(f"   • Overridden: timesteps = {args.timesteps:,}")
    
#     if args.n_envs:
#         config.training.n_envs = args.n_envs
#         print(f"   • Overridden: n_envs = {args.n_envs}")
    
#     if args.lr:
#         config.ppo.learning_rate = args.lr
#         print(f"   • Overridden: learning_rate = {args.lr}")
    
#     # Запускаем обучение
#     train(config)

# src/train.py
"""
train.py — Training different RL algorithms for differential drive robot.

Usage:
    python train.py                                      # Default PPO training
    python train.py --algo sac --config default          # Train SAC
    python train.py --algo td3 --config fast             # Train TD3 with fast config
    python train.py --algo ppo --config improved         # Train PPO with improved config
    python train.py --list-algos                         # List available algorithms
    python train.py --list-configs ppo                   # List configs for specific algo
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, load_config, list_available_configs, create_config_template
from trainers import PPOTrainer, SACTrainer, TD3Trainer


def get_trainer(algo_name, config):
    """Get trainer instance for given algorithm."""
    trainers = {
        'ppo': PPOTrainer,
        'sac': SACTrainer,
        'td3': TD3Trainer,
    }
    
    if algo_name not in trainers:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(trainers.keys())}")
    
    return trainers[algo_name](config)


def list_available_algorithms():
    """List available algorithms."""
    print("\nAvailable algorithms:")
    print("  • ppo  - Proximal Policy Optimization")
    print("  • sac  - Soft Actor-Critic (recommended for continuous control)")
    print("  • td3  - Twin Delayed DDPG")
    print("\nTips:")
    print("  - SAC usually works best for this task")
    print("  - Use --algo sac --config fast for quick testing")
    print("  - Use --algo ppo --config improved for better PPO results")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training RL algorithms for Differential Drive Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                                    # Default PPO training
  python train.py --algo sac --config default        # Train SAC
  python train.py --algo ppo --config improved       # Train PPO with improved config
  python train.py --algo td3 --config fast           # Train TD3 with fast config
  python train.py --list-algos                       # List available algorithms
  python train.py --list-configs ppo                 # List configs for PPO
        """
    )
    
    parser.add_argument("--algo", "-a", type=str, default="ppo",
                       choices=['ppo', 'sac', 'td3'],
                       help="Algorithm to train (default: ppo)")
    
    parser.add_argument("--config", "-c", type=str, default="default",
                       help="Configuration name or path to YAML file")
    
    parser.add_argument("--timesteps", "-t", type=int,
                       help="Override total training steps")
    
    parser.add_argument("--n-envs", "-n", type=int,
                       help="Override number of parallel environments")
    
    parser.add_argument("--lr", type=float,
                       help="Override learning rate")
    
    parser.add_argument("--list-algos", action="store_true",
                       help="List available algorithms")
    
    parser.add_argument("--list-configs", type=str, nargs='?', const='all',
                       help="List available configurations for an algorithm")
    
    parser.add_argument("--create-template", type=str, nargs='?', const='ppo',
                       help="Create a template config for an algorithm")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List algorithms
    if args.list_algos:
        list_available_algorithms()
        return
    
    # List configs
    if args.list_configs:
        if args.list_configs == 'all':
            print("\nAvailable configurations by algorithm:")
            for algo in ['ppo', 'sac', 'td3']:
                configs = list_available_configs(f"configs/{algo}")
                if configs:
                    print(f"\n{algo.upper()}:")
                    for cfg in configs:
                        print(f"  • {cfg}")
                else:
                    print(f"\n{algo.upper()}: No configurations found")
        else:
            configs = list_available_configs(f"configs/{args.list_configs}")
            if configs:
                print(f"\nAvailable configurations for {args.list_configs.upper()}:")
                for cfg in configs:
                    print(f"  • {cfg}")
            else:
                print(f"No configurations found for {args.list_configs}")
        return
    
    # Create template
    if args.create_template:
        algo = args.create_template
        os.makedirs(f"configs/{algo}", exist_ok=True)
        template_path = f"configs/{algo}/template.yaml"
        create_config_template(template_path)
        print(f"\nTemplate created for {algo.upper()}: {template_path}")
        print("Edit this file to configure your training")
        return
    
    # Load configuration
    config_path = None
    if args.config.endswith(('.yaml', '.yml')):
        config_path = args.config
        config = Config.from_yaml(config_path)
        print(f"Loaded configuration from {config_path}")
    else:
        # Try to load from algorithm-specific config folder
        algo_config_path = f"configs/{args.algo}/{args.config}.yaml"
        if os.path.exists(algo_config_path):
            config = Config.from_yaml(algo_config_path)
            print(f"Loaded configuration: {args.algo}/{args.config}")
        else:
            # Fall back to old config location
            config = load_config(args.config)
            print(f"Loaded configuration: {args.config}")
    
    # Override parameters
    if args.timesteps:
        config.training.total_timesteps = args.timesteps
        print(f"   • Overridden: timesteps = {args.timesteps:,}")
    
    if args.n_envs:
        config.training.n_envs = args.n_envs
        print(f"   • Overridden: n_envs = {args.n_envs}")
    
    if args.lr:
        if args.algo == 'ppo':
            config.ppo.learning_rate = args.lr
        elif args.algo == 'sac':
            config.sac.learning_rate = args.lr
        elif args.algo == 'td3':
            config.td3.learning_rate = args.lr
        print(f"   • Overridden: learning_rate = {args.lr}")
    
    # Update model name and paths to include algorithm
    config.training.model_name = f"{args.algo}_{config.training.model_name}"
    config.training.model_dir = f"models/{args.algo}/{config.training.model_name}"
    config.training.plots_dir = f"plots/{args.algo}/{config.training.model_name}"
    config.training.log_dir = f"logs/{args.algo}/{config.training.model_name}"
    
    # Create trainer and start training
    trainer = get_trainer(args.algo, config)
    trainer.train()


if __name__ == "__main__":
    main()