# src/trainers/base_trainer.py
"""
Base trainer class for all RL algorithms.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from environment import DiffDriveEnv
from config import Config


import imageio
import numpy as np
from PIL import Image
import tempfile
import shutil

class GIFRecorder:
    """Record agent performance and save as GIF."""
    
    def __init__(self, env, model, config, max_steps=600):
        self.env = env
        self.model = model
        self.config = config
        self.max_steps = max_steps
        self.frames = []
        
    def record_episode(self, deterministic=True, seed=None):
        """Record one episode and return frames."""
        self.frames = []
        obs, _ = self.env.reset(seed=seed)
        done = False
        step = 0
        
        while not done and step < self.max_steps:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step += 1
            
            # Capture frame (you'll need to implement render method)
            frame = self.env.render(mode='rgb_array')
            if frame is not None:
                self.frames.append(frame)
        
        return self.frames
    
    def save_to_gif(self, filename, fps=30):
        """Save recorded frames as GIF."""
        if not self.frames:
            print("No frames recorded. Run record_episode() first.")
            return
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save as GIF
        with imageio.get_writer(filename, mode='I', fps=fps, loop=0) as writer:
            for frame in self.frames:
                writer.append_data(frame)
        
        print(f"GIF saved to: {filename}")
    
    def save_multiple_episodes(self, n_episodes=5, output_dir="gifs", fps=30):
        """Record and save multiple episodes."""
        os.makedirs(output_dir, exist_ok=True)
        
        for ep in range(n_episodes):
            filename = f"{output_dir}/{self.config.training.model_name}_ep{ep+1}.gif"
            self.record_episode(seed=ep*100)
            self.save_to_gif(filename, fps)
            print(f"Episode {ep+1}/{n_episodes} recorded")


class ProgressCallback(BaseCallback):
    """Callback to track training progress."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.log_every = config.training.log_every
        self.ep_rewards = []
        self.ep_lengths = []
        self.success_flags = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
            if info.get("success"):
                self.success_flags.append(1.0)
            elif "episode" in info:
                self.success_flags.append(0.0)

        if self.n_calls % self.log_every == 0:
            n = len(self.ep_rewards)
            if n > 0:
                recent = self.ep_rewards[-100:]
                sr = np.mean(self.success_flags[-200:]) * 100 if self.success_flags else 0
                print(
                    f"  step={self.num_timesteps:>8,} | "
                    f"ep={n:>5} | "
                    f"reward(last100)={np.mean(recent):>8.1f} | "
                    f"success%={sr:>5.1f}"
                )
        return True


class SaveCallback(BaseCallback):
    """Callback to save model periodically."""
    
    def __init__(self, config: Config, save_freq: int):
        super().__init__()
        self.config = config
        self.save_freq = save_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = f"{self.config.training.model_dir}/{self.config.training.model_name}_step{self.n_calls}"
            self.model.save(path)
            print(f" Model saved to: {path}")
        return True


class BaseTrainer:
    """Base class for all trainers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.callbacks = []
        
    def make_env(self):
        """Create environment with given config."""
        self.config.apply_to_env(DiffDriveEnv)
        return Monitor(DiffDriveEnv())
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        cb_progress = ProgressCallback(self.config)
        cb_save = SaveCallback(self.config, self.config.training.save_freq)
        self.callbacks = [cb_progress, cb_save]
        
        # Eval callback
        if self.config.training.eval_freq > 0:
            eval_env = make_vec_env(lambda: self.make_env(), n_envs=1)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.config.training.model_dir,
                log_path=self.config.training.log_dir,
                eval_freq=self.config.training.eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=self.config.training.eval_episodes
            )
            self.callbacks.append(eval_callback)
        
        return self.callbacks
    
    def create_model(self, vec_env):
        """Create the RL model - to be implemented by subclasses."""
        raise NotImplementedError
    
    def train(self):
        """Train the model."""
        print("=" * 60)
        print(f"  Training {self.get_algo_name()} — Differential Drive → (0, 0, 0)")
        print("=" * 60)
        
        self.config.print_summary()
        
        # Create vectorized environment
        vec_env = make_vec_env(lambda: self.make_env(), n_envs=self.config.training.n_envs)
        
        # Create model
        self.model = self.create_model(vec_env)
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Create directories
        os.makedirs(self.config.training.model_dir, exist_ok=True)
        os.makedirs(self.config.training.plots_dir, exist_ok=True)
        os.makedirs(self.config.training.log_dir, exist_ok=True)
        
        print(f"\n Starting training ({self.config.training.total_timesteps:,} steps)...")
        print(f"   • {self.config.training.n_envs} parallel environments")
        print(f"   • Saving every {self.config.training.save_freq:,} steps")
        print()
        
        t0 = time.time()
        self.model.learn(
            total_timesteps=self.config.training.total_timesteps,
            callback=self.callbacks,
            progress_bar=False
        )
        elapsed = time.time() - t0
        print(f"\nTraining completed in {elapsed:.1f} seconds")
        
        # Save final model
        model_path = f"{self.config.training.model_dir}/{self.config.training.model_name}"
        self.model.save(model_path)
        print(f"Model saved to → {model_path}.zip")
        
        # Save logs
        self._save_logs()
        
        # Save config
        self.config.save(f"{self.config.training.model_dir}/config.yaml")
        print(f"Config saved to → {self.config.training.model_dir}/config.yaml")
        
        # Plot training curves
        self._plot_training()
        
        if self.config.training.record_gifs:
            try:
                self._record_and_save_gifs(
                    n_episodes=self.config.training.get('n_gif_episodes', 5),
                    fps=self.config.training.get('gif_fps', 30)
                )
            except Exception as e:
                print(f"Warning: Could not record GIFs: {e}")
                print("Make sure imageio and pygame are installed properly")
        
        return self.model

    
    def _save_logs(self):
        """Save training logs."""
        # Find the progress callback
        progress_cb = None
        for cb in self.callbacks:
            if isinstance(cb, ProgressCallback):
                progress_cb = cb
                break
        
        if progress_cb:
            log_path = f"{self.config.training.model_dir}/{self.config.training.log_name}"
            np.savez(
                log_path,
                ep_rewards=np.array(progress_cb.ep_rewards),
                ep_lengths=np.array(progress_cb.ep_lengths),
                success_flags=np.array(progress_cb.success_flags),
            )
            print(f"Logs saved to → {log_path}")
    
    def _smooth(self, arr, w=50):
        """Smooth array with moving average."""
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")
    
    def _plot_training(self):
        """Plot training curves."""
        # Find the progress callback
        progress_cb = None
        for cb in self.callbacks:
            if isinstance(cb, ProgressCallback):
                progress_cb = cb
                break
        
        if not progress_cb or len(progress_cb.ep_rewards) == 0:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0f0f1a")
        fig.suptitle(f"Training Curves — {self.get_algo_name()}",
                     color="white", fontsize=14, fontweight="bold", y=0.98)

        COLOR_RAW = "#3a6fc4"
        COLOR_SMOOTH = "#f0a500"
        COLOR_SR = "#2ecc71"

        rewards = np.array(progress_cb.ep_rewards)
        success = np.array(progress_cb.success_flags)

        # Reward plot
        ax1 = axes[0]
        ax1.set_facecolor("#12122a")
        ep_idx = np.arange(len(rewards))
        ax1.plot(ep_idx, rewards, color=COLOR_RAW, alpha=0.25, linewidth=0.5)
        if len(rewards) >= 50:
            ax1.plot(
                ep_idx[49:], self._smooth(rewards),
                color=COLOR_SMOOTH, linewidth=2, label="Smoothed average (50 episodes)"
            )
        ax1.set_ylabel("Total Reward", color="white")
        ax1.set_xlabel("Episode", color="white")
        ax1.tick_params(colors="white")
        ax1.legend(facecolor="#12122a", labelcolor="white")
        ax1.spines[:].set_color("#444")
        ax1.set_title("Total Reward per Episode", color="white", fontsize=11)
        
        ax1.text(0.02, 0.98, 
                 f"V_max={self.config.env.V_MAX} | Goal_dist={self.config.env.GOAL_DIST}",
                 transform=ax1.transAxes, color='gray', fontsize=8,
                 verticalalignment='top')

        # Success rate plot
        ax2 = axes[1]
        ax2.set_facecolor("#12122a")
        if len(success) > 0:
            sr = self._smooth(success, w=100) * 100
            ax2.plot(np.arange(len(sr)), sr, color=COLOR_SR, linewidth=2)
            ax2.axhline(y=80, color="#ff5555", linestyle="--", alpha=0.5, label="80% target")
            ax2.axhline(y=95, color="#ffaa55", linestyle=":", alpha=0.5, label="95% target")
            ax2.set_ylim(0, 105)
        ax2.set_ylabel("Success rate, %", color="white")
        ax2.set_xlabel("Episode", color="white")
        ax2.tick_params(colors="white")
        ax2.legend(facecolor="#12122a", labelcolor="white")
        ax2.spines[:].set_color("#444")
        ax2.set_title("Success Rate (Rolling Average of 100 Episodes)", 
                      color="white", fontsize=11)

        plt.tight_layout()
        path = f"{self.config.training.plots_dir}/training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Graph saved to → {path}")
    
    def get_algo_name(self):
        """Return algorithm name."""
        return self.__class__.__name__.replace('Trainer', '')
    
    def _record_and_save_gifs(self, n_episodes=5, fps=30):
        """Record and save GIFs of the trained agent."""
        print("\n" + "=" * 60)
        print("Recording GIFs of the trained agent...")
        print("=" * 60)
        
        # Create a fresh environment with rendering enabled
        eval_env = self.make_env()
        
        # Add render mode if supported
        if hasattr(eval_env, 'render_mode'):
            eval_env.render_mode = 'rgb_array'
        
        # Create recorder
        recorder = GIFRecorder(eval_env, self.model, self.config)
        
        # Create output directory
        gif_dir = f"{self.config.training.plots_dir}/gifs"
        os.makedirs(gif_dir, exist_ok=True)
        
        # Record episodes from different starting positions
        start_positions = [
            None,  # Random
            (8.0, 8.0, 0.0),      # Far corner, facing forward
            (8.0, 8.0, 2.0),      # Far corner, facing wrong direction
            (-8.0, -8.0, 0.0),    # Opposite corner
            (5.0, -5.0, 1.57),    # Side, facing away
            (0.0, 8.0, 3.14),     # Top, facing down
        ]
        
        for i, start_pos in enumerate(start_positions[:n_episodes]):
            if start_pos is not None:
                # Set specific start position
                eval_env.x, eval_env.y, eval_env.theta = start_pos
                eval_env.trajectory = [(eval_env.x, eval_env.y, eval_env.theta)]
                print(f"\nRecording episode {i+1} from position: "
                      f"x={start_pos[0]:.1f}, y={start_pos[1]:.1f}, "
                      f"angle={np.degrees(start_pos[2]):.1f}°")
            else:
                # Random start
                eval_env.reset()
                print(f"\nRecording episode {i+1} from random start")
            
            # Record episode
            recorder.record_episode(deterministic=True)
            
            # Save GIF
            gif_path = f"{gif_dir}/{self.config.training.model_name}_episode_{i+1}.gif"
            recorder.save_to_gif(gif_path, fps=fps)
        
        print(f"\nAll GIFs saved to: {gif_dir}/")