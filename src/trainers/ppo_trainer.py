# src/trainers/ppo_trainer.py
"""
PPO Trainer for differential drive robot.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .base_trainer import BaseTrainer
from config import Config


class PPOTrainer(BaseTrainer):
    """PPO algorithm trainer."""
    
    def __init__(self, config: Config):
        super().__init__(config)
    
    def create_model(self, vec_env):
        """Create PPO model."""
        return PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self.config.ppo.learning_rate,
            n_steps=self.config.ppo.n_steps,
            batch_size=self.config.ppo.batch_size,
            n_epochs=self.config.ppo.n_epochs,
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
            clip_range=self.config.ppo.clip_range,
            ent_coef=self.config.ppo.ent_coef,
            vf_coef=self.config.ppo.vf_coef,
            max_grad_norm=self.config.ppo.max_grad_norm,
            policy_kwargs=dict(net_arch=self.config.ppo.net_arch),
            verbose=0,
        )