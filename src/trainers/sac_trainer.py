# src/trainers/sac_trainer.py
"""
SAC (Soft Actor-Critic) Trainer for differential drive robot.
"""

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from .base_trainer import BaseTrainer
from config import Config


class SACTrainer(BaseTrainer):
    """SAC algorithm trainer."""
    
    def __init__(self, config: Config):
        super().__init__(config)
    
    def create_model(self, vec_env):
        """Create SAC model."""
        return SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self.config.sac.learning_rate,
            buffer_size=self.config.sac.buffer_size,
            learning_starts=self.config.sac.learning_starts,
            batch_size=self.config.sac.batch_size,
            tau=self.config.sac.tau,
            gamma=self.config.sac.gamma,
            train_freq=self.config.sac.train_freq,
            gradient_steps=self.config.sac.gradient_steps,
            ent_coef=self.config.sac.ent_coef,
            policy_kwargs=dict(net_arch=self.config.sac.net_arch),
            verbose=0,
        )