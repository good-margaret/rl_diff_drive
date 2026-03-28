# src/trainers/td3_trainer.py
"""
TD3 (Twin Delayed DDPG) Trainer for differential drive robot.
"""

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env

from .base_trainer import BaseTrainer
from config import Config


class TD3Trainer(BaseTrainer):
    """TD3 algorithm trainer."""
    
    def __init__(self, config: Config):
        super().__init__(config)
    
    def create_model(self, vec_env):
        """Create TD3 model."""
        return TD3(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self.config.td3.learning_rate,
            buffer_size=self.config.td3.buffer_size,
            learning_starts=self.config.td3.learning_starts,
            batch_size=self.config.td3.batch_size,
            tau=self.config.td3.tau,
            gamma=self.config.td3.gamma,
            train_freq=self.config.td3.train_freq,
            gradient_steps=self.config.td3.gradient_steps,
            policy_delay=self.config.td3.policy_delay,
            policy_kwargs=dict(net_arch=self.config.td3.net_arch),
            verbose=0,
        )