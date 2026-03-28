# src/models/sac_model.py
"""
SAC (Soft Actor-Critic) implementation compatible with Stable-Baselines3 API.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.observations[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_obs
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample random batch from buffer."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        obs = torch.FloatTensor(self.observations[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_obs = torch.FloatTensor(self.next_observations[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return self.size


class SACPolicy(nn.Module):
    """Gaussian policy network for SAC."""
    
    def __init__(self, observation_dim: int, action_dim: int, net_arch: List[int] = [256, 256]):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Build policy network
        layers = []
        input_dim = observation_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)
        
        # Mean and log std heads
        self.mean_net = nn.Linear(input_dim, action_dim)
        self.log_std_net = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.shared_net(obs)
        mean = self.mean_net(features)
        log_std = self.log_std_net(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = self._compute_log_prob(mean, std, action)
        else:
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)
            log_prob = self._compute_log_prob(mean, std, action, z)
        
        return action, log_prob
    
    def _compute_log_prob(self, mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor, 
                          z: torch.Tensor = None) -> torch.Tensor:
        """Compute log probability of action."""
        if z is None:
            # Inverse tanh
            z = torch.atanh(torch.clamp(action, -0.999999, 0.999999))
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob


class SACQNetwork(nn.Module):
    """Q-network for SAC (twin critics)."""
    
    def __init__(self, observation_dim: int, action_dim: int, net_arch: List[int] = [256, 256]):
        super().__init__()
        
        # Build Q-network
        layers = []
        input_dim = observation_dim + action_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.q_net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([obs, action], dim=-1)
        return self.q_net(x)


class SACModel:
    """
    SAC (Soft Actor-Critic) implementation.
    
    Compatible with Stable-Baselines3 API.
    """
    
    def __init__(
        self,
        policy: str = "MlpPolicy",
        env: gym.Env = None,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: float = 0.2,
        target_entropy: float = None,
        net_arch: List[int] = [256, 256],
        device: str = "cpu",
        seed: int = None,
        verbose: int = 0,
    ):
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.ent_coef = ent_coef
        self.net_arch = net_arch
        self.device = torch.device(device)
        self.verbose = verbose
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Create environment
        self.env = env
        if env is not None:
            self._setup_model()
            if target_entropy is None:
                self.target_entropy = -self.action_dim
            else:
                self.target_entropy = target_entropy
    
    def _setup_model(self):
        """Setup the model."""
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Create networks
        self.policy = SACPolicy(self.observation_dim, self.action_dim, self.net_arch).to(self.device)
        self.q1 = SACQNetwork(self.observation_dim, self.action_dim, self.net_arch).to(self.device)
        self.q2 = SACQNetwork(self.observation_dim, self.action_dim, self.net_arch).to(self.device)
        
        self.target_q1 = SACQNetwork(self.observation_dim, self.action_dim, self.net_arch).to(self.device)
        self.target_q2 = SACQNetwork(self.observation_dim, self.action_dim, self.net_arch).to(self.device)
        
        # Copy parameters
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.learning_rate)
        self.q1_optimizer = Adam(self.q1.parameters(), lr=self.learning_rate)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, self.observation_dim, self.action_dim, self.device
        )
        
        # Entropy coefficient (automatic tuning)
        self.log_ent_coef = torch.log(torch.tensor(self.ent_coef)).to(self.device)
        self.log_ent_coef.requires_grad = True
        self.ent_coef_optimizer = Adam([self.log_ent_coef], lr=self.learning_rate)
        
        # Training counters
        self.num_timesteps = 0
        self._episode_num = 0
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        """Predict action for given observation."""
        if isinstance(obs, dict):
            obs = obs['observation']
        
        # Ensure observation is 2D
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        actions = []
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            for o in obs_tensor:
                mean, _ = self.policy.forward(o.unsqueeze(0))
                if deterministic:
                    action = torch.tanh(mean)
                else:
                    action, _ = self.policy.sample_action(o.unsqueeze(0))
                actions.append(action.cpu().numpy().squeeze())
        
        return np.array(actions), None
    
    def learn(self, total_timesteps: int, callback=None, progress_bar: bool = False):
        """Train the agent."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        
        while self.num_timesteps < total_timesteps:
            # Collect experience
            action, _ = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action[0])
            done = terminated or truncated
            
            # Store in replay buffer
            self.replay_buffer.push(obs[0], action[0], reward, next_obs[0], done)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            self.num_timesteps += 1
            
            if done:
                obs, _ = self.env.reset()
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                
                # Callback
                if callback is not None and hasattr(callback, '_on_step'):
                    callback.locals = {
                        'model': self,
                        'num_timesteps': self.num_timesteps,
                        'episode_rewards': episode_rewards,
                        'episode_lengths': episode_lengths,
                        'infos': [info]
                    }
                    callback._on_step()
                
                if self.verbose > 0 and len(episode_rewards) % 10 == 0:
                    mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                    print(f"Step: {self.num_timesteps}, Episode: {len(episode_rewards)}, "
                          f"Mean Reward: {mean_reward:.2f}")
            
            # Update model
            if self.num_timesteps >= self.learning_starts and self.num_timesteps % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self._update()
        
        return self
    
    def _update(self):
        """Update policy and Q-networks."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample_action(next_obs)
            target_q1 = self.target_q1(next_obs, next_actions)
            target_q2 = self.target_q2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.log_ent_coef.exp() * next_log_probs
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample_action(obs)
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.log_ent_coef.exp() * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update entropy coefficient
        ent_coef_loss = -(self.log_ent_coef * (log_probs.detach() + self.target_entropy).mean())
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        
        # Update target networks
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_ent_coef': self.log_ent_coef,
            'net_arch': self.net_arch,
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
        }, path)
    
    @classmethod
    def load(cls, path: str, env: gym.Env = None):
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model instance
        model = cls(
            env=env,
            net_arch=checkpoint['net_arch'],
            device='cpu'
        )
        
        # Load weights
        model.policy.load_state_dict(checkpoint['policy_state_dict'])
        model.q1.load_state_dict(checkpoint['q1_state_dict'])
        model.q2.load_state_dict(checkpoint['q2_state_dict'])
        model.target_q1.load_state_dict(checkpoint['target_q1_state_dict'])
        model.target_q2.load_state_dict(checkpoint['target_q2_state_dict'])
        model.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        model.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        model.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        model.log_ent_coef = checkpoint['log_ent_coef']
        
        return model