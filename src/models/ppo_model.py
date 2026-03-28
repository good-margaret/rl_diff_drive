# src/models/ppo_model.py
"""
PPO (Proximal Policy Optimization) implementation compatible with Stable-Baselines3.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, Type


class MlpPolicy(nn.Module):
    """Multi-Layer Perceptron policy network."""
    
    def __init__(self, observation_dim: int, action_dim: int, net_arch: list = [256, 256]):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Build policy network (actor)
        policy_layers = []
        input_dim = observation_dim
        for hidden_dim in net_arch:
            policy_layers.append(nn.Linear(input_dim, hidden_dim))
            policy_layers.append(nn.Tanh())
            input_dim = hidden_dim
        policy_layers.append(nn.Linear(input_dim, action_dim))
        policy_layers.append(nn.Tanh())
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Build value network (critic)
        value_layers = []
        input_dim = observation_dim
        for hidden_dim in net_arch:
            value_layers.append(nn.Linear(input_dim, hidden_dim))
            value_layers.append(nn.Tanh())
            input_dim = hidden_dim
        value_layers.append(nn.Linear(input_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
        
        # Log standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy and value networks."""
        mean = self.policy_net(obs)
        value = self.value_net(obs)
        return mean, value
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mean, _ = self.forward(obs_tensor)
            
            if deterministic:
                action = mean
            else:
                std = torch.exp(self.log_std)
                action = torch.normal(mean, std)
            
            return action.squeeze(0).numpy()
    
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions and compute log probabilities and values."""
        mean, value = self.forward(obs)
        
        # Compute log probability of actions under Gaussian distribution
        std = torch.exp(self.log_std)
        var = std ** 2
        log_prob = -0.5 * (((actions - mean) ** 2) / var).sum(dim=-1)
        log_prob -= 0.5 * (self.action_dim * np.log(2 * np.pi) + var.log().sum())
        
        return mean, log_prob, value.squeeze(-1)


class PPOModel:
    """
    PPO (Proximal Policy Optimization) implementation.
    
    Compatible with Stable-Baselines3 API.
    """
    
    def __init__(
        self,
        policy: str = "MlpPolicy",
        env: gym.Env = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        net_arch: list = [64, 64],
        device: str = "cpu",
        seed: int = None,
        verbose: int = 0,
    ):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.net_arch = net_arch
        self.device = torch.device(device)
        self.verbose = verbose
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create environment
        self.env = env
        if env is not None:
            self._setup_model()
    
    def _setup_model(self):
        """Setup the model and optimizer."""
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.policy = MlpPolicy(
            observation_dim=observation_dim,
            action_dim=action_dim,
            net_arch=self.net_arch
        ).to(self.device)
        
        self.optimizer = Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Storage for trajectories
        self.obs_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.values_buffer = []
        self.log_probs_buffer = []
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        """Predict action for given observation."""
        if isinstance(obs, dict):
            obs = obs['observation']
        
        # Ensure observation is 2D
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        actions = []
        for o in obs:
            action = self.policy.get_action(o, deterministic)
            actions.append(action)
        
        return np.array(actions), None
    
    def learn(self, total_timesteps: int, callback=None, progress_bar: bool = False):
        """Train the agent."""
        total_steps = 0
        episode_rewards = []
        episode_lengths = []
        
        while total_steps < total_timesteps:
            # Collect trajectory
            trajectory_data = self._collect_trajectory()
            
            # Compute advantages and returns
            advantages, returns = self._compute_advantages_and_returns(
                trajectory_data['rewards'],
                trajectory_data['values'],
                trajectory_data['dones']
            )
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update policy
            policy_loss, value_loss, entropy_loss = self._update(
                trajectory_data['obs'],
                trajectory_data['actions'],
                trajectory_data['log_probs'],
                advantages,
                returns
            )
            
            # Update counters
            steps_this_iter = len(trajectory_data['obs'])
            total_steps += steps_this_iter
            episode_rewards.extend(trajectory_data.get('episode_rewards', []))
            episode_lengths.extend(trajectory_data.get('episode_lengths', []))
            
            # Callback
            if callback is not None:
                callback.locals = {
                    'model': self,
                    'num_timesteps': total_steps,
                    'episode_rewards': episode_rewards,
                    'episode_lengths': episode_lengths
                }
                callback._on_step()
            
            if self.verbose > 0 and total_steps % 10000 == 0:
                mean_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                print(f"Step: {total_steps}, Mean Reward (100 ep): {mean_reward:.2f}")
        
        return self
    
    def _collect_trajectory(self) -> Dict[str, np.ndarray]:
        """Collect one trajectory using current policy."""
        obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        log_probs_list = []
        episode_rewards = []
        episode_lengths = []
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(self.n_steps):
            # Get action and value
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                mean, value = self.policy.forward(obs_tensor)
                action = self.policy.get_action(obs)
                log_prob = self._compute_log_prob(mean, action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store data
            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            values_list.append(value.cpu().numpy().item())
            log_probs_list.append(log_prob)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
        
        return {
            'obs': np.array(obs_list),
            'actions': np.array(actions_list),
            'rewards': np.array(rewards_list),
            'dones': np.array(dones_list),
            'values': np.array(values_list),
            'log_probs': np.array(log_probs_list),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def _compute_log_prob(self, mean: torch.Tensor, action: np.ndarray) -> float:
        """Compute log probability of action under Gaussian policy."""
        with torch.no_grad():
            std = torch.exp(self.policy.log_std)
            action_tensor = torch.FloatTensor(action).to(self.device)
            
            # Compute log probability
            log_prob = -0.5 * ((action_tensor - mean) ** 2 / (std ** 2)).sum()
            log_prob -= 0.5 * (mean.shape[-1] * np.log(2 * np.pi) + (2 * std.log()).sum())
            
            return log_prob.cpu().numpy().item()
    
    def _compute_advantages_and_returns(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE and returns."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        return advantages, returns
    
    def _update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """Update policy using PPO objective."""
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Mini-batch updates
        n_samples = len(obs)
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            
            batch_obs = obs_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            
            # Forward pass
            mean, log_probs, values = self.policy.evaluate(batch_obs, batch_actions)
            
            # Policy loss with clipping
            ratio = torch.exp(log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
            
            # Value loss with clipping
            value_pred_clipped = batch_old_values + torch.clamp(
                values - batch_old_values, -self.clip_range, self.clip_range
            )
            value_loss = 0.5 * torch.max(
                (values - batch_returns) ** 2,
                (value_pred_clipped - batch_returns) ** 2
            ).mean()
            
            # Entropy bonus
            entropy = 0.5 * (torch.log(2 * np.pi * torch.exp(self.policy.log_std)) + 1).sum()
            entropy_loss = -self.ent_coef * entropy
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + entropy_loss
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        return total_policy_loss / (n_samples / self.batch_size), \
               total_value_loss / (n_samples / self.batch_size), \
               total_entropy_loss / (n_samples / self.batch_size)
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'net_arch': self.net_arch,
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
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model