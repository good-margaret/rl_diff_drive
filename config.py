"""
config.py — конфигурация для обучения и визуализации с поддержкой YAML.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path


# @dataclass
# class EnvConfig:
#     """Конфигурация среды."""
#     L: float = 1.0
#     V_MAX: float = 3.0
#     DT: float = 0.05
#     MAX_STEPS: int = 600
#     GOAL_DIST: float = 0.30
#     GOAL_ANGLE: float = 0.15
#     FIELD: float = 10.0
    
#     # Награды
#     REWARD_DIST_COEF: float = 0.10
#     REWARD_ANGLE_COEF: float = 0.03
#     REWARD_BOUNDARY_PENALTY: float = 1.0
#     REWARD_SUCCESS_BONUS: float = 200.0
    
#     @classmethod
#     def from_dict(cls, data: dict):
#         """Создает конфигурацию из словаря."""
#         return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class EnvConfig:
    """Configuration for environment."""
    L: float = 1.0
    V_MAX: float = 3.0
    DT: float = 0.05
    MAX_STEPS: int = 600
    GOAL_DIST: float = 0.30
    GOAL_ANGLE: float = 0.15
    FIELD: float = 10.0
    
    # Reward parameters
    REWARD_DIST_COEF: float = 0.10
    REWARD_ANGLE_COEF: float = 0.03
    REWARD_BOUNDARY_PENALTY: float = 1.0
    REWARD_SUCCESS_BONUS: float = 200.0
    REWARD_PROGRESS_COEF: float = 5.0
    REWARD_STOP_PENALTY: float = 0.5
    REWARD_ALIVE_BONUS: float = 0.01
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create EnvConfig from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
@dataclass
class PPOConfig:
    """Конфигурация PPO алгоритма."""
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    normalize_advantage: bool = True
    use_sde: bool = False
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create EnvConfig from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    


@dataclass
class SACConfig:
    """Configuration for SAC algorithm."""
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])

    @classmethod
    def from_dict(cls, data: dict):
        """Create EnvConfig from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    


@dataclass
class TD3Config:
    """Configuration for TD3 algorithm."""
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    net_arch: List[int] = field(default_factory=lambda: [256, 256])

    @classmethod
    def from_dict(cls, data: dict):
        """Create EnvConfig from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    total_timesteps: int = 500_000
    n_envs: int = 8
    log_every: int = 20_000
    eval_freq: int = 10_000
    eval_episodes: int = 10
    save_freq: int = 100_000
    
    # Paths
    model_dir: str = "models"
    plots_dir: str = "plots"
    log_dir: str = "logs"
    model_name: str = "ppo_diff_drive"
    log_name: str = "training_log.npz"
    
    # GIF recording
    record_gifs: bool = True
    n_gif_episodes: int = 5
    gif_fps: int = 30
    
    def __post_init__(self):
        """Create directories after initialization."""
        for dir_path in [self.model_dir, self.plots_dir, self.log_dir]:
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create TrainingConfig from dictionary."""
        # Filter only keys that exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
@dataclass
class MetadataConfig:
    """Метаданные эксперимента."""
    description: str = ""
    author: str = ""
    version: str = "1.0"
    created: str = ""
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Создает конфигурацию из словаря."""
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})



@dataclass
class Config:
    """Main configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Загружает конфигурацию из YAML файла."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            env=EnvConfig.from_dict(data.get('env', {})),
            ppo=PPOConfig.from_dict(data.get('ppo', {})),
            training=TrainingConfig.from_dict(data.get('training', {})),
            metadata=MetadataConfig.from_dict(data.get('metadata', {}))
        )
    
    def to_yaml(self, path: str):
        """Сохраняет конфигурацию в YAML файл."""
        # Конвертируем в словарь
        data = {
            'env': self.env.__dict__,
            'ppo': self.ppo.__dict__,
            'training': self.training.__dict__,
            'metadata': self.metadata.__dict__
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def save(self, path: str = "config.yaml"):
        """Сохраняет конфигурацию (алиас для to_yaml)."""
        self.to_yaml(path)
    
    @classmethod
    def load(cls, path: str = "config.yaml"):
        """Загружает конфигурацию (алиас для from_yaml)."""
        return cls.from_yaml(path)
    
    def print_summary(self):
        """Выводит краткую сводку конфигурации."""
        print("=" * 60)
        print("КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
        
        if self.metadata.description:
            print(f"\n{self.metadata.description}")
        
        print("\nENVIRONMENT PARAMETERS:")
        print(f"   • Field size: ±{self.env.FIELD} m")
        print(f"   • Robot base: {self.env.L} m")
        print(f"   • Max speed: {self.env.V_MAX} m/s")
        print(f"   • Goal: distance < {self.env.GOAL_DIST} m, angle < {self.env.GOAL_ANGLE:.2f} rad")
        print(f"   • Time step: {self.env.DT} s")
        print(f"   • Max steps: {self.env.MAX_STEPS}")
        
        print("\nPPO PARAMETERS:")
        print(f"   • Learning rate: {self.ppo.learning_rate}")
        print(f"   • Network: {' → '.join(map(str, self.ppo.net_arch))}")
        print(f"   • Batch size: {self.ppo.batch_size}")
        print(f"   • N steps: {self.ppo.n_steps}")
        print(f"   • Gamma: {self.ppo.gamma}")
        print(f"   • Entropy coeff: {self.ppo.ent_coef}")
        
        print("\nTRAINING PROCESS:")
        print(f"   • Total timesteps: {self.training.total_timesteps:,}")
        print(f"   • Parallel environments: {self.training.n_envs}")
        print(f"   • Saving to: {self.training.model_dir}/")
        print("=" * 60)
    
    def apply_to_env(self, env_class):
        """Применяет конфигурацию к классу среды."""
        env_class.L = self.env.L
        env_class.V_MAX = self.env.V_MAX
        env_class.DT = self.env.DT
        env_class.MAX_STEPS = self.env.MAX_STEPS
        env_class.GOAL_DIST = self.env.GOAL_DIST
        env_class.GOAL_ANGLE = self.env.GOAL_ANGLE
        env_class.FIELD = self.env.FIELD
        
        # Дополнительные параметры наград (если нужны в среде)
        if hasattr(env_class, 'REWARD_DIST_COEF'):
            env_class.REWARD_DIST_COEF = self.env.REWARD_DIST_COEF
        if hasattr(env_class, 'REWARD_ANGLE_COEF'):
            env_class.REWARD_ANGLE_COEF = self.env.REWARD_ANGLE_COEF
        if hasattr(env_class, 'REWARD_BOUNDARY_PENALTY'):
            env_class.REWARD_BOUNDARY_PENALTY = self.env.REWARD_BOUNDARY_PENALTY
        if hasattr(env_class, 'REWARD_SUCCESS_BONUS'):
            env_class.REWARD_SUCCESS_BONUS = self.env.REWARD_SUCCESS_BONUS


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты для работы с конфигурациями

def list_available_configs(config_dir: str = "configs") -> List[str]:
    """Возвращает список доступных конфигураций."""
    if not os.path.exists(config_dir):
        return []
    
    configs = []
    for file in os.listdir(config_dir):
        if file.endswith(('.yaml', '.yml')):
            configs.append(file.replace('.yaml', '').replace('.yml', ''))
    return configs


def load_config(config_name: str = "default", config_dir: str = "configs") -> Config:
    """
    Загружает конфигурацию по имени.
    
    Args:
        config_name: имя конфигурации (без расширения)
        config_dir: директория с конфигурациями
    
    Returns:
        Config объект
    """
    # Пробуем разные расширения
    for ext in ['.yaml', '.yml']:
        config_path = os.path.join(config_dir, f"{config_name}{ext}")
        if os.path.exists(config_path):
            return Config.from_yaml(config_path)
    
    raise FileNotFoundError(f"Config '{config_name}' not found in {config_dir}")


def create_config_template(path: str = "config_template.yaml"):
    """Создает шаблон конфигурации."""
    config = Config()
    config.metadata.description = "Шаблон конфигурации для обучения"
    config.metadata.author = "Your Name"
    config.metadata.version = "1.0"
    config.to_yaml(path)
    print(f"Config template created: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Предопределенные конфигурации (для обратной совместимости)

def get_fast_config() -> Config:
    """Быстрая конфигурация для тестирования."""
    return load_config("fast")


def get_precision_config() -> Config:
    """Конфигурация высокой точности."""
    return load_config("precision")


def get_robust_config() -> Config:
    """Конфигурация устойчивого обучения."""
    return load_config("robust")


def get_default_config() -> Config:
    """Стандартная конфигурация."""
    return load_config("default")


if __name__ == "__main__":
    # Пример использования
    create_config_template()
    print("\nAvailable configs:")
    for cfg in list_available_configs():
        print(f"  • {cfg}")