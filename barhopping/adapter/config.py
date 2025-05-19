from dataclasses import dataclass
from typing import Optional
import yaml
import os
from pathlib import Path

@dataclass
class AdapterConfig:
    # Training parameters
    input_dim: int = 1024
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.0001
    warmup_steps: int = 100
    margin: float = 0.5
    device: str = 'cpu'
    
    num_bars: int = 100
    questions_per_bar: int = 10
    
    eval_k: int = 20
    model_save_path: str = 'adapter_model.pt'
    data_dir: str = 'data'
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AdapterConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        config_dict = {
            'input_dim': self.input_dim,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'margin': self.margin,
            'device': self.device,
            'num_bars': self.num_bars,
            'questions_per_bar': self.questions_per_bar,
            'eval_k': self.eval_k,
            'model_save_path': self.model_save_path,
            'data_dir': self.data_dir
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Default config
default_config = AdapterConfig()

def get_config(config_path: Optional[str] = None) -> AdapterConfig:
    """Get config from file or return default."""
    if config_path and os.path.exists(config_path):
        return AdapterConfig.from_yaml(config_path)
    return default_config 