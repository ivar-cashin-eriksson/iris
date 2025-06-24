"""
Configuration manager for embedding pipeline.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

from iris.config.config_manager import BaseConfig, ConfigManager
from iris.config.data_pipeline_config_manager import ShopConfig

@dataclass(frozen=True, kw_only=True)
class ClipConfig(BaseConfig):
    """Configuration for CLIP model."""
    model_name: str
    pretrained: str


class EmbeddingPipelineConfigManager(ConfigManager):
    """Configuration manager for the embedding pipeline component.
    
    Handles loading and managing configurations for CLIP model and embedding database.
    
    Attributes:
        embedding_config_dir (Path): Directory containing embedding pipeline configs
        clip_config (ClipConfig): Configuration for CLIP model
        database_config (EmbeddingDBConfig): Configuration for embedding database
    """

    def _load_all_configs(self) -> None:     
        # Load Clip Config   
        clip_data = self._load_toml(self.base_config.clip_config_path)
        self.clip_config = self._create_clip_config(clip_data)
        

    def _create_clip_config(self, data: dict) -> ClipConfig:
        return ClipConfig(**asdict(self.base_config), **data)
    