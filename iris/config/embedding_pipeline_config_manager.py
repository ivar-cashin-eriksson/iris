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


@dataclass(frozen=True, kw_only=True)
class EmbeddingDBConfig(BaseConfig):
    """Configuration for embedding database."""
    
    storage_path: Path = None
    embedding_dim: int
    index_type: str
    _path_template: str = "data/{env}/{shop_name}/embeddings"
    _shop_name: str
    
    def __post_init__(self) -> None:
        storage_path = self._path_template.format(
            env=self.environment,
            shop_name=self._shop_name
        )
        object.__setattr__(self, "storage_path", self.base_path / storage_path)


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
        
        # TODO: Move to data_pipeline_config_manager.py
        # Load EmbeddingDB Config
        embeddingdb_data = self._load_toml(self.base_config.embeddingdb_config_path)
        self.database_config = self._create_embeddingdb_config(
            embeddingdb_data,
            shop_name="nikolaj_storm"  # TODO: Replace with dynamic shop name
        )

    def _create_clip_config(self, data: dict) -> ClipConfig:
        return ClipConfig(**asdict(self.base_config), **data)
    
    def _create_embeddingdb_config(self, data: dict, shop_name: str) -> EmbeddingDBConfig:
        """Create EmbeddingDBConfig with base config."""
        return EmbeddingDBConfig(
            **asdict(self.base_config), 
            **data, 
            _shop_name=shop_name
        )
