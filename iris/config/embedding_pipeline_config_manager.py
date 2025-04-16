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
    embedding_dim: int
    index_type: str
    _path_template: str = "data/{env}/{shop_name}/embeddings"

    def get_storage_path(self, shop_config: ShopConfig) -> Path:
        """Get the database directory for a specific shop."""
        computed_path = self._path_template.format(
            env=self.environment,
            shop_name=shop_config.shop_name
        )
        return Path(self.base_path) / computed_path


class EmbeddingPipelineConfigManager(ConfigManager):
    """Configuration manager for the embedding pipeline component.
    
    Handles loading and managing configurations for CLIP model and embedding database.
    
    Attributes:
        embedding_config_dir (Path): Directory containing embedding pipeline configs
        clip_config (ClipConfig): Configuration for CLIP model
        database_config (EmbeddingDBConfig): Configuration for embedding database
    """

    def _setup_paths(self) -> None:
        self.embedding_config_dir = self.config_dir / "embedding_pipeline"

    def _load_all_configs(self) -> None:     
        # Load Clip Config   
        clip_data = self._load_toml(self.embedding_config_dir / "clip.toml")
        self.clip_config = self._create_clip_config(clip_data)
        
        # Load EmbeddingDB Config
        embeddingdb_data = self._load_toml(self.embedding_config_dir / "database.toml")
        self.database_config = self._create_embeddingdb_config(embeddingdb_data)

    def _create_clip_config(self, data: dict) -> ClipConfig:
        return ClipConfig(**asdict(self.base_config), **data)
    
    def _create_embeddingdb_config(self, data: dict) -> EmbeddingDBConfig:
        """Create EmbeddingDBConfig with base config."""
        return EmbeddingDBConfig(**asdict(self.base_config), **data)
