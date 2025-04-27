"""
Configuration manager for object localization pipeline.
"""

from typing import Union
from dataclasses import asdict, dataclass
from abc import ABC
from pathlib import Path

from iris.config.config_manager import BaseConfig, ConfigManager


@dataclass(frozen=True)
class ObjectLocalizationModelConfig(ABC, BaseConfig):
    """Base configuration for object_localization models."""

    _checkpoint_path: str
    max_image_size = None  # Maximum size for the longest dimension of images, None for no limit
    model_params: dict[str, Union[int, float, bool]]

    def get_checkpoint_path(self) -> Path:
        """Get the storage path for the model checkpoint."""
        return Path(self.base_path) / self._checkpoint_path

@dataclass(frozen=True, kw_only=True)
class YOLOSConfig(ObjectLocalizationModelConfig):
    """Configuration for YOLOS model."""

@dataclass(frozen=True, kw_only=True)
class SAM2Config(ObjectLocalizationModelConfig):
    """Configuration for SAM2 mask generation."""
    
    model_type: str = "default"


class ObjectLocalizationPipelineConfigManager(ConfigManager):
    """
    Manages configuration loading and access for the object localization 
    pipeline.

    This manager handles configurations for:
    - YOLOS model settings (checkpoint, config)
    - SAM2 model settings (checkpoint, config)

    Attributes:
        config_dir (Path): Directory containing object localization 
                                        configs.
        model_config (ObjectLocalizationModelConfig): Main object localization 
                                                      configuration.
    """

    def _setup_paths(self) -> None:
        self.config_dir = self.config_dir / "object_localization_pipeline"

    def _load_all_configs(self) -> None:
        # Load main object localization config
        self._ol_base_config = self._load_toml(
            self.config_dir / "object_localization_config.toml"
        )

        # Load model config
        model_data = self._load_toml(
            self.config_dir / self._ol_base_config["model_config"]
        )
        self.model_config = self._create_model_config(model_data)

    def _create_model_config(self, data: dict) -> ObjectLocalizationModelConfig:
        if self._ol_base_config['model_type'] == 'yolos':
            return YOLOSConfig(**asdict(self.base_config), **data)
        elif self._ol_base_config['model_type'] == 'sam2':
            return SAM2Config(**asdict(self.base_config), **data)
        else:
            raise ValueError(
                f"Unsupported model type: {self._ol_base_config['model_type']}"
            )
