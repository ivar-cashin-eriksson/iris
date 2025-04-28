"""
Configuration manager for object localization pipeline.
"""

from typing import Union
from dataclasses import asdict, dataclass
from abc import ABC
from pathlib import Path

from iris.config.config_manager import BaseConfig, ConfigManager


@dataclass(frozen=True, kw_only=True)
class LocalizationModelConfig(ABC, BaseConfig):
    """Base configuration for localization models."""

    _checkpoint_path: str
    model_params: dict[str, Union[int, float, bool]]
    max_image_size = None  # Maximum size for the longest dimension of images.
                           # None indicates no limit.
                           # Images are scaled to this size, and then
                           # localizations are rescaled again.

    def get_checkpoint_path(self) -> Path:
        """Get the storage path for the model checkpoint."""
        return Path(self.base_path) / self._checkpoint_path

@dataclass(frozen=True, kw_only=True)
class YolosConfig(LocalizationModelConfig):
    """Configuration for YOLOS model."""

    use_fast: bool = True
    confidence_threshold: float = 0.5

@dataclass(frozen=True, kw_only=True)
class SAM2Config(LocalizationModelConfig):
    """Configuration for SAM2 mask generation."""
    
    model_type: str = "default"
    mask_format: str = "binary"


class LocalizationPipelineConfigManager(ConfigManager):
    """
    Manages configuration loading and access for the object localization 
    pipeline.

    This manager handles configurations for:
    - YOLOS model settings (checkpoint, config)
    - SAM2 model settings (checkpoint, config)

    Attributes:
        config_dir (Path): Directory containing object localization 
                                        configs.
        model_config (LocalizationModelConfig): Main object localization 
                                                      configuration.
    """

    def _setup_paths(self) -> None:
        self.config_dir = self.config_dir / "localization_pipeline"

    def _load_all_configs(self) -> None:
        # Load main object localization config
        self._localization_base_config = self._load_toml(
            self.config_dir / "localization_config.toml"
        )

        # Load model config
        model_data = self._load_toml(
            self.config_dir / self._localization_base_config["model_config"]
        )
        self.model_config = self._create_model_config(model_data)

    def _create_model_config(self, data: dict) -> LocalizationModelConfig:
        if self._localization_base_config['model_type'] == 'yolos':
            return YolosConfig(**asdict(self.base_config), **data)
        elif self._localization_base_config['model_type'] == 'sam2':
            return SAM2Config(**asdict(self.base_config), **data)
        else:
            raise ValueError(
                f"Unsupported model type: {self._localization_base_config['model_type']}"
            )
