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

    checkpoint_path: Path
    model_params: dict[str, Union[int, float, bool]]
    max_image_size = None  # Maximum size for the longest dimension of images.
                           # None indicates no limit.
                           # Images are scaled to this size, and then
                           # localizations are rescaled again.
    
    def __post_init__(self) -> None:
        object.__setattr__(
            self, 
            "checkpoint_path", 
            self.base_path / self.checkpoint_path
        )

@dataclass(frozen=True, kw_only=True)
class YoloConfig(LocalizationModelConfig):
    """Configuration for YOLO model."""

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
        model_config (LocalizationModelConfig): Main object localization 
                                                      configuration.
    """

    def _load_all_configs(self) -> None:
        # Load main object localization config
        self._localization_base_config = self._load_toml(
            self.base_config.localization_config_path
        )

        # Load model config
        model_data = self._load_toml(
            self.base_config.config_dir / self._localization_base_config["model_config"]
        )
        self.model_config = self._create_model_config(model_data)

    def _create_model_config(self, data: dict) -> LocalizationModelConfig:
        match self._localization_base_config['model_type']:
            case 'yolo':
                return YoloConfig(**asdict(self.base_config), **data)
            case 'yolos':
                return YolosConfig(**asdict(self.base_config), **data)
            case 'sam2':
                return SAM2Config(**asdict(self.base_config), **data)
            case _:
                raise ValueError(
                    f"Unsupported model type: {self._localization_base_config['model_type']}"
                )
