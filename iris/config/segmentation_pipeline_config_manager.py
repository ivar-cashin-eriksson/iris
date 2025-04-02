"""
Configuration manager for segmentation pipeline.
"""

from typing import Union
from dataclasses import dataclass
from pathlib import Path

from iris.config.config_manager import BaseConfig, ConfigManager


@dataclass(frozen=True, kw_only=True)  # kw_only=True due to inheritance of BaseConfig
class SAM2Config(BaseConfig):
    """Configuration for SAM2 mask generation."""
    
    _checkpoint_path: str
    apply_postprocessing: bool = False
    model_type: str = "default"
    mask_generator_params: dict[str, Union[int, float, bool]]

    def get_config_path(self) -> Path:
        """Get the storage path for the model config."""
        return Path(self.base_path) / self._config_path
    
    def get_checkpoint_path(self) -> Path:
        """Get the storage path for the model checkpoint."""
        return Path(self.base_path) / self._checkpoint_path
    

class SegmentationPipelineConfigManager(ConfigManager):
    """
    Manages configuration loading and access for the segmentation pipeline.

    This manager handles configurations for:
    - SAM2 model settings (checkpoint, config)

    Attributes:
        segmentation_config_dir (Path): Directory containing segmentation configs.
        sam2_config (SAM2Config): Main segmentation configuration.
    """

    def _setup_paths(self) -> None:
        self.segmentation_config_dir = self.config_dir / "segmentation_pipeline"

    def _load_all_configs(self) -> None:
        # Load main segmentation config
        sam2_data = self._load_toml(self.segmentation_config_dir / "sam2.toml")
        self.sam2_config = self._create_segmentation_config(sam2_data)

    def _create_segmentation_config(self, data: dict) -> SAM2Config:
        return SAM2Config(**data, base_path=self.base_config.base_path)
