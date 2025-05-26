
import tomllib
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import final
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class BaseConfig:
    """Base configuration with environment settings."""
    
    config_dir: Path
    _config_paths: dict[str, Path]
    shop_config_path: Path
    mongodb_config_path: Path
    qdrant_config_path: Path
    scraper_config_path: Path
    storage_config_path: Path
    clip_config_path: Path
    embeddingdb_config_path: Path
    localization_config_path: Path

    base_path: Path = None  # Will be set in __post_init__ from environment variable
    environment: str = "dev"  # Default to development environment
    device: str = "cpu"
    
    def __post_init__(self):
        """Post-initialization to ensure base_path is set."""
        object.__setattr__(self, "base_path", Path(os.getenv('BASE_PATH')))
        object.__setattr__(self, "config_dir", self.base_path / Path(self.config_dir))
        
        # Convert config_paths values to Path objects and prepend base_path
        for config_name, path in self._config_paths.items():
            config_path = self.base_path / self.config_dir / path
            object.__setattr__(self, config_name, config_path)


class ConfigManager(ABC):
    """
    Abstract base class for configuration managers. 
    
    This class provides an abstract base class for configuration 
    managers. These allow secure loading and easy access of multiple 
    `TOML` files and different configuration `@dataclasses`.

    Attributes:
        base_config (BaseConfig): Base configuration containing environment settings.

    Subclasses must implement the following methods:
    - `_load_all_configs()`: Defines loading logic for speciality 
                             configs.

    Subclasses do not implement the following methods:
    - `__init__()`: Creates config path, loads in base config, and calls 
                    abstract methods `_setup_paths()` and 
                    `_load_all_configs()`.
    - `_load_toml()`: Loads `TOML` file from file path.
    """

    def __init__(self) -> None:
        """
        Initialize the configuration manager.
        """

        self.base_config_path = Path(os.getenv('BASE_PATH')) \
                              / os.getenv('BASE_CONFIG_PATH')
        
        # Load Base Config
        base_data = self._load_toml(self.base_config_path)
        self.base_config = BaseConfig(**base_data, **base_data['_config_paths'])

        # Call the abstract method that subclasses must implement
        self._load_all_configs()

    @abstractmethod
    def _load_all_configs(self) -> None:
        """Load all configuration objects specific to the subclass.
        
        This method should load all TOML files and create corresponding
        configuration objects for the specific implementation.
        """
        pass

    @final
    def _load_toml(self, file_path: Path) -> dict:
        """Load a TOML file and return its contents."""
        if not file_path.exists():
            return {}

        with open(file_path, "rb") as f:
            return tomllib.load(f)
        