
import tomllib
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import final



@dataclass(frozen=True)
class BaseConfig:
    """Base configuration with environment settings."""

    base_path: str
    environment: str = "dev"  # Default to development environment
    device: str = "cpu"


class ConfigManager(ABC):
    """
    Abstract base class for configuration managers. 
    
    This class provides an abstract base class for configuration 
    managers. These allow secure loading and easy access of multiple 
    `TOML` files and different configuration `@dataclasses`.

    Attributes:
        config_dir (Path): Root directory containing all configuration files.
        base_config (BaseConfig): Base configuration containing environment settings.

    Subclasses must implement the following methods:
    - `_setup_paths()`: Defines paths to speciality configs.
    - `_load_all_configs()`: Defines loading logic for speciality 
                             configs.

    Subclasses do not implement the following methods:
    - `__init__()`: Creates config path, loads in base config, and calls 
                    abstract methods `_setup_paths()` and 
                    `_load_all_configs()`.
    - `_load_toml()`: Loads `TOML` file from file path.
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_dir: Path to the configuration directory. If None,
                        uses default directory `config/`.
        """

        # Set up configuration paths
        self.config_dir = (
            config_dir
            if config_dir
            else Path(__file__).parent.parent.parent / "configs"
        )
        
        # Load Base Config
        base_data = self._load_toml(self.config_dir / "base.toml")
        self.base_config = BaseConfig(**base_data)

        # Call the abstract method that subclasses must implement
        self._setup_paths()
        self._load_all_configs()

    @abstractmethod
    def _setup_paths(self) -> None:
        """Set up additional configuration paths needed by subclasses."""
        pass

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
        