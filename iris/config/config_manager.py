"""
Configuration manager for loading and managing TOML configurations.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import os
import tomllib


@dataclass(frozen=True)
class MongoDBConfig:
    """MongoDB configuration."""
    connection_string: str
    database: str
    image_metadata_collection: str = "image_metadata"
    products_collection: str = "products"
    scraping_progress_collection: str = "scraping_progress"


@dataclass(frozen=True)
class ScraperConfig:
    """Base scraper configuration."""
    user_agent: str
    rate_limit: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    pagination_link: str = "a[rel='next']"
    product_link: str = "a[href*='/products/']"
    category_link: str = "a[href*='/collections/']"


@dataclass(frozen=True)
class ShopConfig:
    """Shop-specific configuration."""
    base_url: str
    image_selectors: dict[str, str]
    metadata_selectors: dict[str, str]
    scraper_config: ScraperConfig


class ConfigManager:
    """
    Manages configuration loading and access from multiple TOML files.
    Supports base configurations and shop-specific overrides.
    """
    
    def __init__(self, config_dir: str | None = None) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Path to the configuration directory. If None, uses default path.
        """
        # Set up configuration paths
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "configs"
        self.base_config_dir = self.config_dir / "base"
        self.shops_config_dir = self.config_dir / "shops"
        
        # Create config directories if they don't exist
        self.base_config_dir.mkdir(parents=True, exist_ok=True)
        self.shops_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration storage
        self.mongodb_config: MongoDBConfig
        self.scraper_config: ScraperConfig
        self.shop_configs: Dict[str, ShopConfig] = {}
        
        # Load configurations
        self._load_all_configs()
    
    def _load_toml(self, file_path: Path) -> dict:
        """Load a TOML file and return its contents."""
        if not file_path.exists():
            return {}
        
        with open(file_path, 'rb') as f:
            return tomllib.load(f)
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        # Load MongoDB Config
        mongodb_data = self._load_toml(self.base_config_dir / "mongodb.toml")
        self.mongodb_config = self._create_mongodb_config(mongodb_data)

        # Load Scraper Config
        scraper_data = self._load_toml(self.base_config_dir / "scraper.toml")
        self.scraper_config = self._create_scraper_config(scraper_data)

        # Load shop configurations
        for shop_config_path in self.shops_config_dir.glob("*.toml"):
            shop_name = shop_config_path.stem
            shop_config_data = self._load_toml(shop_config_path)
            self.shop_configs[shop_name] = self._create_shop_config(shop_config_data)
    
    def _create_mongodb_config(self, data: dict) -> MongoDBConfig:
        return MongoDBConfig(**data)
    
    def _create_scraper_config(self, data: dict, override_data: dict | None = None) -> ScraperConfig:
        # Merge the default config data with any overrides
        if override_data:
            data.update(override_data)

        # Make ScraperConfig
        return ScraperConfig(**data)
    
    def _create_shop_config(self, data: dict) -> ShopConfig:     
        # Merge the default scraper config with any shop-specific overrides
        scraper_overrides = data.pop('scraper_overrides', {})
        scraper_config = self._create_scraper_config(
            self.scraper_config.__dict__.copy(), 
            scraper_overrides
        )

        # Make ShopConfig
        return ShopConfig(**data, scraper_config=scraper_config)      
