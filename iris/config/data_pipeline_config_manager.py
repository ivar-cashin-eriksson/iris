"""
Configuration manager for data pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from iris.config.config_manager import BaseConfig, ConfigManager


@dataclass(frozen=True)
class ScraperConfig:  # TODO: Should this be an abstract class?
    """Base scraper configuration."""

    user_agent: str
    rate_limit: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    wait_for_selector: str = "img"
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
    shop_name: str | None = None  # Optional, set from filename if None


@dataclass(frozen=True, kw_only=True)  # kw_only=True due to inheritance of BaseConfig
class StorageConfig(BaseConfig):
    """Storage configuration."""

    base_path: str
    path_template: str = "{env}/{shop_name}"

    def get_storage_path(self, shop_config: ShopConfig) -> Path:
        """Get the storage path for a specific shop."""
        computed_path = self.path_template.format(
            env=self.environment,
            shop_name=shop_config.shop_name
        )
        return Path(self.base_path) / computed_path


@dataclass(frozen=True, kw_only=True)  # kw_only=True due to inheritance of BaseConfig
class MongoDBConfig(BaseConfig):
    """MongoDB configuration."""

    connection_string: str
    database_template: str = "iris_{env}_{shop_name}"
    image_metadata_collection: str = "image_metadata"
    products_collection: str = "products"
    scraping_progress_collection: str = "scraping_progress"
    tls_allow_invalid_certificates: bool = True

    def get_database_name(self, shop_config: ShopConfig) -> str:
        """Get the database name for a specific shop."""
        return self.database_template.format(
            env=self.environment,
            shop_name=shop_config.shop_name
        )


class DataPipelineConfigManager(ConfigManager):
    """
    Manages configuration loading and access from multiple TOML files.
    Supports base configurations and shop-specific overrides.

    This manager handles configurations for:
    - Base scraper settings (rate limits, timeouts, selectors)
    - Shop-specific configurations (URLs, selectors, overrides)
    - Storage settings (base paths, templates)
    - MongoDB connection settings

    Attributes:
        data_pipeline_config_dir (Path): Directory containing data pipeline configs.
        shops_config_dir (Path): Directory containing shop-specific configs.
        scraper_config (ScraperConfig): Base scraper configuration.
        shop_configs (Dict[str, ShopConfig]): Shop-specific configurations.
        storage_config (StorageConfig): Storage configuration.
        mongodb_config (MongoDBConfig): MongoDB configuration.
    """

    def _setup_paths(self) -> None:
        self.data_pipeline_config_dir = self.config_dir / "data_pipeline"
        self.shops_config_dir = self.data_pipeline_config_dir / "shops"

    def _load_all_configs(self) -> None:
        # Load Scraper Config
        scraper_data = self._load_toml(self.data_pipeline_config_dir / "scraper.toml")
        self.scraper_config: ScraperConfig = self._create_scraper_config(scraper_data)

        # Load shop configurations
        self.shop_configs: Dict[str, ShopConfig] = {}
        for shop_config_path in self.shops_config_dir.glob("*.toml"):
            shop_name = shop_config_path.stem
            shop_config_data = self._load_toml(shop_config_path)
            self.shop_configs[shop_name] = self._create_shop_config(
                shop_config_data, 
                shop_name
            )

        # Load Storage Config
        storage_data = self._load_toml(self.data_pipeline_config_dir / "storage.toml")
        self.storage_config: StorageConfig = self._create_storage_config(storage_data)

        # Load MongoDB Config
        mongodb_data = self._load_toml(self.data_pipeline_config_dir / "mongodb.toml")
        self.mongodb_config: MongoDBConfig = self._create_mongodb_config(mongodb_data)

    def _create_scraper_config(
        self, data: dict, override_data: dict | None = None
    ) -> ScraperConfig:
        # Merge the default config data with any overrides
        if override_data:
            data.update(override_data)

        # Make ScraperConfig
        return ScraperConfig(**data)

    def _create_shop_config(self, data: dict, shop_name: str) -> ShopConfig:
        # Use configured shop_name if provided, otherwise use filename
        if "shop_name" not in data:
            data["shop_name"] = shop_name

        # Merge the default scraper config with any shop-specific overrides
        scraper_overrides = data.pop("scraper_overrides", {})
        scraper_config = self._create_scraper_config(
            self.scraper_config.__dict__.copy(), scraper_overrides
        )

        # Make ShopConfig
        return ShopConfig(**data, scraper_config=scraper_config)
    
    def _create_storage_config(self, data: dict) -> StorageConfig:
        return StorageConfig(**data, environment=self.base_config.environment)

    def _create_mongodb_config(self, data: dict) -> MongoDBConfig:
        return MongoDBConfig(**data, environment=self.base_config.environment)
