"""
Configuration manager for data pipeline.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import os

from iris.config.config_manager import BaseConfig, ConfigManager
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ScraperConfig:  # TODO: Should this be an abstract class?
    """Base scraper configuration."""

    user_agent: str
    rate_limit: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    wait_for_selector: str = "img"
    patterns: dict[str, str] = field(default_factory=lambda: {
        "product": "/products/.+",
        "collection": "/collections/.+",
        "pagination": "/page/\\d+"
    })


@dataclass(frozen=True)
class ShopConfig:
    """Shop-specific configuration."""

    shop_name: str
    base_url: str
    image_selectors: dict[str, str]
    metadata_selectors: dict[str, str]
    scraper_config: ScraperConfig
    start_url: str | None = None

    def __post_init__(self) -> None:
        if self.start_url is None:
            # Default to the base URL if no start URL is provided
            object.__setattr__(self, "start_url", self.base_url)

@dataclass(frozen=True, kw_only=True)  # kw_only=True due to inheritance of BaseConfig
class ImageStoreConfig(BaseConfig):
    """Storage configuration."""

    storage_backend: str = "local"
    storage_path: Path = None
    _path_template: str = "data/{env}/{shop_name}/images/"
    cache_enabled: bool = True
    timeout: int = 10
    _shop_name: str

    def __post_init__(self) -> None:
        computed_path = self._path_template.format(
            env=self.environment,
            shop_name=self._shop_name
        )

        if self.storage_backend == 'local':
            object.__setattr__(self, "storage_path", self.base_path / computed_path)
        else:
            object.__setattr__(self, "storage_path", Path(computed_path))


@dataclass(frozen=True, kw_only=True)  # kw_only=True due to inheritance of BaseConfig
class MongoDBConfig(BaseConfig):
    """MongoDB configuration."""

    connection_string: str = None
    database_name: str = None
    tls_allow_invalid_certificates: bool = True
    _database_template: str = "iris_{env}_{shop_name}"
    product_collection: str = "products"
    image_metadata_collection: str = "image_metadata"
    localization_collection: str = "localizations"
    scraping_progress_collection: str = "scraping_progress"
    _shop_name: str

    def __post_init__(self) -> None:
        database_name = self._database_template.format(
            env=self.environment,
            shop_name=self._shop_name
        )
        object.__setattr__(self, "database_name", database_name)

        connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        object.__setattr__(self, "connection_string", connection_string)


@dataclass(frozen=True, kw_only=True)  # kw_only=True due to inheritance of BaseConfig
class QdrantConfig(BaseConfig):
    """Qdrant configuration."""

    api_key: str = None
    url: str
    embedding_dim: int = 512
    _collection_template: str = "iris_{env}_{shop_name}_{collection_name}"
    product_collection: str = 'products'
    image_collection: str = 'images'
    localization_collection: str = 'localizations'
    _shop_name: str


    def __post_init__(self) -> None:
        collections = [
            'product_collection',
            'image_collection', 
            'localization_collection'
        ]
        for collection_attr in collections:
            base_name = getattr(self, collection_attr)
            full_collection = self._collection_template.format(
                env=self.environment,
                shop_name=self._shop_name,
                collection_name=base_name
            )
            object.__setattr__(self, collection_attr, full_collection)


        api_key = os.getenv('QDRANT_API_KEY')
        object.__setattr__(self, "api_key", api_key)

        


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
        scraper_config (ScraperConfig): Base scraper configuration.
        shop_config (ShopConfig): Shop-specific configuration.
        storage_config (StorageConfig): Storage configuration.
        mongodb_config (MongoDBConfig): MongoDB configuration.
    """

    def _load_all_configs(self) -> None:
        # Load Scraper Config
        scraper_data = self._load_toml(self.base_config.scraper_config_path)
        self.scraper_config: ScraperConfig = self._create_scraper_config(scraper_data)

        # Load shop configurations
        shop_config_data = self._load_toml(self.base_config.shop_config_path)
        self.shop_config: ShopConfig = self._create_shop_config(shop_config_data)

        # Load Image Store Config
        image_store_data = self._load_toml(self.base_config.image_store_config_path)
        self.image_store_config: ImageStoreConfig = self._create_image_store_config(
            image_store_data,
            self.shop_config.shop_name
        )

        # Load MongoDB Config
        mongodb_data = self._load_toml(self.base_config.mongodb_config_path)
        self.mongodb_config: MongoDBConfig = self._create_mongodb_config(
            mongodb_data,
            self.shop_config.shop_name
        )

        # Load Qdrant Config
        qdrant_data = self._load_toml(self.base_config.qdrant_config_path)
        self.qdrant_config: QdrantConfig = self._create_qdrant_config(
            qdrant_data,
            self.shop_config.shop_name
        )

    def _create_scraper_config(
        self, data: dict, 
        override_data: dict | None = None
    ) -> ScraperConfig:
        # Merge the default config data with any overrides
        if override_data:
            data.update(override_data)

        # Make ScraperConfig
        return ScraperConfig(**data)

    def _create_shop_config(self, data: dict) -> ShopConfig:
        # Merge the default scraper config with any shop-specific overrides
        scraper_overrides = data.pop("scraper_overrides", {})
        scraper_config = self._create_scraper_config(
            asdict(self.scraper_config), 
            scraper_overrides
        )

        # Make ShopConfig
        return ShopConfig(**data, scraper_config=scraper_config)
    
    def _create_image_store_config(self, data: dict, shop_name: str) -> ImageStoreConfig:
        return ImageStoreConfig(**asdict(self.base_config), **data, _shop_name=shop_name)

    def _create_mongodb_config(self, data: dict, shop_name: str) -> MongoDBConfig:
        return MongoDBConfig(**asdict(self.base_config), **data, _shop_name=shop_name)

    def _create_qdrant_config(self, data: dict, shop_name: str) -> QdrantConfig:
        return QdrantConfig(**asdict(self.base_config), **data, _shop_name=shop_name)
