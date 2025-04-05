from typing import Dict

from iris.config.data_pipeline_config_manager import ShopConfig, StorageConfig
from iris.data_pipeline.base_scraper import BaseScraper
from iris.data_pipeline.image_handler import ImageHandler
from iris.data_pipeline.mongodb_manager import MongoDBManager
from iris.data_pipeline.utils import _get_url_hash


class ProductHandler:
    """
    A class for handling product-specific scraping operations.

    Features:
    - Product page loading and parsing
    - Product metadata extraction
    - Product data storage in MongoDB
    - Product-image relationship management
    """

    def __init__(
        self,
        shop_config: ShopConfig,
        storage_config: StorageConfig,
        mongodb_manager: MongoDBManager,
    ) -> None:
        """
        Initialize the ProductHandler.

        Args:
            shop_config (ShopConfig): Shop config instance.
            mongodb_manager (MongoDBManager): MongoDB manager instance.
        """
        self.shop_config = shop_config
        self.storage_config = storage_config
        self.mongodb_manager = mongodb_manager

        # Initialize the base scraper and image handler
        self.scraper = BaseScraper(self.shop_config.scraper_config)
        self.image_handler = ImageHandler(
            self.shop_config, 
            self.storage_config, 
            self.mongodb_manager
        )

    def __del__(self):
        """
        Cleanup: Close the base scraper
        """
        if hasattr(self, "scraper"):
            del self.scraper

    def process_product(self, url: str) -> Dict[str, str] | None:
        """
        Process a product page: load, extract data, and store in MongoDB.

        Args:
            url (str): URL of the product page to process.

        Returns:
            Dict[str, str] | None: Extracted product data if successful, None otherwise.
        """
        # Generate a hash for the product URL
        product_hash = _get_url_hash(url)

        # Check if we already have this product in MongoDB
        query = {"product_hash": product_hash}
        existing_product = self.mongodb_manager.find_one("products", query)
        if existing_product:
            print(f"Product already exists in database: {url}")
            return existing_product

        # Load the page
        soup = self.scraper.load_page(url)
        if not soup:
            print(f"Failed to load product page: {url}")
            return None

        # Extract product data
        product_data = self.scraper.extract_data(
            soup, self.shop_config.metadata_selectors
        )
        if not product_data:
            print(f"No product data found on page: {url}")
            return None

        # Process images and get their hashes
        image_hashes = []
        for image_selector in self.shop_config.image_selectors.values():
            image_hashes.extend(
                self.image_handler.process_images(
                    soup, image_selector=image_selector, product_hash=product_hash
                )
            )

        # Add URL, hash, image references, and timestamp
        product_data["url"] = url
        product_data["product_hash"] = product_hash
        product_data["image_hashes"] = image_hashes  # Store references to images
        product_data = self.mongodb_manager.add_timestamp(product_data)

        # Store in MongoDB
        try:
            # Use product_hash as unique identifier to prevent duplicates
            self.mongodb_manager.update_one(
                "products", {"product_hash": product_hash}, product_data
            )
            print(f"Successfully processed product: {url}")
            return product_data
        except Exception as e:
            print(f"Error saving product data to MongoDB: {e}")
            return None

    def process_products(self, urls: list[str]) -> None:
        """
        Process multiple product pages.

        Args:
            urls (list[str]): List of product page URLs to process.
        """
        for url in urls:
            self.process_product(url)
