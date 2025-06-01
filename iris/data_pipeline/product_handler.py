
from bs4 import BeautifulSoup
from typing import Dict

from iris.config.data_pipeline_config_manager import ShopConfig, StorageConfig
from iris.data_pipeline.base_scraper import BaseScraper
from iris.data_pipeline.image_handler import ImageHandler
from iris.data_pipeline.mongodb_manager import MongoDBManager
from iris.data_pipeline.utils import _get_url_hash
from iris.utils.log import logger

from iris.models.product import Product
from iris.models.image import Image


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

    def process_product_page(
            self, 
            url: str, 
            soup: BeautifulSoup
        ) -> tuple[Product, list[Image]] | None:
        """
        Parse and convert a product page into a structured Product document.

        This method extracts metadata and associated images from the given HTML page
        and constructs a Product instance using the configured scraping logic.

        Args:
            url (str): URL of the product page to process.
            soup (BeautifulSoup): Parsed HTML of the product page.

        Returns:
            tuple[Product, list[Image]] | None: Product object and list of Image objects
                                                if extraction is successful; None 
                                                otherwise.
        """
        # Extract product data
        product_data = self.scraper.extract_data(
            soup,
            self.shop_config.metadata_selectors
        )
        if not product_data:
            logger.warning("No product data found in the page.")
            return None

        # Extract product images
        images: list[Image] = []
        for image_selector in self.shop_config.image_selectors.values():
            images.extend(
                self.image_handler.extract_images(
                    soup,
                    image_selector=image_selector
                )
            )

        # Make product instance
        product = Product.from_raw(
            title=product_data["title"],
            description=product_data["description"],
            url=url,
            image_ids=[image.id for image in images],
            debug_info=product_data.get("debug_info", dict())
        )

        return product, images
