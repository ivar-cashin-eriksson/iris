from bs4 import BeautifulSoup

from iris.config.data_pipeline_config_manager import ShopConfig
from iris.data_pipeline.base_scraper import BaseScraper
from iris.data_pipeline.image_handler import ImageHandler
from iris.models.product import Product
from iris.models.image import Image
from iris.utils.log import logger


class ProductHandler:
    """
    A class for scraping product data.

    Features:
    - Product page loading and parsing
    - Product metadata extraction
    """

    def __init__(
        self,
        shop_config: ShopConfig,
    ) -> None:
        """
        Initialize the ProductHandler.

        Args:
            shop_config (ShopConfig): Shop config instance.
        """
        self.shop_config = shop_config

        # Initialize the base scraper and image handler
        self.scraper = BaseScraper(self.shop_config.scraper_config)

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
            tuple[Product, list[Image]] | None: Product object if extraction is
                                                successful; None otherwise.
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
                ImageHandler.extract_images(
                    soup,
                    image_selector=image_selector
                )
            )

        # Make product instance
        product = Product(
            title=product_data["title"],
            description=product_data["description"],
            url=url,
            image_hashes=[image.hash for image in images],
            debug_info=product_data.get("debug_info", dict())
        )

        return product, images
