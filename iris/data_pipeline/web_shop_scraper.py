import re
import time
from typing import Set
from urllib.parse import urljoin, urlparse
from collections.abc import Iterator


from bs4 import BeautifulSoup

from iris.config.data_pipeline_config_manager import ShopConfig, MongoDBConfig
from iris.data_pipeline.base_scraper import BaseScraper
from iris.data_pipeline.mongodb_manager import MongoDBManager
from iris.data_pipeline.product_handler import ProductHandler

from iris.models.product import Product
from iris.models.image import Image

from iris.utils.log import logger


class WebShopScraper:
    """
    A class for scraping entire web shops.

    Features:
    - Product URL discovery from shop pages
    - Pagination handling
    - Category navigation
    - Progress tracking and resumability
    - Rate limiting
    """

    def __init__(
        self,
        shop_config: ShopConfig,
        mongodb_config: MongoDBConfig,
        mongodb_manager: MongoDBManager,
        product_handler: ProductHandler,
    ) -> None:
        """
        Initialize the WebShopScraper.

        Args:
            shop_config (ShopConfig): Shop configuration
            mongodb_config (MongoDBConfig): MongoDB configuration
            product_handler (ProductHandler): Handler for processing individual products
            mongodb_manager (MongoDBManager): MongoDB manager for storing progress
        """
        self.shop_config = shop_config
        self.mongodb_config = mongodb_config
        self.product_handler = product_handler
        self.mongodb_manager = mongodb_manager

        # Initialize the base scraper
        self.scraper = BaseScraper(self.shop_config.scraper_config)

        # Set to track processed URLs
        self.processed_urls: Set[str] = set()

        # Load progress from MongoDB
        self._load_progress()

    def __del__(self):
        """Cleanup: Close the base scraper"""
        if hasattr(self, "scraper"):
            del self.scraper

    def _load_progress(self) -> None:
        """Load previously processed URLs from MongoDB."""
        processed_docs = self.mongodb_manager.find_all(
            self.mongodb_config.scraping_progress_collection,
            {"shop_url": self.shop_config.base_url},
        )
        self.processed_urls = {doc["url"] for doc in processed_docs}

    def _save_progress(self, url: str) -> None:
        """Save processed URL to MongoDB."""
        self.mongodb_manager.update_one(
            self.mongodb_config.scraping_progress_collection,
            {"url": url},
            {
                "url": url,
                "shop_url": self.shop_config.base_url,
                "processed_at": time.time(),
            },
        )

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL to handle relative paths and remove query parameters/fragments.

        Args:
            url (str): URL to normalize

        Returns:
            str: Normalized URL without query parameters or fragments
        """
        # Join with base URL if relative
        full_url = urljoin(self.shop_config.base_url, url)

        # Parse the URL
        parsed = urlparse(full_url)

        # Reconstruct URL without query parameters and fragments
        # Only keep scheme, netloc, and path
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def _is_valid_product_url(self, url: str) -> bool:
        """
        Check if URL is a valid product URL based on configured selectors.

        Args:
            url (str): URL to validate

        Returns:
            bool: True if URL matches product pattern and doesn't match collection or page patterns
        """
        # Normalize the URL first
        normalized_url = self._normalize_url(url)

        # Convert CSS selectors to regex patterns
        # For example: a[href*='/products/'] -> /products/
        product_link = self.shop_config.scraper_config.selectors['product']
        product_pattern = product_link.split("*=")[
            1
        ].strip("[]'\"")
        category_link = self.shop_config.scraper_config.selectors['category']
        category_pattern = category_link.split("*=")[
            1
        ].strip("[]'\"")

        # Create regex patterns
        product_regex = re.compile(product_pattern)
        category_regex = re.compile(category_pattern)

        # Check if URL matches product pattern and doesn't match category pattern
        if not product_regex.search(normalized_url) or category_regex.search(
            normalized_url
        ):
            return False

        return True

    def _extract_links(self, soup: BeautifulSoup) -> set[str]:
        """
        Process a single page and extract product, category, and pagination links.

        Args:
            url (str): URL of the page to process.

        Returns:
            set[str]: Set of extracted URLs.
        """
        links = set()
        
        for selector in self.shop_config.scraper_config.selectors.values():
            links.update({
            self._normalize_url(a["href"])
            for a in soup.select(selector)
            if a.has_attr("href")
        })

        return links

    def scrape(self) -> Iterator[tuple[Product, list[Image]]]:
        """
        Crawl and process an entire web shop, yielding structured product data.

        This method handles recursive link discovery, pagination, and product extraction 
        with URL deduplication and rate limiting.

        For each valid product page encountered, it yields a tuple containing:
            - A Product instance
            - A list of associated Image instances

        Returns:
            Iterator[tuple[Product, list[Image]]]: Streamed product and image pairs
                                                   as they are discovered and parsed.
        """

        urls_to_process = {self.shop_config.base_url}
        while urls_to_process:
            url = urls_to_process.pop()
            url = self._normalize_url(url)

            # Skip if already processed
            if url in self.processed_urls:
                continue

            # Load the page
            logger.info(f"Scraping page: {url}")
            
            soup = self.scraper.load_page(url)
            if soup is None:
                logger.warning(f"Failed to load page {url}. Skipping.")
                continue

            # Process the page and get all links
            links = self._extract_links(soup)
            new_links = links - self.processed_urls - urls_to_process
            urls_to_process.update(new_links)
            logger.info(
                f"\tLinks: {len(links)}\n"
                f"\tNew links: {len(new_links)}\n"
                f"\tLinks to process: {len(urls_to_process)}"
            )

            # Process product URLs
            if self._is_valid_product_url(url):
                extracted = self.product_handler.process_product_page(url, soup)

                if extracted is not None:
                    extracted_product = extracted[0]
                    extracted_images = extracted[1]
                    yield extracted_product, extracted_images

            # Mark this URL as processed
            self.processed_urls.add(url)
            self._save_progress(url)

            # Rate limiting
            time.sleep(self.shop_config.scraper_config.rate_limit)

        logger.info(f"Scraping completed. Processed {len(self.processed_urls)} URLs.")