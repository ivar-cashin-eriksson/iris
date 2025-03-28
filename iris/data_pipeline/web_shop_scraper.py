import re
import time
from typing import Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from iris.config.config_manager import ConfigManager, MongoDBConfig, ShopConfig
from iris.data_pipeline.base_scraper import BaseScraper
from iris.data_pipeline.mongodb_manager import MongoDBManager
from iris.data_pipeline.product_handler import ProductHandler


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
        mongodb_manager: MongoDBManager,
        product_handler: ProductHandler,
        mongodb_config: MongoDBConfig,
        shop_config: ShopConfig,
    ) -> None:
        """
        Initialize the WebShopScraper.

        Args:
            product_handler (ProductHandler): Handler for processing individual products
            mongodb_manager (MongoDBManager): MongoDB manager for storing progress
            mongodb_config (MongoDBConfig): MongoDB configuration
            shop_config (ShopConfig): Shop configuration
        """
        self.product_handler = product_handler
        self.mongodb_manager = mongodb_manager
        self.mongodb_config = mongodb_config
        self.shop_config = shop_config

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

    def _extract_links(self, soup: BeautifulSoup, selector: str) -> Set[str]:
        """Extract and normalize links from a page using a selector."""
        links = set()
        for element in soup.select(selector):
            href = element.get("href")
            if href:
                links.add(self._normalize_url(href))
        return links

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

        # TODO: Split long lines
        # Convert CSS selectors to regex patterns
        # For example: a[href*='/products/'] -> /products/
        product_pattern = self.shop_config.scraper_config.product_link.split("*=")[
            1
        ].strip("[]'\"")
        category_pattern = self.shop_config.scraper_config.category_link.split("*=")[
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

    def _process_page(self, url: str) -> tuple[Set[str], Set[str], Set[str]]:
        """
        Process a single page and extract product, category, and pagination links.

        Args:
            url (str): URL of the page to process

        Returns:
            tuple[Set[str], Set[str], Set[str]]: Sets of product, category, and pagination URLs
        """
        # Normalize the input URL
        normalized_url = self._normalize_url(url)

        # Skip if we've already processed this normalized URL
        if normalized_url in self.processed_urls:
            return set(), set(), set()

        soup = self.scraper.load_page(url)
        if not soup:
            return set(), set(), set()

        # Extract all types of links
        product_links = {
            url
            for url in self._extract_links(
                soup, self.shop_config.scraper_config.product_link
            )
            if self._is_valid_product_url(url)
        }
        category_links = self._extract_links(
            soup, self.shop_config.scraper_config.category_link
        )
        pagination_links = self._extract_links(
            soup, self.shop_config.scraper_config.pagination_link
        )

        # Debug logging
        print(f"\nProcessing page: {url}")
        print(f"Found {len(product_links)} product links")
        print(f"Found {len(category_links)} category links")
        print(f"Found {len(pagination_links)} pagination links")

        return product_links, category_links, pagination_links

    def scrape(self) -> None:
        """
        Main method to scrape the entire web shop.
        Handles pagination, categories, and product processing.
        """
        urls_to_process = {self.shop_config.base_url}
        processed_categories = set()

        while urls_to_process:
            url = urls_to_process.pop()

            # Skip if already processed
            if url in self.processed_urls:
                continue

            print(f"Processing: {url}")

            # Process the page and get all links
            product_links, category_links, pagination_links = self._process_page(url)

            # Add new category URLs to process
            new_categories = category_links - processed_categories
            urls_to_process.update(new_categories)
            processed_categories.update(new_categories)

            # Add pagination URLs to process
            urls_to_process.update(pagination_links)

            # Process product URLs
            for product_url in product_links:
                if product_url not in self.processed_urls:
                    print(f"Processing product: {product_url}")
                    self.product_handler.process_product(product_url)

                    # Add product page to URLs to process to find more links
                    urls_to_process.add(product_url)

            # Mark this URL as processed
            self.processed_urls.add(url)
            self._save_progress(url)

            # Rate limiting
            time.sleep(self.shop_config.scraper_config.rate_limit)

        print("Finished scraping web shop!")
