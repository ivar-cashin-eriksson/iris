import re
import time
from typing import Set
from urllib.parse import urljoin, urlparse
from collections import deque
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
        product_handler: ProductHandler,
    ) -> None:
        """
        Initialize the WebShopScraper.

        Args:
            shop_config (ShopConfig): Shop configuration
            product_handler (ProductHandler): Handler for processing individual products
        """
        self.shop_config = shop_config
        self.product_handler = product_handler

        # Initialize the base scraper
        self.scraper = BaseScraper(self.shop_config.scraper_config)

        # Set to track processed URLs
        self.processed_urls: Set[str] = set()

    def __del__(self):
        """Cleanup: Close the base scraper"""
        if hasattr(self, "scraper"):
            del self.scraper

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

    def _is_product_url(self, url: str) -> bool:
        """
        Check if URL is a valid product URL based on configured selectors.

        Args:
            url (str): URL to validate

        Returns:
            bool: True if URL matches product pattern and doesn't match 
                  collection or page patterns
        """
        # Normalize the URL first
        normalized_url = self._normalize_url(url)

        pattern = self.shop_config.scraper_config.patterns["product"]
        return re.search(pattern, url) is not None

    def _extract_links(self, soup: BeautifulSoup) -> set[str]:
        """
        Process a single page and extract product, category, and pagination links.

        Args:
            soup (BeautifulSoup): Parsed HTML of the page.

        Returns:
            set[str]: Set of extracted URLs.
        """
        all_links = {
            self._normalize_url(a["href"])
            for a in soup.find_all("a", href=True)
        }

        # Filter: must contain base_url and match one of the patterns
        base_url = self.shop_config.base_url
        patterns = self.shop_config.scraper_config.patterns
        matched_links = {
            url for url in all_links
            if base_url in url and any(re.search(pattern, url) 
                                       for pattern in patterns.values())
        }

        return matched_links

    def scrape(self) -> Iterator[tuple[Product, list[Image]]]:
        """
        Crawl and process an entire web shop, yielding structured product data.

        This method handles recursive link discovery, pagination, and product extraction 
        with URL deduplication and rate limiting.

        For each valid product page encountered, it yields a Product instance.

        Returns:
            Iterator[tuple[Product, list[Image]]]: Streamed products as they are discovered and parsed.
        """

        urls_to_process = deque([self.shop_config.start_url])
        seen_urls = {self.shop_config.start_url}

        while urls_to_process:
            url = urls_to_process.popleft()
            url = self._normalize_url(url)

            if url in self.processed_urls:
                continue

            logger.info(f"Scraping page: {url}")
            soup = self.scraper.load_page(url)
            if soup is None:
                logger.warning(f"Failed to load page {url}. Skipping.")
                continue

            links = self._extract_links(soup)
            new_links = links - self.processed_urls - seen_urls

            for link in new_links:
                urls_to_process.append(link)
                seen_urls.add(link)

            logger.info(
                f"\tLinks: {len(links)}\n"
                f"\tNew links: {len(new_links)}\n"
                f"\tLinks to process: {len(urls_to_process)}"
            )

            if self._is_product_url(url):
                product, images = self.product_handler.process_product_page(url, soup)
                if product is not None:
                    yield product, images

            self.processed_urls.add(url)
            time.sleep(self.shop_config.scraper_config.rate_limit)

        logger.info(f"Scraping completed. Processed {len(self.processed_urls)} URLs.")