import json
from pathlib import Path
from urllib.parse import urlparse, urljoin
import datetime
import hashlib
from typing import Set, Optional
import time
import re

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from pymongo import MongoClient
from webdriver_manager.chrome import ChromeDriverManager
from base_scraper import BaseScraper
from product_handler import ProductHandler
from mongodb_manager import MongoDBManager
from utils import get_url_hash


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
        base_url: str,
        product_handler: ProductHandler,
        mongo_manager: MongoDBManager,
        pagination_selector: str = "a[rel='next']",  # Common pagination selector
        product_link_selector: str = "a[href*='/products/']",  # Common product link selector
        category_selector: str = "a[href*='/collections/']",  # Common category selector
        rate_limit: float = 1.0  # Seconds between requests
    ) -> None:
        """
        Initialize the WebShopScraper.

        Args:
            base_url (str): The main URL of the web shop
            product_handler (ProductHandler): Handler for processing individual products
            mongo_manager (MongoDBManager): MongoDB manager for storing progress
            pagination_selector (str): CSS selector for pagination links
            product_link_selector (str): CSS selector for product links
            category_selector (str): CSS selector for category links
            rate_limit (float): Seconds to wait between requests
        """
        self.base_url = base_url
        self.product_handler = product_handler
        self.mongo_manager = mongo_manager
        self.pagination_selector = pagination_selector
        self.product_link_selector = product_link_selector
        self.category_selector = category_selector
        self.rate_limit = rate_limit
        
        # Initialize the base scraper
        self.scraper = BaseScraper()
        
        # Set to track processed URLs
        self.processed_urls: Set[str] = set()
        
        # Load progress from MongoDB
        self._load_progress()

    def __del__(self):
        """Cleanup: Close the base scraper"""
        if hasattr(self, 'scraper'):
            del self.scraper

    def _load_progress(self) -> None:
        """Load previously processed URLs from MongoDB."""
        processed_docs = self.mongo_manager.find_all('scraping_progress', {'shop_url': self.base_url})
        self.processed_urls = {doc['url'] for doc in processed_docs}

    def _save_progress(self, url: str) -> None:
        """Save processed URL to MongoDB."""
        self.mongo_manager.update_one(
            'scraping_progress',
            {'url': url},
            {
                'url': url,
                'shop_url': self.base_url,
                'processed_at': time.time()
            }
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
        full_url = urljoin(self.base_url, url)
        
        # Parse the URL
        parsed = urlparse(full_url)
        
        # Reconstruct URL without query parameters and fragments
        # Only keep scheme, netloc, and path
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def _extract_links(self, soup: BeautifulSoup, selector: str) -> Set[str]:
        """Extract and normalize links from a page using a selector."""
        links = set()
        for element in soup.select(selector):
            href = element.get('href')
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
        
        # Convert CSS selectors to regex patterns
        # a[href*='/products/'] -> /products/
        product_pattern = self.product_link_selector.split('*=')[1].strip('[]\'"')
        category_pattern = self.category_selector.split('*=')[1].strip('[]\'"')
        
        # Create regex patterns
        product_regex = re.compile(product_pattern)
        category_regex = re.compile(category_pattern)
        
        # Check if URL matches product pattern and doesn't match category pattern
        if not product_regex.search(normalized_url) or category_regex.search(normalized_url):
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
        product_links = {url for url in self._extract_links(soup, self.product_link_selector)
                        if self._is_valid_product_url(url)}
        category_links = self._extract_links(soup, self.category_selector)
        pagination_links = self._extract_links(soup, self.pagination_selector)

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
        urls_to_process = {self.base_url}
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
            time.sleep(self.rate_limit)

        print("Finished scraping web shop!")


def main():
    # MongoDB connection string
    connection_string = "mongodb+srv://test:test@iris-cluster.andes.mongodb.net/?retryWrites=true&w=majority&appName=iris-cluster"
    
    # Initialize MongoDB connection
    mongo_manager = MongoDBManager(connection_string)

    # CSS selectors for product data
    product_filters = {
        "title": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div.text-xl.font-medium.leading-5",
        "price": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div.hidden.md\:block > div > div > button > div > div",
        "color": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > button > div > div.flex.items-center.gap-1 > span",
        "num_colorways": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > button > div > div.text-\[9px\].uppercase.opacity-70 > span:nth-child(1)",
        "description": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div:nth-child(4) > div > div > p",
        "made_in": "#radix-\:r7d\: > div.flex.h-full.w-full.flex-col.border-t.px-5.pt-6.md\:px-8 > div.py-12.md\:py-10 > div.flex.w-full.items-center.justify-start.border-t.border-gray-light.py-6 > div > div.flex.w-56.text-xl.font-medium.md\:text-2xl > div",
        "product_details": "#radix-\:r7d\: > div.flex.h-full.w-full.flex-col.border-t.px-5.pt-6.md\:px-8 > div.py-12.md\:py-10 > div.flex.w-full.items-center.justify-start.border-y.border-gray-light.py-6 > div > div.flex.w-56.text-xs.leading-4"
    }

    # CSS selector for product images
    image_selector = "body > main > div > div > div:nth-child(1) > section.block-wrapper.page-offset-notification.relative.bg-\[\#f1f1f1\].lg\:h-screen.lg\:pt-0 > div.relative.transition-all.lg\:h-screen > div.relative.hidden.lg\:block.h-full"

    # Create product handler
    product_handler = ProductHandler(
        filters=product_filters,
        mongo_manager=mongo_manager,
        image_selector=image_selector
    )

    # Create web shop scraper
    shop_scraper = WebShopScraper(
        base_url="https://pasnormalstudios.com/dk",
        product_handler=product_handler,
        mongo_manager=mongo_manager,
        rate_limit=1.0  # Be nice to the server
    )

    # Start scraping
    shop_scraper.scrape()

    # Close MongoDB connection
    mongo_manager.close()


if __name__ == "__main__":
    main()
