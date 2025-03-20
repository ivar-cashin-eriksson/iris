from typing import Dict
from base_scraper import BaseScraper
from mongodb_manager import MongoDBManager
from image_handler import ImageHandler
from utils import get_url_hash


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
        filters: Dict[str, str],
        mongo_manager: MongoDBManager | None,
        image_selector: str = "img",
        wait_for_selector: str = "img"
    ) -> None:
        """
        Initialize the ProductHandler.

        Args:
            filters (Dict[str, str]): CSS selectors for extracting product data.
            mongo_manager (MongoDBManager): MongoDB manager instance.
                                            If None, creates a new connection.
            image_selector (str): CSS selector for finding image elements.
            wait_for_selector (str): CSS selector to wait for before considering page loaded.
        """
        self.filters = filters
        self.mongo_manager = mongo_manager
        self.image_selector = image_selector
        self.wait_for_selector = wait_for_selector
            
        # Initialize the base scraper and image handler
        self.scraper = BaseScraper()
        self.image_handler = ImageHandler(mongo_manager)

    def __del__(self):
        """
        Cleanup: Close the base scraper
        """
        if hasattr(self, 'scraper'):
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
        product_hash = get_url_hash(url)
        
        # Check if we already have this product in MongoDB
        query = {'product_hash': product_hash}
        existing_product = self.mongo_manager.find_one('products', query)
        if existing_product:
            print(f"Product already exists in database: {url}")
            return existing_product

        # Load the page
        soup = self.scraper.load_page(url, wait_for_selector=self.wait_for_selector)
        if not soup:
            print(f"Failed to load product page: {url}")
            return None

        # Extract product data
        product_data = self.scraper.extract_data(soup, self.filters)
        if not product_data:
            print(f"No product data found on page: {url}")
            return None

        # Process images and get their hashes
        image_hashes = self.image_handler.process_images(
            soup,
            image_selector=self.image_selector,
            product_hash=product_hash
        )

        # Add URL, hash, image references, and timestamp
        product_data['url'] = url
        product_data['product_hash'] = product_hash
        product_data['image_hashes'] = image_hashes  # Store references to images
        product_data = self.mongo_manager.add_timestamp(product_data)

        # Store in MongoDB
        try:
            # Use product_hash as unique identifier to prevent duplicates
            self.mongo_manager.update_one(
                'products',
                {'product_hash': product_hash},
                product_data
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