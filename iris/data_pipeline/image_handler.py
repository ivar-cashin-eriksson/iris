from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from iris.config.config_manager import ShopConfig, StorageConfig
from iris.data_pipeline.mongodb_manager import MongoDBManager
from iris.data_pipeline.utils import get_url_hash


class ImageHandler:
    """
    A class for handling image-related operations including downloading,
    storage, and metadata management.

    Features:
    - Image URL extraction from HTML
    - Image downloading and local storage
    - Image metadata management in MongoDB
    - Product-image relationship management
    """

    def __init__(
        self, 
        shop_config: ShopConfig,
        storage_config: StorageConfig, 
        mongodb_manager: MongoDBManager
    ) -> None:
        """
        Initialize the image handler.

        Args:
            shop_config: Shop configuration
            storage_config: Storage configuration
            mongodb_manager: MongoDB manager for storing metadata

        Attributes:
            shop_config (ShopConfig): Shop configuration
            storage_config (StorageConfig): Storage configuration
            mongodb_manager (MongoDBManager): MongoDB manager instance
            images_dir (Path): The directory for downloaded images
        """
        self.shop_config = shop_config
        self.storage_config = storage_config
        self.mongodb_manager = mongodb_manager

        # Create the images directory within the shop's storage path
        self.images_dir = self.storage_config.get_storage_path(self.shop_config) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _get_element_path(self, element: BeautifulSoup) -> str:
        """
        Get the full path from root to the element, including classes and IDs.

        Args:
            element (BeautifulSoup): The element to get the path for.

        Returns:
            str: The full path from root to the element.
        """
        # Get all parents including the element itself
        path = [element] + list(element.find_parents())
        
        # Build identifiers for each element in the path
        identifiers = []
        for elem in reversed(path):  # Reverse to get root-to-element order
            if identifier := elem.name:
                if classes := elem.get("class"):
                    identifier += f".{' .'.join(classes)}"
                if elem_id := elem.get("id"):
                    identifier += f"#{elem_id}"
                identifiers.append(identifier)
        
        return " > ".join(identifiers)

    def _get_image_data(self, img_element: BeautifulSoup) -> tuple[str, str] | None:
        """
        Extract URL and location data from an image element.

        Args:
            img_element (BeautifulSoup): The image element to process.

        Returns:
            tuple[str, str] | None: Tuple of (url, location) if URL is found, None otherwise.
        """
        src = img_element.get("src")
        if not src:
            return None

        # Get the full path from root to the image
        location = self._get_element_path(img_element)

        return (src, location)

    def extract_image_urls(
        self, 
        soup: BeautifulSoup, 
        image_selector: str
    ) -> list[tuple[str, str]]:
        """
        Extracts image URLs and their locations from the given BeautifulSoup object.

        Args:
            soup (BeautifulSoup): Parsed HTML page.
            image_selector (str): CSS selector for finding image elements.

        Returns:
            List[tuple[str, str]]: A list of tuples containing (url, location) where location describes
                                 where the URL was found in the HTML structure.
        """
        image_data = []
        elements = soup.select(image_selector)

        for element in elements:
            if element.name == "img":
                # Directly an <img>, extract data
                if data := self._get_image_data(element):
                    image_data.append(data)
            else:
                # It's a container (e.g., <div>), find all <img> inside
                for img in element.find_all("img"):
                    if data := self._get_image_data(img):
                        image_data.append(data)

        return image_data

    def save_image_metadata(
        self, 
        img_url: str, 
        image_hash: str, 
        local_path: Path, 
        product_hash: str, 
        html_location: str
    ) -> None:
        """
        Saves image metadata to MongoDB.

        Args:
            img_url (str): Original URL of the image
            image_hash (str): Hash of the image URL
            local_path (Path): Local path where the image is stored
            product_hash (str): Hash of the product where the image was found
            html_location (str): Location in the HTML where the image was found
        """
        try:
            # Prepare the document
            image_doc = {
                "image_hash": image_hash,
                "local_path": str(local_path),
                "original_url": img_url,
                "source_product": product_hash,
                "html_location": html_location,  # Add HTML location information
            }

            # Add timestamp and save to MongoDB
            image_doc = self.mongodb_manager.add_timestamp(image_doc)
            self.mongodb_manager.update_one(
                "image_metadata", {"image_hash": image_hash}, image_doc
            )

        except Exception as e:
            print(f"Error saving to MongoDB: {e}")

    def download_image(self, img_url: str, image_hash: str) -> Path | None:
        """
        Downloads an image and saves it to the structured directory.

        Args:
            img_url (str): The image URL.
            image_hash (str): Hash of the image URL.

        Returns:
            Optional[Path]: Path to the saved image or None if the download fails.
        """

        # Parse the URL to remove query parameters
        parsed_url = urlparse(img_url)
        clean_filename = Path(parsed_url.path).name  # Extracts the actual filename

        # Ensure the filename has an extension
        if "." not in clean_filename:
            clean_filename += ".jpg"  # Default to .jpg if missing

        img_name = f"{image_hash}_{clean_filename}"
        save_path = self.images_dir / img_name

        try:
            response = requests.get(img_url, stream=True, timeout=10)
            response.raise_for_status()

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            return save_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {img_url}: {e}")
            return None

    def process_images(
        self, soup: BeautifulSoup, image_selector: str, product_hash: str
    ) -> list[str]:
        """
        Process all images from a webpage: extract, download, and save metadata.

        Args:
            soup (BeautifulSoup): Parsed HTML page.
            image_selector (str): CSS selector for finding image elements.
            product_hash (str): Hash of the product URL where images were found.

        Returns:
            List[str]: List of image hashes that were processed.
        """
        image_data = self.extract_image_urls(soup, image_selector)
        processed_image_hashes = []

        for img_url, location in image_data:
            # Generate a deterministic hash for the image URL
            image_hash = get_url_hash(img_url)

            # Check if we already have this image in MongoDB
            existing_image = self.mongodb_manager.find_one(
                "image_metadata", {"image_hash": image_hash}
            )

            if existing_image:
                print(f"Image already exists in database: {img_url}")
                processed_image_hashes.append(image_hash)
                continue

            # First download the image locally
            local_path = self.download_image(img_url, image_hash)
            if local_path:
                # Then save the reference to MongoDB
                self.save_image_metadata(
                    img_url, 
                    image_hash, 
                    local_path, 
                    product_hash, 
                    location
                )
                processed_image_hashes.append(image_hash)

        return processed_image_hashes
