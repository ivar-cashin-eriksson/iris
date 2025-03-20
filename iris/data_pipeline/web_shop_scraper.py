import json
from pathlib import Path
from urllib.parse import urlparse
import datetime
import hashlib

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


class WebShopScraper:
    """
    A web scraper that utilizes Selenium and BeautifulSoup to extract 
    image URLs from web pages and download images to a specified directory.

    Features:
    - Uses Selenium to render JavaScript-heavy pages and retrieve HTML.
    - Extracts image URLs based on specified CSS selectors.
    - Downloads images from extracted URLs and saves them locally.
    - Stores image metadata in MongoDB.

    Methods:
    - `extract_html_image_urls(url, filters)`: Extracts image URLs from the page based on given CSS selectors.
    - `download_image_from_url(img_url, directory_path)`: Downloads and saves an image from the given URL.
    """

    def __init__(self, urls: list[str], filters: dict[str, str] | None = None, save_dir: str = "data") -> None:
        """
        Initializes the WebShopScraper.

        Args:
            urls (list[str]): List of product page URLs to scrape.
            filters (dict[str, str] | None): CSS selectors for extracting images & metadata.
                                            Defaults to {"images": "img"}.
            save_dir (str): Directory to store downloaded images.

        Attributes:
            urls (list[str]): The list of product URLs to be scraped.
            filters (dict[str, str]): The CSS selectors used to extract data from web pages.
            save_dir (Path): The directory where images will be stored.
            driver (WebDriver): The Selenium WebDriver instance for loading web pages.
            db: MongoDB database instance.
            image_metadata: MongoDB collection for image metadata.
        """
        self.urls = urls
        self.filters = filters if filters else {"images": "img"}
        self.save_dir = Path(save_dir)
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-software-rasterizer")
        
        # First try to find ChromeDriver in the project directory
        project_root = Path(__file__).parent.parent.parent
        chromedriver_path = project_root / "chromedriver.exe"  # Windows executable
        
        if chromedriver_path.exists():
            print(f"Using ChromeDriver from project directory: {chromedriver_path}")
            service = Service(str(chromedriver_path))
        else:
            print("ChromeDriver not found in project directory, downloading...")
            service = Service(ChromeDriverManager().install())
            
        self.driver: WebDriver = webdriver.Chrome(service=service, options=chrome_options)

        # MongoDB Atlas connection string
        connection_string = "mongodb+srv://test:test@iris-cluster.andes.mongodb.net/?retryWrites=true&w=majority&appName=iris-cluster"
        
        # Initialize MongoDB connection
        self.client = MongoClient(connection_string, tlsAllowInvalidCertificates=True)
        self.db = self.client['iris']
        self.image_metadata = self.db['image_metadata']

    def __del__(self):
        """
        Cleanup: Close MongoDB connection
        """
        if hasattr(self, 'client'):
            self.client.close()

    def __load_page(self, url: str) -> BeautifulSoup | None:
        """
        Loads a web page using Selenium and returns a parsed BeautifulSoup object.

        This method:
        - Opens the specified URL in a headless Chrome browser.
        - Waits for at least one image to be present on the page to ensure content has loaded.
        - Retrieves the page source and parses it into a BeautifulSoup object.

        Args:
            url (str): The URL of the web page to load.

        Returns:
            BeautifulSoup | None: A BeautifulSoup object containing the parsed HTML,
                                or None if the page fails to load.
        """
        self.driver.get(url)

        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img"))
            )
            return BeautifulSoup(self.driver.page_source, "html.parser")
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return None

    def __extract_images(self, soup: BeautifulSoup) -> list[str]:
        """
        Extracts image URLs from the given BeautifulSoup object.

        Args:
            soup (BeautifulSoup): Parsed HTML page.

        Returns:
            List[str]: A list of image URLs.
        """
        image_urls = []
        elements = soup.select(self.filters["images"])  # Select based on provided filter

        for element in elements:
            if element.name == "img":
                # Directly an <img>, extract src
                src = element.get("src")
                if src:
                    image_urls.append(src)
            else:
                # It's a container (e.g., <div>), find all <img> inside
                img_elements = element.find_all("img")
                for img in img_elements:
                    src = img.get("src")
                    if src:
                        image_urls.append(src)

        return image_urls

    def __extract_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        """
        Extracts metadata (title, price, etc.) from the given BeautifulSoup object.

        Args:
            soup (BeautifulSoup): Parsed HTML page.

        Returns:
            Dict[str, str]: Extracted metadata.
        """
        metadata = {}
        for key, selector in self.filters.items():
            if key == "images":
                continue
            metadata[key] = soup.select_one(selector).text.strip() if soup.select_one(selector) else "NOT_FOUND"

        return metadata

    def __get_image_hash(self, img_url: str) -> str:
        """
        Generate a deterministic hash for an image URL.

        Args:
            img_url (str): The image URL to hash.

        Returns:
            str: A hash of the URL.
        """
        return hashlib.md5(img_url.encode()).hexdigest()

    def __save_image_metadata_to_mongodb(self, img_url: str, image_hash: str, local_path: Path, source_url: str) -> None:
        """
        Saves image metadata to MongoDB.

        Args:
            img_url (str): Original URL of the image
            image_hash (str): Hash of the image URL
            local_path (Path): Local path where the image is stored
            source_url (str): URL of the webpage where the image was found
        """
        try:
            # Store reference in image_metadata collection
            image_doc = {
                'image_hash': image_hash,
                'local_path': str(local_path),
                'original_url': img_url,
                'source_url': source_url,
                'created_at': datetime.datetime.utcnow()
            }

            # Use image_hash as the unique identifier to prevent duplicates
            self.image_metadata.update_one(
                {'image_hash': image_hash},
                {'$set': image_doc},
                upsert=True
            )

        except Exception as e:
            print(f"Error saving to MongoDB: {e}")

    def __download_image(self, img_url: str, image_hash: str) -> Path | None:
        """
        Downloads an image and saves it to the structured directory.

        Args:
            img_url (str): The image URL.
            image_hash (str): Hash of the image URL.

        Returns:
            Optional[Path]: Path to the saved image or None if the download fails.
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)

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
        
    def scrape(self) -> None:
        """
        Main method to scrape all product pages, extract images & metadata.
        Save images locally and reference them in MongoDB.
        """
        for url in self.urls:
            print(f"Scraping: {url}")
            soup = self.__load_page(url)

            if not soup:
                continue

            images = self.__extract_images(soup)

            for img_url in images:
                # Generate a deterministic hash for the image URL
                image_hash = self.__get_image_hash(img_url)
                
                # Check if we already have this image in MongoDB
                existing_image = self.image_metadata.find_one({'image_hash': image_hash})
                
                if existing_image:
                    print(f"Image already exists in database: {img_url}")
                    continue
                
                # First download the image locally
                local_path = self.__download_image(img_url, image_hash)
                if local_path:
                    # Then save the reference to MongoDB
                    self.__save_image_metadata_to_mongodb(img_url, image_hash, local_path, url)

            print(f"Completed scraping images from: {url}")


def main():
    urls = [
        "https://pasnormalstudios.com/dk/products/off-race-logo-hoodie",
        "https://pasnormalstudios.com/dk/products/off-race-cotton-twill-pants-limestone"
    ]
    filters = {
        "images": "body > main > div > div > div:nth-child(1) > section.block-wrapper.page-offset-notification.relative.bg-\[\#f1f1f1\].lg\:h-screen.lg\:pt-0 > div.relative.transition-all.lg\:h-screen > div.relative.hidden.lg\:block.h-full", 
        "title": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div.text-xl.font-medium.leading-5", 
        "price": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div.hidden.md\:block > div > div > button > div > div",
        "color": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > button > div > div.flex.items-center.gap-1 > span",
        "num_colorways": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > button > div > div.text-\[9px\].uppercase.opacity-70 > span:nth-child(1)",
        "description": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div:nth-child(4) > div > div > p",
        "made_in": "#radix-\:r7d\: > div.flex.h-full.w-full.flex-col.border-t.px-5.pt-6.md\:px-8 > div.py-12.md\:py-10 > div.flex.w-full.items-center.justify-start.border-t.border-gray-light.py-6 > div > div.flex.w-56.text-xl.font-medium.md\:text-2xl > div",
        "product_details": "#radix-\:r7d\: > div.flex.h-full.w-full.flex-col.border-t.px-5.pt-6.md\:px-8 > div.py-12.md\:py-10 > div.flex.w-full.items-center.justify-start.border-y.border-gray-light.py-6 > div > div.flex.w-56.text-xs.leading-4",
    }
    scraper = WebShopScraper(urls, filters)
    scraper.scrape()


if __name__ == "__main__":
    main()
