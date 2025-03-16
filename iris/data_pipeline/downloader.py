from pathlib import Path

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class ImageScraper:
    """
    A web scraper that utilizes Selenium and BeautifulSoup to extract 
    image URLs from web pages and download images to a specified directory.

    Features:
    - Uses Selenium to render JavaScript-heavy pages and retrieve HTML.
    - Extracts image URLs based on specified CSS selectors.
    - Downloads images from extracted URLs and saves them locally.

    Methods:
    - `extract_html_image_urls(url, filters)`: Extracts image URLs from the page based on given CSS selectors.
    - `download_image_from_url(img_url, directory_path)`: Downloads and saves an image from the given URL.
    """

    def __init__(self) -> None:
        """
        Initializes the `ImageScraper` class.

        This constructor:
        - Sets up a Selenium WebDriver instance with headless Chrome.
        - Configures options to run Chrome in a lightweight mode.
        - Uses a specified ChromeDriver path to manage browser automation.

        This setup enables the class to load and parse JavaScript-heavy web pages efficiently.

        Attributes:
            driver (WebDriver): The Selenium Chrome WebDriver instance used for web scraping.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service("/usr/local/bin/chromedriver")
        self.driver = webdriver.Chrome(service=service, options=chrome_options)


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


    def extract_html_image_urls(self, url: str, filters: list[str] = []) -> list[str]:
        """
        Extracts image URLs from HTML-rendered pages (without JavaScript).

        Args:
            page_url (str): URL of the page to extract images from.
            filters (list[str]): CSS selectors to filter images. Defaults to ['img'].

        Returns:
            list[str]: A list of extracted image URLs.
        """
        if not filters:
            filters = ['img']

        soup = self.__load_page(url)

        image_urls = set()

        for selector in filters:
            elements = soup.select(selector)
            for element in elements:
                if element.name == 'img':
                    img_url = element.get('src')
                    if img_url:
                        image_urls.add(img_url)
                else:
                    imgs = element.find_all('img')
                    for img in imgs:
                        img_url = img.get('src')
                        if img_url:
                            image_urls.add(img_url)

        return list(image_urls)


    def download_image_from_url(self, img_url: str, directory_path: Path) -> Path | None:
        """
        Downloads an image from a given URL and saves it to the specified 
        directory.
        
        Parameters:
            img_url (str): The URL of the image to be downloaded.
            directory (Path): The directory where the image will be saved.
        
        Returns:
            Path: The file path where the image is saved if successful.
            None: If the download fails.
        """

        directory_path.mkdir(parents=True, exist_ok=True)

        img_name = Path(img_url).name
        save_path = directory_path / img_name

        try:
            response = requests.get(img_url, stream=True, timeout=100)  # TODO: Move to config
            response.raise_for_status()

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):  # TODO: Move to config file
                    file.write(chunk)

            print(f"Downloaded: {save_path.name}")
            return save_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {img_url}: {e}")
            return None


def main():
    scraper = ImageScraper()
    page_url = "https://pasnormalstudios.com/dk/products/off-race-logo-hoodie"
    filters = ['body > main > div > div > div:nth-child(1) > section.block-wrapper.page-offset-notification.relative.bg-\[\#f1f1f1\].lg\:h-screen.lg\:pt-0 > div.relative.transition-all.lg\:h-screen > div.relative.hidden.lg\:block.h-full > div > div.swiper-wrapper']
    img_urls  = scraper.extract_html_image_urls(page_url, filters)

    for url in img_urls:
        scraper.download_image_from_url(url, "test/")


if __name__ == "__main__":
    main()