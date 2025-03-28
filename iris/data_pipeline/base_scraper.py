from pathlib import Path
from typing import Dict

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from iris.config.config_manager import ScraperConfig


class BaseScraper:
    """
    A base class for web scraping that handles page loading and basic HTML parsing.

    Features:
    - Selenium WebDriver setup and management
    - Page loading with wait conditions
    - Basic HTML parsing with BeautifulSoup
    """

    def __init__(self, scraper_config: ScraperConfig) -> None:
        """
        Initialize the BaseScraper with a configured Selenium WebDriver.
        """
        self.scraper_config = scraper_config

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

        self.driver: WebDriver = webdriver.Chrome(
            service=service, options=chrome_options
        )

    def __del__(self):
        """
        Cleanup: Close the WebDriver when the object is destroyed.
        """
        if hasattr(self, "driver"):
            self.driver.quit()

    def load_page(self, url: str) -> BeautifulSoup | None:
        """
        Loads a web page using Selenium and returns a parsed BeautifulSoup object.

        Args:
            url (str): The URL of the web page to load.
            wait_for_selector (str): CSS selector to wait for before considering page loaded.
            timeout (int): Maximum time to wait for the selector in seconds.

        Returns:
            BeautifulSoup | None: A BeautifulSoup object containing the parsed HTML,
                                or None if the page fails to load.
        """
        self.driver.get(url)

        try:
            WebDriverWait(self.driver, self.scraper_config.timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.scraper_config.wait_for_selector)
                )
            )
            return BeautifulSoup(self.driver.page_source, "html.parser")
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return None

    def extract_data(
        self, soup: BeautifulSoup, selectors: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Extracts data from the given BeautifulSoup object using provided CSS selectors.

        Args:
            soup (BeautifulSoup): Parsed HTML page.
            selectors (Dict[str, str]): Dictionary mapping data keys to CSS selectors.

        Returns:
            Dict[str, str]: Extracted data with values from matching elements.
        """
        data = {}
        for key, selector in selectors.items():
            element = soup.select_one(selector)
            data[key] = element.text.strip() if element else "NOT_FOUND"
        return data
