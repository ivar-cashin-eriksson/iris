from pathlib import Path
from typing import Dict
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (SessionNotCreatedException, 
                                        TimeoutException, 
                                        WebDriverException)

from iris.config.data_pipeline_config_manager import ScraperConfig
from iris.utils.log import logger


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
        
        # Add browser-like properties
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Add additional headers
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        self.driver: WebDriver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Execute CDP commands to prevent detection
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })

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
            # Wait for initial content
            WebDriverWait(self.driver, self.scraper_config.timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.scraper_config.wait_for_selector)
                )
            )
            
            # Scroll to load all images
            self._scroll_to_load_images()
            
            # Wait additional time for dynamic content
            self.driver.implicitly_wait(2)
            
            return BeautifulSoup(self.driver.page_source, "html.parser")
        except TimeoutException:
            logger.warning(f"Timeout waiting for element on page: {url}")
        except WebDriverException as e:
            logger.warning(f"Selenium error while loading page {url}: {e}")

        return None

    def _scroll_to_load_images(self):
        """
        Scroll the page to trigger lazy loading of images.
        """
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait for potential new content
            time.sleep(1)
            
            # Calculate new scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Break if no more new content
            if new_height == last_height:
                break
                
            last_height = new_height

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
