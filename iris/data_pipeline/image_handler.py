from abc import ABC
from bs4 import BeautifulSoup

from iris.models.image import Image


class ImageHandler(ABC):
    """
    A collection of image scraping operations.
    """

    @classmethod
    def _get_element_path(cls, element: BeautifulSoup) -> str:
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

    @classmethod
    def _get_image_data(cls, img_element: BeautifulSoup) -> tuple[str, str] | None:
        """
        Extract the image source URL and its DOM location from a single <img> element.

        Args:
            img_element (BeautifulSoup): A single <img> tag parsed from the page.

        Returns:
            tuple[str, str] | None: A tuple containing the image URL and a DOM location
            string describing the element's position in the page structure.
        """
        src = img_element.get("src")
        dom_location = cls._get_element_path(img_element)

        return (src, dom_location)

    @classmethod
    def extract_image_urls(
        cls,
        soup: BeautifulSoup,
        image_selector: str
    ) -> tuple[list[str], list[str]]:
        """
        Extract all image URLs and their DOM locations from a parsed HTML document.

        This method uses the given CSS selector to locate image elements or their containers,
        then finds all <img> tags and extracts their "src" URLs along with a description
        of where they appear in the document structure.

        Args:
            soup (BeautifulSoup): Parsed HTML document.
            image_selector (str): CSS selector to locate image containers or image tags.

        Returns:
            tuple[list[str], list[str]]: A pair of lists — one with image URLs,
            and one with their corresponding DOM location descriptions.
    """
        image_urls = []
        dom_locations = []
        elements = soup.select(image_selector)

        for element in elements:
            if element.name == "img":
                # Directly an <img>, extract data
                if data := cls._get_image_data(element):
                    image_urls.append(data[0])
                    dom_locations.append(data[1])
            else:
                # It's a container (e.g., <div>), find all <img> inside
                for img in element.find_all("img"):
                    if data := cls._get_image_data(img):
                        image_urls.append(data[0])
                        dom_locations.append(data[1])

        return image_urls, dom_locations

    @classmethod
    def extract_images(
        cls, 
        soup: BeautifulSoup,
        image_selector: str
    ) -> list[Image]:
        """
        Extract image elements from a parsed HTML document and convert them into Image objects.

        This method uses a CSS selector to locate <img> elements or equivalent, extracts their URLs
        and DOM locations, and creates Image instances for each.

        Args:
            soup (BeautifulSoup): The parsed HTML content (soup object).
            image_selector (str): A CSS selector targeting image elements to extract.

        Returns:
            list[Image]: List of Image instances of extracted images.
        """
        urls, dom_locations = cls.extract_image_urls(soup, image_selector)
        images = []

        for url, dom_location in zip(urls, dom_locations):
            image = Image.from_raw(url, debug_info={"dom_location": dom_location})
            images.append(image)

        return images
