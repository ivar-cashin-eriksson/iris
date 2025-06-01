import os
import io
import re
import requests
from PIL import Image as PILImage
from abc import ABC, abstractmethod
from pathlib import Path

from iris.config.data_pipeline_config_manager import ImageStoreConfig
from iris.utils.log import logger


class ImageStoreManager:
    """
    A manager that resolves images based on information from an Image document.
    Tries storage_location, then downloads from URL.
    """

    def __init__(self, config: ImageStoreConfig):
        self.config = config

        match self.config.storage_backend:
            case "local":
                logger.debug("Using local storage backend.")
                self.storage_backend = LocalStorageHandler(self.config.storage_path)
            case _:
                logger.error(f"Unsupported storage backend: {self.config.storage_backend}")
                raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")


    def resolve(
        self, 
        storage_location: Path | None = None, 
        url: str | None = None,
        image_id: str | None = None
    ) -> tuple[PILImage.Image, Path]:
        """
        Resolve image from storage path or url.

        Loads the image from the specified storage location if available,
        otherwise downloads it from the provided URL and stores the image.

        Args:
            storge_location (str): Storage location of image, can be a local path or a blob key.
            url (str): URL to download the image if storage_location is not provided.

        Returns:
            tuple[PIL.Image, Path]: The loaded image and storage location.

        Raises:
            FileNotFoundError: If the image cannot be resolved from any source.
        """
        if storage_location is not None:
            logger.debug(f"Attempting to resolve image from storage_location: {storage_location}")
            return self.storage_backend.load(storage_location), storage_location
        elif (url is not None) and (image_id is not None):
            logger.debug(f"Falling back to downloading image from URL: {url}")
            image = self._download(url)
            path = self.storage_backend.save(image, image_id)
            return image, path
        else:
            logger.error("No storage location or URL and image ID provided for image resolution.")
            raise FileNotFoundError("No storage location or URL and image ID provided for image resolution.")


    def _download(self, url: str) -> PILImage.Image:
        try:
            logger.debug(f"Downloading image from URL: {url}")
            response = requests.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            image = PILImage.open(io.BytesIO(response.content)).convert("RGB")
            logger.debug(f"Successfully downloaded image from URL: {url}")
            return image
        except (requests.RequestException, OSError) as e:
            logger.error(f"Failed to download or decode image from URL {url}: {e}")
            raise FileNotFoundError(f"Could not download image from URL: {url}")


class StorageBackendHandler(ABC):
    @abstractmethod
    def save(self, image: PILImage.Image, image_id: str) -> Path:
        pass

    @abstractmethod
    def load(self, image_id: str) -> PILImage.Image:
        pass


class LocalStorageHandler(StorageBackendHandler):
    def __init__(self, directory: Path):
        self.directory = directory

    def save(self, image: PILImage.Image, image_id: str) -> Path:
        path = self.directory / f"{image_id}.jpg"
        os.makedirs(path.parent, exist_ok=True)
        image.save(path)
        logger.debug(f"Image saved at path: {path}")
        return path

    def load(self, image_id: str) -> PILImage.Image:
        path = self.directory / f"{image_id}.jpg"
        if not path.exists():
            logger.error(f"Image file not found at path: {path}")
            raise FileNotFoundError(path)
        return PILImage.open(path).convert("RGB")