import os
import io
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


    def get_pil_image(
        self,
        image_id: str | None = None,
        path: Path | None = None, 
        url: str | None = None,
    ) -> tuple[PILImage.Image, Path]:
        """
        Resolve image from id, storage path, or url.

        Loads the image from the specified storage location if available,
        otherwise downloads it from the provided URL and stores the image.

        Args:
            image_id (str | None): Optional image ID to use for storage.
            path (Path | None): File path of image, can be a local path, blob 
                                key or similar.
            url (str | None): URL to download the image if id, and path is not 
                              provided.

        Returns:
            tuple[PIL.Image, Path]: The loaded image and storage location.

        Raises:
            FileNotFoundError: If the image cannot be resolved from any source.
        """
        if all(x is None for x in (image_id, path, url)):
            logger.error("No image ID, file path, or URL provided for image resolution.")
            raise ValueError("At least one of image_id, storage_location, or url must be provided.")
        
        if image_id is not None:
            try:
                logger.debug(f"Attempting to resolve image from id: {image_id}")
                image = self.storage_backend.load_from_id(image_id)
                path = self.storage_backend.path_from_id(image_id)
                return image, path
            except FileNotFoundError:
                logger.warning(f"Image with ID {image_id} not found in storage.")
        if path is not None:
            try:
                logger.debug(f"Attempting to resolve image from storage_location: {path}")
                image = self.storage_backend.load_from_path(path)
                return image, path
            except FileNotFoundError:
                logger.warning(f"Image not found at storage location: {path}")
        if (url is not None) and (image_id is not None):
            logger.debug(f"Falling back to downloading image from URL: {url}")
            image = self._download(url)
            path = self.storage_backend.save_to_id(image, image_id)
            return image, path
        
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
    def save_to_path(self, image: PILImage.Image, path: Path) -> Path:
        ...

    @abstractmethod
    def load_from_path(self, path: Path) -> PILImage.Image:
        ...
    
    @abstractmethod
    def path_from_id(self, image_id: str) -> Path:
        ...

    def save_to_id(self, image: PILImage.Image, image_id: str) -> Path:
        path = self.path_from_id(image_id)
        return self.save_to_path(image, path)

    def load_from_id(self, image_id: str) -> PILImage.Image:
        path = self.path_from_id(image_id)
        return self.load_from_path(path)


class LocalStorageHandler(StorageBackendHandler):
    def __init__(self, directory: Path):
        self.directory = directory

    def path_from_id(self, image_id: str) -> Path:
        return self.directory / f"{image_id}.jpg"
    
    def save_to_path(self, image: PILImage.Image, path: Path) -> Path:
        os.makedirs(path.parent, exist_ok=True)
        image.save(path)
        logger.debug(f"Image saved at path: {path}")
        return path

    def load_from_path(self, path: Path) -> PILImage.Image:
        if not path.exists():
            logger.error(f"Image file not found at path: {path}")
            raise FileNotFoundError(path)
        return PILImage.open(path).convert("RGB")