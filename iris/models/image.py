from typing import Self
import torch

from iris.models.document import Document, DataType
from iris.mixins.renderable import RenderableMixin
from iris.embedding_pipeline.embedder import Embedder

from iris.data_pipeline.image_store_manager import ImageStoreManager
from PIL import Image as PILImage

class Image(Document, RenderableMixin):
    """
    Represents an image document, which may contain localizations.
    """
    def __init__(self, data: DataType):
        """
        Initialize an Image document.

        Args:
            data (dict): The data dictionary containing image metadata.
        """
        super().__init__(data)
        self._loaded_image: PILImage.Image | None = None

    @classmethod
    def from_raw(
        cls, 
        url: str,
        debug_info: dict | None = None
    ) -> Self:
        """
        Convert raw scraped data into a structured Image document.

        Args:
            url (str): The URL of the image.
            storage_path (str): Path where the image is stored.
            debug_info (dict | None): Optional debugging information.

        Returns:
            Image: A structured Image instance.
        """
        data = {
            "url": url,
            "debug_info": debug_info or dict()
        }
        hash = cls.compute_hash_from_data(cls.hash_data_from_data(data))

        # Create complete data structure
        data["_id"] = hash
        data["type"] = "image"

        return cls(data)

    @classmethod
    def hash_data_from_data(cls, data: DataType) -> DataType:
        """
        Fields used to compute the hash for the image.

        Returns:
            dict: URL only.
        """
        return {
            "url": data["url"]
        }

    def load_image(self, store: ImageStoreManager) -> None:
        self._loaded_image, storage_path = store.resolve(
            self.data.get("storage_path"), 
            self.data["url"],
            self.id
        )
        self.data["storage_path"] = storage_path

    def image(self, store: ImageStoreManager | None = None) -> PILImage.Image:
        if self._loaded_image is None:
            if store is None:
                raise RuntimeError("No image loaded and no store provided")
            self.load_image(store)
        return self._loaded_image
    
    def embed(self, embedder: Embedder) -> torch.Tensor:
        return embedder.embed_image(self.image())  # TODO maybe storage manager should be passed here?
