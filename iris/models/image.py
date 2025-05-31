from typing import Self
import torch

from iris.models.document import Document, DataType
from iris.models.product import Product
from iris.mixins.renderable import RenderableMixin
from iris.embedding_pipeline.embedder import Embedder

class Image(Document, RenderableMixin):
    """
    Represents an image document, which may contain localizations.
    """

    @classmethod
    def from_raw(
        cls, 
        url: str,
        debugging_info: list | None = None
    ) -> Self:
        """
        Convert raw scraped data into a structured Image document.

        Args:
            url (str): The URL of the image.
            storage_path (str): Path where the image is stored.
            debugging_info (list | None): Optional debugging information.

        Returns:
            Image: A structured Image instance.
        """
        data = {
            "url": url,
            "debugging_info": debugging_info or []
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

    def embed(self, embedder: Embedder) -> torch.Tensor:
        return embedder.embed_image(self.data["storage_path"])
