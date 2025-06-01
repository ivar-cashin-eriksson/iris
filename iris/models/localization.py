from typing import Self
import torch

from iris.models.document import Document, DataType
from iris.models.image import Image
from iris.mixins.renderable import RenderableMixin
from iris.embedding_pipeline.embedder import Embedder

class Localization(Document, RenderableMixin):
    """
    Represents a cropped bounding box from an image that may point to a product.
    """

    def __init__(self, data: DataType, parent_image: Image):
        """
        Initialize a Localization document with a reference to its parent image.

        Args:
            data (dict): The raw MongoDB document data, including `_id`.
            parent_image (Image): The parent Image instance this localization belongs to.
        """
        super().__init__(data)
        self.parent_image = parent_image

    @classmethod
    def from_raw(
        cls, 
        label: str,
        score: float,
        bbox: list[float],
        model: str,
        parent_image: Image,
        debug_info: dict | None = None
    ) -> Self:
        """
        Convert raw localization data into a structured Localization document.

        Args:
            label (str): The label of the localization.
            score (float): Confidence score for the localization.
            bbox (list[float]): Bounding box coordinates in [x1, y1, x2, y2] format.
            model (str): The model used to generate this localization.
            debug_info (dict | None): Optional debug information for this localization.
            parent_image (Image): The parent Image instance this localization belongs to.

        Returns:
            Localization: A structured instance.
        """
        data = {
            "label": label,
            "score": score,
            "bbox": bbox,
            "model": model,
            "debug_info": debug_info or dict()
        }

        hash = cls.compute_hash_from_data(cls.hash_data_from_data(data, parent_image))

        # Create complete data structure
        data["_id"] = hash
        data["type"] = "localization"

        return cls(data, parent_image)

    @classmethod
    def hash_data_from_data(cls, data: DataType, parent_image: Image) -> DataType:
        """
        Fields used to compute the content-based hash for the localization.

        Returns:
            dict: Identifiers for the image and bounding box.
        """
        return {
            "parent_image_id": parent_image.id,
            "label": data["label"],
            "score": data["score"],
            "bbox": data["bbox"],
            "model": data["model"]
        }

    def embed(self, embedder: Embedder) -> torch.Tensor:
        """
        Generate an embedding for this localization by applying 5 augmentations
        and averaging the resulting embeddings.

        Args:
            embedder: An embedding model capable of cropping and embedding.

        Returns:
            torch.Tensor: The averaged embedding.
        """
        # Perhaps augmentation should be done here instead of in the embedder?
        crops = [
            embedder.crop_and_embed(self.data["image_path"], self.data["bbox"])
            for _ in range(5)
        ]
        return torch.stack(crops).mean(dim=0)
