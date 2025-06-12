from dataclasses import dataclass

from iris.models.document import Document, DataType
from iris.mixins.embeddable import EmbeddingPayload
from iris.mixins.renderable import RenderableMixin
from iris.protocols.context_protocols import HasImageContext


@dataclass(repr=False, kw_only=True)
class Localization(Document, RenderableMixin):
    """
    Represents a cropped bounding box from an image that may point to a product.
    """

    type: str = "localization"
    parent_image_hash: str
    label: str
    score: float
    bbox: list[float]
    model: str

    @property
    def hash_data(self) -> DataType:
        """
        Fields used to compute the content-based hash for the localization.

        Returns:
            DataType: Identifiers for the image and bounding box.
        """
        data = super().hash_data
        data["type"] = self.type
        data["parent_image_id"] = self.parent_image_hash
        data["label"] = self.label
        data["score"] = self.score
        data["bbox"] = self.bbox
        data["model"] = self.model

        return data
    
    @property
    def id(self) -> str:
        """
        Get the unique identifier for the parent image.

        Returns:
            str: The hash of the parent image.
        """
        return self.parent_image_hash
    
    @property
    def storage_path(self) -> str | None:
        """
        Does not return a storage path for the localization as image access is
        handled through the parent image.
        """
        # No-op for now, as localizations do not have a storage path
        return None
    
    @storage_path.setter
    def storage_path(self, value: str | None) -> None:
        """
        Does not set a storage path for the localization as image access is 
        handled through the parent image.
        """
        # No-op for now, as localizations do not have a storage path
        pass

    @property
    def url(self) -> str | None:
        """
        Does not return a URL for the localization as image access is handled
        through the parent image.
        """
        return None
    
    @url.setter
    def url(self, value: str | None) -> None:
        """
        Does not set a URL for the localization as image access is handled 
        through the parent image.
        """
        pass
    
    def get_embedding_data(self, context: HasImageContext) -> EmbeddingPayload:

        pil_image = context.get_pil_image(self.parent_image_hash)
        augmentations = ...  # Generate augmentations from the image using the bounding box

        embedding_payload =  EmbeddingPayload.from_items(
            augmentations,
            [f"augmentation_{i}" for i in range(len(augmentations))]  # Tags for each augmentation
        )

        return embedding_payload
