from dataclasses import dataclass
from PIL import Image as PILImage

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
    label_id: str
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
        data["label_id"] = self.label_id
        data["score"] = self.score
        data["bbox"] = self.bbox
        data["model"] = self.model

        return data
    
    def render(self, context: HasImageContext, **kwargs) -> PILImage.Image:
        """
        Render the localization using the provided context.

        Args:
            context: The context containing necessary configurations and methods.

        Returns:
            PILImage.Image: The rendered image.
        """
        pil_image, _ = context.get_pil_image(self.parent_image_hash)

        # Convert relative bbox coordinates to absolute pixel values
        im_width, im_height = pil_image.size
        x_min, y_min, width, height = self.bbox
        abs_bbox = (
            int(x_min * im_width),
            int(y_min * im_height),
            int((width + x_min) * im_width),
            int((height + y_min) * im_height),
        )
        crop = pil_image.crop(abs_bbox)

        return crop
    
    def get_embedding_data(self, context: HasImageContext) -> EmbeddingPayload:

        pil_image = context.get_pil_image(self.parent_image_hash)
        augmentations = ...  # Generate augmentations from the image using the bounding box

        embedding_payload =  EmbeddingPayload.from_items(
            augmentations,
            [f"augmentation_{i}" for i in range(len(augmentations))]  # Tags for each augmentation
        )

        return embedding_payload
