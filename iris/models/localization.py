from dataclasses import dataclass
from PIL import Image as PILImage
import torchvision.transforms as T

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
    point: list[float] | None = None
    product_predictions: dict[str, float] | None = None


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
        pil_image, _ = context.get_pil_image(self.parent_image_hash)
        img_width, img_height = pil_image.size

        # --- 1. Convert relative bbox (x, y, w, h) to absolute (x0, y0, x1, y1) ---
        rel_x, rel_y, rel_w, rel_h = self.bbox
        x0 = int(rel_x * img_width)
        y0 = int(rel_y * img_height)
        x1 = int((rel_x + rel_w) * img_width)
        y1 = int((rel_y + rel_h) * img_height)

        base_crop = pil_image.crop((x0, y0, x1, y1))

        # --- 2. Define augmentation pipeline ---
        # TODO: Move to config
        augmentation = T.Compose([
            T.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=15
            ),
            T.RandomHorizontalFlip(p=0.5),
        ])

        # --- 3. Convert to tensor ---
        to_tensor = T.ToTensor()
        to_pil = T.ToPILImage()
        base_tensor = to_tensor(base_crop)

        # --- 4. Generate crops + augmentations ---
        augmentations = [base_crop]
        for i in range(5):  # TODO: Move to config
            aug_tensor = augmentation(base_tensor)
            aug_image = to_pil(aug_tensor)
            augmentations.append(aug_image)

        # --- 5. Return payload ---
        embedding_payload = EmbeddingPayload.from_items(
            augmentations,
            [f"augmentation_{i}" for i in range(len(augmentations))]
        )

        return embedding_payload

    def calculate_point(self) -> None:
        """
        Calculate the point to place the product link at for this localization.

        Note: for now this is the center of the bounding box.
        """
        self.point = (
            self.bbox[0] + self.bbox[2] / 2,
            self.bbox[1] + self.bbox[3] / 2
        )
