from dataclasses import dataclass, field
from PIL import Image as PILImage

from iris.models.document import Document, DataType
from iris.mixins.embeddable import EmbeddingPayload
from iris.mixins.renderable import RenderableMixin
from iris.protocols.context_protocols import HasImageContext

@dataclass(repr=False, kw_only=True)
class Image(Document, RenderableMixin):
    """
    Represents an image document, which may contain localizations.
    """

    type: str = "image"
    url: str
    storage_path: str | None = None
    localization_hashes: list[str] = field(default_factory=list)

    @property
    def hash_data(self) -> DataType:
        """
        Fields used to compute the hash for the product.

        Returns:
            dict: Type and URL only.
        """
        data = super().hash_data
        data["type"] = self.type
        data["url"] = self.url

        return data

    def get_embedding_data(self, context: HasImageContext) -> EmbeddingPayload:
        """
        Get the data required to compute an embedding for this image.

        Args:
            context: The context containing necessary configurations and methods.

        Returns:
            EmbeddingPayload: The payload containing components for embedding.
        """
        pil_image = context.get_pil_image(self.hash, self.storage_path, self.url)

        embedding_payload = EmbeddingPayload.from_items(
            [pil_image],
            ["image"]
        )

        return embedding_payload
    
    def render(self, context: HasImageContext) -> PILImage.Image:
        """
        Render the image using the provided context.

        Args:
            context: The context containing necessary configurations and methods.

        Returns:
            PILImage.Image: The rendered image.
        """
        pil_image, path = context.get_pil_image(self.hash, self.storage_path, self.url)
        self.storage_path = path
        return pil_image