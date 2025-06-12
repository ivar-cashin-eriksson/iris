from dataclasses import dataclass, field
import torch

from iris.models.document import Document, DataType
from iris.mixins.embeddable import EmbeddingPayload
from iris.embedding_pipeline.embedder import Embedder
from iris.protocols.context_protocols import HasFullContext

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris.models.localization import Localization
    from iris.models.image import Image


@dataclass(repr=False, kw_only=True)
class Product(Document):
    """
    Represents a product listed in the web shop.
    """
    
    type: str = "product"
    title: str
    description: str
    url: str
    image_hashes: list[str] = field(default_factory=list)
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

    def load_localization_ids(
            self, 
            embedder: 
            Embedder, 
            context: 
            HasFullContext
        ) -> None:
        """
        Load localizations for the product from the context.

        Args:
            context (HasFullContext): The context to load localizations from.
        """
        self.localization_hashes = []
        text_embedding = embedder.embed(self)  # Embed the product's title and description

        images: list["Image"] = context.find_all(
            context.config.image_metadata_collection,
            self.image_hashes
        )
        
        for image in images:
            localizations: list["Localization"] = context.find_all(
                context.config.image_metadata_collection,
                image.localization_hashes
            )

            # Get the localization with the closest embedding to the product's
            # title and description embedding
            closest_localization = None
            closest_distance = float("inf")
            for localization in localizations:
                localization_embedding = embedder.embed(localization)
                # TODO: Let some utility handle this, probably qdrant
                distance = torch.dist(text_embedding, localization_embedding).item()
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_localization = localization

            self.localization_hashes.append(closest_localization.hash)
    
    def get_embedding_data(self, context: HasFullContext) -> EmbeddingPayload:

        embedding_payload =  EmbeddingPayload.from_items(
            [self.title, self.description],
            ["title", "description"]
        )

        localizations: list["Localization"] = context.find_all(
            context.config.image_metadata_collection,
            self.localization_hashes
        )

        for localization in localizations:
            embedding_payload.components.extend(
                localization.get_embedding_data(context).components
            )

        return embedding_payload
