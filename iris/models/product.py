from dataclasses import dataclass, field
import torch

from iris.models.document import Document, DataType
from iris.mixins.embeddable import EmbeddingPayload
from iris.embedding_pipeline.embedding_handler import EmbeddingHandler
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
    metadata: dict[str, str]
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

    def load_localization_hashes(
            self, 
            embedding_handler: EmbeddingHandler, 
            context: HasFullContext
        ) -> None:
        """
        Load localizations for the product from the context.

        Args:
            context (HasFullContext): The context to load localizations from.
        """
        self.localization_hashes = []
        text_embedding = embedding_handler.embedder.embed(self, context)  # Embed the product's title and description

        images: list["Image"] = context.find_all(
            context.config.image_metadata_collection,
            {"hash": {"$in": self.image_hashes}}
        )
        
        for image in images:
            localizations: list["Localization"] = context.find_all(
                context.config.localization_collection,
                {"hash": {"$in": image.localization_hashes}}
            )

            # Get the localization with the closest embedding to the product's
            # title and description embedding
            closest_localization = None
            closest_distance = float("inf")
            for localization in localizations:
                localization_embedding = embedding_handler.get_embedding(
                    localization, 
                    context,
                    embedding_handler.qdrant_manager.qdrant_config.localization_collection
                )
                # TODO: Let some utility handle this, probably qdrant
                distance = torch.dist(text_embedding, localization_embedding).item()
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_localization = localization

            if closest_localization is not None:
                self.localization_hashes.append(closest_localization.hash)
    
    def get_embedding_data(self, context: HasFullContext) -> EmbeddingPayload:

        embedding_payload =  EmbeddingPayload.from_items(
            list(self.metadata.values()),
            list(self.metadata.keys())
        )

        localizations: list["Localization"] = context.find_all(
            context.config.localization_collection,
            {"hash": {"$in": self.localization_hashes}}
        )

        for localization in localizations:
            embedding_payload.components.extend(
                localization.get_embedding_data(context).components
            )

        return embedding_payload
