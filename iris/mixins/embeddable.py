from abc import ABC, abstractmethod
import torch

from iris.data_pipeline.qdrant_manager import QdrantManager
from iris.embedding_pipeline.embedder import Embedder
from iris.utils.log import logger

class Embeddable(ABC):
    @abstractmethod
    def embed(self, embedder: Embedder) -> torch.Tensor:
        """Generate a new embedding for this document."""
        pass

    def get_embedding(self, qdrant_manager: QdrantManager | None, embedder: Embedder | None) -> torch.Tensor:
        
        # Check if the embedding is already cached
        if self.embedding is not None:
            logger.debug(f"Embedding already cached for document {self.hash}")
            return self.embedding

        # Try to retrieve the embedding from Qdrant
        try:
            result = qdrant_manager.retrieve(self)
            if result:
                self.set_embedding(result)
                logger.info(f"Fetched embedding from Qdrant for document {self.hash}")
                return self.embedding
        except Exception as e:
            logger.warning(f"Failed to retrieve embedding from Qdrant for {self.hash}: {e}")

        # If no embedding is cached or retrieved, compute a new one
        try:
            result = self.embed(embedder)
            if result is not None:
                self.set_embedding(result)
                logger.info(f"Computed new embedding for document {self.hash}")
                return self.embedding
        except Exception as e:
            logger.warning(f"Embedding computation failed for {self.hash}: {e}")

        logger.error(f"No embedding available for document {self.hash}")
        raise RuntimeError(f"Embedding could not be retrieved or computed for document {self.hash}")

    def set_embedding(self, embedding: torch.Tensor) -> None:
        """Set the embedding."""
        self.embedding = embedding

    def save_embedding(self, qdrant_manager: QdrantManager, collection_name: str):
        """Save embedding to Qdrant."""
        if self.embedding is None:
            logger.warning(f"Cannot save embedding for document {self.hash} because it is None")
        else:
            raise NotImplementedError("Embedding saving is not implemented yet.")
