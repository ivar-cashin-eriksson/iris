from abc import ABC, abstractmethod
import torch

from iris.data_pipeline.qdrant_manager import QdrantManager
from iris.embedding_pipeline.embedder import Embedder
from iris.utils.log import logger

class Embeddable(ABC):
    """
    Abstract interface for embeddable documents.

    This mixin defines the standard interface for computing, caching,
    retrieving, and saving embeddings. It assumes implementing classes
    define `self.embedding` and `self.id` attributes.
    """

    @abstractmethod
    def embed(self, embedder: Embedder) -> torch.Tensor:
        """
        Generate a new embedding for this document using the provided embedder.

        Args:
            embedder (Embedder): An instance capable of embedding this document.

        Returns:
            torch.Tensor: The computed embedding vector.
        """
        pass

    def get_embedding(self, qdrant_manager: QdrantManager | None, embedder: Embedder | None) -> torch.Tensor:
        """
        Retrieve or compute an embedding for this document.

        The method first checks for a cached embedding in memory, then tries to fetch
        from Qdrant. If no embedding is found, it falls back to computing it using the
        provided embedder.

        Args:
            qdrant_manager (QdrantManager | None): Optional manager to fetch from Qdrant.
            embedder (Embedder | None): Optional embedder to compute the embedding if needed.

        Returns:
            torch.Tensor: The final embedding.

        Raises:
            RuntimeError: If the embedding could not be retrieved or computed.
        """
        if self.embedding is not None:
            logger.debug(f"Embedding already cached for document {self.id}")
            return self.embedding

        try:
            result = qdrant_manager.retrieve(self)
            if result:
                self.set_embedding(result)
                logger.info(f"Fetched embedding from Qdrant for document {self.id}")
                return self.embedding
        except Exception as e:
            logger.warning(f"Failed to retrieve embedding from Qdrant for {self.id}: {e}")

        try:
            result = self.embed(embedder)
            if result is not None:
                self.set_embedding(result)
                logger.info(f"Computed new embedding for document {self.id}")
                return self.embedding
        except Exception as e:
            logger.warning(f"Embedding computation failed for {self.id}: {e}")

        logger.error(f"No embedding available for document {self.id}")
        raise RuntimeError(f"Embedding could not be retrieved or computed for document {self.id}")

    def set_embedding(self, embedding: torch.Tensor) -> None:
        """
        Set the embedding on this document.

        Args:
            embedding (torch.Tensor): The computed embedding vector.
        """
        self.embedding = embedding

    def save_embedding(self, qdrant_manager: QdrantManager, collection_name: str):
        """
        Save the embedding to Qdrant.

        Args:
            qdrant_manager (QdrantManager): The manager responsible for Qdrant operations.
            collection_name (str): The name of the Qdrant collection to save to.

        Raises:
            NotImplementedError: Placeholder — implementation is pending.
        """
        if self.embedding is None:
            logger.warning(f"Cannot save embedding for document {self.id} because it is None")
        else:
            raise NotImplementedError("Embedding saving is not implemented yet.")
