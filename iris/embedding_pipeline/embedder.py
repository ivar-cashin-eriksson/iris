import torch

from iris.mixins.embeddable import EmbeddingPayload, EmbeddableMixin
from iris.data_pipeline.qdrant_manager import QdrantManager
from iris.utils.log import logger


class Embedder:
    def __init__(self, model_name: str):
        ...

    def embed_text(self, text: str) -> list:
        ...
    
    def embed_image(self, image_path: str) -> list:
        ...

    def embed_batch(self, texts: list) -> list:
        ...
        
        
    def embed(
        self, 
        item: EmbeddingPayload | EmbeddableMixin, 
        qdrant_manager: QdrantManager | None
    ) -> torch.Tensor:
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
            logger.debug(f"Embedding already cached for document {self!r}")
            return self.embedding

        try:
            result = qdrant_manager.retrieve(self)
            if result:
                self.set_embedding(result)
                logger.info(f"Fetched embedding from Qdrant for document {self!r}")
                return self.embedding
        except Exception as e:
            logger.warning(f"Failed to retrieve embedding from Qdrant for {self!r}: {e}")

        try:
            result = self.embed(embedder)
            if result is not None:
                self.set_embedding(result)
                logger.info(f"Computed new embedding for document {self!r}")
                return self.embedding
        except Exception as e:
            logger.warning(f"Embedding computation failed for {self!r}: {e}")

        logger.error(f"No embedding available for document {self!r}")
        raise RuntimeError(f"Embedding could not be retrieved or computed for document {self!r}")


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
            logger.warning(f"Cannot save embedding for document {self!r} because it is None")
        else:
            raise NotImplementedError("Embedding saving is not implemented yet.")
