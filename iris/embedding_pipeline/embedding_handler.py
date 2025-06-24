import torch
from typing import Self

from iris.embedding_pipeline.embedder import Embedder
from iris.data_pipeline.qdrant_manager import QdrantManager
from iris.mixins.embeddable import EmbeddableMixin
from iris.protocols.context_protocols import HasFullContext

class EmbeddingHandler:
    def __init__(self, embedder: Embedder, qdrant_manager: QdrantManager):
        """
        Initialize the EmbeddingHandler with embedder and qdrant manager.
        
        Args:
            embedder (Embedder): An instance of the Embedder class for 
                                 generating embeddings.
            qdrant_manager (QdrantManager): An instance of the QdrantManager 
                                            for sotring embeddings.
        """
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager

    def __enter__(self) -> Self:
        self.qdrant_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.qdrant_manager.__exit__(exc_type, exc_val, exc_tb)

    def get_embedding(
        self, 
        item: EmbeddableMixin, 
        context: HasFullContext, 
        qdrant_collection: str
    ) -> torch.Tensor:
        """
        Get embedding for an embeddable.
        
        Args:
            item (EmbeddableMixin): An instance of an embeddable item.
            
        Returns:
            torch.Tensor: Item embedding tensor.
        """
        
        # TODO: Ensure embeddable has a hash
        self.qdrant_manager.create_collection(
            qdrant_collection, 
            self.qdrant_manager.qdrant_config.embedding_dim
        )
        stored_records = self.qdrant_manager.retrieve_points(
            qdrant_collection, 
            [item.hash]
        )
        if len(stored_records) > 0:
            record = stored_records[0]
            return torch.tensor(record.vector, dtype=torch.float32) 
        
        embedding = self.embedder.embed(item, context)
        self.qdrant_manager.upsert_points(
            qdrant_collection, 
            [embedding],
            payloads=[{"type": item.type, "hash": item.hash}],
            ids=[item.hash]
        )

        return embedding
    
    def get_embeddings(
        self, 
        items: list[EmbeddableMixin], 
        context: HasFullContext, 
        qdrant_collection: str
    ) -> list[torch.Tensor]:
        """
        Get embeddings for a list of embeddables in a batched manner.
        Note, all items must live in the same collection in qdrant.
        
        Args:
            items (list[EmbeddableMixin]): List of embeddable items.
            context (HasFullContext): Context for data retreival.
            qdrant_collection (str): Qdrant collection name for fetching embeddings.
            
        Returns:
            list[torch.Tensor]: List of item embedding tensors.
        """
        # Ensure collection exists
        self.qdrant_manager.create_collection(
            qdrant_collection, 
            self.qdrant_manager.qdrant_config.embedding_dim
        )

        # Get all item hashes
        item_hashes = [item.hash for item in items]
        
        # Try to get stored embeddings from Qdrant
        stored_records = self.qdrant_manager.retrieve_points(
            qdrant_collection, 
            item_hashes
        )
        
        # Create a mapping of hash to stored embedding
        stored_embeddings = {
            record.payload['hash']: torch.tensor(record.vector, dtype=torch.float32)
            for record in stored_records
        }
        
        # Identify items that need new embeddings
        missing_items = [
            item for item in items 
            if item.hash not in stored_embeddings
        ]
        
        # Compute new embeddings in batch if needed
        if missing_items:
            new_embeddings = self.embedder.embed_batch(missing_items, context)
            
            # Save new embeddings to Qdrant
            self.qdrant_manager.upsert_points(
                qdrant_collection,
                new_embeddings,
                payloads=[{"type": item.type, "hash": item.hash} for item in missing_items],
                ids=[item.hash for item in missing_items]
            )
            
            # Add new embeddings to stored mapping
            stored_embeddings.update(zip(
                [item.hash for item in missing_items],
                new_embeddings
            ))
        
        # Return embeddings in original order
        return [stored_embeddings[item.hash] for item in items]

