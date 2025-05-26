import faiss
import numpy as np
import json
from pathlib import Path
from iris.config.embedding_pipeline_config_manager import EmbeddingDBConfig
from iris.config.data_pipeline_config_manager import ShopConfig

class EmbeddingDatabase:
    def __init__(self, config: EmbeddingDBConfig, shop_config: ShopConfig):
        """
        Initialize the FAISS index for storing embeddings.
        
        Args:
            config (EmbeddingDBConfig): Configuration for the embedding database
            shop_config (ShopConfig): Shop-specific configuration for path
        """
        self.config = config
        self.database_directory = config.storage_path
        
        if config.index_type == "flat":
            self.index = faiss.IndexFlatL2(config.embedding_dim)
        elif config.index_type == "ivf":
            raise NotImplementedError("Index type 'ivf' is not implemented.")
        else:
            raise ValueError(f"Invalid index type '{config.index_type}'. Supported types are: 'flat'.")
            
        self.embeddings = []
        self.ids = []
    

    def add_embedding(self, embedding: np.ndarray, id: str, allow_overwrite: bool = False) -> bool:
        """
        Add an embedding to the database if it doesn't already exist.
        
        Args:
            embedding (np.ndarray): The embedding vector to add.
            id (str): A unique identifier for the embedding.
            allow_overwrite (bool): Whether to overwrite if ID already exists.
            
        Returns:
            bool: True if embedding was added, False if skipped due to duplicate
        """
        # Check for duplicate ID
        if id in self.ids:
            if not allow_overwrite:
                print(f"Warning: Skipping duplicate hash {id}")
                return False
            else:
                # Remove old embedding
                idx = self.ids.index(id)
                self.ids.pop(idx)
                self.embeddings.pop(idx)
                # Need to rebuild index since FAISS doesn't support removal
                self.index = faiss.IndexFlatL2(self.config.embedding_dim)
                if self.embeddings:
                    self.index.add(np.vstack(self.embeddings))

        # Add new embedding
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        self.index.add(embedding)
        self.embeddings.append(embedding)
        self.ids.append(id)
        return True

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Search for the k nearest neighbors of a query embedding.
        
        Args:
            query_embedding (np.ndarray): The query embedding vector.
            k (int): Number of nearest neighbors to retrieve.
            
        Returns:
            List[Tuple[str, float]]: List of (id, distance) for the nearest neighbors.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        distances, indices = self.index.search(query_embedding, k)
        results = [
            (self.ids[idx], distances[0][i])
            for i, idx in enumerate(indices[0]) if idx != -1
        ]
        return results

    def save(self):
        """Save the FAISS index and metadata to disk"""
        self.database_directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = self.database_directory / "embeddings.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata (ids)
        metadata_path = self.database_directory / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.ids, f)

    def load(self):
        """Load the FAISS index and metadata from disk"""
        
        # Load FAISS index
        index_path = self.database_directory / "embeddings.index"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = self.database_directory / "metadata.json"
        with open(metadata_path) as f:
            self.ids = json.load(f)
        
        # Rebuild embeddings list from index if needed
        if hasattr(self.index, 'reconstruct'):
            self.embeddings = [
                self.index.reconstruct(i) 
                for i in range(self.index.ntotal)
            ]

    def get_embedding(self, id: str) -> np.ndarray:
        """
        Retrieve an embedding by its ID.
        
        Args:
            id (str): The unique identifier for the embedding.
            
        Returns:
            np.ndarray: The embedding vector if found, None otherwise.
            
        Raises:
            KeyError: If the ID is not found in the database.
        """
        try:
            idx = self.ids.index(id)
            if hasattr(self.index, 'reconstruct'):
                return self.index.reconstruct(idx)
            else:
                return self.embeddings[idx]
        except ValueError:
            raise KeyError(f"No embedding found for ID: {id}")
