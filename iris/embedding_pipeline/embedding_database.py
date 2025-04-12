import faiss
import numpy as np
from iris.config.embedding_pipeline_config_manager import EmbeddingDBConfig

class EmbeddingDatabase:
    def __init__(self, config: EmbeddingDBConfig):
        """
        Initialize the FAISS index for storing embeddings.
        
        Args:
            config (EmbeddingDBConfig): Configuration for the embedding database.
                index_type must be:
                - "flat": Exact nearest neighbor search using L2 distance
                    - Simple brute force comparison
                    - 100% accurate results
                    - Best for small to medium datasets (<1M vectors)
                    - Memory usage: O(n * d) where n=vectors, d=dimensions
                
                - "ivf": Approximate nearest neighbor search (Not yet supported)
                    - Uses Voronoi-like space partitioning
                    - Trade accuracy for speed on large datasets
                    - Best for >1M vectors
                    - Requires training step
                    - Configurable precision vs speed tradeoff
        """
        if config.index_type == "flat":
            self.index = faiss.IndexFlatL2(config.embedding_dim)
        elif config.index_type == "ivf":
            raise NotImplementedError("Index type 'ivf' is not implemented.")
        else:
            raise ValueError(f"Invalid index type '{config.index_type}'. Supported types are: 'flat'.")
            
        self.embeddings = []
        self.ids = []

    def add_embedding(self, embedding: np.ndarray, id: str):
        """
        Add an embedding to the database.
        
        Args:
            embedding (np.ndarray): The embedding vector to add.
            id (str): A unique identifier for the embedding.
        """
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        self.index.add(embedding)
        self.embeddings.append(embedding)
        self.ids.append(id)

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
