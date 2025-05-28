from collections.abc import Sequence
from typing import TypeAlias, Any, Self

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    ScoredPoint,
    Record
)

from iris.config.data_pipeline_config_manager import QdrantConfig

# Type aliases
VectorType: TypeAlias = NDArray[np.float32]
PayloadType: TypeAlias = dict[str, Any]


class QdrantManager:
    """
    A class to manage Qdrant vector database operations including connections
    and common vector operations.

    Features:
    - Connection management with automatic cleanup
    - Collection creation and management
    - Vector operations (upsert, search, retrieve)
    - Context manager support
    """

    def __init__(self, qdrant_config: QdrantConfig) -> None:
        """
        Initialize the Qdrant manager.

        Args:
            qdrant_config (QdrantConfig): Qdrant configuration
        """
        self.qdrant_config = qdrant_config
        self._client: QdrantClient | None = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Establish connection to Qdrant."""
        if self._client is None:
            self._client = QdrantClient(
                url=self.qdrant_config.url,
                api_key=self.qdrant_config.api_key
            )

    def close(self) -> None:
        """Close the Qdrant connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> bool:
        """
        Create a new collection or recreate an existing one.

        Args:
            collection_name (str): Name of the collection
            vector_size (int): Size of vectors to store
            distance (Distance): Distance metric to use
            recreate (bool): Whether to recreate if collection exists

        Returns:
            bool: True if operation was successful
        """
        try:
            if recreate:
                self._client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
            else:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
            return True
        except UnexpectedResponse:
            return False

    def upsert_points(
        self,
        collection_name: str,
        vectors: Sequence[VectorType],
        payloads: Sequence[PayloadType],
        ids: Sequence[int] | None = None
    ) -> bool:
        """
        Insert or update points in a collection.

        Args:
            collection_name (str): Name of the collection
            vectors (List[np.ndarray]): List of vectors to insert
            payloads (List[dict]): List of payloads for each vector
            ids (Optional[List[int]]): Optional list of IDs for the points

        Returns:
            bool: True if operation was successful
        """
        try:
            if ids is None:
                ids = list(range(len(vectors)))

            points = [
                PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload
                )
                for id, vector, payload in zip(ids, vectors, payloads)
            ]

            self._client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception:
            return False

    def search_points(
        self,
        collection_name: str,
        query_vector: VectorType,
        limit: int = 10
    ) -> list[ScoredPoint]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name (str): Name of the collection
            query_vector (np.ndarray): Vector to search for
            limit (int): Maximum number of results to return

        Returns:
            List[ScoredPoint]: List of search results with scores
        """
        return self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

    def retrieve_points(
        self,
        collection_name: str,
        ids: Sequence[int],
        with_vectors: bool = True
    ) -> list[Record]:
        """
        Retrieve specific points by their IDs.

        Args:
            collection_name (str): Name of the collection
            ids (List[int]): List of point IDs to retrieve
            with_vectors (bool): Whether to include vector data

        Returns:
            List[Record]: List of retrieved points
        """
        return self._client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=with_vectors
        )

    def delete_points(
        self,
        collection_name: str,
        ids: Sequence[int]
    ) -> bool:
        """
        Delete points from a collection.

        Args:
            collection_name (str): Name of the collection
            ids (List[int]): List of point IDs to delete

        Returns:
            bool: True if operation was successful
        """
        try:
            self._client.delete(
                collection_name=collection_name,
                points_selector=ids
            )
            return True
        except Exception:
            return False

    def get_collections(self) -> list[str]:
        """
        Get list of available collections.

        Returns:
            list[str]: List of collection names
        """
        collections = self._client.get_collections()
        return [collection.name for collection in collections.collections]
