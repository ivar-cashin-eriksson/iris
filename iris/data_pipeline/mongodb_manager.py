import datetime
from typing import Any, TypeAlias, Self
from collections.abc import Iterable

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from iris.config.data_pipeline_config_manager import MongoDBConfig
from iris.models.document import Document
from iris.models.factory import factory as document_factory

# Type aliases
DocumentType: TypeAlias = dict[str, Any]
QueryType: TypeAlias = dict[str, Any]


class MongoDBManager:
    """
    A class to manage MongoDB operations including connections, collections,
    and common database operations.

    Features:
    - Connection management with automatic cleanup
    - Collection access and management
    - Common database operations (insert, update, find)
    - Context manager support
    """

    def __init__(self, config: MongoDBConfig) -> None:
        """
        Initialize the MongoDB manager.

        Args:
            config (MongoDBConfig): MongoDB configuration
        """
        self.config = config
        self._client: MongoClient | None = None
        self._db: Database | None = None
        self._collections: dict[str, Collection] = {}

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Establish connection to MongoDB."""
        if self._client is None:
            self._client = MongoClient(
                self.config.connection_string,
                tlsAllowInvalidCertificates=self.config.tls_allow_invalid_certificates,
            )
            # Use shop-specific database name
            database_name = self.config.database_name
            self._db = self._client[database_name]

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            self._collections.clear()

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a MongoDB collection by name.

        Args:
            collection_name (str): Name of the collection

        Returns:
            Collection: MongoDB collection object
        """
        if collection_name not in self._collections:
            if self._db is None:
                self.connect()
            self._collections[collection_name] = self._db[collection_name]
        return self._collections[collection_name]

    def upsert(
        self,
        collection_name: str,
        docs: Document | Iterable[Document]
    ) -> int:
        """
        Upsert one or many documents into the specified MongoDB collection.

        Args:
            collection_name (str): Target MongoDB collection.
            docs (Document or Iterable[Document]): One or more Document instances.

        Returns:
            int: Number of documents inserted or updated.
        """
        collection = self.get_collection(collection_name)

        if isinstance(docs, Document):
            result = collection.update_one(
                {"_id": docs.id},
                {"$set": docs.to_mongo()},
                upsert=True
            )
            return int(result.modified_count > 0 or result.upserted_id is not None)

        # Assume iterable of documents
        count = 0
        for doc in docs:
            result = collection.update_one(
                {"_id": doc.id},
                {"$set": doc.to_mongo()},
                upsert=True
            )
            if result.modified_count > 0 or result.upserted_id is not None:
                count += 1

        return count

    def find_one(
        self, 
        collection_name: str, 
        query: QueryType
    ) -> Document | None:
        """
        Find a single document in a collection.

        Args:
            collection_name (str): Name of the collection
            query (dict): Query to find the document

        Returns:
            Document | None: Found document or None
        """
        collection = self.get_collection(collection_name)
        data = collection.find_one(query)

        return document_factory(data) if data else None

    def find_all(
        self, 
        collection_name: str, 
        query: QueryType | None = None
    ) -> list[Document]:
        """
        Find documents in a collection.

        Args:
            collection_name (str): Name of the collection
            query (QueryType | None): Query to filter documents. If None, 
                                      returns all documents.

        Returns:
            list[Document]: List of found documents
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query) if query else collection.find()
        return [document_factory(doc) for doc in cursor]
