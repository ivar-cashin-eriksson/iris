import datetime
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from iris.config.data_pipeline_config_manager import ShopConfig, MongoDBConfig


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

    def __init__(self, mongodb_config: MongoDBConfig) -> None:
        """
        Initialize the MongoDB manager.

        Args:
            mongodb_config (MongoDBConfig): MongoDB configuration
        """
        self.mongodb_config = mongodb_config
        self._client: MongoClient | None = None
        self._db: Database | None = None
        self._collections: dict[str, Collection] = {}

    def __enter__(self) -> "MongoDBManager":
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
                self.mongodb_config.connection_string,
                tlsAllowInvalidCertificates=self.mongodb_config.tls_allow_invalid_certificates,
            )
            # Use shop-specific database name
            database_name = self.mongodb_config.database_name
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

    def insert_one(self, collection_name: str, document: dict[str, Any]) -> str:
        """
        Insert a single document into a collection.

        Args:
            collection_name (str): Name of the collection
            document (dict): Document to insert

        Returns:
            str: ID of the inserted document
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)

    def update_one(
        self,
        collection_name: str,
        filter_query: dict[str, Any],
        update_data: dict[str, Any],
        upsert: bool = True,
    ) -> bool:
        """
        Update a single document in a collection.

        Args:
            collection_name (str): Name of the collection
            filter_query (dict): Query to find the document
            update_data (dict): Data to update
            upsert (bool): Whether to create if document doesn't exist

        Returns:
            bool: True if document was updated or created
        """
        collection = self.get_collection(collection_name)
        result = collection.update_one(
            filter_query, {"$set": update_data}, upsert=upsert
        )
        return result.modified_count > 0 or result.upserted_id is not None

    def find_one(
        self, collection_name: str, query: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Find a single document in a collection.

        Args:
            collection_name (str): Name of the collection
            query (dict): Query to find the document

        Returns:
            Optional[dict]: Found document or None
        """
        collection = self.get_collection(collection_name)
        return collection.find_one(query)

    def find_all(
        self, collection_name: str, query: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Find documents in a collection.

        Args:
            collection_name (str): Name of the collection
            query (dict | None): Query to filter documents. If None, returns all documents.

        Returns:
            list[dict]: List of found documents
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query) if query else collection.find()
        return list(cursor)

    def delete_one(self, collection_name: str, query: dict[str, Any]) -> bool:
        """
        Delete a single document from a collection.

        Args:
            collection_name (str): Name of the collection
            query (dict): Query to find the document to delete

        Returns:
            bool: True if document was deleted
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_one(query)
        return result.deleted_count > 0

    def count_documents(self, collection_name: str, query: dict[str, Any]) -> int:
        """
        Count documents in a collection matching a query.

        Args:
            collection_name (str): Name of the collection
            query (dict): Query to match documents

        Returns:
            int: Number of matching documents
        """
        collection = self.get_collection(collection_name)
        return collection.count_documents(query)

    def add_timestamp(self, document: dict[str, Any]) -> dict[str, Any]:
        """
        Add a timestamp to a document.

        Args:
            document (dict): Document to add timestamp to

        Returns:
            dict: Document with timestamp added
        """
        document["created_at"] = datetime.datetime.utcnow()
        return document
