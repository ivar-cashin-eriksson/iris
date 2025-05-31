from iris.utils.log import logger
import torch

class SerializableMixin:
    """
    Mixin for exporting a document to dictionary formats.

    Provides both `.to_dict()` for general-purpose use (debugging, JSON, testing)
    and `.to_mongo()` for database persistence with enforced `_id`.

    Subclasses must define:
        - `self.data`: the raw content dictionary
        - `self.id`: the MongoDB ID
        - `self.embedding`: (optional) torch.Tensor or list
    """

    def to_dict(self) -> dict[str, any]:
        """
        Return a serializable representation of this document for general use.

        Converts tensors to lists and skips internal-only fields.
        By default, uses `self.data` as the base structure.

        Returns:
            dict: A serializable version of the document.
        """
        if not hasattr(self, "data"):
            raise AttributeError("Class using SerializableMixin must define `self.data`")

        doc = self.data.copy()

        if hasattr(self, "embedding") and isinstance(self.embedding, torch.Tensor):
            doc["embedding"] = self.embedding.tolist()

        return doc

    def to_mongo(self) -> dict[str, any]:
        """
        Return a dictionary suitable for inserting into MongoDB.

        Includes `_id` and formats fields for MongoDB compatibility.

        Returns:
            dict: MongoDB-ready document.
        """
        if not hasattr(self, "id"):
            raise AttributeError("Class using SerializableMixin must define `self.id`")

        doc = self.to_dict()
        doc["_id"] = self.id
        return doc
