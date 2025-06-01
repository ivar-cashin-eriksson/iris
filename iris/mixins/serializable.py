import torch
from pathlib import Path
from datetime import datetime, timezone

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
        
        return self.data.copy()

    def to_mongo(self) -> dict[str, any]:
        """
        Return a dictionary suitable for inserting into MongoDB.

        Includes `_id` and formats fields for MongoDB compatibility.

        Returns:
            dict: MongoDB-ready document.
        """
        doc = self.to_dict()

        # Convert to BSON-compatible types
        for key, value in doc.items():
            if isinstance(value, torch.Tensor):
                doc[key] = value.tolist()
            elif isinstance(value, Path):
                doc[key] = str(value)
        
        # Add `_id` field for MongoDB
        if not hasattr(self, "id"):
            raise AttributeError("Class using SerializableMixin must define `self.id`")
        doc["_id"] = self.id

        # Add timestamp field
        doc["created_at"] = datetime.now(timezone.utc)

        return doc
