from pathlib import Path
from datetime import datetime, timezone
from typing import TypeAlias

DataType: TypeAlias = dict[str, any]

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

    def to_dict(self) -> DataType:
        """
        Return a serializable representation of this document for general use.

        Returns:
            dict: A serializable version of the document.
        """
        ...

    def to_mongo(self) -> dict[str, any]:
        """
        Return a dictionary suitable for inserting into MongoDB.

        Includes `_id` and formats fields for MongoDB compatibility. Assumes
        that `self.id` is defined in the subclass.

        Returns:
            dict: MongoDB-ready document.
        """
        doc = self.to_dict()

        # Convert to BSON-compatible types
        for key, value in doc.items():
            if isinstance(value, Path):
                doc[key] = str(value)
        
        # Remove `_id` field if it is None
        if doc.get("_id") is None:
            doc.pop("_id", None)

        # Add timestamp field
        doc["created_at"] = datetime.now(timezone.utc)

        return doc
