from abc import ABC, abstractmethod
import torch
from typing import TypeAlias, Self

from iris.mixins.embeddable import EmbeddableMixin
from iris.mixins.hashable import HashableMixin
from iris.mixins.serializable import SerializableMixin

DataType: TypeAlias = dict[str, any]

class Document(ABC, EmbeddableMixin, HashableMixin, SerializableMixin):
    """
    Abstract base class for all document types in Iris.

    A Document represents a structured, embeddable, hashable, and serializable
    unit of data, typically backed by MongoDB. Subclasses must implement
    `from_raw()` to define how raw scraped data is converted into a structured document.
    """

    def __init__(self, data: DataType):
        """
        Initialize a Document with structured data from MongoDB.

        Args:
            data (dict): The raw MongoDB document data, including `_id`.

        Attributes:
            id (str | ObjectId): The MongoDB identifier for this document.
            data (dict): Raw field data from the database.
            embedding (torch.Tensor | None): Optional in-memory embedding.
        """
        self.id: str = data['_id']
        self.data: DataType = data
        self.embedding: torch.Tensor | None = None

    @classmethod
    @abstractmethod
    def from_raw(cls, raw: dict[str, any]) -> Self:
        """
        Create a document instance from raw scraped or API data.

        This method should normalize and structure the input data to match
        the format expected by the `Document` and its subclasses.

        Args:
            raw (dict): Unstructured raw data.

        Returns:
            Self: A new instance of the subclass.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation of the document.

        Returns:
            str: A concise representation including the document's class and ID.
        """
        return f"<{self.__class__.__name__}(id={self.id})>"
