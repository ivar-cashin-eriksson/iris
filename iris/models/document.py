from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import TypeAlias, Self

from iris.mixins.hashable import HashableMixin
from iris.mixins.serializable import SerializableMixin
from iris.mixins.embeddable import EmbeddableMixin

DataType: TypeAlias = dict[str, any]

@dataclass(kw_only=True)
class Document(ABC, HashableMixin, SerializableMixin, EmbeddableMixin):
    """
    Abstract base class for all document types in Iris.

    A Document represents a structured, embeddable, hashable, and serializable
    unit of data, typically backed by MongoDB.
    """

    _id: str | None = None
    created_at: str | None = None
    hash: str | None = None
    type: str
    debug_info: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Post-initialization to set the document ID and ensure image IDs are unique.
        """
        if self.type is None:
            raise ValueError("Document type must be specified")

        if self.hash is None:
            self.hash = self.compute_hash()

    @classmethod
    def from_dict(cls, data: DataType) -> Self:
        return cls(**data)

    def to_dict(self) -> DataType:
        """
        Convert the document to a dictionary format suitable for serialization.

        Returns:
            DataType: A dictionary representation of the document.
        """
        return asdict(self)

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation of the document.

        Returns:
            str: A concise representation including the document's class and ID.
        """
        return f"<{self.__class__.__name__}(hash={self.hash})>"

    @property
    @abstractmethod
    def hash_data(self) -> DataType:
        """
        Generate the data used for computing the document's hash. Subclass 
        implementations should call this method.

        Returns:
            DataType: The data dictionary used for hashing.
        """
        return {"type": self.type}
