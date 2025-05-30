from abc import ABC, abstractmethod
import torch
from typing import TypeAlias, Self

from iris.mixins.embeddable import EmbeddableMixin
from iris.mixins.hashable import HashableMixin
from iris.mixins.serializable import SerializableMixin

DataType: TypeAlias = dict[str, any]

class Document(ABC, EmbeddableMixin, HashableMixin, SerializableMixin):
    def __init__(self, data: DataType):
        self.data: DataType = data
        self.id = data['_id']
        self.embedding: torch.Tensor | None = None

    @classmethod
    @abstractmethod
    def from_raw(cls, raw: dict[str, any]) -> Self:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"
