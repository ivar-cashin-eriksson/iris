from dataclasses import dataclass
from typing import Literal, Self
from PIL import Image as PILImage

from abc import abstractmethod

from iris.protocols.context_protocols import HasFullContext


@dataclass
class EmbeddingComponent:
    type: Literal["text", "image"]
    content: str | PILImage.Image
    tag: str | None = None # e.g. "title", "desc", "gallery_1"


@dataclass
class EmbeddingPayload:
    components: list[EmbeddingComponent]

    @classmethod
    def from_items(
        cls, 
        items: list[str | PILImage.Image], 
        tags: list[str] | None = None
    ) -> Self:
        if tags is not None and len(tags) != len(items):
            raise ValueError("Length of `tags` must match length of `items`.")

        components = []
        for i, item in enumerate(items):
            tag = tags[i] if tags else None

            match item:
                case str():
                    components.append(EmbeddingComponent(
                        type="text",
                        content=item,
                        tag=tag or f"text_{i}"
                    ))
                case PILImage.Image():
                    components.append(EmbeddingComponent(
                        type="image",
                        content=item,
                        tag=tag or f"image_{i}"
                    ))
                case _:
                    raise TypeError(f"Unsupported item type: {type(item)}")

        return cls(components=components)


class EmbeddableMixin:
    """
    Abstract interface for embeddable documents.

    This mixin defines the standard interface for an embeddable object. 
    To be used together with an embedder.
    """

    @abstractmethod
    def get_embedding_data(self, context: HasFullContext) -> EmbeddingPayload:
        """
        Get the data required to compute an embedding for this document.

        Args:
            context (Embedder): An instance capable of embedding this document.

        Returns:
            torch.Tensor: The computed embedding vector.
        """
        ...
