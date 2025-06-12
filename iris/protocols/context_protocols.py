from PIL import Image as PILImage
from pathlib import Path
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from iris.models.document import Document
    from iris.config.data_pipeline_config_manager import MongoDBConfig


class HasMongoDBContext(Protocol):
    @property
    def config(self) -> "MongoDBConfig":
        ...

    def find_all(self, collection: str, document_hashes: list[str]) -> list["Document"]:
        ...

class HasImageContext(Protocol):
    def get_pil_image(
        self, 
        image_id: str | None = None,
        path: Path | None = None, 
        url: str | None = None
    ) -> tuple[PILImage.Image, Path]:
        ...

class HasFullContext(HasImageContext, HasMongoDBContext, Protocol):
    """Combines image + MongoDB access."""
    pass