from typing import Self
import torch

from iris.models.document import Document, DataType
from iris.embedding_pipeline.embedder import Embedder

class Product(Document):
    """
    Represents a product listed in the web shop.
    """

    @classmethod
    def from_raw(
        cls, 
        title: str,
        description: str,
        url: str,
        image_ids: list[str],
        debugging_info: list | None = None
    ) -> Self:
        """
        Convert raw scraped data into a structured Product document.

        Args:
            title (str): The product's title.
            description (str): The product's description.
            url (str): The URL of the product page.
            debugging_info (list | None): Optional debugging information.

        Returns:
            Product: A structured Product instance.
        """
        data = {
            "title": title,
            "description": description,
            "url": url,
            "image_ids": image_ids,
            "debugging_info": debugging_info or []
        }
        hash = cls.compute_hash_from_data(cls.hash_data_from_data(data))

        # Create complete data structure
        data["_id"] = hash
        data["type"] = "product"
        
        return cls(data)

    @classmethod
    def hash_data_from_data(cls, data: DataType) -> DataType:
        """
        Fields used to compute the hash for the product.

        Returns:
            dict: URL only.
        """
        return {
            "url": data["url"]
        }
    
    @property
    def images(self):
        """
        Lazily access loaded image objects.

        Raises:
            RuntimeError: If images have not been loaded.

        Returns:
            list: List of Image objects associated with this document.
        """
        if self._images is None:
            raise RuntimeError("Images not loaded. Call `load_images()` first.")
        return self._images

    def load_images(self, mongo_handler):
        """
        Load image documents from MongoDB based on stored image IDs.

        Args:
            mongo_handler: An instance of MongoDBManager for loading images.

        Returns:
            list: List of loaded Image objects.
        """
        from iris.document_types.image import Image  # avoid circular import
        image_docs = mongo_handler.find_images_by_ids(self.data.get("image_ids", []))
        self._images = [Image(doc) for doc in image_docs]
        return self._images

    def embed_text(self, embedder: Embedder) -> torch.Tensor:
        """
        Generate an embedding from the product's title and description.

        Args:
            embedder: An embedding model with `embed_text()` method.

        Returns:
            torch.Tensor: The product's embedding vector.
        """
        text = f"{self.data.get('title', '')} {self.data.get('description', '')}".strip()
        return embedder.embed_text(text)

    def embed(self, embedder: Embedder) -> torch.Tensor:
        """
        Generate an embedding from the product.

        Args:
            embedder: An embedding model with `embed_text()` method.

        Returns:
            torch.Tensor: The product's embedding vector.
        """
        embeddings = []
        text_embedding = self.embed_text(embedder)
        embeddings.append(text_embedding)

        # TODO: Add localisation embeddings
        raise NotImplementedError("Localization embeddings not implemented yet.")

        embedding = torch.stack(embeddings).mean(dim=0)

        return embedding
