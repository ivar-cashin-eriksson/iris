from iris.models.document import Document, DataType
from iris.models.product import Product
from iris.models.image import Image
from iris.models.localization import Localization

def factory(data: DataType) -> Document:
    """
    Factory function to create a Document instance from raw data.

    This function determines the appropriate subclass of Document based on
    the data structure and returns an instance of that subclass.

    Args:
        data (dict): The raw data to convert into a Document.

    Returns:
        Document: An instance of a specific Document subclass.
    """
    match data.get("type"):
        case "product":
            return Product(data)
        case "image":
            return Image(data)
        case "localization":
            raise NotImplementedError(
                "Localization documents are not yet implemented in the factory."
            )
        case _:
            raise ValueError(f"Unknown document type: {data.get('type')}")