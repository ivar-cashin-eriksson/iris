from PIL import Image as PILImage
from abc import ABC, abstractmethod

from iris.protocols.context_protocols import HasImageContext

class RenderableMixin(ABC):
    """
    Mixin class to provide a render method for objects.
    """

    @abstractmethod
    def render(self, context: HasImageContext, **kwargs) -> PILImage.Image:
        """
        Render the object to a string representation.

        Returns:
            str: The rendered string representation of the object.
        """
        ...
        