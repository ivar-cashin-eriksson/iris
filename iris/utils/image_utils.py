from typing import Union
from pycocotools import mask as maskUtils
import numpy as np
from glob import glob
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


def convert_image_format(
    image: Union[np.ndarray, Image.Image],
    target_format: str = "numpy",
    ensure_rgb: bool = False
) -> Union[np.ndarray, Image.Image]:
    """
    Convert an image between numpy array and PIL Image formats.
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_format: Desired output format ("numpy" or "pil")
        ensure_rgb: Whether to ensure the output is RGB (converts grayscale/RGBA)
        
    Returns:
        Image in the requested format (numpy array or PIL Image)
        
    Raises:
        ValueError: If target_format is not "numpy" or "pil"
    """
    if target_format not in ["numpy", "pil"]:
        raise ValueError('target_format must be either "numpy" or "pil"')
    
    # Handle RGB conversion if needed
    if ensure_rgb:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:  # numpy array
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image = image[..., :3]
    
    # Convert to target format if needed
    if target_format == "numpy" and isinstance(image, Image.Image):
        return np.array(image)
    elif target_format == "pil" and isinstance(image, np.ndarray):
        return Image.fromarray(image)
    
    return image


def convert_mask_format(
        mask: Union[np.ndarray, dict], 
        target_format: str
    ) -> Union[np.ndarray, dict]:
    """
    Convert between binary mask and RLE mask formats.

    Args:
        mask: Input mask (binary mask or RLE mask).
        target_format: Desired output format ('binary' or 'rle').

    Returns:
        Converted mask in the requested format.
    """
    if target_format == 'rle':
        if isinstance(mask, np.ndarray):  # Binary mask
            rle = maskUtils.encode(np.asfortranarray(mask))
            return rle
        elif isinstance(mask, dict) and 'counts' in mask:  # Already RLE
            return mask
        else:
            raise ValueError("Input mask must be a binary mask to convert to RLE.")
    
    elif target_format == 'binary':
        if isinstance(mask, dict) and 'counts' in mask:  # RLE mask
            binary_mask = maskUtils.decode(mask)
            return binary_mask
        elif isinstance(mask, np.ndarray):  # Already binary
            return mask
        else:
            raise ValueError("Input mask must be an RLE mask to convert to binary.")
    
    else:
        raise ValueError("Invalid target format. Use 'binary' or 'rle'.")
