from typing import Union
import torch
from pycocotools import mask as maskUtils
import numpy as np

def get_device(device_str: str) -> torch.device:
    """Convert a string to a torch.device.
    
    Args:
        device_str: String representing the device ('cuda', 'cpu', or 'mps')
        
    Returns:
        torch.device: The appropriate device
        
    Raises:
        ValueError: If device_str is not one of the supported devices
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Device {device_str} is not available")
    

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