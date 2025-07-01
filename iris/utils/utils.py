import json
import os
from glob import glob
from pathlib import Path
from typing import Union
from urllib.parse import urlparse, urlunparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools import mask
from tqdm import tqdm


def normalize_image_url(url: str) -> str:
    """
    Normalize image URL by removing query parameters that don't affect the core image.
    This ensures consistent storage and prevents duplicates for the same image 
    with different parameters like imwidth, imheight, quality, etc.
    
    Args:
        url (str): The original image URL
        
    Returns:
        str: The normalized URL without query parameters
        
    Examples:
        >>> normalize_image_url("https://media.cos.com/image.jpg?imwidth=657")
        "https://media.cos.com/image.jpg"
        >>> normalize_image_url("https://example.com/photo.png?width=800&height=600")
        "https://example.com/photo.png"
    """
    if not url:
        return url
        
    parsed = urlparse(url)
    # Remove query parameters - keeping only the base URL
    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc, 
            parsed.path,
            '',  # params
            '',  # query - removed
            ''   # fragment
        )
    )
    return normalized
