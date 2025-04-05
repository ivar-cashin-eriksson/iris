"""
Utility functions for the data_pipeline subpackage.

These functions are intended for internal use within the data_pipeline
subpackage only.
"""

import hashlib


def _get_url_hash(url: str) -> str:
    """
    Generate a deterministic hash for a URL.

    Args:
        url (str): The URL to hash.

    Returns:
        str: A hash of the URL.
    """
    return hashlib.md5(url.encode()).hexdigest()
