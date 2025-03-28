import hashlib


def get_url_hash(url: str) -> str:
    """
    Generate a deterministic hash for a URL.

    Args:
        url (str): The URL to hash.

    Returns:
        str: A hash of the URL.
    """
    return hashlib.md5(url.encode()).hexdigest()
