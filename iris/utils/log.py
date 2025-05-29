import logging

logger = logging.getLogger("iris")
logger.setLevel(logging.INFO)  # Default: suppress DEBUG unless explicitly enabled

# Avoid duplicate handlers if this module is imported multiple times
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
