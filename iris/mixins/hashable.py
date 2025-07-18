import hashlib
from typing import TypeAlias
from iris.utils.log import logger

DataType: TypeAlias = dict[str, any]  # TODO: Import from iris.models.document instead

class HashableMixin:
    """
    Mixin for computing a stable, content-based hash from document fields.

    This mixin provides a default method for generating an MD5 hash from a
    subset of the document's data.
    """
    
    @property
    def hash_data(self) -> DataType:
        """
        Returns the dictionary of fields used to compute the hash.

        Returns:
            dict: Canonical content used for hash computation.
        """
        ...

    @classmethod
    def compute_hash_from_data(cls, hash_data: DataType) -> str:
        """
        Compute a stable hash from provided DataType instance.

        This method sorts and stringifies the `hash_data` content, then returns
        an MD5 hash of that canonical form.

        Returns:
            str: MD5 hex digest representing the content hash.
        """        
        canonical = str(sorted(hash_data.items())).encode("utf-8")
        hash_ = hashlib.md5(canonical).hexdigest()

        logger.debug(f"Computed hash {hash_} from fields: {list(hash_data.keys())}")
        return hash_
    
    def compute_hash(self) -> str:
        """
        Compute a stable hash of the document's identity-defining fields.

        This method uses the `hash_data` property to get the relevant fields.

        Returns:
            str: MD5 hex digest representing the content hash.
        """
        return self.compute_hash_from_data(self.hash_data)
