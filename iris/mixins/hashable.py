import hashlib
from abc import ABC

class HashableMixin(ABC):
    @property
    def hash_data(self) -> dict[str, any]:
        """
        Returns the dictionary of fields used to compute the hash.

        By default, this excludes volatile or identity-irrelevant fields like
        '_id', 'hash', and 'created_at'. Subclasses can override this property
        to define a custom identity scope.

        Returns:
            dict: Canonical content used for hash computation.
        """
        if not hasattr(self, "data"):
            raise AttributeError("Class using HashableMixin must define a `data` attribute.")

        exclude = {"_id", "hash", "created_at"}
        return {k: v for k, v in self.data.items() if k not in exclude}

    def compute_hash(self) -> str:
        """
        Compute a stable hash of the document's identity-defining fields.

        This method sorts and stringifies the `hash_data` content, then returns
        an MD5 hash of that canonical form.

        Returns:
            str: MD5 hex digest representing the content hash.
        """
        canonical = str(sorted(self.hash_data.items())).encode("utf-8")
        return hashlib.md5(canonical).hexdigest()
