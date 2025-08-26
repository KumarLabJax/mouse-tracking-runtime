import hashlib
from pathlib import Path


def hash_file(file: Path) -> str:
    """Return hash of file.

    Args:
            file: path to file to hash

    Returns:
            blake2b hash of file
    """
    chunk_size = 8192
    with file.open("rb") as f:
        h = hashlib.blake2b(digest_size=20)
        c = f.read(chunk_size)
        while c:
            h.update(c)
            c = f.read(chunk_size)
    return h.hexdigest()
