"""Content-hash utilities for reconstruction frame trees.

Used to prove pixel-level identity or difference between two reconstruction
runs independently of any *derived* signal (a detector score, a metric
value). A detector score match is not proof of pixel identity -- two
different images could in principle score identically -- so anywhere a claim
of "these outputs are bit-identical" matters (e.g. the G1 v1.2 effective-seed
finding, or invalidating an evaluator cache when input frames change), the
actual frame bytes should be hashed directly instead.

Pure stdlib (hashlib only); safe to import without torch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Union

PathLike = Union[str, Path]


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_tree_sha256(paths: Sequence[PathLike]) -> str:
    """Order-sensitive combined hash of a sequence of frame files.

    Hashes each file's content plus its basename, then folds the per-frame
    hashes together in the given order. Sensitive to content changes,
    reordering, and insertion/removal, without holding every frame's bytes
    in memory at once (each file is hashed and released before the next).

    Raises ``FileNotFoundError`` (propagated from the underlying open) if
    any path does not exist -- callers should not treat a missing frame as
    an empty/skippable contribution to the tree hash.
    """
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b":")
        digest.update(sha256_file(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
