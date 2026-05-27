"""preprocessing.py – Phase 2 compatibility shim.

Image preprocessing utilities have moved to:
  utils/preprocessing.py

This module re-exports the public API so that any code still importing from
``sgdjscc_lab.preprocessing`` continues to work without modification.
"""

from sgdjscc_lab.utils.preprocessing import (  # noqa: F401
    preprocess_image,
    prepare_patches,
    split_patches,
    merge_patches,
)
