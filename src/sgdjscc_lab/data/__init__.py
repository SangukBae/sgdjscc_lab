"""sgdjscc_lab.data – Dataset and DataLoader helpers for training.

Legacy flat-folder helpers (``ImageFolderDataset``, ``build_dataloader``) remain
for the original training scaffold; the stage-aware datasets
(``ImageOnlyDataset``, ``TextImageDataset``, ``TextImageEdgeDataset``) back the
3-stage SGD-JSCC training procedure.
"""
from sgdjscc_lab.data.image_dataset import ImageFolderDataset, build_dataloader
from sgdjscc_lab.data.transforms import ImageTransform, build_transform
from sgdjscc_lab.data.datasets import (
    ImageOnlyDataset,
    TextImageDataset,
    TextImageEdgeDataset,
    build_dataset_for_stage,
    build_dataloader_for_stage,
    collate_stage_batch,
)

__all__ = [
    "ImageFolderDataset",
    "build_dataloader",
    "ImageTransform",
    "build_transform",
    "ImageOnlyDataset",
    "TextImageDataset",
    "TextImageEdgeDataset",
    "build_dataset_for_stage",
    "build_dataloader_for_stage",
    "collate_stage_batch",
]
