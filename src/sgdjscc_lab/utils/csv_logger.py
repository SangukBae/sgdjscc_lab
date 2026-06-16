"""utils/csv_logger.py – Streaming CSV writer for evaluation results.

Supports both write-once and append modes so that long SNR sweeps can
write results incrementally (crash-safe) rather than accumulating in memory.

Usage
-----
>>> with CSVLogger("results.csv", fieldnames=["filename", "snr_db", "psnr"]) as log:
...     log.write_row({"filename": "img.png", "snr_db": 10, "psnr": 32.4})
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CSVLogger:
    """Streaming CSV writer that handles header creation automatically.

    Parameters
    ----------
    path:
        Output file path.  Parent directories are created automatically.
    fieldnames:
        Column names in the order they appear in the CSV header.
    append:
        If True and the file already exists, new rows are appended without
        rewriting the header.  If False (default), the file is overwritten.
    encoding:
        File encoding (default ``'utf-8'``).
    """

    def __init__(
        self,
        path: str | Path,
        fieldnames: List[str],
        append: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.append = append
        self.encoding = encoding
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def _open(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if (self.append and file_exists) else "w"

        self._file = open(self.path, mode, newline="", encoding=self.encoding)
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.fieldnames,
            extrasaction="ignore",  # silently drop unexpected keys
        )
        if mode == "w" or not file_exists:
            self._writer.writeheader()
        logger.debug("CSVLogger opened (%s): %s", mode, self.path)

    def write_row(self, row: Dict) -> None:
        """Write a single row dict.  Opens the file lazily on first call."""
        self._open()
        self._writer.writerow(row)
        self._file.flush()

    def write_rows(self, rows: List[Dict]) -> None:
        """Write a list of row dicts."""
        for row in rows:
            self.write_row(row)

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
            logger.info("CSVLogger closed: %s", self.path)

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Column schema
# ─────────────────────────────────────────────────────────────────────────────

#: Ordered list of CSV columns for the standard evaluation output.
RESULT_COLUMNS: List[str] = [
    "filename",
    "snr_db",
    "psnr",
    "ssim",
    "lpips",
    "clip_image_image",
    "clip_text_image",
    "object_preservation_rate",
    "missing_object_rate",
    "additional_object_rate",
    "hallucination_score",
    "semantic_reliability_score",
    # FID is a dataset/SNR-level metric (paper §VI): the same value is filled into
    # every row of an SNR group (None when the metric is disabled / backend absent).
    "fid",
    # Which FID feature backend produced `fid`: "inception" (true Inception-FID),
    # "proxy" (injected non-Inception feature_fn — NOT comparable to paper FID),
    # "unavailable" / "" (no backend). Persisted so results files are unambiguous.
    "fid_backend",
]

#: Additional columns emitted when packet-aware evaluation is enabled
#: (Phase 4-A).  Appended to RESULT_COLUMNS so legacy rows are unaffected;
#: extra row keys are silently dropped by CSVLogger when not in the header.
PACKET_RESULT_COLUMNS: List[str] = [
    "srs_base",
    "srs_packet",
    "srs_v2",
    "object_match_rate",
    "relation_consistency",
    "attribute_consistency",
    "segmentation_consistency",
    "scene_match",
    "missing_object_count",
    "additional_object_count",
    "relation_error_count",
    "attribute_error_count",
    "guidance_regime",
    "regeneration_strategy",
]

#: Full column set for packet-aware evaluation runs.
PACKET_RESULT_COLUMNS_FULL: List[str] = RESULT_COLUMNS + PACKET_RESULT_COLUMNS
