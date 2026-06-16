"""utils/packet_io.py – Semantic packet (JSON) serialisation helpers (Phase 4-A).

A *semantic packet* is the unified, channel-independent description of a frame's
semantic content (caption, objects, scene, relations, attributes, and structural
summaries).  Phase 4-A does **not** transmit the packet over the wireless
channel; it only serialises it beside each reconstructed image so the
packet-aware verifier (``evaluators/semantic_packet_matcher.py``) can compare the
original-frame packet against the reconstructed-frame packet.

Design notes
------------
- Packets are plain ``dict`` objects containing only JSON-native types
  (str / int / float / bool / list / dict / None).  ``to_jsonable`` converts any
  stray numpy / torch scalars to Python primitives so ``json.dump`` never fails.
- The on-disk layout next to a reconstructed image ``<stem>.png`` is::

      <out_dir>/<stem>.packet.json        # reconstructed-frame packet
      <out_dir>/<stem>.orig_packet.json    # original-frame packet (optional)
      <out_dir>/<stem>.error_report.json   # packet matcher output (optional)

  The helper functions below centralise these naming conventions so the rest of
  the package never hard-codes suffixes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Packet schema version – bump when the packet dict layout changes.
PACKET_VERSION = "phase4-a.1"

# Canonical on-disk suffixes.
PACKET_SUFFIX = ".packet.json"
ORIG_PACKET_SUFFIX = ".orig_packet.json"
ERROR_REPORT_SUFFIX = ".error_report.json"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into JSON-native Python types.

    Handles numpy scalars/arrays and torch tensors by falling back to ``.item()``
    / ``.tolist()`` so that packets built from model outputs serialise cleanly.
    """
    # Fast path for the common primitives.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # numpy / torch scalars expose .item(); arrays/tensors expose .tolist().
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:  # noqa: BLE001 – not a 0-d scalar
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:  # noqa: BLE001
            pass

    # Last resort: stringify so serialisation never crashes a batch.
    return str(obj)


def save_packet(packet: Dict[str, Any], path: str | Path) -> Path:
    """Serialise *packet* to *path* as pretty-printed JSON. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(packet), fh, indent=2, ensure_ascii=False)
    logger.debug("Saved packet → %s", path)
    return path


def load_packet(path: str | Path) -> Dict[str, Any]:
    """Load a packet JSON file into a dict."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── On-disk naming helpers ────────────────────────────────────────────────────

def packet_path(out_dir: str | Path, stem: str) -> Path:
    """Return the reconstructed-frame packet path for *stem* under *out_dir*."""
    return Path(out_dir) / f"{stem}{PACKET_SUFFIX}"


def orig_packet_path(out_dir: str | Path, stem: str) -> Path:
    """Return the original-frame packet path for *stem* under *out_dir*."""
    return Path(out_dir) / f"{stem}{ORIG_PACKET_SUFFIX}"


def error_report_path(out_dir: str | Path, stem: str) -> Path:
    """Return the error-report path for *stem* under *out_dir*."""
    return Path(out_dir) / f"{stem}{ERROR_REPORT_SUFFIX}"


def save_error_report(report: Dict[str, Any], out_dir: str | Path, stem: str) -> Path:
    """Serialise a packet-matcher error report next to the reconstructed image."""
    return save_packet(report, error_report_path(out_dir, stem))
