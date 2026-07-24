"""guidance/packet_cache.py – Opt-in disk cache for ORIGINAL-frame semantic packets.

Video evaluation re-extracts a caption/object/scene packet for every ORIGINAL
frame on every run (BLIP2 caption + CLIP object/scene probing when no
``--captions`` file is given, or CLIP-only probing when one is). For a fixed
video + fixed extraction config, that packet is a pure function of the frame
pixels and the extractor settings — re-running the same evaluation (e.g. a
diffusion-step sweep that only changes the RECONSTRUCTION side) recomputes it
identically every time. This module memoizes it to disk, per video.

Deliberately NOT covered
-------------------------
Reconstructed-frame packets are never cached across runs here: the
reconstruction is the very thing an experiment is varying (diffusion_step,
force_interframe_reuse, SNR, ...) and, for the real diffusion sampler, is not
even deterministic run-to-run. Caching it would silently make different runs
report on stale, unrelated reconstructions. (Within a single run, the
existing in-memory keyframe-reuse path in video/temporal_pipeline.py already
avoids redundant recon-packet extraction for reused frames — that is a
correctness-preserving reuse, not a cross-run cache.)

Cache key / invalidation
-------------------------
One JSON file per video: ``<cache_dir>/<video_stem>.json``, holding a ``meta``
header plus a ``packets`` map keyed by frame_id. On load, every field in
``meta`` is compared against the current run's key; any mismatch (different
video file, different mtime/size because the source was re-exported,
different caption source, different CLIP model, different packet extractor
settings, different packet schema version) invalidates the WHOLE file rather
than trying to patch it entry-by-entry, so a stale cache can never silently
mix packets built under two different configs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from sgdjscc_lab.utils.packet_io import PACKET_VERSION

logger = logging.getLogger(__name__)


def build_meta(
    video_path,
    *,
    caption_source: str,
    clip_model_name: Optional[str],
    packet_caption_objects: bool,
) -> Dict:
    """Build the cache-invalidation key for one video + extraction config.

    caption_source:
        A short label describing where original-frame captions come from —
        e.g. ``"captions:<path>"`` (a --captions file/dir) or ``"blip2"``
        (extracted per frame). Different sources must never share a cache
        entry even if they happen to produce the same string.
    """
    p = Path(video_path)
    st = p.stat()
    return {
        "packet_version": PACKET_VERSION,
        "video_path": str(p.resolve()),
        "video_mtime_ns": st.st_mtime_ns,
        "video_size": st.st_size,
        "caption_source": caption_source,
        "clip_model_name": clip_model_name,
        "packet_caption_objects": bool(packet_caption_objects),
    }


def cache_path_for(cache_dir, video_path) -> Path:
    return Path(cache_dir) / f"{Path(video_path).stem}.json"


class PacketCache:
    """Load-on-construct, save-on-demand original-frame packet cache.

    Usage::

        cache = PacketCache(cache_dir, video_path, meta)
        pkt = cache.get(frame_id)
        if pkt is None:
            pkt = extractor.extract(...)
            cache.put(frame_id, pkt)
        ...
        cache.save()  # once at the end of the run
    """

    def __init__(self, cache_dir, video_path, meta: Dict) -> None:
        self.path = cache_path_for(cache_dir, video_path)
        self.meta = meta
        self._packets: Dict[str, Dict] = {}
        self._dirty = False
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Packet cache unreadable (%s), starting fresh: %s", self.path, exc)
            return
        if data.get("meta") != self.meta:
            logger.info(
                "Packet cache %s built under a different config — ignoring "
                "(will be overwritten on save).", self.path,
            )
            return
        self._packets = dict(data.get("packets", {}))
        logger.info("Packet cache hit: %d cached original-frame packet(s) from %s",
                     len(self._packets), self.path)

    def get(self, frame_id: str) -> Optional[Dict]:
        pkt = self._packets.get(str(frame_id))
        if pkt is not None:
            self._hits += 1
        else:
            self._misses += 1
        return pkt

    def put(self, frame_id: str, packet: Dict) -> None:
        self._packets[str(frame_id)] = packet
        self._dirty = True

    @property
    def stats(self) -> Dict:
        return {"hits": self._hits, "misses": self._misses, "cached_total": len(self._packets)}

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Process-unique temp filename: the same video (and therefore the
        # same cache file path) can legitimately be processed by several
        # concurrent processes at once — e.g. different modes/diffusion-step
        # sweeps or motion-threshold values all reading the same ORIGINAL
        # frames. A fixed ".json.tmp" name would let two processes' writes
        # interleave on the same temp file before either rename()s it
        # (write/write or write/rename races); a pid-suffixed name gives each
        # process its own temp file, so only the final os.replace() (atomic)
        # is ever contended, and that just becomes a last-writer-wins on the
        # cache file itself — never a corrupted intermediate file. (The
        # packets each process computes for a given frame_id are a pure
        # function of the frame + extraction config, so a "last writer wins"
        # outcome is at worst a missed cache entry, never wrong content.)
        tmp = self.path.with_suffix(f".json.tmp{os.getpid()}")
        tmp.write_text(
            json.dumps({"meta": self.meta, "packets": self._packets}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)
        logger.info("Packet cache → %s (%d packet(s), %s)", self.path, len(self._packets), self.stats)
