"""diagnostics/ – float32 digital-quality diagnostic harness.

Compares, at the SAME (video, frame, seed), three Tx/Rx paths built entirely
from the project's real production sender/receiver code
(``pipelines/infer_pipeline.py``, ``transmission/*.py``):

  awgn               – existing production AWGN path (baseline).
  digital_inprocess  – ``DigitalPacketChannel(bit_depth=32)`` swapped into
                        ``jscc.channel_model``; sender output reaches the
                        receiver-side computation without crossing a
                        frame-level bundle byte boundary.
  digital_wire       – the real ``transmission.receiver_runtime`` byte
                        boundary: ``encode_frame_to_bundle_bytes`` →
                        ``reconstruct_frame_from_bundle_bytes``.

See ``docs/protocols/float32_digital_diagnostics.md`` for the full design
note and ``scripts/diagnose_float32_digital_quality.py`` for the CLI.
"""
