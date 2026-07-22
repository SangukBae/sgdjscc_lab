"""tests/test_heldout_remeasurement.py – held-out remeasurement pipeline tests (ETRI 5차, step 9)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet  # noqa: E402


def _item(item_id, ref_objs, recon_objs, role="keyframe", with_image=True):
    from sgdjscc_lab.pipelines.heldout_remeasurement import RemeasurementItem
    return RemeasurementItem(
        item_id=item_id,
        reference_packet=build_packet(objects=ref_objs, scene="s"),
        reconstructed_packet=build_packet(objects=recon_objs, scene="s"),
        reconstructed_image=(torch.rand(1, 3, 4, 4) if with_image else None),
        role=role,
    )


class TestItemsFromTemporalRecords:
    def test_builds_items_from_frame_records(self):
        from sgdjscc_lab.video.temporal_pipeline import FrameRecord
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_temporal_records

        recs = [
            FrameRecord(index=0, role="keyframe",
                        orig_packet=build_packet(objects=["car"]),
                        recon_packet=build_packet(objects=["car"]),
                        recon=torch.rand(1, 3, 4, 4)),
        ]
        items = items_from_temporal_records(recs)
        assert len(items) == 1
        assert items[0].item_id == 0
        assert items[0].role == "keyframe"
        assert items[0].reconstructed_image is not None


class TestItemsFromSavedPackets:
    def test_loads_saved_packet_pairs(self, tmp_path):
        from sgdjscc_lab.utils.packet_io import save_packet, orig_packet_path, packet_path
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_saved_packets

        save_packet(build_packet(objects=["car", "dog"]), orig_packet_path(tmp_path, "f0"))
        save_packet(build_packet(objects=["car"]), packet_path(tmp_path, "f0"))

        pairs = [("f0", str(orig_packet_path(tmp_path, "f0")), str(packet_path(tmp_path, "f0")), "keyframe")]
        items = items_from_saved_packets(pairs)
        assert len(items) == 1
        assert items[0].reference_packet["objects"] == ["car", "dog"]
        assert items[0].reconstructed_packet["objects"] == ["car"]
        assert items[0].reconstructed_image is None   # no pixels saved alongside packets


class TestRemeasureWithoutCalibrator:
    def test_calibrated_equals_clip_only_when_no_calibrator(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure

        items = [
            _item(0, ["car"], ["car"], role="keyframe"),
            _item(1, ["car", "dog"], ["car"], role="inter"),
        ]
        out = remeasure(items, presence_calibrator=None)
        assert out["clip_only"]["metrics"] == out["calibrated"]["metrics"]
        for k, v in out["metric_delta"].items():
            if k.endswith("_diff"):
                assert v == 0 or v is None

    def test_rows_tagged_held_out(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure

        out = remeasure([_item(0, ["car"], ["car"])], presence_calibrator=None)
        assert out["clip_only"]["rows"][0]["metric_role"] == "held_out"
        assert out["calibrated"]["rows"][0]["metric_role"] == "held_out"


class TestRemeasureWithCalibrator:
    def test_calibration_corrects_missing_object(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceResult
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        class AlwaysPresent(PresenceBackend):
            backend_name = "stub"
            def check(self, object_name, image=None, packet=None, gt_metadata=None):
                return PresenceResult(object_name=object_name, present=True, confidence=0.99, backend="stub")

        cal = PresenceCalibrator({"clip": AlwaysPresent()}, mode="clip_only")
        items = [_item(0, ["car", "dog"], ["car"], role="keyframe")]
        out = remeasure(items, presence_calibrator=cal)

        assert out["clip_only"]["rows"][0]["missing_objects"] == ["dog"]
        assert out["calibrated"]["rows"][0]["missing_objects"] == []
        assert out["clip_only"]["metrics"]["mean_severity"] > out["calibrated"]["metrics"]["mean_severity"]
        assert out["metric_delta"]["mean_severity_diff"] < 0

    def test_image_required_backend_without_image_falls_back_to_clip_only(self):
        """A calibrator backend that legitimately NEEDS an image (mirrors
        ClipPresenceBackend) can't do anything useful for --from-packets
        items (no pixels saved) — the row must stay identical to clip_only."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceBackendUnavailableError
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        class NeedsImage(PresenceBackend):
            backend_name = "clip"
            def check(self, object_name, image=None, packet=None, gt_metadata=None):
                if image is None:
                    raise PresenceBackendUnavailableError("needs an image")
                raise AssertionError("should never be reached in this test")

        cal = PresenceCalibrator({"clip": NeedsImage()}, mode="clip_only")
        items = [_item(0, ["car", "dog"], ["car"], with_image=False)]
        out = remeasure(items, presence_calibrator=cal)
        assert out["calibrated"]["rows"][0]["missing_objects"] == ["dog"]
        assert out["calibrated"]["rows"][0]["calibrated_presence_result"] is None

    def test_image_free_backend_calibrates_without_image(self):
        """Mock/gt-style (image-free) backends must be able to calibrate
        --from-packets items even though those items have no
        reconstructed_image — this is the whole point of supporting saved
        packets without pixels."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceResult
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        class AlwaysPresent(PresenceBackend):
            backend_name = "stub"
            def check(self, object_name, image=None, packet=None, gt_metadata=None):
                return PresenceResult(object_name=object_name, present=True, confidence=0.99, backend="stub")

        cal = PresenceCalibrator({"clip": AlwaysPresent()}, mode="clip_only")
        items = [_item(0, ["car", "dog"], ["car"], with_image=False)]
        out = remeasure(items, presence_calibrator=cal)
        assert out["calibrated"]["rows"][0]["missing_objects"] == []
        assert out["clip_only"]["rows"][0]["missing_objects"] == ["dog"]
        assert out["calibrated"]["rows"][0]["calibrated_presence_result"] is not None

    def test_gt_metadata_from_item_reaches_gt_backend(self):
        """RemeasurementItem.gt_metadata must actually be forwarded into
        calibration (previously stored but unused) — proven end-to-end via
        the 'gt' backend with NO default metadata of its own."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import RemeasurementItem, remeasure
        from sgdjscc_lab.evaluators.presence_backends import GtPresenceBackend
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        cal = PresenceCalibrator({"gt": GtPresenceBackend()}, mode="gt_only")
        item = RemeasurementItem(
            item_id=0,
            reference_packet=build_packet(objects=["car", "dog"], scene="s"),
            reconstructed_packet=build_packet(objects=["car"], scene="s"),
            reconstructed_image=None,
            gt_metadata={"dog": True},   # GT says dog IS actually present
        )
        out = remeasure([item], presence_calibrator=cal)
        assert out["clip_only"]["rows"][0]["missing_objects"] == ["dog"]
        assert out["calibrated"]["rows"][0]["missing_objects"] == []

    def test_saved_packets_with_gt_metadata_end_to_end(self, tmp_path):
        """Full --from-packets-style loading (items_from_saved_packets) +
        gt_metadata_by_id + gt-only calibration, exercising the exact path
        scripts/remeasure_video_metrics.py's --from-packets mode uses."""
        from sgdjscc_lab.utils.packet_io import save_packet, orig_packet_path, packet_path
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_saved_packets, remeasure
        from sgdjscc_lab.evaluators.presence_backends import GtPresenceBackend
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        save_packet(build_packet(objects=["car", "dog"]), orig_packet_path(tmp_path, "f0"))
        save_packet(build_packet(objects=["car"]), packet_path(tmp_path, "f0"))
        pairs = [("f0", str(orig_packet_path(tmp_path, "f0")), str(packet_path(tmp_path, "f0")))]
        items = items_from_saved_packets(pairs, gt_metadata_by_id={"f0": {"dog": True}})
        assert items[0].reconstructed_image is None

        cal = PresenceCalibrator({"gt": GtPresenceBackend()}, mode="gt_only")
        out = remeasure(items, presence_calibrator=cal)
        assert out["clip_only"]["rows"][0]["missing_objects"] == ["dog"]
        assert out["calibrated"]["rows"][0]["missing_objects"] == []


class TestWriteRemeasurement:
    def test_writes_requested_artefacts(self, tmp_path):
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure, write_remeasurement

        out = remeasure([_item(0, ["car"], ["car"])], presence_calibrator=None)
        write_remeasurement(
            out,
            clip_only_json=str(tmp_path / "clip.json"),
            clip_only_csv=str(tmp_path / "clip.csv"),
            calibrated_json=str(tmp_path / "cal.json"),
            calibrated_csv=str(tmp_path / "cal.csv"),
            metric_delta_json=str(tmp_path / "delta.json"),
            metric_delta_csv=str(tmp_path / "delta.csv"),
        )
        for name in ("clip.json", "clip.csv", "cal.json", "cal.csv", "delta.json", "delta.csv"):
            assert (tmp_path / name).exists()

        delta = json.loads((tmp_path / "delta.json").read_text(encoding="utf-8"))
        assert "note" in delta

    def test_skips_unrequested_artefacts(self, tmp_path):
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure, write_remeasurement

        out = remeasure([_item(0, ["car"], ["car"])], presence_calibrator=None)
        write_remeasurement(out, clip_only_json=str(tmp_path / "only_this.json"))
        assert (tmp_path / "only_this.json").exists()
        assert list(tmp_path.iterdir()) == [tmp_path / "only_this.json"]
