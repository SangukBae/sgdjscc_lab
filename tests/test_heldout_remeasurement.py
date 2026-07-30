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

    def test_gt_metadata_key_mismatch_no_longer_silently_dropped(self, tmp_path):
        """Regression: items_from_saved_packets used to do a bare
        gt_metadata_by_id.get(item_id) — a gt_presence.json-style file keyed
        "frame_00000" would silently fail to match an integer/differently-
        typed item_id. _lookup_gt_metadata now bridges "frame_00000" <-> 0."""
        from sgdjscc_lab.utils.packet_io import save_packet, orig_packet_path, packet_path
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_saved_packets

        save_packet(build_packet(objects=["car", "dog"]), orig_packet_path(tmp_path, "f0"))
        save_packet(build_packet(objects=["car"]), packet_path(tmp_path, "f0"))
        pairs = [(0, str(orig_packet_path(tmp_path, "f0")), str(packet_path(tmp_path, "f0")))]

        items = items_from_saved_packets(pairs, gt_metadata_by_id={"frame_00000": {"dog": True}})
        assert items[0].gt_metadata == {"dog": True}


class TestLookupGtMetadata:
    """_lookup_gt_metadata: bridges the handful of key spellings different
    GT-file producers/consumers use for "the same" frame."""

    def test_exact_match(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import _lookup_gt_metadata
        assert _lookup_gt_metadata({0: {"car": True}}, 0) == {"car": True}
        assert _lookup_gt_metadata({"frame_00000": {"car": True}}, "frame_00000") == {"car": True}

    def test_int_item_id_matches_frame_prefixed_string_key(self):
        """The key shape scripts/run_etri_video_eval.py's gt_presence.json and
        convert_gt_to_presence() actually produce, looked up by the plain
        integer item_id items_from_recon_frame_dirs() uses."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import _lookup_gt_metadata
        gt = {"frame_00000": {"person": True}, "frame_00001": {"person": False}}
        assert _lookup_gt_metadata(gt, 0) == {"person": True}
        assert _lookup_gt_metadata(gt, 1) == {"person": False}

    def test_frame_prefixed_item_id_matches_bare_int_string_key(self):
        """The reverse direction: a hand-written {"0": {...}} file (JSON object
        keys are always strings) looked up by a "frame_00000"-style item_id."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import _lookup_gt_metadata
        gt = {"0": {"person": True}}
        assert _lookup_gt_metadata(gt, "frame_00000") == {"person": True}

    def test_no_match_returns_none(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import _lookup_gt_metadata
        assert _lookup_gt_metadata({"frame_00000": {"person": True}}, "frame_00099") is None
        assert _lookup_gt_metadata({}, 0) is None


class TestConvertGtToPresence:
    def test_converts_segment_level_gt(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import convert_gt_to_presence
        gt = {
            "video_id": "01_toy", "n_frames": 3,
            "segments": [
                {"start_frame": 0, "end_frame": 1,
                 "objects": [{"label": "person", "count": 1, "presence": "visible"}]},
                {"start_frame": 2, "end_frame": 2,
                 "objects": [{"label": "person", "count": 1, "presence": "absent"},
                            {"label": "car", "count": 1, "presence": "visible"}]},
            ],
        }
        presence = convert_gt_to_presence(gt)
        assert set(presence) == {f"frame_{i:05d}" for i in range(3)}
        assert presence["frame_00000"] == {"car": False, "person": True}
        assert presence["frame_00002"] == {"car": True, "person": False}

    def test_looks_like_segment_level_gt(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import looks_like_segment_level_gt
        assert looks_like_segment_level_gt({"n_frames": 3, "segments": []}) is True
        assert looks_like_segment_level_gt({"frame_00000": {"person": True}}) is False
        assert looks_like_segment_level_gt({}) is False
        assert looks_like_segment_level_gt([1, 2, 3]) is False


class _StubPacketExtractor:
    """Weight-free duck-typed stand-in for SemanticPacketExtractor — records
    every (frame_id, caption) it was called with so tests can assert on the
    exact calls items_from_recon_frame_dirs makes."""

    def __init__(self):
        self.calls = []

    def extract(self, image, frame_id=None, caption=None):
        self.calls.append((frame_id, caption))
        objs = ["car"] if frame_id and frame_id.startswith("recon_") else ["car", "dog"]
        return build_packet(objects=objs, caption=caption or "", frame_id=frame_id)


def _make_frame_dir(root: Path, prefix: str, n: int, color_offset: int = 0) -> Path:
    from PIL import Image
    d = root / prefix
    d.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (4, 4), color=(color_offset + i, 0, 0)).save(d / f"{prefix}_{i:05d}.png")
    return d


class TestItemsFromReconFrameDirs:
    """items_from_recon_frame_dirs: reuses a completed run's saved
    extracted_frames/ + recon_frames/ images (byte-identical reconstruction)
    without re-running any model, re-extracting packets via the injected
    packet_extractor (see the function's docstring for the fidelity split)."""

    def test_builds_items_with_real_recon_image(self, tmp_path):
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs

        extracted_dir = _make_frame_dir(tmp_path, "frame", 2, color_offset=0)
        recon_dir = _make_frame_dir(tmp_path, "recon", 2, color_offset=100)
        extractor = _StubPacketExtractor()

        items = items_from_recon_frame_dirs(extracted_dir, recon_dir, extractor)
        assert len(items) == 2
        assert items[0].item_id == 0
        assert items[0].reconstructed_image is not None
        assert items[0].reconstructed_image.shape[-2:] == (4, 4)
        assert items[0].reference_packet["objects"] == ["car", "dog"]
        assert items[0].reconstructed_packet["objects"] == ["car"]

    def test_frame_count_mismatch_raises(self, tmp_path):
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs

        extracted_dir = _make_frame_dir(tmp_path, "frame", 3)
        recon_dir = _make_frame_dir(tmp_path, "recon", 2)
        with pytest.raises(ValueError, match="Frame count mismatch"):
            items_from_recon_frame_dirs(extracted_dir, recon_dir, _StubPacketExtractor())

    def test_captions_go_to_reference_only_recon_always_none(self, tmp_path):
        """Reconstructed-frame packets always get caption=None passed to the
        extractor (the fidelity note's documented simplification) — only the
        reference/original frame's caption comes from *captions*."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs

        extracted_dir = _make_frame_dir(tmp_path, "frame", 2)
        recon_dir = _make_frame_dir(tmp_path, "recon", 2)
        extractor = _StubPacketExtractor()

        items_from_recon_frame_dirs(extracted_dir, recon_dir, extractor,
                                    captions=["a car", "a dog"])
        ref_calls = {fid: cap for fid, cap in extractor.calls if fid.startswith("frame_")}
        recon_calls = {fid: cap for fid, cap in extractor.calls if fid.startswith("recon_")}
        assert ref_calls == {"frame_00000": "a car", "frame_00001": "a dog"}
        assert recon_calls == {"recon_00000": None, "recon_00001": None}

    def test_roles_passed_through_for_sdi(self, tmp_path):
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs

        extracted_dir = _make_frame_dir(tmp_path, "frame", 2)
        recon_dir = _make_frame_dir(tmp_path, "recon", 2)
        items = items_from_recon_frame_dirs(
            extracted_dir, recon_dir, _StubPacketExtractor(), roles=["keyframe", "inter"],
        )
        assert [it.role for it in items] == ["keyframe", "inter"]

    def test_no_roles_defaults_to_none(self, tmp_path):
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs

        extracted_dir = _make_frame_dir(tmp_path, "frame", 1)
        recon_dir = _make_frame_dir(tmp_path, "recon", 1)
        items = items_from_recon_frame_dirs(extracted_dir, recon_dir, _StubPacketExtractor())
        assert items[0].role is None

    def test_gt_metadata_by_id_keyed_by_index(self, tmp_path):
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs

        extracted_dir = _make_frame_dir(tmp_path, "frame", 2)
        recon_dir = _make_frame_dir(tmp_path, "recon", 2)
        items = items_from_recon_frame_dirs(
            extracted_dir, recon_dir, _StubPacketExtractor(),
            gt_metadata_by_id={0: {"dog": True}},
        )
        assert items[0].gt_metadata == {"dog": True}
        assert items[1].gt_metadata is None

    def test_gt_metadata_accepts_frame_prefixed_string_keys(self, tmp_path):
        """This is the exact scenario the docs point operators at: reusing
        scripts/run_etri_video_eval.py's saved gt_presence.json ("frame_00000"
        string keys) directly as --gt-metadata for --from-recon-frames, whose
        item_id is a plain integer index."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs

        extracted_dir = _make_frame_dir(tmp_path, "frame", 2)
        recon_dir = _make_frame_dir(tmp_path, "recon", 2)
        items = items_from_recon_frame_dirs(
            extracted_dir, recon_dir, _StubPacketExtractor(),
            gt_metadata_by_id={"frame_00000": {"dog": True}, "frame_00001": {"dog": False}},
        )
        assert items[0].gt_metadata == {"dog": True}
        assert items[1].gt_metadata == {"dog": False}

    def test_real_image_enables_image_based_calibration_end_to_end(self, tmp_path):
        """The whole point of this loader vs items_from_saved_packets: a real
        reconstructed_image tensor lets an image-based presence backend
        (owlv2/vqa/clip) actually run instead of reporting itself
        unavailable. Proven here with a stub backend standing in for a real
        one (no weights needed to prove the wiring)."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs, remeasure
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceResult
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        class RequiresImage(PresenceBackend):
            backend_name = "owlv2"
            def check(self, object_name, image=None, packet=None, gt_metadata=None):
                assert image is not None, "expected a real image tensor, got None"
                return PresenceResult(object_name=object_name, present=True, confidence=0.9, backend="owlv2")

        extracted_dir = _make_frame_dir(tmp_path, "frame", 1)
        recon_dir = _make_frame_dir(tmp_path, "recon", 1)
        items = items_from_recon_frame_dirs(extracted_dir, recon_dir, _StubPacketExtractor())

        cal = PresenceCalibrator({"owlv2": RequiresImage()}, mode="owlv2_only")
        out = remeasure(items, presence_calibrator=cal)
        # reference has ["car","dog"], reconstructed (stub) has ["car"] → "dog"
        # is raw-missing, but the (stub) owlv2 backend says everything is present.
        assert out["clip_only"]["rows"][0]["missing_objects"] == ["dog"]
        assert out["calibrated"]["rows"][0]["missing_objects"] == []


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

    def test_csv_header_unchanged_without_filter(self, tmp_path):
        """Backward-compat: with no object_vocabulary_filter, the row CSV's
        header must be byte-identical to before this feature existed — no
        filtered_* columns appear at all."""
        import csv as _csv
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure, write_remeasurement

        out = remeasure([_item(0, ["car"], ["car"])], presence_calibrator=None)
        write_remeasurement(out, clip_only_csv=str(tmp_path / "clip.csv"))
        with open(tmp_path / "clip.csv", newline="", encoding="utf-8") as fh:
            header = next(_csv.reader(fh))
        assert "filtered_objects" not in header
        assert "filtered_missing_objects" not in header
        assert "filtered_additional_objects" not in header

    def test_csv_gains_filtered_columns_when_filter_enabled(self, tmp_path):
        import csv as _csv
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure, write_remeasurement
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter

        items = [_item(0, ["person", "one"], ["person"])]
        out = remeasure(items, presence_calibrator=None, object_vocabulary_filter=ObjectVocabularyFilter(enabled=True))
        write_remeasurement(out, clip_only_csv=str(tmp_path / "clip.csv"))
        with open(tmp_path / "clip.csv", newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        assert "filtered_objects" in rows[0]
        assert rows[0]["filtered_objects"] == '["one"]'


# ─────────────────────────────────────────────────────────────────────────────
# evaluators/object_vocabulary_filter.py — unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectVocabularyFilterUnit:
    def test_disabled_is_passthrough(self):
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        f = ObjectVocabularyFilter(enabled=False)
        kept, removed = f.filter_objects(["person", "one", "walking", "sidewalk"])
        assert kept == ["person", "one", "walking", "sidewalk"]   # order preserved, nothing removed
        assert removed == []

    def test_count_action_scene_terms_excluded(self):
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        f = ObjectVocabularyFilter(enabled=True)
        kept, removed = f.filter_objects(["person", "one", "walking", "sidewalk"])
        assert kept == ["person"]
        assert set(removed) == {"one", "walking", "sidewalk"}

    def test_real_object_noun_kept(self):
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        f = ObjectVocabularyFilter(enabled=True)
        kept, removed = f.filter_objects(["person", "car", "dog"])
        assert kept == ["car", "dog", "person"]
        assert removed == []

    def test_numeric_tokens_excluded(self):
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        f = ObjectVocabularyFilter(enabled=True)
        kept, removed = f.filter_objects(["person", "2", "three"])
        assert kept == ["person"]
        assert set(removed) == {"2", "three"}

    def test_gt_vocabulary_takes_priority_over_heuristic(self):
        """When GT metadata is available, ONLY GT object labels survive —
        the excluded-terms heuristic is not consulted at all, and objects
        not present in the GT label set are dropped too (closed-world GT
        vocabulary — see the class docstring's tradeoff note)."""
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        f = ObjectVocabularyFilter(enabled=True, use_gt_vocabulary=True)
        gt_metadata = {"person": True}
        kept, removed = f.filter_objects(
            ["person", "one", "walking", "sidewalk", "dog"], gt_metadata=gt_metadata,
        )
        assert kept == ["person"]
        assert set(removed) == {"one", "walking", "sidewalk", "dog"}

    def test_no_gt_metadata_falls_back_to_heuristic(self):
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        f = ObjectVocabularyFilter(enabled=True, use_gt_vocabulary=True)
        kept, removed = f.filter_objects(["person", "one", "dog"], gt_metadata=None)
        assert kept == ["dog", "person"]
        assert removed == ["one"]

    def test_use_gt_vocabulary_false_ignores_gt_and_uses_heuristic(self):
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        f = ObjectVocabularyFilter(enabled=True, use_gt_vocabulary=False)
        kept, removed = f.filter_objects(
            ["person", "one", "dog"], gt_metadata={"person": True},
        )
        assert kept == ["dog", "person"]   # "dog" survives — GT ignored, heuristic only
        assert removed == ["one"]

    def test_build_from_cfg_disabled_by_default(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.evaluators.object_vocabulary_filter import build_object_vocabulary_filter
        assert build_object_vocabulary_filter(OmegaConf.create({})) is None

    def test_build_from_cfg_enabled(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.evaluators.object_vocabulary_filter import build_object_vocabulary_filter
        cfg = OmegaConf.create({"verifier": {"object_vocabulary_filter": {"enabled": True}}})
        f = build_object_vocabulary_filter(cfg)
        assert f is not None
        assert f.enabled is True
        assert f.use_gt_vocabulary is True

    def test_build_from_cfg_additional_excluded_terms(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.evaluators.object_vocabulary_filter import build_object_vocabulary_filter
        cfg = OmegaConf.create({
            "verifier": {"object_vocabulary_filter": {
                "enabled": True, "additional_excluded_terms": ["mushroom"],
            }},
        })
        f = build_object_vocabulary_filter(cfg)
        kept, removed = f.filter_objects(["person", "mushroom"])
        assert kept == ["person"]
        assert removed == ["mushroom"]


# ─────────────────────────────────────────────────────────────────────────────
# remeasure() — object vocabulary filter integration (ETRI 5차 follow-up)
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectVocabularyFilterInRemeasure:
    def test_default_no_filter_is_backward_compatible(self):
        """With object_vocabulary_filter=None (the default), rows carry no
        filtered_* keys and metrics are unaffected — proves the new parameter
        does not change any existing behaviour when unused."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure

        items = [_item(0, ["person", "one", "walking", "sidewalk"], ["person", "one"], role="keyframe")]
        baseline = remeasure(items, presence_calibrator=None)
        with_none_filter = remeasure(items, presence_calibrator=None, object_vocabulary_filter=None)

        assert baseline["clip_only"]["metrics"] == with_none_filter["clip_only"]["metrics"]
        assert baseline["clip_only"]["rows"][0] == with_none_filter["clip_only"]["rows"][0]
        for row in with_none_filter["clip_only"]["rows"] + with_none_filter["calibrated"]["rows"]:
            assert "filtered_objects" not in row
            assert "filtered_missing_objects" not in row
            assert "filtered_additional_objects" not in row
        # "walking"/"sidewalk" still count as missing objects without the filter.
        assert set(with_none_filter["clip_only"]["rows"][0]["missing_objects"]) == {"walking", "sidewalk"}

    def test_filter_removes_count_action_scene_from_object_metric(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter

        items = [_item(0, ["person", "one", "walking", "sidewalk"], ["person", "one"], role="keyframe")]
        vocab_filter = ObjectVocabularyFilter(enabled=True)
        out = remeasure(items, presence_calibrator=None, object_vocabulary_filter=vocab_filter)

        row = out["clip_only"]["rows"][0]
        # "one" was in both ref/recon (never missing/additional); "walking"/
        # "sidewalk" were reference-only and would have been "missing" pre-filter.
        assert row["missing_objects"] == []
        assert row["object_match_rate"] == pytest.approx(1.0)
        assert set(row["filtered_objects"]) == {"one", "walking", "sidewalk"}
        assert set(row["filtered_missing_objects"]) == {"walking", "sidewalk"}
        assert row["filtered_additional_objects"] == []

    def test_person_object_survives_filter(self):
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter

        items = [_item(0, ["person", "one", "walking", "sidewalk"], ["person"], role="keyframe")]
        vocab_filter = ObjectVocabularyFilter(enabled=True)
        out = remeasure(items, presence_calibrator=None, object_vocabulary_filter=vocab_filter)
        assert "person" not in out["clip_only"]["rows"][0]["filtered_objects"]
        assert out["clip_only"]["rows"][0]["missing_objects"] == []

    def test_gt_metadata_restricts_to_gt_object_labels_only(self):
        """When an item carries gt_metadata, ONLY the GT's object labels are
        used for the object metric — non-GT tokens (including a real object
        noun like "dog" that GT never mentions) are filtered out too."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import RemeasurementItem, remeasure
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet

        item = RemeasurementItem(
            item_id=0,
            reference_packet=build_packet(objects=["person", "one", "walking", "sidewalk"], scene="s"),
            reconstructed_packet=build_packet(objects=["person", "dog"], scene="s"),
            gt_metadata={"person": True},
        )
        vocab_filter = ObjectVocabularyFilter(enabled=True, use_gt_vocabulary=True)
        out = remeasure([item], presence_calibrator=None, object_vocabulary_filter=vocab_filter)

        row = out["clip_only"]["rows"][0]
        assert row["missing_objects"] == []          # only "person" counted; present in both
        assert row["additional_objects"] == []        # "dog" filtered out (not a GT label)
        assert set(row["filtered_objects"]) == {"one", "walking", "sidewalk", "dog"}
        assert set(row["filtered_additional_objects"]) == {"dog"}

    def test_clip_only_and_calibrated_see_identical_filtering(self):
        """The whole point of applying the filter once in remeasure(): both
        columns must be judged against the SAME filtered vocabulary."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceResult
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        class AlwaysPresent(PresenceBackend):
            backend_name = "stub"
            def check(self, object_name, image=None, packet=None, gt_metadata=None):
                return PresenceResult(object_name=object_name, present=True, confidence=0.99, backend="stub")

        cal = PresenceCalibrator({"clip": AlwaysPresent()}, mode="clip_only")
        # recon lacks "person" entirely (only the noise word "one" survives before
        # filtering) so there is something left for the calibrator to correct
        # AFTER filtering has already run identically for both columns.
        items = [_item(0, ["person", "one", "walking", "sidewalk"], ["one"], role="keyframe")]
        vocab_filter = ObjectVocabularyFilter(enabled=True)
        out = remeasure(items, presence_calibrator=cal, object_vocabulary_filter=vocab_filter)

        clip_row = out["clip_only"]["rows"][0]
        cal_row = out["calibrated"]["rows"][0]
        # Both columns saw the identical filtered vocabulary...
        assert clip_row["filtered_objects"] == cal_row["filtered_objects"]
        assert set(clip_row["filtered_objects"]) == {"one", "walking", "sidewalk"}
        assert clip_row["filtered_missing_objects"] == cal_row["filtered_missing_objects"]
        assert clip_row["filtered_additional_objects"] == cal_row["filtered_additional_objects"]
        # ...and clip_only's remaining (post-filter) missing object is "person" —
        # only the calibrated column then goes on to correct it via AlwaysPresent.
        assert clip_row["missing_objects"] == ["person"]
        assert cal_row["missing_objects"] == []

    def test_temporal_metrics_change_when_filter_enabled(self):
        """PTC/SFR are derived from evaluate_sequence() reading orig_packet/
        recon_packet directly (bypassing PacketVerifier) — the filter must
        reach that path too, not just PacketVerifier's rows."""
        from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure
        from sgdjscc_lab.evaluators.object_vocabulary_filter import ObjectVocabularyFilter

        items = [
            _item(0, ["person", "one", "walking", "sidewalk"], ["person", "one"], role="keyframe"),
            _item(1, ["person", "one", "walking", "sidewalk"], ["person", "one", "running"], role="inter"),
        ]
        unfiltered = remeasure(items, presence_calibrator=None)
        filtered = remeasure(items, presence_calibrator=None, object_vocabulary_filter=ObjectVocabularyFilter(enabled=True))

        assert filtered["clip_only"]["metrics"]["mean_severity"] < unfiltered["clip_only"]["metrics"]["mean_severity"]
        assert filtered["clip_only"]["metrics"]["ptc"] > unfiltered["clip_only"]["metrics"]["ptc"]
