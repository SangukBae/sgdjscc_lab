"""tests/test_transmission_accounting.py – transmission accounting + rate/reliability
report tests (ETRI 6차, step 11-12).
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# accounting/bit_accounting.py – per-component calculators
# ─────────────────────────────────────────────────────────────────────────────

class TestCaptionAndPacketBits:
    def test_caption_bits_stable_utf8_byte_length(self):
        from sgdjscc_lab.accounting.bit_accounting import caption_bits

        c = caption_bits("a red car")
        assert c.unit == "bits"
        assert c.proxy is False
        assert c.value == len("a red car".encode("utf-8")) * 8

    def test_caption_bits_grows_with_multibyte_utf8(self):
        # Korean text is multi-byte in UTF-8 — the byte-length x 8 formula must
        # actually reflect that, not len(str) x 8.
        from sgdjscc_lab.accounting.bit_accounting import caption_bits

        ascii_c = caption_bits("car")
        korean_c = caption_bits("자동차")
        assert korean_c.value == len("자동차".encode("utf-8")) * 8
        assert korean_c.value > ascii_c.value

    def test_caption_bits_empty(self):
        from sgdjscc_lab.accounting.bit_accounting import caption_bits
        assert caption_bits(None).value == 0.0
        assert caption_bits("").value == 0.0

    def test_semantic_packet_bits_stable_json_byte_length(self):
        from sgdjscc_lab.accounting.bit_accounting import semantic_packet_bits

        packet = build_packet(objects=["car", "dog"], scene="street scene", caption="a car and a dog")
        c1 = semantic_packet_bits(packet)
        c2 = semantic_packet_bits(dict(packet))  # same content, different dict instance
        assert c1.unit == "bits"
        assert c1.proxy is False
        assert c1.value == c2.value
        assert c1.value == len(json.dumps(packet, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")) * 8

    def test_semantic_packet_bits_grows_with_more_content(self):
        from sgdjscc_lab.accounting.bit_accounting import semantic_packet_bits

        small = semantic_packet_bits(build_packet(objects=["car"], scene="s"))
        big = semantic_packet_bits(build_packet(
            objects=["car", "dog", "tree", "bus"], scene="a much longer scene description",
            caption="a long and detailed caption describing many objects",
        ))
        assert big.value > small.value

    def test_semantic_packet_bits_empty(self):
        from sgdjscc_lab.accounting.bit_accounting import semantic_packet_bits
        assert semantic_packet_bits(None).value == 0.0
        assert semantic_packet_bits({}).value == 0.0


class TestLatentEdgeMotionProxies:
    def test_visual_latent_symbols_from_frame_shape(self):
        from sgdjscc_lab.accounting.bit_accounting import (
            LATENT_ELEMENTS_PER_PATCH, visual_latent_symbols,
        )

        frame = torch.zeros(1, 3, 128, 128)
        c = visual_latent_symbols(frame)
        assert c.unit == "channel_symbols"
        assert c.proxy is True
        assert c.value == LATENT_ELEMENTS_PER_PATCH   # exactly 1 patch

    def test_visual_latent_symbols_scales_with_patch_count(self):
        from sgdjscc_lab.accounting.bit_accounting import LATENT_ELEMENTS_PER_PATCH, visual_latent_symbols

        one_patch = visual_latent_symbols(torch.zeros(1, 3, 128, 128))
        four_patches = visual_latent_symbols(torch.zeros(1, 3, 256, 256))
        assert four_patches.value == pytest.approx(4 * LATENT_ELEMENTS_PER_PATCH)
        assert four_patches.value == 4 * one_patch.value

    def test_visual_latent_symbols_override(self):
        from sgdjscc_lab.accounting.bit_accounting import visual_latent_symbols

        c = visual_latent_symbols(torch.zeros(1, 3, 128, 128), latent_symbols_override=999.0)
        assert c.value == 999.0
        assert c.proxy is True
        assert "override" in c.note

    def test_visual_latent_symbols_no_frame(self):
        from sgdjscc_lab.accounting.bit_accounting import visual_latent_symbols
        c = visual_latent_symbols(None)
        assert c.value == 0.0
        assert c.proxy is True

    def test_edge_side_info_symbols_is_cr_fraction_of_latent(self):
        from sgdjscc_lab.accounting.bit_accounting import edge_side_info_symbols

        c = edge_side_info_symbols(4096.0, edge_cr=0.1)
        assert c.value == pytest.approx(409.6)
        assert c.unit == "channel_symbols"
        assert c.proxy is True

    def test_motion_side_info_bits_from_block_map(self):
        from sgdjscc_lab.accounting.bit_accounting import motion_side_info_bits

        motion = {"block_map": [[0.1] * 8 for _ in range(8)]}   # 8x8 = 64 blocks
        c = motion_side_info_bits(motion, bits_per_block=8.0)
        assert c.value == 64 * 8.0
        assert c.unit == "bits"
        assert c.proxy is True

    def test_motion_side_info_bits_no_motion(self):
        from sgdjscc_lab.accounting.bit_accounting import motion_side_info_bits
        assert motion_side_info_bits(None).value == 0.0
        assert motion_side_info_bits({}).value == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# accounting/bit_accounting.py – per-decision frame accounting
# ─────────────────────────────────────────────────────────────────────────────

class TestAccountFrame:
    def _frame(self):
        return torch.zeros(1, 3, 128, 128)

    def test_keyframe_charges_latent_edge_and_packet(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame, LATENT_ELEMENTS_PER_PATCH

        packet = build_packet(objects=["car"], scene="s", caption="a car")
        rec = account_frame(0, "keyframe", role="keyframe", frame=self._frame(), packet=packet)
        comps = rec.components
        assert comps["keyframe_visual_latent_symbols"]["value"] == LATENT_ELEMENTS_PER_PATCH
        assert comps["edge_side_info_symbols"]["value"] > 0
        assert comps["semantic_packet_bits"]["value"] > 0
        assert comps["reused_frame_symbols"]["value"] == 0
        assert comps["generated_frame_symbols"]["value"] == 0
        assert comps["recompute_frame_symbols"]["value"] == 0
        assert rec.total_bits > 0
        assert rec.total_channel_symbols > 0

    def test_reuse_charges_nothing(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame

        rec = account_frame(1, "reuse", role="inter", frame=self._frame(), packet=build_packet(objects=["car"]))
        assert rec.total_bits == 0.0
        assert rec.total_channel_symbols == 0.0
        for comp in rec.components.values():
            assert comp["value"] == 0

    def test_recompute_matches_keyframe_visual_cost_under_recompute_bucket(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame

        packet = build_packet(objects=["car"], scene="s")
        kf = account_frame(0, "keyframe", role="keyframe", frame=self._frame(), packet=packet)
        rc = account_frame(1, "recompute_semantic", role="inter", frame=self._frame(), packet=packet)
        assert rc.components["recompute_frame_symbols"]["value"] == kf.components["keyframe_visual_latent_symbols"]["value"]
        assert rc.components["keyframe_visual_latent_symbols"]["value"] == 0
        assert rc.total_bits == kf.total_bits   # same packet -> same packet bits

    def test_recompute_motion_same_as_recompute_semantic_cost(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame

        packet = build_packet(objects=["car"], scene="s")
        a = account_frame(1, "recompute_semantic", role="inter", frame=self._frame(), packet=packet)
        b = account_frame(1, "recompute_motion", role="inter", frame=self._frame(), packet=packet)
        assert a.total_channel_symbols == b.total_channel_symbols

    def test_generate_charges_only_caption_and_motion_no_visual_latent(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame

        motion = {"block_map": [[0.1] * 4 for _ in range(4)]}
        rec = account_frame(2, "generate", role="inter", caption="a red car", motion=motion)
        assert rec.components["keyframe_visual_latent_symbols"]["value"] == 0
        assert rec.components["edge_side_info_symbols"]["value"] == 0
        assert rec.components["semantic_packet_bits"]["value"] == 0
        assert rec.components["generated_frame_symbols"]["value"] == 0
        assert rec.components["caption_bits"]["value"] > 0
        assert rec.components["motion_side_info_bits"]["value"] > 0
        assert rec.total_bits > 0

    def test_unknown_decision_is_all_zero(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame
        rec = account_frame(3, None, role="inter")
        assert rec.total_bits == 0.0
        assert rec.total_channel_symbols == 0.0

    def test_transmitted_units_passthrough(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame
        rec = account_frame(0, "keyframe", role="keyframe", transmitted_units=7)
        assert rec.total_semantic_units == 7.0

    def test_symbols_per_bit_proxy_folds_bits_into_symbol_total(self):
        from sgdjscc_lab.accounting.bit_accounting import account_frame

        packet = build_packet(objects=["car"], scene="s")
        rec_default = account_frame(0, "keyframe", role="keyframe", frame=self._frame(), packet=packet, symbols_per_bit=1.0)
        rec_zero = account_frame(0, "keyframe", role="keyframe", frame=self._frame(), packet=packet, symbols_per_bit=0.0)
        assert rec_default.total_channel_symbols > rec_zero.total_channel_symbols
        assert rec_zero.total_channel_symbols == pytest.approx(
            rec_default.total_channel_symbols - rec_default.total_bits * 1.0
        )

    def test_to_dict_is_json_serialisable(self):
        import json as _json
        from sgdjscc_lab.accounting.bit_accounting import account_frame

        rec = account_frame(0, "keyframe", role="keyframe", frame=self._frame(), packet=build_packet(objects=["car"]))
        _json.dumps(rec.to_dict())


class TestBaselines:
    def test_naive_full_frame_baseline_charges_every_frame_fully(self):
        from sgdjscc_lab.accounting.bit_accounting import compute_baseline_record, BASELINE_NAIVE_FULL_FRAME

        packet = build_packet(objects=["car"], scene="s")
        frame = torch.zeros(1, 3, 128, 128)
        b_keyframe = compute_baseline_record(BASELINE_NAIVE_FULL_FRAME, 0, role="keyframe", frame=frame, packet=packet)
        b_inter = compute_baseline_record(BASELINE_NAIVE_FULL_FRAME, 1, role="inter", frame=frame, packet=packet)
        assert b_keyframe.total_channel_symbols == b_inter.total_channel_symbols > 0

    def test_keyframe_only_lgvsc_baseline_only_charges_keyframes_fully(self):
        from sgdjscc_lab.accounting.bit_accounting import compute_baseline_record, BASELINE_KEYFRAME_ONLY_LGVSC

        packet = build_packet(objects=["car"], scene="s", caption="a car")
        frame = torch.zeros(1, 3, 128, 128)
        b_keyframe = compute_baseline_record(BASELINE_KEYFRAME_ONLY_LGVSC, 0, role="keyframe", frame=frame, packet=packet)
        b_inter = compute_baseline_record(BASELINE_KEYFRAME_ONLY_LGVSC, 1, role="inter", frame=frame, packet=packet)
        assert b_keyframe.total_channel_symbols > b_inter.total_channel_symbols
        assert b_inter.components["caption_bits"]["value"] > 0
        assert b_inter.components["keyframe_visual_latent_symbols"]["value"] == 0

    def test_unknown_baseline_raises(self):
        from sgdjscc_lab.accounting.bit_accounting import compute_baseline_record
        with pytest.raises(NotImplementedError):
            compute_baseline_record("not_a_baseline", 0, role="keyframe")


# ─────────────────────────────────────────────────────────────────────────────
# pipelines/transmission_accounting.py – frame/segment/summary aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _frame_record(index, decision, role, objects=("car",), caption="a car", with_frame=True):
    from sgdjscc_lab.video.temporal_pipeline import FrameRecord

    packet = build_packet(objects=list(objects), scene="s", caption=caption)
    return FrameRecord(
        index=index, role=role, decision=decision,
        orig_packet=packet, recon_packet=packet,
        recon=(torch.zeros(1, 3, 128, 128) if with_frame else None),
        transmitted_units=(0 if decision in ("keyframe", "reuse") else 2),
    )


def _fake_result(records, segment_records=None, naive_units=None, transmitted_units=None):
    n_gen = sum(1 for r in records if r.decision == "generate")
    n_reused = sum(1 for r in records if r.decision == "reuse")
    n_rec = sum(1 for r in records if r.decision in ("recompute_semantic", "recompute_motion"))
    if transmitted_units is None:
        transmitted_units = sum(r.transmitted_units for r in records)
    if naive_units is None:
        naive_units = transmitted_units * 2 + 1   # arbitrary but > transmitted_units
    overhead_reduction = 1.0 - (transmitted_units / naive_units) if naive_units else None
    return {
        "records": records,
        "segment_records": segment_records or [],
        "summary": {
            "n_frames": len(records), "n_generate": n_gen, "n_reused": n_reused,
            "n_recompute_semantic": n_rec, "transmitted_units": transmitted_units,
            "naive_units": naive_units, "overhead_reduction": overhead_reduction,
        },
    }


class TestAccountTransmission:
    def test_frame_records_match_input_length_and_order(self):
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission

        records = [
            _frame_record(0, "keyframe", "keyframe"),
            _frame_record(1, "reuse", "inter"),
            _frame_record(2, "generate", "inter"),
        ]
        out = account_transmission(_fake_result(records))
        assert [r["frame_index"] for r in out["frame_records"]] == [0, 1, 2]
        assert [r["decision"] for r in out["frame_records"]] == ["keyframe", "reuse", "generate"]

    def test_summary_totals_and_reduction(self):
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission
        from sgdjscc_lab.accounting.bit_accounting import BASELINE_NAIVE_FULL_FRAME

        records = [
            _frame_record(0, "keyframe", "keyframe"),
            _frame_record(1, "reuse", "inter"),
            _frame_record(2, "reuse", "inter"),
        ]
        out = account_transmission(_fake_result(records), baseline=BASELINE_NAIVE_FULL_FRAME)
        summary = out["summary"]
        assert summary["n_frames"] == 3
        assert summary["n_keyframes"] == 1
        assert summary["n_reused"] == 2
        assert summary["total_bits"] > 0
        assert summary["total_channel_symbols"] > 0
        # naive baseline charges every frame fully -> policy (with 2 reused
        # frames sending nothing) must be strictly cheaper.
        assert summary["total_channel_symbols"] < summary["baseline_channel_symbols"]
        assert summary["bit_reduction"] > 0
        assert summary["symbol_reduction"] > 0

    def test_semantic_unit_reduction_is_passthrough_from_temporal_summary(self):
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission

        records = [_frame_record(0, "keyframe", "keyframe"), _frame_record(1, "generate", "inter")]
        result = _fake_result(records, transmitted_units=3, naive_units=10)
        out = account_transmission(result)
        assert out["summary"]["baseline_semantic_units"] == 10
        assert out["summary"]["semantic_unit_reduction"] == pytest.approx(0.7)

    def test_proxy_fraction_between_zero_and_one(self):
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission

        records = [_frame_record(0, "keyframe", "keyframe")]
        out = account_transmission(_fake_result(records))
        assert 0.0 <= out["summary"]["proxy_fraction"] <= 1.0

    def test_does_not_mutate_input_result(self):
        # "accounting.enabled=false" in evaluate_video.py just skips calling
        # this function at all; the invariant this module itself must hold is
        # that CALLING it never mutates the TemporalPipeline result it reads.
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission

        records = [_frame_record(0, "keyframe", "keyframe"), _frame_record(1, "reuse", "inter")]
        result = _fake_result(records)
        before = json.loads(json.dumps(result["summary"]))
        account_transmission(result)
        assert result["summary"] == before
        assert result["records"] is records   # same objects, untouched

    def test_segment_summary_aggregates_and_reduces(self):
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission

        records = [
            _frame_record(0, "keyframe", "keyframe"),
            _frame_record(1, "reuse", "inter"),
            _frame_record(2, "generate", "inter"),
        ]
        segment_records = [{
            "segment_id": 0, "keyframe_index": 0, "inter_frame_indices": [1, 2],
        }]
        out = account_transmission(_fake_result(records, segment_records=segment_records))
        seg = out["segment_summaries"][0]
        assert seg["segment_id"] == 0
        assert seg["n_frames"] == 3
        assert seg["n_reused"] == 1
        assert seg["n_generate"] == 1
        assert seg["reduction_vs_naive_bits"] is not None
        assert seg["reduction_vs_naive_symbols"] is not None


class TestWriteAccounting:
    def test_writes_requested_artefacts(self, tmp_path):
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission, write_accounting

        records = [_frame_record(0, "keyframe", "keyframe"), _frame_record(1, "reuse", "inter")]
        segment_records = [{"segment_id": 0, "keyframe_index": 0, "inter_frame_indices": [1]}]
        out = account_transmission(_fake_result(records, segment_records=segment_records))

        write_accounting(
            out,
            frame_json=str(tmp_path / "frame.json"), frame_csv=str(tmp_path / "frame.csv"),
            segment_json=str(tmp_path / "segment.json"), segment_csv=str(tmp_path / "segment.csv"),
            summary_json=str(tmp_path / "summary.json"),
        )
        for name in ("frame.json", "frame.csv", "segment.json", "segment.csv", "summary.json"):
            assert (tmp_path / name).exists()

        frame_rows = json.loads((tmp_path / "frame.json").read_text(encoding="utf-8"))
        assert len(frame_rows) == 2
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert "total_bits" in summary and "note" in summary

    def test_skips_unrequested_artefacts(self, tmp_path):
        from sgdjscc_lab.pipelines.transmission_accounting import account_transmission, write_accounting

        out = account_transmission(_fake_result([_frame_record(0, "keyframe", "keyframe")]))
        write_accounting(out, summary_json=str(tmp_path / "only_this.json"))
        assert list(tmp_path.iterdir()) == [tmp_path / "only_this.json"]


# ─────────────────────────────────────────────────────────────────────────────
# pipelines/rate_reliability_report.py
# ─────────────────────────────────────────────────────────────────────────────

class TestRateReliabilityReport:
    def _summary(self):
        return {
            "n_frames": 4, "total_bits": 1000.0, "total_channel_symbols": 5000.0,
            "bit_reduction": 0.3, "symbol_reduction": 0.4, "semantic_unit_reduction": 0.5,
            "n_generate": 1, "n_reused": 2, "n_recompute": 0, "baseline": "naive_full_frame_packet",
            "proxy_fraction": 0.6,
        }

    def test_row_combines_rate_and_reliability_fields(self):
        from sgdjscc_lab.pipelines.rate_reliability_report import build_rate_reliability_row

        row = build_rate_reliability_row(
            self._summary(), {"ptc": 0.9, "sfr": 0.1, "sdi": 0.05}, mean_severity=0.2, label="policy_a",
        )
        assert row["bits_per_frame"] == pytest.approx(250.0)
        assert row["symbols_per_frame"] == pytest.approx(1250.0)
        assert row["bit_reduction"] == 0.3
        assert row["symbol_reduction"] == 0.4
        assert row["semantic_unit_reduction"] == 0.5
        assert row["ptc"] == 0.9
        assert row["sfr"] == 0.1
        assert row["sdi"] == 0.05
        assert row["mean_severity"] == 0.2
        assert row["n_generate"] == 1
        assert row["label"] == "policy_a"

    def test_row_handles_missing_temporal_metrics(self):
        from sgdjscc_lab.pipelines.rate_reliability_report import build_rate_reliability_row

        row = build_rate_reliability_row(self._summary(), None, mean_severity=None)
        assert row["ptc"] is None and row["sfr"] is None and row["sdi"] is None
        assert row["mean_severity"] is None

    def test_write_summary_json(self, tmp_path):
        from sgdjscc_lab.pipelines.rate_reliability_report import (
            build_rate_reliability_row, write_rate_reliability_summary,
        )

        row = build_rate_reliability_row(self._summary(), {"ptc": 0.9, "sfr": 0.1, "sdi": 0.05})
        out_path = tmp_path / "rr.json"
        write_rate_reliability_summary(row, str(out_path))
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["bits_per_frame"] == pytest.approx(250.0)
        assert "note" in data

    def test_append_creates_and_grows_curve_csv(self, tmp_path):
        from sgdjscc_lab.pipelines.rate_reliability_report import (
            append_rate_reliability_row, build_rate_reliability_row,
        )

        curve = tmp_path / "curve.csv"
        row1 = build_rate_reliability_row(self._summary(), {"ptc": 0.9, "sfr": 0.1, "sdi": 0.05}, label="a")
        row2 = build_rate_reliability_row(self._summary(), {"ptc": 0.8, "sfr": 0.2, "sdi": 0.1}, label="b")
        append_rate_reliability_row(row1, str(curve))
        append_rate_reliability_row(row2, str(curve))

        import csv as csv_mod
        with open(curve, newline="", encoding="utf-8") as fh:
            rows = list(csv_mod.DictReader(fh))
        assert len(rows) == 2
        assert rows[0]["label"] == "a"
        assert rows[1]["label"] == "b"

    def test_merge_curves_dedups_by_label(self, tmp_path):
        from sgdjscc_lab.pipelines.rate_reliability_report import (
            append_rate_reliability_row, build_rate_reliability_row, merge_rate_reliability_curves,
        )

        curve1 = tmp_path / "curve1.csv"
        curve2 = tmp_path / "curve2.csv"
        append_rate_reliability_row(
            build_rate_reliability_row(self._summary(), {"ptc": 0.9, "sfr": 0.1, "sdi": 0.05}, label="a"),
            str(curve1),
        )
        append_rate_reliability_row(
            build_rate_reliability_row(self._summary(), {"ptc": 0.5, "sfr": 0.5, "sdi": 0.5}, label="a"),
            str(curve2),
        )   # same label "a" -> last-wins
        append_rate_reliability_row(
            build_rate_reliability_row(self._summary(), {"ptc": 0.7, "sfr": 0.3, "sdi": 0.2}, label="b"),
            str(curve2),
        )

        merged = tmp_path / "merged.csv"
        n = merge_rate_reliability_curves([str(curve1), str(curve2)], str(merged))
        assert n == 2
        import csv as csv_mod
        with open(merged, newline="", encoding="utf-8") as fh:
            rows = list(csv_mod.DictReader(fh))
        by_label = {r["label"]: r for r in rows}
        assert by_label["a"]["ptc"] == "0.5"   # curve2's row won for label "a"
        assert by_label["b"]["ptc"] == "0.7"
