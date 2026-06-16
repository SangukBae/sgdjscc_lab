"""tests/test_packet.py – Phase 4-A semantic packet generation tests.

Pure-Python / torch-only.  No CLIP, BLIP2 or checkpoints required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# ObjectExtractor (caption path, no model)
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectExtractor:
    def test_from_caption_detects_objects(self):
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        oe = ObjectExtractor()
        objs = oe.from_caption("a red car next to a black dog")
        assert "car" in objs
        assert "dog" in objs

    def test_from_caption_word_boundary(self):
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        oe = ObjectExtractor(vocabulary=["car"])
        # "scarf" must NOT match "car"
        assert oe.from_caption("a warm scarf") == []

    def test_from_caption_plural(self):
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        oe = ObjectExtractor(vocabulary=["cat"])
        assert "cat" in oe.from_caption("two cats playing")

    def test_empty_caption(self):
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        assert ObjectExtractor().from_caption("") == []

    def test_nouns_from_caption_open_vocab(self):
        """Regression: open-vocabulary nouns (outside COCO-80) must be captured —
        previously 'mushroom' was dropped so packet.objects was empty."""
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        oe = ObjectExtractor()
        nouns = oe.nouns_from_caption("a mushroom with a face and a smile on it")
        assert "mushroom" in nouns          # not in COCO-80, but recovered
        assert "face" in nouns
        # function words / prepositions are filtered out
        assert "with" not in nouns and "and" not in nouns and "on" not in nouns

    def test_nouns_filter_adjectives_and_boilerplate(self):
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        oe = ObjectExtractor()
        nouns = oe.nouns_from_caption("a close up photo of a red mushroom")
        assert "mushroom" in nouns
        assert "red" not in nouns           # adjective filtered
        assert "close" not in nouns and "photo" not in nouns   # boilerplate filtered

    def test_extract_objects_combines_vocab_and_nouns(self):
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        oe = ObjectExtractor()
        # COCO word (car) + open-vocab noun (mushroom) both present
        objs = oe.extract_objects("a car next to a mushroom")
        assert "car" in objs and "mushroom" in objs

    def test_extract_objects_can_disable_caption_nouns(self):
        from sgdjscc_lab.guidance.object_extractor import ObjectExtractor
        oe = ObjectExtractor()
        objs = oe.extract_objects("a mushroom", include_caption_nouns=False)
        assert objs == []                   # mushroom not in COCO-80, nouns off


# ─────────────────────────────────────────────────────────────────────────────
# RelationExtractor
# ─────────────────────────────────────────────────────────────────────────────

class TestRelationExtractor:
    def test_extracts_triplet(self):
        from sgdjscc_lab.guidance.relation_extractor import RelationExtractor
        rels = RelationExtractor().extract("a cat on a table", ["cat", "table"])
        assert {"subject": "cat", "predicate": "on", "object": "table"} in rels

    def test_no_relation_without_two_objects(self):
        from sgdjscc_lab.guidance.relation_extractor import RelationExtractor
        assert RelationExtractor().extract("a cat on a mat", ["cat"]) == []

    def test_multiword_predicate_preferred(self):
        from sgdjscc_lab.guidance.relation_extractor import RelationExtractor
        rels = RelationExtractor().extract("a dog next to a car", ["dog", "car"])
        assert any(r["predicate"] == "next to" for r in rels)


# ─────────────────────────────────────────────────────────────────────────────
# attributes + importance
# ─────────────────────────────────────────────────────────────────────────────

class TestAttributes:
    def test_extract_attributes(self):
        from sgdjscc_lab.guidance.semantic_packet_extractor import extract_attributes
        attrs = extract_attributes("a red car and a wooden table", ["car", "table"])
        assert "red" in attrs.get("car", [])
        assert "wooden" in attrs.get("table", [])


class TestImportanceEstimator:
    def test_scores_and_order(self):
        from sgdjscc_lab.guidance.importance_estimator import ImportanceEstimator
        packet = {
            "objects": ["car", "tree"],
            "relations": [{"subject": "car", "predicate": "near", "object": "tree"}],
            "caption": "a car near a tree",
        }
        out = ImportanceEstimator().estimate(packet)
        assert set(out["scores"].keys()) == {"car", "tree"}
        assert out["order"][0] in {"car", "tree"}
        for v in out["scores"].values():
            assert 0.0 <= v <= 1.0

    def test_empty_objects(self):
        from sgdjscc_lab.guidance.importance_estimator import ImportanceEstimator
        out = ImportanceEstimator().estimate({"objects": []})
        assert out == {"scores": {}, "order": []}


# ─────────────────────────────────────────────────────────────────────────────
# build_packet schema + summaries
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPacket:
    def test_required_fields(self):
        from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet
        p = build_packet(caption="a cat on a mat", objects=["cat"], scene="indoor scene")
        for key in ("caption", "scene", "objects", "relations", "attributes",
                    "importance", "meta"):
            assert key in p
        assert p["meta"]["version"]

    def test_objects_deduped_sorted(self):
        from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet
        p = build_packet(objects=["dog", "cat", "cat"])
        assert p["objects"] == ["cat", "dog"]

    def test_summarize_edge(self):
        import torch
        from sgdjscc_lab.guidance.semantic_packet_extractor import summarize_edge
        edge = torch.zeros(1, 32, 32)
        edge[:, :8, :] = 1.0
        s = summarize_edge(edge, threshold=0.5)
        assert 0.0 <= s["density"] <= 1.0
        assert s["density"] == pytest.approx(0.25, abs=0.01)

    def test_summarize_segmentation(self):
        import torch
        from sgdjscc_lab.guidance.semantic_packet_extractor import summarize_segmentation
        lab = torch.zeros(1, 4, 4, dtype=torch.long)
        lab[:, :2, :] = 1
        s = summarize_segmentation(lab, ["sky", "ground"])
        assert s["dominant_class"] in {"sky", "ground"}
        assert pytest.approx(sum(s["class_histogram"].values()), abs=1e-5) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# packet_io round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestPacketIO:
    def test_save_load_round_trip(self, tmp_path):
        from sgdjscc_lab.utils.packet_io import save_packet, load_packet, packet_path
        from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet
        p = build_packet(caption="a cat", objects=["cat"], scene="indoor scene")
        path = packet_path(tmp_path, "frame0")
        save_packet(p, path)
        assert path.exists()
        loaded = load_packet(path)
        assert loaded["objects"] == ["cat"]

    def test_to_jsonable_handles_tensor(self):
        import torch
        from sgdjscc_lab.utils.packet_io import to_jsonable
        out = to_jsonable({"a": torch.tensor(3.0), "b": torch.tensor([1, 2])})
        assert out["a"] == pytest.approx(3.0)
        assert out["b"] == [1, 2]
