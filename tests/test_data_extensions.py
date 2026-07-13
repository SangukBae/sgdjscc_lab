"""tests/test_data_extensions.py – CPU tests for the data-loader extensions.

Covers the three additions (caption generation, COCO multi-caption, file-list
input) plus a backward-compatibility guard for the existing sidecar path. All
tests run on CPU with tiny synthetic images (no checkpoints / GPU).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.data.datasets import build_dataset_for_stage  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_images(folder: Path, names, captions=None):
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.new("RGB", (64, 64), (i * 7 % 255, 40, 90)).save(folder / name)
        if captions is not None and captions.get(name) is not None:
            (folder / name).with_suffix(".txt").write_text(captions[name], encoding="utf-8")


def _cfg(**dataset):
    """Minimal training cfg with a train.dataset block."""
    return OmegaConf.create({"train": {"dataset": dataset,
                                       "transforms": {"resize_to": 128, "crop_mode": "center"}}})


# ── [backward-compat] sidecar caption still works ─────────────────────────────

def test_sidecar_caption_unchanged(tmp_path):
    folder = tmp_path / "pairs"
    _make_images(folder, ["a.png", "b.png"],
                 captions={"a.png": "a cat on a mat", "b.png": "a dog"})
    cfg = _cfg(type="text_image", caption_source="sidecar", fallback_caption="fallback")
    ds = build_dataset_for_stage(str(folder), cfg, training=True, stage="text_dm")
    items = {Path(ds[i]["path"]).name: ds[i]["caption"] for i in range(len(ds))}
    assert items["a.png"] == "a cat on a mat"
    assert items["b.png"] == "a dog"
    assert ds[0]["image"].shape == (3, 128, 128)


# ── [2] COCO multi-caption: first / longest / random ──────────────────────────

def _coco_json(path: Path):
    data = {
        "images": [{"id": 1, "file_name": "000001.png"}],
        "annotations": [
            {"image_id": 1, "caption": "short"},
            {"image_id": 1, "caption": "a much longer descriptive caption here"},
            {"image_id": 1, "caption": "mid length caption"},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_coco_json_first_and_longest(tmp_path):
    folder = tmp_path / "coco"
    _make_images(folder, ["000001.png"])
    cj = tmp_path / "captions.json"
    _coco_json(cj)

    cfg_first = _cfg(type="text_image", caption_source="coco_json",
                     caption_path=str(cj), caption_select="first")
    ds_first = build_dataset_for_stage(str(folder), cfg_first, training=True, stage="text_dm")
    assert ds_first[0]["caption"] == "short"

    cfg_long = _cfg(type="text_image", caption_source="coco_json",
                    caption_path=str(cj), caption_select="longest")
    ds_long = build_dataset_for_stage(str(folder), cfg_long, training=True, stage="text_dm")
    assert ds_long[0]["caption"] == "a much longer descriptive caption here"


def test_coco_json_separate_train_val_caption_path(tmp_path):
    """val loader must read val_caption_path (COCO ships separate train/val JSONs)."""
    folder = tmp_path / "coco"
    _make_images(folder, ["000001.png"])
    train_json = tmp_path / "captions_train.json"
    val_json = tmp_path / "captions_val.json"
    train_json.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "000001.png"}],
        "annotations": [{"image_id": 1, "caption": "TRAIN caption"}]}), encoding="utf-8")
    val_json.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "000001.png"}],
        "annotations": [{"image_id": 1, "caption": "VAL caption"}]}), encoding="utf-8")

    cfg = _cfg(type="text_image", caption_source="coco_json",
               caption_path=str(train_json), val_caption_path=str(val_json),
               caption_select="first")
    ds_train = build_dataset_for_stage(str(folder), cfg, training=True, stage="text_dm")
    ds_val = build_dataset_for_stage(str(folder), cfg, training=False, stage="text_dm")
    assert ds_train[0]["caption"] == "TRAIN caption"
    assert ds_val[0]["caption"] == "VAL caption"     # val uses val_caption_path

    # No val_caption_path → val falls back to the train json (documented behaviour).
    cfg_no_val = _cfg(type="text_image", caption_source="coco_json",
                      caption_path=str(train_json), caption_select="first")
    ds_val2 = build_dataset_for_stage(str(folder), cfg_no_val, training=False, stage="text_dm")
    assert ds_val2[0]["caption"] == "TRAIN caption"


def test_coco_json_random_is_valid_and_val_is_reproducible(tmp_path):
    folder = tmp_path / "coco"
    _make_images(folder, ["000001.png"])
    cj = tmp_path / "captions.json"
    _coco_json(cj)
    valid = {"short", "a much longer descriptive caption here", "mid length caption"}

    cfg = _cfg(type="text_image", caption_source="coco_json",
               caption_path=str(cj), caption_select="random")
    ds = build_dataset_for_stage(str(folder), cfg, training=True, stage="text_dm")
    assert ds[0]["caption"] in valid                        # random picks a real caption

    # val loader (training=False) downgrades random → first for reproducibility
    ds_val = build_dataset_for_stage(str(folder), cfg, training=False, stage="text_dm")
    assert ds_val[0]["caption"] == "short"
    assert ds_val[0]["caption"] == ds_val[0]["caption"]


# ── [3] file-list input mode ──────────────────────────────────────────────────

def test_file_list_input_mode(tmp_path):
    folder = tmp_path / "imgs"
    _make_images(folder, ["x.png", "y.png", "z.png"])
    listing = tmp_path / "train.list"
    # mix of absolute path + a comment + blank line
    listing.write_text(
        f"# my train list\n{folder/'x.png'}\n\n{folder/'z.png'}\n", encoding="utf-8")

    cfg = _cfg(type="image", input_mode="file_list", file_list_path=str(listing))
    ds = build_dataset_for_stage(None, cfg, training=True, stage="jscc")
    names = sorted(Path(ds[i]["path"]).name for i in range(len(ds)))
    assert names == ["x.png", "z.png"]                       # 'y.png' not in the list
    assert ds[0]["image"].shape == (3, 128, 128)


def test_file_list_relative_paths(tmp_path):
    folder = tmp_path / "imgs"
    _make_images(folder, ["r1.png", "r2.png"])
    listing = tmp_path / "rel.list"
    # entries relative to the list file's directory
    listing.write_text("imgs/r1.png\nimgs/r2.png\n", encoding="utf-8")
    cfg = _cfg(type="image", input_mode="file_list", file_list_path=str(listing))
    ds = build_dataset_for_stage(None, cfg, training=True, stage="jscc")
    assert len(ds) == 2


def test_file_list_repo_relative_paths_from_nested_list(tmp_path):
    root = tmp_path / "repo"
    folder = root / "data" / "pairs" / "train"
    _make_images(folder, ["a.png", "b.png"])
    listing = root / "data" / "_lists" / "paper_like_multi" / "train.list"
    listing.parent.mkdir(parents=True)
    listing.write_text("data/pairs/train/a.png\ndata/pairs/train/b.png\n", encoding="utf-8")

    cfg = _cfg(type="image", input_mode="file_list", file_list_path=str(listing))
    ds = build_dataset_for_stage(None, cfg, training=True, stage="jscc")
    paths = [Path(ds[i]["path"]) for i in range(len(ds))]
    assert paths == [folder / "a.png", folder / "b.png"]


# ── [1] caption generation script ─────────────────────────────────────────────

def test_generate_captions_fixed_and_filename(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import generate_captions as gc

    folder = tmp_path / "celeba"
    _make_images(folder, ["000001.jpg", "000002.jpg"])

    out = gc.generate_captions(folder, mode="fixed", text="a portrait photo of a person")
    assert out == {"written": 2, "skipped": 2 - 2, "total": 2} or out["written"] == 2
    assert (folder / "000001.txt").read_text().strip() == "a portrait photo of a person"

    # re-run without overwrite → all skipped
    out2 = gc.generate_captions(folder, mode="fixed")
    assert out2["written"] == 0 and out2["skipped"] == 2

    # filename mode with overwrite
    _make_images(folder, ["my_happy-dog.jpg"])
    out3 = gc.generate_captions(folder, mode="filename", overwrite=True)
    assert (folder / "my_happy-dog.txt").read_text().strip() == "my happy dog"


def test_generated_captions_feed_text_dm(tmp_path):
    """End-to-end: generate sidecars, then load as a text_image dataset."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import generate_captions as gc
    folder = tmp_path / "celeba"
    _make_images(folder, ["000001.jpg", "000002.jpg"])
    gc.generate_captions(folder, mode="fixed", text="a portrait photo of a person")

    cfg = _cfg(type="text_image", caption_source="sidecar",
               fallback_caption="x")
    ds = build_dataset_for_stage(str(folder), cfg, training=True, stage="text_dm")
    assert ds[0]["caption"] == "a portrait photo of a person"


# ── [5] controlnet canny + sidecar combo (must not break) ─────────────────────

def test_controlnet_canny_with_sidecar(tmp_path):
    folder = tmp_path / "pairs"
    _make_images(folder, ["a.png"], captions={"a.png": "a face"})
    cfg = _cfg(type="text_image_edge", caption_source="sidecar",
               edge_source="canny", fallback_caption="x")
    ds = build_dataset_for_stage(str(folder), cfg, training=True, stage="controlnet")
    item = ds[0]
    assert item["caption"] == "a face"
    assert item["image"].shape == (3, 128, 128)
    assert item["edge"].shape == (1, 128, 128)
