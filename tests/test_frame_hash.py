from __future__ import annotations

import pytest

from sgdjscc_lab.utils.frame_hash import frame_tree_sha256, sha256_file


def test_sha256_file_matches_known_content(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello")
    import hashlib
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


def test_frame_tree_sha256_identical_content_gives_identical_hash(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    names = ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    contents = [b"one", b"two", b"three"]
    for name, content in zip(names, contents):
        (a / name).write_bytes(content)
        (b / name).write_bytes(content)
    hash_a = frame_tree_sha256([a / name for name in names])
    hash_b = frame_tree_sha256([b / name for name in names])
    assert hash_a == hash_b


def test_frame_tree_sha256_detects_single_byte_difference(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "frame_00000.png").write_bytes(b"same")
    (b / "frame_00000.png").write_bytes(b"sama")
    assert frame_tree_sha256([a / "frame_00000.png"]) != frame_tree_sha256([b / "frame_00000.png"])


def test_frame_tree_sha256_detects_reordering(tmp_path):
    d = tmp_path
    (d / "x.png").write_bytes(b"1")
    (d / "y.png").write_bytes(b"2")
    forward = frame_tree_sha256([d / "x.png", d / "y.png"])
    backward = frame_tree_sha256([d / "y.png", d / "x.png"])
    assert forward != backward


def test_frame_tree_sha256_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        frame_tree_sha256([tmp_path / "does_not_exist.png"])
