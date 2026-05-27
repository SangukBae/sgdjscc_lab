"""tests/test_io.py – Unit tests for image file discovery and loading.

No GPU, no checkpoints, no SGDJSCC imports required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the package is importable even without editable install
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.io import list_image_files, load_image_as_tensor, save_tensor_as_image


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def image_dir(tmp_path):
    """Create a temporary directory with several test images (1×1 pixel PNGs)."""
    from PIL import Image as PILImage

    for name in ("a.png", "b.jpg", "c.png", "skip.txt"):
        if name.endswith(".txt"):
            (tmp_path / name).write_text("not an image")
        else:
            img = PILImage.new("RGB", (4, 4), color=(128, 64, 32))
            img.save(tmp_path / name)
    return tmp_path


@pytest.fixture()
def single_image(tmp_path):
    """Create a single test image (4×4 PNG)."""
    from PIL import Image as PILImage
    path = tmp_path / "single.png"
    img = PILImage.new("RGB", (4, 4), color=(100, 150, 200))
    img.save(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# list_image_files tests
# ─────────────────────────────────────────────────────────────────────────────

class TestListImageFiles:
    def test_single_file_returns_list_of_one(self, single_image):
        files = list_image_files(single_image)
        assert len(files) == 1
        assert files[0] == single_image

    def test_directory_returns_only_images(self, image_dir):
        files = list_image_files(image_dir)
        names = {f.name for f in files}
        assert "a.png" in names
        assert "b.jpg" in names
        assert "c.png" in names
        assert "skip.txt" not in names

    def test_directory_returns_sorted_list(self, image_dir):
        files = list_image_files(image_dir)
        assert files == sorted(files)

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_image_files(tmp_path / "does_not_exist")

    def test_unsupported_extension_single_file_raises(self, tmp_path):
        txt = tmp_path / "note.txt"
        txt.write_text("hello")
        with pytest.raises(ValueError):
            list_image_files(txt)

    def test_empty_directory_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            list_image_files(empty)

    def test_webp_extension_accepted(self, tmp_path):
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (4, 4))
        img.save(tmp_path / "img.webp", format="WEBP")
        files = list_image_files(tmp_path / "img.webp")
        assert len(files) == 1


# ─────────────────────────────────────────────────────────────────────────────
# load_image_as_tensor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadImageAsTensor:
    def test_returns_4d_tensor(self, single_image):
        import torch
        t = load_image_as_tensor(single_image)
        assert t.ndim == 4
        assert t.shape[0] == 1
        assert t.shape[1] == 3

    def test_values_in_zero_one(self, single_image):
        t = load_image_as_tensor(single_image)
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    def test_spatial_dimensions_match_image(self, tmp_path):
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (16, 32))
        path = tmp_path / "rect.png"
        img.save(path)
        t = load_image_as_tensor(path)
        assert t.shape == (1, 3, 32, 16)


# ─────────────────────────────────────────────────────────────────────────────
# save_tensor_as_image tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveTensorAsImage:
    def test_saves_png(self, tmp_path, single_image):
        import torch
        t = load_image_as_tensor(single_image)
        out = tmp_path / "out.png"
        save_tensor_as_image(t, out)
        assert out.exists()

    def test_parent_dir_created_automatically(self, tmp_path, single_image):
        import torch
        t = load_image_as_tensor(single_image)
        out = tmp_path / "nested" / "dir" / "out.png"
        save_tensor_as_image(t, out)
        assert out.exists()
