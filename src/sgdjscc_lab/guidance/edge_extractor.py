"""edge_extractor.py – MuGE-based soft edge (canny) extractor.

Extracted from _extract_canny() / _build_canny_net() in runtime.py and
pipeline.py.  The preprocessing.py duplicate (extract_canny) is now a thin
re-export of this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from sgdjscc_lab._sgdjscc import ensure_sgdjscc_on_path
from sgdjscc_lab.utils.memory import release_cuda_memory

logger = logging.getLogger(__name__)


class EdgeExtractor:
    """Wraps the MuGE network to produce soft edge maps.

    Implements the SemanticGuideExtractor.extract() interface from the README.
    Returns (canny_data, canny_uncertainty) each of shape [N, 11, 128, 128].
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def extract(
        self,
        img_tensor: torch.Tensor,
        device: torch.device,
        offload_device: Optional[torch.device] = None,
        offload_after: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract soft edge maps using the MuGE network.

        Mirrors generate_canny() call in inference_one.py.

        Parameters
        ----------
        img_tensor:
            ``[N, 3, 128, 128]`` float in [0, 1].
        device:
            Device the model should run on.
        offload_device:
            If offload_after=True, move the model here after extraction.
        offload_after:
            Move model to offload_device after extraction.

        Returns
        -------
        (canny_data, canny_uncertainty)
            Both ``[N, 11, 128, 128]``, on *device*.
        """
        ensure_sgdjscc_on_path()
        from utils.utils import generate_canny

        if offload_after and offload_device is not None:
            self._model.to(device)
        try:
            with torch.inference_mode():
                data, uncertainty = generate_canny(img_tensor, self._model, device)
                return data.to(device), uncertainty.to(device)
        finally:
            if offload_after and offload_device is not None:
                self._model.to(offload_device)
                release_cuda_memory()


def build_edge_extractor(
    model_root: Union[Path, str],
    device: torch.device,
) -> EdgeExtractor:
    """Load muge-epoch-19-checkpoint.pth and return an EdgeExtractor.

    Mirrors _build_canny_net() from runtime.py (originally
    inference_one.py lines 282–285).
    """
    ensure_sgdjscc_on_path()
    from models.test_advanced_network.muge_model import Mymodel as MugeModel

    logger.info("Loading MuGE edge extractor…")
    net = MugeModel()
    ckpt = torch.load(
        Path(model_root) / "muge-epoch-19-checkpoint.pth", map_location="cpu"
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(device)
    logger.info("MuGE ready.")
    return EdgeExtractor(net)
