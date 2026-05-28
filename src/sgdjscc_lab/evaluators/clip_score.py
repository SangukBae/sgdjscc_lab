"""evaluators/clip_score.py – CLIP-based semantic similarity evaluators.

Two similarity measures are provided:

image_image_score(original, reconstructed)
    Cosine similarity between CLIP image embeddings.
    Measures how semantically similar the reconstructed image is to the original.

text_image_score(text_list, reconstructed)
    Cosine similarity between CLIP text and image embeddings.
    Measures alignment between a transmitted caption (or GT text) and the
    reconstructed image — captures whether text-guided reconstruction preserved
    the semantic intent.

Input convention
----------------
Images  : ``[N, 3, H, W]`` float tensors in [0, 1].  Resized to 224×224 internally.
Texts   : list of strings, length must equal batch size N.

Both methods return a single scalar float in [0, 1] (batch average cosine
similarity after L2 normalisation).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)

# CLIP input size
_CLIP_SIZE = 224


def _tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    """Convert [N, 3, H, W] float [0,1] tensor to list of PIL images."""
    tensor = tensor.float().clamp(0, 1).cpu()
    pil_list = []
    for i in range(tensor.shape[0]):
        arr = (tensor[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_list.append(Image.fromarray(arr))
    return pil_list


class CLIPScoreEvaluator:
    """CLIP-based semantic similarity evaluator.

    The CLIP model is loaded lazily on first use and cached for reuse across
    multiple evaluation calls.

    Parameters
    ----------
    model_name:
        CLIP model variant.  Default ``'ViT-B/32'`` balances speed and quality.
        Other options: ``'ViT-L/14'`` (higher quality), ``'RN50'`` (faster).
    device:
        Compute device.  Defaults to CPU.

    Notes
    -----
    Requires the ``openai-clip`` package (``pip install openai-clip``).
    Similarity values are cosine similarities in [-1, 1]; in practice
    [0, 1] for natural images after ReLU-like normalisation.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self._model = None
        self._preprocess = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "openai-clip not found. Install with: pip install openai-clip"
            ) from exc
        self._model, self._preprocess = clip.load(self.model_name, device=self.device)
        self._model.eval()
        logger.info("Loaded CLIP model: %s on %s", self.model_name, self.device)

    def _encode_images(self, tensor: torch.Tensor) -> torch.Tensor:
        """Encode image tensor → L2-normalised CLIP features [N, D]."""
        self._load()
        pil_list = _tensor_to_pil_list(tensor)
        images = torch.stack([self._preprocess(img) for img in pil_list]).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(images).float()
        feats = F.normalize(feats, dim=-1)
        return feats

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode text list → L2-normalised CLIP features [N, D]."""
        self._load()
        import clip
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)
        return feats

    def image_image_score(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> float:
        """Cosine similarity between CLIP embeddings of original and reconstructed images.

        Parameters
        ----------
        original, reconstructed:
            ``[N, 3, H, W]`` float tensors in [0, 1].

        Returns
        -------
        float
            Mean cosine similarity over the batch in [−1, 1].
            Typical range for natural images: [0.7, 1.0].
        """
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Shape mismatch: original={tuple(original.shape)}, "
                f"reconstructed={tuple(reconstructed.shape)}"
            )
        orig_feats  = self._encode_images(original)
        recon_feats = self._encode_images(reconstructed)
        # Per-sample cosine similarity, then average
        sim = (orig_feats * recon_feats).sum(dim=-1)   # [N]
        return float(sim.mean().item())

    def text_image_score(
        self,
        text_list: List[str],
        reconstructed: torch.Tensor,
    ) -> float:
        """Cosine similarity between CLIP text embeddings and reconstructed images.

        Parameters
        ----------
        text_list:
            List of N text prompts / captions (one per image in the batch).
        reconstructed:
            ``[N, 3, H, W]`` float tensor in [0, 1].

        Returns
        -------
        float
            Mean cosine similarity over the batch in [−1, 1].
        """
        n = reconstructed.shape[0]
        if len(text_list) != n:
            raise ValueError(
                f"text_list length ({len(text_list)}) must equal batch size ({n})."
            )
        text_feats  = self._encode_texts(text_list)
        image_feats = self._encode_images(reconstructed)
        sim = (text_feats * image_feats).sum(dim=-1)   # [N]
        return float(sim.mean().item())
