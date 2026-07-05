"""qwen_caption.py – Qwen2.5-VL caption extractor for OFFLINE caption generation.

Used **only** by ``scripts/generate_captions.py --mode model`` to write
``<stem>.txt`` caption sidecars at dataset-prep time (e.g. bulk SA-1B). It is
deliberately kept **separate** from ``guidance/text_extractor.py`` (BLIP-2),
which stays the paper-faithful caption / VQA model used inside the inference &
evaluation pipelines. Swapping the *offline* captioner here therefore does NOT
touch the inference forward pass or its numerics (see the algorithm-preservation
rule in CLAUDE.md).

Why Qwen2.5-VL-3B-Instruct
--------------------------
A modern instruction-tuned VLM produces cleaner, more controllable single-
sentence captions than BLIP-2 for bulk auto-captioning. We drive it with a short
fixed instruction and trim the prompt tokens from the decode, so the output is
the caption only — no chat scaffolding.

Requirements
------------
Qwen2.5-VL is a *native* transformers architecture added in ``transformers``
4.49. The repo's pinned inference env (``transformers==4.44.2``) cannot load it,
so ``--mode model`` needs a newer transformers (see ``requirements.txt`` →
"Optional: Qwen2.5-VL caption generation"). The loader raises a clear, actionable
error when the installed transformers is too old.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Public defaults (kept here so the CLI can reference them without importing torch
# eagerly — scripts/generate_captions.py only imports this module in --mode model).
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = (
    "Describe the image in one concise, factual sentence. "
    "Output only the caption, with no preamble or quotation marks."
)
DEFAULT_MAX_NEW_TOKENS = 64
# Cap the vision-token budget so bulk captioning stays fast/cheap. Qwen resizes
# each image to <= max_pixels (H*W) before tiling into 28x28 patches, so
# 512*512 ≈ 334 vision tokens/image — plenty for a one-sentence caption.
DEFAULT_MAX_PIXELS = 512 * 512


def _pick_dtype(device: torch.device) -> torch.dtype:
    """bf16 on capable CUDA (Qwen-VL's recommended dtype), else fp16 on CUDA,
    fp32 on CPU (no fp16 kernels there)."""
    if device.type != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:  # pragma: no cover - very old torch
        pass
    return torch.float16


def _clean_caption(text: str) -> str:
    """Reduce a raw decode to a single bare caption sentence.

    Trimming the prompt tokens already removes the chat scaffolding, so this is a
    light guard: keep the first non-empty line, drop wrapping quotes/backticks and
    an occasional conversational lead-in, and collapse whitespace.
    """
    s = (text or "").strip()
    if not s:
        return ""
    for line in s.splitlines():
        if line.strip():
            s = line.strip()
            break
    s = s.strip().strip("\"'`").strip()
    leads = ("the caption is:", "the caption is", "the caption:", "caption:",
             "sure,", "here is", "here's")
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for lead in leads:
            if low.startswith(lead):
                s = s[len(lead):].strip().strip("\"'`:").strip()
                changed = True
                break
    return " ".join(s.split())


def _resolve_model_class():
    """Return the Qwen2.5-VL conditional-generation class, or raise a clear error.

    Prefers the native ``Qwen2_5_VLForConditionalGeneration`` (transformers >=
    4.49); falls back to the ``AutoModelForImageTextToText`` auto-class when the
    concrete symbol is absent but the auto-class exists.
    """
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
        return _Cls
    except Exception:
        pass
    try:
        from transformers import AutoModelForImageTextToText as _Cls
        return _Cls
    except Exception as exc:
        try:
            import transformers
            ver = transformers.__version__
        except Exception:
            ver = "unknown"
        raise RuntimeError(
            f"Qwen2.5-VL caption model requires transformers>=4.49 (found {ver}). "
            "Install into a caption-only env, e.g.: "
            "pip install -U 'transformers>=4.49' accelerate qwen-vl-utils"
        ) from exc


class QwenCaptionExtractor:
    """Batched image → single-sentence caption with Qwen2.5-VL-Instruct.

    Exposes ``caption_image`` / ``caption_images`` for the caption-generation
    script, plus a drop-in ``extract(img_tensor, device)`` returning list-of-lists
    for parity with ``TextExtractor`` (so it can stand in wherever that interface
    is expected).
    """

    def __init__(self, model, processor, *, prompt: str, max_new_tokens: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        self._model = model
        self._processor = processor
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.dtype = dtype

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    def _build_inputs(self, pil_images: Sequence):
        texts = []
        for _ in pil_images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }]
            texts.append(self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        inputs = self._processor(
            text=texts, images=list(pil_images), padding=True, return_tensors="pt")
        return inputs.to(self.device)

    def caption_images(self, pil_images: Sequence) -> List[str]:
        """Caption a batch of PIL RGB images → list of bare caption strings."""
        if not pil_images:
            return []
        inputs = self._build_inputs(pil_images)
        with torch.inference_mode():
            # Greedy decode for deterministic bulk captions. Pass ONLY
            # max_new_tokens (no max_length) → no length warning; clear the
            # sampling knobs the model's generation_config ships with so
            # do_sample=False doesn't warn about a stray temperature.
            generated = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [_clean_caption(t) for t in decoded]

    def caption_image(self, pil_image) -> str:
        out = self.caption_images([pil_image])
        return out[0] if out else ""

    def extract(self, img_tensor: torch.Tensor, device: Optional[torch.device] = None,
                *args, **kwargs) -> list:
        """TextExtractor-compatible interface. ``img_tensor``: ``[N,3,H,W]`` in
        [0,1]. Returns list-of-lists, e.g. ``[["a cat on a mat"]]``."""
        from torchvision import transforms
        pil = [transforms.ToPILImage()(img_tensor[i].detach().cpu().clamp(0, 1))
               for i in range(len(img_tensor))]
        return [self.caption_images(pil)]


def build_qwen_caption_extractor(
    device,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    prompt: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_pixels: Optional[int] = DEFAULT_MAX_PIXELS,
    min_pixels: Optional[int] = None,
) -> QwenCaptionExtractor:
    """Load Qwen2.5-VL-3B-Instruct (or ``model_id``) and return an extractor.

    dtype follows the device: bf16 on capable CUDA, fp16 otherwise on CUDA,
    fp32 on CPU. ``max_pixels`` / ``min_pixels`` bound the per-image vision-token
    budget for bulk captioning.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    dtype = _pick_dtype(dev)
    model_cls = _resolve_model_class()

    from transformers import AutoProcessor

    proc_kwargs = {}
    if max_pixels:
        proc_kwargs["max_pixels"] = int(max_pixels)
    if min_pixels:
        proc_kwargs["min_pixels"] = int(min_pixels)

    logger.info("Loading Qwen2.5-VL caption model (%s) [%s, %s]…", model_id, dev, dtype)
    processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
    model = model_cls.from_pretrained(model_id, torch_dtype=dtype)
    model.to(dev)
    model.eval()
    # Decoder-only batched generation needs left padding so new tokens align.
    try:
        processor.tokenizer.padding_side = "left"
    except Exception:  # pragma: no cover
        pass
    logger.info("Qwen2.5-VL caption model ready.")
    return QwenCaptionExtractor(
        model, processor, prompt=prompt, max_new_tokens=max_new_tokens,
        device=dev, dtype=dtype)
