"""End-to-end frame transport used by the transmission-reduction benchmark.

The sender turns an input frame into one serialized ``TransmissionBundle``.
The receiver API accepts only those bytes plus models/configuration; it never
receives the source frame, sender latent, caption, edge tensors, or patch
layout as out-of-band Python objects.

This runtime intentionally supports the benchmark's production default
``mask_method=none``.  Other masking modes depend on sender-only statistics
that are not yet part of the wire format and are rejected explicitly instead
of silently crossing the receiver boundary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


def _caption_list(gt_text, n_patches: int) -> list[str]:
    if gt_text and gt_text[0]:
        values = [str(x) for x in gt_text[0]]
    else:
        values = []
    if not values:
        return [""] * n_patches
    if len(values) == 1 and n_patches > 1:
        return values * n_patches
    if len(values) != n_patches:
        raise ValueError(
            f"caption count {len(values)} does not match patch count {n_patches}"
        )
    return values


def _manifest_patch_layout(meta) -> Dict[str, Any]:
    height, width, positions = meta
    return {
        "height": int(height),
        "width": int(width),
        "positions": [[int(y), int(x)] for y, x in positions],
    }


def encode_frame_to_bundle_bytes(
    frame,
    models,
    cfg,
    *,
    bit_depth: Optional[int],
    granularity: str,
    keyframe_index: int,
    manifest: Optional[Dict[str, Any]] = None,
    selected_keyframes: Optional[Sequence[int]] = None,
    visual_is_analog: bool = False,
    caption_override: Optional[str] = None,
    semantic_packet: Optional[Dict[str, Any]] = None,
    include_quantization_error_metadata: bool = False,
    quantization_diagnostics: Optional[list[Dict[str, Any]]] = None,
    guide_profile: str = "baseline",
    guide_transmission_ordinal: int = 0,
) -> Tuple[bytes, int]:
    """Sender: encode *frame* and return its complete serialized wire bundle."""
    import torch

    from sgdjscc_lab.pipelines.infer_pipeline import (
        _encode_latent,
        _extract_semantic_guidance,
    )
    from sgdjscc_lab.transmission.packet_bundle import build_frame_bundle, serialize_bundle
    from sgdjscc_lab.transmission.guide_transport import (
        GUIDE_PROFILES,
        prepare_guides_for_transport,
    )
    from sgdjscc_lab.utils.preprocessing import prepare_patches

    patches, patch_meta = prepare_patches(frame)
    patches = patches.to(models.device)
    n_patches = int(patches.shape[0])

    with torch.inference_mode():
        gt_text, canny_data, canny_uncertainty = _extract_semantic_guidance(
            patches, models, cfg, models.device
        )
        encode_features, _encode_features_std = _encode_latent(models.jscc_model, patches)

    if guide_profile not in GUIDE_PROFILES:
        raise ValueError(f"unknown guide_profile={guide_profile!r}")
    resolved_guide_profile = GUIDE_PROFILES[guide_profile]
    edge_wire, uncertainty_wire, guide_action_codes = prepare_guides_for_transport(
        canny_data.detach().cpu() if canny_data is not None else None,
        canny_uncertainty.detach().cpu() if canny_uncertainty is not None else None,
        resolved_guide_profile,
        guide_transmission_ordinal,
    )

    frame_manifest = dict(manifest or {})
    frame_manifest.update({
        "patch_layout": _manifest_patch_layout(patch_meta),
        "n_patches": n_patches,
        "selected_keyframes": [int(x) for x in (selected_keyframes or [])],
        # Two one-digit codes are always present, so profile-name length can
        # never create a false byte saving.  0=transmit, 1=receiver-cache
        # reuse, 2=zero map; packet item presence carries tensor shape/bits.
        "guide_actions": list(guide_action_codes),
    })
    bundle = build_frame_bundle(
        visual_latent_patches=None if visual_is_analog else encode_features.detach().cpu(),
        visual_is_analog=visual_is_analog,
        visual_bit_depth=None if visual_is_analog else bit_depth,
        visual_granularity=granularity,
        visual_channel_dim=1,
        visual_channel_symbols=int(encode_features.numel()),
        caption=(
            [str(caption_override)] * n_patches
            if caption_override is not None else _caption_list(gt_text, n_patches)
        ),
        edge_tensor=edge_wire,
        edge_uncertainty_tensor=uncertainty_wire,
        edge_bit_depth=resolved_guide_profile.edge_bit_depth,
        uncertainty_bit_depth=resolved_guide_profile.uncertainty_bit_depth,
        keyframe_index=keyframe_index,
        manifest=frame_manifest,
        semantic_packet=semantic_packet,
        include_quantization_error_metadata=include_quantization_error_metadata,
        quantization_diagnostics=quantization_diagnostics,
    )
    return serialize_bundle(bundle), int(encode_features.numel())


def reconstruct_frame_from_bundle_bytes(
    data: bytes,
    models,
    cfg,
    digital_step_policy: str = "bitdepth_proxy",
    receiver_guide_cache: Optional[Dict[str, object]] = None,
):
    """Receiver: reconstruct a full frame using only serialized bundle bytes.

    ``digital_step_policy`` selects how the diffusion decoder step is derived
    from the digital channel (see ``pipelines/infer_pipeline.py::
    DIGITAL_STEP_POLICIES``). The bit_depth and (for ``"quant_nmse"``) the
    measured quantization SNR are read from EACH patch's own decoded packet
    metadata (``decoded["visual_metadata"]``) — never from a process-global
    channel object — so this receiver only ever uses what the received bytes
    actually contain.
    """
    import torch

    from sgdjscc_lab.pipelines.infer_pipeline import (
        ForwardArtifacts,
        _compute_power_scalar,
        _compute_step,
        _decode_diffusion,
        _preprocess_soft_edge,
    )
    from sgdjscc_lab.transmission.packet_bundle import decode_frame_bundle
    from sgdjscc_lab.transmission.guide_transport import resolve_received_guides
    from sgdjscc_lab.utils.preprocessing import merge_patches

    if str(cfg.get("mask_method", "none")) != "none":
        raise ValueError(
            "bundle-only receiver currently supports mask_method=none; "
            "sender-only mask statistics must not be passed out of band"
        )

    decoded = decode_frame_bundle(data, dtype=torch.float32, device=str(models.device))
    if decoded["visual_is_analog"]:
        raise ValueError(
            "analog visual samples are not contained in bundle bytes; "
            "bundle-only reconstruction is available for digital configs"
        )
    visual_latents = decoded["visual_latents"] or []
    if not visual_latents:
        raise ValueError("bundle contains no digital visual latent patches")
    visual_metadata = decoded.get("visual_metadata") or []
    if len(visual_metadata) != len(visual_latents):
        raise ValueError(
            f"visual metadata count {len(visual_metadata)} does not match "
            f"visual patch count {len(visual_latents)} — cannot determine "
            "each patch's bit_depth from its own packet"
        )

    manifest = decoded.get("manifest") or {}
    layout = manifest.get("patch_layout") or {}
    positions = layout.get("positions")
    if not positions:
        raise ValueError("bundle manifest is missing patch_layout.positions")
    patch_meta = (
        int(layout["height"]),
        int(layout["width"]),
        [(int(y), int(x)) for y, x in positions],
    )
    if len(visual_latents) != len(positions):
        raise ValueError(
            f"visual patch count {len(visual_latents)} does not match layout count {len(positions)}"
        )

    captions = list(decoded.get("captions") or [])
    if len(captions) == 1 and len(visual_latents) > 1:
        captions *= len(visual_latents)
    if len(captions) != len(visual_latents):
        raise ValueError(
            f"caption count {len(captions)} does not match visual patch count {len(visual_latents)}"
        )

    edge = decoded.get("edge")
    edge_uncertainty = decoded.get("edge_uncertainty")
    default_actions = [0 if edge is not None else 2, 0 if edge_uncertainty is not None else 2]
    edge, edge_uncertainty = resolve_received_guides(
        edge,
        edge_uncertainty,
        manifest.get("guide_actions", default_actions),
        receiver_guide_cache,
    )
    if edge is not None and edge.shape[0] != len(visual_latents):
        raise ValueError("edge patch count does not match visual patch count")
    if edge_uncertainty is not None and edge_uncertainty.shape[0] != len(visual_latents):
        raise ValueError("edge-uncertainty patch count does not match visual patch count")

    jscc = models.jscc_model
    pipe = models.sem_pipeline
    device = models.device
    out_patches = []
    snr_scale = 10 ** (float(jscc.snr) / 10)

    with torch.inference_mode():
        for i, received in enumerate(visual_latents):
            encode_features_hat = jscc.normalize(received.to(device))
            dummy_x = torch.empty(
                encode_features_hat.shape[0], 1, 1, 1,
                device=device, dtype=encode_features_hat.dtype,
            )
            edge_i = edge[i:i + 1].to(device) if edge is not None else None
            unc_i = (
                edge_uncertainty[i:i + 1].to(device)
                if edge_uncertainty is not None else None
            )
            if edge_i is None and unc_i is not None:
                edge_i = torch.zeros_like(unc_i)
            if edge_i is not None and unc_i is None:
                unc_i = torch.zeros_like(edge_i)
            soft_edge, soft_uncertainty = _preprocess_soft_edge(
                edge_i, unc_i, dummy_x, device
            )
            power_scalar = _compute_power_scalar(encode_features_hat, None, dummy_x)
            signal_scale = (snr_scale / (snr_scale + 1)) * torch.ones_like(
                encode_features_hat[:, 0:1, 0, 0]
            )
            meta_i = visual_metadata[i]
            cur_step, cur_snr = _compute_step(
                jscc=jscc,
                encode_features_hat=encode_features_hat,
                power_scalar=power_scalar,
                signal_scale=signal_scale,
                pipe=pipe,
                step_style=str(cfg.step_style),
                use_jscc_feat=bool(cfg.use_jscc_feature),
                use_gt_csi=bool(cfg.use_gt_csi),
                device=device,
                digital_bit_depth=meta_i["bit_depth"],
                digital_policy=digital_step_policy,
                digital_quant_snr_db=meta_i.get("quant_snr_db"),
                digital_reference_snr_db=float(
                    cfg.get("digital_fixed_reference_snr_db", 10.0)
                ),
            )
            artifacts = ForwardArtifacts(
                use_semantic=bool(cfg.use_semantic),
                encode_features_hat=encode_features_hat,
                signal_scale=signal_scale,
                device=device,
                batch_size=encode_features_hat.shape[0],
                mask_token=None,
                power_scalar=power_scalar,
                cur_step=cur_step,
                cur_snr=cur_snr,
                soft_edge_image=soft_edge,
                soft_edge_uncertainty=soft_uncertainty,
            )
            gt_text = [[captions[i]]]
            out = _decode_diffusion(
                artifacts, jscc, pipe, gt_text, cfg, device, original_image=None,
                edge_already_received=True,
            )
            out_patches.append(out.cpu())

    return merge_patches(torch.cat(out_patches, dim=0), patch_meta)
