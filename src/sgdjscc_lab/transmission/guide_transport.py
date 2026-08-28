"""Controlled edge/uncertainty transport profiles for rate ablations.

The profile is applied before :mod:`packet_bundle` serialization.  Omitted
guides never cross the receiver boundary: action ``reuse`` may use only a
receiver-side cache populated from an earlier decoded packet, while action
``zero`` produces a zero map at the receiver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


GUIDE_TRANSMIT = 0
GUIDE_REUSE = 1
GUIDE_ZERO = 2
GUIDE_ACTION_NAMES = {
    GUIDE_TRANSMIT: "transmit",
    GUIDE_REUSE: "reuse",
    GUIDE_ZERO: "zero",
}


@dataclass(frozen=True)
class GuideTransportProfile:
    name: str
    family: str
    stage: str
    edge_bit_depth: int = 8
    uncertainty_bit_depth: int = 8
    edge_downsample: int = 1
    uncertainty_downsample: int = 1
    edge_stride: int = 1
    uncertainty_stride: int = 1
    edge_omit: bool = False
    uncertainty_omit: bool = False


def _profile(
    name: str,
    family: str,
    stage: str = "individual",
    **kwargs,
) -> GuideTransportProfile:
    return GuideTransportProfile(name=name, family=family, stage=stage, **kwargs)


# Five isolated ablations for each guide plus a compact set of combined
# candidates.  The names are stable experiment protocol identifiers and are
# recorded in run signatures/analysis tables, but not copied verbatim into
# the wire manifest (avoids label-length byte artefacts).
GUIDE_PROFILES: Dict[str, GuideTransportProfile] = {
    "baseline": _profile("baseline", "baseline", stage="baseline"),
    "edge_q4": _profile("edge_q4", "edge", edge_bit_depth=4),
    "edge_ds2": _profile("edge_ds2", "edge", edge_downsample=2),
    "edge_ds4": _profile("edge_ds4", "edge", edge_downsample=4),
    "edge_reuse2": _profile("edge_reuse2", "edge", edge_stride=2),
    "edge_omit": _profile("edge_omit", "edge", edge_omit=True),
    "uncertainty_q4": _profile(
        "uncertainty_q4", "uncertainty", uncertainty_bit_depth=4,
    ),
    "uncertainty_ds2": _profile(
        "uncertainty_ds2", "uncertainty", uncertainty_downsample=2,
    ),
    "uncertainty_ds4": _profile(
        "uncertainty_ds4", "uncertainty", uncertainty_downsample=4,
    ),
    "uncertainty_reuse2": _profile(
        "uncertainty_reuse2", "uncertainty", uncertainty_stride=2,
    ),
    "uncertainty_omit": _profile(
        "uncertainty_omit", "uncertainty", uncertainty_omit=True,
    ),
    "combined_q4": _profile(
        "combined_q4", "combined", stage="combined",
        edge_bit_depth=4, uncertainty_bit_depth=4,
    ),
    "combined_ds2": _profile(
        "combined_ds2", "combined", stage="combined",
        edge_downsample=2, uncertainty_downsample=2,
    ),
    "combined_ds4": _profile(
        "combined_ds4", "combined", stage="combined",
        edge_downsample=4, uncertainty_downsample=4,
    ),
    "combined_reuse2": _profile(
        "combined_reuse2", "combined", stage="combined",
        edge_stride=2, uncertainty_stride=2,
    ),
    "combined_q4_ds2_reuse2": _profile(
        "combined_q4_ds2_reuse2", "combined", stage="combined",
        edge_bit_depth=4, uncertainty_bit_depth=4,
        edge_downsample=2, uncertainty_downsample=2,
        edge_stride=2, uncertainty_stride=2,
    ),
    # Follow-up operating-point candidates selected after the 16-profile
    # ablation.  The ``candidate_`` prefix deliberately keeps them out of the
    # historical five-per-family accounting while making the protocol names
    # stable in manifests and resume signatures.
    "candidate_edge_ds4_uncertainty_omit": _profile(
        "candidate_edge_ds4_uncertainty_omit", "candidate", stage="integrated",
        edge_downsample=4, uncertainty_omit=True,
    ),
    "candidate_both_omit": _profile(
        "candidate_both_omit", "candidate", stage="integrated",
        edge_omit=True, uncertainty_omit=True,
    ),
}

DEFAULT_GUIDE_ABLATION_PROFILES = tuple(GUIDE_PROFILES)


def parse_guide_profiles(text: str) -> Tuple[GuideTransportProfile, ...]:
    names = [item.strip() for item in str(text).split(",") if item.strip()]
    if not names:
        raise ValueError("--guide-profiles must contain at least one profile")
    if len(set(names)) != len(names):
        raise ValueError("--guide-profiles contains duplicate profile names")
    unknown = [name for name in names if name not in GUIDE_PROFILES]
    if unknown:
        raise ValueError(
            f"unknown guide profiles {unknown}; expected a subset of "
            f"{list(GUIDE_PROFILES)}"
        )
    return tuple(GUIDE_PROFILES[name] for name in names)


def profile_metadata(profile: GuideTransportProfile) -> Dict[str, object]:
    return {
        "guide_profile": profile.name,
        "guide_family": profile.family,
        "guide_stage": profile.stage,
        "edge_bit_depth": profile.edge_bit_depth,
        "uncertainty_bit_depth": profile.uncertainty_bit_depth,
        "edge_downsample": profile.edge_downsample,
        "uncertainty_downsample": profile.uncertainty_downsample,
        "edge_stride": profile.edge_stride,
        "uncertainty_stride": profile.uncertainty_stride,
        "edge_omit": profile.edge_omit,
        "uncertainty_omit": profile.uncertainty_omit,
    }


def _action(*, omit: bool, stride: int, transmission_ordinal: int) -> int:
    if omit:
        return GUIDE_ZERO
    if stride > 1 and transmission_ordinal % stride != 0:
        return GUIDE_REUSE
    return GUIDE_TRANSMIT


def guide_actions(
    profile: GuideTransportProfile, transmission_ordinal: int,
) -> Tuple[int, int]:
    if transmission_ordinal < 0:
        raise ValueError("transmission_ordinal must be >= 0")
    return (
        _action(
            omit=profile.edge_omit,
            stride=profile.edge_stride,
            transmission_ordinal=transmission_ordinal,
        ),
        _action(
            omit=profile.uncertainty_omit,
            stride=profile.uncertainty_stride,
            transmission_ordinal=transmission_ordinal,
        ),
    )


def downsample_guide(tensor, factor: int):
    """Downsample an ``NCHW`` guide while preserving values/range."""
    if tensor is None or factor == 1:
        return tensor
    if factor not in (2, 4):
        raise ValueError(f"guide downsample factor must be 1, 2, or 4; got {factor}")
    if tensor.ndim != 4:
        raise ValueError(f"guide tensor must be NCHW, got shape={tuple(tensor.shape)}")
    import torch.nn.functional as F

    height, width = tensor.shape[-2:]
    target = (max(int(height) // factor, 1), max(int(width) // factor, 1))
    return F.interpolate(tensor, size=target, mode="bilinear", align_corners=False)


def prepare_guides_for_transport(
    edge,
    uncertainty,
    profile: GuideTransportProfile,
    transmission_ordinal: int,
):
    """Return wire tensors and fixed-width action codes for one transmission."""
    edge_action, uncertainty_action = guide_actions(profile, transmission_ordinal)
    edge_wire = (
        downsample_guide(edge, profile.edge_downsample)
        if edge_action == GUIDE_TRANSMIT else None
    )
    uncertainty_wire = (
        downsample_guide(uncertainty, profile.uncertainty_downsample)
        if uncertainty_action == GUIDE_TRANSMIT else None
    )
    return edge_wire, uncertainty_wire, (edge_action, uncertainty_action)


def restore_guide_resolution(tensor, target_hw: Iterable[int] = (128, 128)):
    if tensor is None:
        return None
    target = tuple(int(value) for value in target_hw)
    if tuple(tensor.shape[-2:]) == target:
        return tensor
    import torch.nn.functional as F

    return F.interpolate(tensor, size=target, mode="bilinear", align_corners=False)


def resolve_received_guides(
    edge,
    uncertainty,
    actions: Iterable[int],
    cache: Optional[Dict[str, object]],
    *,
    target_hw: Iterable[int] = (128, 128),
):
    """Resolve decoded guides using only packet data and receiver state."""
    action_values = tuple(int(value) for value in actions)
    if len(action_values) != 2 or any(value not in GUIDE_ACTION_NAMES for value in action_values):
        raise ValueError(f"invalid guide action codes: {action_values}")
    receiver_cache: Dict[str, object] = cache if cache is not None else {}

    def resolve(name: str, decoded, action: int):
        if action == GUIDE_TRANSMIT:
            if decoded is None:
                raise ValueError(f"guide action says transmit but bundle has no {name} item")
            restored = restore_guide_resolution(decoded, target_hw)
            receiver_cache[name] = restored.detach().clone()
            return restored
        if decoded is not None:
            raise ValueError(f"guide action {GUIDE_ACTION_NAMES[action]} conflicts with {name} item")
        if action == GUIDE_REUSE:
            if name not in receiver_cache:
                raise ValueError(f"guide action says reuse but receiver cache has no {name}")
            return receiver_cache[name].detach().clone()
        # Preserve independent edge/uncertainty ablations: the caller fills a
        # zero tensor from the other component's shape when only one is zero.
        return None

    return (
        resolve("edge", edge, action_values[0]),
        resolve("edge_uncertainty", uncertainty, action_values[1]),
    )
