from __future__ import annotations

import pytest
import torch

from sgdjscc_lab.transmission.guide_transport import (
    DEFAULT_GUIDE_ABLATION_PROFILES,
    GUIDE_PROFILES,
    GUIDE_REUSE,
    GUIDE_TRANSMIT,
    GUIDE_ZERO,
    parse_guide_profiles,
    prepare_guides_for_transport,
    resolve_received_guides,
)


def test_protocol_contains_five_isolated_profiles_per_component_and_combined_candidates():
    assert DEFAULT_GUIDE_ABLATION_PROFILES[0] == "baseline"
    assert len([name for name in GUIDE_PROFILES if name.startswith("edge_")]) == 5
    assert len([name for name in GUIDE_PROFILES if name.startswith("uncertainty_")]) == 5
    assert len([name for name in GUIDE_PROFILES if name.startswith("combined_")]) == 5
    assert len([name for name in GUIDE_PROFILES if name.startswith("candidate_")]) == 2
    assert len(DEFAULT_GUIDE_ABLATION_PROFILES) == 18


def test_profile_parser_rejects_unknown_and_duplicate_names():
    assert [profile.name for profile in parse_guide_profiles("baseline,edge_q4")] == [
        "baseline", "edge_q4",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        parse_guide_profiles("baseline,baseline")
    with pytest.raises(ValueError, match="unknown"):
        parse_guide_profiles("not_a_profile")


@pytest.mark.parametrize("name,factor", [("edge_ds2", 2), ("edge_ds4", 4)])
def test_edge_downsample_profiles_change_only_edge_wire_shape(name, factor):
    edge = torch.rand(2, 11, 128, 128)
    uncertainty = torch.rand_like(edge)
    edge_wire, uncertainty_wire, actions = prepare_guides_for_transport(
        edge, uncertainty, GUIDE_PROFILES[name], 0,
    )
    assert edge_wire.shape[-2:] == (128 // factor, 128 // factor)
    assert uncertainty_wire.shape == uncertainty.shape
    assert actions == (GUIDE_TRANSMIT, GUIDE_TRANSMIT)


def test_reuse_profile_sends_first_then_uses_receiver_cache_only():
    edge = torch.rand(1, 11, 128, 128)
    uncertainty = torch.rand_like(edge)
    first_edge, first_uncertainty, first_actions = prepare_guides_for_transport(
        edge, uncertainty, GUIDE_PROFILES["edge_reuse2"], 0,
    )
    cache = {}
    resolved_first = resolve_received_guides(
        first_edge, first_uncertainty, first_actions, cache,
    )
    second_edge, second_uncertainty, second_actions = prepare_guides_for_transport(
        edge * 0, uncertainty * 0, GUIDE_PROFILES["edge_reuse2"], 1,
    )
    assert second_edge is None
    assert second_actions == (GUIDE_REUSE, GUIDE_TRANSMIT)
    resolved_second = resolve_received_guides(
        second_edge, second_uncertainty, second_actions, cache,
    )
    assert torch.equal(resolved_second[0], resolved_first[0])
    assert torch.equal(resolved_second[1], torch.zeros_like(resolved_second[1]))


def test_reuse_without_prior_received_packet_is_rejected():
    with pytest.raises(ValueError, match="receiver cache"):
        resolve_received_guides(None, torch.zeros(1, 11, 128, 128), [1, 0], {})


def test_omit_is_zero_action_and_does_not_put_tensor_on_wire():
    edge = torch.rand(1, 11, 128, 128)
    uncertainty = torch.rand_like(edge)
    edge_wire, uncertainty_wire, actions = prepare_guides_for_transport(
        edge, uncertainty, GUIDE_PROFILES["uncertainty_omit"], 0,
    )
    assert edge_wire is not None
    assert uncertainty_wire is None
    assert actions == (GUIDE_TRANSMIT, GUIDE_ZERO)


def test_integrated_candidates_have_expected_independent_actions():
    edge = torch.rand(1, 11, 128, 128)
    uncertainty = torch.rand_like(edge)
    edge_wire, uncertainty_wire, actions = prepare_guides_for_transport(
        edge, uncertainty, GUIDE_PROFILES["candidate_edge_ds4_uncertainty_omit"], 0,
    )
    assert edge_wire.shape[-2:] == (32, 32)
    assert uncertainty_wire is None
    assert actions == (GUIDE_TRANSMIT, GUIDE_ZERO)
    edge_wire, uncertainty_wire, actions = prepare_guides_for_transport(
        edge, uncertainty, GUIDE_PROFILES["candidate_both_omit"], 0,
    )
    assert edge_wire is None and uncertainty_wire is None
    assert actions == (GUIDE_ZERO, GUIDE_ZERO)
