"""SAVER-JSCC model components and stable contracts.

Neural components are exported lazily.  The wire-packet module imports the
contract enums, while the semantic codec imports the wire-packet module;
eagerly importing every component here would therefore create a packet/codec
cycle for otherwise valid standalone packet consumers.
"""

from __future__ import annotations

from importlib import import_module

from .contracts import (
    AssertionAction,
    EntityObservation,
    ProtectionAction,
    RenderStatus,
    SaverCheckpointContract,
    SemanticState,
    SignedEntityEvent,
    is_state_action_allowed,
    stable_entity_numeric_id,
)

_LAZY_EXPORTS = {
    "SignedAssertionOutput": ("signed_assertion_tokenizer", "SignedAssertionOutput"),
    "SignedAssertionTokenizer": ("signed_assertion_tokenizer", "SignedAssertionTokenizer"),
    "JointAssertionSymbolRouter": ("joint_assertion_symbol_router", "JointAssertionSymbolRouter"),
    "RouterOutput": ("joint_assertion_symbol_router", "RouterOutput"),
    "ActionConditionedSemanticChannelCodec": (
        "semantic_channel_codec", "ActionConditionedSemanticChannelCodec"
    ),
    "WirelessChannelObservation": ("semantic_channel_codec", "WirelessChannelObservation"),
    "WirelessSemanticSymbols": ("semantic_channel_codec", "WirelessSemanticSymbols"),
    "SignedMemoryAdapterOutput": ("signed_memory_dit", "SignedMemoryAdapterOutput"),
    "SignedMemoryDiTAdapterStack": ("signed_memory_dit", "SignedMemoryDiTAdapterStack"),
    "SignedMemoryDiTBlock": ("signed_memory_dit", "SignedMemoryDiTBlock"),
    "MemoryUpdateOutput": ("versioned_entity_memory", "MemoryUpdateOutput"),
    "VersionedMemoryState": ("versioned_entity_memory", "VersionedMemoryState"),
    "VersionedRevocableEntityMemory": (
        "versioned_entity_memory", "VersionedRevocableEntityMemory"
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, symbol = _LAZY_EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), symbol)
    globals()[name] = value
    return value

__all__ = [
    "AssertionAction",
    "EntityObservation",
    "ProtectionAction",
    "RenderStatus",
    "SaverCheckpointContract",
    "SemanticState",
    "SignedEntityEvent",
    "is_state_action_allowed",
    "stable_entity_numeric_id",
    "SignedAssertionOutput",
    "SignedAssertionTokenizer",
    "JointAssertionSymbolRouter",
    "RouterOutput",
    "ActionConditionedSemanticChannelCodec",
    "WirelessChannelObservation",
    "WirelessSemanticSymbols",
    "SignedMemoryAdapterOutput",
    "SignedMemoryDiTAdapterStack",
    "SignedMemoryDiTBlock",
    "MemoryUpdateOutput",
    "VersionedMemoryState",
    "VersionedRevocableEntityMemory",
]
