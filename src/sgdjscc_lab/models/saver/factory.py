"""Config-driven construction of the opt-in SAVER tensor pipeline."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from sgdjscc_lab.pipelines.saver_video_pipeline import SaverVideoPipeline

from .contracts import SaverCheckpointContract
from .joint_assertion_symbol_router import JointAssertionSymbolRouter
from .semantic_channel_codec import ActionConditionedSemanticChannelCodec
from .signed_assertion_tokenizer import SignedAssertionTokenizer
from .signed_memory_dit import SignedMemoryDiTAdapterStack
from .versioned_entity_memory import VersionedRevocableEntityMemory


def _get(cfg: DictConfig, key: str, default):
    return OmegaConf.select(cfg, f"saver.{key}", default=default)


def checkpoint_contract_from_config(cfg: DictConfig) -> SaverCheckpointContract:
    return SaverCheckpointContract(
        architecture_version=str(_get(cfg, "architecture_version", "saver-jscc-v1")),
        packet_schema_version=int(_get(cfg, "packet_schema_version", 1)),
        slot_count=int(_get(cfg, "slot_count", 16)),
        memory_dim=int(_get(cfg, "memory_dim", 512)),
        injection_layers=tuple(int(x) for x in _get(cfg, "dit.injection_layers", [1, 2, 3])),
    )


def build_saver_video_pipeline(cfg: DictConfig) -> SaverVideoPipeline:
    """Build SAVER only from the explicit ``saver`` experiment block."""

    contract = checkpoint_contract_from_config(cfg)
    input_dim = int(_get(cfg, "source_feature_dim", contract.memory_dim))
    channel_feature_dim = int(_get(cfg, "router.channel_feature_dim", 8))
    channel_token_dim = int(_get(cfg, "dit.channel_token_dim", 8))
    hidden_dim = int(_get(cfg, "dit.hidden_dim", contract.memory_dim))
    timestep_dim = int(_get(cfg, "dit.timestep_dim", hidden_dim))
    heads = int(_get(cfg, "num_heads", 8))
    max_symbols = int(_get(cfg, "codec.max_symbols", 64))
    tokenizer = SignedAssertionTokenizer(
        input_dim=input_dim,
        memory_dim=contract.memory_dim,
        slot_count=contract.slot_count,
        num_heads=heads,
    )
    router = JointAssertionSymbolRouter(
        token_dim=contract.memory_dim,
        channel_dim=channel_feature_dim,
        rate_options=tuple(int(x) for x in _get(cfg, "router.rate_options", [0, 8, 16, 32, 64])),
        visual_rate_options=tuple(
            int(x) for x in _get(cfg, "router.visual_rate_options", [0, 128, 256, 512])
        ),
        protection_extra_symbols=tuple(
            int(x) for x in _get(cfg, "router.protection_extra_symbols", [0, 8, 4, 16])
        ),
        hidden_dim=int(_get(cfg, "router.hidden_dim", 128)),
        temperature=float(_get(cfg, "router.temperature", 1.0)),
    )
    codec = ActionConditionedSemanticChannelCodec(
        token_dim=contract.memory_dim,
        max_symbols=max_symbols,
        hidden_dim=int(_get(cfg, "codec.hidden_dim", 256)),
    )
    memory = VersionedRevocableEntityMemory(
        memory_dim=contract.memory_dim, slot_count=contract.slot_count
    )
    adapters = SignedMemoryDiTAdapterStack(
        contract.injection_layers,
        hidden_dim=hidden_dim,
        memory_dim=contract.memory_dim,
        channel_dim=channel_token_dim,
        timestep_dim=timestep_dim,
        bottleneck_dim=int(_get(cfg, "dit.bottleneck_dim", contract.memory_dim)),
        num_heads=heads,
    )
    return SaverVideoPipeline(
        tokenizer, router, codec, memory, adapters,
        enabled=bool(OmegaConf.select(cfg, "use_saver_jscc", default=False)),
    )

