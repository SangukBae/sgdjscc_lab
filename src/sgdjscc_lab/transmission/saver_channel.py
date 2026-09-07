"""Packet loss/reorder/duplicate/corruption simulator for SAVER state packets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from sgdjscc_lab.models.saver.contracts import ProtectionAction

from .saver_packet import (
    ApplyStatus,
    SignedPacketError,
    VersionedEntityLedger,
    parse_signed_packet,
)


@dataclass(frozen=True)
class SignedPacketFaultProfile:
    loss_probability: float = 0.0
    duplicate_probability: float = 0.0
    corruption_probability: float = 0.0
    reorder_window: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("loss_probability", "duplicate_probability", "corruption_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.reorder_window <= 0:
            raise ValueError("reorder_window must be positive")


@dataclass(frozen=True)
class PacketAttempt:
    source_index: int
    replica_index: int
    wire: bytes
    dropped: bool
    corrupted: bool


@dataclass(frozen=True)
class SignedPacketChannelReport:
    attempts: Tuple[PacketAttempt, ...]
    delivered: Tuple[PacketAttempt, ...]
    total_on_wire_bytes: int
    delivered_bytes: int


def _corrupt(wire: bytes, rng: random.Random) -> bytes:
    if not wire:
        return wire
    value = bytearray(wire)
    # Exclude the magic when possible so corruption is diagnosed by CRC rather
    # than being accidentally interpreted as another packet family.
    index = rng.randrange(4, len(value)) if len(value) > 4 else rng.randrange(len(value))
    value[index] ^= 1 << rng.randrange(8)
    return bytes(value)


class SignedPacketChannel:
    """Apply a reproducible fault profile while retaining exact byte counts."""

    def __init__(self, profile: SignedPacketFaultProfile) -> None:
        self.profile = profile

    def transmit(
        self,
        wires: Sequence[bytes],
        *,
        forced_order: Optional[Sequence[int]] = None,
    ) -> SignedPacketChannelReport:
        rng = random.Random(self.profile.seed)
        attempts: List[PacketAttempt] = []
        for source_index, raw in enumerate(wires):
            wire = bytes(raw)
            packet = parse_signed_packet(wire)
            replicas = 2 if packet.protection == ProtectionAction.REPETITION else 1
            if rng.random() < self.profile.duplicate_probability:
                replicas += 1
            for replica in range(replicas):
                dropped = rng.random() < self.profile.loss_probability
                corrupted = not dropped and rng.random() < self.profile.corruption_probability
                candidate = _corrupt(wire, rng) if corrupted else wire
                attempts.append(PacketAttempt(
                    source_index=source_index,
                    replica_index=replica,
                    wire=candidate,
                    dropped=dropped,
                    corrupted=corrupted,
                ))

        delivered = [attempt for attempt in attempts if not attempt.dropped]
        if forced_order is not None:
            order = list(int(x) for x in forced_order)
            if sorted(order) != list(range(len(delivered))):
                raise ValueError("forced_order must be a permutation of delivered attempts")
            delivered = [delivered[index] for index in order]
        elif self.profile.reorder_window > 1:
            reordered = []
            for start in range(0, len(delivered), self.profile.reorder_window):
                window = delivered[start:start + self.profile.reorder_window]
                rng.shuffle(window)
                reordered.extend(window)
            delivered = reordered
        return SignedPacketChannelReport(
            attempts=tuple(attempts),
            delivered=tuple(delivered),
            total_on_wire_bytes=sum(len(attempt.wire) for attempt in attempts),
            delivered_bytes=sum(len(attempt.wire) for attempt in delivered),
        )

    @staticmethod
    def apply_report(
        report: SignedPacketChannelReport,
        ledger: VersionedEntityLedger,
    ) -> Tuple[ApplyStatus, ...]:
        statuses = []
        for attempt in report.delivered:
            try:
                packet = parse_signed_packet(attempt.wire)
            except SignedPacketError:
                statuses.append(ApplyStatus.CORRUPT)
                continue
            statuses.append(ledger.apply(packet))
        return tuple(statuses)
