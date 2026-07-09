# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Analytic ("structured") staging plans for NHD head-mismatched KV bounce transfers.

An NHD head-mismatch transfer is perfectly regular: every fragment address is
``block_ptrs[b] + flat_offsets[k]`` with one constant fragment size, so instead of
materializing millions of per-fragment pointer entries (gather_scatter.Plan), the whole
transfer is described by ``O(n_blocks + n_frags_per_block)`` integers. This module holds
that descriptor and its numpy expansion, which is both the reference the tests compare
against and the source of truth for the fallback copy loop.

The module itself is pure numpy, no CUDA or Triton imports (mirrors core.py), so its
logic is unit-testable on CPU — but importing it through the bounce package executes
``bounce/__init__``, which requires cuda-python (cudart); the CPU unit tests skip when
that is absent. The kernels that consume these specs live in gather_scatter.py.
"""

import ctypes
import struct
from dataclasses import dataclass

import numpy as np

__all__ = [
    "NHDGatherSpec",
    "NHDResultTail",
    "NHDScatterTemplate",
    "copy_expanded_host",
    "decode_nhd_tail",
    "encode_nhd_tail",
    "expand_specs",
    "is_nhd_tail",
    "specs_total_bytes",
]


@dataclass(frozen=True)
class NHDGatherSpec:
    """Analytic gather/scatter plan for one NHD head-mismatch pool pair.

    Fragment ``k`` of block ``b`` lives at ``block_ptrs[b] + flat_offsets[k]`` on the
    paged side and occupies ``frag_bytes`` bytes. Fragments are staged contiguously in
    canonical order — blocks outer, ``flat_offsets`` inner — which is exactly the order
    ``NHDHeadMismatchMapper.map`` emits (``np.add.outer(ptrs, offsets).ravel()``), so a
    structured writer and a materialized (fallback) writer produce identical staging
    bytes. The same spec describes both directions: gather reads the paged side and
    writes the staging region, scatter is the inverse.
    """

    block_ptrs: np.ndarray  # [n_blocks] int64 absolute paged block base addresses
    flat_offsets: np.ndarray  # [n_frags_per_block] int64 byte offsets within one block
    frag_bytes: int  # bytes per fragment: min(self, peer) heads x bytes per token/head

    def __post_init__(self) -> None:
        for name, arr in (("block_ptrs", self.block_ptrs), ("flat_offsets", self.flat_offsets)):
            if not isinstance(arr, np.ndarray) or arr.ndim != 1 or arr.dtype != np.int64:
                raise ValueError(f"NHDGatherSpec.{name} must be a 1-D int64 ndarray, got {arr!r}")
        if not isinstance(self.frag_bytes, int) or self.frag_bytes <= 0:
            raise ValueError(
                f"NHDGatherSpec.frag_bytes must be a positive int, got {self.frag_bytes!r}"
            )

    @property
    def n_frags_per_block(self) -> int:
        return int(self.flat_offsets.size)

    @property
    def n_frags(self) -> int:
        return int(self.block_ptrs.size) * self.n_frags_per_block

    @property
    def total_bytes(self) -> int:
        return self.n_frags * self.frag_bytes

    def paged_ptrs(self) -> np.ndarray:
        """Absolute paged-side fragment addresses in canonical order, one per fragment.

        This is the materialization the structured path avoids on the hot path; it is
        the numpy reference for tests and the fallback, byte-identical to what
        ``NHDHeadMismatchMapper.map`` produces for the same block pointers.
        """
        return np.add.outer(self.block_ptrs, self.flat_offsets).ravel()

    def staged_offsets(self) -> np.ndarray:
        """Byte offset of each fragment within this spec's staging section."""
        return np.arange(self.n_frags, dtype=np.int64) * self.frag_bytes


def specs_total_bytes(specs: list[NHDGatherSpec]) -> int:
    """Total staging bytes across all sections, i.e. the region size to reserve."""
    return sum(spec.total_bytes for spec in specs)


def expand_specs(
    staging_base: int, specs: list[NHDGatherSpec]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the per-fragment tables the specs stand for (the numpy reference).

    Sections are staged back to back starting at ``staging_base``, each in canonical
    order, matching gather_scatter's running-offset layout for the equivalent Plan.

    Returns:
        ``(paged_ptrs, staged_ptrs, sizes)`` int64 arrays, one entry per fragment. For a
        gather the paged side is the source and the staged side the destination; for a
        scatter the roles swap.
    """
    paged: list[np.ndarray] = []
    staged: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    base = int(staging_base)
    for spec in specs:
        paged.append(spec.paged_ptrs())
        staged.append(base + spec.staged_offsets())
        sizes.append(np.full(spec.n_frags, spec.frag_bytes, dtype=np.int64))
        base += spec.total_bytes
    if not specs:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    return np.concatenate(paged), np.concatenate(staged), np.concatenate(sizes)


def copy_expanded_host(
    paged_ptrs: np.ndarray, staged_ptrs: np.ndarray, sizes: np.ndarray, *, gather: bool
) -> None:
    """Reference copy over host memory: memmove every fragment of an expanded plan.

    The addresses must be real host addresses (e.g. ``array.ctypes.data`` plus offsets).
    Used by the CPU unit tests and by the GPU test as the ground truth the Triton kernel
    is compared against.

    Args:
        paged_ptrs: [n_frags] int64 absolute paged-side fragment addresses.
        staged_ptrs: [n_frags] int64 absolute staging-side fragment addresses.
        sizes: [n_frags] int64 bytes per fragment.
        gather: copy paged -> staged when True, staged -> paged when False.
    """
    if not (paged_ptrs.size == staged_ptrs.size == sizes.size):
        raise ValueError(
            f"expanded plan arrays disagree in length: paged={paged_ptrs.size}, "
            f"staged={staged_ptrs.size}, sizes={sizes.size}"
        )
    for k in range(int(sizes.size)):
        paged_addr, staged_addr, n = int(paged_ptrs[k]), int(staged_ptrs[k]), int(sizes[k])
        dst, src = (staged_addr, paged_addr) if gather else (paged_addr, staged_addr)
        ctypes.memmove(dst, src, n)


@dataclass(frozen=True)
class NHDScatterTemplate:
    """Receiver-side scatter template for one structured pool pair.

    Built once at reserve() time from the receiver's OWN page table and block ids
    (``block_ptrs`` covers every block the receiver advertised for the pool's layer
    group). When a writer's structured result tail arrives, :meth:`spec_for` slices the
    template with the sender-side alignment facts carried in the tail (``dst_skip``,
    ``n_blocks``) and rebases the flat offsets to the reporting fan-in writer's head
    sub-range — writer ``i`` contributes heads at byte offset
    ``i * writer_head_stride`` within the receiver's per-token head span, exactly the
    assignment ``HeadMismatchMapper._compute_head_offsets`` makes for the sender's
    tp_rank (fan-in writers are indexed in ``PeerOverlap.ranks`` order, tp innermost).

    ``flat_offsets`` is the writer-0 plan: the receiver's self-side offsets with head
    offset zero.
    """

    block_ptrs: np.ndarray  # [n_blocks_total] int64 receiver paged block base addresses
    flat_offsets: np.ndarray  # [n_frags_per_block] int64, fan-in writer 0 (head offset 0)
    frag_bytes: int  # bytes per fragment: min(self, peer) heads x bytes per token/head
    writer_head_stride: int  # byte offset added per fan-in writer index

    def __post_init__(self) -> None:
        for name, arr in (("block_ptrs", self.block_ptrs), ("flat_offsets", self.flat_offsets)):
            if not isinstance(arr, np.ndarray) or arr.ndim != 1 or arr.dtype != np.int64:
                raise ValueError(
                    f"NHDScatterTemplate.{name} must be a 1-D int64 ndarray, got {arr!r}"
                )
        if not isinstance(self.frag_bytes, int) or self.frag_bytes <= 0:
            raise ValueError(
                f"NHDScatterTemplate.frag_bytes must be a positive int, got {self.frag_bytes!r}"
            )
        if not isinstance(self.writer_head_stride, int) or self.writer_head_stride < 0:
            raise ValueError(
                "NHDScatterTemplate.writer_head_stride must be a non-negative int, "
                f"got {self.writer_head_stride!r}"
            )

    @property
    def total_bytes(self) -> int:
        """Return the bytes needed to stage every advertised block for one writer."""
        return int(self.block_ptrs.size) * int(self.flat_offsets.size) * self.frag_bytes

    def spec_for(self, writer_index: int, dst_skip: int, n_blocks: int) -> NHDGatherSpec:
        """The scatter plan for one writer's section: its aligned block slice, its head
        sub-range. Raises ValueError when the tail's alignment facts do not fit this
        template (corrupt or mismatched tail must never scatter)."""
        if writer_index < 0:
            raise ValueError(f"writer_index must be >= 0, got {writer_index}")
        if dst_skip < 0 or n_blocks < 0 or dst_skip + n_blocks > int(self.block_ptrs.size):
            raise ValueError(
                f"block slice [{dst_skip}, {dst_skip + n_blocks}) is outside the "
                f"{int(self.block_ptrs.size)}-block template"
            )
        offsets = self.flat_offsets
        if writer_index > 0:
            offsets = offsets + writer_index * self.writer_head_stride
        return NHDGatherSpec(
            block_ptrs=np.ascontiguousarray(self.block_ptrs[dst_skip : dst_skip + n_blocks]),
            flat_offsets=offsets,
            frag_bytes=self.frag_bytes,
        )


# --------------------------------------------------------------------------- #
# Structured result-tail wire format (KV_AGENT_RESULT trailing frame)
# --------------------------------------------------------------------------- #
# A structured writer replaces the legacy tail (full dst pointer + size tables, ~16 MB
# for a long head-mismatched request) with one small frame: the writer's staging base,
# one alignment record per structured section (only the sender-side block alignment the
# receiver cannot derive locally), and the materialized remainder tables for any
# trailing non-structured sections (replicated side caches — whole-block fragments, so
# the tables stay tiny). Layout, little-endian:
#
#   magic u8 | src_base i64 | n_records u16
#   (dst_lg u16, dst_pool u16, dst_skip u32, n_blocks u32) x n_records
#   n_rest u32 | rest_dst_ptrs int64 x n_rest | rest_sizes int64 x n_rest
#
# The struct fields are explicitly little-endian while the numpy rest tables use
# native byte order (tobytes/frombuffer) — consistent only on an LE host, which
# covers all supported platforms (x86-64, aarch64).
_NHD_TAIL_MAGIC = 0xB7
_TAIL_HEADER = struct.Struct("<BqH")
_TAIL_RECORD = struct.Struct("<HHII")
_TAIL_REST = struct.Struct("<I")
_U16_MAX = 0xFFFF
_U32_MAX = 0xFFFFFFFF


@dataclass(frozen=True)
class NHDResultTail:
    """Decoded structured result tail: everything the receiver needs, beyond its own
    reserve-time scatter templates, to scatter one writer's staging section."""

    src_base: int  # where this fan-in writer wrote within the recv region
    # per structured section, in staging order: the receiver-side pool key the section
    # scatters into and the sender's block alignment (head skip + aligned length)
    records: tuple[tuple[int, int, int, int], ...]  # (dst_lg, dst_pool, dst_skip, n_blocks)
    rest_dst_ptrs: np.ndarray  # materialized remainder (non-structured sections), staged last
    rest_sizes: np.ndarray


def encode_nhd_tail(tail: NHDResultTail) -> bytes:
    """Serialize a structured result tail into one wire frame."""
    if tail.rest_dst_ptrs.size != tail.rest_sizes.size:
        raise ValueError(
            f"rest tables disagree in length: dst={tail.rest_dst_ptrs.size}, "
            f"sizes={tail.rest_sizes.size}"
        )
    if len(tail.records) > _U16_MAX:
        raise ValueError(f"too many structured sections: {len(tail.records)}")
    parts = [_TAIL_HEADER.pack(_NHD_TAIL_MAGIC, int(tail.src_base), len(tail.records))]
    for dst_lg, dst_pool, dst_skip, n_blocks in tail.records:
        if not (0 <= dst_lg <= _U16_MAX and 0 <= dst_pool <= _U16_MAX):
            raise ValueError(f"pool key ({dst_lg}, {dst_pool}) does not fit u16")
        if not (0 <= dst_skip <= _U32_MAX and 0 <= n_blocks <= _U32_MAX):
            raise ValueError(f"block alignment ({dst_skip}, {n_blocks}) does not fit u32")
        parts.append(_TAIL_RECORD.pack(dst_lg, dst_pool, dst_skip, n_blocks))
    rest_dst = np.ascontiguousarray(tail.rest_dst_ptrs, dtype=np.int64)
    rest_sizes = np.ascontiguousarray(tail.rest_sizes, dtype=np.int64)
    parts.append(_TAIL_REST.pack(int(rest_dst.size)))
    parts.append(rest_dst.tobytes())
    parts.append(rest_sizes.tobytes())
    return b"".join(parts)


def is_nhd_tail(blob: bytes) -> bool:
    """Whether a tail frame carries the structured format (magic byte check)."""
    return len(blob) >= 1 and blob[0] == _NHD_TAIL_MAGIC


def decode_nhd_tail(blob: bytes) -> NHDResultTail:
    """Parse a structured result tail; raises ValueError on any malformed input."""
    if len(blob) < _TAIL_HEADER.size:
        raise ValueError(f"structured tail truncated: {len(blob)} bytes")
    magic, src_base, n_records = _TAIL_HEADER.unpack_from(blob, 0)
    if magic != _NHD_TAIL_MAGIC:
        raise ValueError(f"bad structured tail magic: {magic:#x}")
    pos = _TAIL_HEADER.size
    if len(blob) < pos + n_records * _TAIL_RECORD.size + _TAIL_REST.size:
        raise ValueError(f"structured tail truncated: {len(blob)} bytes, {n_records} records")
    records = []
    for _ in range(n_records):
        records.append(_TAIL_RECORD.unpack_from(blob, pos))
        pos += _TAIL_RECORD.size
    (n_rest,) = _TAIL_REST.unpack_from(blob, pos)
    pos += _TAIL_REST.size
    rest_bytes = n_rest * np.dtype(np.int64).itemsize
    if len(blob) != pos + 2 * rest_bytes:
        raise ValueError(
            f"structured tail length mismatch: {len(blob)} bytes, expected {pos + 2 * rest_bytes}"
        )
    rest_dst = np.frombuffer(blob, dtype=np.int64, count=n_rest, offset=pos)
    rest_sizes = np.frombuffer(blob, dtype=np.int64, count=n_rest, offset=pos + rest_bytes)
    return NHDResultTail(
        src_base=int(src_base),
        records=tuple(records),
        rest_dst_ptrs=rest_dst,
        rest_sizes=rest_sizes,
    )
