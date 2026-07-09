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
"""Bounce configuration and pluggable sizing policy. A config enables bounce; leaving it unset keeps
the per-block path. The size knob doubles as the on and off switch."""

import os
from dataclasses import dataclass, field
from typing import Optional

from tensorrt_llm import logger

_MIB = 1024 * 1024

# Opt-in for the structured NHD staging fast path, layered on the bounce size knob:
# bounce off => nothing new runs; bounce on + flag off => merged bounce behavior only.
STRUCTURED_NHD_ENV = "TRTLLM_KV_BOUNCE_STRUCTURED_NHD"


def structured_nhd_from_env() -> bool:
    """Whether the structured-NHD staging flag is set in the environment (default OFF)."""
    return os.environ.get(STRUCTURED_NHD_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


@dataclass(frozen=True)
class SizingContext:
    free_bytes: int  # free at setup, after the cache pool claimed its fraction
    total_bytes: int
    chunk_bytes: int
    device_id: int
    # The manager's KVCachePageTable, so a sizing can derive capacity from the cache
    # geometry (TokenBudgetSizing). None when the caller has no page table.
    page_table: Optional[object] = None


@dataclass(frozen=True)
class Sizing:
    """Returns the byte size of one region; there are two, one for sending and one for receiving."""

    def resolve(self, ctx: SizingContext) -> int:
        raise NotImplementedError


# Default size in MiB per region. Raise it to bounce larger single transfers, lower it to save
# memory. It is clamped to the free-memory budget at setup.
DEFAULT_CAPACITY_MB = 384


@dataclass(frozen=True)
class FixedSizing(Sizing):
    """A fixed capacity per region, clamped to free memory at setup."""

    capacity_mb: int = DEFAULT_CAPACITY_MB

    def resolve(self, ctx: SizingContext) -> int:
        return max(_round_up(self.capacity_mb * _MIB, ctx.chunk_bytes), ctx.chunk_bytes)


def token_budget_bytes(page_table, max_tokens: Optional[int]) -> Optional[int]:
    """Per-region bytes needed to stage ``max_tokens`` worth of structured NHD KV and
    its bounded mapped side pools, or None when it cannot be derived (no page table,
    no positive budget, or no NHD pools).

    Mirrors C++ ``CacheTransBufferManager::computeTransferBufferSize``: round the token
    budget up to blocks, reserve one extra block for speculative/draft-token destination
    capacity, and multiply by the summed per-block bytes of every explicitly sized mapped
    pool view. At least one NHD view is required because this policy is only selected for
    structured NHD staging. Counting bounded non-NHD views covers the single-writer merged
    layout (for example, MiniMax INDEX_KEY) without changing legacy-only page tables."""
    if page_table is None or max_tokens is None or max_tokens <= 0:
        return None
    # Local import: keep this module importable standalone (mirrors impl.block_bytes_per_group).
    from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup, MapperKind

    tokens_per_block = int(page_table.tokens_per_block)
    if tokens_per_block <= 0:
        return None
    bytes_per_block = 0
    has_nhd_view = False
    for layer_group in page_table.layer_groups:
        if not isinstance(layer_group, AttentionLayerGroup):
            continue
        for view in layer_group.pool_views:
            if view.bytes_per_region is None:
                continue
            bytes_per_block += int(view.bytes_per_region)
            has_nhd_view = has_nhd_view or view.mapper_kind == MapperKind.NHD
    if not has_nhd_view or bytes_per_block == 0:
        return None
    max_blocks = (max_tokens + tokens_per_block - 1) // tokens_per_block + 1
    return max_blocks * bytes_per_block


@dataclass(frozen=True)
class TokenBudgetSizing(Sizing):
    """Capacity derived from the transceiver's token budget and the cache geometry:
    ``(ceil(max_tokens / tokens_per_block) + 1) x sum(mapped pools' bytes_per_region)``,
    including bounded side pools that single-writer structured staging coalesces. This is
    the direct analog of C++ ``CacheTransBufferManager::computeTransferBufferSize``.
    Resolves to the fixed default when the geometry yields no NHD pool bytes (never
    raises: an unusual page table degrades to FixedSizing, not a crash)."""

    max_tokens: int
    fallback_mb: int = DEFAULT_CAPACITY_MB

    def resolve(self, ctx: SizingContext) -> int:
        capacity = token_budget_bytes(ctx.page_table, self.max_tokens)
        if capacity is None:
            return FixedSizing(capacity_mb=self.fallback_mb).resolve(ctx)
        return max(_round_up(capacity, ctx.chunk_bytes), ctx.chunk_bytes)


# bounce takes at most this fraction of the free memory left after the cache pool
_HEADROOM_FRACTION = 0.5


def fit_within_free(
    capacity_bytes: int,
    *,
    free_bytes: int,
    chunk_bytes: int,
    max_free_fraction: float = _HEADROOM_FRACTION,
) -> Optional[int]:
    """Clamp each region so the two together stay within the allowed fraction of free memory, rounded
    to a chunk. Returns None if not even one chunk fits."""
    budget_per_dir = (int(free_bytes * max_free_fraction) // 2 // chunk_bytes) * chunk_bytes
    if budget_per_dir < chunk_bytes:
        return None
    capacity_bytes = min(capacity_bytes, budget_per_dir)
    capacity_bytes = max(capacity_bytes, chunk_bytes)
    return capacity_bytes


@dataclass
class Config:
    sizing: Sizing = field(default_factory=FixedSizing)  # how much memory to reserve (pluggable)
    chunk_mb: int = 32  # physical chunk size; a large chunk keeps the write to a single descriptor
    # skip bounce below this many blocks (roughly 12k tokens at 128 per block); heuristic, tunable
    min_blocks: int = 96
    # analytic NHD head-mismatch staging plans instead of materialized fragment tables;
    # default OFF, driven by TRTLLM_KV_BOUNCE_STRUCTURED_NHD until promoted to a
    # CacheTransceiverConfig field
    structured_nhd: bool = False


def config_from_size(size_mb: int, structured_nhd: Optional[bool] = None) -> Optional[Config]:
    """Build a bounce config from a per-region size in MiB, or None to leave bounce off when the size
    is not positive. The size is both the capacity and the on and off switch. ``structured_nhd``
    defaults to the TRTLLM_KV_BOUNCE_STRUCTURED_NHD environment flag."""
    if size_mb is None or size_mb <= 0:
        return None
    if structured_nhd is None:
        structured_nhd = structured_nhd_from_env()
    return Config(sizing=FixedSizing(capacity_mb=size_mb), structured_nhd=structured_nhd)


def resolve_bounce_config(
    size_mb: Optional[int],
    *,
    max_tokens_in_buffer: Optional[int] = None,
    structured_nhd: Optional[bool] = None,
) -> Optional[Config]:
    """Resolve the two user knobs into a bounce config.

    - size set: bounce on at that fixed capacity (:func:`config_from_size`).
    - size unset/0 + structured flag set: auto-enable bounce — the structured fast path
      is useless without a staging arena, so a bare TRTLLM_KV_BOUNCE_STRUCTURED_NHD=1
      must not silently do nothing. Capacity comes from :class:`TokenBudgetSizing` when
      ``max_tokens_in_buffer`` is a positive token budget, else the fixed default.
    - neither: bounce off (None).

    ``structured_nhd`` defaults to the TRTLLM_KV_BOUNCE_STRUCTURED_NHD environment flag.
    """
    if structured_nhd is None:
        structured_nhd = structured_nhd_from_env()
    cfg = config_from_size(size_mb, structured_nhd=structured_nhd)
    if cfg is not None or not structured_nhd:
        return cfg
    if max_tokens_in_buffer is not None and max_tokens_in_buffer > 0:
        sizing: Sizing = TokenBudgetSizing(max_tokens=max_tokens_in_buffer)
        sizing_desc = f"TokenBudgetSizing(max_tokens={max_tokens_in_buffer})"
    else:
        sizing = FixedSizing()
        sizing_desc = f"FixedSizing({DEFAULT_CAPACITY_MB}MiB)"
    logger.info(
        f"[kv-bounce] auto-enabled by {STRUCTURED_NHD_ENV}=1 with {sizing_desc}: "
        "kv_cache_bounce_size_mb is unset and the structured NHD staging path needs a "
        "bounce arena; set kv_cache_bounce_size_mb to pick the capacity explicitly. "
        "TokenBudgetSizing degrades to FixedSizing at transport setup when the page "
        "table has no explicitly sized NHD pool views (V1/legacy geometries)."
    )
    return Config(sizing=sizing, structured_nhd=True)
