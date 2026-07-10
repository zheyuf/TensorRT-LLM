# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA-backed MiniMax-M3 sparse attention on the `TrtllmAttention` stack.

Mimics `DSATrtllmAttention`:

  * `MiniMaxM3MSATrtllmAttention` subclasses `TrtllmAttention` and reuses
    its inherited `forward`, overriding only the sparse hooks
    (`sparse_attn_predict`, `sparse_kv_predict`) and owning an `MsaIndexer`.
  * The main sparse GQA runs through the registered `MsaSparseGqaFmha`.
  * The indexer calls `fmha_sm100` directly (prefill) or the graph-safe
    decode kernels (decode) to produce the per-query selected block
    indices, which the model layer threads through
    `forward_args.topk_indices`.
  * `MiniMaxM3MSATrtllmAttentionMetadata` subclasses
    `TrtllmAttentionMetadata` and owns the CUDA-graph-stable buffers and
    the built sparse metadata.

The classes are defined inside `get_minimax_m3_msa_attention_backend_cls`
with a deferred `trtllm` import, avoiding a `trtllm` -> `fmha` ->
`interface` import cycle at package init.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .common import (
    _MSA_REQUIRED_HEAD_DIM,
    _MSA_REQUIRED_TOPK,
    build_kv_indices_and_lens,
    msa_paged_kv,
    write_main_kv_slots,
)
from .indexer import MsaIndexer
from .metadata import (
    MiniMaxM3SparseConfig,
    build_m3_sparse_metadata_and_plans,
    build_stable_kv_indices,
    get_global_msa_geometry,
    m3_cache_device,
    rederive_m3_attachment,
    whole_batch_qo_lens,
)

if TYPE_CHECKING:
    from .metadata import MiniMaxM3SparseAttentionMetadata


def _whole_batch_lens(
    m3_meta: "MiniMaxM3SparseAttentionMetadata",
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Whole-batch `(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices)`.

    Prefers the pre-staged plan values on the metadata (CUDA-graph-stable
    buffers) and falls back to an eager rebuild for focused tests and the
    first eager warmup pass.
    """
    kv_indices = getattr(m3_meta, "msa_kv_indices", None)
    if kv_indices is not None:
        return (
            m3_meta.msa_qo_lens_cpu,
            m3_meta.msa_kv_lens_cpu,
            m3_meta.msa_qo_offset_cpu,
            kv_indices,
        )
    lens = whole_batch_qo_lens(m3_meta)
    if lens is None:
        raise RuntimeError("prefill metadata requires extend_seq_lens_cpu / prefix_lens")
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = lens
    kv_indices, _ = build_kv_indices_and_lens(m3_meta, page_size)
    return qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices


def run_msa_sparse_decode(
    config: "MiniMaxM3SparseConfig",
    kv_cache_manager,
    layer_idx: int,
    m3_meta: "MiniMaxM3SparseAttentionMetadata",
    q: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """CUDA-graph-safe decode sparse GQA via the in-tree driver.

    Decode is CUDA-graph captured, so it must not go through the eager
    `fmha_sm100_plan` host driver, which uses unpinned H2D staging,
    per-call device allocations, and a device-side cost sweep. The staged
    page tables come from the metadata's pre-built plan values. Returns
    `[num_tokens, num_q_heads * head_dim]`.
    """
    from .decode_wrapper.dispatch import (
        M3DecodeGeometry,
        decode_sparse_attention,
        resolve_decode_state,
    )

    kv_indices = getattr(m3_meta, "msa_kv_indices", None)
    if kv_indices is None:
        raise RuntimeError(
            "MiniMax-M3 MSA decode requires pre-staged plan values; "
            "prepare() did not build them (missing geometry / use_msa)."
        )
    kv_page_indptr = m3_meta.msa_kv_page_indptr
    k_paged, v_paged = msa_paged_kv(kv_cache_manager, layer_idx)
    page_size = int(k_paged.shape[2])
    seq_lens = m3_meta.seq_lens.to(torch.int32)
    total_q = int(q.shape[0])
    geometry = M3DecodeGeometry.from_config(
        config,
        max_batch=int(getattr(m3_meta, "msa_max_batch", 0))
        or max(64, 1 << (total_q - 1).bit_length()),
        max_kv_len=int(getattr(m3_meta, "msa_max_kv_len", 0)) or int(m3_meta.req_to_token.shape[1]),
        page_size=page_size,
    )
    state = resolve_decode_state(m3_meta, geometry, q.device)
    out = decode_sparse_attention(
        state,
        q,
        k_paged,
        v_paged,
        kv_block_indexes,
        seq_lens=seq_lens,
        kv_page_indptr=kv_page_indptr,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        qo_len=int(m3_meta.decode_qo_len),
    )
    return out.reshape(total_q, config.num_q_heads * config.head_dim)


def rederive_msa_attachment(owner) -> None:
    """Re-derive the M3 attachment + MSA eager-plan mirrors after the
    overlap scheduler's on-device kv-len correction.

    Beyond :func:`rederive_m3_attachment` (which covers the decode
    side), prefill/mixed batches run the EAGER fmha plan, which consumes
    the CPU length mirrors AND the staged flat page table, whose layout
    (cumulative page counts) depends on the corrected lens — a shrink
    across a page boundary would misbase every subsequent row's pages.
    Refresh both; host work is legal here because mixed batches are
    never captured.
    """
    rederive_m3_attachment(owner)
    meta = owner.m3_sparse_metadata
    if meta is None or not meta.is_prefill or meta.msa_kv_lens_cpu is None:
        return
    batch = int(meta.slot_ids.shape[0])
    kv_lens_cpu = owner.kv_lens_cuda[:batch].to("cpu", torch.int32)
    meta.msa_kv_lens_cpu = kv_lens_cpu
    if meta.msa_qo_lens_cpu is not None:
        meta.msa_qo_offset_cpu = (kv_lens_cpu - meta.msa_qo_lens_cpu).to(torch.int32)
    geometry = get_global_msa_geometry()
    if geometry is not None and owner._msa_kv_indices_buf is not None:
        kv_indices, kv_page_indptr = build_stable_kv_indices(
            req_to_token=meta.req_to_token,
            slot_ids=meta.slot_ids,
            seq_lens=meta.seq_lens,
            seq_lens_cpu=kv_lens_cpu,
            page_size=int(geometry.block_size),
            dst=owner._msa_kv_indices_buf,
            kv_page_indptr_dst=owner._msa_kv_page_indptr_buf,
        )
        meta.msa_kv_indices = kv_indices
        meta.msa_kv_page_indptr = kv_page_indptr


@functools.lru_cache(maxsize=1)
def get_minimax_m3_msa_attention_backend_cls():
    """Return `MiniMaxM3MSATrtllmAttention` (selection entry point)."""
    from dataclasses import dataclass

    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

    @dataclass(init=False)
    class MiniMaxM3MSATrtllmAttentionMetadata(TrtllmAttentionMetadata):
        """`TrtllmAttentionMetadata` for MiniMax-M3 MSA sparse layers.

        Owns the CUDA-graph-stable buffers and the per-forward sparse
        metadata; the paged K/V views and sparse GQA dispatch live in
        `MsaSparseGqaFmha`.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.m3_sparse_metadata = None
            self.m3_out_cache_loc = None
            self._m3_static_buffers: Optional[dict] = None
            self._msa_kv_indices_buf = None
            self._msa_kv_page_indptr_buf = None
            # Persistent decode state holding the CUDA-graph-stable kernel
            # buffers. This metadata persists across steps and graph replays
            # for its batch-size bucket, so owning the state here keeps
            # those buffers' data_ptr() stable across replays.
            self._m3_decode_state = None

        # lifecycle

        def prepare(self) -> None:
            super().prepare()
            self.m3_sparse_metadata = None
            self.m3_out_cache_loc = None
            # The per-rank config (geometry) is registered process-wide by
            # the attention layer's constructor, before any CUDA graph
            # capture, so every metadata instance reads it here. The builder
            # allocates and owns all graph-stable buffers on this metadata
            # and publishes m3_sparse_metadata / m3_out_cache_loc.
            geometry = get_global_msa_geometry()
            m3_meta = build_m3_sparse_metadata_and_plans(self, geometry=geometry)
            self._attach_decode_state(m3_meta, geometry, m3_cache_device(self))

        def on_update_kv_lens(self) -> None:
            """Same pattern as ``DSAtrtllmAttentionMetadata`` and the
            Triton-path M3 metadata; see ``rederive_msa_attachment``.
            """
            super().on_update_kv_lens()
            rederive_msa_attachment(self)

        def _attach_decode_state(self, m3_meta, config, cache_device) -> None:
            """Build once and attach the persistent decode state.

            Runs inside prepare(), outside any CUDA graph capture, so the
            decode buffers are allocated before capture and reused across
            replays. It is keyed by the decode geometry and rebuilt only
            when that changes, which does not happen once a bucket's static
            buffers are allocated. Only the pure-decode path uses it;
            prefill / mixed batches run the eager fmha_sm100 path.
            """
            if m3_meta is None or config is None:
                return
            if getattr(m3_meta, "msa_kv_indices", None) is None or m3_meta.is_prefill:
                return
            from .decode_wrapper.dispatch import M3DecodeGeometry, build_m3_decode_state

            decode_geometry = M3DecodeGeometry.from_config(
                config,
                max_batch=int(m3_meta.msa_max_batch),
                max_kv_len=int(m3_meta.msa_max_kv_len),
            )
            state = self._m3_decode_state
            if state is None or state.geom != decode_geometry or state.device != cache_device:
                state = build_m3_decode_state(decode_geometry, cache_device)
                self._m3_decode_state = state
            m3_meta.decode_state = state

        # internal accessors

        @property
        def m3_meta(self) -> "MiniMaxM3SparseAttentionMetadata":
            if self.m3_sparse_metadata is None:
                raise RuntimeError(
                    "MiniMaxM3MSATrtllmAttentionMetadata.m3_sparse_metadata is not built; "
                    "prepare() must run before the sparse forward."
                )
            return self.m3_sparse_metadata

        # Index-K cache access + write, consumed by the MsaIndexer.

        def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
            """Raw paged index-K view for the indexer; HND conversion is done there."""
            return self.kv_cache_manager.get_index_k_buffer(layer_idx)

        def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
            idx_cache = self.msa_idx_k_cache(layer_idx)
            sparse_index_dim = int(idx_cache.shape[-1])
            num_tokens = int(idx_k.shape[0])
            write_main_kv_slots(
                idx_cache, self.m3_out_cache_loc, idx_k.reshape(num_tokens, 1, sparse_index_dim)
            )

    class MiniMaxM3MSATrtllmAttention(TrtllmAttention):
        """MSA-backed MiniMax-M3 sparse attention (mimics `DSATrtllmAttention`)."""

        Metadata = MiniMaxM3MSATrtllmAttentionMetadata

        def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config=None,
            *,
            sparse_params,
            **kwargs,
        ):
            TrtllmAttention.__init__(
                self,
                layer_idx,
                num_heads,
                head_dim,
                num_kv_heads=num_kv_heads,
                quant_config=quant_config,
                sparse_params=sparse_params,
                **kwargs,
            )
            self.m3_config = MiniMaxM3SparseConfig.from_sparse_params(
                sparse_params,
                num_q_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
                head_dim=head_dim,
            )
            self.disable_index_value = bool(sparse_params.disable_index_value)
            self._validate_msa_preconditions()
            self.indexer = MsaIndexer(self.m3_config)
            self._register_global_geometry()

        def _validate_msa_preconditions(self) -> None:
            if not self.disable_index_value:
                raise NotImplementedError(
                    "MSA backend requires disable_index_value=True (MSA's proxy pass "
                    "only consumes max_score; an index-V path is not implemented)."
                )
            if self.m3_config.head_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires head_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.head_dim}."
                )
            if self.m3_config.sparse_index_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires sparse_index_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.sparse_index_dim}."
                )
            if self.m3_config.topk != _MSA_REQUIRED_TOPK:
                raise NotImplementedError(
                    f"MSA backend requires topk={_MSA_REQUIRED_TOPK}, got {self.m3_config.topk}."
                )

        def _register_global_geometry(self) -> None:
            # Register the config process-wide at construction, before any
            # forward or CUDA graph capture, so every metadata's prepare()
            # can pre-build the MSA plans.
            from .metadata import set_global_msa_geometry

            set_global_msa_geometry(self.m3_config)

        @classmethod
        def support_fused_rope(cls) -> bool:
            # The MiniMax-M3 model layer applies partial RoPE to both the
            # main and index branches explicitly.
            return False

        def create_fmha_libs(self) -> None:
            # Restrict the dispatch loop to MsaSparseGqaFmha; the standard
            # libs come first in the registry but cannot handle the M3
            # sparse contract (unfused q/k/v plus selected block indices).
            from tensorrt_llm._torch.attention_backend.fmha import MsaSparseGqaFmha

            self.fmha_libs = [MsaSparseGqaFmha(self)]

        # indexer entry point (called by the model layer, DSA-style)

        def run_indexer(
            self,
            idx_q: torch.Tensor,
            idx_k: torch.Tensor,
            metadata,
            *,
            idx_sm_scale: Optional[float] = None,
        ) -> torch.Tensor:
            """Write the index-K cache and return the selected block indices.

            Mirrors DSA's `indexer.sparse_attn_indexer`: the model layer
            runs this before `forward` and threads the result through
            `forward_args.topk_indices`. Returns `[total_q, num_kv_heads,
            topk]`.
            """
            config = self.m3_config
            idx_sm_scale = (
                idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5
            )
            num_tokens = int(idx_q.shape[0])
            idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
            idx_k_view = idx_k.view(num_tokens, 1, config.sparse_index_dim)

            metadata.msa_write_idx_k(self.layer_idx, idx_k_view)
            idx_k_cache = metadata.msa_idx_k_cache(self.layer_idx)
            m3_meta = metadata.m3_meta
            # The cache's tokens_per_block equals the sparse block size
            # (enforced by the M3 KV cache manager), so read it from the
            # config instead of the cache view.
            page_size = self.m3_config.block_size

            if m3_meta.is_prefill:
                qo, kv, qo_off, kv_indices = _whole_batch_lens(m3_meta, page_size)
                return self.indexer.select_blocks_prefill(
                    idx_q_view,
                    idx_k_cache,
                    m3_meta,
                    idx_sm_scale=idx_sm_scale,
                    qo_lens_cpu=qo,
                    kv_lens_cpu=kv,
                    qo_offset_cpu=qo_off,
                    kv_indices=kv_indices,
                )
            return self.indexer.select_blocks_decode(
                idx_q_view,
                idx_k_cache,
                m3_meta,
                idx_sm_scale=idx_sm_scale,
                page_size=page_size,
            )

        # sparse hooks (consumed by inherited TrtllmAttention.forward)

        def sparse_attn_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            # The model layer runs run_indexer() and passes the selected
            # block indices through forward_args.topk_indices. Publish them
            # as the sparse attention indices the MsaSparseGqaFmha reads.
            return getattr(forward_args, "topk_indices", None), None

        def sparse_kv_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            return None, None

    return MiniMaxM3MSATrtllmAttention


__all__ = [
    "get_minimax_m3_msa_attention_backend_cls",
]
