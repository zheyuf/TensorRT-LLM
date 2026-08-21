# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-shape Triton/MSA tactic selection for MiniMax-M3 sparse decode."""

from __future__ import annotations

import contextlib
from typing import Any, List

import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    DistributedTuningStrategy,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
    autotune,
)
from tensorrt_llm.logger import logger

_CUSTOM_OP = "trtllm::minimax_m3_sparse_decode"
_ATTEMPTED_TUNING_KEYS = set()


class MiniMaxM3SparseDecodeRunner(TunableRunner):
    """Run either complete sparse-decode kernel with identical inputs.

    The fallback tactic is Triton, which is the production route predating the
    MSA sparse-decode port.  MSA is considered only when the caller supplies a
    prebuilt plan; metadata preparation guarantees that for ``adaptive``.
    """

    # All TP ranks capture the same graph key and must embed the same tactic.
    # Profiling independently makes near-tie shapes vulnerable to rank-local
    # timer noise. Rank 0 is representative on homogeneous TP nodes, so tune
    # once and broadcast the cached choice before capture.
    tuning_config = TuningConfig(
        use_cuda_graph=True,
        distributed_tuning_strategy=DistributedTuningStrategy.BROADCAST,
    )

    def __init__(
        self,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        topk: int,
        decode_query_len: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        sm_scale: float,
    ) -> None:
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.topk = int(topk)
        self.decode_query_len = int(decode_query_len)
        self.q_dtype = str(q_dtype)
        self.kv_dtype = str(kv_dtype)
        self.output_dtype = str(output_dtype)
        self.sm_scale = float(sm_scale)

    def unique_id(self) -> tuple:
        # AutoTuner's shape key does not include dtype or scalar kernel
        # geometry, so make all of them explicit here.  Batch and total query
        # tokens are already represented by the input tensor shapes.
        return (
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.topk,
            self.decode_query_len,
            self.q_dtype,
            self.kv_dtype,
            self.output_dtype,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[Any]:
        del inputs, profile, kwargs
        return ["triton", "msa"]

    def forward(
        self,
        inputs: List[torch.Tensor],
        *,
        tactic: Any = -1,
        plan: tuple,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        q, k_paged, v_paged, block_indexes, block_table, seq_lens, output = inputs

        if tactic == -1:
            tactic = "triton"
        if tactic == "triton":
            from .triton_sparse_decode import minimax_m3_sparse_attn_decode

            minimax_m3_sparse_attn_decode(
                q,
                k_paged,
                v_paged,
                block_indexes.permute(1, 0, 2),
                block_table,
                seq_lens,
                sm_scale=self.sm_scale,
                output=output,
                decode_query_len=self.decode_query_len,
            )
            return output
        if tactic == "msa":
            from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import (
                run_msa_sparse_gqa,
            )

            use_fp8 = k_paged.dtype == torch.float8_e4m3fn
            msa_q = q
            if use_fp8 and msa_q.dtype != torch.float8_e4m3fn:
                msa_q = msa_q.to(torch.float8_e4m3fn)
            run_msa_sparse_gqa(
                msa_q,
                k_paged,
                v_paged,
                block_indexes,
                kv_indices=block_table.flatten(),
                sm_scale=self.sm_scale,
                causal=True,
                head_dim=self.head_dim,
                plan=plan,
                out=output,
                use_fp8=use_fp8,
            )
            return output
        raise ValueError(f"Unsupported MiniMax-M3 sparse decode tactic: {tactic!r}.")


def _tuning_key(runner: MiniMaxM3SparseDecodeRunner, inputs: List[torch.Tensor]) -> tuple:
    return runner.unique_id(), tuple(tuple(int(dim) for dim in tensor.shape) for tensor in inputs)


def run_adaptive_sparse_decode(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    block_indexes: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    sm_scale: float,
    decode_query_len: int,
    plan: tuple,
) -> None:
    """Profile once per exact shape, cache the winner, and run it.

    The first non-capturing CUDA call for a shape is the tuning point.  CUDA
    graph warmup provides exactly such a call before capture.  During capture,
    a cache miss deliberately takes AutoTuner's Triton fallback instead of
    attempting nested profiling; the captured graph therefore never changes
    tactic across replays.
    """
    if plan is None:
        raise RuntimeError(
            "MiniMax-M3 adaptive sparse decode requires a preplanned MSA GQA plan."
        )

    runner = MiniMaxM3SparseDecodeRunner(
        num_q_heads=int(q.shape[1]),
        num_kv_heads=int(k_paged.shape[1]),
        head_dim=int(q.shape[2]),
        page_size=int(k_paged.shape[2]),
        topk=int(block_indexes.shape[2]),
        decode_query_len=decode_query_len,
        q_dtype=q.dtype,
        kv_dtype=k_paged.dtype,
        output_dtype=output.dtype,
        sm_scale=sm_scale,
    )
    inputs = [q, k_paged, v_paged, block_indexes, block_table, seq_lens, output]
    tuner = AutoTuner.get()
    input_shapes = tuple(tensor.size() for tensor in inputs)
    cache_hit, _, _, _ = tuner.profiling_cache.search_cache(
        _CUSTOM_OP,
        [runner],
        input_shapes,
        runner.tuning_config,
    )
    tuning_key = _tuning_key(runner, inputs)
    # AutoTuner's tune context performs pipeline cache handoff.  Graph warmup
    # does not run that coordinator, so PP configurations conservatively keep
    # the Triton fallback until graph-tuning orchestration supports them.
    can_profile = (
        q.is_cuda
        and not torch.cuda.is_current_stream_capturing()
        and not tuner.mapping.has_pp()
    )
    should_profile = (
        not cache_hit
        and tuning_key not in _ATTEMPTED_TUNING_KEYS
        and can_profile
        and not tuner.is_tuning_mode
    )
    if should_profile:
        _ATTEMPTED_TUNING_KEYS.add(tuning_key)

    tune_context = autotune() if should_profile else contextlib.nullcontext()
    with tune_context:
        _, tactic = tuner.choose_one(
            _CUSTOM_OP,
            [runner],
            runner.tuning_config,
            inputs,
            plan=plan,
        )
    selected_tactic = "triton" if tactic == -1 else tactic
    logger.info_once(
        "MiniMax-M3 adaptive sparse decode selected "
        f"{selected_tactic} for B={int(block_table.shape[0])}, "
        f"DQL={decode_query_len}, local HQ/HKV={int(q.shape[1])}/{int(k_paged.shape[1])}.",
        key=(_CUSTOM_OP, tuning_key),
    )
    runner(inputs, tactic=tactic, plan=plan)
