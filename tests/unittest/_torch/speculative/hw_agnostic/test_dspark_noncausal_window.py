# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-causal sliding-window DSpark block attention on the TRTLLM backend.

The drafter serves it as the trtllm-gen causal windowed decode, with its log-sum-exp, plus
``dflash_merge_later_block_keys_``; it must equal dense non-causal attention.
"""

import math
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dflash import (
    DFlashForCausalLM,
    dflash_merge_later_block_keys_,
)

HEAD_DIM, PAGE_SIZE = 128, 64


def _trtllm_gen_ops():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from tensorrt_llm._torch.speculative.dflash_attention import (
        dflash_trtllm_gen_unavailability_reason,
        get_dflash_trtllm_gen_ops,
    )

    reason = dflash_trtllm_gen_unavailability_reason()
    if reason is not None:
        pytest.skip(f"DFlash TRTLLM backend unavailable: {reason}")
    return get_dflash_trtllm_gen_ops()


def _check_block_attention(ctx_lens, block_size, num_heads, num_kv_heads, window):
    """Runs the drafter's lowering over a shuffled page pool and compares it to dense fp32 attention."""
    ops = _trtllm_gen_ops()
    dev = torch.device("cuda")
    B = len(ctx_lens)
    seq_lens = [c + block_size for c in ctx_lens]
    num_pages = [math.ceil(s / PAGE_SIZE) for s in seq_lens]
    max_pages = max(num_pages)
    pages = torch.randperm(sum(num_pages), dtype=torch.int32).split(num_pages)
    block_tables = torch.zeros(B, max_pages, dtype=torch.int32, device=dev)
    cache = torch.zeros(
        sum(num_pages), 2, num_kv_heads, PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    keys = [torch.randn(s, num_kv_heads, HEAD_DIM, device=dev).bfloat16() for s in seq_lens]
    values = [torch.randn(s, num_kv_heads, HEAD_DIM, device=dev).bfloat16() for s in seq_lens]
    for b in range(B):
        block_tables[b, : num_pages[b]] = pages[b]
        for pos in range(0, ctx_lens[b], PAGE_SIZE):
            n = min(PAGE_SIZE, ctx_lens[b] - pos)
            page = int(pages[b][pos // PAGE_SIZE])
            cache[page, 0, :, :n] = keys[b][pos : pos + n].transpose(0, 1)
            cache[page, 1, :, :n] = values[b][pos : pos + n].transpose(0, 1)
    query = torch.randn(B, block_size, num_heads, HEAD_DIM, device=dev).bfloat16()
    block_key = torch.stack([k[-block_size:] for k in keys])
    block_value = torch.stack([v[-block_size:] for v in values])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    ops.append_paged_kv_cache(
        append_key=block_key.flatten(0, 1),
        append_value=block_value.flatten(0, 1),
        batch_indices=torch.arange(B, dtype=torch.int32, device=dev).repeat_interleave(block_size),
        positions=(seq_lens_t.view(-1, 1) - block_size + torch.arange(block_size, device=dev))
        .flatten()
        .int(),
        paged_kv_cache=cache,
        kv_indices=block_tables.flatten(),
        kv_indptr=torch.arange(0, (B + 1) * max_pages, max_pages, dtype=torch.int32, device=dev),
        kv_last_page_len=(seq_lens_t - 1) % PAGE_SIZE + 1,
        kv_layout="HND",
    )
    drafter = DFlashForCausalLM.__new__(DFlashForCausalLM)
    torch.nn.Module.__init__(drafter)
    drafter._dflash_trtllm_gen_ops = ops
    drafter._dflash_trtllm_gen_device = drafter._dflash_trtllm_gen_sm_count = None
    drafter._dflash_trtllm_gen_workspace = drafter._dflash_trtllm_gen_counters = None
    with patch.object(DFlashForCausalLM, "_prepare_dflash_index_buffers"):
        drafter._prepare_dflash_trtllm_gen_buffers(
            torch.bfloat16, dev, B, block_size, num_heads, num_kv_heads, HEAD_DIM
        )

    w = window - 1
    out = torch.empty(B * block_size, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    _, lse = ops.batch_decode_with_kv_cache(
        query=query.flatten(0, 1),
        kv_cache=(cache[:, 0], cache[:, 1]),
        workspace_buffer=drafter._dflash_trtllm_gen_workspace,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_pages * PAGE_SIZE,
        bmm1_scale=HEAD_DIM**-0.5,
        bmm2_scale=1.0,
        window_left=w,
        out=out,
        sinks=None,
        enable_pdl=False,
        kv_layout="HND",
        backend="trtllm-gen",
        q_len_per_req=block_size,
        max_q_len=None,
        cum_seq_lens_q=None,
        kv_cache_sf=None,
        uses_shared_paged_kv_idx=True,
        bmm1_scale_log2=None,
        return_lse=True,
        multi_ctas_kv_counter_buffer=drafter._dflash_trtllm_gen_counters,
    )
    dflash_merge_later_block_keys_(out, lse, query, block_key, block_value, HEAD_DIM**-0.5, w)

    group = num_heads // num_kv_heads
    for b, c in enumerate(ctx_lens):
        pos_q = c + torch.arange(block_size, device=dev).view(-1, 1)
        pos_k = torch.arange(c + block_size, device=dev).view(1, -1)
        logits = torch.einsum(
            "jhd,lhd->hjl", query[b].float(), keys[b].float().repeat_interleave(group, 1)
        )
        logits = logits.masked_fill((pos_k - pos_q).abs() > w, float("-inf")) * HEAD_DIM**-0.5
        expected = torch.einsum(
            "hjl,lhd->jhd", logits.softmax(-1), values[b].float().repeat_interleave(group, 1)
        )
        got = out[b * block_size : (b + 1) * block_size].float()
        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("block_size", [3, 7])
@pytest.mark.parametrize("heads", [(16, 1), (64, 4)])
def test_non_causal_window_matches_dense_attention(block_size, heads):
    torch.manual_seed(block_size)
    window = 64
    ctx_lens = [
        1,
        window - block_size - 1,
        window - block_size,
        window - 1,
        window,
        window + 1,
        3 * window + 17,
    ]
    _check_block_attention(ctx_lens, block_size, *heads, window)


def test_workspace_holds_the_lse_stats():
    """At 64 heads, 512 requests and 3 block tokens, the decode's softmax stats alone outgrow the
    plain trtllm-gen workspace, which flashinfer would refuse."""
    torch.manual_seed(0)
    _check_block_attention([1 + (97 * b) % 300 for b in range(512)], 3, 64, 4, 4096)
