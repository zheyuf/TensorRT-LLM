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
"""Paired block reuse of a full-attention target pool and a DSpark-like sliding-window draft pool.

Real KVCacheManagerV2 pools admitted through KVCacheV2Scheduler: which depth both pools claim,
and that the draft pool never publishes a page the drafter did not write.
"""

import contextlib
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, CapacitySchedulerPolicy, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DEFAULT_BEAM_INDEX

TOKENS_PER_BLOCK = 8
WINDOW = 16
MAX_SEQ_LEN = 256


def _manager(policy: str, **kwargs) -> KVCacheManagerV2:
    window = kwargs.pop("window", None)
    return KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=True,
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            max_util_for_resume=1.0,
            max_attention_window=None if window is None else [window],
            block_reuse_config=BlockReuseConfig(policy=policy),
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        max_num_tokens=MAX_SEQ_LEN,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=4096,
        joint_kv_cache_reuse=True,
        **kwargs,
    )


@pytest.fixture(params=["per_conversation", "all_reusable"])
def pools(request):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    target = _manager(request.param)
    draft = _manager(request.param, window=WINDOW, is_draft=True, hidden_state_draft_pool=True)
    scheduler = KVCacheV2Scheduler(
        max_batch_size=4,
        max_num_tokens=MAX_SEQ_LEN,
        kv_cache_manager=target,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        draft_kv_cache_manager=draft,
    )
    try:
        yield scheduler, target, draft
    finally:
        draft.shutdown()
        target.shutdown()


@dataclass
class _Request:
    py_request_id: int
    tokens: list[int]
    context_current_position: int = 0
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    context_chunk_size: int = 0
    lora_task_id: int | None = None
    cache_salt: str | None = None
    py_conversation_params: None = None
    py_draft_tokens: None = None
    py_served_prefix_end: int = 0
    py_num_connector_matched_tokens: int = 0
    is_disagg_generation_init_state: bool = False
    is_dummy_request: bool = False
    return_perf_metrics: bool = False
    multimodal_hashes: None = None
    multimodal_positions: None = None
    multimodal_lengths: None = None

    @property
    def prompt_len(self) -> int:
        return len(self.tokens)

    @property
    def context_remaining_length(self) -> int:
        return self.prompt_len - self.context_current_position

    @property
    def is_dummy(self) -> bool:
        return self.is_dummy_request

    def get_tokens(self, beam_id: int = DEFAULT_BEAM_INDEX) -> list[int]:
        return self.tokens

    def get_tokens_view(self, beam_id: int = DEFAULT_BEAM_INDEX) -> np.ndarray:
        return np.asarray(self.tokens, dtype=np.int32)

    def set_prepopulated_prompt_len(self, length: int, tokens_per_block: int) -> None:
        self.context_current_position = length


def _prefill(scheduler, target, draft, request: _Request, chunks=(MAX_SEQ_LEN,)) -> int:
    """Admit and prefill ``request`` chunk by chunk; return its reuse claim."""
    claim = None
    for index, chunk in enumerate(chunks):
        request.is_first_context_chunk = index == 0
        assert scheduler._prepare_context_pair(request)
        if claim is None:
            claim = request.context_current_position
        chunk = min(chunk, request.context_remaining_length)
        request.is_last_context_chunk = chunk == request.context_remaining_length
        request.context_chunk_size = chunk
        assert scheduler._try_allocate_context(request, chunk)
        batch = ScheduledRequests()
        batch.append_context_request(request)
        draft.prepare_resources(batch)
        request.context_current_position += chunk
        target.update_context_resources(batch)
        draft.update_context_resources(batch)
        if request.context_remaining_length == 0:
            break
    assert request.context_remaining_length == 0
    return claim


def _finish(target, draft, *requests: _Request) -> None:
    for request in requests:
        for manager in (target, draft):
            if request.py_request_id in manager.kv_cache_map:
                manager.free_resources(request)


def _probe(manager: KVCacheManagerV2, tokens: list[int]) -> int:
    return manager.probe_context_reuse(_Request(10_000, tokens))


def _prompt(start: int, length: int) -> list[int]:
    return list(range(start, start + length))


def test_continuation_reuses_both_pools(pools):
    scheduler, target, draft = pools
    first = _Request(1, _prompt(0, 64))
    turn = _Request(2, first.tokens + _prompt(1000, 20))
    try:
        assert _prefill(scheduler, target, draft, first) == 0
        _finish(target, draft, first)
        assert _prefill(scheduler, target, draft, turn) == 64
    finally:
        _finish(target, draft, first, turn)


def test_a_branch_point_window_is_recomputed_once_and_then_shared(pools):
    """The draft pool holds no window at the shared 43-token prefix, so the first branch is a
    fresh mirror that recomputes from 24 (one window before page 40) and publishes the window
    at 40; the next branch claims the prefix in both pools."""
    scheduler, target, draft = pools
    first = _Request(1, _prompt(0, 64))
    second = _Request(2, first.tokens[:43] + _prompt(1000, 9))
    third = _Request(3, first.tokens[:43] + _prompt(2000, 9))
    try:
        _prefill(scheduler, target, draft, first)
        _finish(target, draft, first)

        assert _prefill(scheduler, target, draft, second) == 24
        _finish(target, draft, second)
        assert _prefill(scheduler, target, draft, third) >= 40
        # Nothing below the second request's claim was published.
        assert _probe(draft, second.tokens[:31] + _prompt(3000, 20)) == 0
    finally:
        _finish(target, draft, first, second, third)


@pytest.mark.parametrize("pools", ["all_reusable"], indirect=True)
@pytest.mark.parametrize("gated", [True, False], ids=["gated", "ungated_control"])
def test_commit_waits_until_the_window_leaves_the_unwritten_prefix(pools, gated):
    """A fresh mirror claiming 40 commits its first chunk at 48, while the window still covers
    the never-written page 32..39; committed then, a request ending at 47 would read it."""
    scheduler, target, draft = pools
    first = _Request(1, _prompt(0, 64))
    branch = _Request(2, first.tokens[:40] + _prompt(1000, 40))
    probe = branch.tokens[:47] + _prompt(2000, 20)
    try:
        _prefill(scheduler, target, draft, first)
        _finish(target, draft, first)

        gate = (
            contextlib.nullcontext()
            if gated
            else patch.object(
                KVCacheManagerV2, "_commit_publishes_written_pages_only", return_value=True
            )
        )
        with gate:
            assert _prefill(scheduler, target, draft, branch, chunks=(8, 32)) == 40
        _finish(target, draft, branch)

        assert _probe(draft, probe) == (0 if gated else 47)
    finally:
        _finish(target, draft, first, branch)


def test_an_attention_dp_pad_dummy_is_admitted_only_with_its_draft_cache(pools):
    scheduler, target, draft = pools
    for draft_manager, admitted in ((None, False), (draft, True)):
        (dummy,) = target.add_dummy_requests(
            [ATTENTION_DP_DUMMY_REQUEST_ID],
            is_gen=True,
            max_num_draft_tokens=7,
            draft_kv_cache_manager=draft_manager,
        )
        try:
            assert scheduler._try_allocate_generation(dummy) is admitted
        finally:
            _finish(target, draft, dummy)
