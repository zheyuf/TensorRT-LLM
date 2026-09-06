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
"""Pure-logic tests for MiniMax-M3's shared one-model draft layers.

The drafter's layers share the target KV cache manager; the manager presents
each of them to the attention op as its own virtual pool rooted at the
layer's K page inside M3's non-uniform mega-slot.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
    cache_manager as m3_cache_manager,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
    derive_shared_draft_layout,
    extend_attention_op_pools_for_shared_draft_layers,
    shared_draft_layer_count,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

DRAFT_LOCAL_LAYER = 60
SCALE = 179  # sub-pages per M3 mega-slot: 3 dense x 2 + 57 sparse x 3 + draft x 2
DRAFT_K_ADDR = 0x7000_0000


def test_virtual_pool_is_rooted_at_the_draft_k_page():
    pool_pointers = torch.tensor([[0x6000_0000, 0]], dtype=torch.int64)
    pool_mapping = torch.tensor([[0, i] for i in range(61)], dtype=torch.int32)

    pointers, mapping, index_scales, kv_offsets, op_pools = (
        extend_attention_op_pools_for_shared_draft_layers(
            pool_pointers, pool_mapping, 1, [(DRAFT_LOCAL_LAYER, DRAFT_K_ADDR, SCALE)]
        )
    )

    assert pointers.tolist() == [[0x6000_0000, 0], [DRAFT_K_ADDR, 0]]
    assert pointers.dtype == torch.int64
    # The draft layer moves to the new pool at offset 0; target rows are untouched.
    assert mapping[DRAFT_LOCAL_LAYER].tolist() == [1, 0]
    assert mapping[:DRAFT_LOCAL_LAYER].tolist() == pool_mapping[:DRAFT_LOCAL_LAYER].tolist()
    # Slot s -> page s * SCALE for K and s * SCALE + 1 for V.
    assert index_scales.tolist() == [SCALE]
    assert kv_offsets.tolist() == [1]
    assert op_pools == [(1, 0)]


def test_virtual_pool_keeps_nvfp4_pointer_pairs():
    pool_pointers = torch.tensor([[[0x6000_0000, 0x6100_0000], [0, 0]]], dtype=torch.int64)
    pool_mapping = torch.tensor([[0, 0]], dtype=torch.int32)

    pointers, _, _, _, _ = extend_attention_op_pools_for_shared_draft_layers(
        pool_pointers, pool_mapping, 1, [(0, DRAFT_K_ADDR, SCALE)]
    )

    assert pointers.shape == (2, 2, 2)
    assert pointers[1].tolist() == [[DRAFT_K_ADDR, 0], [0, 0]]


def test_no_draft_layers_leaves_the_tables_alone():
    pool_pointers = torch.tensor([[0x6000_0000, 0]], dtype=torch.int64)
    pool_mapping = torch.tensor([[0, 3]], dtype=torch.int32)

    pointers, mapping, index_scales, kv_offsets, op_pools = (
        extend_attention_op_pools_for_shared_draft_layers(pool_pointers, pool_mapping, 1, [])
    )

    assert pointers.tolist() == pool_pointers.tolist()
    assert mapping.tolist() == pool_mapping.tolist()
    assert index_scales.numel() == 0 and kv_offsets.numel() == 0
    assert op_pools == []


def test_block_offset_copy_feeds_the_virtual_pool_from_the_source_pool(monkeypatch):
    """The base copy fills the storage pools; the override runs the same device
    copy once more over the source pool's slot ids with the draft scale."""
    calls = []

    def fake_base_copy(
        self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=None
    ):
        calls.append(("base", request_ids, num_seqs, max_blocks))

    def fake_device_copy(host, dst, copy_idx, index_scales, kv_offsets, stream):
        calls.append(
            ("virtual", host, dst, copy_idx, index_scales.tolist(), kv_offsets.tolist(), stream)
        )

    monkeypatch.setattr(KVCacheManagerV2, "copy_batch_block_offsets", fake_base_copy)
    monkeypatch.setattr(m3_cache_manager, "copy_batch_block_offsets_to_device", fake_device_copy)

    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager._draft_op_pools = ((1, 0),)
    manager._draft_index_scales = torch.tensor([SCALE], dtype=torch.int32)
    manager._draft_kv_offsets = torch.tensor([1], dtype=torch.int32)
    manager.host_kv_cache_block_offsets = torch.zeros((1, 4, 2, 8), dtype=torch.int32)
    copy_idx = torch.tensor([2, 0], dtype=torch.int32)
    manager.index_mapper = SimpleNamespace(get_copy_index=lambda ids, nc, bw: copy_idx)
    manager._stream = SimpleNamespace(cuda_stream=1234)
    dst = torch.zeros((2, 4, 2, 8), dtype=torch.int32)

    manager.copy_batch_block_offsets(dst, [7, 9], 1, 0, 2, max_blocks=5)

    assert calls[0] == ("base", [7, 9], 2, 5)
    kind, host, dst_slice, idx, scales, offsets, stream = calls[1]
    assert kind == "virtual"
    assert (
        host.shape == (1, 4, 2, 8)
        and host.data_ptr() == manager.host_kv_cache_block_offsets.data_ptr()
    )
    assert dst_slice.shape == (1, 4, 2, 8) and dst_slice.data_ptr() == dst[1].data_ptr()
    assert idx is copy_idx
    assert scales == [SCALE] and offsets == [1] and stream == 1234


def test_block_offset_copy_is_untouched_without_draft_layers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        KVCacheManagerV2,
        "copy_batch_block_offsets",
        lambda self, *a, **k: calls.append(("base", a, k)),
    )
    monkeypatch.setattr(
        m3_cache_manager,
        "copy_batch_block_offsets_to_device",
        lambda *a: pytest.fail("no virtual pool to fill"),
    )
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.copy_batch_block_offsets(torch.zeros(1), [1], 1, 0, 1)
    assert len(calls) == 1


def test_draft_layout_scalar_heads_extend_the_target_count():
    # The creation site passes the pretrained target count; the base manager
    # appends the shared draft layers after it.
    assert derive_shared_draft_layout(60, 4, 1) == ([60], 60)


def test_draft_layout_head_list_is_the_total():
    heads = [4] * 60 + [64]
    assert derive_shared_draft_layout(60, heads, 1) == ([60], 60)


def test_draft_layout_equal_head_list():
    assert derive_shared_draft_layout(60, [4] * 61, 1) == ([60], 60)


def test_draft_layout_pre_extended_num_layers():
    heads = [4] * 60 + [64]
    assert derive_shared_draft_layout(61, heads, 1) == ([60], 60)


def test_draft_layout_no_draft():
    assert derive_shared_draft_layout(60, [4] * 60, 0) == ([], 60)
    assert derive_shared_draft_layout(60, 4, 0) == ([], 60)


def test_draft_layout_unpinned_range():
    assert derive_shared_draft_layout(None, 4, 1) == ([], None)


def _eagle3_one_model_config(num_draft_hidden_layers=None):
    mode = SimpleNamespace(
        is_mtp_eagle_one_model=lambda: False,
        is_mtp_vanilla=lambda: False,
        is_eagle3_one_model=lambda: True,
    )
    return SimpleNamespace(spec_dec_mode=mode, _num_draft_hidden_layers=num_draft_hidden_layers)


def test_shared_draft_layer_count_follows_the_base_manager():
    # Appended only when speculation is on and the layer set is not pinned by
    # a mask, exactly when get_pp_layers appends the speculative layers.
    assert shared_draft_layer_count(_eagle3_one_model_config(), None) == 1
    assert shared_draft_layer_count(_eagle3_one_model_config(2), None) == 2
    assert shared_draft_layer_count(_eagle3_one_model_config(), [True] * 60) == 0
    assert shared_draft_layer_count(None, None) == 0
