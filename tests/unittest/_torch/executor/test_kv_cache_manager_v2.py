from types import SimpleNamespace

import torch

import tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 as kv_cache_manager_v2_module
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int):
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens = None
        self.stopped_committing = False

    def commit(self, tokens):
        self.committed_tokens = tokens
        self.num_committed_tokens += len(tokens)

    def stop_committing(self):
        self.stopped_committing = True


def test_try_commit_blocks_commits_uncommitted_tokens_and_stops_at_context_end():
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        context_current_position=8,
        context_remaining_length=0,
        get_tokens=lambda beam_id: list(range(10)),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert kv_cache.committed_tokens == [4, 5, 6, 7]
    assert kv_cache.num_committed_tokens == 8
    assert kv_cache.stopped_committing


class _PoolMappingOverride(KVCacheManagerV2):
    def _build_pool_mapping_tensors(self):
        self.pool_mapping_override_called = True
        return self.expected_pool_pointers, self.expected_pool_mapping


class _FakePoolImpl:
    layer_grouping = [[0]]

    @staticmethod
    def get_page_index_scale(_layer_id, _role):
        return 1

    @staticmethod
    def get_mem_pool_base_address(_layer_id, role, _page_index_mode):
        return 16 if role == Role.VALUE else 0

    @staticmethod
    def get_page_stride(_layer_id, _role):
        return 16


def test_prepare_page_table_uses_subclass_pool_mapping(monkeypatch):
    """Keep sparse-manager pool mapping overrides on the initialization path."""
    monkeypatch.setattr(kv_cache_manager_v2_module, "prefer_pinned", lambda: False)

    manager = object.__new__(_PoolMappingOverride)
    manager.expected_pool_pointers = torch.tensor([[11, 0]], dtype=torch.int64)
    manager.expected_pool_mapping = torch.tensor([[0, 7]], dtype=torch.int32)
    manager.pool_mapping_override_called = False
    manager.num_pools = 1
    manager.max_beam_width = 1
    manager.max_blocks_per_seq = 4
    manager.kv_cache_type = CacheTypeCpp.SELF
    manager.enable_swa_scratch_reuse = False
    manager.impl = _FakePoolImpl()

    manager._prepare_page_table_tensor(index_mapper_capacity=2)

    assert manager.pool_mapping_override_called
    assert manager.kv_cache_pool_pointers is manager.expected_pool_pointers
    assert manager.kv_cache_pool_mapping is manager.expected_pool_mapping
    assert manager.index_scales.tolist() == [1]
    assert manager.kv_offset.tolist() == [1]
    assert manager.host_kv_cache_block_offsets.shape == (1, 2, 2, 4)


def test_disagg_role_mapper_kinds_default_to_indexed():
    manager = object.__new__(KVCacheManagerV2)

    assert manager.get_disagg_role_mapper_kinds() == {Role.ALL: MapperKind.INDEXED}
