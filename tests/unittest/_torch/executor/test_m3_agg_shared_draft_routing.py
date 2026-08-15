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
"""Routing guards for MiniMax-M3 aggregated shared draft KV."""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

_DISABLE_ENV = MiniMaxM3KVCacheManagerV2.aggregated_shared_draft_disable_env


class _Eagle3OneModelMode:
    @staticmethod
    def use_one_engine() -> bool:
        return True

    @staticmethod
    def is_dspark() -> bool:
        return False

    @staticmethod
    def is_eagle3_one_model() -> bool:
        return True


class _SpecConfig:
    def __init__(
        self,
        *,
        max_draft_len: int = 3,
        max_total_draft_tokens: int = 3,
        eagle_choices: list[list[int]] | None = None,
        use_dynamic_tree: bool = False,
        sa_config: object | None = None,
        allow_separate: bool = True,
    ) -> None:
        self.spec_dec_mode = _Eagle3OneModelMode()
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.eagle_choices = eagle_choices
        self.use_dynamic_tree = use_dynamic_tree
        self.sa_config = sa_config
        self._allow_separate_draft_kv_cache = allow_separate


class _OtherSharedManager:
    supports_shared_draft_layers = True


def _make_creator(
    manager_cls: type,
    *,
    spec_config: _SpecConfig | None = None,
    attention_dp: bool = False,
    is_disagg: bool = False,
    tp_size: int = 4,
) -> KvCacheCreator:
    creator = object.__new__(KvCacheCreator)
    creator._mapping = Mock(enable_attention_dp=attention_dp, tp_size=tp_size)
    creator._kv_cache_manager_cls = manager_cls
    creator._is_disagg = is_disagg
    creator._speculative_config = spec_config or _SpecConfig()
    return creator


def test_linear_agentx_config_shares_in_aggregated_attention_tp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    creator = _make_creator(MiniMaxM3KVCacheManagerV2)

    assert not creator._should_create_separate_draft_kv_cache()


def test_emergency_disable_uses_separate_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_DISABLE_ENV, "1")
    creator = _make_creator(MiniMaxM3KVCacheManagerV2)

    assert creator._should_create_separate_draft_kv_cache()


def test_aggregated_route_is_model_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    creator = _make_creator(_OtherSharedManager)

    assert creator._should_create_separate_draft_kv_cache()


def test_disaggregated_route_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    creator = _make_creator(MiniMaxM3KVCacheManagerV2, is_disagg=True)

    assert creator._should_create_separate_draft_kv_cache()


def test_disaggregated_shared_route_is_preserved_when_separate_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The policy and rollback variable are scoped to the newly enabled
    # aggregated path. They must not change a pre-existing disaggregated route,
    # including a tree configuration that this patch does not claim to fix.
    monkeypatch.setenv(_DISABLE_ENV, "1")
    creator = _make_creator(
        MiniMaxM3KVCacheManagerV2,
        spec_config=_SpecConfig(
            max_total_draft_tokens=6,
            use_dynamic_tree=True,
            allow_separate=False,
        ),
        is_disagg=True,
    )

    assert not creator._should_create_separate_draft_kv_cache()


def test_attention_dp_existing_tree_route_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    creator = _make_creator(
        MiniMaxM3KVCacheManagerV2,
        spec_config=_SpecConfig(max_total_draft_tokens=6, use_dynamic_tree=True),
        attention_dp=True,
    )

    assert not creator._should_create_separate_draft_kv_cache()


@pytest.mark.parametrize(
    "spec_config",
    [
        pytest.param(
            _SpecConfig(eagle_choices=[[0], [0, 0], [0, 0, 0]]),
            id="static-tree",
        ),
        pytest.param(
            _SpecConfig(max_total_draft_tokens=6, use_dynamic_tree=True),
            id="dynamic-tree",
        ),
        pytest.param(
            _SpecConfig(max_total_draft_tokens=4),
            id="non-linear-token-count",
        ),
        pytest.param(_SpecConfig(max_draft_len=4, max_total_draft_tokens=4), id="draft-len-4"),
        pytest.param(_SpecConfig(sa_config=object()), id="sa-enhanced"),
    ],
)
def test_unqualified_config_falls_back_to_separate_manager(spec_config: _SpecConfig) -> None:
    creator = _make_creator(MiniMaxM3KVCacheManagerV2, spec_config=spec_config)

    assert creator._should_create_separate_draft_kv_cache()


def test_tp2_h2_noncontiguous_relocation_is_routed_away_from_shared_pool() -> None:
    # MSA overrides the manager's class-default NHD layout to HND at runtime.
    # MiniMax-M3 TP2 then has H=2 target KV heads per rank. A dynamic tree
    # emits accepted-token indices and would enter relocation, but its P128
    # [H, token, D] offsets do not match the draft view's P32
    # [subpage, H, token, D] offsets. Relocation also ignores M3's mega-slot
    # mapping and stride. Preserve the pre-existing aggregated separate-manager
    # route instead of expanding the new shared path; this guard does not claim
    # to validate M3 tree relocation itself.
    spec_config = _SpecConfig(max_total_draft_tokens=6, use_dynamic_tree=True)
    creator = _make_creator(
        MiniMaxM3KVCacheManagerV2,
        spec_config=spec_config,
        tp_size=2,
    )

    assert creator._mapping.tp_size == 2
    assert creator._should_create_separate_draft_kv_cache()
