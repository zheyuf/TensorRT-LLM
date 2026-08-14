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
"""Tests for the non-production MiniMax-M3 aggregated shared-draft A/B switch."""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

_OPT_IN_ENV = "TRTLLM_M3_ENABLE_AGG_SHARED_DRAFT_KV"


class _OtherSharedManager:
    supports_shared_draft_layers = True


def _make_creator(
    manager_cls: type, *, attention_dp: bool = False, is_disagg: bool = False
) -> KvCacheCreator:
    creator = object.__new__(KvCacheCreator)
    creator._mapping = Mock(enable_attention_dp=attention_dp)
    creator._kv_cache_manager_cls = manager_cls
    creator._is_disagg = is_disagg
    spec_config = Mock()
    spec_config.spec_dec_mode.use_one_engine.return_value = True
    spec_config.spec_dec_mode.is_dspark.return_value = False
    spec_config._allow_separate_draft_kv_cache = True
    creator._speculative_config = spec_config
    return creator


@pytest.mark.parametrize("env_value", [None, "0", "true"])
def test_agg_m3_defaults_to_separate_manager(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv(_OPT_IN_ENV, raising=False)
    else:
        monkeypatch.setenv(_OPT_IN_ENV, env_value)
    creator = _make_creator(MiniMaxM3KVCacheManagerV2)

    assert creator._should_create_separate_draft_kv_cache()


def test_agg_m3_explicit_opt_in_selects_shared_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_OPT_IN_ENV, "1")
    creator = _make_creator(MiniMaxM3KVCacheManagerV2)

    assert not creator._should_create_separate_draft_kv_cache()


def test_opt_in_is_model_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_OPT_IN_ENV, "1")
    creator = _make_creator(_OtherSharedManager)

    assert creator._should_create_separate_draft_kv_cache()


def test_opt_in_does_not_change_disaggregated_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_OPT_IN_ENV, "1")
    creator = _make_creator(MiniMaxM3KVCacheManagerV2, is_disagg=True)

    assert creator._should_create_separate_draft_kv_cache()


def test_attention_dp_keeps_existing_shared_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_OPT_IN_ENV, raising=False)
    creator = _make_creator(MiniMaxM3KVCacheManagerV2, attention_dp=True)

    assert not creator._should_create_separate_draft_kv_cache()
