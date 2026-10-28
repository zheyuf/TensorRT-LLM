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

from collections import OrderedDict, deque

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker

pytestmark = pytest.mark.cpu_only


def _prepare(worker: DFlashWorker, request_ids: list[int]) -> None:
    metadata = object.__new__(DFlashSpecMetadata)
    metadata.request_ids = request_ids
    metadata.num_generations = len(request_ids)
    metadata.batch_indices_cuda = torch.zeros(8, dtype=torch.int)
    metadata._dflash_worker = worker
    metadata.prepare()


def test_a_returning_generation_request_resumes_its_stored_context():
    num_slots = 4
    worker = object.__new__(DFlashWorker)
    worker._ctx_buf_inited = True
    worker._ctx_len = torch.zeros(num_slots, dtype=torch.long)
    worker._ctx_len_host = [0] * num_slots
    worker._ctx_base_pages = torch.zeros(num_slots, dtype=torch.long)
    worker._ctx_base_host = [0] * num_slots
    worker._ctx_stored = torch.zeros(num_slots, dtype=torch.bool)
    worker._ctx_stored_host = [False] * num_slots
    worker._parked_ctx_base = OrderedDict()
    worker._parked_ctx_capacity = 4 * num_slots
    worker._req_to_slot = {}
    worker._req_ctx_pos = {}
    worker._dummy_slot = num_slots - 1
    worker._free_slots = deque(range(num_slots - 1))
    worker._graph_dummy_id_floor = 1 << 62
    worker._batch_to_slot = torch.zeros(num_slots - 1, dtype=torch.long)
    worker._ctx_block_tables = torch.zeros((num_slots, 4), dtype=torch.int32)
    slot = worker._assign_slot(11)
    worker._write_ctx_len({slot: 500}, {slot: 3})

    # Request 11 sits one iteration out (e.g. suspended by the scheduler), then returns.
    _prepare(worker, [12])
    assert dict(worker._parked_ctx_base) == {11: 3}
    _prepare(worker, [12, 11])

    slot = worker._req_to_slot[11]
    assert worker._ctx_stored_host[slot] and bool(worker._ctx_stored[slot])
    assert worker._ctx_base_host[slot] == 3 and int(worker._ctx_base_pages[slot]) == 3
    assert not worker._parked_ctx_base
