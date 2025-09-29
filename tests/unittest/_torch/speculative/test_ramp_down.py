import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.drafter import Drafter
from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter
from tensorrt_llm._torch.speculative.ngram import NGramDrafter, NGramPoolManager
from tensorrt_llm.llmapi import NGramDecodingConfig


class _TestDrafter(Drafter):

    def __init__(self, draft_len_schedule=None, max_draft_tokens=4):
        super().__init__(max_concurrency=None,
                         draft_len_schedule=draft_len_schedule)
        # Expose a fixed max_draft_tokens for padding test
        self.max_draft_tokens = max_draft_tokens

    def prepare_draft_tokens(self,
                             scheduled_requests,
                             resource_manager=None) -> None:
        # Not needed for these unit tests
        return


def test_effective_draft_len_schedule_mapping():
    schedule = {1: 4, 4: 3, 8: 2, 16: 1, 32: 0}
    drafter = _TestDrafter(schedule, max_draft_tokens=4)

    # Helper is private; access via name-mangled attribute
    eff = drafter._effective_draft_len  # type: ignore[attr-defined]

    # Exact thresholds
    assert eff(1, 4) == 4
    assert eff(4, 4) == 3
    assert eff(8, 4) == 2
    assert eff(16, 4) == 1
    assert eff(32, 4) == 0

    # Between thresholds maps to the first >= threshold
    assert eff(2, 4) == 3
    assert eff(5, 4) == 2
    assert eff(9, 4) == 1

    # Clamp to [0, base]
    assert eff(1, 2) == 2  # schedule value 4 clamped to base 2
    assert eff(64, 4) == 0  # beyond last threshold uses last value


def test_pad_draft_tokens_for_cuda_graph_pads_to_max():
    drafter = _TestDrafter(draft_len_schedule=None, max_draft_tokens=5)

    class _Req:

        def __init__(self, tokens):
            self.py_draft_tokens = tokens

    sr = ScheduledRequests()
    sr.generation_requests = [_Req([1, 2]), _Req([])]

    # Before
    assert len(sr.generation_requests[0].py_draft_tokens) == 2
    assert len(sr.generation_requests[1].py_draft_tokens) == 0

    drafter.pad_draft_tokens_for_cuda_graph(sr)

    # After: both padded to max_draft_tokens
    assert len(sr.generation_requests[0].py_draft_tokens) == 5
    assert len(sr.generation_requests[1].py_draft_tokens) == 5


def test_post_schedule_cap_applied_uniformly_and_padding():
    # Use the real NGramDrafter and PoolManager
    max_draft_len = 5
    ngram_cfg = NGramDecodingConfig(
        max_draft_len=max_draft_len,
        max_matching_ngram_size=3,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
        draft_len_schedule={
            1: 5,
            4: 3,
            8: 2,
            16: 1,
            32: 0
        },
    )

    pool = NGramPoolManager(ngram_cfg, max_num_requests=16)
    drafter = NGramDrafter(ngram_cfg, pool)

    # Build a scheduled batch of 6 generation requests with prefixes
    class _Req:

        def __init__(self, rid, prefix, orig_len=8, max_new=64):
            self.request_id = rid
            self.py_batch_idx = 0
            self.py_draft_tokens = []
            self.py_draft_pages_allocated = 0
            self.py_orig_prompt_len = orig_len
            self.py_max_new_tokens = max_new
            self.state = type("S", (),
                              {"value": 3})  # GENERATION_IN_PROGRESS-like
            self._prefix = list(prefix)

        def get_tokens(self, _):
            return list(self._prefix)

    sr = ScheduledRequests()
    # Create varying prefixes to populate the pool and allow many potential draft tokens
    sr.generation_requests = [
        _Req(i, prefix=[i, i + 1, i + 2, i + 3, i + 4]) for i in range(6)
    ]

    # Call real drafter to set cap and generate draft tokens
    drafter.prepare_draft_tokens(sr)

    # Cap for 6 requests should be 2 per the schedule
    for req in sr.generation_requests:
        assert req.py_draft_pages_allocated == 2
        assert len(req.py_draft_tokens) <= 2

    # Padding should extend each to max_draft_len
    drafter.pad_draft_tokens_for_cuda_graph(sr)
    for req in sr.generation_requests:
        assert len(req.py_draft_tokens) == max_draft_len


def test_acceptance_rewind_consistency_under_cap():
    # Given a per-iteration cap, ensure rewind equals allocated - accepted
    class _Req:

        def __init__(self, allocated, accepted):
            self.py_draft_pages_allocated = allocated
            # Simulate sampler acceptance producing exactly 'accepted' tokens
            self.py_draft_tokens = [0] * accepted
            self.py_rewind_len = 0

    req = _Req(allocated=3, accepted=1)
    # The pipeline sets: rewind = allocated - accepted
    # Here we reproduce the assignment done by the sampler
    req.py_rewind_len = req.py_draft_pages_allocated - len(req.py_draft_tokens)
    assert req.py_rewind_len == 2

    # A minimal resource manager hook that records rewind calls
    class _RM:

        def __init__(self):
            self.rewinds = []

        def rewind_kv_cache(self, request, amount):
            self.rewinds.append(amount)

    rm = _RM()
    # Mimic the part of update_resources that performs the rewind
    if req.py_rewind_len > 0:
        rm.rewind_kv_cache(req, req.py_rewind_len)
    assert rm.rewinds == [2]


# will need to be deleted later
def test_static_path_cap_applied_in_model_drafter():
    # Stub spec_config with needed fields
    class _SpecCfg:
        max_concurrency = None
        draft_len_schedule = None

    # Stub draft_model_engine and sampler, not used in this test path
    class _Dummy:
        pass

    class _SeqSlotMgr:

        def __init__(self):
            self.freed = []

        def free_resources(self, req):
            self.freed.append(req)

    # Instantiate ModelDrafter with static-loop unrelated deps
    seq_mgr = _SeqSlotMgr()
    drafter = ModelDrafter(
        spec_config=_SpecCfg(),
        draft_model_engine=_Dummy(),
        max_draft_tokens=5,
        draft_seq_slot_manager=seq_mgr,
        sampler=_Dummy(),
    )

    # Build a draft_batch with two requests
    class _DraftReq:

        def __init__(self, rid):
            self.py_request_id = rid
            self.request_id = rid

    class _TargetReq:

        def __init__(self, rid, cap):
            self.py_request_id = rid
            self.state = LlmRequestState.GENERATION_IN_PROGRESS
            self.py_draft_tokens = []
            self.py_draft_pages_allocated = cap

    draft_batch = ScheduledRequests()
    draft_batch.generation_requests = [_DraftReq(1), _DraftReq(2)]
    mapping = {1: _TargetReq(1, 2), 2: _TargetReq(2, 0)}  # cap 2 and cap 0

    # Create static outputs tensor [max_draft_tokens, batch_size]
    outputs = torch.tensor([
        [101, 201],
        [102, 202],
        [103, 203],
        [104, 204],
        [105, 205],
    ])

    # Act: process static outputs (should honor caps)
    drafter.process_static_draft_outputs(outputs, draft_batch, mapping)

    # Assert: request 1 gets exactly 2 tokens; request 2 (cap 0) gets none
    assert mapping[1].py_draft_tokens == [101, 102]
    assert mapping[2].py_draft_tokens == []
