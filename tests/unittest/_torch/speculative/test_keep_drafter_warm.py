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
"""Keep-warm: an Eagle3 one-model drafter stays accurate across a zero-draft phase.

The draft KV cache is only written by the draft forward. When a
draft_len_schedule resolves to 0 the target keeps committing one token per
iteration, so unless the drafter's first forward still runs (keep-warm) those
positions are never written into the draft KV cache and the acceptance rate
collapses for the rest of the request once drafting resumes.

The schedule is driven by a mock that cycles K -> 0 -> K over fixed iteration
windows. The executor's draft-length resolution is instrumented to record, per
iteration, the resolved draft length and the number of draft tokens the
request had accepted in its most recently verified step.
"""

import os
from unittest.mock import patch

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

DRAFT_LEN = 3
# Executor iterations per phase of the K -> 0 -> K cycle in the long run that
# measures acceptance ...
PHASE_ITERATIONS = 40
MAX_TOKENS = 320
# ... and in the short run that checks the output against a fixed-draft-length
# run. Greedy outputs of different execution paths (1 vs. K+1 tokens per step,
# different accepted counts) drift apart numerically after a few hundred
# tokens, so the exact comparison is kept short, like the other spec tests.
SHORT_PHASE_ITERATIONS = 6
SHORT_MAX_TOKENS = 50
# The accepted count a record carries belongs to the step verified one
# iteration earlier (two with the overlap scheduler), so this many records
# around each phase transition are left out of the phase averages.
TRANSITION_MARGIN = 4
PROMPT = (
    "Here is a detailed, step-by-step explanation of how photosynthesis "
    "works, written for a curious high-school student:\n\n"
)


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    """The schedule and the executor are patched in-process."""
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


def _model_dirs():
    models_root = llm_models_root()
    target = f"{models_root}/Qwen3/Qwen3-8B"
    draft = f"{models_root}/Qwen3/qwen3_8b_eagle3"
    if not os.path.exists(target) or not os.path.exists(draft):
        pytest.skip("Qwen3-8B and its EAGLE3 head are required")
    return target, draft


def _llm_kwargs(target_model_dir):
    return dict(
        model=target_model_dir,
        max_batch_size=1,
        max_num_tokens=8192,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.6),
        cuda_graph_config=CudaGraphConfig(),
    )


def _spec_config(draft_model_dir, with_schedule):
    return Eagle3DecodingConfig(
        max_draft_len=DRAFT_LEN,
        speculative_model=draft_model_dir,
        # Any value works: the mock decides the draft length per iteration.
        draft_len_schedule={1: DRAFT_LEN} if with_schedule else None,
    )


def _phased_draft_len(draft_len_schedule, batch_size, max_draft_len, min_draft_len=0):
    """K for phase_iterations calls, then 0 for phase_iterations, then K."""
    call = _phased_draft_len.calls
    _phased_draft_len.calls += 1
    phase = min(call // _phased_draft_len.phase_iterations, 2)
    return max(0 if phase == 1 else DRAFT_LEN, min_draft_len)


def _generate_with_phased_schedule(llm, sampling_params, phase_iterations):
    """Generate under the K -> 0 -> K schedule.

    Returns the output text and one record per executor iteration:
    ``(resolved draft length, mean accepted draft tokens of the generation
    requests' most recently verified step, or None without generation
    requests)``.
    """
    executor = llm._executor.engine
    original = executor._handle_dynamic_draft_len
    records = []

    def instrumented(scheduled_batch):
        original(scheduled_batch)
        gens = scheduled_batch.generation_requests
        accepted = (sum(r.py_num_accepted_draft_tokens for r in gens) / len(gens)) if gens else None
        records.append((executor.model_engine.runtime_draft_len, accepted))

    _phased_draft_len.calls = 0
    _phased_draft_len.phase_iterations = phase_iterations
    with (
        patch(
            "tensorrt_llm._torch.speculative.utils.get_draft_len_for_batch_size",
            new=_phased_draft_len,
        ),
        patch.object(executor, "_handle_dynamic_draft_len", new=instrumented),
    ):
        outputs = llm.generate([PROMPT], sampling_params)
    return outputs[0].outputs[0].text, records


def _mean_acceptance_before_and_after(records, phase_iterations=PHASE_ITERATIONS):
    """Mean acceptance length (1 + accepted draft tokens) of the drafting
    iterations before the zero-draft phase and after it."""
    zero = [i for i, (k, _) in enumerate(records) if k == 0]
    assert len(zero) >= phase_iterations // 2, (
        f"only {len(zero)} zero-draft iterations recorded: {records[:12]}..."
    )
    first_zero, last_zero = zero[0], zero[-1]

    def phase_mean(indices):
        values = [
            1 + records[i][1] for i in indices if records[i][1] is not None and records[i][0] > 0
        ]
        assert len(values) >= phase_iterations // 2, (
            f"{len(values)} usable drafting iterations: {records}"
        )
        return sum(values) / len(values)

    before = phase_mean(range(TRANSITION_MARGIN, first_zero))
    after = phase_mean(range(last_zero + 1 + TRANSITION_MARGIN, len(records)))
    return before, after


@pytest.mark.high_cuda_memory
def test_keep_warm_preserves_acceptance_across_zero_draft_phase(enforce_single_worker):
    """With keep-warm the drafter resumes from a KV cache that tracked the
    target through the zero-draft phase, so acceptance recovers to its
    pre-phase level, and a short K -> 0 -> K run produces the same output as a
    fixed-draft-length run."""
    target_model_dir, draft_model_dir = _model_dirs()
    spec_config = _spec_config(draft_model_dir, with_schedule=True)
    assert spec_config.keep_drafter_warm
    assert spec_config.min_runtime_draft_len == 0
    long_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0, ignore_eos=True)
    short_params = SamplingParams(max_tokens=SHORT_MAX_TOKENS, temperature=0)

    llm_kwargs = _llm_kwargs(target_model_dir)
    with LLM(**llm_kwargs, speculative_config=spec_config) as llm:
        _, records = _generate_with_phased_schedule(llm, long_params, PHASE_ITERATIONS)
        text_short, _ = _generate_with_phased_schedule(llm, short_params, SHORT_PHASE_ITERATIONS)
    before, after = _mean_acceptance_before_and_after(records)
    print(
        f"[keep-warm] mean acceptance length before / after the zero-draft "
        f"phase: {before:.3f} / {after:.3f}"
    )
    assert before >= 1.5, f"Eagle3 acceptance length too low to start with: {before:.2f}"
    assert after >= 0.7 * before, (
        f"acceptance length collapsed after the zero-draft phase: "
        f"{before:.2f} before vs {after:.2f} after"
    )

    # Reference: the same drafter with a fixed draft length and no schedule.
    # Greedy speculative decoding is lossless, so this is the plain-decoding
    # output; it also exercises the same one-model sampler as the run above.
    with LLM(
        **llm_kwargs, speculative_config=_spec_config(draft_model_dir, with_schedule=False)
    ) as llm_ref:
        text_ref = llm_ref.generate([PROMPT], short_params)[0].outputs[0].text
    assert text_short == text_ref


@pytest.mark.high_cuda_memory
def test_skipping_the_drafter_at_zero_draft_len_goes_stale(enforce_single_worker, monkeypatch):
    """Control for the test above: with keep-warm disabled the zero-draft phase
    leaves the committed positions unwritten in the draft KV cache and the
    acceptance length after the phase drops below the level before it."""
    target_model_dir, draft_model_dir = _model_dirs()
    # Let the schedule reach 0 but take the target-only skip path there, i.e.
    # the behaviour before keep-warm existed.
    monkeypatch.setattr(DecodingBaseConfig, "keep_drafter_warm", property(lambda self: False))
    monkeypatch.setattr(DecodingBaseConfig, "min_runtime_draft_len", property(lambda self: 0))
    spec_config = _spec_config(draft_model_dir, with_schedule=True)
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0, ignore_eos=True)
    with LLM(**_llm_kwargs(target_model_dir), speculative_config=spec_config) as llm:
        spec_worker = llm._executor.engine.model_engine.model.spec_worker
        assert not spec_worker._keep_drafter_warm
        _, records = _generate_with_phased_schedule(llm, sampling_params, PHASE_ITERATIONS)
    before, after = _mean_acceptance_before_and_after(records)
    print(
        f"[stale] mean acceptance length before / after the zero-draft "
        f"phase: {before:.3f} / {after:.3f}"
    )
    assert after < 0.9 * before, (
        f"expected a stale draft KV cache to lower acceptance after the "
        f"zero-draft phase, got {before:.2f} before vs {after:.2f} after"
    )
