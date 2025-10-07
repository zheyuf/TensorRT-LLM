"""
test_draft_len_schedule.py

Tests for dynamic draft length (draft_len_schedule) feature - Stage 1.

Stage 1 covers:
- NGramDrafter with dynamic draft_len
- ModelDrafter (2-model) with dynamic draft_len
- Draft-side compute savings only (target model still processes padded tokens)

Not covered in Stage 1:
- ChainDrafter/Eagle3 static loops (Stage 3)
- Target model compute savings (Stage 2)
"""

import os
import sys

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (DraftTargetDecodingConfig, KvCacheConfig,
                                 NGramDecodingConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import similar


# ============================================================================
# P0-1: Correctness check - generation quality doesn't change
# ============================================================================
@pytest.mark.parametrize("drafter_type,schedule", [
    ("ngram", {
        1: 3,
        4: 2,
        8: 1
    }),
    ("model_drafter", {
        1: 3,
        4: 2,
        8: 1
    }),
])
@pytest.mark.high_cuda_memory
def test_correctness_across_batch_sizes(drafter_type: str, schedule: dict):
    """
    Test output correctness with various schedules and batch sizes.

    This is the primary correctness test that validates:
    - Multiple different schedules work correctly
    - Output matches non-speculative baseline
    - Works across different batch size transitions
    - Both NGram and ModelDrafter function correctly

    This test replaces separate basic tests for each drafter type.
    """
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    memory_required = 30 if drafter_type == "model_drafter" else 20
    if total_mem_gb < memory_required:
        pytest.skip(
            f"Not enough memory (need {memory_required}GB, have {total_mem_gb:.1f}GB)"
        )

    models_path = llm_models_root()
    target_model = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"
    draft_model = f"{models_path}/llama-3.2-models/Llama-3.2-3B-Instruct"

    max_batch_size = 4
    max_draft_len = max(schedule.values())  # Use max from schedule
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=8192)

    llm_common_config = dict(
        model=target_model,
        backend='pytorch',
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )

    if drafter_type == "ngram":
        spec_config = NGramDecodingConfig(
            max_draft_len=max_draft_len,
            max_matching_ngram_size=2,
            draft_len_schedule=schedule,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=False,
        )
    else:
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=max_draft_len,
            speculative_model_dir=str(
                draft_model),  # Use smaller 1B model as draft
            draft_len_schedule=schedule,
        )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
        "Machine learning is",
        "The future of AI",
        "What is the capital of Australia?",
        "Explain in one sentence why the sky is blue.",
        "Who wrote the book 'Pride and Prejudice'?",
        "List three U.S. national holidays in the year 2025.",
        "Who painted the Mona Lisa?",
    ]
    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0,
        seed=42,
    )

    # With dynamic draft_len
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    # Reference without speculation
    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    # Verify correctness
    if drafter_type == "ngram":
        for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
            assert similar(text_spec, text_ref), \
                f"NGram output should be similar. Got:\nSpec: {text_spec}\nRef:  {text_ref}"
    else:
        for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
            assert similar(text_spec, text_ref), \
                f"ModelDrafter output should be similar. Got:\nSpec: {text_spec}\nRef:  {text_ref}"
