# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit and integration tests for the MiniMax-M3 text bring-up.

Helper tests exercise config normalization, layer scheduling, and
routing-method scaling. ``test_text_checkpoint_loading`` loads the real
MiniMax-M3 checkpoint config / tokenizer / chat template, runs the
static keyspace coverage classifier on every key in the checkpoint's
safetensors index, and confirms that each ``language_model.*`` weight
is either mapped to a TRT-LLM text parameter (loaded) or intentionally
ignored with a documented reason. CUDA tests exercise attention module
construction and the multi-rank ADP negative-control path.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3Attention,
    _build_swiglu_oai_dense_mlp,
    _strip_language_model_prefix,
    _wrap_dict_as_config,
    get_moe_layer_ids,
    get_sparse_disable_index_value_layer_ids,
    get_sparse_layer_ids,
    get_text_config,
    is_minimax_m3_vl_config,
)
from tensorrt_llm._torch.models.modeling_utils import _load_weights_impl
from tensorrt_llm._torch.modules.fused_moe.routing import (
    MiniMaxM2MoeRoutingMethod,
    MiniMaxM3MoeRoutingMethod,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.mapping import Mapping

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NUM_HIDDEN_LAYERS = 7
_SPARSE_FREQ = [0, 0, 0, 1, 1, 1, 1]
_DISABLE_INDEX_VALUE = [0, 0, 0, 1, 1, 1, 1]
_MOE_LAYER_FREQ = [0, 0, 0, 1, 1, 1, 1]


def _make_text_config():
    """Build a SimpleNamespace mimicking the real M3 text config (trimmed)."""
    sparse_attention_config = {
        "use_sparse_attention": True,
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": list(_DISABLE_INDEX_VALUE),
        "sparse_attention_freq": list(_SPARSE_FREQ),
    }
    return SimpleNamespace(
        model_type="minimax_m3",
        hidden_size=6144,
        intermediate_size=3072,
        num_hidden_layers=_NUM_HIDDEN_LAYERS,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=200064,
        max_position_embeddings=524288,
        rms_norm_eps=1e-06,
        use_gemma_norm=True,
        attention_output_gate=False,
        rope_theta=5000000,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        hidden_act="swigluoai",
        use_qk_norm=True,
        qk_norm_type="per_head",
        tie_word_embeddings=False,
        dense_intermediate_size=12288,
        shared_intermediate_size=3072,
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        use_routing_bias=True,
        moe_layer_freq=list(_MOE_LAYER_FREQ),
        num_mtp_modules=1,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        routed_scaling_factor=2.0,
        sparse_attention_config=sparse_attention_config,
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
    )


def _make_vl_config():
    return SimpleNamespace(
        model_type="minimax_m3_vl",
        text_config=_make_text_config(),
        vision_config=SimpleNamespace(
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
        ),
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        image_token_index=200025,
        video_token_index=200026,
    )


# ---------------------------------------------------------------------------
# Shared helpers used by both CPU and CUDA tests
# ---------------------------------------------------------------------------


_DEFAULT_CHECKPOINT_PATH = f"{llm_models_root()}/MiniMax-M3"


def _checkpoint_path() -> str:
    return _DEFAULT_CHECKPOINT_PATH


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU-only unit tests
# ---------------------------------------------------------------------------


def test_is_minimax_m3_vl_config_detects_vl():
    assert is_minimax_m3_vl_config(_make_vl_config()) is True


def test_is_minimax_m3_vl_config_detects_text_only():
    assert is_minimax_m3_vl_config(_make_text_config()) is False


def test_is_minimax_m3_vl_config_falls_back_to_architectures():
    cfg = SimpleNamespace(
        model_type="custom",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
    )
    assert is_minimax_m3_vl_config(cfg) is True


def test_get_text_config_returns_text_subconfig():
    vl_cfg = _make_vl_config()
    text_cfg = get_text_config(vl_cfg)
    assert text_cfg is vl_cfg.text_config
    assert text_cfg.num_hidden_layers == _NUM_HIDDEN_LAYERS


def test_get_text_config_passthrough_for_text_only():
    text_cfg = _make_text_config()
    assert get_text_config(text_cfg) is text_cfg


def test_get_text_config_propagates_dtype_when_missing():
    vl_cfg = _make_vl_config()
    vl_cfg.text_config.torch_dtype = None
    out = get_text_config(vl_cfg)
    assert out.torch_dtype == "bfloat16"


def test_get_text_config_missing_text_attribute_raises():
    bad = SimpleNamespace(model_type="minimax_m3_vl")
    with pytest.raises(ValueError, match="text_config"):
        get_text_config(bad)


def test_get_sparse_layer_ids_splits_dense_and_sparse():
    dense, sparse = get_sparse_layer_ids(_make_text_config())
    assert dense == [0, 1, 2]
    assert sparse == [3, 4, 5, 6]


def test_get_sparse_layer_ids_falls_back_when_disabled():
    cfg = _make_text_config()
    cfg.sparse_attention_config["use_sparse_attention"] = False
    dense, sparse = get_sparse_layer_ids(cfg)
    assert dense == list(range(_NUM_HIDDEN_LAYERS))
    assert sparse == []


def test_get_sparse_layer_ids_falls_back_without_config():
    cfg = _make_text_config()
    cfg.sparse_attention_config = None
    dense, sparse = get_sparse_layer_ids(cfg)
    assert dense == list(range(_NUM_HIDDEN_LAYERS))
    assert sparse == []


def test_get_sparse_layer_ids_length_mismatch_raises():
    cfg = _make_text_config()
    cfg.sparse_attention_config["sparse_attention_freq"] = [0] * (_NUM_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="sparse_attention_freq length"):
        get_sparse_layer_ids(cfg)


def test_get_sparse_disable_index_value_layer_ids_matches_sparse():
    ids = get_sparse_disable_index_value_layer_ids(_make_text_config())
    assert ids == [3, 4, 5, 6]


def test_get_sparse_disable_index_value_no_config():
    cfg = _make_text_config()
    cfg.sparse_attention_config = None
    assert get_sparse_disable_index_value_layer_ids(cfg) == []


def test_get_moe_layer_ids_splits_dense_and_moe():
    dense, moe = get_moe_layer_ids(_make_text_config())
    assert dense == [0, 1, 2]
    assert moe == [3, 4, 5, 6]


def test_get_moe_layer_ids_all_moe_without_freq():
    cfg = _make_text_config()
    cfg.moe_layer_freq = None
    dense, moe = get_moe_layer_ids(cfg)
    assert dense == []
    assert moe == list(range(_NUM_HIDDEN_LAYERS))


def test_get_moe_layer_ids_length_mismatch_raises():
    cfg = _make_text_config()
    cfg.moe_layer_freq = [0] * (_NUM_HIDDEN_LAYERS - 1)
    with pytest.raises(ValueError, match="moe_layer_freq length"):
        get_moe_layer_ids(cfg)


# ---------------------------------------------------------------------------
# attention module transforms
# ---------------------------------------------------------------------------
#
# These tests construct :class:`MiniMaxM3Attention` with a tiny synthetic
# geometry and ``skip_create_weights_in_init=True`` so the Linear modules
# exist (with ``.in_features`` / ``.out_features`` set) but no weights are
# allocated. The base :class:`Attention` constructor reaches into CUDA-only
# paths (e.g. backend selection), so the tests run under
# ``pytest.mark.gpu`` + ``skipif(not _has_cuda())``. Geometry is bounded to
# a few KB.
#
# Coverage:
#  * Dense / sparse attention construction shapes match the configured
#    head_dim, head counts, and sparse index branch dimensions.
#  * Partial RoPE only rotates ``rotary_dim`` of ``head_dim`` channels.
#  * Per-head Gemma Q/K RMSNorm: q_norm / k_norm are RMSNorm with
#    ``use_gemma=True`` and ``hidden_size=head_dim``; the
#    :meth:`apply_qk_norm` reshape matches an independent hand-written
#    reference.
#  * Sparse index branch: ``index_q_proj`` is column-parallel and
#    projects to ``num_index_heads * sparse_index_dim``;
#    ``index_k_proj`` is **replicated** (tp_mode is None) and projects
#    to **only** ``sparse_index_dim`` (single K per token, broadcast
#    across all index heads for block-selection scoring) — this is the
#    SGLang reference contract, confirmed by the M3 checkpoint shape
#    ``(sparse_index_dim, hidden_size)``.
#  * Dense layers do not expose any index branch attributes (negative
#    control).
#  * Real M3 checkpoint shape for ``index_k_proj.weight`` is
#    ``(sparse_index_dim, hidden_size)`` = ``(128, 6144)``.


def _make_attention_test_config():
    """Return ``(text_config, ModelConfig)`` for the attention tests.

    Geometry is a scaled-down M3-shaped config: hidden_size=128, head_dim=32,
    num_heads=4, num_kv_heads=2, num_index_heads=2, sparse_index_dim=32,
    rotary_dim=16 (= head_dim * 0.5 — partial RoPE), 1 dense + 3 sparse
    layers. With ``skip_create_weights_in_init=True`` no Linear weight
    tensors are allocated, only metadata.
    """
    n_layers = 4
    sparse_cfg = {
        "use_sparse_attention": True,
        "sparse_index_dim": 32,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 4,
        "sparse_block_size": 16,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": [0, 1, 1, 1],
        "sparse_attention_freq": [0, 1, 1, 1],
    }
    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 128,
            "intermediate_size": 64,
            "num_hidden_layers": n_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 256,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "use_gemma_norm": True,
            "rope_theta": 10000.0,
            "rotary_dim": 16,
            "partial_rotary_factor": 0.5,
            "qk_norm_type": "per_head",
            "use_qk_norm": True,
            "sparse_attention_config": sparse_cfg,
            "torch_dtype": torch.bfloat16,
        }
    )
    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    return text_cfg, model_cfg


def _per_head_gemma_rms_norm_reference(x, weight, eps):
    """Hand-written reference for per-head Gemma RMSNorm.

    Matches :class:`RMSNorm.forward` with ``use_gemma=True``,
    ``residual=None``, ``is_nvfp4=False``: cast to float32 to compute
    variance, normalise, cast back to input dtype, then scale by
    ``(weight + 1)``. The per-head structure comes from reshaping the
    input to ``(-1, head_dim)`` before applying this function.
    """
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_norm = (x_f32 * torch.rsqrt(variance + eps)).to(input_dtype)
    return (weight + 1.0) * x_norm


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_dense_construction_matches_config():
    """Dense layer's QKV/O projection and per-head Q/K norm match config."""
    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
        disable_index_value=False,
    )

    head_dim = int(text_cfg.head_dim)
    num_heads = int(text_cfg.num_attention_heads)
    num_kv = int(text_cfg.num_key_value_heads)
    hidden = int(text_cfg.hidden_size)

    # Q/K/V projection (fused into qkv_proj).
    assert attn.num_heads == num_heads
    assert attn.num_key_value_heads == num_kv
    assert attn.head_dim == head_dim
    assert attn.head_dim_value == head_dim
    assert attn.q_size == num_heads * head_dim
    assert attn.kv_size == num_kv * head_dim
    assert attn.qkv_proj.in_features == hidden
    # With tp_size=1, out_features == q_size + 2 * kv_size.
    assert attn.qkv_proj.out_features == num_heads * head_dim + 2 * num_kv * head_dim

    # Output projection.
    assert attn.o_proj.in_features == num_heads * head_dim
    assert attn.o_proj.out_features == hidden

    # Per-head Gemma Q/K RMSNorm.
    assert attn.use_gemma_norm is True
    assert attn.qk_norm_type == "per_head"
    assert attn.q_norm.use_gemma is True
    assert attn.k_norm.use_gemma is True
    assert tuple(attn.q_norm.weight.shape) == (head_dim,)
    assert tuple(attn.k_norm.weight.shape) == (head_dim,)

    # Dense layers must not expose any index-branch attributes.
    assert attn.is_sparse_attention_layer is False
    for name in (
        "index_q_proj",
        "index_k_proj",
        "index_q_norm",
        "index_k_norm",
    ):
        assert not hasattr(attn, name), f"dense layer should not declare {name!r}"


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_partial_rope_dim_is_rotary_dim():
    """Partial RoPE rotates only ``rotary_dim`` of ``head_dim`` channels."""
    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    rotary_dim = int(text_cfg.rotary_dim)
    head_dim = int(text_cfg.head_dim)
    assert attn.pos_embd_params is not None
    assert attn.pos_embd_params.rope.dim == rotary_dim
    assert attn.pos_embd_params.rope.dim < head_dim, (
        f"partial RoPE expects rope.dim < head_dim, got {attn.pos_embd_params.rope.dim} >= {head_dim}"
    )
    # The base Attention class also stores the rotary embedding when
    # ``rope_fusion=False``; M3 sets ``rope_fusion=False`` explicitly.
    assert attn.rope_fusion is False
    assert attn.rotary_emb is not None


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_apply_qk_norm_matches_reference():
    """Verify ``apply_qk_norm`` does per-head Gemma RMSNorm and reshape-back."""
    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    head_dim = int(text_cfg.head_dim)
    eps = float(text_cfg.rms_norm_eps)

    # Set non-zero norm weights so the test catches any reshape /
    # weight-broadcast bugs (zero weights would mask many errors).
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    attn.q_norm.weight = torch.nn.Parameter(q_weight)
    attn.k_norm.weight = torch.nn.Parameter(k_weight)

    seq = 3
    q = torch.randn(seq, attn.q_size, dtype=dtype, device=device)
    k = torch.randn(seq, attn.kv_size, dtype=dtype, device=device)

    q_out, k_out = attn.apply_qk_norm(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

    # Hand-written reference: per-head reshape -> Gemma RMSNorm ->
    # reshape back. Identical computation, independent code path.
    q_ref = _per_head_gemma_rms_norm_reference(q.reshape(-1, head_dim), q_weight, eps).reshape(
        q.shape
    )
    k_ref = _per_head_gemma_rms_norm_reference(k.reshape(-1, head_dim), k_weight, eps).reshape(
        k.shape
    )

    # BF16 + possible flashinfer kernel: use a looser tolerance.
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_sparse_construction_matches_config():
    """Sparse layer adds index branch with the SGLang-correct shapes.

    Verifies the **bug fix** from the iter-4 work:
      * ``index_q_proj`` is column-parallel and projects to
        ``num_index_heads * sparse_index_dim``.
      * ``index_k_proj`` is replicated (``tp_mode is None``) and projects
        to **only** ``sparse_index_dim`` — a single replicated K per
        token, *not* per-head. This matches SGLang's ``ReplicatedLinear``
        and the M3 checkpoint's ``index_k_proj.weight`` shape
        ``(sparse_index_dim, hidden_size)``.
      * ``index_q_norm`` / ``index_k_norm`` are per-head Gemma RMSNorm
        of width ``sparse_index_dim``.
    """
    text_cfg, model_cfg = _make_attention_test_config()
    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])
    hidden = int(text_cfg.hidden_size)

    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )
    assert attn.is_sparse_attention_layer is True
    assert attn.disable_index_value is True

    # index_q_proj: per-head Q for the index branch. As of iter-15 this
    # is **replicated** (tp_mode=None) across TP ranks, not
    # column-parallel: the sparse forward consumes ``idx_q`` reshaped to
    # ``[num_tokens, num_index_heads, sparse_index_dim]`` and a
    # column-parallel split would slice the head dimension (breaking the
    # reshape at any ``tp_size > num_index_heads`` geometry, including
    # the TP=8 configuration the real-checkpoint smoke test now uses).
    # The replicated weight is small (~3 MiB BF16) so the per-rank
    # memory cost is negligible.
    assert attn.index_q_proj.in_features == hidden
    assert attn.index_q_proj.out_features == num_index_heads * sparse_index_dim
    assert attn.index_q_proj.tp_mode is None, (
        f"index_q_proj must be replicated (tp_mode=None) so the sparse "
        f"forward's `idx_q.view(num_tokens, num_index_heads, sparse_index_dim)` "
        f"reshape is well-defined at any TP geometry, got "
        f"{attn.index_q_proj.tp_mode!r}"
    )

    # index_k_proj: REPLICATED, only sparse_index_dim outputs.
    assert attn.index_k_proj.in_features == hidden
    assert attn.index_k_proj.out_features == sparse_index_dim, (
        f"index_k_proj.out_features must be sparse_index_dim={sparse_index_dim}, "
        f"got {attn.index_k_proj.out_features} (regression of the iter-4 fix)"
    )
    assert attn.index_k_proj.tp_mode is None, (
        f"index_k_proj must be replicated (tp_mode=None), got {attn.index_k_proj.tp_mode!r}"
    )

    # Per-head Gemma RMSNorm of width sparse_index_dim.
    assert attn.index_q_norm.use_gemma is True
    assert attn.index_k_norm.use_gemma is True
    assert tuple(attn.index_q_norm.weight.shape) == (sparse_index_dim,)
    assert tuple(attn.index_k_norm.weight.shape) == (sparse_index_dim,)

    # The sparse forward path now dispatches through the MiniMax-M3
    # sparse algorithm. Calling forward without metadata must raise a
    # clear RuntimeError pointing at the missing kv_cache_manager
    # (rather than silently returning garbage or crashing inside the
    # algorithm).
    try:
        attn.forward()
    except RuntimeError as e:
        msg = str(e)
        assert "attn_metadata" in msg or "kv_cache_manager" in msg, msg
    else:  # pragma: no cover
        raise AssertionError("sparse forward must raise RuntimeError when attn_metadata is None")


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_apply_index_qk_norm_matches_reference():
    """Sparse index per-head Gemma QK norm matches the hand-written reference.

    ``apply_index_qk_norm`` reshapes ``idx_q`` (``num_index_heads`` heads)
    and ``idx_k`` (1 replicated head) to ``(-1, sparse_index_dim)`` rows,
    applies the per-head Gemma RMSNorm, and reshapes back. The test sets
    non-zero norm weights, drives synthetic input, and compares against
    the same pure-torch reference used for the main Q/K norm.
    """
    text_cfg, model_cfg = _make_attention_test_config()
    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])

    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = float(text_cfg.rms_norm_eps)
    iq_weight = torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    ik_weight = torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    attn.index_q_norm.weight = torch.nn.Parameter(iq_weight)
    attn.index_k_norm.weight = torch.nn.Parameter(ik_weight)

    seq = 5
    idx_q = torch.randn(seq, num_index_heads * sparse_index_dim, dtype=dtype, device=device)
    idx_k = torch.randn(seq, sparse_index_dim, dtype=dtype, device=device)
    iq_out, ik_out = attn.apply_index_qk_norm(idx_q, idx_k)
    assert iq_out.shape == idx_q.shape
    assert ik_out.shape == idx_k.shape

    iq_ref = _per_head_gemma_rms_norm_reference(
        idx_q.reshape(-1, sparse_index_dim), iq_weight, eps
    ).reshape(idx_q.shape)
    ik_ref = _per_head_gemma_rms_norm_reference(
        idx_k.reshape(-1, sparse_index_dim), ik_weight, eps
    ).reshape(idx_k.shape)

    torch.testing.assert_close(iq_out, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ik_out, ik_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_dense_apply_index_qk_norm_raises():
    """Dense layers must reject ``apply_index_qk_norm`` calls."""
    _, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    idx_q = torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda")
    idx_k = torch.zeros(2, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="only valid on sparse attention layers"):
        attn.apply_index_qk_norm(idx_q, idx_k)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_real_config_index_branch_shapes():
    """Real M3 config → sparse-layer index branch has the checkpoint's shapes.

    Asserts the iter-4 fix in numbers:
      * ``index_q_proj.out_features == 512`` (= 4 * 128
        = ``num_index_heads * sparse_index_dim``).
      * ``index_k_proj.out_features == 128`` (= ``sparse_index_dim``)
        and ``tp_mode is None`` (replicated). The real
        ``index_k_proj.weight`` in the checkpoint has shape
        ``(128, 6144)``; a regression to the old
        ``num_index_heads * sparse_index_dim`` (512) would break weight
        loading at runtime.
      * ``index_q_norm.weight.shape == (128,)`` and
        ``index_k_norm.weight.shape == (128,)``.
    """
    pytest.importorskip("transformers")
    cfg = AutoConfig.from_pretrained(_checkpoint_path(), trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    # ``MiniMaxM3Model.__init__`` normalises ``torch_dtype`` to a real
    # ``torch.dtype`` before constructing layers (the HF config stores it
    # as the string ``"bfloat16"``). Mirror that here so the standalone
    # attention construction does not blow up inside the RMSNorm
    # ``torch.zeros(..., dtype=dtype)`` call.
    if isinstance(getattr(text_cfg, "torch_dtype", None), str):
        text_cfg.torch_dtype = torch.bfloat16
    elif getattr(text_cfg, "torch_dtype", None) is None:
        text_cfg.torch_dtype = torch.bfloat16

    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )

    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])
    assert num_index_heads == 4
    assert sparse_index_dim == 128

    # index_q_proj: 4 * 128 = 512 out, replicated (tp_mode=None) as of
    # iter-15. The downstream sparse forward reshapes ``idx_q`` to
    # ``[num_tokens, num_index_heads, sparse_index_dim]``; a
    # column-parallel split would slice the head dimension and break
    # that reshape at any ``tp_size > num_index_heads`` geometry
    # (including TP=8 used by the real-checkpoint smoke test). The
    # replicated weight is ~3 MiB BF16 — the per-rank memory cost is
    # negligible.
    assert attn.index_q_proj.in_features == int(text_cfg.hidden_size)
    assert attn.index_q_proj.out_features == num_index_heads * sparse_index_dim
    assert attn.index_q_proj.tp_mode is None

    # index_k_proj: 128 out (NOT 512), replicated.
    assert attn.index_k_proj.in_features == int(text_cfg.hidden_size)
    assert attn.index_k_proj.out_features == sparse_index_dim
    assert attn.index_k_proj.tp_mode is None

    # Per-head Gemma index norms: width sparse_index_dim.
    assert tuple(attn.index_q_norm.weight.shape) == (sparse_index_dim,)
    assert tuple(attn.index_k_norm.weight.shape) == (sparse_index_dim,)
    assert attn.index_q_norm.use_gemma is True
    assert attn.index_k_norm.use_gemma is True

    # Main Q/K norm shapes follow head_dim, not hidden_size.
    head_dim = int(text_cfg.head_dim)
    assert tuple(attn.q_norm.weight.shape) == (head_dim,)
    assert tuple(attn.k_norm.weight.shape) == (head_dim,)

    # Partial RoPE rotates rotary_dim of head_dim.
    assert attn.pos_embd_params.rope.dim == int(text_cfg.rotary_dim)
    assert attn.pos_embd_params.rope.dim < head_dim


# ---------------------------------------------------------------------------
# Routing-method unit tests (CPU)
# ---------------------------------------------------------------------------


def test_minimax_m3_routing_method_applies_routed_scaling_factor():
    """MiniMaxM3MoeRoutingMethod multiplies renormalized weights by scaling."""
    num_experts = 8
    top_k = 2
    bias = torch.zeros(num_experts, dtype=torch.float32)

    def bias_fn():
        return bias

    torch.manual_seed(0)
    logits = torch.randn(4, num_experts, dtype=torch.float32) * 3.0

    base = MiniMaxM2MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=bias_fn,
    )
    scaled = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=bias_fn,
        routed_scaling_factor=2.0,
    )

    base_idx, base_weights = base.apply(logits)
    scaled_idx, scaled_weights = scaled.apply(logits)

    torch.testing.assert_close(base_idx, scaled_idx)
    torch.testing.assert_close(scaled_weights, base_weights * 2.0, rtol=0, atol=0)


def test_minimax_m3_routing_method_default_scale_is_identity():
    num_experts = 8
    top_k = 2
    bias = torch.zeros(num_experts, dtype=torch.float32)

    torch.manual_seed(0)
    logits = torch.randn(4, num_experts, dtype=torch.float32) * 3.0

    base = MiniMaxM2MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
    )
    same = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=1.0,
    )
    _, base_weights = base.apply(logits)
    _, same_weights = same.apply(logits)
    torch.testing.assert_close(same_weights, base_weights, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 needs CUDA")
def test_text_norm_weights_real_loader_smoke():
    """real ``_load_weights_impl`` populates norm parameters.

    Constructs a memory-safe stub containing the top-level ``model.norm``
    and the first decoder layer's ``input_layernorm`` and
    ``post_attention_layernorm`` (each a 6144-dim BF16
    :class:`RMSNorm`, ~12 KB on CUDA), reads the corresponding tensors
    from the real checkpoint via ``safetensors``, strips the
    ``language_model.`` prefix exactly as the M3 VL wrapper does, and
    invokes :func:`_load_weights_impl` end-to-end. The test fails if any
    target parameter remains at its zero-initialisation, proving the
    canonical loader walks the module tree and copies the correct source
    keys for these BF16 parameters.

    Why this slice: ``input_layernorm`` / ``post_attention_layernorm`` /
    ``model.norm`` exercise the loader's ``filter_weights`` + per-module
    copy path on real tensor handles.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)

    eps = float(text_cfg.rms_norm_eps)
    use_gemma = bool(getattr(text_cfg, "use_gemma_norm", False))
    hidden = int(text_cfg.hidden_size)
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Memory-safe stub matching the M3 module tree for the three norms.
    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )
            self.post_attention_layernorm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )

    class _ModelInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )
            self.layers = nn.ModuleList([_Layer()])

    class _LoaderStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_config = ModelConfig(pretrained_config=text_cfg, mapping=Mapping())
            self.config = text_cfg
            self.model = _ModelInner()

    stub = _LoaderStub()

    # Sanity: the RMSNorm init zero-fills the gemma path; if any tensor is
    # already equal to its checkpoint value we would not be testing the copy.
    assert torch.all(stub.model.norm.weight == 0)
    assert torch.all(stub.model.layers[0].input_layernorm.weight == 0)
    assert torch.all(stub.model.layers[0].post_attention_layernorm.weight == 0)

    targets = [
        "language_model.model.norm.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
    ]

    # Group source keys by safetensors shard for efficient reads.
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: dict = {}
    for key in targets:
        shard = weight_map[key]
        by_shard.setdefault(shard, []).append(key)

    raw_weights: dict = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as sf:
            for k in keys:
                raw_weights[k] = sf.get_tensor(k)

    # Strip `language_model.` exactly as `MiniMaxM3VLForConditionalGeneration`
    # does at load time. Confirm the stripped keyspace matches the inner
    # loader's expectation.
    text_weights, ignored = _strip_language_model_prefix(raw_weights)
    assert ignored == {}, f"unexpectedly stripped {len(ignored)} entries: {ignored!r}"
    assert set(text_weights.keys()) == {
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    }

    # Invoke the canonical loader. `_load_weights_impl` walks the stub's
    # module tree and uses the generic per-parameter copy fallback because
    # RMSNorm does not define ``load_weights``. Disable the parallel
    # executor so a failure surfaces immediately rather than as a thread
    # traceback (the parallel path is exercised in production; for this
    # tiny 3-module slice the serial walk is what the test should observe).
    os.environ["TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL"] = "True"
    try:
        _load_weights_impl(stub, text_weights, allow_partial_loading=True)
    finally:
        os.environ.pop("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", None)

    # The three norms should now hold the source tensors' values.
    torch.testing.assert_close(
        stub.model.norm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.norm.weight"].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        stub.model.layers[0].input_layernorm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.layers.0.input_layernorm.weight"].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        stub.model.layers[0].post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.layers.0.post_attention_layernorm.weight"].to(
            torch.bfloat16
        ),
    )

    # Independent sanity: at least one tensor must be non-zero, i.e. the
    # loader actually performed the copy.
    assert torch.any(stub.model.norm.weight != 0)


# ---------------------------------------------------------------------------
# Attention DP construction for the dense MLP / MoE shared expert
# ---------------------------------------------------------------------------
#
# Under ``enable_attention_dp=True`` each rank processes a rank-local
# set of tokens. The base ``Attention`` re-maps tp_size=1 internally so
# qkv_proj/o_proj are replicated. The MiniMax-M3 dense MLP / MoE shared
# expert (a ``GatedMLP`` built by ``_build_swiglu_oai_dense_mlp``) must
# follow the same pattern: ``overridden_tp_size=1`` + ``reduce_output=
# False`` so it runs replicated under ADP. A ROW-parallel all-reduce
# across ADP ranks would mix outputs from independent rank-local token
# sets and produce wrong results. This test pins the construction-level
# invariants of that contract.


@pytest.mark.gpu
@pytest.mark.skipif(
    not _has_cuda(), reason="MiniMax-M3 MLP construction needs CUDA backend selection"
)
def test_minimax_m3_swiglu_oai_dense_mlp_under_adp_is_replicated():
    """``_build_swiglu_oai_dense_mlp`` under ``enable_attention_dp=True``
    must produce a ``GatedMLP`` whose Linear layers are replicated
    (full-width in_features/out_features) with ``reduce_output=False``
    on ``down_proj``. Without this the dense MLP / shared expert would
    all-reduce across ADP ranks and mix outputs from independent
    rank-local token sets.
    """
    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 128,
            "intermediate_size": 64,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "torch_dtype": torch.bfloat16,
        }
    )
    # Simulate ADP with tp_size=4; world_size matches so Mapping
    # validation passes.
    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(world_size=4, tp_size=4, pp_size=1, rank=0, enable_attention_dp=True),
        skip_create_weights_in_init=True,
    )

    intermediate = 64
    hidden = int(text_cfg.hidden_size)
    mlp = _build_swiglu_oai_dense_mlp(
        model_config=model_cfg,
        intermediate_size=intermediate,
    )
    # Under ADP the gate_up_proj and down_proj must be replicated:
    # in_features and out_features keep their full size and the
    # Linear-internal tp_size is 1.
    assert mlp.gate_up_proj.in_features == hidden
    assert mlp.gate_up_proj.out_features == 2 * intermediate, (
        "ADP gate_up_proj must be full-width (replicated), not sharded by the global TP size"
    )
    assert mlp.gate_up_proj.tp_size == 1
    assert mlp.down_proj.in_features == intermediate, (
        "ADP down_proj must be full-width (replicated), not ROW-sharded across the global TP group"
    )
    assert mlp.down_proj.out_features == hidden
    assert mlp.down_proj.tp_size == 1
    assert mlp.down_proj.reduce_output is False, (
        "ADP down_proj must skip the cross-rank all-reduce; otherwise it "
        "mixes outputs across independent rank-local token sets"
    )


# ---------------------------------------------------------------------------
# Overlap-scheduler correction hook (on_update_kv_lens)
# ---------------------------------------------------------------------------
#
# With the overlap scheduler + speculative decoding, prepare() builds the M3
# attachment from optimistic cached counts (full draft acceptance assumed);
# the engine later corrects kv_lens_cuda on device and invokes
# on_update_kv_lens(). These tests build an attachment with optimistic
# values, apply per-row rejection deltas to kv_lens_cuda, run the hook, and
# assert the attachment matches a from-scratch build using the true counts.


def _build_extend_attachment(device, req_to_token, prefix_lens, ext_lens):
    """Mirror build_runtime_metadata_from_kv_manager's extend construction."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseAttentionMetadata,
    )

    batch = len(prefix_lens)
    table_width = int(req_to_token.shape[1])
    seq_lens = [p + e for p, e in zip(prefix_lens, ext_lens)]
    cu = [0]
    out_cache_loc = []
    q_batch_row = []
    q_positions = []
    for b in range(batch):
        for offset in range(ext_lens[b]):
            pos = prefix_lens[b] + offset
            # Mirror derive_q_positions_and_cache_slots: optimistic
            # positions overhanging the page table gather a clamped
            # placeholder slot; q_positions keep the optimistic value.
            out_cache_loc.append(int(req_to_token[b, min(pos, table_width - 1)].item()))
            q_batch_row.append(b)
            q_positions.append(pos)
        cu.append(cu[-1] + ext_lens[b])
    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=True,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=list(ext_lens),
        q_batch_row=torch.tensor(q_batch_row, dtype=torch.int32, device=device),
        q_positions=torch.tensor(q_positions, dtype=torch.int32, device=device),
    )
    return meta, torch.tensor(out_cache_loc, dtype=torch.int32, device=device)


def _make_hook_host(device, meta, out_cache_loc, kv_lens):
    """Minimal MiniMaxM3AttentionMetadata carrying just what the hook reads."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        get_minimax_m3_attention_metadata_cls,
    )

    cls = get_minimax_m3_attention_metadata_cls()
    host = cls.__new__(cls)  # skip dataclass init; the hook reads 3 attrs
    host.enable_flash_mla = False  # consumed by super().on_update_kv_lens()
    host.kv_lens_cuda = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    host.minimax_m3 = {"metadata": meta, "out_cache_loc": out_cache_loc}
    return host


@pytest.mark.skipif(not _has_cuda(), reason="device tensors required")
def test_on_update_kv_lens_rederives_extend_attachment():
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, max_kv_len, draft_w = 3, 32, 4
    # Random-but-distinct slot table so wrong gathers are detectable.
    req_to_token = (
        torch.randperm(batch * max_kv_len, dtype=torch.int32).reshape(batch, max_kv_len).to(device)
    )

    true_prefix = [7, 12, 5]
    rejected = [2, 0, 3]  # per-row rejected draft tokens from step N
    optimistic_prefix = [p + r for p, r in zip(true_prefix, rejected)]
    ext = [draft_w] * batch

    meta, out_loc = _build_extend_attachment(device, req_to_token, optimistic_prefix, ext)
    truth_meta, truth_out_loc = _build_extend_attachment(device, req_to_token, true_prefix, ext)

    corrected_kv_lens = [p + e for p, e in zip(true_prefix, ext)]
    host = _make_hook_host(device, meta, out_loc, corrected_kv_lens)
    host.on_update_kv_lens()

    torch.testing.assert_close(meta.seq_lens, truth_meta.seq_lens)
    torch.testing.assert_close(meta.prefix_lens, truth_meta.prefix_lens)
    torch.testing.assert_close(meta.q_positions, truth_meta.q_positions)
    torch.testing.assert_close(out_loc, truth_out_loc)
    # Idempotent: a second invocation (the hook fires pre- and
    # post-correction each step) must be an identity.
    host.on_update_kv_lens()
    torch.testing.assert_close(meta.prefix_lens, truth_meta.prefix_lens)
    torch.testing.assert_close(out_loc, truth_out_loc)


@pytest.mark.skipif(not _has_cuda(), reason="device tensors required")
def test_on_update_kv_lens_identity_without_correction():
    device = torch.device("cuda")
    batch, max_kv_len = 2, 16
    req_to_token = (
        torch.arange(batch * max_kv_len, dtype=torch.int32).reshape(batch, max_kv_len).to(device)
    )
    prefix = [4, 9]
    ext = [3, 3]
    meta, out_loc = _build_extend_attachment(device, req_to_token, prefix, ext)
    before = (
        meta.seq_lens.clone(),
        meta.prefix_lens.clone(),
        meta.q_positions.clone(),
        out_loc.clone(),
    )
    kv_lens = [p + e for p, e in zip(prefix, ext)]  # uncorrected == optimistic
    host = _make_hook_host(device, meta, out_loc, kv_lens)
    host.on_update_kv_lens()
    torch.testing.assert_close(meta.seq_lens, before[0])
    torch.testing.assert_close(meta.prefix_lens, before[1])
    torch.testing.assert_close(meta.q_positions, before[2])
    torch.testing.assert_close(out_loc, before[3])


@pytest.mark.skipif(not _has_cuda(), reason="device tensors required")
def test_on_update_kv_lens_decode_branch():
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseAttentionMetadata,
    )

    device = torch.device("cuda")
    batch, max_kv_len = 2, 16
    req_to_token = (
        torch.arange(batch * max_kv_len, dtype=torch.int32).reshape(batch, max_kv_len).to(device)
    )
    optimistic = [10, 8]
    true_lens = [9, 6]
    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(optimistic, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(optimistic, dtype=torch.int32),
    )
    out_loc = torch.tensor(
        [int(req_to_token[b, optimistic[b] - 1].item()) for b in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    host = _make_hook_host(device, meta, out_loc, true_lens)
    host.on_update_kv_lens()
    expected = torch.tensor(
        [int(req_to_token[b, true_lens[b] - 1].item()) for b in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(
        meta.seq_lens, torch.tensor(true_lens, dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(out_loc, expected)


# prepare() derives max_seqlen_k host-side from the optimistic seq_lens_cpu.
# When a request's true length sits exactly at a page boundary, the
# optimistic (full-acceptance) length overhangs the last allocated page —
# e.g. a 384-token request in a 3-page x 128-slot table with one pending
# bonus token claims 385. The sparse kernels tolerate the overhang (they
# arange over max_seqlen_k and mask by the corrected seq_lens), but the
# dense fallback consumes max_seqlen_k as the exact SDPA mask/gather width:
# _gather_paged_batched silently clamps the gather to req_to_token's width
# while the mask keeps the optimistic width, and SDPA rejects the mismatch
# (found by TestMiniMaxM3::test_nvfp4_eagle3 on MMLU: mask 385 vs K 384).
# prepare() must clamp max_seqlen_k to the page table's width.


@pytest.mark.skipif(not _has_cuda(), reason="device tensors required")
def test_on_update_kv_lens_rederives_page_boundary_overhang():
    """Optimistic positions overhanging the page table must not crash the build.

    A request whose true length lands exactly on the table's last slot
    (page-aligned) plus a rejected draft token from the previous step gives
    an optimistic position one past the table. The eager device-side slot
    gather used to trip the CUDA scatter/gather assert on the last batch
    row (and silently read the next request's slots on inner rows); with
    the clamp it returns a placeholder, and the correction hook re-derives
    the true slots before any forward consumes them.
    """
    device = torch.device("cuda")
    batch, table_width, draft_w = 2, 12, 4
    req_to_token = (
        torch.randperm(batch * table_width, dtype=torch.int32)
        .reshape(batch, table_width)
        .to(device)
    )
    # Row 0: true final length 12 == table_width (page boundary), one
    # rejected token → optimistic length 13 overhangs the table by one.
    # Row 0 is intentionally NOT the last row: the un-clamped gather read
    # row 1's slots silently, so slot values (not just shapes) must match.
    true_prefix = [8, 3]
    rejected = [1, 0]
    optimistic_prefix = [p + r for p, r in zip(true_prefix, rejected)]
    ext = [draft_w] * batch

    meta, out_loc = _build_extend_attachment(device, req_to_token, optimistic_prefix, ext)
    meta.prepare()
    assert meta.max_seqlen_k == table_width  # clamped, not 13

    truth_meta, truth_out_loc = _build_extend_attachment(device, req_to_token, true_prefix, ext)
    corrected_kv_lens = [p + e for p, e in zip(true_prefix, ext)]
    host = _make_hook_host(device, meta, out_loc, corrected_kv_lens)
    host.on_update_kv_lens()

    torch.testing.assert_close(meta.seq_lens, truth_meta.seq_lens)
    torch.testing.assert_close(meta.prefix_lens, truth_meta.prefix_lens)
    torch.testing.assert_close(meta.q_positions, truth_meta.q_positions)
    torch.testing.assert_close(out_loc, truth_out_loc)


def test_derive_decode_cache_slots_clamps_overhang_and_empty_rows():
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        derive_decode_cache_slots,
    )

    batch, table_width = 3, 8
    req_to_token = torch.arange(batch * table_width, dtype=torch.int32).reshape(
        batch, table_width
    )
    # Row 0 overhangs (optimistic 9 > 8), row 1 in-bounds, row 2 is a
    # zero-length dummy row (would index -1 unclamped).
    seq_lens = torch.tensor([table_width + 1, 5, 0], dtype=torch.int32)
    slots = derive_decode_cache_slots(req_to_token, seq_lens)
    assert slots[0] == req_to_token[0, table_width - 1]  # clamped placeholder
    assert slots[1] == req_to_token[1, 4]
    assert slots[2] == req_to_token[2, 0]  # clamped from -1


def test_prepare_clamps_prefill_max_seqlen_k_to_page_table_width():
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseAttentionMetadata,
    )

    batch, table_width = 2, 12  # 3 pages x 4 slots
    req_to_token = torch.arange(batch * table_width, dtype=torch.int32).reshape(
        batch, table_width
    )
    # Row 0: true length 12 (exactly page-aligned), optimistic 13 overhangs.
    optimistic = [table_width + 1, 7]
    ext = [4, 3]
    prefix = [o - e for o, e in zip(optimistic, ext)]
    cu = [0, ext[0], ext[0] + ext[1]]
    total_q = cu[-1]
    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=True,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch, dtype=torch.int32),
        seq_lens=torch.tensor(optimistic, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(optimistic, dtype=torch.int32),
        prefix_lens=torch.tensor(prefix, dtype=torch.int32),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        extend_seq_lens_cpu=list(ext),
        q_batch_row=torch.empty(total_q, dtype=torch.int32),
        q_positions=torch.empty(total_q, dtype=torch.int32),
    )
    meta.prepare()
    assert meta.max_seqlen_k == table_width
    assert meta.max_seqlen_q == max(ext)


def test_prepare_clamps_decode_max_seqlen_k_to_page_table_width():
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseAttentionMetadata,
    )

    batch, table_width = 2, 8  # 2 pages x 4 slots
    req_to_token = torch.arange(batch * table_width, dtype=torch.int32).reshape(
        batch, table_width
    )
    optimistic = [table_width + 1, 5]
    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch, dtype=torch.int32),
        seq_lens=torch.tensor(optimistic, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(optimistic, dtype=torch.int32),
    )
    meta.prepare()
    assert meta.max_seqlen_k == table_width


def test_prepare_keeps_in_bounds_max_seqlen_k():
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseAttentionMetadata,
    )

    batch, table_width = 2, 8
    req_to_token = torch.arange(batch * table_width, dtype=torch.int32).reshape(
        batch, table_width
    )
    in_bounds = [6, 5]
    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=req_to_token,
        slot_ids=torch.arange(batch, dtype=torch.int32),
        seq_lens=torch.tensor(in_bounds, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(in_bounds, dtype=torch.int32),
    )
    meta.prepare()
    assert meta.max_seqlen_k == max(in_bounds)


# The extend/decode routing predicate in MiniMaxM3AttentionMetadata.prepare()
# is the line that makes one-model speculative decoding representable at all:
# spec verify presents pure-generation rows carrying 1 + draft_len tokens,
# which the decode attachment (hard-wired to one slot per row) cannot hold.
# Mutation testing showed no unit test pinned it (reverting to the pre-spec
# `num_contexts > 0` predicate passed the whole suite), so pin the routing
# decision itself: run prepare() with the TRTLLM base prepare stubbed and the
# M3 builder mocked, and assert which attachment kind prepare() requested.


def _prepare_routing_decision(new_tokens, num_cached, num_contexts):
    from unittest import mock

    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata as m3_metadata
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

    cls = m3_metadata.get_minimax_m3_attention_metadata_cls()
    obj = object.__new__(cls)
    obj.request_ids = list(range(len(new_tokens)))
    obj.seq_lens = torch.tensor(new_tokens, dtype=torch.int32)
    obj.seq_lens_cpu = torch.tensor(new_tokens, dtype=torch.int32)
    obj.num_contexts = num_contexts
    obj._m3_static_buffers = None
    obj.kv_cache_params = mock.Mock(num_cached_tokens_per_seq=num_cached)

    fake_manager = mock.Mock(spec=["get_index_k_buffer", "get_buffers"])
    fake_manager.get_buffers.return_value = torch.empty(1)  # cache_device = cpu
    obj.kv_cache_manager = fake_manager

    with (
        mock.patch.object(TrtllmAttentionMetadata, "prepare", lambda self: None),
        mock.patch.object(
            m3_metadata,
            "build_runtime_metadata_from_kv_manager",
            return_value=(mock.MagicMock(), mock.MagicMock()),
        ) as builder,
    ):
        obj.prepare()
    assert builder.call_count == 1
    return builder.call_args.kwargs["is_prefill"]


def test_prepare_routes_multitoken_gen_rows_to_extend():
    # Spec verify: pure-gen batch, every row carries 1 + draft_len tokens.
    assert _prepare_routing_decision([4, 4], [10, 20], num_contexts=0) is True
    # Mixed batch with one multi-token gen row (iter-131 class of bug).
    assert _prepare_routing_decision([5, 1], [0, 20], num_contexts=1) is True


def test_prepare_routes_pure_single_token_decode_to_decode():
    # Non-speculative decode keeps the one-slot-per-row perf specialization
    # (and with it the CUDA-graph warmup geometry).
    assert _prepare_routing_decision([1, 1], [10, 20], num_contexts=0) is False
