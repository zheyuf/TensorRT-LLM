# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the MiniMax-M3 MSA multi-token (Eagle3 verify)
glue: optimistic metadata builder -> overlap-correction hook -> ladder
KV/idx_k writes -> indexer decode -> sparse decode driver, checked
against fp32 first-principles references computed straight from the
cache pools + block tables.

The driver suite (test_minimax_m3_decode_driver_vs_msa.py) bit-diffs
the kernels at full-model geometry (64/4/4 heads, pf_proxy=4); this
file runs the glue at the deployed TP4 per-rank shape (16/1/1 heads,
pf_proxy=1), which that suite never exercises.

Requires SM100 + the `fmha_sm100` package.
"""

import math
from types import SimpleNamespace

import pytest
import torch

# Deployed TP4 per-rank M3 geometry.
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
NUM_INDEX_HEADS = 1
HEAD_DIM = 128
PAGE = 128
TOPK = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
QO_LEN = 4  # 1 + max_draft_len(3), the Eagle3 verify-row width

SM_SCALE = HEAD_DIM**-0.5
IDX_SM_SCALE = HEAD_DIM**-0.5
DEV = torch.device("cuda")


def _require_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("SM100 (Blackwell) required")
    try:
        import fmha_sm100  # noqa: F401
    except ImportError:
        pytest.skip("fmha_sm100 (MSA) not importable")


def _m3():
    from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import run_msa_sparse_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import metadata as metadata_mod
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import (
        write_main_kv_slots,
        write_msa_main_kv,
    )
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.decode_wrapper.dispatch import (
        decode_proxy_max_score,
    )
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.indexer import MsaIndexer
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import (
        MiniMaxM3SparseConfig,
        build_m3_sparse_metadata_and_plans,
    )
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        _whole_batch_lens,
        rederive_msa_attachment,
        run_msa_sparse_decode,
    )

    return SimpleNamespace(**locals())


def _config(m):
    return m.MiniMaxM3SparseConfig(
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_index_heads=NUM_INDEX_HEADS,
        sparse_index_dim=HEAD_DIM,
        block_size=PAGE,
        topk=TOPK,
        init_blocks=INIT_BLOCKS,
        local_blocks=LOCAL_BLOCKS,
    )


class FakeKvParams:

    def __init__(self, cached):
        self.num_cached_tokens_per_seq = cached


class FakeManager:
    """Duck-typed KV cache manager: shuffled non-contiguous block tables,
    randomly initialized pools (so any read of a never-written slot
    diverges from the position-addressed references)."""

    use_msa = True
    tokens_per_block = PAGE

    def __init__(self, num_pages, block_table):
        self.block_table = block_table
        r = _randn(99)
        # main pool [pages, 2, page, kvh, dim]; index pool [pages, page, 1, dim]
        self.pool = r(num_pages, 2, PAGE, NUM_KV_HEADS, HEAD_DIM)
        self.idx_pool = r(num_pages, PAGE, 1, HEAD_DIM)

    def get_buffers(self, layer_idx):
        return self.pool

    def get_index_k_buffer(self, layer_idx):
        return self.idx_pool

    def get_block_ids_per_seq(self, request_ids):
        rows = [self.block_table[r] for r in request_ids]
        width = max(len(x) for x in rows)
        out = torch.zeros(len(rows), width, dtype=torch.int32)
        for i, x in enumerate(rows):
            out[i, : len(x)] = torch.tensor(x, dtype=torch.int32)
        return out.to(DEV)


class FakeMeta:
    """Duck-typed attention metadata carrying exactly the fields
    `build_m3_sparse_metadata_and_plans` / the re-derivation hooks read."""

    is_cuda_graph = False

    def __init__(self, manager, request_ids, seq_lens_cpu, cached_opt):
        self.kv_cache_manager = manager
        self.request_ids = request_ids
        self.seq_lens = seq_lens_cpu.clone()
        self.seq_lens_cpu = seq_lens_cpu.clone()
        self.num_contexts = 0
        self.kv_cache_params = FakeKvParams(cached_opt)
        self.max_num_sequences = 16
        self.max_total_draft_tokens = QO_LEN - 1
        self.m3_sparse_metadata = None
        self.m3_out_cache_loc = None
        self._msa_kv_indices_buf = None
        self._msa_kv_page_indptr_buf = None
        self.kv_lens_cuda = torch.zeros(16, dtype=torch.int32, device=DEV)


def flat_slot(block_ids, pos):
    return block_ids[pos // PAGE] * PAGE + pos % PAGE


def _alloc_block_tables(pages_needed, spare=4):
    total_pages = sum(pages_needed) + spare
    perm = torch.randperm(total_pages).tolist()
    block_table, ofs = {}, 0
    for b, n in enumerate(pages_needed):
        block_table[b] = perm[ofs : ofs + n]
        ofs += n
    return block_table, total_pages


def _randn(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return (torch.randn(*shape, generator=gen, device=DEV, dtype=torch.float32) * 0.5).to(
            torch.bfloat16
        )

    return r


def _pool_rows(manager, block_table, b, eff, which):
    """fp32 rows for positions 0..eff-1 of request b ('k'/'v'/'i')."""
    pages = block_table[b]
    chunks, p = [], 0
    while p * PAGE < eff:
        n = min(PAGE, eff - p * PAGE)
        if which == "k":
            chunks.append(manager.pool[pages[p], 0, :n, 0])
        elif which == "v":
            chunks.append(manager.pool[pages[p], 1, :n, 0])
        else:
            chunks.append(manager.idx_pool[pages[p], :n, 0])
        p += 1
    return torch.cat(chunks, 0).float()


def _sparse_ref(q_tok, k_rows, v_rows, blocks_row, eff):
    """fp32 attention over the selected blocks clipped to the causal-ladder
    bound `eff`; None if no block intersects the bound."""
    blks = [int(x) for x in blocks_row.tolist() if x >= 0]
    chunks = [
        torch.arange(bk * PAGE, min((bk + 1) * PAGE, eff)) for bk in blks if bk * PAGE < eff
    ]
    if not chunks:
        return None
    pos = torch.cat(chunks).to(q_tok.device)
    scores = q_tok @ k_rows[pos].T * SM_SCALE
    return torch.softmax(scores, -1) @ v_rows[pos]


def _staged_verify_batch(m, config):
    """Optimistic builder + corrected-hook staging for the shared
    adversarial verify batch: page-boundary-crossing true lens with
    per-row pending rejections (optimistic cached = true + rejected)."""
    torch.manual_seed(5)
    true_cached = [122, 125, 252, 124, 33, 380]
    rejected = [3, 3, 3, 3, 0, 3]
    B = len(true_cached)
    block_table, total_pages = _alloc_block_tables(
        [(true_cached[b] + rejected[b] + QO_LEN + PAGE - 1) // PAGE for b in range(B)], spare=8
    )
    manager = FakeManager(total_pages, block_table)
    seq_lens_cpu = torch.full((B,), QO_LEN, dtype=torch.int32)
    cached_opt = [true_cached[b] + rejected[b] for b in range(B)]
    meta = FakeMeta(manager, list(range(B)), seq_lens_cpu, cached_opt)
    m3_meta = m.build_m3_sparse_metadata_and_plans(meta, geometry=config)
    corrected = torch.tensor([c + QO_LEN for c in true_cached], dtype=torch.int32, device=DEV)
    meta.kv_lens_cuda[:B] = corrected
    m.rederive_msa_attachment(meta)
    torch.cuda.synchronize()
    return manager, meta, m3_meta, block_table, true_cached, corrected


def test_verify_step_glue_at_page_boundaries():
    """One verify step: builder staged with OPTIMISTIC cached counts (as
    the overlap scheduler's prepare() sees them), then the real hook
    re-derivation with corrected kv lens; ladder slots, indexer block
    selection, and sparse output all checked against fp32 ground truth
    read back from the pools."""
    _require_env()
    m = _m3()
    config = _config(m)
    manager, meta, m3_meta, block_table, true_cached, corrected = _staged_verify_batch(m, config)
    B = len(true_cached)
    assert not m3_meta.is_prefill
    assert m3_meta.decode_qo_len == QO_LEN
    assert torch.equal(m3_meta.seq_lens[:B].cpu(), corrected.cpu())

    # re-derived ladder slots vs block-table ground truth
    ocl = meta.m3_out_cache_loc.cpu().tolist()
    for b in range(B):
        for t in range(QO_LEN):
            pos = true_cached[b] + t
            assert ocl[b * QO_LEN + t] == flat_slot(block_table[b], pos), f"slot b={b} t={t}"

    # simulate the layer: new-token tensors in engine order
    r = _randn(7)
    total_q = B * QO_LEN
    q3 = r(total_q, NUM_Q_HEADS, HEAD_DIM)
    k_new = r(total_q, NUM_KV_HEADS, HEAD_DIM)
    v_new = r(total_q, NUM_KV_HEADS, HEAD_DIM)
    idx_q = r(total_q, NUM_INDEX_HEADS, HEAD_DIM)
    idx_k_new = r(total_q, 1, HEAD_DIM)

    m.write_main_kv_slots(manager.get_index_k_buffer(0), meta.m3_out_cache_loc, idx_k_new)
    indexer = m.MsaIndexer(config)
    blocks = indexer.select_blocks_decode(
        idx_q, manager.get_index_k_buffer(0), m3_meta, idx_sm_scale=IDX_SM_SCALE, page_size=PAGE
    ).clone()
    m.write_msa_main_kv(
        manager, 0, meta.m3_out_cache_loc, k_new.reshape(total_q, -1), v_new.reshape(total_q, -1)
    )
    out = m.run_msa_sparse_decode(config, manager, 0, m3_meta, q3, blocks, SM_SCALE).clone()
    torch.cuda.synchronize()

    # proxy max scores from the staged plan: semantically the fp32
    # per-block max times one CONSTANT scale (constancy is what per-token
    # top-k ordering needs). This is the guard that scores are computed
    # from the right positions — the block reference below is derived
    # from these same kernel scores, so it can't catch wrong-page reads.
    ms_kernel = m.decode_proxy_max_score(
        m3_meta.decode_state,
        idx_q,
        manager.idx_pool.permute(0, 2, 1, 3).contiguous(),
        seq_lens=m3_meta.seq_lens.to(torch.int32),
        kv_page_indptr=m3_meta.msa_kv_page_indptr,
        kv_indices=m3_meta.msa_kv_indices,
        sm_scale=IDX_SM_SCALE,
        qo_len=QO_LEN,
    ).clone()
    torch.cuda.synchronize()
    scale_seen, max_rel_dev = None, 0.0
    for b in range(B):
        for t in range(QO_LEN):
            tok = b * QO_LEN + t
            eff = true_cached[b] + t + 1
            scores = idx_q[tok, 0].float() @ _pool_rows(manager, block_table, b, eff, "i").T
            for blk in range((eff + PAGE - 1) // PAGE):
                ref = scores[blk * PAGE : min((blk + 1) * PAGE, eff)].max()
                if abs(ref) > 0.5:
                    ratio = (ms_kernel[0, blk, tok] / ref).item()
                    if scale_seen is None:
                        scale_seen = ratio
                    max_rel_dev = max(max_rel_dev, abs(ratio - scale_seen))
    assert scale_seen is not None and max_rel_dev < 0.05, (scale_seen, max_rel_dev)

    # block selection vs python per-token ladder top-k over kernel scores
    blocks_ref = torch.full((total_q, NUM_KV_HEADS, TOPK), -1, dtype=torch.int32)
    ms_cpu = ms_kernel.float().cpu().numpy()
    for b in range(B):
        for t in range(QO_LEN):
            tok = b * QO_LEN + t
            eff = true_cached[b] + t + 1
            valid = (eff + PAGE - 1) // PAGE
            row = ms_cpu[0, :, tok].copy()
            row[valid:] = -math.inf
            for k in range(max(valid - LOCAL_BLOCKS, 0), valid):
                row[k] = math.inf
            order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:TOPK]
            picked = sorted(i for i in order if row[i] != -math.inf)
            for j, blk in enumerate(picked):
                blocks_ref[tok, 0, j] = blk
    assert torch.equal(blocks_ref, blocks.cpu())

    # sparse output vs fp32 over the selected blocks with the ladder bound
    out3 = out.view(total_q, NUM_Q_HEADS, HEAD_DIM).float()
    for b in range(B):
        for t in range(QO_LEN):
            tok = b * QO_LEN + t
            eff = true_cached[b] + t + 1
            ref = _sparse_ref(
                q3[tok].float(),
                _pool_rows(manager, block_table, b, eff, "k"),
                _pool_rows(manager, block_table, b, eff, "v"),
                blocks.cpu()[tok, 0],
                eff,
            )
            d = (out3[tok] - ref).abs().max().item()
            assert d < 0.06, f"sparse mismatch tok={tok} (b={b},t={t}): {d}"

    # K landed at the corrected slots (not the optimistic ones)
    for b in range(B):
        for t in range(QO_LEN):
            s = flat_slot(block_table[b], true_cached[b] + t)
            assert torch.equal(
                manager.pool[s // PAGE, 0, s % PAGE, 0], k_new[b * QO_LEN + t, 0]
            ), f"K write b={b} t={t}"


def test_multistep_verify_no_rot():
    """Evolve two requests across 30 consecutive verify steps (optimistic
    staging -> correction with a rejection schedule -> writes -> indexer
    -> sparse decode), comparing each step against a position-addressed
    fp32 ground truth. Cross-step corruption (wrong slots, stale plans,
    wrong ladder) shows up as a step-N divergence. Crosses several page
    boundaries (lens ~100 -> ~200)."""
    _require_env()
    m = _m3()
    torch.manual_seed(11)
    r = _randn(3)
    config = _config(m)
    STEPS = 30

    B = 2
    max_len = 100 + STEPS * QO_LEN + PAGE
    pages_per_req = (max_len + PAGE - 1) // PAGE
    block_table, total_pages = _alloc_block_tables([pages_per_req] * B)
    manager = FakeManager(total_pages, block_table)
    indexer = m.MsaIndexer(config)

    # position-addressed ground truth (rejected tokens' KV stays in the
    # pool until overwritten by the next step's write at those positions)
    true_len = [100, 97]
    GT_K = torch.zeros(B, max_len, HEAD_DIM, device=DEV, dtype=torch.float32)
    GT_V = torch.zeros(B, max_len, HEAD_DIM, device=DEV, dtype=torch.float32)
    for b in range(B):
        n = true_len[b]
        hk, hv, hi = r(n, HEAD_DIM), r(n, HEAD_DIM), r(n, HEAD_DIM)
        GT_K[b, :n] = hk.float()
        GT_V[b, :n] = hv.float()
        for p in range(0, n, PAGE):
            e = min(p + PAGE, n)
            pg = block_table[b][p // PAGE]
            manager.pool[pg, 0, : e - p, 0] = hk[p:e]
            manager.pool[pg, 1, : e - p, 0] = hv[p:e]
            manager.idx_pool[pg, : e - p, 0] = hi[p:e]

    # rejection schedule: rej_seq[step] = tokens of THIS step later rejected
    rej_seq = [0, 3, 1, 2, 0, 0, 3, 2, 1, 0, 2, 3, 0, 1, 0] * 3
    pending_rej = [0, 0]

    for step in range(STEPS):
        # optimistic staging: the previous step's rejected tokens still
        # counted as cached (overlap prepares before the truth is known)
        cached_opt = [true_len[b] + pending_rej[b] for b in range(B)]
        seq_lens_cpu = torch.full((B,), QO_LEN, dtype=torch.int32)
        meta = FakeMeta(manager, list(range(B)), seq_lens_cpu, cached_opt)
        m3_meta = m.build_m3_sparse_metadata_and_plans(meta, geometry=config)
        assert m3_meta.decode_qo_len == QO_LEN

        corrected = torch.tensor(
            [true_len[b] + QO_LEN for b in range(B)], dtype=torch.int32, device=DEV
        )
        meta.kv_lens_cuda[:B] = corrected
        m.rederive_msa_attachment(meta)

        total_q = B * QO_LEN
        q3 = r(total_q, NUM_Q_HEADS, HEAD_DIM)
        k_new, v_new = r(total_q, 1, HEAD_DIM), r(total_q, 1, HEAD_DIM)
        idx_k_new = r(total_q, 1, HEAD_DIM)
        idx_q = r(total_q, NUM_INDEX_HEADS, HEAD_DIM)

        m.write_main_kv_slots(manager.get_index_k_buffer(0), meta.m3_out_cache_loc, idx_k_new)
        blocks = indexer.select_blocks_decode(
            idx_q,
            manager.get_index_k_buffer(0),
            m3_meta,
            idx_sm_scale=IDX_SM_SCALE,
            page_size=PAGE,
        ).clone()
        m.write_msa_main_kv(
            manager, 0, meta.m3_out_cache_loc, k_new.reshape(total_q, -1),
            v_new.reshape(total_q, -1)
        )
        out = m.run_msa_sparse_decode(config, manager, 0, m3_meta, q3, blocks, SM_SCALE).clone()
        torch.cuda.synchronize()

        for b in range(B):
            s = b * QO_LEN
            GT_K[b, true_len[b] : true_len[b] + QO_LEN] = k_new[s : s + QO_LEN, 0].float()
            GT_V[b, true_len[b] : true_len[b] + QO_LEN] = v_new[s : s + QO_LEN, 0].float()

        for b in range(B):
            for t in range(QO_LEN):
                tok = b * QO_LEN + t
                eff = true_len[b] + t + 1
                ref = _sparse_ref(q3[tok].float(), GT_K[b], GT_V[b], blocks[tok, 0], eff)
                err = (out[tok].view(NUM_Q_HEADS, HEAD_DIM).float() - ref).abs().max().item()
                assert err < 0.05, f"rot at step={step} b={b} t={t}: {err}"

        this_rej = [rej_seq[step], rej_seq[(step + 5) % len(rej_seq)]]
        for b in range(B):
            true_len[b] += QO_LEN - this_rej[b]
        pending_rej = this_rej


def test_mixed_batch_correction_restages_page_table():
    """Mixed ctx+gen (eager extend) batch where the correction shrinks a
    gen row across a page boundary (optimistic 129 kv -> 2 pages,
    corrected 126 -> 1 page). The eager fmha plan rebuilds its indptr
    from the CORRECTED lens, so the staged flat page table must be
    re-staged to the same layout — a stale optimistic layout misbases
    every row after the shrink (the victim row here read wrong pages,
    max err 0.137, before the hook re-staged the table)."""
    _require_env()
    m = _m3()
    torch.manual_seed(21)
    config = _config(m)

    # Row 0: fresh context (300 tokens, no correction).
    # Row 1 (gen): true_cached=122, rejected=3 -> page-count SHRINK.
    # Row 2 (gen): true_cached=200, rejected=1 -> the victim after the shrink.
    ctx_len = 300
    gen_true_cached = [122, 200]
    gen_rejected = [3, 1]
    B = 3
    qo_lens = [ctx_len, QO_LEN, QO_LEN]
    cached_opt = [0] + [gen_true_cached[i] + gen_rejected[i] for i in range(2)]
    kv_true = [ctx_len] + [gen_true_cached[i] + QO_LEN for i in range(2)]

    block_table, total_pages = _alloc_block_tables(
        [(cached_opt[b] + qo_lens[b] + PAGE - 1) // PAGE for b in range(B)]
    )
    manager = FakeManager(total_pages, block_table)
    seq_lens_cpu = torch.tensor(qo_lens, dtype=torch.int32)
    meta = FakeMeta(manager, list(range(B)), seq_lens_cpu, cached_opt)
    meta.num_contexts = 1  # mixed batch -> extend path
    m3_meta = m.build_m3_sparse_metadata_and_plans(meta, geometry=config)
    assert m3_meta.is_prefill, "mixed batch must route to the extend path"

    # the real hook's re-stage only runs with a registered geometry
    saved = m.metadata_mod._GLOBAL_MSA_GEOMETRY
    m.metadata_mod._GLOBAL_MSA_GEOMETRY = config
    try:
        corrected = torch.tensor(kv_true, dtype=torch.int32, device=DEV)
        meta.kv_lens_cuda[:B] = corrected
        m.rederive_msa_attachment(meta)
        torch.cuda.synchronize()
    finally:
        m.metadata_mod._GLOBAL_MSA_GEOMETRY = saved

    # structural pin: staged indptr matches the corrected-lens layout
    corr_indptr = [0]
    for kv in kv_true:
        corr_indptr.append(corr_indptr[-1] + (kv + PAGE - 1) // PAGE)
    assert m3_meta.msa_kv_page_indptr.cpu().tolist() == corr_indptr
    assert m3_meta.msa_kv_lens_cpu.tolist() == kv_true

    # run the full eager extend path and check every row against fp32
    r = _randn(4)
    total_q = sum(qo_lens)
    q3 = r(total_q, NUM_Q_HEADS, HEAD_DIM)
    k_new = r(total_q, NUM_KV_HEADS, HEAD_DIM)
    v_new = r(total_q, NUM_KV_HEADS, HEAD_DIM)
    idx_q = r(total_q, 1, HEAD_DIM)
    idx_k_new = r(total_q, 1, HEAD_DIM)

    m.write_main_kv_slots(manager.get_index_k_buffer(0), meta.m3_out_cache_loc, idx_k_new)
    indexer = m.MsaIndexer(config)
    qo, kv, qo_off, kv_indices = m._whole_batch_lens(m3_meta, PAGE)
    blocks = indexer.select_blocks_prefill(
        idx_q,
        manager.get_index_k_buffer(0),
        m3_meta,
        idx_sm_scale=IDX_SM_SCALE,
        qo_lens_cpu=qo,
        kv_lens_cpu=kv,
        qo_offset_cpu=qo_off,
        kv_indices=kv_indices,
    ).clone()
    m.write_msa_main_kv(
        manager, 0, meta.m3_out_cache_loc, k_new.reshape(total_q, -1), v_new.reshape(total_q, -1)
    )
    k_paged = manager.pool[:, 0].permute(0, 2, 1, 3).contiguous()
    v_paged = manager.pool[:, 1].permute(0, 2, 1, 3).contiguous()
    out = m.run_msa_sparse_gqa(
        q3,
        k_paged,
        v_paged,
        blocks,
        qo_lens_cpu=qo,
        kv_lens_cpu=kv,
        qo_offset_cpu=qo_off,
        kv_indices=kv_indices,
        sm_scale=SM_SCALE,
        causal=True,
    )
    torch.cuda.synchronize()

    out3 = out.view(total_q, NUM_Q_HEADS, HEAD_DIM).float()
    row_start = [0, ctx_len, ctx_len + QO_LEN]
    prefix_true = [0] + gen_true_cached
    for b in range(B):
        for t in range(qo_lens[b]):
            tok = row_start[b] + t
            eff = prefix_true[b] + t + 1
            ref = _sparse_ref(
                q3[tok].float(),
                _pool_rows(manager, block_table, b, eff, "k"),
                _pool_rows(manager, block_table, b, eff, "v"),
                blocks.cpu()[tok, 0],
                eff,
            )
            if ref is None:
                continue
            d = (out3[tok] - ref).abs().max().item()
            assert d < 0.06, f"row {b} tok {t}: max abs err {d}"


def test_dense_decode_ladder():
    """`MiniMaxM3Attention._dense_attention_core` DECODE branch (layers
    0-2 causal-ladder SDPA) for a verify step, vs an fp32 per-token
    reference, using the real metadata builder + hook."""
    _require_env()
    m = _m3()
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention

    config = _config(m)
    manager, meta, _, block_table, true_cached, _ = _staged_verify_batch(m, config)
    B = len(true_cached)

    r = _randn(3)
    total_q = B * QO_LEN
    q = r(total_q, NUM_Q_HEADS * HEAD_DIM)
    k = r(total_q, NUM_KV_HEADS * HEAD_DIM)
    v = r(total_q, NUM_KV_HEADS * HEAD_DIM)
    output = torch.zeros(total_q, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=DEV)

    fake_self = SimpleNamespace(
        num_heads=NUM_Q_HEADS,
        head_dim=HEAD_DIM,
        num_key_value_heads=NUM_KV_HEADS,
        layer_idx=0,
    )
    MiniMaxM3Attention._dense_attention_core(fake_self, q, k, v, meta, output)
    torch.cuda.synchronize()

    # fp32 reference from the cache pools (the method wrote K/V itself);
    # dense = all pages selected up to the ladder bound.
    out3 = output.view(total_q, NUM_Q_HEADS, HEAD_DIM).float()
    for b in range(B):
        for t in range(QO_LEN):
            tok = b * QO_LEN + t
            eff = true_cached[b] + t + 1
            all_blocks = torch.arange((eff + PAGE - 1) // PAGE)
            ref = _sparse_ref(
                q.view(total_q, NUM_Q_HEADS, HEAD_DIM)[tok].float(),
                _pool_rows(manager, block_table, b, eff, "k"),
                _pool_rows(manager, block_table, b, eff, "v"),
                all_blocks,
                eff,
            )
            d = (out3[tok] - ref).abs().max().item()
            assert d < 0.06, f"dense decode mismatch tok={tok} (b={b},t={t}): {d}"
    # the K written by the method landed at the corrected ladder slots
    for b in range(B):
        for t in range(QO_LEN):
            s = flat_slot(block_table[b], true_cached[b] + t)
            got = manager.pool[s // PAGE, 0, s % PAGE, 0]
            want = k.view(total_q, NUM_KV_HEADS, HEAD_DIM)[b * QO_LEN + t, 0]
            assert torch.equal(got, want), f"dense K write mismatch b={b} t={t}"
