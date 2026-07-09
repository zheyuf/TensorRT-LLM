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
"""Phase-2 tests for structured NHD bounce staging (TRTLLM_KV_BOUNCE_STRUCTURED_NHD).

CPU-testable parts (no GPU): the config flag and its env plumbing, the structured
result-tail encode/decode wire format, the receiver-side scatter template slicing, the
eligibility/fallback decision helpers, the structured-tail -> scatter-descriptor
resolution (including the min_blocks bypass and rejection of mismatched tails), and the
fallback materialization of structured sections.

GPU e2e (marked cuda): the threaded-NIXL MiniMax-M3 harness with a head-mismatched
topology, run with the flag on and off; the exact-equality cache verification makes the
flag-on result byte-identical to the flag-off path, and monkeypatched counters assert
the structured gather/scatter actually ran — including the pure-TP fan-in +
replicated-INDEX_KEY mix, which takes split routing (NHD sections through the slot,
INDEX_KEY via direct descriptors on the same write).
"""

import queue
from types import SimpleNamespace

import numpy as np
import pytest

# The bounce package init pulls CUDA-binding modules; skip gracefully when those are
# absent (CPU-only env without cuda-python). Catch only ImportError so a genuine bug in
# the module still fails CI instead of being silently turned into a skip.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg
    from tensorrt_llm._torch.disaggregation.native.bounce import core as bcore
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr
    from tensorrt_llm._torch.disaggregation.native.bounce.nhd_plan import (
        NHDGatherSpec,
        NHDResultTail,
        NHDScatterTemplate,
        decode_nhd_tail,
        encode_nhd_tail,
        is_nhd_tail,
    )

    _HAVE_BOUNCE = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_BOUNCE = False

pytestmark = pytest.mark.skipif(not _HAVE_BOUNCE, reason="bounce import needs cuda-python")

_EMPTY = np.zeros(0, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Config flag — env plumbing layered on the kv_cache_bounce_size_mb opt-in
# --------------------------------------------------------------------------- #
class TestStructuredFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(bcfg.STRUCTURED_NHD_ENV, raising=False)
        assert bcfg.structured_nhd_from_env() is False
        assert bcfg.Config().structured_nhd is False
        cfg = bcfg.config_from_size(256)
        assert cfg is not None and cfg.structured_nhd is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_env_enables(self, monkeypatch, value):
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, value)
        assert bcfg.structured_nhd_from_env() is True
        cfg = bcfg.config_from_size(256)
        assert cfg is not None and cfg.structured_nhd is True

    @pytest.mark.parametrize("value", ["0", "false", "off", ""])
    def test_env_disables(self, monkeypatch, value):
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, value)
        assert bcfg.structured_nhd_from_env() is False

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, "1")
        cfg = bcfg.config_from_size(256, structured_nhd=False)
        assert cfg is not None and cfg.structured_nhd is False
        monkeypatch.delenv(bcfg.STRUCTURED_NHD_ENV, raising=False)
        cfg = bcfg.config_from_size(256, structured_nhd=True)
        assert cfg is not None and cfg.structured_nhd is True

    def test_size_stays_the_master_switch(self, monkeypatch):
        # flag alone must not enable bounce: the size knob is the opt-in it layers on.
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, "1")
        assert bcfg.config_from_size(0) is None
        assert bcfg.config_from_size(None) is None

    def test_transport_carries_flag(self):
        assert bcore.BounceTransport.structured_nhd is False
        assert btr.NoBounceTransport().structured_nhd is False


# --------------------------------------------------------------------------- #
# Capacity sizing — TokenBudgetSizing (C++ computeTransferBufferSize analog) and
# the single-knob auto-enable (flag alone must not silently do nothing)
# --------------------------------------------------------------------------- #
def _fake_page_table(nhd_region_bytes=(4096, 2048), other_region_bytes=(1024,), tpb=128):
    """Build a minimal page table for token-budget sizing.

    Uses real AttentionLayerGroup instances
    (isinstance-checked) carrying PoolViews of mixed mapper kinds.
    """
    from tensorrt_llm._torch.disaggregation.resource.page import (
        AttentionLayerGroup,
        MapperKind,
        PoolView,
    )

    views = [
        PoolView(
            pool_idx=i,
            buffer_entries=np.zeros(0),
            mapper_kind=MapperKind.NHD,
            bytes_per_region=b,
        )
        for i, b in enumerate(nhd_region_bytes)
    ] + [
        PoolView(
            pool_idx=len(nhd_region_bytes) + i,
            buffer_entries=np.zeros(0),
            mapper_kind=MapperKind.REPLICATED,
            bytes_per_region=b,
        )
        for i, b in enumerate(other_region_bytes)
    ]
    lg = AttentionLayerGroup(pool_group_idx=0, pool_views=views)
    return SimpleNamespace(tokens_per_block=tpb, layer_groups=[lg])


class TestTokenBudgetSizing:
    def _ctx(self, page_table, chunk_mb=32):
        return bcfg.SizingContext(
            free_bytes=1 << 34,
            total_bytes=1 << 35,
            chunk_bytes=chunk_mb * 1024 * 1024,
            device_id=0,
            page_table=page_table,
        )

    def test_token_budget_bytes_counts_bounded_mapped_pools(self):
        pt = _fake_page_table(nhd_region_bytes=(4096, 2048), other_region_bytes=(1024,), tpb=128)
        # ceil(512/128) + 1 draft block = 5 blocks. Single-writer staging may
        # coalesce the bounded replicated view, so auto-sizing must include it.
        per_block = 4096 + 2048 + 1024
        assert bcfg.token_budget_bytes(pt, 512) == 5 * per_block
        # non-block-aligned budget rounds up: ceil(129/128) + 1 = 3
        assert bcfg.token_budget_bytes(pt, 129) == 3 * per_block

    def test_token_budget_bytes_degrades_to_none(self):
        assert bcfg.token_budget_bytes(None, 512) is None
        assert bcfg.token_budget_bytes(_fake_page_table(), None) is None
        assert bcfg.token_budget_bytes(_fake_page_table(), 0) is None
        no_nhd = _fake_page_table(nhd_region_bytes=(), other_region_bytes=(1024,))
        assert bcfg.token_budget_bytes(no_nhd, 512) is None

    def test_resolve_rounds_to_chunk(self):
        pt = _fake_page_table()
        got = bcfg.TokenBudgetSizing(max_tokens=512).resolve(self._ctx(pt))
        chunk = 32 * 1024 * 1024
        assert got == chunk  # 5 * 6144 rounds up to one chunk
        big = bcfg.TokenBudgetSizing(max_tokens=3_000_000).resolve(self._ctx(pt))
        assert big % chunk == 0
        assert big >= bcfg.token_budget_bytes(pt, 3_000_000)

    def test_resolve_falls_back_to_fixed_default(self):
        # no page table / no NHD pools: never crash, resolve like FixedSizing (the
        # codex reference raised here — graceful fallback is the required behavior)
        ctx = self._ctx(None)
        fixed = bcfg.FixedSizing().resolve(ctx)
        assert bcfg.TokenBudgetSizing(max_tokens=512).resolve(ctx) == fixed


class TestResolveBounceConfig:
    """The four knob combinations of kv_cache_bounce_size_mb x TRTLLM_KV_BOUNCE_STRUCTURED_NHD."""

    def test_neither_knob_keeps_bounce_off(self, monkeypatch):
        monkeypatch.delenv(bcfg.STRUCTURED_NHD_ENV, raising=False)
        assert bcfg.resolve_bounce_config(0, max_tokens_in_buffer=512) is None
        assert bcfg.resolve_bounce_config(None, max_tokens_in_buffer=None) is None

    def test_size_alone_gives_fixed_sizing(self, monkeypatch):
        monkeypatch.delenv(bcfg.STRUCTURED_NHD_ENV, raising=False)
        cfg = bcfg.resolve_bounce_config(256, max_tokens_in_buffer=512)
        assert cfg is not None and cfg.structured_nhd is False
        assert cfg.sizing == bcfg.FixedSizing(capacity_mb=256)

    def test_both_knobs_size_wins(self, monkeypatch):
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, "1")
        cfg = bcfg.resolve_bounce_config(256, max_tokens_in_buffer=512)
        assert cfg is not None and cfg.structured_nhd is True
        assert cfg.sizing == bcfg.FixedSizing(capacity_mb=256)

    def test_flag_alone_auto_enables_with_token_budget(self, monkeypatch):
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, "1")
        cfg = bcfg.resolve_bounce_config(0, max_tokens_in_buffer=512)
        assert cfg is not None and cfg.structured_nhd is True
        assert cfg.sizing == bcfg.TokenBudgetSizing(max_tokens=512)

    def test_flag_alone_without_token_budget_falls_back_to_fixed(self, monkeypatch):
        # max_tokens_in_buffer is None: must NOT crash (codex bug), fall back to fixed
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, "1")
        for budget in (None, 0):
            cfg = bcfg.resolve_bounce_config(None, max_tokens_in_buffer=budget)
            assert cfg is not None and cfg.structured_nhd is True
            assert cfg.sizing == bcfg.FixedSizing()

    def test_explicit_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv(bcfg.STRUCTURED_NHD_ENV, "1")
        assert bcfg.resolve_bounce_config(0, structured_nhd=False) is None
        monkeypatch.delenv(bcfg.STRUCTURED_NHD_ENV, raising=False)
        cfg = bcfg.resolve_bounce_config(0, max_tokens_in_buffer=64, structured_nhd=True)
        assert cfg is not None and cfg.sizing == bcfg.TokenBudgetSizing(max_tokens=64)


# --------------------------------------------------------------------------- #
# Structured result tail — wire format encode/decode
# --------------------------------------------------------------------------- #
class TestNHDTailCodec:
    def _tail(self, records=((3, 1, 2, 5),), rest_n=4, src_base=0x7000):
        rng = np.random.default_rng(0)
        rest = rng.integers(1, 1 << 40, size=rest_n).astype(np.int64)
        sizes = np.full(rest_n, 512, dtype=np.int64)
        return NHDResultTail(
            src_base=src_base, records=tuple(records), rest_dst_ptrs=rest, rest_sizes=sizes
        )

    def test_roundtrip(self):
        tail = self._tail(records=((0, 0, 0, 64), (3, 1, 2, 62)), rest_n=5)
        blob = encode_nhd_tail(tail)
        assert is_nhd_tail(blob)
        out = decode_nhd_tail(blob)
        assert out.src_base == tail.src_base
        assert out.records == tail.records
        assert np.array_equal(out.rest_dst_ptrs, tail.rest_dst_ptrs)
        assert np.array_equal(out.rest_sizes, tail.rest_sizes)

    def test_roundtrip_empty_rest_and_records(self):
        tail = NHDResultTail(src_base=0, records=(), rest_dst_ptrs=_EMPTY, rest_sizes=_EMPTY)
        out = decode_nhd_tail(encode_nhd_tail(tail))
        assert out.records == () and out.rest_dst_ptrs.size == 0

    def test_tail_is_one_small_frame(self):
        # the point of the structured tail: dozens of bytes, not a 16 MB dst table.
        blob = encode_nhd_tail(self._tail(rest_n=0))
        assert len(blob) < 100

    def test_bad_magic_rejected(self):
        blob = bytearray(encode_nhd_tail(self._tail()))
        blob[0] ^= 0xFF
        with pytest.raises(ValueError, match="magic"):
            decode_nhd_tail(bytes(blob))
        assert not is_nhd_tail(bytes(blob))

    def test_truncation_rejected(self):
        blob = encode_nhd_tail(self._tail(rest_n=3))
        for cut in (1, 5, len(blob) - 8):
            with pytest.raises(ValueError, match="truncated|length mismatch"):
                decode_nhd_tail(blob[:cut])

    def test_out_of_range_fields_rejected(self):
        with pytest.raises(ValueError, match="u16"):
            encode_nhd_tail(self._tail(records=((0x1_0000, 0, 0, 1),)))
        with pytest.raises(ValueError, match="u32"):
            encode_nhd_tail(self._tail(records=((0, 0, 0, 1 << 33),)))
        with pytest.raises(ValueError, match="disagree"):
            encode_nhd_tail(
                NHDResultTail(
                    src_base=0,
                    records=(),
                    rest_dst_ptrs=np.array([1], dtype=np.int64),
                    rest_sizes=_EMPTY,
                )
            )

    def test_legacy_and_structured_dispatch(self):
        # decode_result_tail must recognize both forms by frame shape + magic.
        structured = [b"KV_AGENT_RESULT", b"prefix", encode_nhd_tail(self._tail())]
        dst, sizes, src_base, tail = btr.decode_result_tail(structured)
        assert dst is None and sizes is None and src_base is None
        assert isinstance(tail, NHDResultTail)

        wm = SimpleNamespace(
            dst_ptrs=np.array([1, 2], dtype=np.int64),
            sizes=np.array([8, 8], dtype=np.int64),
            bounce_dst_base=0x10,
        )
        legacy = [b"KV_AGENT_RESULT", b"prefix"] + btr.encode_result_tail(wm)
        dst, sizes, src_base, tail = btr.decode_result_tail(legacy)
        assert tail is None and src_base == 0x10 and dst.size == 2

    def test_encode_result_tail_picks_structured_form(self):
        # a WriteMeta with structured sections emits the single-frame form whose
        # records mirror the sections and whose rest tables are the materialized
        # remainder (e.g. replicated side caches).
        spec = NHDGatherSpec(
            block_ptrs=np.array([0x100], dtype=np.int64),
            flat_offsets=np.array([0, 16], dtype=np.int64),
            frag_bytes=16,
        )
        section = SimpleNamespace(spec=spec, dst_lg=2, dst_pool=1, dst_skip=3, n_blocks=1)
        wm = SimpleNamespace(
            dst_ptrs=np.array([0x9000], dtype=np.int64),
            sizes=np.array([256], dtype=np.int64),
            bounce_dst_base=0x4000,
            nhd_sections=[section],
        )
        frames = btr.encode_result_tail(wm)
        assert len(frames) == 1
        tail = decode_nhd_tail(frames[0])
        assert tail.src_base == 0x4000
        assert tail.records == ((2, 1, 3, 1),)
        assert np.array_equal(tail.rest_dst_ptrs, wm.dst_ptrs)
        assert np.array_equal(tail.rest_sizes, wm.sizes)


# --------------------------------------------------------------------------- #
# Receiver scatter template — slicing + fan-in writer head stride
# --------------------------------------------------------------------------- #
class TestScatterTemplate:
    def _template(self, n_blocks=4, n_off=3, frag=16, stride=32):
        return NHDScatterTemplate(
            block_ptrs=(0x1000 + 0x100 * np.arange(n_blocks, dtype=np.int64)),
            flat_offsets=np.arange(n_off, dtype=np.int64) * frag,
            frag_bytes=frag,
            writer_head_stride=stride,
        )

    def test_spec_for_slices_blocks(self):
        t = self._template()
        spec = t.spec_for(0, dst_skip=1, n_blocks=2)
        assert np.array_equal(spec.block_ptrs, t.block_ptrs[1:3])
        assert np.array_equal(spec.flat_offsets, t.flat_offsets)
        assert spec.frag_bytes == t.frag_bytes

    def test_spec_for_applies_writer_stride(self):
        t = self._template(stride=32)
        spec = t.spec_for(2, dst_skip=0, n_blocks=4)
        assert np.array_equal(spec.flat_offsets, t.flat_offsets + 64)

    def test_total_bytes_covers_full_writer_template(self):
        assert self._template(n_blocks=4, n_off=3, frag=16).total_bytes == 4 * 3 * 16

    @pytest.mark.parametrize(
        "writer_index,dst_skip,n_blocks",
        [(-1, 0, 1), (0, -1, 1), (0, 0, 5), (0, 3, 2)],
    )
    def test_spec_for_rejects_out_of_range(self, writer_index, dst_skip, n_blocks):
        with pytest.raises(ValueError):
            self._template().spec_for(writer_index, dst_skip=dst_skip, n_blocks=n_blocks)

    def test_validation(self):
        with pytest.raises(ValueError, match="int64"):
            NHDScatterTemplate(
                block_ptrs=np.zeros(1, dtype=np.int32),
                flat_offsets=_EMPTY,
                frag_bytes=16,
                writer_head_stride=16,
            )
        with pytest.raises(ValueError, match="frag_bytes"):
            NHDScatterTemplate(
                block_ptrs=_EMPTY, flat_offsets=_EMPTY, frag_bytes=0, writer_head_stride=16
            )
        with pytest.raises(ValueError, match="writer_head_stride"):
            NHDScatterTemplate(
                block_ptrs=_EMPTY, flat_offsets=_EMPTY, frag_bytes=16, writer_head_stride=-1
            )


# --------------------------------------------------------------------------- #
# Eligibility / fallback decision logic (receiver side)
# --------------------------------------------------------------------------- #
def test_structured_bounce_eligibility_gate():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap

    eligible = tfr.Receiver._structured_bounce_eligible
    template = {(0, 0): object()}
    tp2 = PeerOverlap(overlap_pp_size=1, overlap_tp_size=2, overlap_cp_size=1, ranks=[0, 1])
    tp1 = PeerOverlap(overlap_pp_size=1, overlap_tp_size=1, overlap_cp_size=1, ranks=[0])
    # no structured section -> nothing to gain
    assert eligible({}, 1, tp1) is False
    assert eligible({}, 2, tp2) is False
    # Single writer is eligible with a non-structured remainder; the later split
    # policy decides whether that remainder is merged or sent directly.
    assert eligible(template, 1, tp1) is True
    # pure-TP fan-in is eligible even with non-structured pools in the mix: the slot is
    # sized for the structured sections alone (structured_only split routing) and the
    # replicated remainder goes via direct descriptors, so the even split holds.
    assert eligible(template, 2, tp2) is True


def test_structured_bounce_eligibility_rejects_pp_and_cp_fanin():
    # NHDScatterTemplate.spec_for's writer_index -> head-stride math assumes writers
    # differ by TP head shard. PP (or CP) fan-in has writers differing by LAYER window
    # (or sequence chunk) with identical byte counts — _fanin_bounce_safe admits even-PP
    # for the plain coalesced bounce, but structured staging must decline or a head
    # shift is silently applied where a layer-window shift belongs.
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    from tensorrt_llm._torch.disaggregation.native.peer import PeerOverlap

    eligible = tfr.Receiver._structured_bounce_eligible
    template = {(0, 0): object()}
    # ctx tp2/pp2 -> gen tp1/pp1: 4-writer fan-in, even PP layer split, head mismatch
    pp2_tp2 = PeerOverlap(
        overlap_pp_size=2, overlap_tp_size=2, overlap_cp_size=1, ranks=[0, 1, 2, 3]
    )
    assert eligible(template, 4, pp2_tp2) is False
    cp2 = PeerOverlap(overlap_pp_size=1, overlap_tp_size=1, overlap_cp_size=2, ranks=[0, 1])
    assert eligible(template, 2, cp2) is False
    # single-writer overlaps are unaffected even if sizes are recorded as pp/cp
    pp1 = PeerOverlap(overlap_pp_size=1, overlap_tp_size=1, overlap_cp_size=1, ranks=[0])
    assert eligible(template, 1, pp1) is True


def test_fanin_writer_set_is_exact_for_single_dp_broadcast():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    exact = tfr.Receiver._fanin_writer_set_is_exact

    assert exact(sender_dp_rank=0, peer_dp_size=4) is True
    assert exact(sender_dp_rank=None, peer_dp_size=1) is True
    assert exact(sender_dp_rank=None, peer_dp_size=2) is False


def test_adp_broadcast_rank_union_preserves_topology_order():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    ordered_union = tfr.Receiver._ordered_rank_union

    # set({7, 15}) is not guaranteed to preserve TP order. Duplicate ranks across
    # groups are removed at their first topology-ordered occurrence.
    overlaps = [SimpleNamespace(ranks=[7, 15]), SimpleNamespace(ranks=[15, 23])]
    assert ordered_union(overlaps) == [7, 15, 23]


def test_legacy_fanin_rejects_replicated_owner_pool():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind

    def page_table(*kinds):
        views = [SimpleNamespace(mapper_kind=kind) for kind in kinds]
        return SimpleNamespace(layer_groups=[SimpleNamespace(pool_views=views)])

    mapping = [((0, 0), (0, 0)), ((0, 1), (0, 1))]
    uniform = page_table(MapperKind.NHD, MapperKind.NHD)
    mixed = page_table(MapperKind.NHD, MapperKind.REPLICATED)
    assert tfr.Receiver._legacy_fanin_pools_uniform(mapping, uniform) is True
    assert tfr.Receiver._legacy_fanin_pools_uniform(mapping, mixed) is False


def test_legacy_bounce_layout_requires_all_views_in_sized_physical_pool():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    complete = tfr.Receiver._legacy_bounce_layout_complete
    mapping = [((0, 0), (0, 0)), ((0, 1), (0, 1))]

    def page_table(*physical_pool_ids):
        views = [SimpleNamespace(pool_idx=pool_id) for pool_id in physical_pool_ids]
        return SimpleNamespace(layer_groups=[SimpleNamespace(pool_views=views)])

    # Logical NHD and replicated views may be coalesced in physical pool zero.
    assert complete(mapping, page_table(0, 0)) is True
    # A separately backed INDEX_KEY view is absent from legacy pool-zero sizing.
    assert complete(mapping, page_table(0, 1)) is False


def test_structured_split_scope_merges_bounded_single_writer_remainder():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    required = tfr.Receiver._structured_split_required
    plan = {(0, 0): object()}

    assert required(None, expected_transfers=1, has_unbounded_remainder=True) is False
    assert required(plan, expected_transfers=1, has_unbounded_remainder=False) is False
    assert required(plan, expected_transfers=1, has_unbounded_remainder=True) is True
    assert required(plan, expected_transfers=2, has_unbounded_remainder=False) is True


# --------------------------------------------------------------------------- #
# RecvReqInfo wire format — the structured_nhd key must be absent when False so
# an old sender/new receiver (or the reverse) mix keeps working with the flag off
# (from_bytes is cls(**d): an unknown key raises TypeError on old builds)
# --------------------------------------------------------------------------- #
class TestRecvReqInfoWireFormat:
    def _req(self, tfr, **kwargs):
        return tfr.RecvReqInfo(
            sender_req_id=7,
            instance_name="gen0",
            instance_rank=1,
            block_ids_per_layer_groups=[np.arange(3, dtype=np.int64)],
            unique_rid=42,
            **kwargs,
        )

    def test_flag_off_wire_format_has_no_structured_key(self):
        import msgpack

        tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
        blob = self._req(tfr).to_bytes()
        d = msgpack.unpackb(blob, raw=False)
        assert "structured_nhd" not in d
        assert "structured_only" not in d
        # byte-identical to a hand-built pre-Phase-2 payload (no structured keys)
        assert blob == msgpack.packb(d)

    def test_flag_on_roundtrips(self):
        tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
        req = self._req(tfr, structured_nhd=True, structured_only=True, bounce_dst_base=0x2000)
        out = tfr.RecvReqInfo.from_bytes(req.to_bytes())
        assert out.structured_nhd is True
        assert out.structured_only is True
        assert out.bounce_dst_base == 0x2000
        assert np.array_equal(out.block_ids_per_layer_groups[0], req.block_ids_per_layer_groups[0])

    def test_payload_without_key_decodes_flag_off(self):
        import msgpack

        tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
        d = msgpack.unpackb(self._req(tfr).to_bytes(), raw=False)
        assert "structured_nhd" not in d  # simulate an old sender's payload
        out = tfr.RecvReqInfo.from_bytes(msgpack.packb(d))
        assert out.structured_nhd is False

    def test_unknown_keys_from_newer_peer_are_dropped(self):
        import msgpack

        tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
        d = msgpack.unpackb(self._req(tfr, structured_nhd=True).to_bytes(), raw=False)
        d["some_future_feature_key"] = 123  # newer peer's optional key
        out = tfr.RecvReqInfo.from_bytes(msgpack.packb(d))
        assert out.structured_nhd is True
        assert not hasattr(out, "some_future_feature_key")


# --------------------------------------------------------------------------- #
# Structured tail -> scatter descriptor resolution + transport integration
# (GPU allocators, streams and the scatter worker are mocked as in test_bounce)
# --------------------------------------------------------------------------- #
def _template_for_desc(n_blocks=4, n_off=2, frag=16):
    return NHDScatterTemplate(
        block_ptrs=(0x9000 + 0x40 * np.arange(n_blocks, dtype=np.int64)),
        flat_offsets=np.arange(n_off, dtype=np.int64) * frag,
        frag_bytes=frag,
        writer_head_stride=frag * n_off,
    )


def _ctx(scatter_plan, base=0x1000, per_writer=4096, num_writers=2):
    return bcore.TransferContext(
        rid_slice=(1, 0),
        slot_id=0,
        base_addr=base,
        per_writer_bytes=per_writer,
        num_writers=num_writers,
        scatter_plan=scatter_plan,
    )


class TestStructuredScatterDesc:
    def test_resolves_writer_index_and_slice(self):
        template = _template_for_desc()
        ctx = _ctx({(0, 0): template})
        tail = NHDResultTail(
            src_base=0x1000 + 4096,  # writer 1
            records=((0, 0, 1, 2),),
            rest_dst_ptrs=_EMPTY,
            rest_sizes=_EMPTY,
        )
        src_base, specs, rest_dst, rest_sizes = btr.VmmBounceTransport._structured_scatter_desc(
            ctx, tail
        )
        assert src_base == 0x1000 + 4096
        (spec,) = specs
        assert np.array_equal(spec.block_ptrs, template.block_ptrs[1:3])
        # writer 1's head sub-range: offsets shifted by one head stride
        assert np.array_equal(
            spec.flat_offsets, template.flat_offsets + template.writer_head_stride
        )

    def test_rejections(self):
        template = _template_for_desc()
        rest = (_EMPTY, _EMPTY)
        # no plan reserved
        with pytest.raises(ValueError, match="no scatter templates"):
            btr.VmmBounceTransport._structured_scatter_desc(
                _ctx(None), NHDResultTail(0x1000, ((0, 0, 0, 1),), *rest)
            )
        # unknown pool key
        with pytest.raises(ValueError, match="no scatter template for pool"):
            btr.VmmBounceTransport._structured_scatter_desc(
                _ctx({(0, 0): template}), NHDResultTail(0x1000, ((7, 7, 0, 1),), *rest)
            )
        # src_base not on a writer boundary
        with pytest.raises(ValueError, match="writer base"):
            btr.VmmBounceTransport._structured_scatter_desc(
                _ctx({(0, 0): template}), NHDResultTail(0x1001, ((0, 0, 0, 1),), *rest)
            )
        # claimed bytes exceed the writer's sub-region
        small = _ctx({(0, 0): template}, per_writer=16)
        with pytest.raises(ValueError, match="exceeding"):
            btr.VmmBounceTransport._structured_scatter_desc(
                small, NHDResultTail(0x1000, ((0, 0, 0, 4),), *rest)
            )

    def test_empty_tail_yields_no_desc(self):
        ctx = _ctx({(0, 0): _template_for_desc()})
        tail = NHDResultTail(0x1000, ((0, 0, 0, 0),), _EMPTY, _EMPTY)
        assert btr.VmmBounceTransport._structured_scatter_desc(ctx, tail) is None


class _FakeAlloc:
    """Linear SlotAllocator stand-in that records adjacent reservations and releases."""

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        self._cap = capacity_bytes
        self.base = 0x100000
        self.next_id = 0
        self.next_offset = 0
        self.reservations = []
        self.released = []
        self.quarantined = []

    @property
    def capacity(self):
        return self._cap

    def reserve(self, size, timeout=None):
        if self.next_offset + size > self._cap:
            return None
        sid = self.next_id
        self.next_id += 1
        start = self.base + self.next_offset
        self.next_offset += size
        self.reservations.append((sid, start, size))
        return sid, start

    def release(self, slot_id):
        self.released.append(slot_id)

    def quarantine(self, slot_id, grace_s):
        self.quarantined.append(slot_id)

    def reclaim_expired(self):
        return 0

    def reg_descs(self):
        return []


def _make_transport(monkeypatch, block_bytes_per_group, capacity=1 << 30, min_blocks=96):
    monkeypatch.setattr(btr, "SlotAllocator", _FakeAlloc)
    monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self: 0)
    monkeypatch.setattr(
        btr.VmmBounceTransport,
        "_start_scatter_worker",
        lambda self, name: setattr(self, "_scatter_q", queue.Queue()),
    )
    agent = SimpleNamespace(register_memory=lambda d: None)
    return btr.VmmBounceTransport(
        agent,
        device_id=0,
        capacity_bytes=capacity,
        phys_chunk_size=32 * 1024 * 1024,
        block_bytes_per_group=block_bytes_per_group,
        min_blocks=min_blocks,
        structured_nhd=True,
    )


def _recv_req(block_counts, rid=1, slice_id=0):
    return SimpleNamespace(
        block_ids_per_layer_groups=[SimpleNamespace(size=n) for n in block_counts],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
    )


class TestReserveAndRecordStructured:
    def test_min_blocks_bypass_only_when_descriptor_dominated(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=96)
        # 2 blocks < 96: the size heuristic skips bounce...
        assert t.reserve(_recv_req([2]), num_writers=1) is False
        # ...unless the transfer is descriptor-dominated (NHD head mismatch)
        req = _recv_req([2], rid=2)
        assert t.reserve(req, num_writers=1, descriptor_dominated=True) is True
        assert req.bounce_dst_base == 0x100000

    def test_record_result_structured_tail_builds_desc(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        template = _template_for_desc(n_blocks=2, n_off=2, frag=16)
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1, scatter_plan={(0, 0): template}) is True
        rid_slice = (req.unique_rid, req.slice_id)
        tail = NHDResultTail(
            src_base=req.bounce_dst_base,
            records=((0, 0, 0, 2),),
            rest_dst_ptrs=np.array([0xAA00], dtype=np.int64),
            rest_sizes=np.array([32], dtype=np.int64),
        )
        t.record_result(rid_slice, 3, structured_tail=tail)
        ctx, descs = t._scatter_q.get_nowait()
        (desc,) = descs
        assert len(desc) == 4  # structured form: (src_base, specs, rest_dst, rest_sizes)
        src_base, specs, rest_dst, rest_sizes = desc
        assert src_base == req.bounce_dst_base
        assert np.array_equal(specs[0].block_ptrs, template.block_ptrs)
        assert np.array_equal(rest_dst, tail.rest_dst_ptrs)

    def test_mismatched_tail_fails_writer_without_scatter(self, monkeypatch):
        # a tail that does not fit the reservation must fail the writer (drain, release)
        # rather than scatter garbage or strand the region.
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        template = _template_for_desc(n_blocks=2)
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1, scatter_plan={(0, 0): template}) is True
        rid_slice = (req.unique_rid, req.slice_id)
        bad_tail = NHDResultTail(
            src_base=req.bounce_dst_base,
            records=((9, 9, 0, 1),),  # pool key the receiver never planned
            rest_dst_ptrs=_EMPTY,
            rest_sizes=_EMPTY,
        )
        calls = []
        t.record_result(rid_slice, 3, structured_tail=bad_tail, on_done=lambda ok: calls.append(ok))
        assert t._scatter_q.empty()  # nothing scattered
        assert calls == [False]  # completion reports failure
        assert t._recv_alloc.released  # region drained and freed

    def test_oversized_structured_tail_cannot_consume_next_reservation(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        template = _template_for_desc(n_blocks=2)
        first = _recv_req([2], rid=1)
        second = _recv_req([2], rid=2)
        reserve_args = {
            "num_writers": 1,
            "scatter_plan": {(0, 0): template},
            "descriptor_dominated": True,
            "staged_bytes_per_writer": 1024,
        }
        assert t.reserve(first, **reserve_args) is True
        assert t.reserve(second, **reserve_args) is True
        first_ctx = t._reserved_map[(first.unique_rid, first.slice_id)]
        second_ctx = t._reserved_map[(second.unique_rid, second.slice_id)]
        assert first_ctx.base_addr + first_ctx.per_writer_bytes == second_ctx.base_addr

        tail = NHDResultTail(
            src_base=first.bounce_dst_base,
            records=((0, 0, 0, 2),),
            rest_dst_ptrs=np.array([0xAA00], dtype=np.int64),
            rest_sizes=np.array([1024], dtype=np.int64),
        )
        calls = []
        t.record_result(
            (first.unique_rid, first.slice_id),
            3,
            structured_tail=tail,
            on_done=lambda ok: calls.append(ok),
        )

        assert t._scatter_q.empty()
        assert calls == [False]
        assert first_ctx.slot_id in t._recv_alloc.released
        assert second_ctx.slot_id not in t._recv_alloc.released
        assert (second.unique_rid, second.slice_id) in t._reserved_map

    @pytest.mark.parametrize("num_writers", [1, 2])
    def test_oversized_legacy_tail_fails_writer_without_scatter(self, monkeypatch, num_writers):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        req = _recv_req([2])
        assert t.reserve(req, num_writers=num_writers) is True
        rid_slice = (req.unique_rid, req.slice_id)
        ctx = t._reserved_map[rid_slice]
        calls = []
        t.record_result(
            rid_slice,
            3,
            dst_ptrs=np.array([0xAA00], dtype=np.int64),
            sizes=np.array([ctx.per_writer_bytes + 1], dtype=np.int64),
            src_base=req.bounce_dst_base,
            on_done=lambda ok: calls.append(ok),
        )
        if num_writers == 2:
            t.record_result(
                rid_slice,
                4,
                dst_ptrs=np.array([0xCC00], dtype=np.int64),
                sizes=np.array([ctx.per_writer_bytes], dtype=np.int64),
                src_base=req.bounce_dst_base + ctx.per_writer_bytes,
            )
        assert t._scatter_q.empty()
        assert calls == [False]
        assert t._recv_alloc.released


# --------------------------------------------------------------------------- #
# Structured split routing (structured_only): NHD sections stage in the slot, the
# materialized remainder (replicated INDEX_KEY et al.) rides the SAME NIXL write
# as direct per-fragment descriptors — never the slot
# --------------------------------------------------------------------------- #
class TestStructuredSplitRouting:
    def _wm(self, direct_rest=True):
        spec = NHDGatherSpec(
            block_ptrs=np.array([0x100, 0x200], dtype=np.int64),
            flat_offsets=np.array([0, 16], dtype=np.int64),
            frag_bytes=16,
        )
        return SimpleNamespace(
            src_ptrs=np.array([0x111, 0x222], dtype=np.int64),
            dst_ptrs=np.array([0x911, 0x922], dtype=np.int64),
            sizes=np.array([64, 64], dtype=np.int64),
            nhd_sections=[SimpleNamespace(spec=spec, dst_lg=0, dst_pool=0, dst_skip=0, n_blocks=2)],
            bounce_dst_base=0x8000,
            bounce_direct_rest=direct_rest,
            dst_device_id=1,
            peer_name="gen0",
        )

    def test_reserve_and_gather_sizes_slot_with_specs_only(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        launched = []
        monkeypatch.setattr(
            t, "_launch_gather", lambda addr, meta, total: launched.append(total) or None
        )
        wm = self._wm(direct_rest=True)
        got = t._reserve_and_gather(wm, timeout=0.01)
        assert got is not None
        _, _, total, _ = got
        assert total == wm.nhd_sections[0].spec.total_bytes  # 2 blocks x 2 frags x 16B
        assert launched == [total]
        # The lower-level merged layout remains decodable for compatibility.
        wm2 = self._wm(direct_rest=False)
        _, _, total2, _ = t._reserve_and_gather(wm2, timeout=0.01)
        assert total2 == wm2.nhd_sections[0].spec.total_bytes + int(wm2.sizes.sum())

    def test_make_write_appends_direct_descriptors(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        recorded = []

        def record(mem_type, ptrs, sizes, dev):
            recorded.append((ptrs.copy(), sizes.copy(), dev))
            return ("descs", len(recorded))

        monkeypatch.setattr(btr, "MemoryDescs", SimpleNamespace(from_arrays_uniform_device=record))
        monkeypatch.setattr(btr, "TransferRequest", lambda *args: SimpleNamespace(args=args))
        wm = self._wm(direct_rest=True)
        req = t._make_write(0x7000, wm, total=64)
        # one request: [slot desc] + the remainder's direct per-fragment descriptors,
        # so a single submit/wait covers the bounce write AND the direct writes
        (src_ptrs, src_sizes, src_dev), (dst_ptrs, dst_sizes, dst_dev) = recorded
        assert src_ptrs.tolist() == [0x7000, 0x111, 0x222]
        assert dst_ptrs.tolist() == [0x8000, 0x911, 0x922]
        assert src_sizes.tolist() == dst_sizes.tolist() == [64, 64, 64]
        assert src_dev == 0 and dst_dev == 1
        assert req.args[0] == btr.TransferOp.WRITE
        # The lower-level merged layout keeps the single coalesced descriptor.
        recorded.clear()
        t._make_write(0x7000, self._wm(direct_rest=False), total=192)
        (src_ptrs, _, _), (dst_ptrs, _, _) = recorded
        assert src_ptrs.tolist() == [0x7000] and dst_ptrs.tolist() == [0x8000]

    def test_direct_rest_tail_carries_empty_rest_tables(self):
        wm = self._wm(direct_rest=True)
        (frame,) = btr.encode_result_tail(wm)
        tail = decode_nhd_tail(frame)
        assert tail.records == ((0, 0, 0, 2),)
        assert tail.rest_dst_ptrs.size == 0 and tail.rest_sizes.size == 0
        # The lower-level merged layout still carries the remainder in the tail.
        (frame2,) = btr.encode_result_tail(self._wm(direct_rest=False))
        tail2 = decode_nhd_tail(frame2)
        assert np.array_equal(tail2.rest_dst_ptrs, wm.dst_ptrs)

    def test_reserve_structured_bytes_per_writer_splits_evenly(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        req = _recv_req([4])
        # block-table sizing would be 4 x 4096; the structured override wins and the
        # per-writer sections split exactly
        assert (
            t.reserve(
                req,
                num_writers=2,
                scatter_plan={(0, 0): _template_for_desc()},
                descriptor_dominated=True,
                staged_bytes_per_writer=1024,
            )
            is True
        )
        ctx = t._reserved_map[(req.unique_rid, req.slice_id)]
        assert ctx.per_writer_bytes == 1024 and ctx.num_writers == 2

    @pytest.mark.parametrize("staged_bytes_per_writer", [0, -1])
    def test_reserve_rejects_nonpositive_structured_bytes(
        self, monkeypatch, staged_bytes_per_writer
    ):
        t = _make_transport(monkeypatch, block_bytes_per_group=[4096], min_blocks=1)
        assert (
            t.reserve(
                _recv_req([4]),
                num_writers=2,
                scatter_plan={(0, 0): _template_for_desc()},
                descriptor_dominated=True,
                staged_bytes_per_writer=staged_bytes_per_writer,
            )
            is False
        )


class _FakeExtractor:
    """Page-table stand-in: block base pointer = base + block_id x slot + pool offset."""

    _POOL_STRIDE = 0x100000

    def __init__(self, base, slot_bytes, tpb=16, pool_region_bytes=None):
        self._base = base
        self._slot = slot_bytes
        self._pool_region_bytes = pool_region_bytes or {}
        self.page_table = SimpleNamespace(
            tokens_per_block=tpb,
            layer_groups=[SimpleNamespace(sliding_window_size=None)],
        )

    def extract(self, block_ids, layer_group_id, pool_idx):
        ptrs = (
            self._base
            + np.asarray(block_ids, dtype=np.int64) * self._slot
            + pool_idx * self._POOL_STRIDE
        )
        return SimpleNamespace(
            memory=SimpleNamespace(
                ptrs=ptrs,
                bytes_per_region=self._pool_region_bytes.get(pool_idx, self._slot),
            )
        )


class _FakeNHDMapper:
    supports_structured_staging = True

    def __init__(self, n_off=4, frag_bytes=32):
        self.flat_offsets = np.arange(n_off, dtype=np.int64) * frag_bytes
        self.frag_bytes = frag_bytes

    def src_gather_spec(self, ptrs):
        return NHDGatherSpec(
            block_ptrs=np.ascontiguousarray(ptrs, dtype=np.int64),
            flat_offsets=self.flat_offsets,
            frag_bytes=self.frag_bytes,
        )

    def recv_scatter_template(self, ptrs):
        return NHDScatterTemplate(
            block_ptrs=np.ascontiguousarray(ptrs, dtype=np.int64),
            flat_offsets=self.flat_offsets,
            frag_bytes=self.frag_bytes,
            writer_head_stride=self.frag_bytes,
        )

    def map(self, src, dst):
        raise AssertionError("structured pool must not materialize fragment tables")


class _FakeReplicatedMapper:
    supports_structured_staging = False

    def map(self, src, dst):
        return SimpleNamespace(
            src=SimpleNamespace(memory=SimpleNamespace(ptrs=src.memory.ptrs, bytes_per_region=64)),
            dst=SimpleNamespace(memory=SimpleNamespace(ptrs=dst.memory.ptrs, bytes_per_region=64)),
        )


class _FakeRegistrar:
    def __init__(self, mappers):
        self.self_extractor = _FakeExtractor(0x100000, 4096)
        self._peer_ext = _FakeExtractor(0x900000, 4096)
        self._mappers = mappers
        self.self_rank_info = SimpleNamespace(instance_name="ctx", instance_rank=0, device_id=0)

    def get_peer_rank_info(self, name, rank):
        return SimpleNamespace(
            instance_name=name, instance_rank=rank, dp_rank=0, device_id=1, self_endpoint="ep"
        )

    def get_peer_overlap(self, peer_ri, dp_rank):
        return SimpleNamespace(ranks=[0])

    def peer_extractor(self, name, rank):
        return self._peer_ext

    def get_pool_mapping(self, peer_ri):
        return [(key, key) for key in self._mappers]

    def should_send_pool(
        self, targets, peer_ri, layer_group_id, pool_idx, peer_layer_group_id, peer_pool_idx
    ):
        return True

    def get_kv_map(self, peer_ri, self_key, peer_key):
        return self._mappers[self_key]


def _build_split_meta(tfr, monkeypatch, *, structured_only, mappers=None):
    monkeypatch.setattr(
        tfr.MambaPolicy, "collect_frags", staticmethod(lambda **kwargs: ([], [], []))
    )
    sender = object.__new__(tfr.Sender)
    sender._registrar = _FakeRegistrar(
        mappers
        if mappers is not None
        else {(0, 0): _FakeNHDMapper(), (0, 1): _FakeReplicatedMapper()}
    )
    sender._bounce = SimpleNamespace(structured_nhd=True)
    task = SimpleNamespace(
        _slice=SimpleNamespace(
            block_ids_per_layer_groups=[np.array([3, 4], dtype=np.int64)],
            token_range=SimpleNamespace(end=32),
            is_last_slice=True,
            mamba_state_index=None,
        ),
        _perf_timer=None,
        _beam_width=1,
        _prompt_len=32,
        _unique_rid=42,
        slice_id=0,
    )
    req_info = tfr.RecvReqInfo(
        sender_req_id=1,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[np.array([7, 8], dtype=np.int64)],
        unique_rid=42,
        bounce_dst_base=0x8000,
        structured_nhd=True,
        structured_only=structured_only,
    )
    return tfr.Sender._build_kv_write_meta(sender, task, req_info)


def test_single_writer_capacity_includes_mapped_remainder(monkeypatch):
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    receiver = object.__new__(tfr.Receiver)
    receiver._registrar = _FakeRegistrar(
        {(0, 0): _FakeNHDMapper(), (0, 1): _FakeReplicatedMapper()}
    )
    receiver._registrar.self_extractor = _FakeExtractor(0x900000, 4096, pool_region_bytes={1: 64})
    task = SimpleNamespace(
        _kv_slice=SimpleNamespace(block_ids_per_layer_groups=[np.array([7, 8], dtype=np.int64)])
    )

    templates, remainder_capacity = receiver._structured_scatter_plan(task, object())

    assert receiver._structured_section_bytes(templates) == 2 * 4 * 32
    assert remainder_capacity == 2 * 64
    receiver_reservation = receiver._structured_section_bytes(templates) + remainder_capacity

    # Pin the cross-rank safety invariant: the sender's actual staged bytes may never
    # exceed the receiver-derived reservation, including the merged INDEX_KEY tail.
    wm = _build_split_meta(tfr, monkeypatch, structured_only=False)
    sender_staged = sum(section.spec.total_bytes for section in wm.nhd_sections) + int(
        wm.sizes.sum()
    )
    assert sender_staged <= receiver_reservation


def test_fanin_mixed_pool_split_routing(monkeypatch):
    """Verify mixed-pool fan-in split routing.

    NHD K/V goes structured through the slot, the replicated
    INDEX_KEY pool goes through the direct descriptor path, and the per-writer
    section bytes match the receiver's reserve-time accounting exactly.
    """
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")

    wm = _build_split_meta(tfr, monkeypatch, structured_only=True)
    assert wm.bounce_direct_rest is True
    assert wm.bounce_dst_base == 0x8000
    (section,) = wm.nhd_sections  # the NHD pool staged structurally
    assert (section.dst_lg, section.dst_pool) == (0, 0)
    assert np.array_equal(
        section.spec.block_ptrs, 0x100000 + np.array([3, 4], dtype=np.int64) * 4096
    )
    # the replicated INDEX_KEY (pool 1) went to the DIRECT tables: real peer pointers,
    # never the bounce slot
    expected_dst = 0x900000 + np.array([7, 8], dtype=np.int64) * 4096 + _FakeExtractor._POOL_STRIDE
    assert np.array_equal(wm.dst_ptrs, expected_dst)
    assert wm.sizes.tolist() == [64, 64]

    # equal per-writer section bytes: every fan-in writer of the same geometry stages
    # exactly the receiver's per-writer accounting (reserve's even split is exact)
    mapper = _FakeNHDMapper()
    template = NHDScatterTemplate(
        block_ptrs=np.zeros(2, dtype=np.int64),
        flat_offsets=mapper.flat_offsets,
        frag_bytes=mapper.frag_bytes,
        writer_head_stride=mapper.frag_bytes,
    )
    per_writer = tfr.Receiver._structured_section_bytes({(0, 0): template})
    assert per_writer == section.spec.total_bytes == 2 * 4 * 32


def test_v2_pool_buffer_mapper_builds_structured_section(monkeypatch):
    """V2 MiniMax-style buffer metadata must engage structured staging too.

    The production M3 page table uses non-empty buffer_entries plus per-buffer
    mapper kinds, which makes PeerRegistrar return PoolBufferMapper instead of
    the legacy NHDHeadMismatchMapper. This pins that transfer.py consumes the
    PoolBufferMapper structured-staging interface directly.
    """
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    from test_peer import make_page_table, make_rankinfo

    from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import PoolBufferMapper
    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
    from tensorrt_llm._torch.disaggregation.resource.page import BUFFER_ENTRY_DTYPE, MapperKind

    monkeypatch.setattr(
        tfr.MambaPolicy, "collect_frags", staticmethod(lambda **kwargs: ([], [], []))
    )

    self_pt = make_page_table(pool_ptrs=[0x100000], block_bytes=[32], global_layer_ids=[0])
    peer_pt = make_page_table(pool_ptrs=[0x900000], block_bytes=[16], global_layer_ids=[0])
    self_pt.tokens_per_block = peer_pt.tokens_per_block = 2
    self_entries = np.array([(0, 0, 16), (0, 16, 16)], dtype=BUFFER_ENTRY_DTYPE)
    peer_entries = np.array([(0, 0, 8), (0, 8, 8)], dtype=BUFFER_ENTRY_DTYPE)
    for page_table, entries in ((self_pt, self_entries), (peer_pt, peer_entries)):
        pool_view = page_table.layer_groups[0].pool_views[0]
        pool_view.buffer_entries = entries
        pool_view.pool_role = frozenset({"key", "value"})
        pool_view.mapper_kind = MapperKind.NHD
        pool_view.buffer_roles = ("key", "value")
        pool_view.buffer_mapper_kinds = (MapperKind.NHD, MapperKind.NHD)

    self_ri = make_rankinfo(
        instance_name="ctx",
        tp_size=2,
        kv_heads_per_rank=2,
        tokens_per_block=2,
        dims_per_head=2,
        element_bytes=2,
        page_table=self_pt,
        layer_num_per_pp=[1],
    )
    peer_ri = make_rankinfo(
        instance_name="gen",
        instance_rank=0,
        tp_size=1,
        kv_heads_per_rank=1,
        tokens_per_block=2,
        dims_per_head=2,
        element_bytes=2,
        page_table=peer_pt,
        layer_num_per_pp=[1],
    )
    registrar = PeerRegistrar(self_ri, KVRegionExtractorV1(self_pt))
    registrar.register(peer_ri.instance_name, peer_ri.instance_rank, peer_ri)
    mapper = registrar.get_kv_map(peer_ri, (0, 0), (0, 0))
    assert isinstance(mapper, PoolBufferMapper)
    assert mapper.supports_structured_staging is True

    sender = object.__new__(tfr.Sender)
    sender._registrar = registrar
    sender._bounce = SimpleNamespace(structured_nhd=True)
    task = SimpleNamespace(
        _slice=SimpleNamespace(
            block_ids_per_layer_groups=[np.array([3, 4], dtype=np.int64)],
            token_range=SimpleNamespace(end=4),
            is_last_slice=True,
            mamba_state_index=None,
        ),
        _perf_timer=None,
        _beam_width=1,
        _prompt_len=4,
        _unique_rid=42,
        slice_id=0,
    )
    req_info = tfr.RecvReqInfo(
        sender_req_id=1,
        instance_name=peer_ri.instance_name,
        instance_rank=peer_ri.instance_rank,
        block_ids_per_layer_groups=[np.array([7, 8], dtype=np.int64)],
        unique_rid=42,
        bounce_dst_base=0x8000,
        structured_nhd=True,
        structured_only=False,
    )

    wm = tfr.Sender._build_kv_write_meta(sender, task, req_info)

    assert wm.nhd_sections is not None and len(wm.nhd_sections) == 1
    (section,) = wm.nhd_sections
    assert section.mapper is mapper
    assert section.spec.total_bytes == 2 * 2 * 2 * 4
    assert section.spec.frag_bytes == 4
    assert section.spec.flat_offsets.tolist() == [0, 8, 16, 24]
    assert np.array_equal(
        section.spec.block_ptrs,
        0x100000 + np.array([3, 4], dtype=np.int64) * 32,
    )
    assert wm.dst_ptrs.size == 0
    assert wm.sizes.size == 0


def test_single_writer_coalesces_bounded_remainder(monkeypatch):
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    wm = _build_split_meta(tfr, monkeypatch, structured_only=False)
    assert wm.bounce_direct_rest is False
    assert wm.bounce_dst_base == 0x8000
    assert wm.nhd_sections is not None and wm.dst_ptrs.size == 2


def test_single_writer_unbounded_remainder_routes_direct(monkeypatch):
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    wm = _build_split_meta(tfr, monkeypatch, structured_only=True)
    assert wm.bounce_direct_rest is True
    assert wm.bounce_dst_base == 0x8000
    assert wm.nhd_sections is not None and wm.dst_ptrs.size == 2


def test_structured_only_without_sections_goes_fully_direct(monkeypatch):
    # A writer that stages nothing structurally must not touch the structured-sized
    # slot: gathering materialized bytes there could overflow a sibling's sub-region.
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    wm = _build_split_meta(
        tfr,
        monkeypatch,
        structured_only=True,
        mappers={(0, 1): _FakeReplicatedMapper()},
    )
    assert wm.bounce_dst_base is None
    assert wm.bounce_direct_rest is False
    assert wm.nhd_sections is None and wm.dst_ptrs.size == 2


# --------------------------------------------------------------------------- #
# Sender fallback — structured sections materialize on demand
# --------------------------------------------------------------------------- #
def test_materialize_structured_expands_sections():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")

    frag_ptrs = np.array([0x100, 0x200], dtype=np.int64)
    dst_ptrs = np.array([0x900, 0xA00], dtype=np.int64)
    mapper = SimpleNamespace(
        map=lambda src, dst: SimpleNamespace(
            src=SimpleNamespace(memory=SimpleNamespace(ptrs=frag_ptrs, bytes_per_region=64)),
            dst=SimpleNamespace(memory=SimpleNamespace(ptrs=dst_ptrs, bytes_per_region=64)),
        )
    )
    section = tfr.NHDStagedSection(
        spec=SimpleNamespace(total_bytes=128, n_frags=2),
        dst_lg=0,
        dst_pool=0,
        dst_skip=0,
        n_blocks=2,
        mapper=mapper,
        src_region=object(),
        dst_region=object(),
    )
    wm = SimpleNamespace(
        src_ptrs=np.array([0x1], dtype=np.int64),
        dst_ptrs=np.array([0x2], dtype=np.int64),
        sizes=np.array([16], dtype=np.int64),
        nhd_sections=[section],
    )
    out = tfr.Sender._materialize_structured(wm)
    assert out is wm
    assert wm.nhd_sections is None  # sections consumed; encode_result_tail goes legacy
    assert np.array_equal(wm.src_ptrs, np.array([0x1, 0x100, 0x200], dtype=np.int64))
    assert np.array_equal(wm.dst_ptrs, np.array([0x2, 0x900, 0xA00], dtype=np.int64))
    assert np.array_equal(wm.sizes, np.array([16, 64, 64], dtype=np.int64))
    # no sections -> no-op
    assert tfr.Sender._materialize_structured(wm) is wm


def test_align_kv_blocks_with_skip_reports_dst_skip():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    # The source covers tokens [32, 64), while dst covers [0, 64): alignment
    # skips the first two destination blocks and reports that receiver offset.
    src = np.array([10, 11], dtype=np.int64)
    dst = np.array([20, 21, 22, 23], dtype=np.int64)
    a_src, a_dst, dst_skip = tfr.Sender._align_kv_blocks_with_skip(
        src,
        dst,
        src_token_start=32,
        dst_token_start=0,
        tokens_per_block=16,
    )
    assert dst_skip == 2
    assert np.array_equal(a_src, src)
    assert np.array_equal(a_dst, dst[2:])
    # the 2-tuple wrapper stays signature-compatible
    b_src, b_dst = tfr.Sender._align_kv_blocks(
        src,
        dst,
        src_token_start=32,
        dst_token_start=0,
        tokens_per_block=16,
    )
    assert np.array_equal(b_src, a_src) and np.array_equal(b_dst, a_dst)


# --------------------------------------------------------------------------- #
# GPU e2e — MiniMax-M3 head-mismatch topology through the threaded NIXL harness
# --------------------------------------------------------------------------- #
# (ctx_tp, ctx_pp, ctx_dp, gen_tp, gen_pp, gen_dp, expect_structured, test_id)
# tep1_to_tep2: 2 ctx heads -> 1 head/gen-rank, single writer per receiver: the
#   structured path must engage (NHD specs + replicated index as the rest section).
# tep2_to_dep2: fan-in of 2 writers with a replicated index pool: split routing —
#   the NHD K/V sections stage structurally (equal per-writer bytes by construction)
#   while the replicated INDEX_KEY, sent by one elected writer, goes via direct
#   descriptors on the same write. The structured path must engage.
_E2E_TOPOLOGIES = [
    (1, 1, False, 2, 1, False, True, "tep1_to_tep2"),
    (2, 1, False, 2, 1, True, True, "tep2_to_dep2"),
]


def _run_m3_e2e(monkeypatch, topology, structured_env: str, request_lengths=None):
    import test_deepseek_v4_kv_transfer as transfer_harness
    import test_minimax_m3_kv_transfer as m3

    import tensorrt_llm._torch.disaggregation.native.bounce.impl as bounce_impl
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

    ctx_tp, ctx_pp, ctx_dp, gen_tp, gen_pp, gen_dp, _, _ = topology
    monkeypatch.setenv("TRTLLM_KV_BOUNCE_STRUCTURED_NHD", structured_env)

    calls = {"gather": 0, "scatter": 0}
    real_gather = bounce_impl.gather_structured
    real_scatter = bounce_impl.scatter_structured

    def counting_gather(dst_base, specs, *, stream):
        calls["gather"] += 1
        return real_gather(dst_base, specs, stream=stream)

    def counting_scatter(src_base, specs, *, stream):
        calls["scatter"] += 1
        return real_scatter(src_base, specs, stream=stream)

    monkeypatch.setattr(bounce_impl, "gather_structured", counting_gather)
    monkeypatch.setattr(bounce_impl, "scatter_structured", counting_scatter)

    transfer_harness.run_deepseek_v4_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_dp,
        gen_enable_dp=gen_dp,
        compress_ratios=[1] * m3.NUM_LAYERS,
        update_before_transfer=True,
        manager_factory=lambda tp, pp, enable_dp, layout: m3._create_managers(
            tp, pp, enable_dp, layout, DataType.BF16
        ),
        init_fn=m3._initialize_cache,
        verify_fn=m3._verify_cache,
        transceiver_config=CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            max_tokens_in_buffer=512,
            kv_cache_bounce_size_mb=64,
        ),
        request_lengths=request_lengths,
    )
    return calls


@pytest.mark.cuda
@pytest.mark.timeout(180)
@pytest.mark.parametrize("topology", _E2E_TOPOLOGIES, ids=[t[7] for t in _E2E_TOPOLOGIES])
@pytest.mark.parametrize("structured_env", ["0", "1"], ids=["flag_off", "flag_on"])
def test_m3_head_mismatch_structured_e2e(monkeypatch, topology, structured_env):
    """Head-mismatched M3 transfer with bounce enabled, flag off vs on.

    The harness verifier asserts EXACT (rtol=0, atol=0) equality of every gen-side
    cache byte against its ctx-side source, so a passing flag-on run is byte-identical
    to the flag-off path. The counters assert the structured kernels actually ran
    (including the fan-in + replicated INDEX_KEY mix, which now takes split routing).
    """
    expect_structured = topology[6] and structured_env == "1"
    calls = _run_m3_e2e(monkeypatch, topology, structured_env)
    if expect_structured:
        assert calls["gather"] > 0, "structured gather never engaged"
        assert calls["scatter"] > 0, "structured scatter never engaged"
    else:
        assert calls["gather"] == 0 and calls["scatter"] == 0


@pytest.mark.cuda
@pytest.mark.timeout(180)
@pytest.mark.parametrize("structured_env", ["0", "1"], ids=["flag_off", "flag_on"])
def test_m3_tep2_to_dep2_long_context_forces_bounce(monkeypatch, structured_env):
    """Long-context TEP2 ctx -> DEP2 gen fan-in with the replicated INDEX_KEY mix.

    Multi-block requests (4 and 3 blocks of 128 tokens, the harness ceiling) engage
    the bounce path through the descriptor_dominated bypass of min_blocks; TASK-3
    split routing stages the NHD K/V sections structurally per fan-in writer while
    the elected writer's INDEX_KEY rides direct descriptors on the same write. The
    position-dependent fill plus the verifier's exact-equality check make the flag-on
    run byte-identical to flag-off, for K/V and INDEX_KEY alike.
    """
    calls = _run_m3_e2e(
        monkeypatch,
        (2, 1, False, 2, 1, True, True, "tep2_to_dep2_longctx"),
        structured_env,
        request_lengths=[512, 384],  # 4 blocks and 3 blocks at TOKENS_PER_BLOCK=128
    )
    if structured_env == "1":
        assert calls["gather"] > 0, "structured gather never engaged"
        assert calls["scatter"] > 0, "structured scatter never engaged"
    else:
        assert calls["gather"] == 0 and calls["scatter"] == 0
