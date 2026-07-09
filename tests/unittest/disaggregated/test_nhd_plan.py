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
"""Unit tests for the structured NHD staging plan (bounce/nhd_plan.py).

The core property under test: for any NHD head-mismatch geometry, the analytic
NHDGatherSpec expansion must be byte-identical — same (address, size) fragments in the
same order, on both the paged and the staging side — to what NHDHeadMismatchMapper.map()
plus the existing bounce dst-table path (gather_scatter.Plan running offsets) produce.
Randomized geometries cover layers, buffers per layer, tokens per block, self/peer head
splits, layer offsets, prefix-skip/draft-trim block slicing, and sub-byte (NVFP4-adjacent
0.5-byte) element sizes.

Plan construction and the numpy/memmove reference run on CPU; the Triton kernel test is
gated on CUDA and compares against the same reference.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_peer import make_rankinfo

from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    HeadMismatchMapper,
    IdentityMapper,
    NHDHeadMismatchMapper,
    PoolBufferMapper,
    PoolBufferMapping,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind

# The bounce package init pulls CUDA-binding modules; skip gracefully when those are
# absent (CPU-only env without cuda-python). Catch only ImportError so a genuine bug in
# the module still fails CI instead of being silently turned into a skip.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce.gather_scatter import (
        _HAVE_TRITON,
        Plan,
        gather_structured,
        scatter_structured,
    )
    from tensorrt_llm._torch.disaggregation.native.bounce.nhd_plan import (
        NHDGatherSpec,
        copy_expanded_host,
        expand_specs,
        specs_total_bytes,
    )

    _HAVE_BOUNCE = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_BOUNCE = False

pytestmark = pytest.mark.skipif(not _HAVE_BOUNCE, reason="bounce import needs cuda-python")


# --------------------------------------------------------------------------- #
# geometry fixtures — randomized NHD head-mismatch cases
# --------------------------------------------------------------------------- #
def _random_geometry(rng: np.random.Generator) -> SimpleNamespace:
    """One random NHD head-mismatch geometry plus the mapper built from it."""
    tokens_per_block = int(rng.choice([1, 2, 4, 16]))
    buffers_per_layer = int(rng.choice([1, 2]))  # 1 also covers MLA-like single-buffer pools
    dims_per_head = int(rng.choice([2, 4, 64, 128]))
    element_bytes = float(rng.choice([0.5, 1.0, 2.0]))  # 0.5 = NVFP4-adjacent sub-byte
    bytes_per_token_head = int(dims_per_head * element_bytes)

    # a genuine head mismatch: same total model heads, different TP splits
    total_heads = int(rng.choice([2, 4, 8]))
    tp_choices = [t for t in (1, 2, 4, 8) if t <= total_heads and total_heads % t == 0]
    self_tp, peer_tp = rng.choice(tp_choices, size=2, replace=False)
    self_tp, peer_tp = int(self_tp), int(peer_tp)
    self_heads = total_heads // self_tp
    peer_heads = total_heads // peer_tp

    transfer_layers = int(rng.integers(1, 5))
    src_layer_off = int(rng.integers(0, 3))
    peer_layer_off = int(rng.integers(0, 3))
    self_pool_num_layers = src_layer_off + transfer_layers + int(rng.integers(0, 3))
    peer_pool_num_layers = peer_layer_off + transfer_layers + int(rng.integers(0, 3))

    def region_bytes(num_layers: int, heads: int) -> int:
        return num_layers * buffers_per_layer * tokens_per_block * heads * bytes_per_token_head

    self_region_bytes = region_bytes(self_pool_num_layers, self_heads)
    peer_region_bytes = region_bytes(peer_pool_num_layers, peer_heads)

    self_ri = make_rankinfo(
        tp_size=self_tp,
        tp_rank=int(rng.integers(0, self_tp)),
        kv_heads_per_rank=self_heads,
        tokens_per_block=tokens_per_block,
        dims_per_head=dims_per_head,
        element_bytes=element_bytes,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=peer_tp,
        tp_rank=int(rng.integers(0, peer_tp)),
        kv_heads_per_rank=peer_heads,
        tokens_per_block=tokens_per_block,
        dims_per_head=dims_per_head,
        element_bytes=element_bytes,
    )
    mapper = NHDHeadMismatchMapper(
        transfer_layers=transfer_layers,
        src_layer_off=src_layer_off,
        peer_layer_off=peer_layer_off,
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_region_bytes=self_region_bytes,
        peer_region_bytes=peer_region_bytes,
        self_pool_num_layers=self_pool_num_layers,
        peer_pool_num_layers=peer_pool_num_layers,
        self_buffers_per_layer=buffers_per_layer,
        peer_buffers_per_layer=buffers_per_layer,
    )

    # aligned block lists, sliced as _prepare_kv_blocks_for_transfer would for prefix
    # cache skip / spec-decode draft trim (both paths consume the ALIGNED arrays)
    n_total = int(rng.integers(1, 8))
    skip = int(rng.integers(0, n_total))
    n_blocks = n_total - skip
    src_block_ptrs = rng.integers(1 << 16, 1 << 40, size=n_total, dtype=np.int64)
    dst_block_ptrs = rng.integers(1 << 16, 1 << 40, size=n_total, dtype=np.int64)
    return SimpleNamespace(
        mapper=mapper,
        self_region_bytes=self_region_bytes,
        peer_region_bytes=peer_region_bytes,
        src_block_ptrs=src_block_ptrs[skip : skip + n_blocks],
        dst_block_ptrs=dst_block_ptrs[skip : skip + n_blocks],
    )


def _legacy_tables(geo: SimpleNamespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The materialized fragment tables today's path produces: mapper.map() output."""
    pair = geo.mapper.map(
        SpecRegion(
            memory=MemRegionGroup(ptrs=geo.src_block_ptrs, bytes_per_region=geo.self_region_bytes)
        ),
        SpecRegion(
            memory=MemRegionGroup(ptrs=geo.dst_block_ptrs, bytes_per_region=geo.peer_region_bytes)
        ),
    )
    frag = pair.src.memory.bytes_per_region
    assert frag == pair.dst.memory.bytes_per_region
    sizes = np.full(pair.src.memory.ptrs.size, frag, dtype=np.int64)
    return pair.src.memory.ptrs, pair.dst.memory.ptrs, sizes


def _coverage(ptrs: np.ndarray, sizes: np.ndarray) -> set:
    return set(zip(ptrs.tolist(), sizes.tolist()))


# --------------------------------------------------------------------------- #
# NHDGatherSpec — validation and basic geometry
# --------------------------------------------------------------------------- #
class TestSpecValidation:
    def _ok(self):
        return dict(
            block_ptrs=np.array([100, 200], dtype=np.int64),
            flat_offsets=np.array([0, 8], dtype=np.int64),
            frag_bytes=8,
        )

    def test_valid_spec_geometry(self):
        s = NHDGatherSpec(**self._ok())
        assert s.n_frags_per_block == 2
        assert s.n_frags == 4
        assert s.total_bytes == 32
        assert s.paged_ptrs().tolist() == [100, 108, 200, 208]  # blocks outer, offsets inner
        assert s.staged_offsets().tolist() == [0, 8, 16, 24]

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("block_ptrs", np.array([1.0, 2.0])),  # wrong dtype
            ("block_ptrs", np.array([[1], [2]], dtype=np.int64)),  # wrong ndim
            ("flat_offsets", [0, 8]),  # not an ndarray
            ("frag_bytes", 0),
            ("frag_bytes", -8),
            ("frag_bytes", 8.0),  # not an int
        ],
    )
    def test_invalid_spec_raises(self, field, bad):
        kwargs = self._ok()
        kwargs[field] = bad
        with pytest.raises(ValueError):
            NHDGatherSpec(**kwargs)

    def test_specs_total_bytes(self):
        a = NHDGatherSpec(**self._ok())
        assert specs_total_bytes([]) == 0
        assert specs_total_bytes([a, a]) == 64

    def test_expand_empty(self):
        paged, staged, sizes = expand_specs(0x1000, [])
        assert paged.size == staged.size == sizes.size == 0


# --------------------------------------------------------------------------- #
# equivalence — structured expansion vs NHDHeadMismatchMapper.map() + Plan offsets
# --------------------------------------------------------------------------- #
class TestMapperEquivalence:
    def test_randomized_geometries(self):
        """Verify analytic plans across randomized head-mismatch geometries.

        The expansion is byte-identical to the mapper's materialized tables plus the
        bounce destination table's running staging offsets on both sides.
        """
        rng = np.random.default_rng(0xB0)
        staging_base = 0x7F0000000000
        for _ in range(200):
            geo = _random_geometry(rng)
            legacy_src, legacy_dst, sizes = _legacy_tables(geo)
            # the staging addresses today's bounce path uses: base + Plan running offsets
            plan = Plan(legacy_src, legacy_dst, sizes, int(sizes.sum()))
            legacy_staged = staging_base + plan.offsets

            src_spec = geo.mapper.src_gather_spec(geo.src_block_ptrs)
            dst_spec = geo.mapper.dst_scatter_spec(geo.dst_block_ptrs)

            # sender gather: paged source fragments -> contiguous staging
            paged, staged, exp_sizes = expand_specs(staging_base, [src_spec])
            assert np.array_equal(paged, legacy_src)
            assert np.array_equal(staged, legacy_staged)
            assert np.array_equal(exp_sizes, sizes)
            assert _coverage(paged, exp_sizes) == _coverage(legacy_src, sizes)

            # receiver scatter: contiguous staging -> paged destination fragments
            paged, staged, exp_sizes = expand_specs(staging_base, [dst_spec])
            assert np.array_equal(paged, legacy_dst)
            assert np.array_equal(staged, legacy_staged)
            assert np.array_equal(exp_sizes, sizes)
            assert _coverage(paged, exp_sizes) == _coverage(legacy_dst, sizes)

    def test_multi_section_matches_concatenated_plan(self):
        """Verify two staged pool-pair sections against one legacy plan.

        Section k's staging range starts where section k-1 ends, preserving the
        canonical ordering that keeps structured and fallback writers byte-identical.
        """
        rng = np.random.default_rng(0xB1)
        staging_base = 0x1000
        for _ in range(20):
            geos = [_random_geometry(rng), _random_geometry(rng)]
            tables = [_legacy_tables(g) for g in geos]
            legacy_src = np.concatenate([t[0] for t in tables])
            legacy_dst = np.concatenate([t[1] for t in tables])
            sizes = np.concatenate([t[2] for t in tables])
            plan = Plan(legacy_src, legacy_dst, sizes, int(sizes.sum()))

            src_specs = [g.mapper.src_gather_spec(g.src_block_ptrs) for g in geos]
            dst_specs = [g.mapper.dst_scatter_spec(g.dst_block_ptrs) for g in geos]
            assert specs_total_bytes(src_specs) == plan.total_size

            paged, staged, exp_sizes = expand_specs(staging_base, src_specs)
            assert np.array_equal(paged, legacy_src)
            assert np.array_equal(staged, staging_base + plan.offsets)
            assert np.array_equal(exp_sizes, sizes)

            paged, staged, _ = expand_specs(staging_base, dst_specs)
            assert np.array_equal(paged, legacy_dst)
            assert np.array_equal(staged, staging_base + plan.offsets)

    def test_minimax_m3_like_geometry(self):
        """Deterministic spot check: an M3-flavored ctx TP2 -> gen TP8 shape."""
        self_ri = make_rankinfo(
            tp_size=2,
            tp_rank=1,
            kv_heads_per_rank=4,
            tokens_per_block=16,
            dims_per_head=128,
            element_bytes=2,
        )
        peer_ri = make_rankinfo(
            instance_name="peer",
            tp_size=8,
            tp_rank=5,
            kv_heads_per_rank=1,
            tokens_per_block=16,
            dims_per_head=128,
            element_bytes=2,
        )
        bpth = 128 * 2
        mapper = NHDHeadMismatchMapper(
            transfer_layers=3,
            src_layer_off=1,
            peer_layer_off=0,
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_region_bytes=4 * 2 * 16 * 4 * bpth,
            peer_region_bytes=3 * 2 * 16 * 1 * bpth,
            self_pool_num_layers=4,
            peer_pool_num_layers=3,
            self_buffers_per_layer=2,
            peer_buffers_per_layer=2,
        )
        geo = SimpleNamespace(
            mapper=mapper,
            self_region_bytes=4 * 2 * 16 * 4 * bpth,
            peer_region_bytes=3 * 2 * 16 * 1 * bpth,
            src_block_ptrs=np.array([1 << 20, 5 << 20, 3 << 20], dtype=np.int64),
            dst_block_ptrs=np.array([2 << 20, 9 << 20, 4 << 20], dtype=np.int64),
        )
        legacy_src, legacy_dst, sizes = _legacy_tables(geo)
        spec = mapper.src_gather_spec(geo.src_block_ptrs)
        assert spec.frag_bytes == 1 * bpth  # min(4, 1) heads
        assert spec.n_frags_per_block == 3 * 2 * 16  # layers x buffers x tokens
        paged, _, exp_sizes = expand_specs(0, [spec])
        assert np.array_equal(paged, legacy_src)
        assert np.array_equal(exp_sizes, sizes)
        paged, _, _ = expand_specs(0, [mapper.dst_scatter_spec(geo.dst_block_ptrs)])
        assert np.array_equal(paged, legacy_dst)


# --------------------------------------------------------------------------- #
# mapper accessors — structured-staging capability flags
# --------------------------------------------------------------------------- #
class TestMapperAccessors:
    def test_supports_structured_staging_flags(self):
        assert IdentityMapper().supports_structured_staging is False
        # HND HeadMismatchMapper keeps the materialized path (few, large fragments)
        hnd = HeadMismatchMapper(
            transfer_layers=1,
            src_layer_off=0,
            peer_layer_off=0,
            self_ri=make_rankinfo(),
            peer_ri=make_rankinfo(instance_name="peer", tp_size=1, kv_heads_per_rank=4),
        )
        assert hnd.supports_structured_staging is False
        rng = np.random.default_rng(0xB2)
        assert _random_geometry(rng).mapper.supports_structured_staging is True

    def test_specs_share_mapper_offsets(self):
        rng = np.random.default_rng(0xB3)
        geo = _random_geometry(rng)
        src_spec = geo.mapper.src_gather_spec(geo.src_block_ptrs)
        dst_spec = geo.mapper.dst_scatter_spec(geo.dst_block_ptrs)
        # flat_offsets are the mapper's precomputed arrays (no copies), so a device-side
        # cache keyed on them stays valid for the mapper's lifetime
        assert src_spec.flat_offsets is geo.mapper._src_flat_offsets
        assert dst_spec.flat_offsets is geo.mapper._dst_flat_offsets
        assert src_spec.frag_bytes == dst_spec.frag_bytes == geo.mapper._bytes_cont_heads
        assert np.array_equal(src_spec.block_ptrs, geo.src_block_ptrs)
        assert src_spec.block_ptrs.dtype == np.int64

    def test_accessor_coerces_block_ptr_dtype(self):
        rng = np.random.default_rng(0xB4)
        geo = _random_geometry(rng)
        spec = geo.mapper.src_gather_spec(geo.src_block_ptrs.astype(np.uint64))
        assert spec.block_ptrs.dtype == np.int64
        assert np.array_equal(spec.block_ptrs, geo.src_block_ptrs)

    def test_pool_buffer_mapper_v2_nhd_specs_match_map(self):
        self_ri = make_rankinfo(
            instance_name="local",
            tp_size=2,
            kv_heads_per_rank=2,
            tokens_per_block=2,
            dims_per_head=2,
            element_bytes=2,
        )
        peer_ri = make_rankinfo(
            instance_name="peer",
            tp_size=1,
            kv_heads_per_rank=1,
            tokens_per_block=2,
            dims_per_head=2,
            element_bytes=2,
        )
        mapper = PoolBufferMapper(
            mappings=[
                PoolBufferMapping(0, 0, 16, 8, MapperKind.NHD),
                PoolBufferMapping(16, 8, 16, 8, MapperKind.NHD),
            ],
            self_ri=self_ri,
            peer_ri=peer_ri,
            self_region_bytes=32,
            peer_region_bytes=16,
            full_region_identity=False,
            include_sharded=True,
            include_replicated=True,
        )
        src_block_ptrs = np.array([1000, 2000], dtype=np.int64)
        dst_block_ptrs = np.array([5000, 6000], dtype=np.int64)

        pair = mapper.map(
            SpecRegion(memory=MemRegionGroup(ptrs=src_block_ptrs, bytes_per_region=32)),
            SpecRegion(memory=MemRegionGroup(ptrs=dst_block_ptrs, bytes_per_region=16)),
        )[0]
        src_spec = mapper.src_gather_spec(src_block_ptrs)
        dst_spec = mapper.dst_scatter_spec(dst_block_ptrs)

        paged, _, sizes = expand_specs(0, [src_spec])
        assert mapper.supports_structured_staging is True
        assert np.array_equal(paged, pair.src.memory.ptrs)
        assert np.array_equal(sizes, np.full(pair.src.memory.ptrs.size, 4, dtype=np.int64))
        paged, _, sizes = expand_specs(0, [dst_spec])
        assert np.array_equal(paged, pair.dst.memory.ptrs)
        assert np.array_equal(sizes, np.full(pair.dst.memory.ptrs.size, 4, dtype=np.int64))

        template = mapper.recv_scatter_template(src_block_ptrs)
        assert template.frag_bytes == 4
        assert template.writer_head_stride == 4
        assert np.array_equal(template.spec_for(0, 0, 2).flat_offsets, src_spec.flat_offsets)
        assert np.array_equal(
            template.spec_for(1, 0, 2).flat_offsets,
            src_spec.flat_offsets + template.writer_head_stride,
        )


# --------------------------------------------------------------------------- #
# host reference — gather + scatter over real host memory moves the right bytes
# --------------------------------------------------------------------------- #
class TestHostReference:
    def test_gather_scatter_roundtrip_moves_mapped_fragments(self):
        """Verify a gather/scatter round trip over host memory.

        Every mapped fragment must land at the mapper's exact destination address with
        the source bytes.
        """
        rng = np.random.default_rng(0xB5)
        for _ in range(20):
            geo = _random_geometry(rng)
            n_blocks = geo.src_block_ptrs.size
            src_pool = rng.integers(0, 256, size=n_blocks * geo.self_region_bytes, dtype=np.uint8)
            dst_pool = np.zeros(n_blocks * geo.peer_region_bytes, dtype=np.uint8)
            # rebase the random block ids onto the real buffers, keeping the block order
            slot = np.arange(n_blocks, dtype=np.int64)
            geo.src_block_ptrs = src_pool.ctypes.data + slot * geo.self_region_bytes
            geo.dst_block_ptrs = dst_pool.ctypes.data + slot * geo.peer_region_bytes

            src_spec = geo.mapper.src_gather_spec(geo.src_block_ptrs)
            staging = np.zeros(src_spec.total_bytes, dtype=np.uint8)
            base = staging.ctypes.data

            copy_expanded_host(*expand_specs(base, [src_spec]), gather=True)
            copy_expanded_host(
                *expand_specs(base, [geo.mapper.dst_scatter_spec(geo.dst_block_ptrs)]),
                gather=False,
            )

            legacy_src, legacy_dst, sizes = _legacy_tables(geo)
            src_off = legacy_src - src_pool.ctypes.data
            dst_off = legacy_dst - dst_pool.ctypes.data
            for k in range(sizes.size):
                n = int(sizes[k])
                expect = src_pool[int(src_off[k]) : int(src_off[k]) + n]
                got = dst_pool[int(dst_off[k]) : int(dst_off[k]) + n]
                assert np.array_equal(got, expect)

    def test_copy_expanded_host_length_mismatch_raises(self):
        one = np.array([0], dtype=np.int64)
        two = np.array([0, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="disagree in length"):
            copy_expanded_host(one, two, one, gather=True)


# --------------------------------------------------------------------------- #
# GPU — Triton structured kernel and the per-fragment fallback vs the host reference
# --------------------------------------------------------------------------- #
def _make_gpu_mapper(
    dims_per_head,
    element_bytes,
    *,
    layers=3,
    heads_self=2,
    heads_peer=1,
    tokens_per_block=4,
    buffers=2,
):
    """One NHD head-mismatch mapper plus its region sizes for the GPU tests."""
    bpth = int(dims_per_head * element_bytes)
    self_ri = make_rankinfo(
        tp_size=1,
        kv_heads_per_rank=heads_self,
        tokens_per_block=tokens_per_block,
        dims_per_head=dims_per_head,
        element_bytes=element_bytes,
    )
    peer_ri = make_rankinfo(
        instance_name="peer",
        tp_size=2,
        tp_rank=1,
        kv_heads_per_rank=heads_peer,
        tokens_per_block=tokens_per_block,
        dims_per_head=dims_per_head,
        element_bytes=element_bytes,
    )
    self_region = layers * buffers * tokens_per_block * heads_self * bpth
    peer_region = layers * buffers * tokens_per_block * heads_peer * bpth
    mapper = NHDHeadMismatchMapper(
        transfer_layers=layers,
        src_layer_off=0,
        peer_layer_off=0,
        self_ri=self_ri,
        peer_ri=peer_ri,
        self_region_bytes=self_region,
        peer_region_bytes=peer_region,
        self_pool_num_layers=layers,
        peer_pool_num_layers=layers,
        self_buffers_per_layer=buffers,
        peer_buffers_per_layer=buffers,
    )
    return SimpleNamespace(mapper=mapper, self_region=self_region, peer_region=peer_region)


def _run_gpu_roundtrip(torch, geos, n_blocks, rng, *, expect_kernel):
    """Run one structured GPU gather/scatter round trip.

    Gather every geometry's paged fragments into one staging region, scatter them back,
    compare against the NumPy/memmove host reference, and verify the expected launch path.
    """
    src_hosts = [rng.integers(0, 256, size=n_blocks * g.self_region, dtype=np.uint8) for g in geos]
    src_devs = [torch.from_numpy(h).cuda() for h in src_hosts]
    dst_devs = [
        torch.zeros(n_blocks * g.peer_region, dtype=torch.uint8, device="cuda") for g in geos
    ]

    def block_ptrs(base: int, region: int) -> np.ndarray:
        return base + np.arange(n_blocks, dtype=np.int64) * region

    src_specs = [
        g.mapper.src_gather_spec(block_ptrs(d.data_ptr(), g.self_region))
        for g, d in zip(geos, src_devs)
    ]
    dst_specs = [
        g.mapper.dst_scatter_spec(block_ptrs(d.data_ptr(), g.peer_region))
        for g, d in zip(geos, dst_devs)
    ]
    total = specs_total_bytes(src_specs)
    staging_dev = torch.zeros(64 + total, dtype=torch.uint8, device="cuda")
    staging_base = staging_dev.data_ptr() + 64  # offset: catch base-handling bugs

    stream = torch.cuda.current_stream()
    took_kernel = gather_structured(staging_base, src_specs, stream=stream.cuda_stream)
    assert took_kernel is expect_kernel  # the path ids/params claim must be honest
    # gather and scatter on the same stream: sync between them so the second call's
    # pinned-metadata refill cannot race the first call's H2D copy (see the buffer
    # reuse contract in gather_scatter._launch_structured_sections)
    stream.synchronize()
    took_kernel = scatter_structured(staging_base, dst_specs, stream=stream.cuda_stream)
    assert took_kernel is expect_kernel
    stream.synchronize()

    # host ground truth over mirrored buffers via the numpy/memmove reference
    staging_host = np.zeros(total, dtype=np.uint8)
    dst_hosts = [np.zeros(n_blocks * g.peer_region, dtype=np.uint8) for g in geos]
    host_src_specs = [
        g.mapper.src_gather_spec(block_ptrs(h.ctypes.data, g.self_region))
        for g, h in zip(geos, src_hosts)
    ]
    copy_expanded_host(*expand_specs(staging_host.ctypes.data, host_src_specs), gather=True)
    host_dst_specs = [
        g.mapper.dst_scatter_spec(block_ptrs(h.ctypes.data, g.peer_region))
        for g, h in zip(geos, dst_hosts)
    ]
    copy_expanded_host(*expand_specs(staging_host.ctypes.data, host_dst_specs), gather=False)

    got_staging = staging_dev[64 : 64 + total].cpu().numpy()
    assert np.array_equal(got_staging, staging_host)
    for dst_dev, dst_host in zip(dst_devs, dst_hosts):
        assert np.array_equal(dst_dev.cpu().numpy(), dst_host)


def _gpu_setup(expect_kernel):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if expect_kernel and not _HAVE_TRITON:
        pytest.skip("kernel-path case needs Triton; the fallback is covered separately")
    return torch


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dims_per_head,element_bytes,expect_kernel",
    # 16B-multiple frags take the Triton kernel; 4B frags take the memcpy fallback
    [(64, 2.0, True), (128, 0.5, True), (2, 2.0, False)],
    ids=["bf16_kernel", "nvfp4_kernel", "tiny_frag_fallback"],
)
def test_gpu_structured_matches_host_reference(dims_per_head, element_bytes, expect_kernel):
    torch = _gpu_setup(expect_kernel)
    rng = np.random.default_rng(0xB6)
    geo = _make_gpu_mapper(dims_per_head, element_bytes)
    _run_gpu_roundtrip(
        torch, [geo], n_blocks=5, rng=rng, expect_kernel=expect_kernel and _HAVE_TRITON
    )


@pytest.mark.cuda
def test_gpu_multi_section_matches_host_reference():
    """Verify a multi-section GPU round trip against the host reference.

    The two sections use different fragment sizes and fragment counts, exercising the
    metadata packing loop and per-section staging-base accumulation on device.
    """
    torch = _gpu_setup(expect_kernel=True)
    rng = np.random.default_rng(0xB7)
    geos = [
        _make_gpu_mapper(64, 2.0),  # frag 128 B, 3 layers x 2 buffers x 4 tokens
        _make_gpu_mapper(128, 2.0, layers=2, heads_self=4, heads_peer=2, tokens_per_block=8),
    ]
    _run_gpu_roundtrip(torch, geos, n_blocks=3, rng=rng, expect_kernel=True)
