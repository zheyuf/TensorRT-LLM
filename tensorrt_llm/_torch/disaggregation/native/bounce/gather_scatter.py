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
"""Device-to-device gather and scatter for KV bounce. It coalesces the scattered fragments into one
contiguous buffer, and the inverse, in a single batched kernel launch, and falls back to a
per-fragment copy loop when Triton or the GPU is unavailable.

Two entry-point families share the kernels' launch machinery:
- gather_contiguous / scatter_contiguous consume a materialized fragment table (Plan);
- gather_structured / scatter_structured consume analytic NHD plans (nhd_plan.NHDGatherSpec)
  whose fragment addresses the kernel computes from geometry instead of pointer tables.
"""

import threading
from dataclasses import dataclass

import numpy as np

from .nhd_plan import NHDGatherSpec

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.runtime.generation import CUASSERT

try:
    import torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # keep importable without GPU or Triton at import time
    _HAVE_TRITON = False

_D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

# copy in 128-bit elements, the widest per-thread transaction, as two 64-bit words
_ELEM_BYTES = 16

# elements per program; an architecture-agnostic default of about 64 KiB of copy
_BLOCK = 4096
# fixed rather than autotuned, to avoid recompiling per shape on the hot path
_NUM_WARPS = 4
_CHUNK_BYTES = _BLOCK * _ELEM_BYTES


@dataclass
class Plan:
    """One transfer's coalescing plan: the source and destination fragment addresses, their sizes,
    and the total. Both sides walk the fragments in the same order, and the offsets are the running
    sum of the sizes."""

    src_ptrs: np.ndarray  # source fragment addresses
    dst_ptrs: np.ndarray  # destination fragment addresses
    sizes: np.ndarray  # bytes per fragment
    total_size: int  # total bytes; must not exceed the bounce buffer capacity

    @property
    def offsets(self) -> np.ndarray:
        if self.sizes.size == 0:
            return np.zeros(0, dtype=np.int64)
        off = np.empty(self.sizes.size, dtype=np.int64)
        off[0] = 0
        np.cumsum(self.sizes[:-1], out=off[1:])
        return off


if _HAVE_TRITON:

    @triton.jit
    def _batched_copy_kernel(
        dst_ptrs_ptr,  # destination address per fragment
        src_ptrs_ptr,  # source address per fragment
        nvec_ptr,  # 128-bit element count per fragment
        n_frags,  # number of fragments
        BLOCK: tl.constexpr,  # elements copied per program
    ):
        # each program copies one block-sized slice of one fragment
        frag = tl.program_id(0)
        chunk = tl.program_id(1)
        if frag >= n_frags:
            return

        dst_addr = tl.load(dst_ptrs_ptr + frag)
        src_addr = tl.load(src_ptrs_ptr + frag)
        nvec = tl.load(nvec_ptr + frag)

        vec = chunk * BLOCK + tl.arange(0, BLOCK)  # element index within the fragment
        mask = vec < nvec  # masks off the fragment's tail

        # each row is two 64-bit words the compiler coalesces into one 128-bit access
        i64 = vec[:, None] * 2 + tl.arange(0, 2)[None, :]
        m2 = mask[:, None]
        src_i64 = src_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))
        dst_i64 = dst_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))

        vals = tl.load(src_i64 + i64, mask=m2)
        tl.store(dst_i64 + i64, vals, mask=m2)

    @triton.jit
    def _structured_copy_kernel(
        block_ptrs_ptr,  # absolute paged block base address per block
        flat_offsets_ptr,  # byte offset of each fragment within one block
        staging_base_ptr,  # one int64: base address of this section's staging range
        n_off,  # fragments per block
        nvec,  # 128-bit element count per fragment (uniform)
        n_frags,  # number of fragments (n_blocks * n_off)
        ELEM_BYTES: tl.constexpr,  # bytes copied by one vector element
        GATHER: tl.constexpr,  # paged -> staging when True, the inverse when False
        BLOCK: tl.constexpr,  # elements copied per program
    ):
        # The structured sibling of _batched_copy_kernel: fragment addresses are computed
        # from geometry (n_blocks + n_off metadata entries) instead of loaded from
        # per-fragment tables (3 entries per fragment).
        frag = tl.program_id(0)
        chunk = tl.program_id(1)
        if frag >= n_frags:
            return

        block = frag // n_off
        off = frag % n_off
        paged_addr = tl.load(block_ptrs_ptr + block) + tl.load(flat_offsets_ptr + off)
        # fragments are staged contiguously in emission order: blocks outer, offsets inner
        # ELEM_BYTES is an explicit constexpr for compatibility across Triton versions.
        staged_addr = tl.load(staging_base_ptr) + frag.to(tl.int64) * nvec * ELEM_BYTES

        vec = chunk * BLOCK + tl.arange(0, BLOCK)  # element index within the fragment
        mask = vec < nvec  # masks off the fragment's tail

        # each row is two 64-bit words the compiler coalesces into one 128-bit access
        i64 = vec[:, None] * 2 + tl.arange(0, 2)[None, :]
        m2 = mask[:, None]
        paged_i64 = paged_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))
        staged_i64 = staged_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))

        if GATHER:
            vals = tl.load(paged_i64 + i64, mask=m2)
            tl.store(staged_i64 + i64, vals, mask=m2)
        else:
            vals = tl.load(staged_i64 + i64, mask=m2)
            tl.store(paged_i64 + i64, vals, mask=m2)


def _uniform_nelem(sizes: np.ndarray):
    """Return the element counts, the maximum, and whether every fragment is 16-byte aligned, which
    the 128-bit copy requires."""
    if sizes.size == 0:
        return None, 0, False
    if not np.all((sizes % _ELEM_BYTES) == 0):
        return None, 0, False
    nvec = (sizes // _ELEM_BYTES).astype(np.int64)
    return nvec, int(nvec.max()), True


# Reusable pinned staging for the metadata copy, one per stream, so the copy is a true
# async transfer.
#
# Race fixed here (see the PR description, "pinned metadata buffer race"): the pinned host buffer is
# refilled synchronously on the host before each launch, but the H2D copy that consumes
# it is asynchronous. Two gather/scatter calls issued back to back on the same stream
# without a host-side wait (the sender's structured+contiguous pair, and the scatter
# worker's per-writer fan-in loop — a pre-existing hazard) could overwrite metadata the
# first copy has not yet read. Each per-stream entry therefore carries a CUDA event
# recorded right after its H2D copy; the next fill waits on it (event-per-fill). The
# DEVICE buffer needs no guard: it is only written by H2D copies and read by kernels on
# the same stream, so stream ordering already serializes it.
_meta_lock = threading.Lock()
_meta_buffers = {}  # stream handle -> _MetaEntry


class _MetaEntry:
    """One stream's pinned+device metadata buffers plus the fill-drain event.

    Thread contract: wait_previous_fill -> fill pinned -> record_fill is not atomic
    on its own; callers must serialize all uses of one stream handle's entry
    externally (the send-stream lock / the single scatter worker thread)."""

    __slots__ = ("pinned", "device", "capacity", "fill_event")

    def __init__(self, pinned, device, capacity: int):
        self.pinned = pinned
        self.device = device
        self.capacity = capacity
        self.fill_event = None  # created lazily on the first fill

    def wait_previous_fill(self) -> None:
        """Block (host-side) until the previous fill's H2D copy has executed, so the
        pinned buffer is safe to overwrite. Call before writing into ``pinned``."""
        if self.fill_event is not None:
            self.fill_event.synchronize()

    def record_fill(self, ext_stream) -> None:
        """Mark the point on the stream after which the pinned buffer is reusable.
        Call right after issuing the H2D copy of ``pinned``."""
        if self.fill_event is None:
            self.fill_event = torch.cuda.Event()
        self.fill_event.record(ext_stream)


def _get_meta_buffers(stream_handle: int, need: int, dev) -> "_MetaEntry":
    """Return the stream's metadata entry, large enough for the request, growing it
    under a lock. Growth drains the old entry's in-flight fill first so the old pinned
    tensor is never freed while an H2D copy may still read it."""
    with _meta_lock:
        ent = _meta_buffers.get(stream_handle)
        if ent is None or ent.capacity < need:
            if ent is not None:
                ent.wait_previous_fill()
            new_cap = max(need, (ent.capacity * 2 if ent else 0), 4096)
            pinned = torch.empty(new_cap, dtype=torch.int64, pin_memory=prefer_pinned())
            devt = torch.empty(new_cap, dtype=torch.int64, device=dev)
            ent = _MetaEntry(pinned, devt, new_cap)
            _meta_buffers[stream_handle] = ent
        return ent


def _launch_batched_copy(
    dst_addrs: np.ndarray, src_addrs: np.ndarray, sizes: np.ndarray, stream
) -> bool:
    """Run the single batched copy on the stream. Returns False when the caller must use the loop
    fallback."""
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False

    n = int(src_addrs.size)
    if n == 0:
        return True  # nothing to copy, trivially done

    nvec, max_nvec, ok = _uniform_nelem(sizes)
    if not ok or max_nvec == 0:
        return False

    dev = torch.device("cuda", torch.cuda.current_device())
    # pack all three address arrays into one pinned buffer and one async copy, so the kernel is
    # ordered after it
    stream_handle = int(stream)
    entry = _get_meta_buffers(stream_handle, 3 * n, dev)
    # wait for the previous fill's H2D copy before overwriting the pinned buffer (pinned-buffer race fix)
    entry.wait_previous_fill()
    pinned, devt = entry.pinned, entry.device
    host = pinned.numpy()
    host[:n] = dst_addrs
    host[n : 2 * n] = src_addrs
    host[2 * n : 3 * n] = nvec

    n_chunks = triton.cdiv(max_nvec, _BLOCK)
    grid = (n, n_chunks)

    ext_stream = torch.cuda.ExternalStream(stream_handle)
    with torch.cuda.stream(ext_stream):
        devt[: 3 * n].copy_(pinned[: 3 * n], non_blocking=True)
        entry.record_fill(ext_stream)  # pinned buffer reusable once this point drains
        _batched_copy_kernel[grid](
            devt[:n],
            devt[n : 2 * n],
            devt[2 * n : 3 * n],
            n,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return True


def _copy_frags(pairs, sizes: np.ndarray, stream) -> None:
    """Fallback used when the batched copy is unavailable: one async copy per fragment."""
    # strict zip: a length mismatch must fail fast rather than silently drop part of the copy
    for (dst, src), n in zip(pairs, sizes, strict=True):
        CUASSERT(cudart.cudaMemcpyAsync(int(dst), int(src), int(n), _D2D, stream))


def gather_contiguous(
    dst_base: int,
    src_ptrs: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    *,
    stream,
) -> None:
    """Gather each source fragment into its place in the contiguous buffer, asynchronously. The
    caller syncs before issuing the write."""
    src_addrs = np.asarray(src_ptrs, dtype=np.int64)
    dst_addrs = np.int64(dst_base) + np.asarray(offsets, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)

    if _launch_batched_copy(dst_addrs, src_addrs, sizes, stream):
        return

    _copy_frags(
        ((int(dst_addrs[k]), int(src_addrs[k])) for k in range(src_addrs.size)),
        sizes,
        stream,
    )


def scatter_contiguous(
    src_base: int,
    dst_ptrs: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    *,
    stream,
) -> None:
    """The inverse of gather: scatter each piece of the contiguous buffer back to its destination
    fragment, asynchronously. The caller syncs before signaling completion."""
    dst_addrs = np.asarray(dst_ptrs, dtype=np.int64)
    src_addrs = np.int64(src_base) + np.asarray(offsets, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)

    if _launch_batched_copy(dst_addrs, src_addrs, sizes, stream):
        return

    _copy_frags(
        ((int(dst_addrs[k]), int(src_addrs[k])) for k in range(dst_addrs.size)),
        sizes,
        stream,
    )


def _launch_structured_sections(
    specs: list[NHDGatherSpec], staging_bases: list[int], *, gather: bool, stream
) -> bool:
    """Run every spec section as a structured kernel launch on the stream. Returns False when
    the caller must use the loop fallback; all-or-nothing so the pinned metadata buffer is
    filled once."""
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False
    # The 128-bit copy path requires both the fragment SIZE and every source/destination
    # ADDRESS to be 16-byte aligned. Per fragment those addresses are
    # block_ptr + flat_offset (paged side) and section_base + frag * frag_bytes (staging
    # side); block_ptrs come from pool bases, which are page-aligned, so flat_offsets
    # alignment suffices for the paged side. Staging slot bases are 512-aligned
    # (SlotAllocator._ALIGN) and section bases advance by total_bytes — a 16-byte
    # multiple once frag_bytes is — but check each section base directly rather than
    # rely on that invariant.
    for spec, base in zip(specs, staging_bases, strict=True):
        if (
            spec.frag_bytes % _ELEM_BYTES != 0
            or base % _ELEM_BYTES != 0
            or not (spec.flat_offsets % _ELEM_BYTES == 0).all()
        ):
            return False

    dev = torch.device("cuda", torch.cuda.current_device())
    stream_handle = int(stream)
    # pack every section's metadata into one pinned buffer and one async copy, so all the
    # kernels are ordered after it: [block_ptrs, flat_offsets, staging_base] per section
    need = sum(spec.block_ptrs.size + spec.flat_offsets.size + 1 for spec in specs)
    entry = _get_meta_buffers(stream_handle, need, dev)
    # wait for the previous fill's H2D copy before overwriting the pinned buffer (pinned-buffer race fix)
    entry.wait_previous_fill()
    pinned, devt = entry.pinned, entry.device
    host = pinned.numpy()
    pos = 0
    slices: list[tuple[int, int, int]] = []  # each section's metadata start positions
    for spec, base in zip(specs, staging_bases, strict=True):
        nb, no = int(spec.block_ptrs.size), int(spec.flat_offsets.size)
        host[pos : pos + nb] = spec.block_ptrs
        host[pos + nb : pos + nb + no] = spec.flat_offsets
        host[pos + nb + no] = base
        slices.append((pos, pos + nb, pos + nb + no))
        pos += nb + no + 1

    ext_stream = torch.cuda.ExternalStream(stream_handle)
    with torch.cuda.stream(ext_stream):
        devt[:need].copy_(pinned[:need], non_blocking=True)
        entry.record_fill(ext_stream)  # pinned buffer reusable once this point drains
        for spec, (block_start, offset_start, staging_start) in zip(specs, slices, strict=True):
            if spec.n_frags == 0:
                continue  # an empty section stages nothing; a zero-sized grid cannot launch
            nvec = spec.frag_bytes // _ELEM_BYTES
            grid = (spec.n_frags, triton.cdiv(nvec, _BLOCK))
            _structured_copy_kernel[grid](
                devt[block_start:offset_start],
                devt[offset_start:staging_start],
                devt[staging_start : staging_start + 1],
                spec.n_frags_per_block,
                nvec,
                spec.n_frags,
                ELEM_BYTES=_ELEM_BYTES,
                GATHER=gather,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
    return True


def _copy_spec_frags(spec: NHDGatherSpec, staging_base: int, *, gather: bool, stream) -> None:
    """Fallback used when the structured kernel is unavailable: one async copy per fragment,
    expanding the spec lazily so the host never materializes the fragment tables."""
    frag_bytes = spec.frag_bytes
    staged = int(staging_base)
    for block_ptr in spec.block_ptrs.tolist():
        for off in spec.flat_offsets.tolist():
            paged = block_ptr + off
            dst, src = (staged, paged) if gather else (paged, staged)
            CUASSERT(cudart.cudaMemcpyAsync(dst, src, frag_bytes, _D2D, stream))
            staged += frag_bytes


def _structured_copy(
    staging_base: int, specs: list[NHDGatherSpec], *, gather: bool, stream
) -> bool:
    """Copy every spec section between its paged fragments and its contiguous staging range,
    sections back to back from staging_base, in the canonical order the specs define.
    Returns True when the structured kernel path ran, False when the per-fragment fallback
    did (both are byte-identical; the return value makes the choice observable to tests)."""
    staging_bases: list[int] = []
    base = int(staging_base)
    for spec in specs:
        staging_bases.append(base)
        base += spec.total_bytes

    if _launch_structured_sections(specs, staging_bases, gather=gather, stream=stream):
        return True

    for spec, section_base in zip(specs, staging_bases, strict=True):
        _copy_spec_frags(spec, section_base, gather=gather, stream=stream)
    return False


def gather_structured(dst_base: int, specs: list[NHDGatherSpec], *, stream) -> bool:
    """Gather each spec's paged fragments into its contiguous staging section, asynchronously.
    The caller syncs before issuing the write. Byte-identical to gather_contiguous over the
    expanded plan (nhd_plan.expand_specs), without materializing it. Returns True when the
    structured kernel path ran, False when the per-fragment fallback did."""
    return _structured_copy(dst_base, specs, gather=True, stream=stream)


def scatter_structured(src_base: int, specs: list[NHDGatherSpec], *, stream) -> bool:
    """The inverse of gather_structured: scatter each staging section back to its spec's paged
    fragments, asynchronously. The caller syncs before signaling completion. Returns True when
    the structured kernel path ran, False when the per-fragment fallback did."""
    return _structured_copy(src_base, specs, gather=False, stream=stream)
