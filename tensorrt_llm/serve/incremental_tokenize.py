# SPDX-License-Identifier: Apache-2.0
"""Exact incremental tokenization for multi-turn chat requests (prototype for tensorrt_llm.serve).

Motivation: agentic/multi-turn traffic re-sends the whole conversation every turn (100k-600k tokens, 97% of which
is a byte-identical prefix of the previous turn). Tokenizing the full rendered prompt costs 2-3 ms per 1k tokens
on the API server (300 ms - 1.5 s per request) and is the largest context-linear component of TTFT.

Exactness argument: HF added/special tokens (chat-role markers) are matched before BPE and never merge with
neighbouring text, so the rendered prompt can be cut right *before* any added-token occurrence and the pieces
encoded independently; concatenating the ids equals the serial encode. A conversation's new prompt shares a long
prefix with its previous prompt; we reuse the cached ids of every whole part inside the common prefix and encode
only the tail (in parallel with the Rust ``encode_batch``). Cold turns fall back to the parallel encode.
"""
from __future__ import annotations

import bisect
import collections
import os
import re
import threading
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _env_flag(name: str) -> bool:
    v = os.environ.get(name)
    if not v:
        return False
    if v.lower() in ("1", "true", "on", "yes"):
        return True
    return os.path.exists(v)  # path-valued: enabled iff the file exists (runtime toggle for A/B tests)


def incremental_tokenize_enabled() -> bool:
    return _env_flag("TLLM_INCREMENTAL_TOKENIZE")


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the common prefix of a and b using memcmp-speed slice compares (O(log n) compares)."""
    n = min(len(a), len(b))
    if n == 0 or a[0] != b[0]:
        return 0
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # invariant: a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


class _ConversationEntry:
    __slots__ = ("text", "part_starts", "part_ids", "n_tokens")

    def __init__(self, text: str, part_starts: List[int], part_ids: List[array]):
        self.text = text
        self.part_starts = part_starts  # char offset where part i starts (part_starts[0] == 0)
        self.part_ids = part_ids
        self.n_tokens = sum(len(p) for p in part_ids)


class IncrementalTokenizer:
    """Conversation-keyed exact incremental encoder over an HF fast tokenizer (or TRT-LLM's wrapper)."""

    def __init__(self, tokenizer, max_conversations: int = 128, min_chars: int = 16384):
        self.hf = getattr(tokenizer, "tokenizer", tokenizer)
        self.rt = getattr(self.hf, "_tokenizer", None) or getattr(self.hf, "backend_tokenizer", None)
        if self.rt is None or not hasattr(self.rt, "encode_batch"):
            raise ValueError("IncrementalTokenizer needs an HF fast tokenizer")
        markers = set()
        try:
            for tok in self.rt.get_added_tokens_decoder().values():
                if getattr(tok, "lstrip", False) or getattr(tok, "single_word", False):
                    continue
                if tok.content:
                    markers.add(tok.content)
        except Exception:
            pass
        for attr in ("all_special_tokens_extended", "all_special_tokens"):
            for t in (getattr(self.hf, attr, None) or []):
                if isinstance(t, str) and t:
                    markers.add(t)
        # Prefer the markers the chat template can actually emit (role/turn delimiters): scanning the
        # template source once keeps the per-request presence scan cheap even for tokenizers with
        # >1000 added tokens (DeepSeek). Any omitted added token only means fewer split points; the
        # result stays exact because every added token is still a hard boundary for the encoder.
        template = getattr(self.hf, "chat_template", None)
        if isinstance(template, dict):
            template = "\n".join(str(v) for v in template.values())
        if isinstance(template, str) and template:
            in_template = {m for m in markers if m in template}
            specials = {t for t in (getattr(self.hf, "all_special_tokens", None) or []) if isinstance(t, str)}
            chosen = in_template | specials
            if chosen:
                markers = chosen
        self.markers: List[str] = sorted(markers, key=len, reverse=True)
        self._regex_cache: Dict[tuple, Any] = {}
        self._cache: "collections.OrderedDict[str, _ConversationEntry]" = collections.OrderedDict()
        self._lock = threading.Lock()
        self.max_conversations = max_conversations
        self.min_chars = min_chars
        self.stats = collections.Counter()

    # ---- splitting -------------------------------------------------------------------------------------
    def _regex_for(self, text: str):
        present = tuple(m for m in self.markers if m in text)
        if not present:
            return None
        pat = self._regex_cache.get(present)
        if pat is None:
            pat = re.compile("(?=" + "|".join(re.escape(m) for m in present) + ")")
            if len(self._regex_cache) < 256:
                self._regex_cache[present] = pat
        return pat

    def _split_starts(self, text: str, offset: int = 0) -> List[int]:
        """Char offsets (absolute, given text starts at ``offset``) where parts begin; first is ``offset``."""
        pat = self._regex_for(text)
        starts = [offset]
        if pat is None:
            return starts
        for m in pat.finditer(text):
            p = m.start()
            if p != 0:
                starts.append(offset + p)
        return starts

    def _encode_pieces(self, pieces: Sequence[str]) -> List[array]:
        if not pieces:
            return []
        return [array("i", e.ids) for e in self.rt.encode_batch(list(pieces), add_special_tokens=False)]

    # ---- public API ------------------------------------------------------------------------------------
    def encode(self, conversation_id: Optional[str], text: str) -> List[int]:
        """Exact token ids of ``text`` (no special tokens added), reusing the conversation's previous prompt."""
        if len(text) < self.min_chars:
            self.stats["small_serial"] += 1
            return self.hf.encode(text, add_special_tokens=False)
        prev = None
        if conversation_id:
            with self._lock:
                prev = self._cache.get(conversation_id)
        reuse_parts = 0
        if prev is not None:
            lcp = _common_prefix_len(prev.text, text)
            # last part boundary strictly inside the common prefix; never reuse prev's final part unless the
            # new text continues past it AND the next char in the new text starts a marker (rare) -> keep simple:
            j = bisect.bisect_right(prev.part_starts, lcp) - 1  # parts 0..j-1 end at or before lcp
            # part j starts <= lcp; it is fully inside the prefix only if the next part starts <= lcp
            reuse_parts = max(0, min(j, len(prev.part_starts) - 1))
            # ensure every reused part lies entirely within the common prefix
            while reuse_parts > 0 and not (reuse_parts < len(prev.part_starts) and prev.part_starts[reuse_parts] <= lcp):
                reuse_parts -= 1
        if reuse_parts > 0:
            tail_offset = prev.part_starts[reuse_parts]
            tail = text[tail_offset:]
            tail_starts = self._split_starts(tail, tail_offset) if tail else []
            starts = prev.part_starts[:reuse_parts] + tail_starts
            pieces = [text[starts[i]:(starts[i + 1] if i + 1 < len(starts) else len(text))] for i in range(reuse_parts, len(starts))]
            new_ids = self._encode_pieces(pieces)
            part_ids = prev.part_ids[:reuse_parts] + new_ids
            self.stats["hits"] += 1
            self.stats["reused_tokens"] += sum(len(p) for p in prev.part_ids[:reuse_parts])
            self.stats["encoded_tokens"] += sum(len(p) for p in new_ids)
        else:
            starts = self._split_starts(text)
            pieces = [text[starts[i]:(starts[i + 1] if i + 1 < len(starts) else len(text))] for i in range(len(starts))]
            part_ids = self._encode_pieces(pieces)
            self.stats["misses"] += 1
            self.stats["encoded_tokens"] += sum(len(p) for p in part_ids)
        if conversation_id:
            entry = _ConversationEntry(text, starts, part_ids)
            with self._lock:
                self._cache[conversation_id] = entry
                self._cache.move_to_end(conversation_id)
                while len(self._cache) > self.max_conversations:
                    self._cache.popitem(last=False)
        out: List[int] = []
        for p in part_ids:
            out.extend(p)
        return out


_singleton: Optional[IncrementalTokenizer] = None
_singleton_lock = threading.Lock()


def get_incremental_tokenizer(tokenizer) -> Optional[IncrementalTokenizer]:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                try:
                    _singleton = IncrementalTokenizer(
                        tokenizer,
                        max_conversations=int(os.environ.get("TLLM_INCREMENTAL_TOKENIZE_MAX_CONVERSATIONS", "128")),
                    )
                except Exception:
                    return None
    return _singleton
