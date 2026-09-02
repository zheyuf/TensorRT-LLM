# SPDX-License-Identifier: Apache-2.0
"""Exactness tests for tensorrt_llm.serve.incremental_tokenize (parallel split encode + conversation-keyed
incremental encode). Uses a small BPE tokenizer built in-test (no network / no checkpoint); if
TLLM_TEST_TOKENIZER_DIR points at an HF tokenizer directory it is exercised as well."""
import os
import random

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.serve.incremental_tokenize import IncrementalTokenizer

ROLE_MARKERS = ["<|im_start|>", "<|im_end|>", "<|tool|>"]


def _build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=800, special_tokens=["<unk>"] + ROLE_MARKERS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = ["def f(x): return x + 1\n" * 50, "the quick brown fox jumps over the lazy dog " * 50,
              "SELECT * FROM t WHERE a = 1;\n" * 30, "你好世界 " * 40]
    tok.train_from_iterator(corpus, trainer)
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                 additional_special_tokens=ROLE_MARKERS)
    hf.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return hf


def _tokenizers():
    yield "tiny-bpe", _build_tiny_tokenizer()
    d = os.environ.get("TLLM_TEST_TOKENIZER_DIR")
    if d:
        from transformers import AutoTokenizer
        yield "env", AutoTokenizer.from_pretrained(d, trust_remote_code=True)


def _random_text(rng, n_words):
    words = ["alpha", "beta", "gamma", "delta", "x=1", "return", "你好", "SELECT", "<|im_start|>", "  ", "\n", "}", "{"]
    return " ".join(rng.choice(words) for _ in range(n_words))


@pytest.mark.parametrize("name,hf", list(_tokenizers()))
def test_parallel_split_encode_is_exact(name, hf):
    rng = random.Random(0)
    inc = IncrementalTokenizer(hf, min_chars=64)
    for n in (200, 2000, 20000):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(8):
            msgs.append({"role": "user", "content": _random_text(rng, n // 8)})
            msgs.append({"role": "assistant", "content": "ok"})
        text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert inc.encode(None, text) == hf.encode(text, add_special_tokens=False)
    # markers glued to text, unicode, and no-marker text
    weird = ("<|im_start|>user\n<|im_end|>" * 20 + "def f(x): return x<|tool|> <|im_start|>ai\n" + "α<|im_start|>" * 10) * 20
    assert inc.encode(None, weird) == hf.encode(weird, add_special_tokens=False)
    plain = "no markers here " * 500
    assert inc.encode(None, plain) == hf.encode(plain, add_special_tokens=False)


@pytest.mark.parametrize("name,hf", list(_tokenizers()))
def test_incremental_conversation_is_exact(name, hf):
    rng = random.Random(1)
    inc = IncrementalTokenizer(hf, min_chars=64, max_conversations=8)
    convs = {f"c{i}": [{"role": "system", "content": f"session {i}"}] for i in range(6)}
    for turn in range(12):
        for cid, msgs in convs.items():
            msgs.append({"role": "user", "content": _random_text(rng, rng.choice([50, 300, 1500]))})
            text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            assert inc.encode(cid, text) == hf.encode(text, add_special_tokens=False), (cid, turn)
            msgs.append({"role": "assistant", "content": _random_text(rng, 20)})
    assert inc.stats["hits"] > 0
    # edited history (divergence in the middle) and truncated history must stay exact
    msgs = convs["c0"]
    edited = list(msgs)
    edited[1] = {"role": "user", "content": "EDITED " + edited[1]["content"]}
    text = hf.apply_chat_template(edited, tokenize=False, add_generation_prompt=True)
    assert inc.encode("c0", text) == hf.encode(text, add_special_tokens=False)
    shorter = hf.apply_chat_template(msgs[:3], tokenize=False, add_generation_prompt=True)
    assert inc.encode("c0", shorter) == hf.encode(shorter, add_special_tokens=False)
    assert len(inc._cache) <= 8  # LRU bound honoured
    # a smaller LRU than the number of live conversations must still be exact (only the hit rate drops)
    small = IncrementalTokenizer(hf, min_chars=64, max_conversations=2)
    for cid, msgs in convs.items():
        text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert small.encode(cid, text) == hf.encode(text, add_special_tokens=False)
    assert len(small._cache) <= 2
