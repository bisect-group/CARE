"""Tests of the frozen-encoder cache: keys, fingerprints, shards and the exactness of the cached vectors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)

import cache_frozen
from cache_frozen import Store, encode, split_entities, text_key
from common import DEFAULT_OUT_ROOT, train_args
from model_utils import PROTEIN_HUB_ID, load_reaction_tokenizer
from testing import TINY_TEXT_VOCAB

CPU = torch.device("cpu")
FINGERPRINT = {"model": "m", "recipe": "r"}


def test_key_is_stable_and_distinct():
    assert text_key("MKV") == text_key("MKV") != text_key("MKVA")


def test_store_roundtrip_and_missing(tmp_path):
    store = Store(tmp_path, "protein", FINGERPRINT)
    texts = ["AAA", "CC", "D"]
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    store.write(texts, vectors)
    assert not list(store.dir.glob(".*tmp"))
    assert store.missing([*texts, "EE"]) == ["EE"]
    assert np.array_equal(store.load(["D", "AAA"]), vectors[[2, 0]])
    with pytest.raises(KeyError):
        store.load(["EE"])


def test_stale_fingerprint_is_rejected(tmp_path):
    Store(tmp_path, "protein", FINGERPRINT)
    Store(tmp_path, "protein", FINGERPRINT)
    with pytest.raises(RuntimeError, match="other settings"):
        Store(tmp_path, "protein", {**FINGERPRINT, "recipe": "other"})


def test_shards_split_the_work_without_overlap(tmp_path):
    store = Store(tmp_path, "text", FINGERPRINT)
    texts = [f"text {i}" for i in range(200)]
    parts = [store.missing(texts, shard, 3) for shard in range(3)]
    assert sorted(sum(parts, [])) == sorted(texts)
    assert all(a == sorted(a, key=lambda t: (len(t), t)) for a in parts)


def test_split_entities_reads_the_tables(tmp_path):
    for name, text in {
        "protein2EC.csv": "Sequence,EC number\nAAA,1.1.1.1\n",
        "reference_train.csv": "Sequence,EC number\nCCC,1.1.1.1\nAAA,1.1.1.2\n",
        "reaction_train.csv": "Reaction,EC number\nC>>O,1.1.1.1\n",
        "test_queries.csv": "Reaction,EC number\nCC>>O,1.1.1.1\n",
        "text2EC.csv": "EC number,Text\n1.1.1.1,oxidoreductase\n",
    }.items():
        (tmp_path / name).write_text(text)
    assert split_entities(tmp_path, "train") == {
        "protein": {"AAA", "CCC"},
        "reaction": {"C>>O", "CC>>O"},
        "text": {"oxidoreductase"},
    }


def tiny_bert():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(TINY_TEXT_VOCAB),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    return BertModel(config).eval()


def vocab_tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(TINY_TEXT_VOCAB) + "\n")
    return BertTokenizer(str(vocab))


def test_text_vectors_equal_the_untrimmed_original(tmp_path):
    model, tokenizer = tiny_bert(), vocab_tokenizer(tmp_path)
    texts = ["oxidoreductase; acting on ch-oh", "lyase", "transferase (a); b", "x"]
    got = encode("text", tokenizer, model, texts, CPU)
    for text, vector in zip(texts, got):
        e = tokenizer(
            text, truncation=True, max_length=512, padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            want = model(e.input_ids, e.attention_mask).pooler_output[0]
        assert torch.allclose(torch.from_numpy(vector), want, atol=1e-5, rtol=1e-5)


def test_reaction_vectors_equal_the_untrimmed_original():
    tokenizer = load_reaction_tokenizer()
    model = tiny_bert()
    model.embeddings.word_embeddings = torch.nn.Embedding(tokenizer.vocab_size, 64)
    texts = ["CC(=O)O.OCC>>CC(=O)OCC.O", "C>>O", "CCO>>CC=O"]
    got = encode("reaction", tokenizer, model, texts, CPU)
    for text, vector in zip(texts, got):
        e = tokenizer(
            text, truncation=True, max_length=512, padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            want = model(e.input_ids, e.attention_mask).last_hidden_state[0, 0]
        assert torch.allclose(torch.from_numpy(vector), want, atol=1e-5, rtol=1e-5)


def test_protein_vectors_equal_the_original_mean_over_all_positions():
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            PROTEIN_HUB_ID,
            do_lower_case=False,
            cache_dir=str(DEFAULT_OUT_ROOT / "pretrained" / "pretrained_ProtT5"),
            local_files_only=True,
        )
    except OSError:
        pytest.skip("needs the cached ProtT5 tokenizer")
    torch.manual_seed(0)
    model = T5EncoderModel(
        T5Config(vocab_size=128, d_model=64, d_ff=128, d_kv=16, num_layers=2, num_heads=4)
    ).eval()
    seqs = ["MKVLAAGU", "ACDEFGHIKLMNPQRSTVWYZOB", "M"]
    got = encode("protein", tokenizer, model, seqs, CPU)
    max_len = train_args().protein_max_sequence_len
    for seq, vector in zip(seqs, got):
        e = tokenizer(
            " ".join(seq),
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            want = model(e.input_ids, e.attention_mask)["last_hidden_state"].mean(dim=1)[0]
        assert torch.allclose(torch.from_numpy(vector), want, atol=1e-5, rtol=1e-5)


def test_main_encodes_only_the_missing_entities(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_frozen, "revision", lambda *a: "rev")
    stores = cache_frozen.open_stores(tmp_path, tmp_path)
    assert set(stores) == set(cache_frozen.MODALITIES)
    assert stores["protein"].dir == cache_frozen.cache_root(tmp_path) / "protein"
