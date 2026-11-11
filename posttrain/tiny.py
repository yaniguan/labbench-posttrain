"""A tiny causal LM built in memory. No download, no checkpoint file.

Smoke tests and CPU demos use this model. It is not a lab model.
"""
from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def build_tiny_lm(seed: int = 0, n_layer: int = 1, n_embd: int = 32, n_positions: int = 64):
    """Random GPT-2 sized for a CPU smoke run.

    Dropout is off so a train step and an eval step see the same function.
    """
    torch.manual_seed(seed)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {word: i for i, word in enumerate(["<pad>", "<eos>", "<unk>"] + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=n_positions,
    )
    cfg = GPT2Config(
        vocab_size=len(tok),
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=2,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok
