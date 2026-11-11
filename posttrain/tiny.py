"""A tiny GPT-2 built in memory.

Smoke tests and CPU demos use this so they do not download weights.
The model is randomly initialised. Its loss is not a lab score.
"""
from __future__ import annotations

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def build_tiny(seed: int = 0, n_layer: int = 1, n_embd: int = 32, n_head: int = 2, n_positions: int = 64):
    """Return (model, tokenizer). Dropout is off so a smoke step is deterministic given the seed."""
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
    tok.padding_side = "right"
    cfg = GPT2Config(
        vocab_size=len(tok),
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    return model, tok
