"""Offline, small GPT-2 and byte tokenizer for real CPU trainer smoke runs."""
from pathlib import Path


def ensure_tiny_model(teacher=False):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    path = Path("outputs/smoke/local_teacher" if teacher else "outputs/smoke/local_model")
    if (path / "config.json").exists():
        return str(path)
    torch.manual_seed(17 if teacher else 0)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {word: i for i, word in enumerate(["<pad>", "<eos>", "<unk>"] + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>", model_max_length=512)
    tok.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% else %}{{ eos_token }}{% endif %}"
    cfg = GPT2Config(vocab_size=len(tok), n_positions=512, n_embd=32, n_layer=2 if teacher else 1,
                     n_head=2, bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                     pad_token_id=tok.pad_token_id, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    model = GPT2LMHeadModel(cfg)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)
