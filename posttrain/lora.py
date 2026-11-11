"""Optional LoRA. Used so a Colab run can fit a 0.5B model. Smoke tests the wrap on the tiny GPT-2."""
from __future__ import annotations

from peft import LoraConfig, get_peft_model


def apply_lora(model, r: int = 4, alpha: int = 8, target_modules: list[str] | None = None, dropout: float = 0.0):
    """Freeze the base weights and train a low-rank update.

    GPT-2's fused attention projection is named c_attn. Llama-style models use q_proj and v_proj.
    Pass target_modules from the config when the architecture is not GPT-2.
    """
    targets = target_modules or ["c_attn"]
    # GPT-2 stores attention in a Conv1D. peft wants fan_in_fan_out for that layout.
    # Llama-style q_proj/v_proj are ordinary linear layers and should leave it off.
    fan_in_fan_out = "c_attn" in targets
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=fan_in_fan_out,
    )
    return get_peft_model(model, cfg)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
