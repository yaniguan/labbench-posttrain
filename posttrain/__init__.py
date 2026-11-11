"""Post-training for an automated lab: SFT, DPO, PPO, and GRPO.

The reference policy for every KL term is the SFT checkpoint. KL stays on
unless a caller turns it off on purpose and accepts the ablation warning.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
