# Post-training status

**not run yet.**

No GPU job has been launched from `posttrain/`. No ladder, no KL sweep, and no contamination eval on a real model has a number here.

CPU smokes of the tiny random model are code checks. Their logs belong in the pull request, not in this file as if they were lab results. Re-run:

```bash
python -m posttrain.sft --smoke
python -m posttrain.dpo --smoke
python -m posttrain.ppo --smoke
python -m posttrain.grpo --smoke
```

The design notes are in `docs/posttrain/README.md`.
