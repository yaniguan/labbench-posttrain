# Post-training results

Task metrics: **not run**.

No GPU and no LLM API key were available. No model was downloaded. The numbers below are CPU smoke losses on a tiny randomly initialised GPT-2 and toy strings. They are not PQA, ORD, ERR, GEN, LAB-Bench, or MATH scores.

Logged by `python -m posttrain.<method> --smoke` in this workspace:

```
sft step=0.0 loss=5.509847640991211 finite=True grad_norm=0.05017386146947318
sft step=1.0 loss=5.509403705596924 finite=True grad_norm=0.05017927705328831
sft smoke done steps=2

dpo step=0.0 loss=0.6931471824645996 finite=True grad_norm=0.9500709916575939 margin=0.0
dpo step=1.0 loss=0.6507917046546936 finite=True grad_norm=0.8265233550852693 margin=0.866705596446991
dpo smoke done steps=2

ppo step=0.0 loss=0.04629525542259216 finite=True grad_norm=4.631547411116782 approx_kl=0.0 entropy=5.550373554229736 reward_mean=0.25 clip_frac=0.0
ppo step=1.0 loss=0.050411805510520935 finite=True grad_norm=3.923989970884607 approx_kl=0.0002031475305557251 entropy=5.550426006317139 reward_mean=0.5 clip_frac=0.0
ppo smoke done steps=2

grpo step=0.0 loss=-0.005550514906644821 finite=True grad_norm=1.7882155837249678 approx_kl=0.0 entropy=5.550514221191406 reward_mean=0.5833333134651184 clip_frac=0.0 zero_var_frac=0.0
grpo step=1.0 loss=-0.08718131482601166 finite=True grad_norm=1.6454291915008745 approx_kl=0.004464741796255112 entropy=5.550451755523682 reward_mean=0.5833333134651184 clip_frac=0.0 zero_var_frac=0.0
grpo step=2.0 loss=-0.005542226135730743 finite=True grad_norm=1.7132825530252769 approx_kl=0.000785384327173233 entropy=5.550141334533691 reward_mean=0.4166666269302368 clip_frac=0.0 zero_var_frac=0.0
grpo step=3.0 loss=-0.048925548791885376 finite=True grad_norm=1.6991746358256397 approx_kl=0.0025099292397499084 entropy=5.549945831298828 reward_mean=0.4166666269302368 clip_frac=0.0 zero_var_frac=0.0
grpo step=0.0 loss=0.0 finite=True grad_norm=0.0 approx_kl=0.0 entropy=0.0 reward_mean=0.4166666865348816 clip_frac=0.0 zero_var_frac=0.0
grpo smoke prior_only unchanged=True
grpo smoke done steps=2

rft step=0.0 loss=5.573727130889893 finite=True grad_norm=2.5524877887835484
rft step=1.0 loss=5.471493721008301 finite=True grad_norm=2.510389133204516
rft smoke kept=12 best_of_n=4 steps=2
```

DPO step 0 is `log(2)` because the policy and the reference start identical, so the margin is 0. Clip fractions are 0 in these short on-policy steps when the probability ratio stays inside the clip range. That is expected and is not a claim about the clipping-bias result in Shao et al.

pytest: 108 passed, 3 xfailed. The three xfails are known holes in the existing ERR and GEN checkers. See `docs/posttrain/reward_hacking.md`.

Reward ladder, Colab stages, and contamination on a real model: **not run**.
