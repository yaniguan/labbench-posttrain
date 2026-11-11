"""What a KL sweep is supposed to answer. The sweep itself is not run.

The question: does turning the KL coefficient down raise the proxy reward
while the held-out gold metric, the format rate, or the style distance
gets worse? If yes, the extra proxy reward is drift or hacking. The knee
is the smallest coefficient that still keeps gold and style near the SFT
checkpoint.
"""
from __future__ import annotations


def render_kl_report(rows: list[dict] | None) -> str:
    lines = [
        "# KL anchor report",
        "",
        "Question: does a lower KL coefficient raise proxy reward while the held-out gold metric or the style gets worse, and where is the knee?",
        "",
    ]
    if not rows:
        lines.append("Status: not run. No step logs were passed in.")
        lines.append("")
        lines.append("When a log exists, each row needs proxy_reward, gold_metric, kl_to_sft, kl_coef, format_compliance, response length, entropy, style_distance_to_sft, and the hacking flags.")
        return "\n".join(lines)
    lines.append("Status: computed from the rows passed to this function. This text does not add scores.")
    lines.append("")
    lines.append("| kl_coef | proxy | gold | KL to SFT | format | length | entropy | style distance | hacking |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        flags = row.get("hacking_flags") or []
        lines.append(
            "| {coef} | {proxy} | {gold} | {kl} | {fmt} | {length} | {ent} | {style} | {flags} |".format(
                coef=_fmt(row.get("kl_coef")),
                proxy=_fmt(row.get("proxy_reward", row.get("reward_mean"))),
                gold=_fmt(row.get("gold_metric")),
                kl=_fmt(row.get("kl_to_sft")),
                fmt=_fmt(row.get("format_compliance")),
                length=_fmt(row.get("response_length_mean")),
                ent=_fmt(row.get("entropy")),
                style=_fmt(row.get("style_distance_to_sft")),
                flags=", ".join(flags) if flags else "none",
            )
        )
    lines.append("")
    lines.append(_knee_sentence(rows))
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "not run"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _knee_sentence(rows: list[dict]) -> str:
    usable = [
        r for r in rows
        if r.get("kl_coef") is not None and r.get("proxy_reward", r.get("reward_mean")) is not None
        and r.get("gold_metric") is not None
    ]
    if len(usable) < 2:
        return "Not enough rows to talk about a knee."
    ordered = sorted(usable, key=lambda r: float(r["kl_coef"]))
    low, high = ordered[0], ordered[-1]
    proxy_low = float(low.get("proxy_reward", low.get("reward_mean")))
    proxy_high = float(high.get("proxy_reward", high.get("reward_mean")))
    gold_low = float(low["gold_metric"])
    gold_high = float(high["gold_metric"])
    if proxy_low > proxy_high and gold_low < gold_high:
        return (
            "In these rows, the lower KL coefficient has a higher proxy and a lower gold metric. "
            "That is the drift pattern. The knee is the higher coefficient that still holds the gold metric, "
            "not the coefficient with the highest proxy."
        )
    if proxy_low > proxy_high and gold_low >= gold_high:
        return (
            "In these rows, a lower KL coefficient raises the proxy and does not lower the gold metric. "
            "That does not by itself prove the reward is clean. Check format, style distance, and the hacking flags."
        )
    return (
        "In these rows, lowering the KL coefficient does not raise the proxy while hurting gold. "
        "Read the style distance and the hacking flags before treating the proxy as the task."
    )
