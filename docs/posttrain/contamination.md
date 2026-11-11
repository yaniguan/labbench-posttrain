# Contamination and memorization

LAB-Bench text is evaluation only (CC-BY-SA). The public jsonl files keep a canary string so a training corpus that swallowed the benchmark can be spotted. Do not train on `data/labbench/`. BioProBench questions can be scored by the same functions later. The test split should stay out of the training reward if you want it as held-out gold.

No contamination number for a real model is claimed here. The CPU demo runs the metrics on a tiny random model and a toy sentence. Status of that demo is `ran on tiny model`. It is not a finding.

## Prefix completion

Give the model the first 10%, 25%, 50%, or 75% of a question and greedy-decode the rest. Compare the continuation to the true remainder:

- exact match
- normalised edit distance (0 means identical)
- ROUGE-L, implemented as LCS length divided by the reference length
- token Jaccard overlap

A high score on the original is not enough. The same function runs on a shuffled-character control and should be reported next to it. If both are high, the metric is easy, not specific. `shuffled_control` builds that string.

## Option order

Shuffle the options. Record the old letter and the new letter of the same answer text. If the model still emits the old letter, it memorised the letter. If it emits the new letter, it followed the text. `letter_was_memorised` is that check. A small accuracy drop after a shuffle is what you want. A sharp drop is a memorised letter.

Answer given a partial question is the prefix test above: if greedy decoding reproduces the rest of the question verbatim, the model is completing a memorised item rather than answering.

## Perplexity ratio and min-k%

`mean_logprob` is the average token log-probability (higher is more likely). `min_k_logprob` is the average log-probability of the least likely 20% of tokens, a simple min-k% score. The membership gap is original minus paraphrase. Positive means the original is more likely.

**Weakness.** A paraphrase can be rarer English even when the model never saw the original. A gap is not proof of membership. The shuffled control and a post-cutoff set are the checks that keep this from being the only evidence.

## Canary in outputs

`canary_in_texts` searches generations for the LAB-Bench canary. A hit means the model is emitting a string that was planted to detect training on the benchmark. The CPU demo's toy sentence does not contain it.

## Post-cutoff holdout

Shao et al., Appendix D, Figure 13 caption: AIME 2025 was released after the models in that paper, so it is a contamination control. The same caption says each AIME set has 30 questions and gaps under about 2 points may not be significant.

`configs/posttrain/post_cutoff_holdout.yaml` lists AIME 2025, AIME 2024, public LAB-Bench, and the BioProBench test split. **Not run.** No scores are filled in.

`interpret_holdout` only writes a reading when you pass scores. If the benchmark was released after the model, a high post-cutoff score is not explained by that benchmark sitting in pretraining. If the pre-cutoff score is much higher, that is consistent with contamination and also with the new set being harder. It is not a verdict. If you pass no scores, the function says **not run**.

## How to run it later

`posttrain/contamination.py` `cpu_demo()` is the CPU path. On Colab, point `score_prefixes` and `sequence_token_logprobs` at a real model and at LAB-Bench or BioProBench text. Do not add those texts to a training jsonl as part of that run.
