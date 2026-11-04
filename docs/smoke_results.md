# Local CPU smoke results

Computed from `outputs/smoke_report.json`: 15/15 jobs passed; sum of process wall times 63.960 seconds.

These are pipeline checks on real data with tiny local models. They are not GPU or scientific benchmark results. The GRPO smoke model received zero structured-answer rewards, producing a measured zero loss.

| job | seconds | last reported training loss |
|---|---:|---:|
| scientist_encoders | 4.380 | 1.4196 |
| scientist_align | 5.582 | 0.3201 |
| scientist_projector | 4.078 | 5.5684 |
| scientist_distill | 3.825 | 2.7044 |
| scientist_sft | 4.429 | 5.5325825 |
| scientist_dpo | 4.528 | 0.69299366 |
| scientist_rm | 4.428 | 0.64314663 |
| scientist_grpo | 4.627 | 0 |
| operator_sft | 4.077 | 5.5495695 |
| operator_dpo | 4.276 | 0.69293682 |
| operator_rm | 4.132 | 0.68590172 |
| operator_grpo | 4.677 | 0 |
| operator_orpo | 4.124 | 5.6267612 |
| operator_bc | 3.324 | 0.14408 |
| operator_wm | 3.474 | 0.02451 |

For jobs with multiple modality loops, the displayed loss is the last recorded loop loss; all step losses and profile summaries remain in the JSON report. Loss scales differ by objective and must not be compared across methods.

Validation: `.venv/bin/python -m pytest -q` passed 53 tests. Both A100 notebooks contain valid Python cells and no stored GPU execution results. Candidate-file audit found no file above 50,000,000 bytes; the largest is `data/v0/tasks/sft_train.jsonl` at 15,544,625 bytes.

**GPU results: PENDING (A100 Colab).**
