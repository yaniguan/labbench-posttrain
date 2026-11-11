"""Hand-written toy strings for CPU smoke tests.

These are not LAB-Bench and not BioProBench. They exist so a trainer can take a step
without a download and without touching a benchmark.
"""

SFT_PAIRS = [
    ("Add reagent. Volume:", " 10 ml."),
    ("Incubate at", " 37 C."),
    ("Spin the tube at", " 300 g."),
    ("The next step is", " wash."),
]

DPO_PAIRS = [
    {"prompt": "Incubate at", "chosen": " 37 C.", "rejected": " 4 C."},
    {"prompt": "Spin the tube at", "chosen": " 300 g.", "rejected": " 10 rpm."},
    {"prompt": "Add reagent. Volume:", "chosen": " 10 ml.", "rejected": " none."},
]

RL_PROMPTS = [
    "Incubate at",
    "Spin the tube at",
    "Add reagent. Volume:",
    "The next step is",
]
