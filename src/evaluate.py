"""Shared evaluation helpers for Part I

    - every model predicts test tokens 1..N-1 of the test stream, each from the
      preceding test tokens only (no training text as context);
    - average next-token NLL and perplexity = exp(NLL)
    - generation: the same prompts, temperature, length, and seed for every model.

Temperature sampling follows the DSA4213 Lecture 1 and Lecture 3 demos.
"""
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SEED = 4213

PROMPTS = [
    "i was",
    "miss havisham",
    "joe said",
    "it was a",
]
TEMPERATURE = 0.8
GENERATION_LENGTH = 30


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def perplexity(mean_nll):
    return math.exp(mean_nll)


def count_parameters(model):
    """Number of trainable scalar parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    """Wall-clock timer: `with Timer() as t: ...` then read t.seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start


def hardware_summary():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def apply_temperature(probabilities, temperature):
    """q_T(w) proportional to p(w)^(1/T)"""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    adjusted = probabilities ** (1.0 / temperature)
    return adjusted / adjusted.sum()


def generate(next_token_probs, prompt_ids, length=GENERATION_LENGTH,
             temperature=TEMPERATURE, seed=SEED):
    """Sample `length` tokens after the prompt.

    next_token_probs(history_ids) must return a NumPy probability vector over
    the vocabulary. Every model uses this same sampler and seed, so differences
    in output come from the models rather than the decoding procedure.
    """
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    for _ in range(length):
        probabilities = apply_temperature(next_token_probs(ids), temperature)
        ids.append(int(rng.choice(len(probabilities), p=probabilities)))
    return ids


def save_results(name, results):
    """Write one model's metrics to results/<name>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
