"""Shared evaluation helpers for Part I, used by every language model.

Common evaluation setup (identical for all models):
    - every model predicts test tokens 1..N-1 of the test stream, each from the
      preceding test tokens only (no training text as context);
    - average next-token NLL and perplexity = exp(NLL); <pad> never appears in
      the stream, and neural training losses also use ignore_index=<pad>;
    - recurrent models carry state through the whole stream; the Transformer
      uses a sliding window of max_length tokens (see neural_mean_nll);
    - the inference workload is scoring the full test stream once;
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

# Matched decoding settings for the generated-text comparison.
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
    """q_T(w) proportional to p(w)^(1/T), as in the lecture demos."""
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


def neural_mean_nll(model, ids, stride=None):
    """Average next-token NLL over target positions 1..N-1 of one token stream.

    Recurrent models read the whole stream in one pass and carry their state
    forward, as in the Lecture 3 demo. A Transformer can only see max_length
    tokens, so it slides a window forward by `stride` (default: half the
    window) and scores each target exactly once, with at least
    max_length - stride tokens of context after the first window.
    """
    model.eval()
    ids = torch.as_tensor(ids, dtype=torch.long)
    total = 0.0
    with torch.no_grad():
        if model.is_recurrent:
            logits, _ = model(ids[:-1].unsqueeze(0))
            total = torch.nn.functional.cross_entropy(
                logits[0], ids[1:], reduction="sum").item()
        else:
            window = model.max_length
            stride = stride or window // 2
            scored_until = 0  # targets ids[1..scored_until] are already scored
            for begin in range(0, len(ids) - 1, stride):
                end = min(begin + window, len(ids) - 1)
                logits, _ = model(ids[begin:end].unsqueeze(0))
                targets = ids[begin + 1 : end + 1]
                new = scored_until - begin  # skip targets scored by the previous window
                total += torch.nn.functional.cross_entropy(
                    logits[0, new:], targets[new:], reduction="sum").item()
                scored_until = end
                if end == len(ids) - 1:
                    break
    return total / (len(ids) - 1)


def neural_next_token_probs(model):
    """Return a next_token_probs(history_ids) function for generate()."""
    def next_token_probs(history_ids):
        model.eval()
        if not model.is_recurrent:
            history_ids = history_ids[-model.max_length:]
        with torch.no_grad():
            logits, _ = model(torch.tensor([history_ids], dtype=torch.long))
            return torch.softmax(logits[0, -1].double(), dim=-1).numpy()
    return next_token_probs


def save_results(name, results):
    """Write one model's metrics to results/<name>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
