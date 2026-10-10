"""Interpolated n-gram language model with add-k smoothing (Part I baseline).

Adapted from the DSA4213 Lecture 3 demo (fit_interpolated_ngram and ngram_distribution).
    
Additional features:
    - interpolation weights are fitted on the validation split by
      maximising validation log-likelihood instead of being fixed by hand
    - target probabilities are computed directly instead of building the full
      vocabulary distribution at every position, which is much faster.
"""
from collections import Counter, defaultdict

import numpy as np

MAX_HISTORY = 4  # histories of 0..4 tokens, i.e. an interpolated 1- to 5-gram


class InterpolatedNgram:
    def __init__(self, vocab_size, max_history=MAX_HISTORY, add_k=0.01):
        self.vocab_size = vocab_size
        self.max_history = max_history
        self.add_k = add_k
        # Equal weights until fit_weights is called.
        self.weights = np.full(max_history + 1, 1.0 / (max_history + 1))

    def fit(self, ids):
        """Count next tokens after every history of length 0..max_history."""
        self.counts = []
        self.totals = []
        for history_length in range(self.max_history + 1):
            next_counts = defaultdict(Counter)
            totals = Counter()
            for position in range(history_length, len(ids)):
                history = tuple(ids[position - history_length : position])
                next_counts[history][ids[position]] += 1
                totals[history] += 1
            self.counts.append(next_counts)
            self.totals.append(totals)
        return self

    def stored_count_entries(self):
        """Number of nonzero (history, next token) counts: the model's table size."""
        return sum(len(row) for counts in self.counts for row in counts.values())

    def _history(self, history_ids, history_length):
        return tuple(history_ids[len(history_ids) - history_length :]) if history_length else ()

    def component_probabilities(self, history_ids, target):
        """Smoothed P(target | last h tokens) for h = 0..max_history.

        Near the start of a stream, orders longer than the available history
        reuse the longest available one.
        """
        k, V = self.add_k, self.vocab_size
        components = np.empty(self.max_history + 1)
        for h in range(self.max_history + 1):
            usable = min(h, len(history_ids))
            history = self._history(history_ids, usable)
            count = self.counts[usable].get(history, {}).get(target, 0)
            total = self.totals[usable].get(history, 0)
            components[h] = (count + k) / (total + k * V)
        return components

    def component_matrix(self, ids):
        """One row of component probabilities per target position 1..N-1."""
        return np.array([
            self.component_probabilities(ids[max(0, position - self.max_history) : position],
                                         ids[position])
            for position in range(1, len(ids))
        ])

    def fit_weights(self, valid_ids, iterations=100):
        """EM for the interpolation weights that maximise validation log-likelihood."""
        components = self.component_matrix(valid_ids)
        weights = np.full(self.max_history + 1, 1.0 / (self.max_history + 1))
        for _ in range(iterations):
            weighted = components * weights
            responsibilities = weighted / weighted.sum(axis=1, keepdims=True)
            weights = responsibilities.mean(axis=0)
        self.weights = weights
        return self

    def mean_nll(self, ids):
        """Average next-token NLL over target positions 1..N-1."""
        mixture = self.component_matrix(ids) @ self.weights
        return float(-np.log(mixture).mean())

    def next_token_probs(self, history_ids):
        """Full interpolated distribution over the vocabulary (for generation)."""
        k, V = self.add_k, self.vocab_size
        distribution = np.zeros(V)
        for h, weight in enumerate(self.weights):
            usable = min(h, len(history_ids))
            history = self._history(history_ids, usable)
            denominator = self.totals[usable].get(history, 0) + k * V
            component = np.full(V, k / denominator)
            for token, count in self.counts[usable].get(history, {}).items():
                component[token] += count / denominator
            distribution += weight * component
        return distribution
