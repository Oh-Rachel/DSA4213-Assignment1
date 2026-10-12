"""Part III: Downstream Representation Analysis for IMDB sentiment classification.

Representations: each review is tokenized with the Part I tokenizer, encoded
with the Dickens vocabulary (unknown words -> <unk>), followed by <eos> (as at
the end of every training paragraph), and read by a frozen LSTM. The top-layer
hidden states are pooled into one fixed-length vector in two ways:
    - mean: average of the hidden states over the real tokens (padding excluded)
    - final: the hidden state at the last real token (<eos>)

Classifier: the same L2-regularised logistic regression for every
representation, with the regularisation strength C chosen on the validation
set.
"""
import json
import re
import tarfile
import urllib.request

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import EOS, ROOT, tokenize
from src.evaluate import SEED

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE = ROOT / "data" / "raw" / "aclImdb_v1.tar.gz"
IMDB_SUBSET = ROOT / "data" / "processed" / "imdb_subset.json"
SUBSET_SIZES = {"train": 5000, "valid": 1000, "test": 5000}
LABEL_NAMES = ["negative", "positive"]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

MEMBER_PATTERN = re.compile(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt$")


def download_imdb():
    """Download the IMDB data to data/raw/ unless it is already there."""
    if not IMDB_ARCHIVE.exists():
        IMDB_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {IMDB_URL} ...")
        urllib.request.urlretrieve(IMDB_URL, IMDB_ARCHIVE)
    return IMDB_ARCHIVE


def clean_review(text):
    """Remove the <br /> line-break tags that appear in the raw reviews."""
    return re.sub(r"<br\s*/?>", " ", text)


def read_imdb_archive(path=IMDB_ARCHIVE):
    """Return {"train": [...], "test": [...]} of (file_id, text, label), label 1 = positive."""
    reviews = {"train": [], "test": []}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            match = MEMBER_PATTERN.search(member.name)
            if not match:
                continue  # skips the unlabelled reviews and the documentation files
            split, polarity, review_id, rating = match.groups()
            text = archive.extractfile(member).read().decode("utf-8")
            reviews[split].append((f"{polarity}/{review_id}_{rating}", clean_review(text),
                                   int(polarity == "pos")))
    for split in reviews:
        reviews[split].sort()
    return reviews


def balanced_sample(items, sizes, rng):
    """Split items into consecutive balanced groups, e.g. sizes = {"train": 5000, "valid": 1000}."""
    by_label = {label: [item for item in items if item[2] == label] for label in (0, 1)}
    for label in by_label:
        order = rng.permutation(len(by_label[label]))
        by_label[label] = [by_label[label][i] for i in order]
    groups, start = {}, 0
    for name, size in sizes.items():
        half = size // 2
        group = by_label[0][start:start + half] + by_label[1][start:start + half]
        groups[name] = [group[i] for i in rng.permutation(len(group))]
        start += half
    return groups


def load_imdb_subset(sizes=SUBSET_SIZES, seed=SEED):
    """Return {"train"|"valid"|"test": {"ids", "texts", "labels"}}, cached in data/processed/.

    train and valid come from the official train split (no overlap), test comes
    from the official test split. Every group is 50% positive.
    """
    if IMDB_SUBSET.exists():
        cached = json.loads(IMDB_SUBSET.read_text(encoding="utf-8"))
        if cached["sizes"] == sizes and cached["seed"] == seed:
            return cached["splits"]

    reviews = read_imdb_archive(download_imdb())
    rng = np.random.default_rng(seed)
    groups = balanced_sample(reviews["train"], {"train": sizes["train"], "valid": sizes["valid"]}, rng)
    groups.update(balanced_sample(reviews["test"], {"test": sizes["test"]}, rng))

    splits = {name: {"ids": [item[0] for item in group],
                     "texts": [item[1] for item in group],
                     "labels": [item[2] for item in group]}
              for name, group in groups.items()}
    IMDB_SUBSET.parent.mkdir(parents=True, exist_ok=True)
    IMDB_SUBSET.write_text(json.dumps({"sizes": sizes, "seed": seed, "splits": splits}),
                           encoding="utf-8")
    return splits


def encode_reviews(texts, vocab):
    """Tokenize and encode each review with the Dickens vocabulary, ending with <eos>."""
    return [vocab.encode(tokenize(text) + [EOS]) for text in texts]


def load_lstm(path, model_class, vocab_size, config):
    """Rebuild a model with the saved config and load its trained weights (frozen, eval mode)."""
    model = model_class(vocab_size, cell="lstm", **config)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.eval()


def extract_features(model, id_lists, batch_size=64, pad_id=0):
    """Return {"mean": [n, hidden], "final": [n, hidden]} pooled LSTM hidden states.

    Reviews are sorted by length and right-padded in batches. The LSTM reads
    left to right, so padding after a review cannot change its earlier states.
    both poolings use only the real positions.
    """
    model.eval()
    hidden_size = model.recurrent.hidden_size
    n = len(id_lists)
    features = {"mean": np.zeros((n, hidden_size), dtype=np.float32),
                "final": np.zeros((n, hidden_size), dtype=np.float32)}
    order = np.argsort([len(ids) for ids in id_lists], kind="stable")
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = order[start:start + batch_size]
            lengths = torch.tensor([len(id_lists[i]) for i in rows])
            batch = torch.full((len(rows), int(lengths.max())), pad_id, dtype=torch.long)
            for position, i in enumerate(rows):
                batch[position, :lengths[position]] = torch.tensor(id_lists[i])
            states, _ = model.hidden_states(batch)  # [batch, length, hidden]
            mask = torch.arange(batch.shape[1])[None, :] < lengths[:, None]
            summed = (states * mask.unsqueeze(-1)).sum(dim=1)
            features["mean"][rows] = (summed / lengths[:, None]).numpy()
            features["final"][rows] = states[torch.arange(len(rows)), lengths - 1].numpy()
    return features


def bag_of_words(id_lists, vocab_size):
    """Binary word-presence features over the same vocabulary ([n, vocab_size] float32)."""
    features = np.zeros((len(id_lists), vocab_size), dtype=np.float32)
    for row, ids in enumerate(id_lists):
        features[row, ids] = 1.0
    return features


def train_classifier(train_X, train_y, valid_X, valid_y, c_grid=C_GRID, scale=True, seed=SEED):
    """Fit logistic regression for each C on train, keep the C with the best validation accuracy.

    scale=True standardises each feature first (used for the dense LSTM features).
    Returns (fitted classifier, best C, {C: validation accuracy}).
    """
    best_classifier, best_c, valid_scores = None, None, {}
    for c in c_grid:
        classifier = LogisticRegression(C=c, max_iter=5000, random_state=seed)
        if scale:
            classifier = make_pipeline(StandardScaler(), classifier)
        classifier.fit(train_X, train_y)
        valid_scores[c] = float(accuracy_score(valid_y, classifier.predict(valid_X)))
        if best_c is None or valid_scores[c] > valid_scores[best_c]:
            best_classifier, best_c = classifier, c
    return best_classifier, best_c, valid_scores


def evaluate_classifier(classifier, X, y):
    """Accuracy, macro-F1, confusion matrix, predictions and P(positive) on one split."""
    predictions = classifier.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),  # rows = true, cols = predicted
        "predictions": predictions.tolist(),
        "prob_positive": classifier.predict_proba(X)[:, 1].tolist(),
    }
