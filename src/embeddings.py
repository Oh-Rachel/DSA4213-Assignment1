"""Embedding tables for Part II: own Word2Vec and public GloVe.

Both tables are built the same way, so settings B and C differ only in where
the vectors come from:
    - one row per vocabulary id, in vocabulary order
    - <pad> is all zeros
    - tokens without a vector get N(0, std of the found vectors), seed 4213

Word2Vec settings follow the gensim call in the DSA4213 Lecture 2 demo, with
vector_size=100 to match GloVe 100d and the Part I models. min_count=1 because
the vocabulary is already filtered (min_count=3), so every vocabulary token,
including <unk> and <eos>, gets a vector.
"""
import numpy as np
from gensim.models import Word2Vec

from src.data import EOS, load_paragraphs, tokenize
from src.evaluate import SEED

GLOVE_NAME = "glove-wiki-gigaword-100"  # GloVe 6B 100d (Wikipedia 2014 + Gigaword 5)

WORD2VEC_SETTINGS = dict(
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,        
    negative=5,
    ns_exponent=0.75,
    sample=0.0,
    shrink_windows=False,
    epochs=15,
    workers=1,            
    seed=SEED,
)


def word2vec_sentences(vocab, split="train"):
    """One token list per paragraph, mapped through the vocabulary.

    Out-of-vocabulary words become <unk> and each paragraph ends with <eos>,
    exactly as the language models see the text.
    """
    return [vocab.decode(vocab.encode(tokenize(paragraph))) + [EOS]
            for paragraph in load_paragraphs(split)]


def train_word2vec(sentences, **overrides):
    """Train gensim Word2Vec with WORD2VEC_SETTINGS (overrides allowed)."""
    return Word2Vec(sentences=sentences, **{**WORD2VEC_SETTINGS, **overrides})


def load_glove(name=GLOVE_NAME):
    """Load public GloVe vectors. gensim downloads them on first use (~130 MB)."""
    import gensim.downloader
    return gensim.downloader.load(name)


def build_embedding_matrix(vocab, vectors, train_ids=None, seed=SEED):
    """Return (matrix [len(vocab), dim] float32, coverage report).

    vectors: gensim KeyedVectors (Word2Vec .wv or GloVe).
    train_ids: if given, also report the share of train tokens that are covered.
    """
    dim = vectors.vector_size
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    found, missing = [], []
    for index, token in enumerate(vocab.id_to_token):
        if index == vocab.pad_id:
            continue
        if token in vectors:
            matrix[index] = vectors[token]
            found.append(index)
        else:
            missing.append(index)

    init_std = float(matrix[found].std())
    rng = np.random.default_rng(seed)
    for index in missing:
        matrix[index] = rng.normal(0.0, init_std, dim)

    report = {
        "vocabulary_size": len(vocab) - 1,  # excluding <pad>
        "found": len(found),
        "missing": len(missing),
        "type_coverage": len(found) / (len(vocab) - 1),
        "missing_examples": vocab.decode(missing[:20]),  # most frequent first
        "missing_init": f"normal(0, {init_std:.4f}) with seed {seed}; <pad> = zeros",
        "mean_vector_norm": float(np.linalg.norm(matrix[found], axis=1).mean()),
    }
    if train_ids is not None:
        is_found = np.zeros(len(vocab), dtype=bool)
        is_found[found] = True
        report["train_token_coverage"] = float(is_found[np.asarray(train_ids)].mean())
    return matrix, report


def nearest_neighbours(word, matrix, vocab, k=5):
    """The k vocabulary words with the highest cosine similarity to `word`."""
    normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    query = vocab.token_to_id[word]
    scores = normed @ normed[query]
    scores[[vocab.pad_id, vocab.unk_id, vocab.eos_id, query]] = -np.inf
    return vocab.decode(np.argsort(-scores)[:k])
