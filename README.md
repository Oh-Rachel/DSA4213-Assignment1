# DSA4213 Assignment 1: Language Models, Embeddings and Representations

This repository contains the code and results for DSA4213 Assignment 1. It has three parts:

1. **Part I:** an interpolated n-gram, a vanilla RNN, an LSTM and a causal Transformer are trained on the same corpus and compared.
2. **Part II:** the LSTM is trained with three input-embedding tables: random and trainable, our own Word2Vec (frozen) and public GloVe (frozen).
3. **Part III:** the two Part II LSTMs are frozen and used as feature extractors for IMDB sentiment classification.

## Repository layout

```
src/                  reusable code (data, models, training, evaluation)
  prepare_data.py     downloads, cleans and splits the corpus
  data.py             tokenizer, vocabulary and encoded splits
  ngram.py            interpolated n-gram model
  models.py           RNN/LSTM and Transformer language models
  train.py            shared training loop
  evaluate.py         perplexity, timing, generation, saving results
  experiment.py       train, test, generate and save one neural model
  embeddings.py       Word2Vec training, GloVe loading, embedding tables
  representations.py  IMDB data, feature extraction and classifiers
notebooks/
  part1.ipynb         Part I experiments
  part2.ipynb         Part II experiments
  part3.ipynb         Part III experiments
results/
  part1/              Part I metrics (JSON) and plots
  part2/              Part II metrics (JSON) and plots
  part3/              Part III metrics (JSON) and plots
  models/             saved Part II LSTM weights, used in Part III
```

The `data/` folder is created by the code and is not included in the repository.

## Data access

All data is downloaded by the code. No data is redistributed in this repository.

| Data | Used in | Source | How it is downloaded |
|---|---|---|---|
| *Great Expectations*, Charles Dickens (1861), Project Gutenberg eBook #1400 | Parts I and II | https://www.gutenberg.org/cache/epub/1400/pg1400.txt | `src/prepare_data.py` saves it to `data/raw/` |
| GloVe 6B 100d (Pennington et al., 2014) | Part II | https://nlp.stanford.edu/projects/glove/ | `notebooks/part2.ipynb` loads `glove-wiki-gigaword-100` through `gensim.downloader` (cached in `~/gensim-data`) |
| IMDB Large Movie Review Dataset (Maas et al., 2011) | Part III | https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | `notebooks/part3.ipynb` saves it to `data/raw/` |


## Preprocessing

**Corpus (`src/prepare_data.py`):**
- The Gutenberg header and footer are removed, as in the Lecture 1 to 3 demos. The text starts at "My father's family name being Pirrip".
- Chapter headings, `[Illustration]` markers, scene-break markers, italics underscores and the final "THE END" are removed. Curly quotes are converted to straight ones.
- Wrapped lines are joined, so each paragraph is one line.
- Paragraphs are split 90/5/5 into train, validation and test sets by position (not randomly), so the test set is unseen later text.
- Output: `data/processed/{train,valid,test}.txt`, with 3,451 / 192 / 192 paragraphs.

**Tokenizer and vocabulary (`src/data.py`):**
- Text is lowercased and split into words and six punctuation marks (`. , ! ? ; :`), following the Lecture 1 tokenizer.
- Possessives and contractions are split the way GloVe splits them (`father's` to `father 's`, `don't` to `do n't`). Dashes and quotation marks are dropped.
- An `<eos>` token ends each paragraph.
- The vocabulary is built from the training split only and keeps tokens seen at least 3 times. All other tokens are mapped to `<unk>`. Special ids are `<pad>` = 0, `<unk>` = 1 and `<eos>` = 2.
- This gives 4,103 vocabulary types and 199,118 / 12,653 / 9,889 train / validation / test tokens.

**IMDB (`src/representations.py`):**
- `<br />` tags are removed. Reviews are tokenized with the same tokenizer, encoded with the Dickens vocabulary and ended with `<eos>`.
- A seeded and balanced subset is used: 5,000 training and 1,000 validation reviews from the official training split (no overlap), and 5,000 test reviews from the official test split. The chosen review ids are cached in `data/processed/imdb_subset.json` and saved in `results/part3/classification.json`.

## Dependencies

- Python 3.13.9. Every run used a CPU only (Intel Core Ultra 7 155H, Windows 11, PyTorch with 16 threads, no GPU).
- Packages are pinned in `requirements.txt`: numpy, scipy, gensim, torch (CPU build), matplotlib, scikit-learn, jupyter and ipykernel.

## How to reproduce

Run these commands from the project root (Windows PowerShell shown):

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python src/prepare_data.py
```

Then run the three notebooks in order. Each notebook should be run top to bottom (Restart & Run All), and it must be run from the `notebooks/` folder, since it adds the project root to `sys.path` to import `src`.

| Step | Notebook | Needs | Writes | Approximate time on our CPU |
|---|---|---|---|---|
| 1 | `notebooks/part1.ipynb` | `data/processed/` | `results/part1/` | about 60 minutes |
| 2 | `notebooks/part2.ipynb` | `results/part1/lstm.json` | `results/part2/`, `results/models/` | about 45 minutes, plus the first GloVe download |
| 3 | `notebooks/part3.ipynb` | `results/models/` | `results/part3/` | a few minutes, plus the first IMDB download |

Part II reuses the Part I LSTM as setting A, so Part I must be run first. Part III loads the two Part II models from `results/models/`, so Part II must be run before Part III.

## Configurations

**Part I: shared neural training (same for all three neural models)**

| Setting | Value |
|---|---|
| Input embedding | 100-d, randomly initialised and trainable |
| Training chunks | non-overlapping 64-token chunks, reshuffled every epoch |
| Batch size | 32 |
| Optimiser | Adam, learning rate 1e-3 |
| Gradient clipping | 1.0 (gradient norm) |
| Dropout | 0.2 |
| Epochs | 40, keeping the weights of the epoch with the lowest validation loss |
| Loss | cross-entropy, padding excluded |

**Part I: models**

| Model | Settings |
|---|---|
| n-gram | 1- to 5-gram, add-k in each order (k chosen from 1e-5 to 1 on validation, best k = 1e-4), interpolation weights fitted by EM on validation |
| RNN | 1 layer, 256 hidden units |
| LSTM | 1 layer, 256 hidden units |
| Transformer | 2 pre-norm layers, 4 heads, feed-forward 256, learned positions, causal mask, 64-token context |

**Part I: evaluation**
- Every model predicts test tokens 2 to N from the preceding test tokens only (9,888 predictions).
- The RNN and LSTM carry their state through the whole test set. The Transformer uses a 64-token sliding window with stride 32.
- Inference timing: 1 warm-up pass, then 5 timed passes over the test set, median reported.
- Generation: the same 4 prompts ("i was", "miss havisham", "joe said", "it was a"), 30 tokens, temperature 0.8, seed 4213 + prompt index.

**Part II: embeddings**
- Same LSTM and training as Part I. Only the input-embedding table changes.
- Word2Vec (gensim): `sg=1, vector_size=100, window=5, negative=5, ns_exponent=0.75, sample=0.0, shrink_windows=False, min_count=1, epochs=15, workers=1, seed=4213`. It is trained on the training split only.
- GloVe: `glove-wiki-gigaword-100` (GloVe 6B 100d).
- In the frozen settings the entire embedding table is frozen. `<pad>` is a zero vector. Missing words get random values from N(0, s²), where s is the standard deviation of the vectors that were found (seed 4213).

**Part III: classification**
- Representations: mean of the LSTM hidden states over real tokens (padding excluded), and the final hidden state at `<eos>`.
- References: an untrained LSTM with the same architecture and seed, and a binary bag-of-words over the same vocabulary.
- Classifier: scikit-learn `LogisticRegression` (L2, `max_iter=5000`). LSTM features are standardised on the training set, and bag-of-words features are left unscaled.
- C is chosen from {0.001, 0.01, 0.1, 1, 10} by validation accuracy. The test set is scored once.

## Random seeds

- One seed, **4213**, is used everywhere. It is set in `src/evaluate.py` (`SEED`).
- `set_seed(4213)` is called before each model is built, so the Part II models start from the same non-embedding weights as the Part I LSTM.
- The same seed is used for chunk shuffling, Word2Vec, missing-word initialisation, the IMDB subset and the classifiers. Generation uses seed 4213 + prompt index.
- Perplexities and samples were identical across repeated runs on our machine. Timings vary between runs because the laptop CPU is shared with other processes.

## Main results

| Part | Result |
|---|---|
| I | Test perplexity: LSTM 73.06, RNN 77.22, Transformer 94.06, n-gram 94.07 |
| II | Test perplexity: A. random 73.06, B. Word2Vec 73.38, C. GloVe 77.75 |
| III | Test accuracy: GloVe LSTM mean 76.3%, Word2Vec LSTM mean 71.8%, untrained LSTM mean 73.9%, bag-of-words 82.1% |

Full results are in `results/` and in the report.

## Attribution

- The tokenizer, Gutenberg cleaning, positional split, n-gram interpolation, chunked training and temperature sampling are adapted from the DSA4213 Lecture 1 to 3 demo notebooks. The Word2Vec settings are adapted from the Lecture 2 demo.
- Data and embeddings: Dickens (1861) from Project Gutenberg, Pennington, Socher and Manning (2014) for GloVe, and Maas et al. (2011) for IMDB.
