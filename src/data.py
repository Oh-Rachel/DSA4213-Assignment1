"""Shared tokenization and vocabulary for all parts of the assignment.

Usage:
    from src.data import load_corpus
    vocab, splits = load_corpus()   
    python src/data.py 

Adapted from the DSA4213 Lecture 1 and Lecture 3 demos:
    - tokenizer regex from Lecture 1 (lowercase words + . , ! ? ; :)
    - training-only vocabulary with <unk> from Lecture 3
Changes:
    - vocabulary keeps tokens seen at least MIN_COUNT times
    - <eos> marks each paragraph end
    - <pad> is reserved for batching so it can be excluded from the loss
"""
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

PAD = "<pad>"
UNK = "<unk>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, EOS]  # ids 0, 1, 2

MIN_COUNT = 3

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")
PUNCTUATION = {".", ",", "!", "?", ";", ":"}


def tokenize(text):
    """Convert a string to a lowercase list of word/punctuation tokens."""
    normalized = text.lower().replace("’", "'")
    return TOKEN_PATTERN.findall(normalized)


def detokenize(tokens):
    """Join tokens into readable text, attaching punctuation to the previous word."""
    text = ""
    for token in tokens:
        if token in PUNCTUATION:
            text = text.rstrip() + token + " "
        elif token == EOS:
            text = text.rstrip() + "\n"
        else:
            text += token + " "
    return text.strip()


def load_paragraphs(split):
    """Return the paragraphs of one split ("train", "valid", or "test")."""
    path = PROCESSED_DIR / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python src/prepare_data.py first.")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_split(split):
    """Return one split as a single token stream with <eos> after each paragraph."""
    tokens = []
    for paragraph in load_paragraphs(split):
        tokens.extend(tokenize(paragraph))
        tokens.append(EOS)
    return tokens


class Vocabulary:
    """Token <-> id mapping built from training tokens only."""

    def __init__(self, tokens, min_count=MIN_COUNT):
        counts = Counter(token for token in tokens if token not in SPECIAL_TOKENS)
        # Sort by frequency, then alphabetically, so ids are deterministic.
        kept = sorted(
            (token for token, count in counts.items() if count >= min_count),
            key=lambda token: (-counts[token], token),
        )
        self.id_to_token = SPECIAL_TOKENS + kept
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.min_count = min_count

    def __len__(self):
        return len(self.id_to_token)

    def __contains__(self, token):
        return token in self.token_to_id

    @property
    def pad_id(self):
        return self.token_to_id[PAD]

    @property
    def unk_id(self):
        return self.token_to_id[UNK]

    @property
    def eos_id(self):
        return self.token_to_id[EOS]

    def encode(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids):
        return [self.id_to_token[int(index)] for index in ids]


def unk_rate(ids, vocab):
    """Fraction of ids that are <unk>."""
    return sum(index == vocab.unk_id for index in ids) / len(ids) if ids else 0.0


def load_corpus(min_count=MIN_COUNT):
    """Build the vocabulary from train and encode all three splits."""
    tokens = {split: load_split(split) for split in ("train", "valid", "test")}
    vocab = Vocabulary(tokens["train"], min_count=min_count)
    splits = {split: vocab.encode(split_tokens) for split, split_tokens in tokens.items()}
    return vocab, splits


def main():
    vocab, splits = load_corpus()
    print(f"Vocabulary size (min_count={vocab.min_count}): {len(vocab):,}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    for split, ids in splits.items():
        print(f"  {split:<5}: {len(ids):>8,} tokens, <unk> rate {unk_rate(ids, vocab):.2%}")
    print("First 20 training tokens:")
    print(vocab.decode(splits["train"][:20]))


if __name__ == "__main__":
    main()
