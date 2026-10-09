"""Download, clean, and split the Great Expectations corpus.

Usage:
    python src/prepare_data.py

Outputs (one paragraph per line):
    data/raw/<book>.txt               original Project Gutenberg files
    data/processed/train.txt          first 90% of each book
    data/processed/valid.txt          next 5% of each book
    data/processed/test.txt           last 5% of each book

Gutenberg header/footer handling follows the DSA4213 Lecture 1-3 demos.
"""
import re
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# opening: first words of the story, used to skip the title page and contents.
# chapter_titles: whether each "Chapter N." heading is followed by a title line.
BOOKS = {
    "great_expectations": {
        "url": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
        "opening": r"My father's family name being Pirrip",
        "chapter_titles": False,
    },
}

TRAIN_FRACTION = 0.90
VALID_FRACTION = 0.05  # the remaining 5% is test

CHAPTER_HEADING = re.compile(r"^\s*CHAPTER\s+[IVXLC]+\.?\s*$", re.I)


def download(name, url):
    path = RAW_DIR / f"{name}.txt"
    if path.exists():
        print(f"  exists, skipping download: {path.relative_to(ROOT)}")
    else:
        path.write_bytes(urlopen(url).read())
        print(f"  downloaded: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def extract_story(raw_text, opening):
    """Remove the Gutenberg header/footer, title page, and contents."""
    start = re.search(r"\*\*\* START OF .*?\*\*\*", raw_text, flags=re.I)
    end = re.search(r"\*\*\* END OF .*?\*\*\*", raw_text, flags=re.I)
    if not start or not end:
        raise ValueError("Gutenberg START/END markers not found")
    body = raw_text[start.end() : end.start()]

    # Allow either a straight or curly apostrophe in the opening phrase.
    opening_match = re.search(opening.replace("'", "['’]"), body)
    if not opening_match:
        raise ValueError(f"Opening phrase not found: {opening!r}")
    # Start at the beginning of the line containing the opening phrase.
    line_start = body.rfind("\n", 0, opening_match.start()) + 1
    return body[line_start:]


def clean_paragraphs(story_text, chapter_titles):
    """Return cleaned paragraphs, each joined onto a single line."""
    lines = story_text.replace("\r\n", "\n").split("\n")

    # Drop each chapter heading, and its title line if the book has one.
    kept = []
    skip_title = False
    for line in lines:
        if CHAPTER_HEADING.match(line):
            skip_title = chapter_titles
            continue
        if skip_title and line.strip():
            skip_title = False
            continue
        kept.append(line)

    paragraphs = []
    for block in re.split(r"\n\s*\n", "\n".join(kept)):
        text = " ".join(block.split())
        text = re.sub(r"\[Illustration[^\]]*\]", " ", text, flags=re.I)
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("_", "")  # Gutenberg italics markup
        text = " ".join(text.split())
        # Skip scene-break ornaments such as "* * * * *".
        if not text or not re.search(r"[A-Za-z]", text):
            continue
        if text.upper() == "THE END":
            continue
        paragraphs.append(text)
    return paragraphs


def split_by_position(paragraphs):
    n = len(paragraphs)
    train_end = int(n * TRAIN_FRACTION)
    valid_end = int(n * (TRAIN_FRACTION + VALID_FRACTION))
    return {
        "train": paragraphs[:train_end],
        "valid": paragraphs[train_end:valid_end],
        "test": paragraphs[valid_end:],
    }


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    combined = {"train": [], "valid": [], "test": []}
    for name, book in BOOKS.items():
        print(name)
        raw_text = download(name, book["url"])
        story = extract_story(raw_text, book["opening"])
        paragraphs = clean_paragraphs(story, book["chapter_titles"])
        splits = split_by_position(paragraphs)
        for split_name, split_paragraphs in splits.items():
            combined[split_name].extend(split_paragraphs)
        words = sum(len(p.split()) for p in paragraphs)
        print(f"  {len(paragraphs):,} paragraphs, {words:,} words")
        print(f"  first: {paragraphs[0][:70]}...")
        print(f"  last:  {paragraphs[-1][:70]}...")

    print("\nsplits")
    for split_name, split_paragraphs in combined.items():
        path = PROCESSED_DIR / f"{split_name}.txt"
        path.write_text("\n".join(split_paragraphs) + "\n", encoding="utf-8")
        words = sum(len(p.split()) for p in split_paragraphs)
        print(f"  {split_name:<5}: {len(split_paragraphs):>5,} paragraphs, "
              f"{words:>7,} words -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
