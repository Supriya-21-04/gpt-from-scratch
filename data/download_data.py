from pathlib import Path
import requests


# Project Gutenberg's plain-text version of The Count of Monte Cristo.
URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"

# This points to:
# gpt-from-scratch/data/corpus.txt
OUTPUT_PATH = Path(__file__).parent / "corpus.txt"


def download_corpus():
    """Download the corpus and save it locally."""

    print("Downloading corpus...")

    response = requests.get(URL, timeout=30)

    # Raise an error immediately if the server returned something
    # like 404 or 500 instead of silently saving bad data.
    response.raise_for_status()

    text = response.text

    # UTF-8 lets us preserve characters that are not plain ASCII.
    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print(f"Saved corpus to: {OUTPUT_PATH}")
    print(f"Characters: {len(text):,}")
    print(f"Approximate size: {OUTPUT_PATH.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    download_corpus()