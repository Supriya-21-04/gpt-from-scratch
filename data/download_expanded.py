from pathlib import Path
import re
import requests


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "corpus_expanded.txt"


# ============================================================
# GUTENBERG URL
# ============================================================

def gutenberg_url(book_id):
    return (
        f"https://www.gutenberg.org/cache/epub/"
        f"{book_id}/pg{book_id}.txt"
    )


# ============================================================
# 20-BOOK CORPUS
#
# Diverse English text:
# - Literature
# - Philosophy
# - Science
# - History / Biography
# - Essays
# - Reference / Technical
#
# These are the exact books whose Gutenberg URLs
# were verified successfully.
# ============================================================

BOOKS = {
    "literature": [
        ("Pride and Prejudice", 1342),
        ("Alice's Adventures in Wonderland", 11),
        ("The Adventures of Sherlock Holmes", 1661),
        ("Frankenstein", 84),
        ("The Picture of Dorian Gray", 174),
        ("A Tale of Two Cities", 98),
        ("Moby Dick", 2701),
    ],

    "philosophy": [
        ("The Republic", 1497),
        ("Meditations", 2680),
    ],

    "science": [
        ("On the Origin of Species", 2009),
        ("The Descent of Man", 2300),
        ("The Expression of the Emotions in Man and Animals", 1227),
    ],

    "history_biography": [
        ("The Autobiography of Benjamin Franklin", 148),
        ("The Life of George Washington", 12540),
        ("The History of the Peloponnesian War", 7142),
    ],

    "essays": [
        ("Essays by Ralph Waldo Emerson", 16643),
        ("Walden", 205),
        ("Civil Disobedience", 71),
    ],

    "reference_technical": [
        ("The Elements of Style", 37134),
        ("The Art of War", 17405),
    ],
}


# ============================================================
# HTTP SETTINGS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; GPT-from-scratch-corpus-builder/1.0)"
    )
}


# ============================================================
# DOWNLOAD
# ============================================================

def download_text(url):
    """
    Download a Gutenberg text file and decode it as UTF-8.
    """

    print(f"  Downloading: {url}")

    response = requests.get(
        url,
        timeout=60,
        headers=HEADERS,
    )

    response.raise_for_status()

    return response.content.decode(
        "utf-8",
        errors="replace",
    )


# ============================================================
# REMOVE GUTENBERG BOILERPLATE
# ============================================================

def clean_gutenberg_text(text):
    """
    Remove Gutenberg header/footer and normalize whitespace.

    We preserve:
    - words
    - punctuation
    - paragraph boundaries
    - capitalization
    - dialogue structure

    We remove:
    - Gutenberg metadata
    - excessive blank lines
    - unnecessary whitespace
    """

    # --------------------------------------------------------
    # Normalize line endings
    # --------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # --------------------------------------------------------
    # Remove Gutenberg START marker
    # --------------------------------------------------------

    start_patterns = [
        r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]

    for pattern in start_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            text = text[match.end():]
            break

    # --------------------------------------------------------
    # Remove Gutenberg END marker
    # --------------------------------------------------------

    end_patterns = [
        r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*END OF PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]

    for pattern in end_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            text = text[:match.start()]
            break

    # --------------------------------------------------------
    # Normalize spaces
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # --------------------------------------------------------
    # Remove spaces on otherwise empty lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    # --------------------------------------------------------
    # Collapse excessive blank lines
    #
    # Keep paragraph boundaries, but avoid huge gaps.
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # --------------------------------------------------------
    # Remove leading/trailing whitespace from each line
    # --------------------------------------------------------

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)
        else:
            # Preserve paragraph boundary
            if lines and lines[-1] != "":
                lines.append("")

    text = "\n".join(lines)

    return text.strip()


# ============================================================
# QUALITY CHECK
# ============================================================

def quality_check(text, title):
    """
    Perform basic corpus-quality checks.

    Returns True if the text looks usable.
    """

    passed = True

    character_count = len(text)

    # --------------------------------------------------------
    # Size check
    # --------------------------------------------------------

    if character_count < 10_000:
        print(
            f"  WARNING: {title} is unusually short "
            f"({character_count:,} characters)."
        )
        passed = False

    # --------------------------------------------------------
    # Replacement-character check
    #
    # \ufffd means decoding encountered a problematic
    # character.
    # --------------------------------------------------------

    replacement_count = text.count("\ufffd")

    if replacement_count > 0:
        print(
            f"  WARNING: {title} contains "
            f"{replacement_count} replacement characters."
        )

        if replacement_count > 100:
            passed = False

    # --------------------------------------------------------
    # Alphabetic-character ratio
    # --------------------------------------------------------

    if character_count > 0:

        alphabetic_count = sum(
            char.isalpha()
            for char in text
        )

        alphabetic_ratio = (
            alphabetic_count / character_count
        )

    else:
        alphabetic_ratio = 0.0

    if alphabetic_ratio < 0.50:
        print(
            f"  WARNING: {title} has low alphabetic "
            f"content ratio: {alphabetic_ratio:.2%}"
        )
        passed = False

    return passed


# ============================================================
# BUILD CORPUS
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING EXPANDED GPT CORPUS")
    print("=" * 70)

    print()
    print(f"Output file:")
    print(f"  {OUTPUT_PATH}")

    print()
    print("Original corpus will NOT be modified.")
    print()

    sections = []

    total_books = sum(
        len(books)
        for books in BOOKS.values()
    )

    completed = 0
    failed = []

    # --------------------------------------------------------
    # Download each category
    # --------------------------------------------------------

    for category, books in BOOKS.items():

        print()
        print("=" * 70)
        print(category.upper())
        print("=" * 70)

        for title, book_id in books:

            completed += 1

            print()
            print(
                f"Book {completed}/{total_books}: "
                f"{title}"
            )

            url = gutenberg_url(book_id)

            try:

                # ------------------------------------------------
                # Download
                # ------------------------------------------------

                raw_text = download_text(url)

                print(
                    f"  Downloaded: "
                    f"{len(raw_text):,} raw characters"
                )

                # ------------------------------------------------
                # Clean
                # ------------------------------------------------

                clean_text = clean_gutenberg_text(
                    raw_text
                )

                print(
                    f"  After cleaning: "
                    f"{len(clean_text):,} characters"
                )

                # ------------------------------------------------
                # Quality check
                # ------------------------------------------------

                quality_check(
                    clean_text,
                    title,
                )

                # ------------------------------------------------
                # Add category/title boundary
                #
                # This is useful because the model should learn
                # that documents are separate rather than having
                # the end of one book immediately merge into the
                # beginning of another.
                # ------------------------------------------------

                section = (
                    "\n\n"
                    + "=" * 70
                    + "\n"
                    + f"CATEGORY: {category.upper()}\n"
                    + f"TITLE: {title}\n"
                    + "=" * 70
                    + "\n\n"
                    + clean_text
                    + "\n"
                )

                sections.append(section)

                print("  STATUS: OK")

            except Exception as error:

                print(
                    f"  FAILED: {error}"
                )

                failed.append(
                    (category, title, book_id)
                )

    # ========================================================
    # CHECK WHETHER ANY BOOKS WERE SUCCESSFULLY DOWNLOADED
    # ========================================================

    if not sections:
        raise RuntimeError(
            "No books were successfully downloaded. "
            "Corpus was not created."
        )

    # ========================================================
    # COMBINE
    # ========================================================

    corpus = "\n".join(sections)

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.write_text(
        corpus,
        encoding="utf-8",
    )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    vocabulary = sorted(
        set(corpus)
    )

    print()
    print("=" * 70)
    print("CORPUS CREATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(f"Output file:")
    print(f"  {OUTPUT_PATH}")

    print()
    print(f"Books attempted:    {total_books}")
    print(f"Books downloaded:    {len(sections)}")
    print(f"Books failed:        {len(failed)}")

    print()
    print(
        f"Total characters:    {len(corpus):,}"
    )

    print(
        f"Vocabulary size:     {len(vocabulary)}"
    )

    # --------------------------------------------------------
    # Character statistics
    # --------------------------------------------------------

    alphabetic = sum(
        char.isalpha()
        for char in corpus
    )

    whitespace = sum(
        char.isspace()
        for char in corpus
    )

    digits = sum(
        char.isdigit()
        for char in corpus
    )

    punctuation = sum(
        not char.isalnum() and not char.isspace()
        for char in corpus
    )

    print()
    print("Character statistics:")
    print(
        f"  Alphabetic:         {alphabetic:,}"
    )
    print(
        f"  Whitespace:         {whitespace:,}"
    )
    print(
        f"  Digits:             {digits:,}"
    )
    print(
        f"  Punctuation:        {punctuation:,}"
    )

    # --------------------------------------------------------
    # Category statistics
    # --------------------------------------------------------

    print()
    print("Categories:")

    for category, books in BOOKS.items():
        print(
            f"  {category:22s} "
            f"{len(books):2d} books"
        )

    # --------------------------------------------------------
    # Failed downloads
    # --------------------------------------------------------

    if failed:

        print()
        print("FAILED BOOKS:")

        for category, title, book_id in failed:
            print(
                f"  {category}: "
                f"{title} "
                f"(ID {book_id})"
            )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print()
    print("First 1,000 characters of the corpus:")
    print("-" * 70)
    print(corpus[:1000])
    print("-" * 70)

    print()
    print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()