
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "corpus_expanded.txt"
OUTPUT_PATH = BASE_DIR / "corpus_expanded_clean.txt"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Could not find {INPUT_PATH}")

text = INPUT_PATH.read_text(encoding="utf-8")
original_chars = len(text)

# ---------------------------------------------------------
# 1. Remove Gutenberg START/END markers
# ---------------------------------------------------------
text = re.sub(
    r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    "",
    text,
    flags=re.IGNORECASE | re.DOTALL,
)

text = re.sub(
    r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    "",
    text,
    flags=re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------
# 2. Remove illustration blocks
# ---------------------------------------------------------
text = re.sub(
    r"\[Illustration:.*?\]",
    "",
    text,
    flags=re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------
# 3. Remove standalone illustration/image markers
# ---------------------------------------------------------
text = re.sub(
    r"(?m)^\s*\[(?:Illustration|Image)\]\s*$",
    "",
    text,
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------
# 4. Remove common Gutenberg image/caption markers
# ---------------------------------------------------------
text = re.sub(
    r"\[Image:.*?\]",
    "",
    text,
    flags=re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------
# 5. Remove obvious OCR/image caption lines
# ---------------------------------------------------------
text = re.sub(
    r"(?m)^\s*_[^_\n]{1,120}_\s*$",
    "",
    text,
)

# ---------------------------------------------------------
# 6. Remove separator lines
# ---------------------------------------------------------
text = re.sub(
    r"(?m)^\s*[=_-]{10,}\s*$",
    "",
    text,
)

# ---------------------------------------------------------
# 7. Remove our synthetic CATEGORY/TITLE metadata
# ---------------------------------------------------------
text = re.sub(
    r"(?m)^\s*CATEGORY:\s*.*$",
    "",
    text,
)

text = re.sub(
    r"(?m)^\s*TITLE:\s*.*$",
    "",
    text,
)

# ---------------------------------------------------------
# 8. Normalize line endings
# ---------------------------------------------------------
text = text.replace("\r\n", "\n")
text = text.replace("\r", "\n")

# ---------------------------------------------------------
# 9. Remove trailing spaces
# ---------------------------------------------------------
text = re.sub(
    r"[ \t]+$",
    "",
    text,
    flags=re.MULTILINE,
)

# ---------------------------------------------------------
# 10. Collapse excessive blank lines
# ---------------------------------------------------------
text = re.sub(
    r"\n{4,}",
    "\n\n\n",
    text,
)

# ---------------------------------------------------------
# 11. Remove leading/trailing whitespace
# ---------------------------------------------------------
text = text.strip()

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
OUTPUT_PATH.write_text(
    text,
    encoding="utf-8"
)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------
clean_chars = len(text)

print("=" * 70)
print("CORPUS CLEANING COMPLETE")
print("=" * 70)

print(f"Input file : {INPUT_PATH}")
print(f"Output file: {OUTPUT_PATH}")

print()
print(f"Original characters : {original_chars:,}")
print(f"Clean characters    : {clean_chars:,}")
print(f"Characters removed  : {original_chars - clean_chars:,}")

if original_chars > 0:
    reduction = (
        (original_chars - clean_chars)
        / original_chars
        * 100
    )
    print(f"Reduction           : {reduction:.2f}%")

print()
print("First 2000 characters of cleaned corpus:")
print("-" * 70)
print(text[:2000])
print("-" * 70)
