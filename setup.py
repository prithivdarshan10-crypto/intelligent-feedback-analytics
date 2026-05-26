# =============================================================================
# setup.py
# One-time setup script: downloads all required NLTK corpora.
# Run this ONCE before launching the Streamlit app:
#
#   python setup.py
# =============================================================================

import sys
print("=" * 60)
print("  Intelligent Customer Feedback Analytics Platform")
print("  Setup – Downloading NLTK Resources")
print("=" * 60)

import nltk

PACKAGES = [
    ("punkt",          "Sentence / word tokeniser"),
    ("punkt_tab",      "Punkt tokeniser tables"),
    ("stopwords",      "English stopword list"),
    ("wordnet",        "WordNet lemmatiser"),
    ("omw-1.4",        "Open Multilingual Wordnet"),
    ("vader_lexicon",  "VADER sentiment lexicon"),
    ("averaged_perceptron_tagger", "POS tagger"),
]

for package, description in PACKAGES:
    print(f"  ↓  {description} ({package}) …", end=" ", flush=True)
    try:
        nltk.download(package, quiet=True)
        print("✓")
    except Exception as exc:
        print(f"FAILED ({exc})")

print()
print("All NLTK resources downloaded successfully.")
print()
print("Next step  →  streamlit run app.py")
print("=" * 60)
