"""End-to-end tests: raw text -> NLTK preprocessing -> TF-IDF -> cosine similarity.

These cover the five required scenarios. They need NLTK (the first run may
download stopwords and wordnet, so an internet connection is required once).
Run from the project root:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plagiarism import HIGH_LABEL, SimilarityError, classify_similarity, compare_documents  # noqa: E402
from preprocessing import (  # noqa: E402
    clean_text,
    ensure_nltk_resources,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    tokenize,
)

SAMPLES_DIR = PROJECT_ROOT / "samples"


def setUpModule():
    ready, message = ensure_nltk_resources()
    if not ready:
        raise unittest.SkipTest(message)


def read_sample(name: str) -> str:
    return (SAMPLES_DIR / name).read_text(encoding="utf-8")


def similarity_of(text_a: str, text_b: str) -> float:
    result = compare_documents(preprocess_text(text_a), preprocess_text(text_b))
    return result.similarity_percentage


class FiveRequiredScenarios(unittest.TestCase):
    """Expected behaviour for the five scenarios listed in the project brief."""

    def test_1_identical_documents(self):
        # Expected: 100% similarity, 0% difference, High Similarity.
        text = read_sample("original.txt")
        result = compare_documents(preprocess_text(text), preprocess_text(text))
        self.assertEqual(result.similarity_percentage, 100.0)
        self.assertEqual(result.difference_percentage, 0.0)
        self.assertEqual(classify_similarity(result.similarity_percentage), HIGH_LABEL)

    def test_2_completely_different_documents(self):
        # Expected: essentially no shared vocabulary, so a very low score.
        score = similarity_of(read_sample("original.txt"), read_sample("different.txt"))
        self.assertLess(score, 10)

    def test_3_highly_similar_documents(self):
        # Expected: only a few words changed, so the score stays high.
        score = similarity_of(read_sample("original.txt"), read_sample("similar.txt"))
        self.assertGreater(score, 70)
        self.assertLess(score, 100)

    def test_4_paraphrased_documents(self):
        # Expected: TF-IDF measures shared wording, not meaning. A paraphrase
        # scores above unrelated text but well below a lightly edited copy.
        original = read_sample("original.txt")
        paraphrase = similarity_of(original, read_sample("paraphrased.txt"))
        different = similarity_of(original, read_sample("different.txt"))
        similar = similarity_of(original, read_sample("similar.txt"))
        self.assertGreater(paraphrase, different)
        self.assertLess(paraphrase, similar)
        self.assertLess(paraphrase, 50)

    def test_5_empty_or_invalid_documents(self):
        # Expected: a clear SimilarityError, never a crash or a fake score.
        stopwords_only = read_sample("stopwords_only.txt")
        for bad_text in ("", "   \n ", "!!! ??? ...", stopwords_only):
            with self.subTest(bad_text=bad_text[:20]):
                with self.assertRaises(SimilarityError):
                    compare_documents(preprocess_text(bad_text), preprocess_text("valid words here"))


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_lowercases_and_strips_punctuation(self):
        self.assertEqual(clean_text("Hello, WORLD!!! Visit https://example.com now."), "hello world visit now")

    def test_clean_text_keeps_contractions_but_drops_quotes(self):
        self.assertEqual(clean_text("Don't 'quote' me"), "don't quote me")

    def test_clean_text_folds_accents(self):
        self.assertEqual(clean_text("Café"), "cafe")

    def test_clean_text_handles_invalid_input(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")
        self.assertEqual(clean_text("   "), "")

    def test_tokenize_splits_words(self):
        self.assertEqual(tokenize("one two  three"), ["one", "two", "three"])
        self.assertEqual(tokenize(""), [])

    def test_remove_stopwords_drops_fillers_and_single_letters(self):
        self.assertEqual(remove_stopwords(["the", "cat", "is", "on", "a", "mat", "x"]), ["cat", "mat"])

    def test_lemmatize_tokens(self):
        self.assertEqual(lemmatize_tokens(["cats", "running", "studies"]), ["cat", "run", "study"])

    def test_full_pipeline(self):
        self.assertEqual(preprocess_text("The cats are running!"), "cat run")

    def test_pipeline_returns_empty_string_for_unusable_input(self):
        self.assertEqual(preprocess_text(None), "")
        self.assertEqual(preprocess_text("the and of"), "")


if __name__ == "__main__":
    unittest.main()
