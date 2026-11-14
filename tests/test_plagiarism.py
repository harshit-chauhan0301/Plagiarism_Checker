"""Unit tests for plagiarism.py (TF-IDF + cosine similarity).

These tests use already-preprocessed text, so they need only scikit-learn.
Run from the project root:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plagiarism import (  # noqa: E402
    HIGH_LABEL,
    LOW_LABEL,
    MODERATE_LABEL,
    SimilarityError,
    classify_similarity,
    compare_documents,
)


class CompareDocumentsTests(unittest.TestCase):
    def test_identical_documents_score_100(self):
        result = compare_documents("cat dog bird", "cat dog bird")
        self.assertEqual(result.similarity_percentage, 100.0)
        self.assertEqual(result.difference_percentage, 0.0)

    def test_no_shared_words_score_0(self):
        result = compare_documents("cat dog bird", "volcano ash lava")
        self.assertEqual(result.similarity_percentage, 0.0)
        self.assertEqual(result.difference_percentage, 100.0)
        self.assertEqual(result.common_keywords, ())

    def test_partial_overlap_matches_hand_calculation(self):
        # Shared terms get idf 1.0, unique terms get idf ln(3/2)+1 ~ 1.405:
        # cosine = (1*1 + 1*1) / (1.405^2 + 1 + 1) = 2 / 3.975 ~ 0.503
        result = compare_documents("cat dog bird", "cat dog fish")
        self.assertAlmostEqual(result.similarity_percentage, 50.31, places=1)

    def test_score_is_symmetric(self):
        a, b = "red green blue yellow", "green blue purple"
        self.assertEqual(
            compare_documents(a, b).similarity_percentage,
            compare_documents(b, a).similarity_percentage,
        )

    def test_common_keywords_contain_only_shared_terms(self):
        result = compare_documents("cat dog bird", "cat dog fish")
        words = [word for word, _ in result.common_keywords]
        self.assertEqual(words, ["cat", "dog"])
        self.assertEqual(result.shared_term_count, 2)
        self.assertEqual(result.vocabulary_size, 4)

    def test_top_n_limits_keywords_but_not_shared_count(self):
        result = compare_documents("cat dog bird", "cat dog fish", top_n_keywords=1)
        self.assertEqual(len(result.common_keywords), 1)
        self.assertEqual(result.shared_term_count, 2)


class InvalidInputTests(unittest.TestCase):
    def test_empty_string_raises(self):
        with self.assertRaises(SimilarityError):
            compare_documents("", "cat dog")

    def test_whitespace_only_raises(self):
        with self.assertRaises(SimilarityError):
            compare_documents("cat dog", "   \n\t ")

    def test_non_string_input_raises(self):
        with self.assertRaises(SimilarityError):
            compare_documents(None, "cat dog")  # type: ignore[arg-type]

    def test_document_of_single_letters_has_no_comparable_words(self):
        # scikit-learn ignores one-character tokens, leaving an all-zero vector.
        with self.assertRaises(SimilarityError):
            compare_documents("a b c", "cat dog")

    def test_empty_vocabulary_raises(self):
        with self.assertRaises(SimilarityError):
            compare_documents("a b", "c d")

    def test_error_message_names_the_bad_document(self):
        with self.assertRaises(SimilarityError) as context:
            compare_documents("cat dog", "")
        self.assertIn("Document 2", str(context.exception))


class ClassificationTests(unittest.TestCase):
    def test_default_thresholds(self):
        self.assertEqual(classify_similarity(10), LOW_LABEL)
        self.assertEqual(classify_similarity(55), MODERATE_LABEL)
        self.assertEqual(classify_similarity(85), HIGH_LABEL)

    def test_boundaries_belong_to_the_higher_band(self):
        self.assertEqual(classify_similarity(40, 40, 70), MODERATE_LABEL)
        self.assertEqual(classify_similarity(70, 40, 70), HIGH_LABEL)
        self.assertEqual(classify_similarity(39.99, 40, 70), LOW_LABEL)

    def test_custom_thresholds_change_the_label(self):
        self.assertEqual(classify_similarity(30, 20, 25), HIGH_LABEL)
        self.assertEqual(classify_similarity(30, 50, 90), LOW_LABEL)

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            classify_similarity(50, 80, 40)
        with self.assertRaises(ValueError):
            classify_similarity(50, -5, 40)
        with self.assertRaises(ValueError):
            classify_similarity(50, 40, 120)


if __name__ == "__main__":
    unittest.main()
