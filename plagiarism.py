"""TF-IDF vectorization and cosine similarity for two documents.

The functions here receive text that has ALREADY been preprocessed
(see preprocessing.py) and know nothing about Streamlit, so they can be
tested and reused on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOW_LABEL = "Low Similarity"
MODERATE_LABEL = "Moderate Similarity"
HIGH_LABEL = "High Similarity"

_DOCUMENT_NAMES = ("Document 1", "Document 2")


class SimilarityError(ValueError):
    """Raised when two documents cannot be compared (empty text, empty vocabulary, ...)."""


@dataclass(frozen=True)
class SimilarityResult:
    """Everything the UI needs to display after comparing two documents."""

    similarity_percentage: float
    difference_percentage: float
    # (keyword, average TF-IDF weight) pairs, strongest first
    common_keywords: Tuple[Tuple[str, float], ...]
    vocabulary_size: int
    shared_term_count: int


def _validate_documents(doc_a: object, doc_b: object) -> None:
    for text, name in zip((doc_a, doc_b), _DOCUMENT_NAMES):
        if not isinstance(text, str) or not text.strip():
            raise SimilarityError(
                f"{name} has no usable words after preprocessing. It may be empty "
                "or contain only stopwords or symbols."
            )


def vectorize_documents(
    doc_a: str,
    doc_b: str,
    ngram_range: Tuple[int, int] = (1, 1),
):
    """Convert two preprocessed documents into TF-IDF vectors.

    Returns:
        (sparse matrix with one row per document, array of feature names)

    Raises:
        SimilarityError: for empty documents or an empty TF-IDF vocabulary.
    """
    _validate_documents(doc_a, doc_b)

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    try:
        matrix = vectorizer.fit_transform([doc_a, doc_b])
    except ValueError as exc:  # scikit-learn: "empty vocabulary"
        raise SimilarityError(
            "The documents produced an empty TF-IDF vocabulary. Make sure they "
            "contain real words, not only stopwords or symbols."
        ) from exc

    # A document whose words were all filtered out would give an all-zero vector.
    for row, name in enumerate(_DOCUMENT_NAMES):
        if matrix[row].nnz == 0:
            raise SimilarityError(f"{name} has no words that can be compared.")

    return matrix, vectorizer.get_feature_names_out()


def compute_cosine_similarity(matrix) -> float:
    """Cosine similarity (0.0 - 1.0) between the two rows of a TF-IDF matrix."""
    score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    if np.isnan(score):
        return 0.0
    # Floating-point noise can give 1.0000000000000002 for identical texts.
    return min(max(score, 0.0), 1.0)


def find_common_keywords(
    matrix,
    feature_names,
    top_n: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Terms that appear in both documents, ranked by average TF-IDF weight."""
    dense = matrix.toarray()
    shared_indices = np.flatnonzero((dense[0] > 0) & (dense[1] > 0))

    keywords = [
        (str(feature_names[i]), round(float((dense[0][i] + dense[1][i]) / 2), 4))
        for i in shared_indices
    ]
    # Strongest first; ties broken alphabetically so the output is deterministic.
    keywords.sort(key=lambda item: (-item[1], item[0]))
    return keywords if top_n is None else keywords[:top_n]


def compare_documents(
    preprocessed_a: str,
    preprocessed_b: str,
    ngram_range: Tuple[int, int] = (1, 1),
    top_n_keywords: Optional[int] = None,
) -> SimilarityResult:
    """Compare two preprocessed documents with TF-IDF + cosine similarity."""
    matrix, feature_names = vectorize_documents(preprocessed_a, preprocessed_b, ngram_range)

    score = compute_cosine_similarity(matrix)
    similarity = round(score * 100, 2)
    keywords = find_common_keywords(matrix, feature_names, top_n_keywords)

    # shared_term_count is computed from the full list, not the truncated one.
    shared_count = len(keywords) if top_n_keywords is None else len(
        find_common_keywords(matrix, feature_names)
    )

    return SimilarityResult(
        similarity_percentage=similarity,
        difference_percentage=round(100 - similarity, 2),
        common_keywords=tuple(keywords),
        vocabulary_size=len(feature_names),
        shared_term_count=shared_count,
    )


def classify_similarity(
    similarity_percentage: float,
    low_threshold: float = 40,
    high_threshold: float = 70,
) -> str:
    """Label a similarity percentage using adjustable thresholds.

    below low_threshold          -> Low Similarity
    low_threshold .. high_threshold -> Moderate Similarity
    high_threshold and above     -> High Similarity
    """
    if not 0 <= low_threshold <= high_threshold <= 100:
        raise ValueError("Thresholds must satisfy 0 <= low <= high <= 100.")

    if similarity_percentage >= high_threshold:
        return HIGH_LABEL
    if similarity_percentage >= low_threshold:
        return MODERATE_LABEL
    return LOW_LABEL
