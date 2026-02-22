"""Text preprocessing utilities for LDA topic modeling.

Provides functions for cleaning raw text and converting document collections
into document-term matrices suitable for topic model training.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


def custom_preprocessor(text):
    """Convert text to lowercase and strip punctuation.

    Args:
        text: Raw input string.

    Returns:
        Cleaned lowercase string with punctuation removed.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def create_count_dataframe(data, vectorizer):
    """Build a document-term count DataFrame from raw text.

    Applies the given vectorizer to the ``"content"`` column of *data* and
    returns a DataFrame whose rows are documents, columns are vocabulary
    terms, and the index is the document ``"id"``.

    Args:
        data: DataFrame with at least ``"content"`` and ``"id"`` columns.
        vectorizer: A fitted or unfitted
            :class:`~sklearn.feature_extraction.text.CountVectorizer`.

    Returns:
        DataFrame of shape ``(n_documents, n_features)`` indexed by
        document id.
    """
    count_values = vectorizer.fit_transform(data["content"])
    count_df = pd.DataFrame(
        count_values.toarray(), columns=vectorizer.get_feature_names_out()
    )
    count_df["id"] = data["id"].reset_index(drop=True)
    count_df.set_index("id", inplace=True)
    return count_df


def build_vectorizer(stop_words=None, min_df=0.25, max_df=0.85):
    """Create a :class:`CountVectorizer` with sensible defaults for LDA.

    Args:
        stop_words: Iterable of stop words to exclude.  Defaults to
            scikit-learn's built-in English stop words.
        min_df: Minimum document frequency (fraction or count).
        max_df: Maximum document frequency (fraction or count).

    Returns:
        A configured :class:`CountVectorizer` instance (not yet fitted).
    """
    if stop_words is None:
        stop_words = list(ENGLISH_STOP_WORDS)
    return CountVectorizer(
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r'(?u)\b\w\w+\b',
        preprocessor=custom_preprocessor,
    )


def load_documents(folder_path):
    """Read all ``.txt`` files in a directory into a DataFrame.

    Args:
        folder_path: Path to the directory containing text files.

    Returns:
        DataFrame with columns ``"id"`` (filename) and ``"content"``
        (file text).
    """
    import os

    data = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                data.append({"id": filename, "content": f.read()})
    return pd.DataFrame(data)
