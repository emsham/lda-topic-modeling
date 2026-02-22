"""Tests for the preprocessing module."""

import pandas as pd
import pytest

from lda_topic_modeling.preprocessing import (
    build_vectorizer,
    create_count_dataframe,
    custom_preprocessor,
)


# ---------------------------------------------------------------------------
# custom_preprocessor
# ---------------------------------------------------------------------------

class TestCustomPreprocessor:
    def test_lowercases_text(self):
        assert custom_preprocessor("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert custom_preprocessor("it's a test!") == "its a test"

    def test_preserves_whitespace(self):
        assert custom_preprocessor("a  b") == "a  b"

    def test_empty_string(self):
        assert custom_preprocessor("") == ""

    def test_numeric_strings_preserved(self):
        assert custom_preprocessor("year 2023") == "year 2023"

    def test_mixed_punctuation(self):
        result = custom_preprocessor("hello-world, foo.bar!baz")
        assert result == "helloworld foobarbaz"


# ---------------------------------------------------------------------------
# create_count_dataframe
# ---------------------------------------------------------------------------

class TestCreateCountDataframe:
    @pytest.fixture()
    def sample_data(self):
        return pd.DataFrame({
            "id": ["doc1", "doc2", "doc3", "doc4"],
            "content": [
                "apple banana apple",
                "banana cherry cherry",
                "apple cherry banana",
                "cherry apple cherry",
            ],
        })

    def test_returns_dataframe(self, sample_data):
        vec = build_vectorizer(stop_words=[], min_df=1, max_df=1.0)
        result = create_count_dataframe(sample_data, vec)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_id(self, sample_data):
        vec = build_vectorizer(stop_words=[], min_df=1, max_df=1.0)
        result = create_count_dataframe(sample_data, vec)
        assert result.index.name == "id"
        assert list(result.index) == ["doc1", "doc2", "doc3", "doc4"]

    def test_word_counts_correct(self, sample_data):
        vec = build_vectorizer(stop_words=[], min_df=1, max_df=1.0)
        result = create_count_dataframe(sample_data, vec)
        assert result.loc["doc1", "apple"] == 2
        assert result.loc["doc1", "banana"] == 1

    def test_stop_words_excluded(self, sample_data):
        vec = build_vectorizer(stop_words=["banana"], min_df=1, max_df=1.0)
        result = create_count_dataframe(sample_data, vec)
        assert "banana" not in result.columns


# ---------------------------------------------------------------------------
# build_vectorizer
# ---------------------------------------------------------------------------

class TestBuildVectorizer:
    def test_default_returns_vectorizer(self):
        vec = build_vectorizer()
        assert hasattr(vec, "fit_transform")

    def test_custom_stop_words(self):
        vec = build_vectorizer(stop_words=["the", "a"])
        assert vec.stop_words == ["the", "a"]

    def test_min_max_df_set(self):
        vec = build_vectorizer(min_df=0.1, max_df=0.9)
        assert vec.min_df == 0.1
        assert vec.max_df == 0.9
