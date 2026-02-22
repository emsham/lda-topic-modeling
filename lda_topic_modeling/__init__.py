"""LDA Topic Modeling — Latent Dirichlet Allocation via CAVI.

A from-scratch implementation of LDA using Coordinate Ascent
Variational Inference for unsupervised topic discovery in text corpora.
"""

from lda_topic_modeling.model import LDA_CAVI
from lda_topic_modeling.visualization import LDA_Visualizer
from lda_topic_modeling.preprocessing import (
    build_vectorizer,
    create_count_dataframe,
    custom_preprocessor,
    load_documents,
)

__all__ = [
    "LDA_CAVI",
    "LDA_Visualizer",
    "build_vectorizer",
    "create_count_dataframe",
    "custom_preprocessor",
    "load_documents",
]
