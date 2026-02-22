"""Tests for the visualization module."""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

from lda_topic_modeling.model import LDA_CAVI
from lda_topic_modeling.visualization import LDA_Visualizer


@pytest.fixture()
def fitted_vis():
    """Return an LDA_Visualizer backed by a tiny fitted model."""
    rng = np.random.RandomState(0)
    dtm = pd.DataFrame(
        rng.poisson(3, size=(20, 10)),
        columns=[f"w{i}" for i in range(10)],
        index=[f"d{i}" for i in range(20)],
    )
    model = LDA_CAVI(dtm, K=2, random_state=0, iter_for_theta=5)
    model.fit(max_iter=3)
    vocab = list(dtm.columns)
    return LDA_Visualizer(model, vocab, top_n=5)


class TestVisualizerInit:
    def test_top_n_stored(self, fitted_vis):
        assert fitted_vis.top_n == 5

    def test_vocabulary_stored(self, fitted_vis):
        assert len(fitted_vis.vocabulary) == 10


class TestGetTopWords:
    def test_returns_correct_count(self, fitted_vis):
        words = fitted_vis.get_top_words(0)
        assert len(words) == 5

    def test_words_are_strings(self, fitted_vis):
        words = fitted_vis.get_top_words(0)
        assert all(isinstance(w, str) for w in words)


class TestPlots:
    """Smoke tests: ensure plotting functions complete without error."""

    def test_plot_topics(self, fitted_vis):
        import matplotlib.pyplot as plt
        fitted_vis.plot_topics()
        plt.close("all")

    def test_plot_lambda(self, fitted_vis):
        import matplotlib.pyplot as plt
        fitted_vis.plot_lambda()
        plt.close("all")

    def test_plot_elbo(self, fitted_vis):
        import matplotlib.pyplot as plt
        fitted_vis.plot_elbo(fitted_vis.lda.elbo_values)
        plt.close("all")

    def test_plot_gammas(self, fitted_vis):
        import matplotlib.pyplot as plt
        fitted_vis.plot_gammas()
        plt.close("all")

    def test_plot_betas(self, fitted_vis):
        import matplotlib.pyplot as plt
        fitted_vis.plot_betas()
        plt.close("all")

    def test_plot_thetas(self, fitted_vis):
        import matplotlib.pyplot as plt
        fitted_vis.plot_thetas()
        plt.close("all")
