"""Tests for the LDA_CAVI model."""

import numpy as np
import pandas as pd
import pytest

from lda_topic_modeling.model import LDA_CAVI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dtm(n_docs=20, vocab_size=15, seed=0):
    """Create a small synthetic document-term matrix for testing."""
    rng = np.random.RandomState(seed)
    counts = rng.poisson(lam=3, size=(n_docs, vocab_size))
    columns = [f"word_{i}" for i in range(vocab_size)]
    index = [f"doc_{i}" for i in range(n_docs)]
    return pd.DataFrame(counts, columns=columns, index=index)


@pytest.fixture()
def small_dtm():
    return _make_synthetic_dtm()


@pytest.fixture()
def fitted_model(small_dtm):
    model = LDA_CAVI(small_dtm, K=2, alpha=1.0, eta=1.0,
                     test_size=0.2, random_state=42, iter_for_theta=10)
    model.fit(max_iter=5)
    return model


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:
    def test_shapes(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=3, test_size=0.25, random_state=0)
        model._init()
        N, V, K = model.N, model.V, model.K
        assert model.lambda_kv.shape == (K, V)
        assert model.gamma_ik.shape == (N, K)
        assert model.phi_ijk.shape == (N, V, K)

    def test_phi_sums_to_one(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=0)
        model._init()
        totals = model.phi_ijk.sum(axis=2)
        np.testing.assert_allclose(totals, 1.0)

    def test_lambda_init_methods(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=0)
        for method in ("uniform", "small_values", "dirichlet", "uniform_noisy"):
            lam = model._lambda_init(method)
            assert lam.shape == (model.K, model.V)
            assert np.all(lam >= 0)

    def test_lambda_init_bad_method(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=0)
        with pytest.raises(ValueError):
            model._lambda_init("nonexistent")


# ---------------------------------------------------------------------------
# ELBO helpers
# ---------------------------------------------------------------------------

class TestELBOHelpers:
    def test_expected_log_dirichlet_shape(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=0)
        model._init()
        result = model._expected_log_dirichlet(model.gamma_ik)
        assert result.shape == model.gamma_ik.shape

    def test_dirichlet_entropy_scalar(self):
        alpha = np.array([1.0, 1.0, 1.0])
        ent = LDA_CAVI._dirichlet_entropy(alpha)
        assert np.isfinite(ent)

    def test_dirichlet_entropy_uniform_is_negative_log_beta(self):
        # For alpha = [1, 1], Dirichlet entropy = 0
        alpha = np.array([1.0, 1.0])
        ent = LDA_CAVI._dirichlet_entropy(alpha)
        np.testing.assert_allclose(ent, 0.0, atol=1e-10)

    def test_elbo_is_finite(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=0)
        model._init()
        assert np.isfinite(model.get_elbo())


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

class TestDerivedQuantities:
    def test_beta_hat_sums_to_one(self, fitted_model):
        beta = fitted_model.beta_hat()
        row_sums = beta.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_beta_hat_shape(self, fitted_model):
        beta = fitted_model.beta_hat()
        assert beta.shape == (fitted_model.K, fitted_model.V)

    def test_beta_k_shape(self, fitted_model):
        beta = fitted_model.beta_k()
        assert beta.shape == (fitted_model.K, fitted_model.V)

    def test_beta_k_rows_sum_to_one(self, fitted_model):
        beta = fitted_model.beta_k()
        np.testing.assert_allclose(beta.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

class TestHasConverged:
    def test_not_enough_values(self):
        assert not LDA_CAVI.has_converged([1.0, 2.0], consecutive_iters=10)

    def test_converged_flat_sequence(self):
        vals = [100.0] * 20
        assert LDA_CAVI.has_converged(vals, epsilon=1e-5, consecutive_iters=10)

    def test_not_converged_increasing(self):
        vals = list(range(20))
        assert not LDA_CAVI.has_converged(vals, epsilon=1e-5, consecutive_iters=10)

    def test_converged_tiny_changes(self):
        base = 1000.0
        vals = [base + i * 1e-8 for i in range(20)]
        assert LDA_CAVI.has_converged(vals, epsilon=1e-5, consecutive_iters=10)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_log_likelihood_is_negative(self, fitted_model):
        theta = fitted_model.theta_i_hat()
        beta = fitted_model.beta_hat()
        ll = fitted_model.calculate_log_likelihood(
            theta, beta, fitted_model.test_new_df
        )
        assert ll <= 0

    def test_log_likelihood_is_finite(self, fitted_model):
        theta = fitted_model.theta_i_hat()
        beta = fitted_model.beta_hat()
        ll = fitted_model.calculate_log_likelihood(
            theta, beta, fitted_model.test_new_df
        )
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# End-to-end sanity check
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_fit_runs_without_error(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, alpha=1.0, eta=1.0,
                         test_size=0.2, random_state=42, iter_for_theta=5)
        model.fit(max_iter=3)
        assert len(model.elbo_values) >= 2

    def test_elbo_recorded_each_iter(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=42, iter_for_theta=5)
        model.fit(max_iter=5)
        # initial + up to 5 iterations
        assert len(model.elbo_values) >= 2
        assert len(model.elbo_values) <= 7

    def test_find_optimal_k(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=42, iter_for_theta=5)
        best_k, likelihoods = model.find_optimal_k([1, 2], max_iter=3)
        assert best_k in (1, 2)
        assert len(likelihoods) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_topic(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=1, random_state=0, iter_for_theta=5)
        model.fit(max_iter=3)
        assert model.beta_hat().shape[0] == 1

    def test_generate_obs_new_splits_columns(self, small_dtm):
        model = LDA_CAVI(small_dtm, K=2, random_state=0)
        obs, new = model._generate_obs_new_df(small_dtm)
        total_cols = obs.shape[1] + new.shape[1]
        assert total_cols == small_dtm.shape[1]
