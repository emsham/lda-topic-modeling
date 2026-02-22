"""Latent Dirichlet Allocation with Coordinate Ascent Variational Inference.

Implements the LDA generative model for topic discovery in text corpora,
optimised via CAVI (Coordinate Ascent Variational Inference).
"""

import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.special import psi, gammaln, digamma
from sklearn.model_selection import train_test_split


class LDA_CAVI:
    """LDA trained with Coordinate-Ascent Variational Inference (CAVI).

    Args:
        X: Document-term matrix (DataFrame).  Rows are documents, columns
            are vocabulary terms.
        K: Number of topics.
        alpha: Dirichlet prior on per-document topic proportions.
        eta: Dirichlet prior on per-topic word distributions.
        test_size: Fraction of documents held out for evaluation.
        random_state: Seed for reproducibility.
        iter_for_theta: Number of iterations when estimating topic
            proportions for new documents.
    """

    def __init__(self, X, K=2, alpha=1, eta=1, test_size=0.2,
                 random_state=42, iter_for_theta=1000):
        self.X = X
        self.random_state = random_state
        self.X_train, self.X_test = train_test_split(
            X, test_size=test_size, random_state=self.random_state
        )
        self.K = K
        self.alpha = alpha
        self.eta = eta
        self.N, self.V = self.X_train.shape
        self.iter_for_theta = iter_for_theta
        self.test_obs_df, self.test_new_df = self._generate_obs_new_df(self.X_test)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init(self):
        """Initialise all variational parameters."""
        self.lambda_kv = self._lambda_init(method="small_values")
        self.gamma_ik = self._gamma_init()
        self.phi_ijk = self._phi_init()

    def _generate_obs_new_df(self, df):
        """Split columns of *df* into two halves for held-out evaluation.

        Args:
            df: DataFrame whose columns will be shuffled and split.

        Returns:
            Tuple ``(observed_df, new_df)`` each containing half the
            vocabulary columns.
        """
        shuffled_columns = (
            pd.Series(df.columns.to_list())
            .sample(frac=1, random_state=self.random_state)
            .tolist()
        )
        midpoint = len(shuffled_columns) // 2
        return df[shuffled_columns[midpoint:]], df[shuffled_columns[:midpoint]]

    def _lambda_init(self, method="uniform"):
        """Initialise the topic-word variational parameter lambda.

        Args:
            method: One of ``"uniform"``, ``"small_values"``,
                ``"dirichlet"``, or ``"uniform_noisy"``.

        Returns:
            Array of shape ``(K, V)``.
        """
        if method == "uniform":
            return np.ones((self.K, self.V))
        elif method == "small_values":
            return np.random.uniform(0.001, 0.01, size=(self.K, self.V))
        elif method == "dirichlet":
            return np.random.dirichlet([self.alpha] * self.V, size=self.K)
        elif method == "uniform_noisy":
            noise = np.random.uniform(0.001, 0.01, size=(self.K, self.V))
            return np.ones((self.K, self.V)) + noise
        raise ValueError(f"Unknown init method: {method}")

    def _gamma_init(self):
        """Initialise gamma (document-topic) with ones.

        Returns:
            Array of shape ``(N, K)``.
        """
        return np.ones((self.N, self.K))

    def _phi_init(self):
        """Initialise phi (word-topic assignment) uniformly.

        Returns:
            Array of shape ``(N, V, K)``.
        """
        phi_ijk = np.empty((self.N, self.V, self.K))
        phi_ijk[:] = 1.0 / self.K
        return phi_ijk

    # ------------------------------------------------------------------
    # ELBO computation
    # ------------------------------------------------------------------

    def get_elbo(self):
        """Compute the Evidence Lower Bound (ELBO).

        Returns:
            Scalar ELBO value.
        """
        return self._expected_log_joint() + self._negative_entropy()

    def _expected_log_joint(self):
        """Expected log-joint under the variational distribution.

        Returns:
            Scalar value.
        """
        expected_log_theta = self._expected_log_dirichlet(self.gamma_ik)
        expected_log_beta = self._expected_log_dirichlet(self.lambda_kv)
        return np.sum(
            self.phi_ijk
            * (expected_log_theta[:, np.newaxis, :] + expected_log_beta.T[np.newaxis, :, :])
        )

    @staticmethod
    def _expected_log_dirichlet(param_matrix):
        """Compute E[log Dir(x; params)] element-wise.

        Args:
            param_matrix: Parameter array where rows are independent
                Dirichlet distributions.

        Returns:
            Array of same shape as *param_matrix*.
        """
        return psi(param_matrix) - psi(np.sum(param_matrix, axis=1, keepdims=True))

    def _negative_entropy(self):
        """Negative entropy of the variational distribution.

        Returns:
            Scalar value.
        """
        neg_entropy_beta = -np.sum(
            [self._dirichlet_entropy(alpha) for alpha in self.lambda_kv]
        )
        neg_entropy_theta = -np.sum(
            [self._dirichlet_entropy(alpha) for alpha in self.gamma_ik]
        )
        neg_entropy_z = -np.sum(self.phi_ijk * np.log(self.phi_ijk + 1e-10))
        return neg_entropy_beta + neg_entropy_theta + neg_entropy_z

    @staticmethod
    def _dirichlet_entropy(alpha):
        """Entropy of a Dirichlet distribution.

        Args:
            alpha: Parameter vector.

        Returns:
            Scalar entropy value.
        """
        return (
            gammaln(np.sum(alpha))
            - np.sum(gammaln(alpha))
            + np.sum((alpha - 1.0) * (psi(alpha) - psi(np.sum(alpha))))
        )

    # ------------------------------------------------------------------
    # Variational parameter updates
    # ------------------------------------------------------------------

    def _update_phi_ijk(self):
        """Update word-topic assignment probabilities phi."""
        for i in range(self.N):
            for j in range(self.V):
                for k in range(self.K):
                    self.phi_ijk[i, j, k] = np.exp(
                        self._expected_log_beta_kv(k, j)
                        + self._expected_log_theta_ik(i, k)
                    )
        phi_sum = np.sum(self.phi_ijk, axis=2)[:, :, np.newaxis]
        phi_sum = np.where(phi_sum == 0, 1e-10, phi_sum)
        self.phi_ijk /= phi_sum

    def _update_gamma_ik(self):
        """Update document-topic variational parameters gamma."""
        for i in range(self.N):
            for k in range(self.K):
                self.gamma_ik[i, k] = self.alpha + np.sum(self.phi_ijk[i, :, k])

    def _update_lambda_kv(self):
        """Update topic-word variational parameters lambda."""
        if self.phi_ijk.ndim != 3:
            raise ValueError("phi_ijk must be a 3-dimensional array")
        X_train = (
            self.X_train.values
            if isinstance(self.X_train, pd.DataFrame)
            else self.X_train
        )
        expected_n_kv = np.einsum('ij,ijk->kj', X_train, self.phi_ijk)
        self.lambda_kv = self.eta + expected_n_kv

    def _expected_log_theta_ik(self, i, k):
        """E[log theta_{ik}].

        Args:
            i: Document index.
            k: Topic index.

        Returns:
            Scalar expected log value.
        """
        return digamma(self.gamma_ik[i, k]) - digamma(np.sum(self.gamma_ik[i, :]))

    def _expected_log_beta_kv(self, k, v):
        """E[log beta_{kv}].

        Args:
            k: Topic index.
            v: Word index.

        Returns:
            Scalar expected log value.
        """
        return digamma(self.lambda_kv[k, v]) - digamma(np.sum(self.lambda_kv[k, :]))

    def _cavi(self):
        """Perform one full CAVI iteration (phi -> gamma -> lambda)."""
        self._update_phi_ijk()
        self._update_gamma_ik()
        self._update_lambda_kv()

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def beta_hat(self):
        """Point estimate of the word-topic distribution.

        Returns:
            Array of shape ``(K, V)`` with rows summing to 1.
        """
        return self.lambda_kv / self.lambda_kv.sum(axis=1, keepdims=True)

    def beta_k(self):
        """Sample word-topic distributions from the posterior Dirichlet.

        Returns:
            Array of shape ``(K, V)`` sampled from Dir(eta + n_k).
        """
        cond_beta_k = np.empty((self.K, self.V))
        for k in range(self.K):
            n_k = self._compute_n_k(k)
            post_params = self.eta + n_k
            cond_beta_k[k, :] = np.random.dirichlet(alpha=post_params)
        return cond_beta_k

    def _compute_n_k(self, k):
        """Expected word counts for topic *k* across all documents.

        Args:
            k: Topic index.

        Returns:
            Array of shape ``(V,)``.
        """
        X_vals = (
            self.X_train.values
            if isinstance(self.X_train, pd.DataFrame)
            else self.X_train
        )
        return np.sum(X_vals * self.phi_ijk[:, :, k], axis=0)

    def theta_i_hat(self):
        """Estimate topic proportions for each test document.

        Returns:
            Array of shape ``(n_test, K)``.
        """
        theta_hat = []
        for _index, new_doc in self.test_obs_df.iterrows():
            theta_hat.append(self._compute_topic_proportions(new_doc))
        return np.array(theta_hat)

    def _compute_topic_proportions(self, new_doc):
        """Infer topic proportions for a single unseen document.

        Args:
            new_doc: Series of word counts indexed by vocabulary terms.

        Returns:
            Array of shape ``(K,)`` summing to 1.
        """
        gamma_new = np.ones(self.K) * self.alpha

        observed_words = new_doc.index.intersection(self.X_train.columns)
        observed_word_indices = [
            self.X_train.columns.get_loc(w) for w in observed_words
        ]
        observed_counts = new_doc[observed_words].values

        phi_new = np.ones((len(observed_word_indices), self.K)) / self.K

        for _ in range(self.iter_for_theta):
            for j, word_idx in enumerate(observed_word_indices):
                phi_new[j, :] = np.exp(
                    sp.digamma(gamma_new)
                    + sp.digamma(self.lambda_kv[:, word_idx])
                    - sp.digamma(np.sum(self.lambda_kv, axis=1))
                )
            phi_new /= np.sum(phi_new, axis=1, keepdims=True)
            gamma_new = self.alpha + np.dot(observed_counts, phi_new)

        return gamma_new / np.sum(gamma_new)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def calculate_log_likelihood(self, theta_hat, beta_hat, X_out):
        """Log-likelihood of held-out data.

        Args:
            theta_hat: Estimated topic proportions, shape ``(n_docs, K)``.
            beta_hat: Estimated word-topic matrix, shape ``(K, V)``.
            X_out: Held-out document-term DataFrame.

        Returns:
            Scalar log-likelihood.
        """
        log_likelihood = 0.0
        for i, (_index, doc) in enumerate(X_out.iterrows()):
            theta_i = theta_hat[i]
            for word, count in doc.items():
                if word in self.X_train.columns:
                    word_idx = self.X_train.columns.get_loc(word)
                    prob = np.dot(theta_i, beta_hat[:, word_idx])
                    log_likelihood += count * np.log(prob + 1e-10)
        return log_likelihood

    @staticmethod
    def has_converged(elbo_values, epsilon=1e-5, consecutive_iters=10):
        """Check whether the ELBO sequence has converged.

        Args:
            elbo_values: List of ELBO values recorded during training.
            epsilon: Relative-change threshold.
            consecutive_iters: Number of consecutive iterations that must
                satisfy the threshold.

        Returns:
            ``True`` if converged, ``False`` otherwise.
        """
        if len(elbo_values) < consecutive_iters + 1:
            return False
        for i in range(1, consecutive_iters + 1):
            if abs(
                (elbo_values[-i] - elbo_values[-i - 1]) / elbo_values[-i - 1]
            ) >= epsilon:
                return False
        return True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, max_iter=1000):
        """Train the LDA model via CAVI.

        Args:
            max_iter: Maximum number of CAVI iterations.

        Returns:
            ``self`` with fitted variational parameters.
        """
        self._init()
        self.elbo_values = [self.get_elbo()]
        self.lambda_kv_history = [self.lambda_kv.copy()]
        self.gamma_ik_history = [self.gamma_ik.copy()]

        for iter_ in range(1, max_iter + 1):
            self._cavi()
            self.lambda_kv_history.append(self.lambda_kv.copy())
            self.gamma_ik_history.append(self.gamma_ik.copy())
            elbo_current = self.get_elbo()
            self.elbo_values.append(elbo_current)

            if iter_ % 10 == 0:
                print(f'Iteration {iter_}: ELBO {elbo_current:.3f}')

            if self.has_converged(self.elbo_values, epsilon=1e-5,
                                  consecutive_iters=10):
                print(
                    f'ELBO converged with ll {elbo_current:.3f} '
                    f'at iteration {iter_} for k = {self.K}'
                )
                break
        else:
            print(f'ELBO ended with ll {self.elbo_values[-1]:.3f}')

        beta_hat_values = self.beta_hat()
        theta_i_hat_values = self.theta_i_hat()
        self.held_out = self.calculate_log_likelihood(
            theta_i_hat_values, beta_hat_values, self.test_new_df
        )

        return self

    def find_optimal_k(self, k_values, max_iter=1000):
        """Evaluate multiple topic counts and return the best.

        Args:
            k_values: Iterable of integers to try as the number of topics.
            max_iter: Maximum CAVI iterations per model.

        Returns:
            Tuple ``(best_k, likelihoods)`` where *best_k* is the topic
            count with the highest held-out likelihood and *likelihoods*
            is a list of likelihoods corresponding to each *k*.
        """
        best_k = None
        highest_likelihood = -np.inf
        likelihoods = []

        for k in k_values:
            print(f"Testing K={k}")
            lda_model = LDA_CAVI(
                self.X, K=k, alpha=self.alpha, eta=self.eta,
                test_size=0.2, random_state=self.random_state,
            )
            lda_model.fit(max_iter=max_iter)
            held_out_likelihood = lda_model.held_out
            likelihoods.append(held_out_likelihood)
            if held_out_likelihood > highest_likelihood:
                best_k = k
                highest_likelihood = held_out_likelihood

        return best_k, likelihoods
