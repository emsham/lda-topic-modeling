"""Visualisation utilities for inspecting trained LDA models."""

import numpy as np
import matplotlib.pyplot as plt


class LDA_Visualizer:
    """Plots for an :class:`~lda_topic_modeling.model.LDA_CAVI` model.

    Args:
        lda_cavi_instance: A fitted :class:`LDA_CAVI` object.
        vocabulary: Sequence of vocabulary terms matching the columns of
            the training document-term matrix.
        top_n: Number of top words to display per topic.
    """

    def __init__(self, lda_cavi_instance, vocabulary, top_n=10):
        self.lda = lda_cavi_instance
        self.vocabulary = vocabulary
        self.top_n = top_n

    def get_top_words(self, topic_id):
        """Return the *top_n* most probable words for a topic.

        Args:
            topic_id: Zero-based topic index.

        Returns:
            List of vocabulary strings ordered by decreasing probability.
        """
        word_probs = self.lda.beta_k()[topic_id]
        top_word_indices = np.argsort(word_probs)[-self.top_n:][::-1]
        return [self.vocabulary[idx] for idx in top_word_indices]

    def plot_topics(self):
        """Display the top words for every topic as a text figure."""
        beta = self.lda.beta_k()
        n_topics = beta.shape[0]

        fig, axes = plt.subplots(nrows=n_topics, ncols=1,
                                 figsize=(10, n_topics * 3))
        if n_topics == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            word_probs = beta[i]
            top_indices = np.argsort(word_probs)[-self.top_n:][::-1]
            words = [self.vocabulary[idx] for idx in top_indices]
            ax.axis('off')
            ax.text(
                0.5, 0.5,
                f'Topic {i + 1}:\n' + '\n'.join(words),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                bbox=dict(facecolor='gray', alpha=0.1),
            )

        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_lambda(self):
        """Heatmap of the lambda (topic-word) variational parameters."""
        plt.figure(figsize=(10, 6))
        plt.imshow(self.lda.lambda_kv, aspect='auto')
        plt.colorbar()
        plt.title('Lambda values over iterations')
        plt.xlabel('Word Index')
        plt.ylabel('Topic Index')
        plt.show()

    def plot_elbo(self, elbo_values):
        """Line plot of ELBO values across training iterations.

        Args:
            elbo_values: Sequence of ELBO scalars.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(elbo_values)
        plt.title('ELBO values over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('ELBO value')
        plt.show()

    def plot_gammas(self):
        """Heatmap of gamma (document-topic) variational parameters."""
        plt.figure(figsize=(10, 6))
        plt.imshow(self.lda.gamma_ik, aspect='auto')
        plt.colorbar()
        plt.title('Gamma values over iterations')
        plt.xlabel('Document Index')
        plt.ylabel('Topic Index')
        plt.show()

    def plot_betas(self):
        """Heatmap of sampled beta (word-topic) distributions."""
        plt.figure(figsize=(10, 6))
        plt.imshow(self.lda.beta_k(), aspect='auto')
        plt.colorbar()
        plt.title('Beta values over topics and words')
        plt.xlabel('Word Index')
        plt.ylabel('Topic Index')
        plt.show()

    def plot_thetas(self):
        """Heatmap of normalised theta (document-topic) proportions."""
        theta_values = (
            self.lda.gamma_ik
            / np.sum(self.lda.gamma_ik, axis=1)[:, np.newaxis]
        )
        plt.figure(figsize=(10, 6))
        plt.imshow(theta_values, aspect='auto')
        plt.colorbar()
        plt.title('Theta values over documents and topics')
        plt.xlabel('Topic Index')
        plt.ylabel('Document Index')
        plt.show()
