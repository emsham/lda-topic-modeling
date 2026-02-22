#!/usr/bin/env python
"""CLI entry point for training and evaluating an LDA topic model.

Example usage::

    python run.py --data-dir ./NEURIPS_abstracts --topics 4
    python run.py --data-dir ./NEURIPS_abstracts --find-optimal-k 1 10
    python run.py --data-dir ./NEURIPS_abstracts --topics 8 --max-iter 500 --alpha 0.5
"""

import argparse
import sys

import numpy as np
import matplotlib

from lda_topic_modeling.preprocessing import (
    build_vectorizer,
    create_count_dataframe,
    load_documents,
)
from lda_topic_modeling.model import LDA_CAVI
from lda_topic_modeling.visualization import LDA_Visualizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train a Latent Dirichlet Allocation (LDA) topic model using "
            "Coordinate Ascent Variational Inference (CAVI)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python run.py --data-dir ./NEURIPS_abstracts --topics 4\n"
            "  python run.py --data-dir ./NEURIPS_abstracts --find-optimal-k 1 10\n"
            "  python run.py --data-dir ./NEURIPS_abstracts --topics 8 --alpha 0.5 --eta 0.5\n"
        ),
    )

    parser.add_argument(
        "--data-dir", required=True,
        help="Path to directory containing .txt document files.",
    )
    parser.add_argument(
        "--topics", "-k", type=int, default=4,
        help="Number of topics K (default: 4).",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Dirichlet prior on document-topic proportions (default: 1.0).",
    )
    parser.add_argument(
        "--eta", type=float, default=1.0,
        help="Dirichlet prior on topic-word distributions (default: 1.0).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000,
        help="Maximum CAVI iterations (default: 1000).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of documents held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--min-df", type=float, default=0.25,
        help="Minimum document frequency for vocabulary (default: 0.25).",
    )
    parser.add_argument(
        "--max-df", type=float, default=0.85,
        help="Maximum document frequency for vocabulary (default: 0.85).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--find-optimal-k", nargs=2, type=int, metavar=("START", "END"),
        help="Search for the optimal K in range [START, END].",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable matplotlib plots (useful for headless environments).",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.no_plot:
        matplotlib.use("Agg")

    # ---- Load & preprocess ------------------------------------------------
    print(f"Loading documents from {args.data_dir} ...")
    documents_df = load_documents(args.data_dir)
    print(f"  {len(documents_df)} documents loaded.")

    vectorizer = build_vectorizer(min_df=args.min_df, max_df=args.max_df)
    count_df = create_count_dataframe(documents_df, vectorizer)
    vocabulary = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(vocabulary)}")

    np.random.seed(args.seed)

    # ---- Optimal-K search -------------------------------------------------
    if args.find_optimal_k:
        start_k, end_k = args.find_optimal_k
        k_range = range(start_k, end_k + 1)
        lda = LDA_CAVI(
            count_df, K=start_k, alpha=args.alpha, eta=args.eta,
            test_size=args.test_size, random_state=args.seed,
        )
        best_k, likelihoods = lda.find_optimal_k(k_range, max_iter=args.max_iter)
        print(f"\nOptimal K = {best_k}")
        for k, ll in zip(k_range, likelihoods):
            print(f"  K={k:>3d}  held-out ll = {ll:.3f}")

        if not args.no_plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.plot(list(k_range), likelihoods, marker="o")
            plt.title("Held-out Likelihood vs Number of Topics")
            plt.xlabel("Number of Topics (K)")
            plt.ylabel("Held-out Likelihood")
            plt.axvline(x=best_k, color="red", linestyle="--",
                        label=f"Best K = {best_k}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return

    # ---- Single-K training ------------------------------------------------
    print(f"\nTraining LDA with K={args.topics} ...")
    lda = LDA_CAVI(
        count_df, K=args.topics, alpha=args.alpha, eta=args.eta,
        test_size=args.test_size, random_state=args.seed,
    )
    lda.fit(max_iter=args.max_iter)

    # ---- Report -----------------------------------------------------------
    print("\n--- Top words per topic ---")
    vis = LDA_Visualizer(lda, vocabulary)
    for t in range(args.topics):
        words = vis.get_top_words(t)
        print(f"  Topic {t + 1}: {', '.join(words)}")

    if not args.no_plot:
        vis.plot_elbo(lda.elbo_values)
        vis.plot_topics()
        vis.plot_betas()
        vis.plot_thetas()


if __name__ == "__main__":
    main()
