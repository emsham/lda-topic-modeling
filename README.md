# LDA Topic Modeling via Coordinate Ascent Variational Inference

A from-scratch implementation of **Latent Dirichlet Allocation (LDA)** using **Coordinate Ascent Variational Inference (CAVI)** in Python. Developed as a class research project for the PhD-level *Probabilistic Models and Machine Learning* course at Columbia University.

## Overview

LDA is a generative probabilistic model that discovers latent topics in document collections. This project implements the full CAVI optimisation loop — including variational parameter updates for **phi** (word-topic assignments), **gamma** (document-topic proportions), and **lambda** (topic-word distributions) — with automatic convergence detection, held-out likelihood evaluation, and a suite of visualisation tools.

## Project Structure

```
lda-topic-modeling/
├── lda_topic_modeling/          # Python package
│   ├── __init__.py              # Public API exports
│   ├── model.py                 # LDA_CAVI model (training, inference, evaluation)
│   ├── preprocessing.py         # Text cleaning, vectorisation, document loading
│   └── visualization.py         # Plotting utilities (ELBO, topics, heatmaps)
├── tests/                       # Unit tests (pytest)
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_visualization.py
├── notebook.ipynb               # Interactive walkthrough and analysis
├── run.py                       # CLI entry point
├── pyproject.toml               # Build config and dependencies
├── requirements.txt             # Runtime dependencies
├── .gitignore
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/lda-topic-modeling.git
cd lda-topic-modeling

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Dataset

The project analyses **NeurIPS conference paper abstracts** to discover latent research topics. Place the dataset directory (containing one `.txt` file per abstract) at the path expected by the notebook or passed via `--data-dir`.

You can obtain the NeurIPS abstracts dataset from [Kaggle](https://www.kaggle.com/datasets) or scrape them from the [NeurIPS proceedings](https://papers.nips.cc/).

## Usage

### CLI

Train a model with 4 topics:

```bash
python run.py --data-dir ./NEURIPS_abstracts --topics 4
```

Search for the optimal number of topics:

```bash
python run.py --data-dir ./NEURIPS_abstracts --find-optimal-k 1 10
```

Custom hyperparameters:

```bash
python run.py --data-dir ./NEURIPS_abstracts --topics 8 --alpha 0.5 --eta 0.5 --max-iter 500
```

Full flag reference:

```bash
python run.py --help
```

### Notebook

Open `notebook.ipynb` in Jupyter for an interactive walkthrough with inline visualisations. The notebook imports all functionality from the `lda_topic_modeling` package.

```bash
jupyter notebook notebook.ipynb
```

### Library API

```python
from lda_topic_modeling import (
    LDA_CAVI, LDA_Visualizer,
    load_documents, build_vectorizer, create_count_dataframe,
)

# Load and preprocess
docs = load_documents("./NEURIPS_abstracts")
vectorizer = build_vectorizer(min_df=0.25, max_df=0.85)
dtm = create_count_dataframe(docs, vectorizer)

# Train
model = LDA_CAVI(dtm, K=4, alpha=1.0, eta=1.0)
model.fit(max_iter=1000)

# Visualise
vis = LDA_Visualizer(model, vectorizer.get_feature_names_out())
vis.plot_topics()
vis.plot_elbo(model.elbo_values)

# Model selection
best_k, likelihoods = model.find_optimal_k(range(1, 11))
```

## Implementation Details

| Component | Description |
|---|---|
| **Inference** | Coordinate Ascent Variational Inference (CAVI) |
| **ELBO** | `E_q[log p(w,z,theta,beta)] + H(q)` — monitored each iteration |
| **Convergence** | Relative ELBO change < epsilon for *n* consecutive iterations |
| **Model selection** | Held-out log-likelihood over a grid of K values |
| **Preprocessing** | Lowercasing, punctuation removal, stop-word filtering, TF thresholds |
| **Evaluation** | Log-likelihood on held-out words from test documents |

## Results Summary

- ELBO convergence typically within 20-30 iterations.
- Optimal topic count K = 4 for the NeurIPS abstracts corpus (selected via held-out likelihood).
- Discovered topics show clear thematic separation across machine learning sub-fields (e.g., optimisation theory, deep learning, probabilistic methods, applications).

## Testing

All tests use small synthetic data and require no external datasets.

```bash
pytest
```

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. JMLR.
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). *Stochastic Variational Inference*. JMLR.

---

*Developed as a class research project for PhD-level Probabilistic Models and Machine Learning at Columbia University.*
