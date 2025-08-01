# Latent Dirichlet Allocation (LDA) Implementation using Coordinate Ascent Variational Inference

A complete implementation of Latent Dirichlet Allocation (LDA) topic modeling using Coordinate Ascent Variational Inference (CAVI) from scratch in Python. This project demonstrates advanced probabilistic machine learning techniques for unsupervised topic discovery in text corpora.

## Project Overview

This project implements LDA, a generative probabilistic model for topic modeling, using variational inference optimization. The implementation includes:

- **Custom LDA with CAVI**: Full implementation of the coordinate ascent variational inference algorithm
- **Topic Discovery**: Automatic identification of latent topics in document collections
- **Model Selection**: Optimal number of topics selection using held-out likelihood
- **Comprehensive Visualization**: Multiple visualization tools for model interpretation

## Dataset

The project analyzes **NEURIPS conference abstracts**, processing academic papers to discover underlying research topics and themes in machine learning and AI.

## Key Features

### Core Implementation

- **LDA_CAVI Class**: Complete implementation of LDA with variational inference
- **Evidence Lower Bound (ELBO)**: Convergence monitoring and optimization tracking
- **Model Parameters**: Automatic updates of θ (topic-document), β (word-topic), and φ (word-topic assignment) distributions
- **Hyperparameter Tuning**: Configurable Dirichlet priors (α, η)

### Advanced Functionality

- **Optimal K Selection**: Automated selection of optimal number of topics using cross-validation
- **Predictive Likelihood**: Held-out data evaluation for model comparison
- **Text Preprocessing**: Custom preprocessing pipeline with stop word removal and vectorization
- **Convergence Detection**: Automated convergence checking with configurable thresholds

### Visualization Suite

- **Topic Word Clouds**: Top words visualization for each discovered topic
- **Parameter Heatmaps**: Visual representation of λ, γ, β, and θ parameters
- **ELBO Convergence Plots**: Training progress monitoring
- **Model Comparison**: Likelihood comparison across different K values

## Technical Stack

- **Python 3.x**
- **NumPy**: Numerical computations and matrix operations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical functions (digamma, gamma functions)
- **Scikit-learn**: Text preprocessing and vectorization
- **Matplotlib/Seaborn**: Data visualization
- **Regular Expressions**: Text preprocessing

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Usage Example

```python
from LDA_CAVI import LDA_CAVI, LDA_Visualizer

# Load and preprocess data
lda_model = LDA_CAVI(document_term_matrix, K=4)

# Fit the model
lda_model.fit(max_iter=1000)

# Find optimal number of topics
best_k, likelihoods = lda_model.find_optimal_k(range(1, 11))

# Visualize results
visualizer = LDA_Visualizer(lda_model, vocabulary)
visualizer.plot_topics()
visualizer.plot_elbo(lda_model.elbo_values)
```

## Results & Analysis

### Model Performance

- **Convergence**: ELBO convergence achieved within 20-30 iterations
- **Optimal Topics**: Analysis shows optimal K=4 topics for the NEURIPS dataset
- **Topic Coherence**: Discovered topics show clear thematic separation in machine learning domains

### Key Findings

- **Topic 1**: Deep Learning and Neural Networks
- **Topic 2**: Optimization and Learning Theory
- **Topic 3**: Computer Vision and Image Processing
- **Topic 4**: Natural Language Processing and Text Analysis

## Mathematical Foundation

The implementation is based on the variational inference framework:

**ELBO Optimization**:

```
L(γ, λ, φ) = E_q[log p(w, z, θ, β)] + H(q)
```

**Parameter Updates**:

- **λ**: Word-topic distribution parameters
- **γ**: Document-topic distribution parameters
- **φ**: Variational topic assignment probabilities

## Project Structure

```
├── notebook.ipynb              # Main implementation and analysis
└── README.md                   # Project documentation
```

## Experimental Design

1. **Data Preprocessing**: Custom text preprocessing with stop word removal and frequency filtering
2. **Model Training**: CAVI algorithm with convergence monitoring
3. **Model Selection**: Cross-validation using held-out likelihood
4. **Evaluation**: Quantitative analysis using perplexity and likelihood metrics
5. **Visualization**: Comprehensive visual analysis of learned parameters

## Educational Value

This project demonstrates:

- **Probabilistic Modeling**: Implementation of complex generative models
- **Variational Inference**: Advanced optimization techniques for intractable posteriors
- **Topic Modeling**: Practical application of unsupervised learning
- **Scientific Computing**: Efficient numerical implementations using NumPy/SciPy

## Performance Metrics

- **ELBO Convergence**: Monitoring of optimization progress
- **Held-out Likelihood**: Model generalization evaluation
- **Topic Coherence**: Qualitative assessment of discovered topics
- **Computational Efficiency**: Optimized matrix operations for scalability

## Future Enhancements

- Implementation of other inference methods (Gibbs sampling, structured variational inference)
- Extension to hierarchical topic models (hLDA)
- Online learning capabilities for streaming data
- GPU acceleration for large-scale datasets

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic variational inference

---

_This project was developed as part of PhD coursework in Probabilistic Models and Machine Learning at Columbia University._
