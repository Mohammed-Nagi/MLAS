# Engineering a Scalable Recommender System for the MovieLens 32M Dataset

## Overview
[cite_start]This repository contains the implementation of a high-performance Collaborative Filtering system designed to handle the **MovieLens 32M dataset**[cite: 116, 120]. [cite_start]The project documents an engineering transition from a naive Python-based Matrix Factorisation approach to a highly optimised solution utilising **Just-In-Time (JIT) compilation** and **Sparse Data Structures**[cite: 116, 121].

[cite_start]The core objective is to predict user preferences via **Regularized Matrix Factorization** using the **Alternating Least Squares (ALS)** algorithm[cite: 120, 125].

## Technical Highlights
* [cite_start]**Scalability:** Successfully processes 32,000,204 ratings across 200,948 users and 84,432 movies[cite: 122, 142].
* [cite_start]**Optimization:** Achieved orders-of-magnitude speedups by moving from standard iterative loops to **Numba-optimized, parallelized kernels**.
* [cite_start]**Memory Efficiency:** Implemented a **Dual Compressed Sparse Row (CSR)** structure to manage a dataset with 99.81% sparsity, allowing for efficient row and column-wise access during the ALS update steps.
* [cite_start]**Latent Space Analysis:** The model learns semantically meaningful embeddings that reveal genre structures (e.g., "Children" vs. "Horror") purely from interaction data.

## Repository Structure
* [cite_start]`Baseline_Implementation.ipynb`: A naive Python implementation demonstrating the computational bottlenecks of standard iterative approaches at scale[cite: 121, 128].
* [cite_start]`Optimised_Implementation.ipynb`: The high-performance solution featuring Numba JIT compilation, vectorised residuals, and parallelised ALS updates[cite: 121].
* `esameldin_AMLAS_draft.pdf`: A detailed technical report documenting the engineering decisions, mathematical framework, and performance evaluation.

## Performance
[cite_start]The final optimised model reduces training time from computationally infeasible durations to **minutes on limited compute resources**, while maintaining competitive **RMSE (Root Mean Square Error)** scores[cite: 122, 129].

| Metric | Value |
| :--- | :--- |
| Total Ratings | 32,000,204 |
| Optimization Tool | Numba (JIT) |
| Data Structure | Dual CSR Indexing |
| Training Time | < 10 Minutes (Optimized) |

## Requirements
* Python 3.x
* NumPy
* Numba
* Pandas
* Matplotlib / Seaborn

## References
* Koren, Y., Bell, R. M., and Volinsky, C. (2009). Matrix factorization techniques for recommender systems.
* Koenigstein, N. and Paquet, U. (2011). The Xbox recommender system.
