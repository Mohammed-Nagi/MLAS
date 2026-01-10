# Engineering a Recommender System for the MovieLens 32M Dataset

This repository contains a high-performance implementation of **Regularized Matrix Factorization** using **Alternating Least Squares (ALS)**, specifically optimized for the MovieLens 32M dataset. The system addresses the computational challenges of scaling collaborative filtering to 32 million interactions by utilizing hardware-aware systems engineering, transitioning from a standard interpreted Python baseline to a JIT-compiled parallel engine.

## Repository Structure

To ensure the successful execution of the provided notebook, the data files and project structure must be organized as follows:

* `ml-32m/`: Contains the primary production dataset.
    * `ratings.csv`: User-item interaction data (32,000,204 ratings).
    * `movies.csv`: Movie metadata including titles and pipe-delimited genre labels.
* `ml-latest-small/`: Contains the baseline evaluation dataset.
    * `ratings.csv`: Small-scale interaction data for rapid prototyping.
* `MovieLens_32M_HighPerformance_ALS.ipynb`: The primary research and execution notebook.
* `32m_model_k2.npz`: Compressed archive of trained (k=2) model parameters used for direct 2D projection of the item embedding space.
* `Engineering a Recommender System for the MovieLens 32M Dataset.pdf`: Technical report documenting the methodology and results.

## Performance Benchmarks

The implementation was evaluated on a consumer-grade workstation (Intel Core i7-14650HX, 16GB RAM). By transitioning to a hardware-aware parallel engine, the system achieves a cumulative **8,767x speedup**.



| Optimization Tier | Method | Epoch Time | Speedup |
| :--- | :--- | :--- | :--- |
| Naive Baseline | Python Interpreter | 3.55 Hours | 1x |
| Dual CSR Architecture | Algorithmic Refinement | 20.0 Minutes | 10.7x |
| JIT Engine (Sequential) | LLVM JIT Compilation | 38.0 Seconds | 336x |
| JIT Engine (Parallel) | Thread-Level Parallelism | **1.46 Seconds** | **8,767x** |

* **Peak Throughput**: 21.9 million ratings per second.
* **Final Accuracy**: Test RMSE of **0.7654** (Hierarchical Model).

## Installation and Usage

### Prerequisites
The implementation requires a Python 3.x environment. The following libraries are essential for maintaining the reported throughput:
* **Numba**: Just-In-Time (JIT) compilation and LLVM-backed parallelization.
* **NumPy & SciPy**: High-performance linear algebra and sparse matrix operations.
* **Pandas**: Efficient data ingestion and manipulation.
* **Matplotlib & Seaborn**: Scientific visualization.

### Setup Instructions
1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/Mohammed-Nagi/MLAS.git](https://github.com/Mohammed-Nagi/MLAS.git)
    cd MLAS
    ```
2.  **Install Dependencies**:
    ```bash
    pip install numpy pandas numba scipy matplotlib seaborn scikit-learn
    ```
3.  **Execute the Notebook**: Launch `MovieLens_32M_HighPerformance_ALS.ipynb`. The code is structured sequentially to allow for reproducibility:
    * **Seeding**: A global seed (42) is set at the start to ensure consistent latent initialization and data splitting.
    * **Data Ingestion**: Section 3 utilizes vectorized mapping to convert raw IDs into zero-indexed contiguous arrays.
    * **Training**: Section 10 contains the final **Hierarchical ALS** implementation.

## Engineering Optimizations

### 1. Dual Compressed Sparse Representation
ALS requires row-wise access for user updates and column-wise access for item updates. To eliminate $O(N)$ pointer-chasing overhead during column slicing, the system maintains synchronized, read-only views:



* **User-CSR (Compressed Sparse Row)**: Optimized for $O(1)$ retrieval of item vectors during user updates.
* **Item-CSC (Compressed Sparse Column)**: Optimized for $O(1)$ retrieval of user vectors during item updates.

### 2. Hardware Acceleration via Numba JIT
The implementation utilizes Numba to translate Python logic into native machine code at runtime, bypassing the Global Interpreter Lock (GIL).



* **LLVM-Driven Vectorization**: Enables SIMD execution for dot products and Ridge Regression solves.
* **Thread-Level Parallelism**: Distributes independent parameter updates across all available hardware threads via `numba.prange`.

## Hierarchical Integration of Metadata

To mitigate noise in sparse interaction data (the "Long-Tail" problem), the system utilizes a **Hierarchical Latent Factor** framework. Item latent vectors are regularized toward a genre-informed prior—the average embedding of the item's associated genres—rather than the origin ($\mathbf{0}$). This "Bayesian Shrinkage" stabilizes the latent manifold for items with low interaction counts.



## Qualitative Analysis

The model recovers meaningful semantic clusters autonomously. Analysis of the $K=2$ projection reveals clear thematic organization along the latent axes:



* **Vertical Axis ($K_2$)**: Differentiates between demographic-specific content (e.g., Children's Animation vs. Adult Horror).
* **Horizontal Axis ($K_1$)**: Separates mainstream fictional narratives from factual or niche documentary formats.

## Utility Beyond Recommendation
The learned embeddings and calculated residuals serve as a generalized foundation for system integrity:
* **Account Sharing Detection**: Identifying profiles with sustained high residuals indicative of divergent taste profiles.
* **Anomaly Monitoring**: Flagging "shilling attacks" or malicious ratings that deviate from the organic user-item manifold.



## Author
**Mohammed Nagi** MSc Artificial Intelligence Student (Google DeepMind Scholar)  
African Institute for Mathematical Sciences (AIMS), South Africa.
