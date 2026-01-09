# Engineering a Recommender System for the MovieLens 32M Dataset

This repository contains a high-performance implementation of **Regularized Matrix Factorization** using **Alternating Least Squares (ALS)**, specifically optimized for the MovieLens 32M dataset. The system addresses the computational challenges of scaling collaborative filtering to 32 million interactions by utilizing hardware-aware systems engineering.

## Performance Benchmarks

The implementation was evaluated on a consumer-grade workstation (Intel Core i7-14650HX, 16GB RAM) across four architectural tiers. By transitioning from an interpreted baseline to a hardware-aware parallel engine, the system achieves a cumulative **8,767x speedup**.



| Optimization Tier | Method | Epoch Time | Speedup |
| :--- | :--- | :--- | :--- |
| Naive Baseline | Python Interpreter | 3.55 Hours | 1x |
| Dual CSR Architecture | Algorithmic Refinement | 20.0 Minutes | 10.7x |
| JIT Engine (Sequential) | LLVM JIT Compilation | 38.0 Seconds | 336x |
| JIT Engine (Parallel) | Thread-Level Parallelism | **1.46 Seconds** | **8,767x** |

* **Total Training Duration**: 72.92 seconds for 50 iterations.
* **Peak Throughput**: Approximately 21.9 million ratings per second.
* **Predictive Accuracy**: Final Test RMSE of 0.7659.

## Engineering Optimizations

### 1. Dual Compressed Sparse Representation
ALS requires efficient row-wise access for user updates and column-wise access for item updates. To eliminate the $O(N)$ complexity of slicing a standard CSR matrix by column, the system maintains two synchronized views:



* **User-CSR (Compressed Sparse Row)**: Facilitates $O(1)$ retrieval of all items associated with a specific user.
* **Item-CSC (Compressed Sparse Column)**: Facilitates $O(1)$ retrieval of all users associated with a specific item.

### 2. Hardware Acceleration via Numba JIT
The implementation utilizes Numba to translate high-level Python logic into optimized machine code at runtime.



* **LLVM-Driven Vectorization**: Enables SIMD (Single Instruction, Multiple Data) vectorization and loop unrolling, allowing multiple floating-point operations to execute in a single clock cycle.
* **Thread-Level Parallelism**: Utilizes `numba.prange` to distribute independent user and item updates across all available CPU cores, scaling throughput linearly with hardware threads.

## Qualitative Analysis and Utility

### Semantic Manifold Discovery
The model autonomously recovers meaningful semantic structures and genre clusters from interaction patterns without access to item metadata.



Analysis of the $K=2$ projection reveals a thematic organization along latent axes:
* **Vertical Axis ($K_2$)**: Acts as a thematic intensity gradient separating family-oriented media ($K_2 < 0$) from visceral, adult-oriented cinema ($K_2 > 0$).
* **Horizontal Axis ($K_1$)**: Differentiates between mainstream fictional narratives and factual or niche content ($K_1 < -1$).

### Downstream Applications
The learned latent representations provide a foundation for various business intelligence tasks:
* **User Churn Prediction**: Monitoring "interest drift" as users move toward low-density regions of the latent manifold.
* **Customer Lifetime Value (LTV)**: Utilizing learned taste profiles (user vectors $p_u$) as inputs for regression models to predict future revenue.
* **Anomaly Detection**: Utilizing residuals ($\epsilon_{ui} = |r_{ui} - \hat{r}_{ui}|$) to detect account sharing or malicious shilling attacks.



## Author
**Mohammed Nagi**
African Institute for Mathematical Sciences (AIMS), South Africa.
