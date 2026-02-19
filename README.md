# Parallel Time Series Reduction Engine

## 🚀 Project Overview
This project implements a high-performance computing (HPC) framework designed to generate and analyze massive datasets of synthetic financial time series. Built using **NVIDIA CUDA**, the engine offloads intensive mathematical computations—specifically random walk generation and parallel reduction from the CPU to the GPU.

## 🎯 Objectives
* **Parallel Synthesis:** Utilize `curand` to generate thousands of independent time series simultaneously.
* **Latency Optimization:** Implement a custom parallel reduction algorithm to aggregate data with minimal overhead.
* **Precision Engineering:** Evaluate the trade-off between **accuracy and speed** by testing different numerical data types.
* **Validation:** Provide a rigorous comparison between CPU and GPU execution to ensure numerical integrity.

---

## 🛠️ Technical Implementation

### 1. Memory Hierarchy Optimization
The engine is designed to maximize throughput by respecting the NVIDIA memory hierarchy:
* **Shared Memory Tiling:** Data is loaded into shared memory (L1 cache level) to facilitate high-speed inter-thread communication.
* **PCIe Bottleneck Reduction:** The system minimizes host-to-device transfers, focusing on keeping data on the device for the duration of the computation.
* **Coalesced Access:** Kernels are designed to ensure global memory accesses are coalesced, saturating GPU memory bandwidth.



### 2. Parallel Reduction Kernel (`Algo`)
The core `Algo` kernel performs a sum reduction:
* It handles multiple series in parallel, where each block processes a unique time series.
* Uses **Interleaved Addressing** during the final reduction phase to minimize warp divergence and bank conflicts.
* Includes synchronization primitives (`__syncthreads()`) to ensure data consistency across parallel threads.

### 3. Synthetic Data Generation
The `generateSerie` kernel simulates a "random walk" for price action:
* **Initial Value:** $100.0$.
* **Volatility:** Configurable max percentage change per step.
* **Randomness:** Uses `curand_init` with a unique seed per thread to ensure statistical independence.

---

## 📊 Performance & Verification
The system produces a `report.txt` containing:
1.  **Execution Time:** Measured in seconds for the GPU kernel vs. CPU baseline.
2.  **Error Counter:** Tracks any result where the GPU-CPU difference exceeds the defined `epsilon` ($1.0e-1$).
3.  **Success Rate:** A percentage representing the reliability of the parallel computation.

---

## 💻 How to Run
### Prerequisites
* NVIDIA GPU with Compute Capability 3.0+
* CUDA Toolkit installed

### Compilation
```bash
nvcc -o series_engine AlgoCUDA.cu
```
---
### Execution
```bash
./series_engine
```
