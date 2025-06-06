# Compte Rendu - OpenCL Matrix Multiplication TP
## A. Matrix Multiplication Implementations: performances comparison
## 1. Introduction

This report presents the results and analysis of implementing matrix multiplication using OpenCL. The work focuses on optimizing performance across different implementations (coalesced, uncoalesced, and block-tiled) and devices (NVIDIA GPU, Intel integrated GPU, and Intel CPU).

## 2. Experimental Setup

### Hardware Configuration:
- NVIDIA GeForce RTX 3070 Laptop GPU (Platform 0)
    - Global memory:          7.99951171875 GB
    - Global cache:           1120.0 KB (READ_WRITE_CACHE)
    - Global cache line:      128 B
    - Local memory:           48.0 KB (LOCAL)
    - Constant memory:        64.0 KB
    - Compute units:          40
    - Max work-group size:    1024
    - Max work-item size:     [1024, 1024, 64]
    - Lockstep unit:          32
    - Max concurrent workgroups (est): 1280

- Intel UHD Graphics (Platform 1)
    - Global memory:          12.683395385742188 GB
    - Global cache:           512.0 KB (READ_WRITE_CACHE)
    - Global cache line:      64 B
    - Local memory:           64.0 KB (LOCAL)
    - Constant memory:        4194296.0 KB
    - Compute units:          32
    - Max work-group size:    256
    - Max work-item size:     [256, 256, 256]
    - Lockstep unit:          64
    - Max concurrent workgroups (est): 1792

- Intel Core i7-11800H CPU (Platform 2)
    - Global memory:          31.70849609375 GB
    - Global cache:           256.0 KB (READ_WRITE_CACHE)
    - Global cache line:      64 B
    - Local memory:           256.0 KB (GLOBAL)
    - Constant memory:        128.0 KB
    - Compute units:          16
    - Max work-group size:    8192
    - Max work-item size:     [8192, 8192, 8192]
    - Lockstep unit:          128
    - Max concurrent workgroups (est): 16

## 3. Performance Comparison: Sequential vs. OpenCL

| Implementation                | Matrix Size | Execution Time (s) | Performance (GFLOPS) | Operations         |
| ----------------------------- | ----------- | ------------------ | -------------------- | ------------------ |
| CPU Sequential                | 256         | 4.23               | 0.00792                | 1 matrix product   |
| CPU Sequential                | 512         | 36.67              | 0.00732               | 1 matrix product   |
| CPU OpenCL (Coalesced, LS=16) | 256         | 0.011              | 58.93                | 20 matrix products |
| CPU OpenCL (Coalesced, LS=16) | 512         | 0.093              | 57.79                | 20 matrix products |

### Analysis:
- OpenCL implementation achieves approximately 7,440× higher performance than sequential code for N=256
- The massive speedup is due to parallelization, cache optimization, and OpenCL compiler optimizations
- Sequential implementation scales poorly with problem size, while OpenCL maintains consistent performance

## 4. Devices Performance Analysis

### 4.1 NVIDIA GeForce RTX 3070 (Platform 0)

| Work Group Size | Coalesced (GFLOPS) | Uncoalesced (GFLOPS) | Block-Tiled (GFLOPS) |
| --------------- | ------------------ | -------------------- | -------------------- |
| **N=2048**      |                    |                      |                      |
| 2×2             | 1006.33            | 270.91               | 866.68               |
| 4×4             | 400.52             | 334.19               | 197.47               |
| 8×8             | 805.85             | 484.66               | 729.50               |
| 16×16           | 1011.30            | 270.31               | 877.66               |
| 32×32           | 971.21             | 142.84               | 828.26               |
| **N=4096**      |                    |                      |                      |
| 2×2             | 992.25             | 274.51               | 884.18               |
| 4×4             | 336.00             | 300.46               | 182.25               |
| 8×8             | 762.09             | 494.02               | 704.57               |
| 16×16           | 991.17             | 277.33               | 887.85               |
| 32×32           | 995.13             | 145.32               | 864.16               |
| **N=8192**      |                    |                      |                      |
| 2×2             | 993.93             | 268.55               | 916.74               |
| 4×4             | 316.56             | 301.24               | 177.03               |
| 8×8             | 749.49             | 494.68               | 699.77               |
| 16×16           | 978.75             | 268.28               | 913.93               |
| 32×32           | 1047.43            | 145.97               | 908.69    

#### **1. Observations:**
- **Coalesced access** consistently delivers the best performance
- **Uncoalesced memory access** is the worst performer, especially for larger work group sizes.
- For **coalesced access**, 32×32 work group size achieves peak performance (~1047 GFLOPS)
- **Uncoalesced access** performs best with 8×8 work groups (~495 GFLOPS)
- Block-tiled implementation performance is close to **coalesced access**

#### **2. Impact of Work Group Size**
- For **coalesced access**, 16×16 and 32×32 perform best—16×16 for smaller matrices and 32×32 for larger ones—due to efficient, aligned memory transactions that match NVIDIA's memory hierarchy. 
- **Uncoalesced access** performs best at 8×8, as smaller groups reduce memory divergence, larger sizes worsen performance due to scattered memory access patterns. 

- **Block-tiled** access shows stable performance across sizes, with 2×2 and 16×16 slightly better, benefiting from shared memory usage while minimizing synchronization overhead.

#### **3. Memory Access Patterns**

- **NVIDIA’s architecture is optimized for coalesced access**, where adjacent threads access adjacent memory locations, minimizing memory latency
- **Uncoalesced access suffers from memory divergence**, leading to poor performance at larger work group sizes.
- **Block-tiled implementations** benefit from shared memory optimizations but can be affected by synchronization overhead.

#### **4. Performance Scaling with Matrix Size**

- The **performance does not degrade significantly** with increasing matrix size.
- The **best implementations (coalesced, 16×16 & 32×32) maintain a high GFLOPS rate across all N values**.
- **For uncoalesced access, performance remains poor regardless of N**, but 8×8 continues to be the best configuration.

### 4.2 Intel UHD Graphics (Platform 1)

| Work Group Size | Coalesced (GFLOPS) | Uncoalesced (GFLOPS) | Block-Tiled (GFLOPS) |
| --------------- | ------------------ | -------------------- | -------------------- |
| **N=2048**      |                    |                      |                      |
| 2×2             | 33.34              | 4.90                 | 60.84                |
| 4×4             | 15.31              | 12.69                | 9.40                 |
| 8×8             | 26.18              | 8.13                 | 52.99                |
| 16×16           | 30.52              | 4.59                 | 60.81                |
| 32×32           | Error              | Error                | Error                |
| **N=4096**      |                    |                      |                      |
| 2×2             | 22.72              | 4.66                 | 59.24                |
| 4×4             | 12.43              | 8.09                 | 8.35                 |
| 8×8             | 17.60              | 7.29                 | 49.90                |
| 16×16           | 18.84              | 4.68                 | 59.83                |
| 32×32           | Error              | Error                | Error                |
| **N=8192**      |                    |                      |                      |
| 2×2             | 16.50              | 4.62                 | 57.29                |
| 4×4             | 9.14               | 5.48                 | 7.68                 |
| 8×8             | 14.49              | 6.72                 | 46.28                |
| 16×16           | 17.96              | 4.55                 | 57.56                |
| 32×32           | Error              | Error                | Error                |

#### **1. Observations:**
- Block-tiled implementation significantly outperforms other variants (57.56 GFLOPS with 16×16)
- Across all matrix sizes, block-tiled maintains a stable range of **~46-60 GFLOPS** due to **cache reuse and efficient local memory access.**

#### **2. Error Cases: 32×32 Work Group Size Fails** 

- The **32×32 work group size fails across all implementations** due to **Intel’s hardware limitation (max work group size = 256 threads).**

#### **3. Performance Trends** 

- **Coalesced Memory Access Degrades as Matrix Size Increases**, this is because Intel’s memory subsystem is not optimized for global memory coalescing and more cache misses occur as matrix size increases, reducing efficiency.

- **Uncoalesced Memory Access is Always Poor (~4-8 GFLOPS).** This is expected because **Uncoalesced memory access patterns** cause high latency.The reason is Intel’s **iGPU has limited L3 cache**, making **uncoalesced memory access** highly inefficient.

- **Block-Tiled Stays Strong (~55-60 GFLOPS)**. Performance remains high because **this approach maximizes L1/L3 cache reuse.**

### 4.3 Intel CPU (Platform 2)

| Work Group Size | Coalesced (GFLOPS) | Uncoalesced (GFLOPS) | Block-Tiled (GFLOPS) |
| --------------- | ------------------ | -------------------- | -------------------- |
| **N=2048**      |                    |                      |                      |
| 2×2             | 25.41              | 18.67                | 44.44                |
| 4×4             | 2.09               | 2.25                 | 4.94                 |
| 8×8             | 2.16               | 2.23                 | 9.60                 |
| 16×16           | 19.14              | 19.47                | 43.62                |
| 32×32           | 18.74              | 19.21                | 51.65                |

#### **1. Observations:**
- **Block-tiled** implementation performs best across all work group sizes peaking at 51.65 GFLOPS (32×32). This is because **Block-tiled** **reuses data within cache** before fetching new elements, reducing cache misses.

- **Coalesced and uncoalesced** access perform similarly, this is because unlike GPUs, CPUs **don’t have a concept of memory coalescing** at thread level. Instead, CPUs benefit **purely from cache-friendly memory access patterns.** This is why **coalesced and uncoalesced** perform similarly on CPUs.

- 4x4 and 8×8 workgroup sizes are the worst-performing configurations across all implementations, likely due to **cache thrashing**. This occurs when multiple threads repeatedly access memory addresses that map to the same cache lines, causing frequent evictions and reloadsand significantly degrading performance.

## 5. Analysis for NVIDIA GPU (N=8192)

|Configuration|Performance (GFLOPS)|
|---|---|
|Uncoalesced (1×32)|815.65|
|Coalesced (32×32)|1047.43|
|Coalesced (1×32)|144.10|
|Uncoalesced (32×32)|146.53|

### Analysis of Special Configurations:

1. **Uncoalesced (1×32) vs. Coalesced (32×32)**:
    
    - Coalesced (32×32) is 28% faster than Uncoalesced (1×32)
    - Coalesced access with a full 32×32 work group maximizes both memory throughput and computation parallelism
    - The uncoalesced result is close to the coalesced one because the 1×32 group size is effectively bunched up into a 32-thread warps, each thread accesses contiguous memory, similar to a coalesced 32×1 pattern. This reduces the memory access inefficiency, leading to minimal performance loss.

2. **Coalesced (1×32) vs. Uncoalesced (32×32)**:
    
    - Nearly identical performance (~146 GFLOPS)
    - This reveals an important insight: the combined effect of work-group shape and memory access pattern matters more than either factor alone
    - The 1×32 configuration struggles with coalesced access because it penalizes row-major memory layouts, where warps access non-contiguous memory locations. This leads to inefficient memory access comparable to the 32x32 uncoalsed access.

### How 32×32 Work Groups Are Organized

A 32×32 work group is organized as 32 rows × 32 columns = 1024 threads total. This means:

- 32 warps of 32 threads each
- In the 32x32 work group Individual rows of 32 threads form warps, which is highly beneficial for coalesced memory access (since adjacent threads access adjacent memory) but detrimental for uncoalesced access (since threads in the same warp access memory with large strides, forcing serialized memory transactions)

## B. Running the kernel on multiple OpenCL devices
## 1. Multi-Device Matrix Multiplication Strategy

### 5.1 Approach to Distributing Work Across Devices

For the uncoalesced implementation with N=8192 and 16×16 work group size, we implemented an iterative optimization approach to distribute the matrix multiplication workload across all three available devices.

#### Hardware Used
We used the following hardware to test our approach:
(insert specs iyed's pc here)


#### Device Performance Analysis

From the benchmark data:

NVIDIA RTX 3050 (Platform 0): ~137 GFLOPS (for uncoalesced with 16×16)
Intel UHD Graphics (Platform 1): ~2.5 GFLOPS (for uncoalesced with 16×16)
Intel CPU (Platform 2): ~19 GFLOPS (estimated based on N=2048 data)

#### Dynamic Row Distribution Strategy
We developed an iterative approach to optimize workload distribution:

##### 1. Proportional Row Distribution:

- We created a function that allocates matrix rows to each device **proportionally** to its measured **GFLOPS performance**.
- All row counts are adjusted to be multiples of 16 to align with our 16×16 work group size

##### 2. Iterative Optimization Process:

- Start with an initial distribution based on individual device benchmarks (throughuts)
- Run the parallel computation and measure actual throughput for each device (usually throughputs of devices run in parallel are a little different that when run individually)
- Feed these new throughput measurements back into the distribution function
- Generate an improved row distribution for the next iteration
- Repeat until speedup values converge (typically 2-3 iterations)

##### 3. Initial Distribution:

- NVIDIA GPU: Rows 0-7088 (86.52% of matrix)
- Intel CPU: Rows 7088-8064 (11.91% of matrix)
- Intel iGPU: Rows 8064-8191 (1.56% of matrix)

##### 4. Converged Distribution: 
- After several iterations, we arrived at an optimized distribution:

```Better row distribution based on throughput: {1: 128, 2: 480, 0: 7584}```


This dynamic distribution approach was chosen for several reasons:

1. **Adaptive Performance Allocation**: Accounts for performance variations when devices run in parallel
2. **Hardware-Aware Optimization**: Captures the impact of power sharing and thermal constraints
3. **Simplified Memory Management**: Row-based partitioning avoids complex data dependencies
4. **Minimal Communication Overhead**: No need for intermediate data transfers between devices

Remark :
The iterative process was crucial because we observed that device performance in isolation differs from performance when running concurrently. For example, the GPU received 140 watts when running alone but only 120 watts when running alongside the CPU, affecting its throughput.

#### Code and results

##### CPU Core Optimization Strategy

A key innovation in our implementation was the separation of CPU cores for computation and data transfer:

```python
# CPU core configuration
CPU_PLATFORM_ID = 2  # Intel CPU platform
CPU_TOTAL_CORES = 16  # Total logical processors
CPU_COMPUTE_CORES = 4  # Cores dedicated to computation
CPU_TRANSFER_CORES = 12  # Cores dedicated to data transfer
```

By dedicating specific cores to data transfer and others to computation, we prevented memory transfer bottlenecks that would otherwise limit CPU performance.

##### Distribution function
(insert pic here)

##### Thread Affinity Management

(insert screen shot here)

##### Implementation of matrix multiplaction for a device

(insert pic here)

##### Implementation of Multi-Device Execution

The core of our implementation uses Python's `ThreadPoolExecutor` to manage concurrent execution across all devices:

(insert screen shot here)

##### other code snippets

##### Results 

###### 1st iteration

The best results from our multi-device implementation were:
(example to setup tomorrow)
```

Device: Intel(R) UHD Graphics completed in 19.3849s at 4.99 GFLOPS
Device: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz completed in 19.5601s at 6.04 GFLOPS
Device: NVIDIA GeForce RTX 3070 Laptop GPU completed in 20.5257s at 257.38 GFLOPS

===== RESULTS =====
Single NVIDIA GPU: 20.775176s
Multi-device: 20.525678s
Speedup: 1.012155x

Better row distribution based on throughput: {1: 128, 2: 480, 0: 7584}

```

===== RESULTS =====
Single NVIDIA GPU: 44.127476s
Multi-device: 44.821002s
Speedup: 0.984527x

Final speedup achieved: 0.98x
Execution times for each device: {0: 41.17525100708008, 2: 41.30077934265137, 1: 44.82100248336792}
Throughput for each device: {0: 122.824363357746, 2: 8.059411244481169, 1: 2.395622062220589}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 144, 2: 480, 0: 7568}

===== RESULTS =====
Single NVIDIA GPU: 43.953746s
Multi-device: 41.883055s
Speedup: 1.049440x

Final speedup achieved: 1.05x
Execution times for each device: {1: 37.065372705459595, 0: 40.86957097053528, 2: 41.883055210113525}
Throughput for each device: {1: 2.3175092991132216, 0: 124.0057404305466, 2: 8.203732558579713}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 128, 2: 496, 0: 7568}

then

===== RESULTS =====
Single NVIDIA GPU: 44.027929s
Multi-device: 41.061944s
Speedup: 1.072232x

Final speedup achieved: 1.07x
Execution times for each device: {1: 36.08024263381958, 2: 38.760624170303345, 0: 41.061944246292114}
Throughput for each device: {1: 2.3807862599982297, 2: 8.310561403363465, 0: 123.94776572761978}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 144, 2: 496, 0: 7552}

#### Best speedup value

speed up = ...