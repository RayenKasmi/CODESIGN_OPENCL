
2025-04-02 03:35

Status:

Tags:

# open cl tp 1

Platform: NVIDIA CUDA
Device: NVIDIA GeForce RTX 3070 Laptop GPU (ALL | GPU)
        Global memory:          7.99951171875 GB
        Global cache:           1120.0 KB (READ_WRITE_CACHE)
        Global cache line:      128 B
        Local memory:           48.0 KB (LOCAL)
        Constant memory:        64.0 KB
        Compute units:          40
        Max work-group size:    1024
        Max work-item size:     [1024, 1024, 64]
        Lockstep unit:          32
        Max concurrent workgroups (est): 1280


Platform: Intel(R) OpenCL HD Graphics
Device: Intel(R) UHD Graphics (ALL | GPU)
        Global memory:          12.683395385742188 GB
        Global cache:           512.0 KB (READ_WRITE_CACHE)
        Global cache line:      64 B
        Local memory:           64.0 KB (LOCAL)
        Constant memory:        4194296.0 KB
        Compute units:          32
        Max work-group size:    256
        Max work-item size:     [256, 256, 256]
        Lockstep unit:          64
        Max concurrent workgroups (est): 1792


Platform: Intel(R) OpenCL
Device: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (ALL | CPU)
        Global memory:          31.70849609375 GB
        Global cache:           256.0 KB (READ_WRITE_CACHE)
        Global cache line:      64 B
        Local memory:           256.0 KB (GLOBAL)
        Constant memory:        128.0 KB
        Compute units:          16
        Max work-group size:    8192
        Max work-item size:     [8192, 8192, 8192]
        Lockstep unit:          128
        Max concurrent workgroups (est): 16
        
# Performance Analysis of Matrix Multiplication Implementations

## 1. CPU Sequential vs. OpenCL Implementation

| Implementation                | Matrix Size | Execution Time (s) | Performance (GFLOPS) | Operations         |
| ----------------------------- | ----------- | ------------------ | -------------------- | ------------------ |
| CPU Sequential                | 256         | 4.23               | 0.008                | 1 matrix product   |
| CPU Sequential                | 512         | 36.67              | 0.007                | 1 matrix product   |
| CPU OpenCL (Coalesced, LS=16) | 256         | 0.011              | 58.93                | 20 matrix products |
| CPU OpenCL (Coalesced, LS=16) | 512         | 0.093              | 57.79                | 20 matrix products |

### Analysis:

- **Sequential vs. OpenCL Performance Gap**: OpenCL implementation on CPU achieves ~7,400× higher performance than sequential code for N=256 and ~7,900× higher for N=512
- **Reasons for Massive Speedup**:
    1. **Parallelization**: OpenCL utilizes all CPU cores and SIMD units
    2. **Cache Optimization**: Better memory access patterns in OpenCL implementation
    3. **Compiler Optimizations**: OpenCL runtime applies additional optimizations
- **Scaling with Problem Size**: Sequential implementation scales poorly (performance drops slightly as N increases), while OpenCL maintains consistent performance

## 2. Performance Comparison Across Devices and Implementations

### NVIDIA GeForce RTX 3070 Laptop GPU (Platform 0)

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
| 32×32           | 1047.43            | 145.97               | 908.69               |

### **Analysis for NVIDIA GPU**

#### **1. Best Implementation**

- **Coalesced access** is consistently the best performer across different work group sizes and matrix sizes.
    
- **Block-tiled** access is also efficient but generally slightly slower than coalesced.
    
- **Uncoalesced memory access** is the worst performer, especially for larger work group sizes.
    
#### **2. Impact of Work Group Size**

- **For Coalesced:**
    - 16×16 and 32×32 perform the best.
    - For smaller matrices (N=2048, 4096), 16×16 is slightly better.
    - For larger matrices (N=8192), 32×32 reaches peak performance.
        
- **For Uncoalesced:**
    - 8×8 achieves the best performance, nearly **3.5× faster than 32×32**, due to memory access bottlenecks.
    - Larger work group sizes (16×16, 32×32) perform worse because of increased memory access divergence.
        
- **For Block-Tiled:**
    - Performance remains relatively **stable** across different work group sizes.
    - Best performance is seen at **2×2 and 16×16** for most cases.
#### **3. Memory Access Patterns**

- **NVIDIA’s architecture is optimized for coalesced access**, where adjacent threads access adjacent memory locations, minimizing memory latency
- **Uncoalesced access suffers from memory divergence**, leading to poor performance at larger work group sizes.
- **Block-tiled implementations** benefit from shared memory optimizations but can be affected by bank conflicts and synchronization overhead.
#### **4. Performance Scaling with Matrix Size**

- The **performance does not degrade significantly** with increasing matrix size.
- The **best implementations (coalesced, 16×16 & 32×32) maintain a high GFLOPS rate across all N values**.
- **For uncoalesced access, performance remains poor regardless of N**, but 8×8 continues to be the best configuration.

### Intel UHD Graphics (Platform 1)

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

#### Analysis for Intel Integrated GPU:

#### **1. Best Implementation: Block-Tiled Dominance** 

- **Block-tiled implementation consistently outperforms others, achieving 2-10× better performance.**
    
- Across all matrix sizes, block-tiled maintains a stable range of **~46-60 GFLOPS** due to **cache reuse and efficient local memory access.**
    

#### **2. Error Cases: 32×32 Work Group Size Fails** 

- The **32×32 work group size fails across all implementations** due to **Intel’s hardware limitation (max work group size = 256 threads).**
    
- Intel’s integrated GPUs typically **favor smaller work groups** due to their **suboptimal scheduling for very large thread blocks.**
    

#### **3. Performance Trends** 

| Implementation  | **N=2048**   | **N=4096**   | **N=8192**   | **Trend**         |
| --------------- | ------------ | ------------ | ------------ | ----------------- |
| **Coalesced**   | 33.34 GFLOPS | 22.72 GFLOPS | 16.50 GFLOPS | **Decreasing**    |
| **Uncoalesced** | 4.90 GFLOPS  | 4.66 GFLOPS  | 4.62 GFLOPS  | **Low & Stable**  |
| **Block-Tiled** | 60.84 GFLOPS | 59.24 GFLOPS | 57.29 GFLOPS | **Stable & High** |

- **Coalesced Memory Access Degrades as Matrix Size Increases**
    
    - Starts at **33 GFLOPS (N=2048)** but **drops to 16 GFLOPS (N=8192)** → **Intel’s memory subsystem is not optimized for global memory coalescing.**
        
    - Unlike NVIDIA GPUs, Intel **doesn’t have strong warp-based memory coalescing mechanisms.**
        
    - More cache misses occur as matrix size increases, reducing efficiency.
        
- **Uncoalesced Memory Access is Always Poor (~4-8 GFLOPS)**
    
    - This is expected because **random memory access patterns cause high latency.**
        
    - Intel’s **iGPU has limited L3 cache**, making **uncoalesced memory access highly inefficient.**
        
- **Block-Tiled Stays Strong (~55-60 GFLOPS)**
    
    - Performance remains high because **this approach maximizes L1/L3 cache reuse.**
        
    - **Tiling improves data locality**, which is crucial for **Intel’s shared memory hierarchy.**
        

---

### **4. Why Does Intel’s iGPU Favor Block-Tiled Over Coalesced Access?**

**Key Differences from NVIDIA GPUs:**

| Feature                     | **Intel Integrated GPU**                 | **NVIDIA GPU**                            |
| --------------------------- | ---------------------------------------- | ----------------------------------------- |
| **Memory Coalescing**       | **Weak**  (No warp-level coalescing)     | **Strong**  (Warp-wide memory coalescing) |
| **Cache Hierarchy**         | **L1 → L3 (No dedicated shared memory)** | **L1 → L2 + Dedicated Shared Memory**     |
| **Optimal Work Group Size** | **Small (≤16×16)**                       | **Large (up to 32×32)**                   |

**Intel’s iGPU benefits more from cache locality than memory coalescing.**

- **Block-Tiled method leverages cache reuse**, leading to **high GFLOPS**.
    
- **Coalesced method isn’t fully optimized** because Intel lacks **NVIDIA-style warp-wide memory coalescing.**
    
- **Uncoalesced memory access is always slow** because Intel’s cache architecture cannot compensate for random access patterns.

### Intel CPU (Platform 2) - For N=2048 Only

| Work Group Size        | Coalesced (GFLOPS) | Uncoalesced (GFLOPS) | Block-Tiled (GFLOPS) |
| ---------------------- | ------------------ | -------------------- | -------------------- |
| **N=2048**             |                    |                      |                      |
| 2×2                    | 25.41              | 18.67                | 44.44                |
| 4×4                    | 2.09               | 2.25                 | 4.94                 |
| 8×8                    | 2.16               | 2.23                 | 9.60                 |
| 16×16                  | 19.14              | 19.47                | 43.62                |
| 32×32                  | 18.74              | 19.21                | 51.65                |

#### Analysis for Intel CPU:

#### **1. Best Implementation: Block-Tiled Wins Again ✅**

- Block-tiled is the fastest **across all work group sizes, peaking at 51.65 GFLOPS (32×32).**
    
- Performance remains **consistently high for 16×16 and 32×32**, confirming that **cache-friendly memory access is critical.**
    

#### **2. Work Group Size Impact**

|Work Group Size|**Performance Trend**|
|---|---|
|**2×2**|Strong across all implementations (**cache reuse is effective at small scales**)|
|**4×4, 8×8**|**Very poor performance** (likely cache thrashing)|
|**16×16, 32×32**|Performance **recovers significantly** (better memory locality)|

🔴 **8×8 is the worst-performing size across all implementations**

- **Hypothesis:** CPU cache organization **conflicts with this access pattern**, causing cache thrashing.
    
- **Evidence:** **All methods (coalesced, uncoalesced, block) perform abnormally bad at 8×8.**
    
- **Recommendation:** Avoid 8×8 **for CPU-based matrix operations** if possible.
    

#### **3. Why Does Block-Tiled Work So Well on CPU?**

- **Cache Blocking Improves Locality:**
    
    - CPUs rely **heavily on L1/L2 cache** to avoid expensive memory accesses.
        
    - Block-tiled **reuses data within cache** before fetching new elements, reducing cache misses.
        
    - This is why **performance remains high (~40-50 GFLOPS) even at larger work sizes.**
        
- **Coalesced vs. Uncoalesced Matters Less:**
    
    - Unlike GPUs, CPUs **don’t have a concept of memory coalescing** at warp/thread level.
        
    - Instead, CPUs benefit **purely from cache-friendly memory access patterns.**
        
    - This is why **coalesced and uncoalesced perform similarly on CPUs.**

## 3. Specialized Analysis for NVIDIA GPU (N=8192)

|Configuration|Performance (GFLOPS)|
|---|---|
|Uncoalesced (1×32)|815.65|
|Coalesced (32×32)|1047.43|
|Coalesced (1×32)|144.10|
|Uncoalesced (32×32)|146.53|

### Analysis of Special Configurations:

1. **Uncoalesced (1×32) vs. Coalesced (32×32)**:
    
    - Coalesced (32×32) is 28.4% faster than Uncoalesced (1×32)
    - This demonstrates the importance of both coalesced memory access and optimal work-group size
    - Coalesced access with a full 32×32 work group maximizes both memory throughput and computation parallelism
2. **Coalesced (1×32) vs. Uncoalesced (32×32)**:
    
    - Nearly identical performance (~146 GFLOPS)
    - This reveals an important insight: the combined effect of work-group shape and memory access pattern matters more than either factor alone
    - The 1×32 work group is inherently unsuitable for coalesced access (penalizes row-major memory layout)
    - The 32×32 configuration for uncoalesced access creates inefficient memory access patterns that waste bandwidth
### How 32×32 Work Groups Are Organized

A 32×32 work group is organized as 32 rows × 32 columns = 1024 threads total. This means:

- 32 warps of 32 threads each
- Each warp processes one row of the work group in the coalesced case
- Each warp processes one column of the work group in the uncoalesced case
## Key Insights and Recommendations

1. **Device-Specific Optimization**:
    
    - NVIDIA GPUs: Use coalesced access with large work groups (16×16 or 32×32)
    - Intel GPUs: Use block-tiled implementation with moderate work groups (16×16)
    - CPUs: Block-tiled algorithm with either very small (2×2) or larger (32×32) work groups
2. **Memory Access Patterns**:
    
    - Memory access pattern is often more important than raw computational capability
    - Coalesced access is critical for NVIDIA GPUs, less important for Intel architectures
3. **Work Group Sizing**:
    
    - Must consider hardware limits (work group size, local memory)
    - Optimal size varies by device and implementation
    - Size should match architecture characteristics (SIMD width, cache hierarchy)
4. **Implementation Selection**:
    
    - Block-tiled approach is most portable across different architectures
    - Coalesced implementation delivers best performance on NVIDIA GPUs
    - Consider hybrid approaches for cross-platform performance




SOME BRAINSTORMING:

# Multi-Device Matrix Multiplication Implementation

To optimize the UNCOALESCED implementation for N=8192 with work-group size 16×16, I'll create a solution that effectively distributes work across all three available OpenCL devices. This approach will leverage the strengths of each device while minimizing their weaknesses.

## Strategy for Work Distribution

Based on the benchmark data, I'll use a performance-weighted distribution approach:

1. **Device Performance Analysis**:
    
    - NVIDIA RTX 3070 (Platform 0): ~268 GFLOPS for uncoalesced with 16×16
    - Intel UHD Graphics (Platform 1): ~4.5 GFLOPS for uncoalesced with 16×16
    - Intel CPU (Platform 2): ~19 GFLOPS (estimated based on 2048 size)
2. **Optimal Work Distribution**:
    
    - NVIDIA GPU: 92% of computation (high performance)
    - Intel Integrated GPU: 1% of computation (lowest performance)
    - Intel CPU: 7% of computation (medium performance)
3. **Matrix Splitting Strategy**:
    
    - For an 8192×8192 matrix, we'll split rows to simplify data partitioning
    - NVIDIA: First 7538 rows (92%)
    - Intel CPU: Next 573 rows (7%)
    - Intel iGPU: Last 81 rows (1%)

This distribution aligns with the relative performance of each device while ensuring work is divided on row boundaries to simplify implementation.

## Implementation Code

```python

#!/usr/bin/env python

"""

Multi-Device Matrix Multiplication with Uncoalesced Access Pattern

For N=8192 matrices with 16×16 work group size

"""

import numpy as np

import pyopencl as cl

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from helper import *

from definitions import *

# Matrix size

N = 8192

# Work group size

WORKGROUP_SIZE = 16

  

# Optimization 1: Define CPU-specific parameters

CPU_PLATFORM_ID = 2  # Change based on your system

CPU_RESERVED_CORES = [6, 7]  # Reserve last 2 cores

CPU_MAX_THREADS = 8

  

def create_context_and_queue(platform_idx, device_idx=0):

    """Create an OpenCL context and command queue for the specified platform."""

    platforms = cl.get_platforms()

    if 0 <= platform_idx < len(platforms):

        platform = platforms[platform_idx]

        devices = platform.get_devices()

        if 0 <= device_idx < len(devices):

            device = devices[device_idx]

            context = cl.Context([device])

            queue = cl.CommandQueue(context)

            return context, queue, device

    return None, None, None

  

def load_kernel():

    """Load the uncoalesced kernel code."""

    with open("C_elem_ij.cl", "r") as f:

        return f.read()

  

def matrix_multiply_device(platform_idx, start_row, end_row, A_full, B_full, C_full):

    """Execute matrix multiplication on a specific device for a subset of rows."""

    # Create OpenCL context and queue for this device

    context, queue, device = create_context_and_queue(platform_idx)

    if context is None:

        print(f"Failed to create context for platform {platform_idx}")

        return False

    device_name = device.name

    print(f"Device: {device_name} processing rows {start_row} to {end_row}")

    # Prepare data for this device's portion

    num_rows = end_row - start_row

    A_device = A_full[start_row*N:end_row*N].copy()

    C_device = np.zeros(num_rows*N, dtype=np.float32)

    # Create OpenCL buffers

    mf = cl.mem_flags

    d_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_device)

    d_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_full)

    d_c = cl.Buffer(context, mf.WRITE_ONLY, C_device.nbytes)

    # Build program

    kernel_source = load_kernel()

    program = cl.Program(context, kernel_source).build()

    mmul = program.mmul

    mmul.set_scalar_arg_dtypes([np.int32, None, None, None])

    # Execute kernel

    start_time = time.time()

    # Launch the kernel

    try:

        event = mmul(queue, (num_rows, N), (WORKGROUP_SIZE, WORKGROUP_SIZE),

                     np.int32(N), d_a, d_b, d_c)

        event.wait()

        cl.enqueue_copy(queue, C_device, d_c)

        queue.finish()

    except Exception as e:

        print(f"Error on device {device_name}: {str(e)}")

        return None

    execution_time = time.time() - start_time

    gflops = 2.0 * num_rows * N * N/(1000000000.0 * execution_time)

    print(f"Device: {device_name} completed in {execution_time:.4f}s at {gflops:.2f} GFLOPS")

    # Copy the result back to the appropriate part of C_full

    C_full[start_row*N:end_row*N] = C_device

    return platform_idx,execution_time, gflops

  

def multi_device_matrix_multiply():

    """Execute matrix multiplication across multiple devices."""

    # Initialize matrices

    A = np.full(N*N, AVAL, dtype=np.float32)

    B = np.full(N*N, BVAL, dtype=np.float32)

    C = np.zeros(N*N, dtype=np.float32)

    # Work distribution based on performance analysis

    # NVIDIA GPU: 89.782% (~7355 rows)

    # Intel CPU: 8.522% (~698 rows)

    # Intel iGPU: 1.695% (~139 rows)

    devices = [

        {"platform_idx": 0, "start_row": 0, "end_row": 8048}, #make the rows multiple of 16

        {"platform_idx": 2, "start_row": 8048, "end_row": 8064}, #Intel CPU

        {"platform_idx": 1, "start_row": 8048, "end_row": N}

    ]

    # Measure single device performance first (NVIDIA GPU with all rows)

    print("Starting single device benchmark (NVIDIA GPU only)...")

    _,single_device_time,_ = matrix_multiply_device(0, 0, N, A, B, np.zeros(N*N, dtype=np.float32))

    print(f"Single device time: {single_device_time:.4f}s")

    # Now run multi-device version

    print("\nStarting multi-device execution...")

    # Execute in parallel across all devices

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:

        futures = []

        for device_config in devices:

            futures.append(

                executor.submit(

                    matrix_multiply_device,

                    device_config["platform_idx"],

                    device_config["start_row"],

                    device_config["end_row"],

                    A, B, C

                )

            )

        # Wait for all tasks to complete and get execution times

        execution_times = {}

        throughput_dict = {}

        for future in as_completed(futures):

            if future.result() is not None:

                plat_id, exec_time, gflops = future.result()

                execution_times[plat_id] = exec_time

                throughput_dict[plat_id] = gflops

    multi_device_time = max(execution_times.values())

    # Calculate speedup

    speedup = single_device_time / multi_device_time

    print("\n===== RESULTS =====")

    print(f"Single NVIDIA GPU: {single_device_time:.6f}s")

    print(f"Multi-device: {multi_device_time:.6f}s")

    print(f"Speedup: {speedup:.6f}x")

    return C, speedup,execution_times,throughput_dict

  

def compute_row_distribution(N, throughput_dict):

    """Compute row distribution based on throughput."""

    total_throughput = sum(throughput_dict.values())

    distribution = {}

    #starting with the lowest throught put device

    for device, throughput in sorted(throughput_dict.items(), key=lambda x: x[1]):

        # Calculate the number of rows for this device based on its throughput

        num_rows = int(N * (throughput / total_throughput))

        num_rows = round(num_rows)

        distribution[device] = num_rows

    id_device_lowest_throughput = min(distribution, key=distribution.get)

    id_device_highest_throughput = max(distribution, key=distribution.get)

    # Adjust the distribution to ensure it sums to N

    if(sum(distribution.values()) < N):

        distribution[id_device_highest_throughput] += N - sum(distribution.values())

    elif(sum(distribution.values()) > N):

        distribution[id_device_lowest_throughput] -= sum(distribution.values()) - N

    leftover = 0

  

    #start with the lowest distribution device to the highest distribution device

    for device in sorted(distribution, key=distribution.get):

        if(device == id_device_highest_throughput):

            distribution[device] += leftover

        else:

            leftover += distribution[device]%16

            distribution[device] -= distribution[device]%16

  

    return distribution

  

result_matrix, speedup, execution_times,throughput_dict = multi_device_matrix_multiply()

print(f"\nFinal speedup achieved: {speedup:.2f}x")

print(f"Execution times for each device: {execution_times}")

print(f"Throughput for each device: {throughput_dict}")

#print(f"there are {error(N,result_matrix)} errors in the multiplication")

print(f"matrix first 10 elements:\n{result_matrix[:10]}")

  

distribution = compute_row_distribution(N,throughput_dict)

print(f"Better row distribution based on throughput: {distribution}")
    
```

## Optimizations and Key Considerations

### 1. Performance-Based Distribution

The work distribution is optimized based on the relative performance of each device:

- **NVIDIA GPU (92%)**: Handles the majority of computation due to its superior performance
- **CPU (7%)**: Handles a moderate portion
- **iGPU (1%)**: Handles the smallest portion due to its low performance with uncoalesced access

### 2. Row-Based Partitioning

I chose row-based partitioning for several reasons:

- **Simplified Memory Management**: Each device processes complete rows, avoiding complex data dependencies
- **Reduced Communication Overhead**: No need for intermediate data transfer between devices
- **Optimal for Matrix Multiplication**: Each device can compute its portion independently

### 3. ThreadPoolExecutor for Parallel Execution

Using Python's ThreadPoolExecutor to manage concurrent execution across devices:

- **Asynchronous Execution**: All devices compute in parallel
- **Simplified Management**: The executor handles thread creation and synchronization
- **Error Handling**: Robust error handling for device-specific failures

### 4. Device-Specific Parameters

For optimal performance, we adjust work distribution based on device capabilities:

- **Global Work Size**: Adjusted for each device based on its assigned rows
- **Local Work Size**: Fixed at 16×16 as specified in the requirements

### 5. Memory Buffer Management

Careful allocation of OpenCL buffers to minimize data transfer overhead:

- **A Matrix**: Only the relevant portion is transferred to each device
- **B Matrix**: Full matrix is transferred to all devices (required for matrix multiplication)
- **C Matrix**: Each device only computes its assigned portion

## Expected Performance Improvements

The theoretical speedup is limited by:

1. **Amdahl's Law**: The fastest device (NVIDIA) still handles 92% of the work
2. **Memory Transfer Overhead**: Data transfer between host and devices
3. **Synchronization Overhead**: Waiting for the slowest device

Given these constraints, a realistic expected speedup is approximately 1.05-1.15x over the single NVIDIA GPU implementation.

The main benefit comes from offloading some computation to other devices while the NVIDIA GPU is already busy, effectively using resources that would otherwise be idle.

## Error Handling and Robustness

The implementation includes several error handling mechanisms:

- Context creation validation
- Kernel execution error catching
- Result validation
- Graceful failure handling for any device

This ensures that even if one device fails, the others can continue processing.


# RESULTS
## 1 iteration
Device: Intel(R) UHD Graphics completed in 3.3806s at 5.08 GFLOPS
Device: NVIDIA GeForce RTX 3070 Laptop GPU completed in 3.9892s at 248.17 GFLOPS
Device: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz completed in 13.7177s at 6.73 GFLOPS

===== RESULTS =====
Single NVIDIA GPU: 4.193498s
Multi-device: 13.717669s
Speedup: 0.305700x

Final speedup achieved: 0.31x
Execution times for each device: {1: 3.3806142807006836, 0: 3.9892148971557617, 2: 13.717669486999512}
Throughput for each device: {1: 5.081877953979184, 0: 248.16661605115456, 2: 6.731595111801901}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 160, 2: 208, 0: 7824}

## 2 iteration

===== RESULTS =====
Single NVIDIA GPU: 4.148237s
Multi-device: 4.607841s
Speedup: 0.900256x

Final speedup achieved: 0.90x
Execution times for each device: {0: 4.1707799434661865, 2: 4.542567729949951, 1: 4.6078410148620605}
Throughput for each device: {0: 251.78012700408334, 2: 6.145706367774419, 1: 4.660498574220636}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 144, 2: 176, 0: 7872}
## 3 iteration
Single NVIDIA GPU: 4.154142s
Multi-device: 4.168373s
Speedup: 0.996586x

Final speedup achieved: 1.00x
Execution times for each device: {2: 3.6710219383239746, 1: 3.8614578247070312, 0: 4.168373107910156}
Throughput for each device: {2: 6.43480767069044, 1: 5.005195889577369, 0: 253.47106112238473}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 144, 2: 192, 0: 7856}

# screw the cpu
## 1 iteration
===== RESULTS =====
Single NVIDIA GPU: 4.150173s
Multi-device: 4.100549s
Speedup: 1.012102x

Final speedup achieved: 1.01x
Execution times for each device: {1: 3.733700752258301, 0: 4.10054874420166}
Throughput for each device: {1: 5.17646006319869, 0: 263.42432253035}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 144, 0: 8048}

# CPU WAS ACTUALLY GOOD JUST HAD TO SPLIT some CORES FOR DATA TRANSFER AND THE REST FOR COMPUTATION
## iteration 1
===== RESULTS =====
Single NVIDIA GPU: 4.183502s
Multi-device: 4.147134s
Speedup: 1.008769x

Final speedup achieved: 1.01x
Execution times for each device: {2: 0.254042387008667, 1: 3.4027929306030273, 0: 4.147134304046631}
Throughput for each device: {2: 8.453249370258577, 1: 5.0487554001575585, 0: 260.465226286497}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 144, 2: 240, 0: 7808}

Single NVIDIA GPU: 4.130121s
Multi-device: 5.327380s
Speedup: 0.775263x

Final speedup achieved: 0.78x
Execution times for each device: {1: 3.4226577281951904, 0: 4.160138845443726, 2: 5.327380418777466}
Throughput for each device: {1: 5.019452877942065, 0: 251.90794325813468, 2: 6.449649859223854}
matrix first 10 elements:
[152703.17 152703.17 152703.17 152703.17 152703.17 152703.17 152703.17
 152703.17 152703.17 152703.17]
Better row distribution based on throughput: {1: 144, 2: 192, 0: 7856}

## iteration 2
the previous one was the best result xd

# References
